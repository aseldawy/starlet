from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely import from_wkb
from shapely.geometry import (
    Point, LineString, LinearRing, Polygon,
    MultiPoint, MultiLineString, MultiPolygon, GeometryCollection
)
from pyproj import Transformer

from starlet._internal.progress import iter_with_progress
from starlet._internal.tiling.crs import WGS84_CRS, geoparquet_crs
from starlet._internal.tiling.datasource import GeoParquetSource

logger = logging.getLogger(__name__)

# Global Web Mercator extent
LIM = 20037508.342789244
GLOBAL_BBOX = (-LIM, -LIM, LIM, LIM)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

@dataclass
class HistConfig:
    grid_size: int = 4096
    out_crs: str = "EPSG:3857"
    dtype: str = "float64"
    max_parallel_tiles: int = 8
    rg_parallel: int = 4


# ---------------------------------------------------------------------------
# GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def _geometry_vertices_iter(g):
    if g is None or g.is_empty:
        return

    if isinstance(g, Point):
        yield (g.x, g.y)

    elif isinstance(g, (LineString, LinearRing)):
        for x, y in g.coords:
            yield (x, y)

    elif isinstance(g, Polygon):
        for x, y in g.exterior.coords:
            yield (x, y)
        for ring in g.interiors:
            for x, y in ring.coords:
                yield (x, y)

    elif isinstance(g, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        for sub in g.geoms:
            yield from _geometry_vertices_iter(sub)

    else:
        coords = getattr(g, "coords", None)
        if coords is not None:
            for x, y in coords:
                yield (x, y)


# ---------------------------------------------------------------------------
# PER TILE HISTOGRAM
# ---------------------------------------------------------------------------

def _accumulate_vertices_hist(
    table,
    histogram: np.ndarray,
    geom_col: str,
    bbox_out,
    transformer,
):
    """Transform each vertex and increment its histogram cell in place."""
    geoms = from_wkb(table[geom_col].to_numpy(zero_copy_only=False))

    minx, miny, maxx, maxy = bbox_out
    inv_w = 1.0 / (maxx - minx)
    inv_h = 1.0 / (maxy - miny)
    grid_size = histogram.shape[0]

    for g in geoms:
        if g is None or g.is_empty:
            continue

        for x, y in _geometry_vertices_iter(g):
            transformed_x, transformed_y = transformer.transform(x, y)
            if not (np.isfinite(transformed_x) and np.isfinite(transformed_y)):
                logger.debug(
                    "Skipping histogram vertex with non-finite transformed coordinates: "
                    "input=(%s, %s) transformed=(%s, %s)",
                    x,
                    y,
                    transformed_x,
                    transformed_y,
                )
                continue
            ix = int((transformed_x - minx) * inv_w * grid_size)
            iy = int((transformed_y - miny) * inv_h * grid_size)
            ix = min(max(ix, 0), grid_size - 1)
            iy = min(max(iy, 0), grid_size - 1)
            histogram[grid_size - 1 - iy, ix] += 1


# ---------------------------------------------------------------------------
# PROCESS SPLIT GROUPS
# ---------------------------------------------------------------------------

def _split_groups(splits: Sequence, max_workers: int) -> List[List]:
    """Divide splits into balanced, non-empty groups for worker processes."""
    if max_workers <= 0:
        raise ValueError("hist_max_parallel must be positive")

    worker_count = min(max_workers, len(splits))
    base_size, remainder = divmod(len(splits), worker_count)
    groups: List[List] = []
    offset = 0
    for index in range(worker_count):
        group_size = base_size + (1 if index < remainder else 0)
        groups.append(list(splits[offset:offset + group_size]))
        offset += group_size
    return groups


def _process_split_group(
    source_path: str,
    splits: Sequence,
    cfg,
    geom_col: str,
) -> np.ndarray:
    """Sequentially accumulate one balanced group of source splits."""
    logger.info("Processing histogram group with %d splits", len(splits))

    source = GeoParquetSource(
        source_path,
        geometry_only=True,
        geom_col=geom_col,
    )
    dtype = np.dtype(cfg.dtype)
    bbox = GLOBAL_BBOX
    base = np.zeros((cfg.grid_size, cfg.grid_size), dtype=dtype)
    transformers = {}

    for split in splits:
        for table in source.iter_tables(split):
            source_crs = geoparquet_crs(table.schema, geom_col) or WGS84_CRS
            transformer = transformers.get(str(source_crs))
            if transformer is None:
                transformer = Transformer.from_crs(source_crs, cfg.out_crs, always_xy=True)
                transformers[str(source_crs)] = transformer
            _accumulate_vertices_hist(
                table.combine_chunks(),
                base,
                geom_col,
                bbox,
                transformer,
            )

    return base


# ---------------------------------------------------------------------------
# GLOBAL SUM AND PREFIX SUM
# ---------------------------------------------------------------------------

def _sum_all_tiles(tile_hists: List[np.ndarray], outdir: Path, dtype="float64") -> Path:

    if not tile_hists:
        raise RuntimeError("No tile histograms generated")

    example = tile_hists[0]
    total = np.zeros_like(example, dtype=dtype)

    for hist in tile_hists:
        if hist.shape != total.shape:
            raise ValueError(
                f"Tile histogram has shape {hist.shape}, expected {total.shape}"
            )
        total += hist

    global_path = outdir / "global.npy"
    np.save(global_path, total, allow_pickle=False)

    global_json = {
        "filename": "global.npy",
        "dtype": str(total.dtype),
        "grid_size": int(total.shape[0]),
        "shape": list(total.shape),
        "crs": "EPSG:3857",
        "bbox": list(GLOBAL_BBOX),
        "sum": float(total.sum()),
        "nonzero": int(np.count_nonzero(total)),
    }
    with open(outdir / "global.json", "w") as f:
        json.dump(global_json, f, indent=2)

    prefix = total.cumsum(axis=0).cumsum(axis=1)

    prefix_path = outdir / "global_prefix.npy"
    np.save(prefix_path, prefix, allow_pickle=False)

    prefix_json = {
        "filename": "global_prefix.npy",
        "dtype": str(prefix.dtype),
        "grid_size": int(prefix.shape[0]),
        "shape": list(prefix.shape),
        "crs": "EPSG:3857",
        "bbox": list(GLOBAL_BBOX),
        "desc": "2D prefix sum histogram"
    }
    with open(outdir / "global_prefix.json", "w") as f:
        json.dump(prefix_json, f, indent=2)

    logger.info(f"Wrote global histogram: {global_path}")
    logger.info(f"Wrote global prefix sum histogram: {prefix_path}")

    return prefix_path


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def build_histograms_for_dir(
    tiles_dir: str,
    outdir: str,
    geom_col="geometry",
    grid_size=4096,
    dtype="float64",
    hist_max_parallel=8,
    hist_rg_parallel=4,
):
    cfg = HistConfig(
        grid_size=grid_size,
        dtype=dtype,
        max_parallel_tiles=hist_max_parallel,
        rg_parallel=hist_rg_parallel,
    )

    source = GeoParquetSource(
        tiles_dir,
        geometry_only=True,
        geom_col=geom_col,
    )
    splits = source.create_splits()
    if not splits:
        logger.error("No parquet tiles found")
        return

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    tile_outputs = []
    split_groups = _split_groups(splits, cfg.max_parallel_tiles)
    logger.info(
        "Histogram computation: %d splits grouped into %d worker tasks",
        len(splits),
        len(split_groups),
    )

    with ProcessPoolExecutor(max_workers=cfg.max_parallel_tiles) as ex:
        futures = {
            ex.submit(_process_split_group, source.path, split_group, cfg, geom_col): split_group
            for split_group in split_groups
        }

        for f in iter_with_progress(
            as_completed(futures),
            total=len(futures),
            logger=logger,
            label="histogram",
        ):
            tile_outputs.append(f.result())

    _sum_all_tiles(tile_outputs, outdir_p, dtype=dtype)
