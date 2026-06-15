"""Buffered tile writer with spatial sorting and parallel flush.

Provides :class:`WriterPool` which buffers Arrow Tables per tile in
memory and writes them to GeoParquet files in bounded-parallel rounds.
Each tile's rows are optionally sorted (Z-order, Hilbert, or by columns)
before writing, and GeoParquet ``geo`` metadata is updated with per-tile
bounding boxes.
"""
from __future__ import annotations

import json
import math
import os
import multiprocessing
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import shapely
from shapely import from_wkb

from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)

# Per-geometry bounding-box "covering" columns written alongside each tile so
# that the on-demand tile server can prune row groups and rows at read time
# (pyarrow predicate pushdown) instead of decoding the whole partition.
BBOX_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")

# Rows per Parquet row group.  Tiles are spatially sorted (Z-order) before
# writing, so bounding the row-group size makes each row group spatially
# coherent — its min/max bbox statistics become a tight filter the reader can
# use to skip row groups that don't intersect a requested tile.
DEFAULT_ROW_GROUP_SIZE = 16384

# ------------------------- Sorting configuration -------------------------

@dataclass
class SortKey:
    column: str
    ascending: bool = True

class SortMode:
    NONE = "none"
    COLUMNS = "columns"
    ZORDER = "zorder"
    HILBERT = "hilbert"


@dataclass(frozen=True)
class _WriterPoolConfig:
    geom_col: str
    sort_mode: str
    sort_keys: List[SortKey]
    sfc_bits: int
    global_extent: Optional[Tuple[float, float, float, float]]
    compression: str
    pq_args: Dict[str, Any]
    outdir: str

# ------------------------- Utility: Morton (Z-order) ----------------------

def _scale_to_uint(v: np.ndarray, vmin: float, vmax: float, bits: int) -> np.ndarray:
    """Normalize *v* into [0, 2^bits - 1] for space-filling curve interleaving."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(v, dtype=np.uint64)
    rng = vmax - vmin
    x = (v - vmin) / rng
    x = np.clip(x, 0.0, 1.0)
    return (x * ((1 << bits) - 1)).astype(np.uint64, copy=False)

def _interleave_bits_2d(x: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    """Compute Morton (Z-order) codes by interleaving bits of *x* and *y*."""
    x = x.astype(np.uint64, copy=False)
    y = y.astype(np.uint64, copy=False)

    def part1by1(n):
        n &= 0x00000000FFFFFFFF
        n = (n | (n << 16)) & 0x0000FFFF0000FFFF
        n = (n | (n << 8))  & 0x00FF00FF00FF00FF
        n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F
        n = (n | (n << 2))  & 0x3333333333333333
        n = (n | (n << 1))  & 0x5555555555555555
        return n

    return (part1by1(y) << 1) | part1by1(x)

def _maybe_sort_and_bbox(
    tbl: pa.Table,
    geom_col: str,
    sort_mode: str,
    sort_keys: List[SortKey],
    sfc_bits: int,
    global_extent: Optional[Tuple[float, float, float, float]],
) -> Tuple[Tuple[float, float, float, float], pa.Table, np.ndarray]:
    """Compute the tile bbox, optionally sort rows, and return per-row bounds.

    Returns ``(bbox, table, per_row_bounds)`` where ``per_row_bounds`` is an
    ``(N, 4)`` float64 array of ``[xmin, ymin, xmax, ymax]`` aligned with the
    rows of the **returned** (possibly sorted) table; rows with no geometry are
    ``NaN``.  The aggregate bbox and per-row bounds come from a single
    vectorised ``shapely.bounds`` call (no per-geometry Python loop).
    """
    N = tbl.num_rows
    geoms = from_wkb(tbl[geom_col].to_numpy(zero_copy_only=False))
    per = shapely.bounds(geoms)  # (N, 4), NaN rows for None/empty geometries
    per = np.asarray(per, dtype=np.float64).reshape(N, 4)
    valid = np.isfinite(per[:, 0])

    if not valid.any():
        bbox = (np.inf, np.inf, -np.inf, -np.inf)
        return bbox, tbl, per

    bbox = (
        float(np.nanmin(per[:, 0])), float(np.nanmin(per[:, 1])),
        float(np.nanmax(per[:, 2])), float(np.nanmax(per[:, 3])),
    )

    if sort_mode == SortMode.NONE:
        return bbox, tbl, per

    if sort_mode == SortMode.COLUMNS:
        if not sort_keys:
            return bbox, tbl, per
        spec = [(sk.column, "ascending" if sk.ascending else "descending") for sk in sort_keys]
        logger.debug("Sorting by columns: %s", spec)
        order = np.asarray(pc.sort_indices(tbl, sort_keys=spec), dtype=np.int64)
        return bbox, tbl.take(pa.array(order, type=pa.int64())), per[order]

    if sort_mode in (SortMode.ZORDER, SortMode.HILBERT):
        cen = shapely.centroid(geoms)
        cx = np.nan_to_num(shapely.get_x(cen), nan=0.0)
        cy = np.nan_to_num(shapely.get_y(cen), nan=0.0)
        gxmin, gymin, gxmax, gymax = global_extent or bbox

        X = _scale_to_uint(cx, gxmin, gxmax, sfc_bits)
        Y = _scale_to_uint(cy, gymin, gymax, sfc_bits)
        z = _interleave_bits_2d(X, Y, sfc_bits)

        max_code = np.uint64((1 << (2 * min(sfc_bits, 31))) - 1)
        zfull = np.where(valid, z, max_code).astype(np.uint64, copy=False)
        order = np.argsort(zfull, kind="mergesort")
        logger.debug(f"Sorting {N} rows by Z-order (sfc_bits={sfc_bits})")
        return bbox, tbl.take(pa.array(order, type=pa.int64())), per[order]

    return bbox, tbl, per


def _append_bbox_columns(tbl: pa.Table, per_bounds: np.ndarray) -> pa.Table:
    """Append the per-row bbox covering columns used for read-time pruning."""
    if tbl.num_rows == 0:
        return tbl
    for i, name in enumerate(BBOX_COLS):
        col = pa.array(np.ascontiguousarray(per_bounds[:, i], dtype=np.float64), type=pa.float64())
        tbl = tbl.append_column(name, col)
    return tbl


def _with_updated_geo_metadata(tbl: pa.Table, bbox: Tuple[float, float, float, float]) -> pa.Table:
    schema = tbl.schema
    meta = dict(schema.metadata or {})

    geo_raw = meta.get(b"geo")
    geo = {}
    if geo_raw is not None:
        try:
            geo = json.loads(geo_raw.decode("utf8"))
        except Exception:
            geo = {}

    # Update per-column bbox (correct place)
    col = geo.setdefault("columns", {}).setdefault("geometry", {})
    col["bbox"] = list(map(float, bbox))

    # Optional but allowed: update table-level bbox too
    geo["bbox"] = list(map(float, bbox))

    meta[b"geo"] = json.dumps(geo).encode("utf8")

    return tbl.replace_schema_metadata(meta)


def _finalize_one_tile(tile_id: int, batches: List[pa.Table], config: _WriterPoolConfig) -> str:
    label = f"tile_{tile_id:06d}"
    logger.debug(f"[{label}] Concatenating {len(batches)} batches.")
    full = pa.concat_tables(batches, promote=True)
    full = ensure_large_types(full, config.geom_col)

    # 🔽 Drop the internal routing column if it exists
    if "geo_parquet_tile_num" in full.column_names:
        logger.debug(f"[{label}] Dropping internal column 'geo_parquet_tile_num'")
        full = full.drop(["geo_parquet_tile_num"])

    bbox, full, per_bounds = _maybe_sort_and_bbox(
        full,
        geom_col=config.geom_col,
        sort_mode=config.sort_mode,
        sort_keys=config.sort_keys,
        sfc_bits=config.sfc_bits,
        global_extent=config.global_extent,
    )
    full = _append_bbox_columns(full, per_bounds)
    full = _with_updated_geo_metadata(full, bbox)
    minx, miny, maxx, maxy = bbox
    safe = lambda v: f"{v:.6f}".replace(".", "_")
    bbox_str = f"{safe(minx)}_{safe(miny)}_{safe(maxx)}_{safe(maxy)}"

    filename = f"{label}__{bbox_str}.parquet"
    out_path = os.path.join(config.outdir, filename)

    # Bound the row-group size so a spatially-sorted tile is split into several
    # spatially-coherent row groups; combined with the bbox columns this lets
    # the server skip row groups that don't intersect a requested tile.
    write_kwargs = dict(config.pq_args)
    if full.num_rows:
        write_kwargs.setdefault("row_group_size", min(full.num_rows, DEFAULT_ROW_GROUP_SIZE))

    logger.info(f"[{label}] Writing to disk → {out_path}")
    pq.write_table(full, out_path, compression=config.compression, **write_kwargs)
    logger.debug(f"[{label}] Flush complete, rows={full.num_rows}")
    return out_path


# ------------------------- Writer Pool -------------------------

class WriterPool:
    """
    Buffer-everything writer:
      - append(tile_id, table): buffer Arrow Tables per tile (no IO)
      - flush_all(): writes once, in rounds of up to `max_parallel_files` concurrent files
    """

    def __init__(
        self,
        outdir: str,
        compression: str = "zstd",
        geom_col: str = "geometry",
        max_parallel_files: Optional[int] = None,
        sort_mode: str = SortMode.ZORDER,
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]] = None,
        sfc_bits: int = 16,
        parquet_writer_args: Optional[dict] = None,
        global_extent: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.outdir = outdir
        self.compression = compression
        self.geom_col = geom_col
        self.sort_mode = sort_mode
        self._sort_keys = self._normalize_sort_keys(sort_keys)
        self.sfc_bits = int(sfc_bits)
        self._pq_args = dict(parquet_writer_args or {})
        self.global_extent = global_extent

        if max_parallel_files is None:
            cpu = max(1, multiprocessing.cpu_count())
            self.max_parallel_files = max(2, cpu // 2)
        else:
            self.max_parallel_files = max(1, int(max_parallel_files))

        self._buffers: Dict[int, List[pa.Table]] = defaultdict(list)

    # --------------------------- Public API ---------------------------

    def append(self, tile_id: int, table: pa.Table) -> None:
        logger.debug("\n--- DEBUG: WriterPool.append ---")
        logger.debug("Incoming metadata: %s", table.schema.metadata)

        if table is None or table.num_rows == 0:
            return
        if self.geom_col not in table.column_names:
            raise ValueError(f"WriterPool.append: missing geometry column '{self.geom_col}'")
        table = table.combine_chunks()
        table = ensure_large_types(table, self.geom_col)
        self._buffers[tile_id].append(table)

    def flush_all(self) -> None:
        if not self._buffers:
            logger.info("WriterPool.flush_all(): no buffered tiles to flush.")
            return

        os.makedirs(self.outdir, exist_ok=True)
        items = list(self._buffers.items())
        self._buffers.clear()

        total = len(items)
        mpf = min(self.max_parallel_files, total)
        rounds = math.ceil(total / mpf)
        logger.info(
            f"WriterPool.flush_all(): {total} tiles buffered → flushing in {rounds} round(s), "
            f"{mpf} parallel writes per round."
        )

        config = _WriterPoolConfig(
            geom_col=self.geom_col,
            sort_mode=self.sort_mode,
            sort_keys=list(self._sort_keys),
            sfc_bits=self.sfc_bits,
            global_extent=self.global_extent,
            compression=self.compression,
            pq_args=dict(self._pq_args),
            outdir=self.outdir,
        )

        for r in range(rounds):
            start = r * mpf
            batch = items[start : start + mpf]
            logger.info(f"WriterPool: round {r+1}/{rounds} — writing {len(batch)} tiles to disk.")
            if len(batch) == 1:
                tid, b = batch[0]
                _finalize_one_tile(tid, b, config)
                continue
            with ProcessPoolExecutor(max_workers=len(batch)) as ex:
                futs = {ex.submit(_finalize_one_tile, tid, b, config): tid for tid, b in batch}
                for f in as_completed(futs):
                    try:
                        _ = f.result()
                    except Exception as e:
                        logger.error(f"Error writing tile {futs[f]}: {e}")

        logger.info("WriterPool.flush_all(): all tiles successfully flushed to disk.")

    def close(self) -> None:
        self.flush_all()

    def set_sort_keys(self, sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]]) -> None:
        self._sort_keys = self._normalize_sort_keys(sort_keys)

    # ------------------------- Internal helpers -----------------------

    @staticmethod
    def _normalize_sort_keys(
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]]
    ) -> List[SortKey]:
        out: List[SortKey] = []
        if not sort_keys:
            return out
        for k in sort_keys:
            if isinstance(k, SortKey):
                out.append(k)
            elif isinstance(k, tuple):
                name, asc = k
                out.append(SortKey(str(name), bool(asc)))
            elif isinstance(k, str):
                out.append(SortKey(k, True))
            else:
                raise TypeError(f"Unsupported sort key type: {type(k)}")
        return out
