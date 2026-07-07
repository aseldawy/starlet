"""Two-stage dataset-to-MVT generator using intermediate vector tiles."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import logging
import math
import multiprocessing
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import shapely
from shapely import from_wkb

from starlet._internal.histogram.loader import HistogramLoader
from starlet._internal.config import resolve_temp_dir
from starlet._internal.mvt.helpers import (
    WORLD_MAXX,
    WORLD_MAXY,
    WORLD_MINX,
    WORLD_MINY,
    mercator_tile_bounds,
)
from starlet._internal.mvt.intermediate_tile import IntermediateVectorTile
from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner
from starlet._internal.pmtiles.exporter import export_to_pmtiles
from starlet._internal.progress import iter_with_progress
from starlet._internal.server.tiler.parquet_index import ParquetIndex
from starlet._internal.tiling.crs import WEB_MERCATOR_CRS, WGS84_CRS, geoparquet_crs, reproject_geometries
from starlet._internal.tiling.geoparquet_source import GeoParquetSource, GeoParquetSplit

logger = logging.getLogger(__name__)

_INTERNAL_ATTRIBUTE_COLUMNS = {
    "_tile_id",
    "_bbox_xmin",
    "_bbox_ymin",
    "_bbox_xmax",
    "_bbox_ymax",
}
_SINGLE_TILE_INDEX_CACHE_SIZE = 16
_single_tile_index_cache: "OrderedDict[str, ParquetIndex]" = OrderedDict()


@dataclass(frozen=True)
class DatasetMVTGenerationResult:
    outdir: str
    tile_count: int
    zoom_levels: list[int]
    pmtiles_path: str | None = None


@dataclass(frozen=True)
class _MapStageResult:
    intermediate_dir: str
    tile_ids: list[int]


@dataclass(frozen=True)
class _ReduceTileInput:
    tile_id: int
    intermediate_dirs: tuple[str, ...]


@dataclass(frozen=True)
class _TableBatch:
    table: pa.Table


_MapInput = GeoParquetSplit | _TableBatch


class DatasetMVTGenerator:
    """Generate MVT tiles from a Starlet tiled dataset.

    This class is intentionally separate from the existing streaming MVT
    generator while the intermediate-tile workflow is developed.
    """

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_zoom_levels: int,
        threshold: float,
        output_format: str = "mvt",
        outdir: str | None = None,
        pmtiles_path: str | None = None,
        pmtiles_compression: str = "gzip",
        workers: int | None = None,
        feature_capacity: int = 10_000,
        extent: int = 4096,
        buffer: int = 256,
        geom_col: str = "geometry",
        seed: int = 42,
        temp_dir: str | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.parquet_dir = self.dataset_dir / "parquet_tiles"
        self.hist_path = self.dataset_dir / "histograms" / "global_prefix.npy"
        self.num_zoom_levels = int(num_zoom_levels)
        self.threshold = float(threshold)
        self.output_format = output_format.strip().lower()
        self.outdir = Path(outdir) if outdir is not None else self.dataset_dir / "mvt"
        self.pmtiles_path = Path(pmtiles_path) if pmtiles_path is not None else self.dataset_dir.with_suffix(".pmtiles")
        self.pmtiles_compression = pmtiles_compression
        cpu_default = max(1, multiprocessing.cpu_count() - 1)
        self.workers = max(1, int(workers or cpu_default))
        self.feature_capacity = int(feature_capacity)
        self.extent = int(extent)
        self.buffer = int(buffer)
        self.partition_buffer = float(self.buffer) / float(self.extent)
        self.geom_col = geom_col
        self.seed = int(seed)
        self.temp_dir = temp_dir

        if self.num_zoom_levels <= 0:
            raise ValueError("num_zoom_levels must be positive")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.output_format not in {"mvt", "pmtiles"}:
            raise ValueError("output_format must be 'mvt' or 'pmtiles'")
        if self.extent <= 0:
            raise ValueError("extent must be positive")

    def run(self) -> DatasetMVTGenerationResult:
        if not self.parquet_dir.is_dir():
            raise FileNotFoundError(f"GeoParquet tile directory not found: {self.parquet_dir}")
        if not self.hist_path.exists():
            raise FileNotFoundError(f"Prefix histogram not found: {self.hist_path}")

        source = GeoParquetSource(str(self.parquet_dir), geom_col=self.geom_col)
        map_groups = _create_map_groups(source, self.workers)
        if not map_groups:
            return DatasetMVTGenerationResult(str(self.outdir), 0, [], None)

        temp_parent = resolve_temp_dir(self.temp_dir, self.dataset_dir / "tmp")
        with tempfile.TemporaryDirectory(prefix="starlet_mvt_", dir=temp_parent) as temp_dir:
            map_results = self._run_map_stage(map_groups, source, Path(temp_dir))
            self._run_reduce_stage(map_results)

        pmtiles_path = None
        if self.output_format == "pmtiles":
            pmtiles_path = str(self.pmtiles_path)
            export_to_pmtiles(
                mvt_dir=str(self.outdir),
                output_path=pmtiles_path,
                tile_type="mvt",
                compression=self.pmtiles_compression,
            )

        zoom_levels = _discover_zoom_levels(self.outdir)
        tile_count = len(list(self.outdir.rglob("*.mvt"))) if self.outdir.exists() else 0
        return DatasetMVTGenerationResult(
            outdir=str(self.outdir),
            tile_count=tile_count,
            zoom_levels=zoom_levels,
            pmtiles_path=pmtiles_path,
        )

    def _run_map_stage(
        self,
        map_groups: Sequence[Sequence[_MapInput]],
        source: GeoParquetSource,
        temp_root: Path,
    ) -> list[_MapStageResult]:
        logger.info(
            "DatasetMVTGenerator map stage: groups=%d workers=%d",
            len(map_groups),
            self.workers,
        )
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    _map_split_group,
                    group,
                    source,
                    str(self.hist_path),
                    self.num_zoom_levels,
                    self.threshold,
                    self.partition_buffer,
                    self.feature_capacity,
                    self.extent,
                    self.buffer,
                    self.seed + index,
                    str(temp_root),
                    index,
                )
                for index, group in enumerate(map_groups)
            ]
            return [
                future.result()
                for future in iter_with_progress(
                    as_completed(futures),
                    total=len(futures),
                    logger=logger,
                    label="dataset-mvt: map",
                )
            ]

    def _run_reduce_stage(self, map_results: list[_MapStageResult]) -> None:
        if not map_results:
            logger.info("DatasetMVTGenerator reduce stage: no intermediate tiles")
            return

        self.outdir.mkdir(parents=True, exist_ok=True)
        tile_locations: dict[int, list[str]] = defaultdict(list)
        for result in map_results:
            for tile_id in result.tile_ids:
                tile_locations[tile_id].append(result.intermediate_dir)

        reduce_groups: list[list[_ReduceTileInput]] = [
            [] for _ in range(self.workers)
        ]
        for tile_id, intermediate_dirs in tile_locations.items():
            group_index = tile_id % self.workers
            reduce_groups[group_index].append(
                _ReduceTileInput(tile_id, tuple(intermediate_dirs))
            )
        logger.info(
            "DatasetMVTGenerator reduce stage: tile_ids=%d groups=%d workers=%d",
            len(tile_locations),
            len(reduce_groups),
            self.workers,
        )
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    _reduce_tile_group,
                    tuple(group),
                    str(self.outdir),
                    self.feature_capacity,
                    self.extent,
                    self.buffer,
                )
                for group in reduce_groups
                if group
            ]
            for future in iter_with_progress(
                as_completed(futures),
                total=len(futures),
                logger=logger,
                label="dataset-mvt: reduce",
            ):
                future.result()


def _map_split_group(
    inputs: Sequence[_MapInput],
    source: GeoParquetSource,
    hist_path: str,
    num_zoom_levels: int,
    threshold: float,
    partition_buffer: float,
    feature_capacity: int,
    extent: int,
    buffer: int,
    seed: int,
    temp_root: str,
    mapper_index: int,
) -> _MapStageResult:
    prefix = HistogramLoader(hist_path).load()
    partitioner = PyramidPartitioner(
        (WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY),
        num_zoom_levels,
        prefix_histogram=prefix,
        size_threshold=threshold,
        buffer=partition_buffer,
    )
    tiles: dict[int, IntermediateVectorTile] = {}

    for table in _iter_map_input_tables(source, inputs):
        for geom, attrs in _iter_web_mercator_features(table, source.geom_col):
            bounds = _positive_bounds_tuple(geom.bounds)
            tile_ids = partitioner.overlapping_tile_ids(bounds)
            if not tile_ids:
                continue
            for tile_id in tile_ids:
                tile = tiles.get(tile_id)
                if tile is None:
                    z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
                    tile = IntermediateVectorTile(
                        z,
                        x,
                        y,
                        feature_capacity=feature_capacity,
                        extent=extent,
                        buffer=buffer,
                        rng=random.Random(seed + tile_id),
                    )
                    tiles[tile_id] = tile
                tile.add_feature(
                    geom,
                    attrs,
                )
    intermediate_dir = Path(temp_root) / f"mapper-{mapper_index:06d}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    tile_ids = []
    for tile_id, tile in tiles.items():
        if tile.feature_count == 0:
            continue
        z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
        tile.write_features(intermediate_dir / _intermediate_tile_filename(z, x, y))
        tile_ids.append(tile_id)
    return _MapStageResult(str(intermediate_dir), tile_ids)


def _iter_map_input_tables(
    source: GeoParquetSource,
    inputs: Sequence[_MapInput],
) -> Iterable[pa.Table]:
    for item in inputs:
        if isinstance(item, _TableBatch):
            yield item.table
        else:
            yield from source.iter_tables(item)


def _reduce_tile_group(
    reduce_inputs: Sequence[_ReduceTileInput],
    outdir: str,
    feature_capacity: int,
    extent: int,
    buffer: int,
) -> None:
    out_path = Path(outdir)
    for reduce_input in reduce_inputs:
        tile_id = reduce_input.tile_id
        z, x, y = PyramidPartitioner.decode_tile_id(tile_id)
        filename = _intermediate_tile_filename(z, x, y)
        merged = IntermediateVectorTile(
            z,
            x,
            y,
            feature_capacity=feature_capacity,
            extent=extent,
            buffer=buffer,
            rng=random.Random(tile_id),
        )
        first_tile = True
        for intermediate_dir in reduce_input.intermediate_dirs:
            path = Path(intermediate_dir) / filename
            if not path.exists():
                continue
            if first_tile:
                merged.load_features(path)
                first_tile = False
            else:
                partial = IntermediateVectorTile(
                    z,
                    x,
                    y,
                    feature_capacity=feature_capacity,
                    extent=extent,
                    buffer=buffer,
                )
                partial.load_features(path)
                merged.merge(partial)

        if merged.feature_count == 0:
            continue
        x_dir = out_path / str(z) / str(x)
        x_dir.mkdir(parents=True, exist_ok=True)
        with open(x_dir / f"{y}.mvt", "wb") as output:
            output.write(merged.encode())


def _intermediate_tile_filename(z: int, x: int, y: int) -> str:
    return f"{z}-{x}-{y}.pyarrow"


def generate_single_mvt_tile(
    dataset_path: str,
    tile_id: tuple[int, int, int],
    *,
    feature_capacity: int = 10_000,
    extent: int = 4096,
    buffer: int = 256,
    layer_name: str = "layer0",
) -> bytes:
    """Generate one MVT tile directly from an indexed Starlet dataset."""
    dataset_dir = Path(dataset_path)
    parquet_dir = dataset_dir / "parquet_tiles"
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"GeoParquet tile directory not found: {parquet_dir}")

    z, x, y = tile_id
    tile_bounds = mercator_tile_bounds(int(z), int(x), int(y))
    index = _single_tile_parquet_index(parquet_dir)
    tile_bounds_4326 = index._transform_bbox(tile_bounds, WEB_MERCATOR_CRS, WGS84_CRS)
    tile = IntermediateVectorTile(
        int(z),
        int(x),
        int(y),
        feature_capacity=feature_capacity,
        extent=extent,
        buffer=buffer,
    )

    for gdf in index.iter_query_batches(tile_bounds_4326, target_crs=WEB_MERCATOR_CRS):
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            attrs = {
                column: _property_value(value)
                for column, value in row.items()
                if column != "geometry" and value is not None
            }
            tile.add_feature(
                geom,
                attrs,
            )

    return tile.encode(layer_name=layer_name)


def _single_tile_parquet_index(parquet_dir: Path) -> ParquetIndex:
    key = str(parquet_dir.resolve())
    index = _single_tile_index_cache.get(key)
    if index is not None:
        _single_tile_index_cache.move_to_end(key)
        return index

    index = ParquetIndex(parquet_dir)
    _single_tile_index_cache[key] = index
    _single_tile_index_cache.move_to_end(key)
    while len(_single_tile_index_cache) > _SINGLE_TILE_INDEX_CACHE_SIZE:
        _single_tile_index_cache.popitem(last=False)
    return index


def _iter_web_mercator_features(table: Any, geom_col: str) -> Iterable[tuple[Any, dict[str, Any]]]:
    source_crs = geoparquet_crs(table.schema, geom_col) or WGS84_CRS
    geometries = from_wkb(table[geom_col].to_numpy(zero_copy_only=False))
    geometries = shapely.make_valid(geometries)
    geometries, _ = reproject_geometries(geometries, source_crs, WEB_MERCATOR_CRS)

    attr_columns = [
        column
        for column in table.column_names
        if column != geom_col and column not in _INTERNAL_ATTRIBUTE_COLUMNS
    ]
    attrs_by_column = {column: table[column].to_pylist() for column in attr_columns}

    for index, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue
        attrs = {
            column: _property_value(values[index])
            for column, values in attrs_by_column.items()
            if values[index] is not None
        }
        yield geom, attrs


def _positive_bounds_tuple(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = map(float, bounds)
    if maxx <= minx:
        maxx = np.nextafter(minx, math.inf)
    if maxy <= miny:
        maxy = np.nextafter(miny, math.inf)
    return minx, miny, maxx, maxy


def _property_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _group_splits(splits: Sequence[GeoParquetSplit], num_groups: int) -> list[list[GeoParquetSplit]]:
    if not splits:
        return []
    group_count = max(1, min(int(num_groups), len(splits)))
    groups: list[list[GeoParquetSplit]] = [[] for _ in range(group_count)]
    for index, split in enumerate(splits):
        groups[index % group_count].append(split)
    return groups


def _create_map_groups(source: GeoParquetSource, num_groups: int) -> list[list[_MapInput]]:
    splits = source.create_splits()
    if len(splits) >= max(1, int(num_groups)):
        return _group_splits(splits, num_groups)

    tables = [table for split in splits for table in source.iter_tables(split)]
    if not tables:
        return []

    table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
    logger.info(
        "DatasetMVTGenerator map fallback: %d row-group splits for %d workers; "
        "loaded %d rows into memory and repartitioned by row batches",
        len(splits),
        max(1, int(num_groups)),
        table.num_rows,
    )
    return _group_table_batches(table, num_groups)


def _group_table_batches(table: pa.Table, num_groups: int) -> list[list[_TableBatch]]:
    if table.num_rows == 0:
        return []
    group_count = max(1, min(int(num_groups), table.num_rows))
    batch_size = max(1, (table.num_rows + group_count - 1) // group_count)
    groups: list[list[_TableBatch]] = []
    for start in range(0, table.num_rows, batch_size):
        groups.append([_TableBatch(table.slice(start, min(batch_size, table.num_rows - start)))])
    return groups


def _bucket_tile_ids(tile_ids: Sequence[int], num_groups: int) -> list[list[int]]:
    group_count = max(1, int(num_groups))
    groups: list[list[int]] = [[] for _ in range(group_count)]
    for tile_id in tile_ids:
        groups[tile_id % group_count].append(tile_id)
    return groups


def _discover_zoom_levels(outdir: Path) -> list[int]:
    if not outdir.exists():
        return []
    levels = []
    for child in outdir.iterdir():
        if child.is_dir() and child.name.isdigit():
            levels.append(int(child.name))
    return sorted(levels)
