from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass, replace
import json
import logging
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from starlet._internal.executor import create_process_executor
from starlet._internal.tiling.datasource import (
    DataSource,
    SpatialSample,
    _GEOJSON_SUFFIXES,
    _attach_geoparquet_metadata,
    _combine_spatial_samples,
    _normalize_decimal_columns,
    _properties_dataframe_to_arrow_table,
    _unify_tabular_schemas,
    _reservoir_add,
    _spatial_sample_from_state,
    _split_sample_cap,
    _source_files,
)
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader

logger = logging.getLogger(__name__)


def is_geojson_path(path: str) -> bool:
    return path.lower().endswith(_GEOJSON_SUFFIXES)


@dataclass(frozen=True)
class GeoJSONSplit:
    """Byte range to read from one GeoJSON source."""

    path: str
    offset: int
    length: int


class GeoJSONSource(DataSource):
    """
    Streams GeoJSON / GeoJSONL as Arrow Tables, converting geometry to WKB.

    - For standard FeatureCollection GeoJSON, reads byte partitions in parallel.
    - For GeoJSON Lines (one Feature per line), reads and batches by line.
    - Geometry dicts → shapely.shape → WKB bytes (binary Arrow column 'geometry').
    - Attaches minimal GeoParquet metadata (version, primary_column, encoding, crs hint).
    """

    def __init__(
        self,
        path: str,
        batch_rows: int = 1_000,
        src_crs: str = "EPSG:4326",
        target_crs: Optional[str] = None,
        keep_null_geoms: bool = False,
    ):
        self.path = str(path)
        self._files = _source_files(self.path, _GEOJSON_SUFFIXES)
        if not self._files:
            raise ValueError(f"No GeoJSON files found in {self.path}")
        self.batch_rows = int(batch_rows)
        self.src_crs = src_crs
        self.target_crs = target_crs
        self.keep_null_geoms = keep_null_geoms

        if target_crs:
            logger.warning(
                "target_crs requested (%s) but GeoJSONSource preserves source CRS; "
                "projection is handled by downstream stages.",
                target_crs,
            )

        self._schema: Optional[pa.Schema] = None
        self._crs_hint: Optional[str] = _extract_feature_collection_crs_hint(str(self._files[0]))

        logger.info(
            "GeoJSONSource opened %s with %d files (batch_rows=%d, src_crs=%s)",
            path, len(self._files), self.batch_rows, self.src_crs
        )

    # ---------------- schema ---------------- #
    def schema(self) -> pa.Schema:
        if self._schema is None:
            property_schemas = []
            for features in self._iter_feature_batches_for_split(None):
                rows = [feature.get("properties") or {} for feature in features]
                props_df = _normalize_decimal_columns(pd.DataFrame.from_records(rows))
                property_schemas.append(
                    _properties_dataframe_to_arrow_table(props_df).schema
                )

            properties_schema = _unify_tabular_schemas(property_schemas)
            base = properties_schema.append(pa.field("geometry", pa.binary()))
            self._schema = _attach_geoparquet_metadata(
                base, self._crs_hint or self.src_crs
            )

        return self._schema

    def set_schema(self, schema: pa.Schema) -> None:
        """Use a schema discovered by an earlier scan of this source."""
        if "geometry" not in schema.names:
            raise ValueError("GeoJSON schema must contain a geometry column")
        self._schema = schema

    def input_size_bytes(self) -> int:
        return sum(file_path.stat().st_size for file_path in self._files)

    # ---------------- iterator ---------------- #
    def create_splits(self) -> List[GeoJSONSplit]:
        target_partition_size = 32 * 1024 * 1024
        splits: List[GeoJSONSplit] = []
        for file_path in self._files:
            file_size = file_path.stat().st_size
            num_splits = max(1, (file_size + target_partition_size - 1) // target_partition_size)
            splits.extend(
                GeoJSONSplit(path=str(file_path), offset=offset, length=length)
                for offset, length in _geojson_partition_ranges(file_size, int(num_splits))
            )
        return splits

    def iter_tables(self, split: Optional[GeoJSONSplit] = None) -> Iterable[pa.Table]:
        batch_index = 0
        crs_value = self._crs_hint or self.src_crs
        schema = self.schema()
        properties_schema = pa.schema(
            [field for field in schema if field.name != "geometry"]
        )

        import geopandas as gpd

        for features in self._iter_feature_batches_for_split(split):
            if not features:
                continue

            gdf = gpd.GeoDataFrame.from_features(features, crs=crs_value)
            geometry_col = pa.array(gdf.geometry.to_wkb(), type=pa.binary())

            props_df = gdf.drop(columns="geometry")
            props_df = _normalize_decimal_columns(props_df)
            props_table = _properties_dataframe_to_arrow_table(
                props_df,
                schema=properties_schema,
            )

            table = (
                pa.table([geometry_col], names=["geometry"])
                if props_table.num_columns == 0
                else props_table.append_column("geometry", geometry_col)
            )

            table = table.cast(schema)
            table = table.combine_chunks()

            logger.debug(
                "GeoJSON batch %d (%d rows) -> %d columns (including 'geometry')",
                batch_index,
                table.num_rows,
                len(table.column_names),
            )
            batch_index += 1
            yield table

    def _iter_feature_batches_for_split(
        self,
        split: Optional[GeoJSONSplit],
    ) -> Iterable[List[Dict[str, Any]]]:
        if split is None:
            for source_split in self.create_splits():
                yield from self._iter_feature_batches_for_split(source_split)
            return

        reader = GeoJSONPartitionReader(split.path, split.offset, split.length, batch_size=self.batch_rows)
        for feature_batch in reader.batches():
            yield [json.loads(feature) for feature in feature_batch]

    # ---------------- internal helpers ---------------- #
    @classmethod
    def read_spatial_sample(
        cls,
        path: str,
        *,
        sample_ratio: float,
        sample_cap: Optional[int],
        seed: int,
        workers: Optional[int],
        src_crs: str = "EPSG:4326",
    ) -> SpatialSample:
        source = cls(path, src_crs=src_crs)
        splits = source.create_splits()
        sample_caps = _split_sample_cap(sample_cap, len(splits))

        logger.info(
            "Reading GeoJSON spatial sample from %s in %d partitions with %s process workers",
            path,
            len(splits),
            workers or "auto",
        )

        with create_process_executor(
            max_workers=workers,
            logger=logger,
            context="GeoJSON spatial sampling",
        ) as ex:
            futures = [
                ex.submit(
                    _read_geojson_partition_spatial_sample,
                    split.path,
                    split.offset,
                    split.length,
                    sample_ratio,
                    sample_caps[idx],
                    seed + idx,
                )
                for idx, split in enumerate(splits)
            ]
            parts: List[SpatialSample] = []
            for future in as_completed(futures):
                parts.append(future.result())
            sample = _combine_spatial_samples(parts)
            properties_schema = _unify_tabular_schemas(
                part.schema for part in parts if part.schema is not None
            )
            schema = _attach_geoparquet_metadata(
                properties_schema.append(pa.field("geometry", pa.binary())),
                source._crs_hint or source.src_crs,
            )
            return replace(sample, schema=schema)


def _read_geojson_spatial_sample(
    path: str,
    *,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
    geojson_workers: Optional[int],
    src_crs: str = "EPSG:4326",
) -> SpatialSample:
    return GeoJSONSource.read_spatial_sample(
        path,
        sample_ratio=sample_ratio,
        sample_cap=sample_cap,
        seed=seed,
        workers=geojson_workers,
        src_crs=src_crs,
    )


def _iter_geojson_geometry_xy(geometry):
    stack = [geometry]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if v.get("type") == "GeometryCollection":
                stack.extend(reversed(v.get("geometries") or []))
            else:
                coordinates = v.get("coordinates")
                if coordinates is not None:
                    stack.append(coordinates)
        elif isinstance(v, list):
            if len(v) >= 2 and isinstance(v[0], Number) and isinstance(v[1], Number):
                yield float(v[0]), float(v[1])
            else:
                stack.extend(reversed(v))


def iter_geojson_xy(feature_json):
    feature = json.loads(feature_json)
    yield from _iter_geojson_geometry_xy(feature.get("geometry"))


def _read_geojson_partition_spatial_sample(
    path: str,
    offset: int,
    length: int,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
) -> SpatialSample:
    reader = GeoJSONPartitionReader(path, offset, length, batch_size=1_024)
    rng = np.random.default_rng(seed)
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    n_batches = 0
    property_schemas: List[pa.Schema] = []

    for batch in reader:
        property_rows = []
        for feature_json in batch:
            feature = json.loads(feature_json)
            property_rows.append(feature.get("properties") or {})
            first_point = True
            for x, y in _iter_geojson_geometry_xy(feature.get("geometry")):
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

                if first_point:
                    first_point = False
                    n_seen += 1
                    _reservoir_add(
                        rng=rng,
                        sample_cap=sample_cap,
                        sample_ratio=sample_ratio,
                        x_sample=x_sample,
                        y_sample=y_sample,
                        n_seen=n_seen,
                        x=x,
                        y=y,
                    )

        props_df = _normalize_decimal_columns(pd.DataFrame.from_records(property_rows))
        property_schemas.append(
            _properties_dataframe_to_arrow_table(props_df).schema
        )
        n_batches += 1

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        mins=np.array([min_x, min_y], dtype=float),
        maxs=np.array([max_x, max_y], dtype=float),
        n_seen=n_seen,
        batches_read=n_batches,
        schema=_unify_tabular_schemas(property_schemas),
    )


def _geometries_to_wkb(geometries: List[Any]) -> List[Any]:
    """
    Vectorized geometry -> WKB conversion using shapely's GeoJSON reader.

    Converting via shapely.geometry.shape per-feature is expensive for large
    files. Using shapely.from_geojson on an array of compact JSON strings keeps
    the heavy work inside GEOS and removes most Python-level loops.
    """
    from shapely import from_geojson, to_wkb

    wkb: List[Any] = [None] * len(geometries)
    non_null_idx: List[int] = []
    geojson_strings: List[str] = []

    for idx, geom in enumerate(geometries):
        if geom is None:
            continue
        non_null_idx.append(idx)
        geojson_strings.append(json.dumps(geom, separators=(",", ":")))

    if not geojson_strings:
        return wkb

    shapely_geoms = from_geojson(geojson_strings)
    encoded = to_wkb(shapely_geoms, hex=False).tolist()

    for idx, val in zip(non_null_idx, encoded):
        wkb[idx] = val

    return wkb


def _geojson_partition_ranges(file_size: int, num_splits: int) -> List[tuple[int, int]]:
    if file_size <= 0:
        return []

    num_splits = max(1, min(int(num_splits), file_size))
    partition_size = max(1, (file_size + num_splits - 1) // num_splits)
    ranges: List[tuple[int, int]] = []

    for offset in range(0, file_size, partition_size):
        ranges.append((offset, min(partition_size, file_size - offset)))

    return ranges


def _extract_feature_collection_crs_hint(buffer: str) -> Optional[str]:
    """
    Try to read the CRS from the header of a FeatureCollection without loading the whole file.
    Looks for a 'crs' object and returns its 'properties.name' if present.
    """
    if not buffer:
        return None

    idx = buffer.lower().find('"features"')
    if idx == -1:
        return None

    header = buffer[:idx]
    first_brace = header.find("{")
    if first_brace == -1:
        return None

    candidate = header[first_brace:]
    candidate = candidate.rstrip(", \r\n\t")
    candidate = candidate + "}"

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    crs = parsed.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties") or {}
        name = props.get("name")
        if isinstance(name, str):
            return name

    return None
