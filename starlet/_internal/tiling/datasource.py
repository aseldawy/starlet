from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import os
from queue import Queue
from typing import Iterable, Optional, List, Dict, Any, Tuple
import logging
import json
from pathlib import Path
from decimal import Decimal
import ijson
from numbers import Number
import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from shapely import from_wkb

from starlet._internal.tiling.RSGrove import EnvelopeNDLite
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader
from starlet._internal.tiling.utils_large import ensure_large_types

logger = logging.getLogger(__name__)
_QUEUE_SENTINEL = object()
_GEOPARQUET_SUFFIXES = (".parquet", ".geoparquet")
_GEOJSON_SUFFIXES = (".geojson", ".geojsonl", ".json", ".jsonl")


class DataSource:
    def schema(self) -> pa.Schema:
        raise NotImplementedError

    def create_splits(self, num_splits: Optional[int] = None) -> List[Any]:
        raise NotImplementedError

    def iter_tables(self, split: Optional[Any] = None) -> Iterable[pa.Table]:
        raise NotImplementedError

    def input_size_bytes(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class SpatialSample:
    """Centroid sample and global bounds prepared for spatial partitioning."""

    sample_points: np.ndarray
    mbr: EnvelopeNDLite
    total_seen: int
    total_sampled: int
    batches_read: int


# ------------------------- GeoParquet source ------------------------- #
@dataclass(frozen=True)
class GeoParquetSplit:
    """Row groups to read from one GeoParquet file."""

    path: str
    row_groups: Tuple[int, ...]


class GeoParquetSource(DataSource):
    def __init__(
        self,
        path: str,
        *,
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ):
        self.path = str(path)
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self._files = _source_files(self.path, _GEOPARQUET_SUFFIXES)
        if not self._files:
            raise ValueError(f"No GeoParquet files found in {self.path}")

        pf = pq.ParquetFile(str(self._files[0]))
        self._schema = pf.schema_arrow
        self._row_group_counts = {
            str(file_path): pq.ParquetFile(str(file_path)).num_row_groups
            for file_path in self._files
        }
        self._num_row_groups = sum(self._row_group_counts.values())
        if self.geometry_only and self.geom_col not in self._schema.names:
            raise ValueError(
                f"Geometry column {self.geom_col!r} was not found in {self.path}"
            )
        logger.info(
            "GeoParquetSource opened %s with %d files and %d row groups (geometry_only=%s)",
            path,
            len(self._files),
            self._num_row_groups,
            self.geometry_only,
        )

    def schema(self) -> pa.Schema:
        logger.debug("GeoParquet source schema metadata: %s", self._schema.metadata)
        return self._schema

    def input_size_bytes(self) -> int:
        return sum(file_path.stat().st_size for file_path in self._files)

    def create_splits(self, num_splits: Optional[int] = None) -> List[GeoParquetSplit]:
        row_groups = [
            (str(file_path), row_group)
            for file_path in self._files
            for row_group in range(self._row_group_counts[str(file_path)])
        ]
        if num_splits is None:
            return [
                GeoParquetSplit(path=path, row_groups=(row_group,))
                for path, row_group in row_groups
            ]

        split_count = max(1, min(int(num_splits), max(1, len(row_groups))))
        chunk_size = max(1, (len(row_groups) + split_count - 1) // split_count)
        splits: List[GeoParquetSplit] = []
        for file_path in self._files:
            groups = list(range(self._row_group_counts[str(file_path)]))
            for index in range(0, len(groups), chunk_size):
                splits.append(
                    GeoParquetSplit(
                        path=str(file_path),
                        row_groups=tuple(groups[index:index + chunk_size]),
                    )
                )
        return splits

    def iter_tables(
        self,
        split: Optional[GeoParquetSplit] = None,
        columns: Optional[List[str]] = None,
    ) -> Iterable[pa.Table]:
        selected_columns = [self.geom_col] if self.geometry_only else columns
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            pf = pq.ParquetFile(source_split.path)
            num_row_groups = self._row_group_counts.get(source_split.path, pf.num_row_groups)
            for row_group in source_split.row_groups:
                logger.debug(
                    "Reading row group %d/%d from %s",
                    row_group,
                    num_row_groups,
                    source_split.path,
                )
                yield pf.read_row_group(row_group, columns=selected_columns)


# ------------------------- Helpers ------------------------- #
def is_geojson_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(_GEOJSON_SUFFIXES)


def _source_files(path: str, suffixes: Tuple[str, ...]) -> List[Path]:
    source_path = Path(path)
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        return sorted(
            file_path
            for file_path in source_path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in suffixes
        )
    raise FileNotFoundError(f"Source path does not exist: {path}")


def _is_geojson_source(path: str) -> bool:
    source_path = Path(path)
    if source_path.is_file():
        return is_geojson_path(path)

    geojson_files = _source_files(path, _GEOJSON_SUFFIXES)
    geoparquet_files = _source_files(path, _GEOPARQUET_SUFFIXES)
    if geojson_files and not geoparquet_files:
        return True
    if geoparquet_files and not geojson_files:
        return False
    if geojson_files and geoparquet_files:
        raise ValueError(f"Source directory contains both GeoJSON and GeoParquet files: {path}")
    raise ValueError(f"No GeoJSON or GeoParquet files found in {path}")


def source_for_path(path: str, **geojson_kwargs) -> DataSource:
    """Create the appropriate source reader for a GeoParquet/GeoJSON path."""
    if _is_geojson_source(path):
        return GeoJSONSource(path, **geojson_kwargs)
    return GeoParquetSource(path)


def read_spatial_sample(
    path: str,
    *,
    geom_col: str = "geometry",
    sample_ratio: float = 1.0,
    sample_cap: Optional[int] = None,
    seed: int = 42,
    geojson_workers: Optional[int] = None,
    geojson_executor: str = "process",
    geoparquet_workers: Optional[int] = None,
) -> SpatialSample:
    """Read a file once and return centroid sample points plus the global MBR."""
    if _is_geojson_source(path):
        return _read_geojson_spatial_sample(
            path,
            sample_ratio=sample_ratio,
            sample_cap=sample_cap,
            seed=seed,
            geojson_workers=geojson_workers,
            geojson_executor=geojson_executor,
        )
    return _read_geoparquet_spatial_sample(
        path,
        geom_col=geom_col,
        sample_ratio=sample_ratio,
        sample_cap=sample_cap,
        seed=seed,
        geoparquet_workers=geoparquet_workers,
    )


def _reservoir_add(
    *,
    rng: np.random.Generator,
    sample_cap: Optional[int],
    sample_ratio: float,
    x_sample: List[float],
    y_sample: List[float],
    n_seen: int,
    x: float,
    y: float,
) -> None:
    if sample_cap is None:
        if rng.random() < sample_ratio:
            x_sample.append(x)
            y_sample.append(y)
        return

    if sample_cap <= 0:
        return

    if n_seen <= sample_cap:
        if len(x_sample) < sample_cap:
            x_sample.append(x)
            y_sample.append(y)
        else:
            j = rng.integers(0, n_seen)
            if j < sample_cap:
                x_sample[j] = x
                y_sample[j] = y
    else:
        j = rng.integers(0, n_seen)
        if j < sample_cap:
            x_sample[j] = x
            y_sample[j] = y


def _combine_spatial_samples(parts: List[SpatialSample]) -> SpatialSample:
    logger.info("Finished the partitions ... merging")
    non_empty = [part for part in parts if part.total_seen > 0]
    if not non_empty:
        raise ValueError(
            "No geometries sampled to build RSGrove index. "
            "Increase --sample-ratio or provide --sample-cap."
        )

    sampled = [part.sample_points for part in non_empty if part.sample_points.shape[1] > 0]
    if not sampled:
        raise ValueError(
            "No geometries sampled to build RSGrove index. "
            "Increase --sample-ratio or provide --sample-cap."
        )

    mins = np.minimum.reduce([part.mbr.mins for part in non_empty])
    maxs = np.maximum.reduce([part.mbr.maxs for part in non_empty])
    sample_points = np.concatenate(sampled, axis=1)
    logger.info("Finished the merge")
    return SpatialSample(
        sample_points=sample_points,
        mbr=EnvelopeNDLite(mins, maxs),
        total_seen=sum(part.total_seen for part in parts),
        total_sampled=sample_points.shape[1],
        batches_read=sum(part.batches_read for part in parts),
    )


def _split_sample_cap(sample_cap: Optional[int], num_parts: int) -> List[Optional[int]]:
    if sample_cap is None:
        return [None] * num_parts

    total = max(0, int(sample_cap))
    base, remainder = divmod(total, max(1, num_parts))
    return [base + (1 if i < remainder else 0) for i in range(num_parts)]


def _spatial_sample_from_state(
    *,
    x_sample: List[float],
    y_sample: List[float],
    mins: np.ndarray,
    maxs: np.ndarray,
    n_seen: int,
    batches_read: int,
) -> SpatialSample:
    return SpatialSample(
        sample_points=(
            np.stack(
                [np.asarray(x_sample, dtype=np.float64), np.asarray(y_sample, dtype=np.float64)],
                axis=0,
            )
            if x_sample
            else np.empty((2, 0), dtype=np.float64)
        ),
        mbr=EnvelopeNDLite(mins, maxs),
        total_seen=n_seen,
        total_sampled=len(x_sample),
        batches_read=batches_read,
    )


def _read_geoparquet_spatial_sample(
    path: str,
    *,
    geom_col: str,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
    geoparquet_workers: Optional[int],
) -> SpatialSample:
    """Sample GeoParquet row-group splits in parallel processes."""
    source = GeoParquetSource(path, geometry_only=True, geom_col=geom_col)
    splits = source.create_splits()
    sample_caps = _split_sample_cap(sample_cap, len(splits))

    logger.info(
        "Reading GeoParquet spatial sample from %s in %d row-group partitions with %s process workers",
        path,
        len(splits),
        geoparquet_workers or "auto",
    )

    with ProcessPoolExecutor(max_workers=geoparquet_workers) as executor:
        futures = [
            executor.submit(
                _read_geoparquet_split_spatial_sample,
                path,
                split,
                geom_col,
                sample_ratio,
                sample_caps[index],
                seed + index,
            )
            for index, split in enumerate(splits)
        ]
        return _combine_spatial_samples([future.result() for future in futures])


def _read_geoparquet_split_spatial_sample(
    path: str,
    split: GeoParquetSplit,
    geom_col: str,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
) -> SpatialSample:
    """Read one GeoParquet row-group split for parallel spatial sampling."""
    source = GeoParquetSource(path, geometry_only=True, geom_col=geom_col)
    rng = np.random.default_rng(seed)
    mins = np.array([+np.inf, +np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf], dtype=np.float64)
    x_sample: List[float] = []
    y_sample: List[float] = []
    n_seen = 0
    n_batches = 0

    for table in source.iter_tables(split):
        table = table.combine_chunks()
        if table.num_rows == 0:
            continue
        n_batches += 1
        table = ensure_large_types(table, geom_col)
        geometries = from_wkb(table[geom_col].to_numpy(zero_copy_only=False))

        for geom in geometries:
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            if minx < mins[0]:
                mins[0] = minx
            if miny < mins[1]:
                mins[1] = miny
            if maxx > maxs[0]:
                maxs[0] = maxx
            if maxy > maxs[1]:
                maxs[1] = maxy

            centroid = geom.centroid
            n_seen += 1
            _reservoir_add(
                rng=rng,
                sample_cap=sample_cap,
                sample_ratio=sample_ratio,
                x_sample=x_sample,
                y_sample=y_sample,
                n_seen=n_seen,
                x=float(centroid.x),
                y=float(centroid.y),
            )

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        mins=mins,
        maxs=maxs,
        n_seen=n_seen,
        batches_read=n_batches,
    )


def _read_geojson_spatial_sample(
    path: str,
    *,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
    geojson_workers: Optional[int],
    geojson_executor: str,
) -> SpatialSample:
    source = GeoJSONSource(path)
    splits = source.create_splits()
    sample_caps = _split_sample_cap(sample_cap, len(splits))

    executor_cls = _geojson_executor_class(geojson_executor)
    logger.info(
        "Reading GeoJSON spatial sample from %s in %d partitions with %s %s workers",
        path,
        len(splits),
        geojson_workers or "auto",
        geojson_executor,
    )

    with executor_cls(max_workers=geojson_workers) as executor:
        futures = [
            executor.submit(
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
        return _combine_spatial_samples([future.result() for future in futures])


def _geojson_executor_class(kind: str):
    normalized = kind.strip().lower()
    if normalized in {"process", "processes", "multiprocessing"}:
        return ProcessPoolExecutor
    if normalized in {"thread", "threads", "threading"}:
        return ThreadPoolExecutor
    raise ValueError(
        "geojson_executor must be 'process' or 'thread' "
        f"(got {kind!r})"
    )


def _iter_geojson_xy(feature_json):
    try:
        geometry = next(ijson.items(io.BytesIO(feature_json.encode("utf-8")), "geometry", use_float=True), None)
    except:
        print("Failed to parse feature_json:")
        raise

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

    for batch in reader:
        for feature_json in batch:
            first_point = True
            for x, y in _iter_geojson_xy(feature_json):
                # Update MBR
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

        n_batches += 1

    return _spatial_sample_from_state(
        x_sample=x_sample,
        y_sample=y_sample,
        mins=np.array([min_x, min_y], dtype=float),
        maxs=np.array([max_x, max_y], dtype=float),
        n_seen=n_seen,
        batches_read=n_batches,
    )


def _attach_geoparquet_metadata(schema: pa.Schema, crs_hint: Optional[str]) -> pa.Schema:
    """
    Return a copy of `schema` with a minimal GeoParquet 'geo' JSON block so
    downstream writers (WriterPool) can inject tile bbox.

    Includes:
      - version: 1.1.0
      - primary_column: geometry
      - columns.geometry.encoding: WKB
      - columns.geometry.crs: <crs_hint> (string hint if provided)
    """
    md = dict(schema.metadata or {})
    if b"geo" in md:
        return pa.schema(schema, metadata=md)

    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB"}},
    }
    if crs_hint:
        try:
            geo["columns"]["geometry"]["crs"] = crs_hint
        except Exception:
            pass

    md[b"geo"] = json.dumps(geo, separators=(",", ":")).encode("utf-8")
    return pa.schema(schema, metadata=md)


def _normalize_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize object columns that commonly infer unstable Arrow types across
    GeoJSON batches.

    Decimal values become float64 so Arrow does not infer different
    decimal128 precision/scale. Nested JSON-like values become compact JSON
    strings so dynamic tag maps do not infer a different struct field set for
    every batch.
    """
    if df.empty:
        return df

    df = df.copy()

    def is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (dict, list)):
            return False
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    for col in df.columns:
        s = df[col]

        # Only object dtype can hold Decimal values in pandas here.
        if s.dtype != "object":
            continue

        sample = None
        for v in s:
            if not is_missing(v):
                sample = v
                break

        if isinstance(sample, Decimal):
            df[col] = s.map(lambda x: None if is_missing(x) else float(x))
        elif isinstance(sample, (dict, list)):
            df[col] = s.map(
                lambda x: json.dumps(x, separators=(",", ":"), sort_keys=True, default=str)
                if not is_missing(x)
                else None
            )

    return df


# ------------------------- GeoJSON source (streaming → Arrow) ------------------------- #
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
        self.target_crs = target_crs  # informational only here
        self.keep_null_geoms = keep_null_geoms

        if target_crs:
            logger.warning(
                "target_crs requested (%s) but GeoJSON reader does not reproject on the fly; data will be read as-is.",
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
            first = self._read_first_batch()
            if first is None or first.num_rows == 0:
                # Empty input file. Create a minimal schema with geometry column.
                base = pa.schema([("geometry", pa.binary())])
                self._schema = _attach_geoparquet_metadata(
                    base, self._crs_hint or self.target_crs or self.src_crs
                )
            else:
                # Lock schema with GeoParquet metadata
                self._schema = _attach_geoparquet_metadata(
                    first.schema, self._crs_hint or self.target_crs or self.src_crs
                )

        return self._schema

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
        crs_value = self._crs_hint or self.target_crs or self.src_crs

        import geopandas as gpd

        for features in self._iter_feature_batches_for_split(split):
            if not features:
                continue

            gdf = gpd.GeoDataFrame.from_features(features, crs=crs_value)
            geometry_col = pa.array(gdf.geometry.to_wkb(), type=pa.binary())

            props_df = gdf.drop(columns="geometry")
            props_df = _normalize_decimal_columns(props_df)
            props_table = pa.Table.from_pandas(props_df, preserve_index=False)

            table = (
                pa.table([geometry_col], names=["geometry"])
                if props_table.num_columns == 0
                else props_table.append_column("geometry", geometry_col)
            )

            # Attach GeoParquet metadata with CRS
            schema_with_geo = _attach_geoparquet_metadata(table.schema, crs_value)
            table = table.replace_schema_metadata(schema_with_geo.metadata)

            if split is None and not self._schema:
                self._schema = schema_with_geo

            if self._schema is not None:
                table = self._coerce_to_schema(table, self._schema)
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
    def _read_first_batch(self) -> Optional[pa.Table]:
        """Read the first batch of features to establish the schema."""
        first_path = self._files[0]
        file_size = first_path.stat().st_size
        batches = GeoJSONPartitionReader(first_path, 0, file_size, batch_size=max(1, self.batch_rows)).batches()
        try:
            first_batch = next(batches)
            features = [json.loads(feature_str) for feature_str in first_batch]
        except StopIteration:
            logger.info("GeoJSON read returned 0 rows when inferring schema")
            return None
        finally:
            batches.close()

        rows_props: List[Dict[str, Any]] = []
        geometries: List[Any] = []

        for feat in features:
            rows_props.append(feat.get("properties") or {})
            geometries.append(feat.get("geometry", None))

        props_df = pd.DataFrame.from_records(rows_props)
        props_df = _normalize_decimal_columns(props_df)
        props_table = pa.Table.from_pandas(props_df, preserve_index=False)

        wkb_list = _geometries_to_wkb(geometries)
        geometry_col = pa.array(wkb_list, type=pa.binary())

        if props_table.num_columns == 0:
            return pa.table([geometry_col], names=["geometry"])

        return props_table.append_column("geometry", geometry_col)

    def _coerce_to_schema(self, t: pa.Table, schema: pa.Schema) -> pa.Table:
        if t.schema.equals(schema):
            return t

        out_cols = []
        for fld in schema:
            name = fld.name
            if name in t.column_names:
                col = t[name]
                if not col.type.equals(fld.type):
                    try:
                        col = col.cast(fld.type)
                    except Exception:
                        logger.warning(
                            "Type mismatch for column '%s': %s -> %s (kept original)",
                            name, col.type, fld.type
                        )
                out_cols.append(col)
            else:
                out_cols.append(pa.nulls(t.num_rows, type=fld.type))

        return pa.table(out_cols, names=[f.name for f in schema])


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


def _geojson_partition_ranges(file_size: int, num_splits: int) -> List[Tuple[int, int]]:
    if file_size <= 0:
        return []

    num_splits = max(1, min(int(num_splits), file_size))
    partition_size = max(1, (file_size + num_splits - 1) // num_splits)
    ranges: List[Tuple[int, int]] = []

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
