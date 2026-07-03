from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
import numpy as np
from shapely import from_wkb

from starlet._internal.tiling.RSGrove import EnvelopeNDLite
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader
from starlet._internal.tiling.utils_large import ensure_large_types
from starlet._internal.progress import iter_with_progress

logger = logging.getLogger(__name__)
_GEOPARQUET_SUFFIXES = (".parquet", ".geoparquet")
_GEOJSON_SUFFIXES = (".geojson", ".geojsonl", ".json", ".jsonl")
_CSV_SUFFIXES = (".csv",)
_SHAPEFILE_SUFFIXES = (".shp",)
_ZIP_SUFFIXES = (".zip",)
_GDB_SUFFIXES = (".gdb",)


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



# ------------------------- Helpers ------------------------- #
def is_geojson_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(_GEOJSON_SUFFIXES)


def is_csv_path(path: str) -> bool:
    p = path.lower()
    return p.endswith(_CSV_SUFFIXES)


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
    return _source_kind(path) == "geojson"


def _source_kind(path: str) -> str:
    source_path = Path(path)
    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in _GEOJSON_SUFFIXES:
            return "geojson"
        if suffix in _GEOPARQUET_SUFFIXES:
            return "geoparquet"
        if suffix in _CSV_SUFFIXES:
            return "csv"
        if suffix in _SHAPEFILE_SUFFIXES or suffix in _ZIP_SUFFIXES:
            return "shapefile"
        raise ValueError(f"Unsupported source file type: {path}")

    if source_path.is_dir() and source_path.suffix.lower() in _GDB_SUFFIXES:
        return "gdb"

    source_types = []
    if _source_files(path, _GEOJSON_SUFFIXES):
        source_types.append("geojson")
    if _source_files(path, _GEOPARQUET_SUFFIXES):
        source_types.append("geoparquet")
    if _source_files(path, _CSV_SUFFIXES):
        source_types.append("csv")
    if _source_files(path, _SHAPEFILE_SUFFIXES) or _source_files(path, _ZIP_SUFFIXES):
        source_types.append("shapefile")
    if sorted(child for child in source_path.rglob("*.gdb") if child.is_dir()):
        source_types.append("gdb")

    if len(source_types) == 1:
        return source_types[0]
    if source_types:
        raise ValueError(f"Source directory contains multiple supported source types: {path}")
    raise ValueError(f"No supported geospatial files found in {path}")


def source_for_path(
    path: str,
    *,
    geom_col: str = "geometry",
    csv_x_col: str | None = None,
    csv_y_col: str | None = None,
    csv_wkt_col: str | None = None,
    csv_split_size: int = 32 * 1024 * 1024,
    csv_batch_rows: int | None = None,
    src_crs: str = "EPSG:4326",
    **geojson_kwargs,
) -> DataSource:
    """Create the appropriate source reader for a supported geospatial path."""
    kind = _source_kind(path)
    if kind == "geojson":
        return GeoJSONSource(path, **geojson_kwargs)
    if kind == "geoparquet":
        return GeoParquetSource(path)
    if kind == "csv":
        return CSVSource(
            path,
            x_col=csv_x_col,
            y_col=csv_y_col,
            wkt_col=csv_wkt_col,
            split_size=csv_split_size,
            batch_rows=csv_batch_rows,
            src_crs=src_crs,
            geom_col=geom_col,
        )
    if kind == "shapefile":
        return ShapefileSource(path, geom_col=geom_col)
    if kind == "gdb":
        return GDBSource(path, geom_col=geom_col)
    raise ValueError(f"Unsupported source: {path}")


def read_spatial_sample(
    path: str,
    *,
    geom_col: str = "geometry",
    csv_x_col: str | None = None,
    csv_y_col: str | None = None,
    csv_wkt_col: str | None = None,
    csv_split_size: int = 32 * 1024 * 1024,
    csv_batch_rows: int | None = None,
    src_crs: str = "EPSG:4326",
    sample_ratio: float = 1.0,
    sample_cap: Optional[int] = None,
    seed: int = 42,
    geojson_workers: Optional[int] = None,
    geojson_executor: str = "process",
    geoparquet_workers: Optional[int] = None,
    source_workers: Optional[int] = None,
) -> SpatialSample:
    """Read a file once and return centroid sample points plus the global MBR."""
    kind = _source_kind(path)
    if kind == "geojson":
        return _read_geojson_spatial_sample(
            path,
            sample_ratio=sample_ratio,
            sample_cap=sample_cap,
            seed=seed,
            geojson_workers=geojson_workers,
            geojson_executor=geojson_executor,
        )
    if kind == "geoparquet":
        return _read_geoparquet_spatial_sample(
            path,
            geom_col=geom_col,
            sample_ratio=sample_ratio,
            sample_cap=sample_cap,
            seed=seed,
            geoparquet_workers=geoparquet_workers,
        )

    source = source_for_path(
        path,
        geom_col=geom_col,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_split_size=csv_split_size,
        csv_batch_rows=csv_batch_rows,
        src_crs=src_crs,
    )
    if isinstance(source, CSVSource):
        source = CSVSource(
            path,
            x_col=csv_x_col,
            y_col=csv_y_col,
            wkt_col=csv_wkt_col,
            split_size=csv_split_size,
            batch_rows=csv_batch_rows,
            src_crs=src_crs,
            geometry_only=True,
            geom_col=geom_col,
        )
    elif isinstance(source, ShapefileSource):
        source = ShapefileSource(path, geometry_only=True, geom_col=geom_col)
    elif isinstance(source, GDBSource):
        source = GDBSource(path, geometry_only=True, geom_col=geom_col)
    return _read_datasource_spatial_sample(
        source,
        geom_col=geom_col,
        sample_ratio=sample_ratio,
        sample_cap=sample_cap,
        seed=seed,
        source_workers=source_workers,
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


_WKB_GEOMETRY_TYPES = {
    1: "Point",
    2: "LineString",
    3: "Polygon",
    4: "MultiPoint",
    5: "MultiLineString",
    6: "MultiPolygon",
    7: "GeometryCollection",
    8: "CircularString",
    9: "CompoundCurve",
    10: "CurvePolygon",
    11: "MultiCurve",
    12: "MultiSurface",
    13: "Curve",
    14: "Surface",
    15: "PolyhedralSurface",
    16: "TIN",
    17: "Triangle",
}


def _decode_wkb_geometries(values: Any, *, geom_col: str, context: str) -> Any:
    try:
        return from_wkb(values)
    except Exception:
        logger.exception(
            "Failed decoding WKB geometries (%s, geom_col=%s, wkb_geometry_types=%s)",
            context,
            geom_col,
            _wkb_geometry_type_summary(values),
        )
        raise


def _split_context(source_path: Any, split: Any) -> str:
    parts = [f"path={getattr(split, 'path', source_path)}"]
    context_attrs = (
        "layer",
        "geometry_type",
        "row_groups",
        "offset",
        "length",
        "skip_features",
        "max_features",
    )
    for attr in context_attrs:
        if hasattr(split, attr):
            parts.append(f"{attr}={getattr(split, attr)!r}")
    return " ".join(parts)


def _wkb_geometry_type_summary(values: Any, limit: int = 64) -> str:
    counts: Dict[str, int] = {}
    inspected = 0
    unavailable = 0
    for value in values:
        if inspected >= limit:
            break
        inspected += 1
        geometry_type = _wkb_geometry_type(value)
        if geometry_type is None:
            unavailable += 1
            continue
        counts[geometry_type] = counts.get(geometry_type, 0) + 1

    pieces = [f"{name}={count}" for name, count in sorted(counts.items())]
    if unavailable:
        pieces.append(f"unavailable={unavailable}")
    return ", ".join(pieces) or "<none>"


def _wkb_geometry_type(value: Any) -> str | None:
    if value is None:
        return None
    try:
        data = bytes(value)
    except Exception:
        return None
    if len(data) < 5:
        return None

    byte_order = data[0]
    if byte_order == 0:
        raw_type = int.from_bytes(data[1:5], "big")
    elif byte_order == 1:
        raw_type = int.from_bytes(data[1:5], "little")
    else:
        return None

    has_z = bool(raw_type & 0x80000000)
    has_m = bool(raw_type & 0x40000000)
    base_type = raw_type & 0x0000FFFF
    if base_type >= 3000:
        has_z = True
        has_m = True
        base_type -= 3000
    elif base_type >= 2000:
        has_m = True
        base_type -= 2000
    elif base_type >= 1000:
        has_z = True
        base_type -= 1000

    name = _WKB_GEOMETRY_TYPES.get(base_type, f"type_code_{base_type}")
    suffix = ("Z" if has_z else "") + ("M" if has_m else "")
    return f"{name}{suffix}" if suffix else name


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
        parts: List[SpatialSample] = []
        for future in iter_with_progress(
            as_completed(futures),
            total=len(futures),
            logger=logger,
            label="MBR + sample",
        ):
            parts.append(future.result())
        return _combine_spatial_samples(parts)


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
        geometries = _decode_wkb_geometries(
            table[geom_col].to_numpy(zero_copy_only=False),
            geom_col=geom_col,
            context=_split_context(path, split),
        )

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


def _read_datasource_spatial_sample(
    source: DataSource,
    *,
    geom_col: str,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
    source_workers: Optional[int],
) -> SpatialSample:
    splits = source.create_splits()
    sample_caps = _split_sample_cap(sample_cap, len(splits))

    logger.info(
        "Reading spatial sample from %s in %d source partitions with %s thread workers",
        getattr(source, "path", "<source>"),
        len(splits),
        source_workers or "auto",
    )

    with ThreadPoolExecutor(max_workers=source_workers) as executor:
        futures = [
            executor.submit(
                _read_datasource_split_spatial_sample,
                source,
                split,
                geom_col,
                sample_ratio,
                sample_caps[index],
                seed + index,
            )
            for index, split in enumerate(splits)
        ]
        parts: List[SpatialSample] = []
        for future in iter_with_progress(
            as_completed(futures),
            total=len(futures),
            logger=logger,
            label="MBR + sample",
        ):
            parts.append(future.result())
        return _combine_spatial_samples(parts)


def _read_datasource_split_spatial_sample(
    source: DataSource,
    split: Any,
    geom_col: str,
    sample_ratio: float,
    sample_cap: Optional[int],
    seed: int,
) -> SpatialSample:
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
        geometries = _decode_wkb_geometries(
            table[geom_col].to_numpy(zero_copy_only=False),
            geom_col=geom_col,
            context=_split_context(getattr(source, "path", "<source>"), split),
        )

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
        parts: List[SpatialSample] = []
        for future in iter_with_progress(
            as_completed(futures),
            total=len(futures),
            logger=logger,
            label="MBR + sample",
        ):
            parts.append(future.result())
        return _combine_spatial_samples(parts)


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


# Backward-compatible re-exports for callers importing concrete sources from
# this module.
from starlet._internal.tiling.geojson_source import GeoJSONSource, GeoJSONSplit
from starlet._internal.tiling.geoparquet_source import GeoParquetSource, GeoParquetSplit
from starlet._internal.tiling.csv_source import CSVSource, CSVSplit
from starlet._internal.tiling.vector_source import GDBSource, ShapefileSource, VectorLayerSplit
