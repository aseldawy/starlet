from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import pyarrow as pa
import shapely
from pyproj import CRS, Transformer
from shapely import from_wkb, to_wkb

logger = logging.getLogger(__name__)

WGS84_CRS = "EPSG:4326"
WEB_MERCATOR_CRS = "EPSG:3857"


def crs_equal(left: Any, right: Any = WGS84_CRS) -> bool:
    try:
        return CRS.from_user_input(left) == CRS.from_user_input(right)
    except Exception:
        return False


def reproject_geometries_to_wgs84(geometries: Any, src_crs: Any) -> tuple[Any, bool]:
    return reproject_geometries(geometries, src_crs, WGS84_CRS)


def reproject_geometries(geometries: Any, src_crs: Any, dst_crs: Any) -> tuple[Any, bool]:
    if not src_crs or not dst_crs or crs_equal(src_crs, dst_crs):
        return geometries, False
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely.transform(geometries, transformer.transform, interleaved=False), True


def reproject_wkb_to_wgs84(values: Iterable[Any], src_crs: Any) -> tuple[list[Any], bool]:
    return reproject_wkb(values, src_crs, WGS84_CRS)


def reproject_wkb(values: Iterable[Any], src_crs: Any, dst_crs: Any) -> tuple[list[Any], bool]:
    values = list(values)
    if not src_crs or not dst_crs or crs_equal(src_crs, dst_crs):
        return values, False

    non_null_indices = [index for index, value in enumerate(values) if value is not None]
    if not non_null_indices:
        return values, False

    geometries = from_wkb([values[index] for index in non_null_indices])
    transformed, _ = reproject_geometries(geometries, src_crs, dst_crs)
    transformed_wkb = to_wkb(transformed, hex=False).tolist()
    for index, value in zip(non_null_indices, transformed_wkb):
        values[index] = value
    return values, True


def reproject_table_to_wgs84(table: pa.Table, geom_col: str, src_crs: Any) -> tuple[pa.Table, bool]:
    return reproject_table(table, geom_col, src_crs, WGS84_CRS)


def reproject_table(
    table: pa.Table,
    geom_col: str,
    src_crs: Any,
    dst_crs: Any,
) -> tuple[pa.Table, bool]:
    if geom_col not in table.column_names:
        return table, False
    geometries, transformed = reproject_wkb(
        table[geom_col].to_pylist(),
        src_crs,
        dst_crs,
    )
    if not transformed:
        return table, False
    geom_index = table.column_names.index(geom_col)
    return table.set_column(
        geom_index,
        geom_col,
        pa.array(geometries, type=pa.binary()),
    ), True


def geoparquet_crs(schema: pa.Schema, geom_col: str = "geometry") -> Any:
    raw_geo = (schema.metadata or {}).get(b"geo")
    if not raw_geo:
        return None
    try:
        geo = json.loads(raw_geo.decode("utf-8"))
    except Exception:
        logger.debug("Could not parse GeoParquet CRS metadata", exc_info=True)
        return None
    columns = geo.get("columns") or {}
    geometry_meta = columns.get(geom_col) or columns.get("geometry") or {}
    return geometry_meta.get("crs")
