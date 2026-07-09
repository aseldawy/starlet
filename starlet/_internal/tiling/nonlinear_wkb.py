from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from shapely import to_wkb
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


_SUPPORTED_WKB_TYPES = {1, 2, 3, 4, 5, 6, 7, 15, 16, 17}
_NONLINEAR_WKB_TYPES = {8, 9, 10, 11, 12, 13, 14}
_XY = Tuple[float, float]


@dataclass(frozen=True)
class _Header:
    endian: str
    raw_type: int
    base_type: int
    has_z: bool
    has_m: bool
    offset: int


def linearize_wkb(value: Any, *, segments_per_quarter: int = 8) -> bytes | None:
    """Return Shapely-supported WKB, approximating nonlinear WKB when needed."""
    if value is None:
        return None
    data = bytes(value)
    if _wkb_base_type(data) in _SUPPORTED_WKB_TYPES:
        return data
    geom, offset = _read_geometry(data, 0, max(1, segments_per_quarter))
    if offset > len(data):
        raise ValueError("WKB parser read past end of geometry")
    return to_wkb(geom, hex=False)


def is_nonlinear_wkb(value: Any) -> bool:
    try:
        base_type = _wkb_base_type(bytes(value))
    except Exception:
        return False
    return base_type in _NONLINEAR_WKB_TYPES


def _wkb_base_type(data: bytes) -> int:
    header = _read_header(data, 0)
    return header.base_type


def _read_geometry(data: bytes, offset: int, segments_per_quarter: int) -> Tuple[Any, int]:
    header = _read_header(data, offset)
    offset = header.offset

    if header.base_type == 1:
        point, offset = _read_point(data, offset, header)
        return Point(point), offset
    if header.base_type == 2:
        points, offset = _read_point_list(data, offset, header)
        return _line_string(points), offset
    if header.base_type == 3:
        rings, offset = _read_polygon_rings(data, offset, header)
        return _polygon(rings), offset
    if header.base_type == 4:
        geoms, offset = _read_geometry_collection(data, offset, header, segments_per_quarter)
        return MultiPoint([geom for geom in geoms if not geom.is_empty]), offset
    if header.base_type == 5:
        geoms, offset = _read_geometry_collection(data, offset, header, segments_per_quarter)
        return MultiLineString([list(geom.coords) for geom in geoms if not geom.is_empty]), offset
    if header.base_type == 6:
        geoms, offset = _read_geometry_collection(data, offset, header, segments_per_quarter)
        return MultiPolygon([geom for geom in geoms if not geom.is_empty]), offset
    if header.base_type == 7:
        geoms, offset = _read_geometry_collection(data, offset, header, segments_per_quarter)
        return GeometryCollection(geoms), offset
    if header.base_type == 8:
        points, offset = _read_point_list(data, offset, header)
        return _line_string(_linearize_circular_string(points, segments_per_quarter)), offset
    if header.base_type == 9:
        points, offset = _read_compound_curve(data, offset, header, segments_per_quarter)
        return _line_string(points), offset
    if header.base_type == 10:
        rings, offset = _read_curve_polygon(data, offset, header, segments_per_quarter)
        return _polygon(rings), offset
    if header.base_type == 11:
        curves, offset = _read_multi_curve(data, offset, header, segments_per_quarter)
        return MultiLineString(curves), offset
    if header.base_type == 12:
        polygons, offset = _read_multi_surface(data, offset, header, segments_per_quarter)
        return MultiPolygon(polygons), offset
    if header.base_type == 17:
        points, offset = _read_point_list(data, offset, header)
        return _polygon([_closed_ring(points)]), offset

    raise ValueError(f"Unsupported WKB geometry type code {header.base_type}")


def _read_header(data: bytes, offset: int) -> _Header:
    if offset + 5 > len(data):
        raise ValueError("Truncated WKB geometry header")
    byte_order = data[offset]
    if byte_order == 0:
        endian = ">"
    elif byte_order == 1:
        endian = "<"
    else:
        raise ValueError(f"Invalid WKB byte order {byte_order}")

    raw_type = struct.unpack_from(f"{endian}I", data, offset + 1)[0]
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
    return _Header(endian, raw_type, base_type, has_z, has_m, offset + 5)


def _read_uint32(data: bytes, offset: int, header: _Header) -> Tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("Truncated WKB count")
    return struct.unpack_from(f"{header.endian}I", data, offset)[0], offset + 4


def _read_point(data: bytes, offset: int, header: _Header) -> Tuple[_XY, int]:
    coord_count = 2 + int(header.has_z) + int(header.has_m)
    byte_count = coord_count * 8
    if offset + byte_count > len(data):
        raise ValueError("Truncated WKB coordinate")
    values = struct.unpack_from(f"{header.endian}{coord_count}d", data, offset)
    return (float(values[0]), float(values[1])), offset + byte_count


def _read_point_list(data: bytes, offset: int, header: _Header) -> Tuple[List[_XY], int]:
    count, offset = _read_uint32(data, offset, header)
    points = []
    for _ in range(count):
        point, offset = _read_point(data, offset, header)
        points.append(point)
    return points, offset


def _read_polygon_rings(data: bytes, offset: int, header: _Header) -> Tuple[List[List[_XY]], int]:
    count, offset = _read_uint32(data, offset, header)
    rings = []
    for _ in range(count):
        ring, offset = _read_point_list(data, offset, header)
        rings.append(_closed_ring(ring))
    return rings, offset


def _read_geometry_collection(
    data: bytes,
    offset: int,
    header: _Header,
    segments_per_quarter: int,
) -> Tuple[List[Any], int]:
    count, offset = _read_uint32(data, offset, header)
    geoms = []
    for _ in range(count):
        geom, offset = _read_geometry(data, offset, segments_per_quarter)
        geoms.append(geom)
    return geoms, offset


def _read_compound_curve(
    data: bytes,
    offset: int,
    header: _Header,
    segments_per_quarter: int,
) -> Tuple[List[_XY], int]:
    count, offset = _read_uint32(data, offset, header)
    points: List[_XY] = []
    for _ in range(count):
        geom, offset = _read_geometry(data, offset, segments_per_quarter)
        coords = list(geom.coords)
        points = _append_coords(points, coords)
    return points, offset


def _read_curve_polygon(
    data: bytes,
    offset: int,
    header: _Header,
    segments_per_quarter: int,
) -> Tuple[List[List[_XY]], int]:
    count, offset = _read_uint32(data, offset, header)
    rings = []
    for _ in range(count):
        geom, offset = _read_geometry(data, offset, segments_per_quarter)
        rings.append(_closed_ring(list(geom.coords)))
    return rings, offset


def _read_multi_curve(
    data: bytes,
    offset: int,
    header: _Header,
    segments_per_quarter: int,
) -> Tuple[List[List[_XY]], int]:
    count, offset = _read_uint32(data, offset, header)
    curves = []
    for _ in range(count):
        geom, offset = _read_geometry(data, offset, segments_per_quarter)
        coords = list(geom.coords)
        if len(coords) >= 2:
            curves.append(coords)
    return curves, offset


def _read_multi_surface(
    data: bytes,
    offset: int,
    header: _Header,
    segments_per_quarter: int,
) -> Tuple[List[Polygon], int]:
    count, offset = _read_uint32(data, offset, header)
    polygons = []
    for _ in range(count):
        geom, offset = _read_geometry(data, offset, segments_per_quarter)
        if isinstance(geom, Polygon) and not geom.is_empty:
            polygons.append(geom)
    return polygons, offset


def _linearize_circular_string(points: Sequence[_XY], segments_per_quarter: int) -> List[_XY]:
    if len(points) < 3:
        return list(points)
    output = [points[0]]
    for index in range(0, len(points) - 2, 2):
        arc = _linearize_arc(
            points[index],
            points[index + 1],
            points[index + 2],
            segments_per_quarter,
        )
        output = _append_coords(output, arc[1:])
    if len(points) % 2 == 0:
        output = _append_coords(output, [points[-1]])
    return output


def _linearize_arc(p0: _XY, p1: _XY, p2: _XY, segments_per_quarter: int) -> List[_XY]:
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    determinant = 2.0 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    if abs(determinant) < 1e-12:
        return [p0, p2]

    ux = (
        (x0 * x0 + y0 * y0) * (y1 - y2)
        + (x1 * x1 + y1 * y1) * (y2 - y0)
        + (x2 * x2 + y2 * y2) * (y0 - y1)
    ) / determinant
    uy = (
        (x0 * x0 + y0 * y0) * (x2 - x1)
        + (x1 * x1 + y1 * y1) * (x0 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x0)
    ) / determinant

    radius = math.hypot(x0 - ux, y0 - uy)
    a0 = math.atan2(y0 - uy, x0 - ux)
    a1 = math.atan2(y1 - uy, x1 - ux)
    a2 = math.atan2(y2 - uy, x2 - ux)
    ccw_sweep = (a2 - a0) % (2.0 * math.pi)
    ccw_mid = (a1 - a0) % (2.0 * math.pi)
    sweep = ccw_sweep if ccw_mid <= ccw_sweep else -((a0 - a2) % (2.0 * math.pi))

    segment_count = max(2, math.ceil(abs(sweep) / (math.pi / 2.0) * segments_per_quarter))
    return [
        (ux + radius * math.cos(a0 + sweep * i / segment_count),
         uy + radius * math.sin(a0 + sweep * i / segment_count))
        for i in range(segment_count + 1)
    ]


def _append_coords(existing: List[_XY], addition: Sequence[_XY]) -> List[_XY]:
    if not addition:
        return existing
    if existing and existing[-1] == addition[0]:
        return existing + list(addition[1:])
    return existing + list(addition)


def _closed_ring(points: Sequence[_XY]) -> List[_XY]:
    ring = list(points)
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _line_string(points: Sequence[_XY]) -> LineString:
    coords = list(points)
    if len(coords) == 1:
        coords.append(coords[0])
    return LineString(coords)


def _polygon(rings: Sequence[Sequence[_XY]]) -> Polygon:
    if not rings:
        return Polygon()
    shell = _closed_ring(rings[0])
    holes = [_closed_ring(ring) for ring in rings[1:] if len(ring) >= 3]
    return Polygon(shell, holes)
