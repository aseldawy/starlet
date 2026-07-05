"""Standalone intermediate vector tile helper.

This module intentionally does not participate in the current MVT generation
pipeline. It provides a small in-memory tile object that can collect Web
Mercator geometries, simplify them into tile pixel coordinates, reservoir-sample
them under a coordinate budget, merge with another intermediate tile, and encode
the retained features as MVT bytes.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import mapbox_vector_tile
from shapely import get_coordinates
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
)

from .helpers import EXTENT, explode_geom, mercator_tile_bounds


DEFAULT_COORDINATE_CAPACITY = 10_000


@dataclass(frozen=True)
class _TileFeature:
    priority: float
    seq: int
    geometry: Any
    properties: dict[str, Any]
    coordinate_count: int


class IntermediateVectorTile:
    """Collect simplified tile-space geometries before final MVT encoding."""

    def __init__(
        self,
        z: int,
        x: int,
        y: int,
        *,
        coordinate_capacity: int = DEFAULT_COORDINATE_CAPACITY,
        extent: int = EXTENT,
        buffer: int = 256,
        simplify_tolerance: float | None = None,
        layer_name: str = "layer0",
        rng: random.Random | None = None,
    ) -> None:
        self.z = int(z)
        self.x = int(x)
        self.y = int(y)
        self.coordinate_capacity = max(1, int(coordinate_capacity))
        self.extent = int(extent)
        self.buffer = int(buffer)
        self.simplify_tolerance = (
            float(simplify_tolerance)
            if simplify_tolerance is not None
            else self.extent * 0.0005
        )
        self.layer_name = layer_name
        self.rng = rng or random.Random()

        minx, miny, maxx, maxy = mercator_tile_bounds(self.z, self.x, self.y)
        width = maxx - minx
        height = maxy - miny
        x_scale = self.extent / width if width != 0 else 0.0
        y_scale = self.extent / height if height != 0 else 0.0
        self.affine_params = (
            x_scale,
            0.0,
            0.0,
            y_scale,
            -minx * x_scale,
            -miny * y_scale,
        )
        self._min_pixel = -self.buffer
        self._max_pixel = self.extent + self.buffer

        self._heap: list[tuple[float, int, _TileFeature]] = []
        self._seq = 0
        self._coordinate_count = 0

    @property
    def coordinate_count(self) -> int:
        """Number of coordinates currently retained by this tile."""
        return self._coordinate_count

    @property
    def feature_count(self) -> int:
        """Number of retained geometry parts."""
        return len(self._heap)

    def add_feature(
        self,
        geometry: Any,
        properties: dict[str, Any] | None = None,
        *,
        priority: float | None = None,
    ) -> bool:
        """Transform, simplify, and reservoir-sample a Web Mercator geometry."""
        if geometry is None or geometry.is_empty:
            return False

        feature_priority = self.rng.random() if priority is None else float(priority)
        if self._can_skip_without_processing(feature_priority):
            return False

        tile_geometries = self.simplify_geometry(geometry)
        if not tile_geometries:
            return False

        clean_properties = {
            key: value
            for key, value in (properties or {}).items()
            if value is not None
        }

        retained = False
        for tile_geometry in tile_geometries:
            coordinate_count = self.count_coordinates(tile_geometry)
            if coordinate_count == 0:
                continue
            retained |= self._add_tile_feature(
                _TileFeature(
                    priority=feature_priority,
                    seq=self._next_seq(),
                    geometry=tile_geometry,
                    properties=clean_properties,
                    coordinate_count=coordinate_count,
                )
            )
        return retained

    # Compatibility alias for users following the referenced Scala naming.
    addFeature = add_feature

    def simplify_geometry(self, geometry: Any) -> list[Any]:
        """Return simplified tile-pixel geometries ready for MVT encoding."""
        if geometry is None or geometry.is_empty:
            return []

        try:
            simplified = self._simplify_geometry(geometry)
        except Exception:
            return []

        if simplified is None or simplified.is_empty:
            return []

        out = []
        for part in explode_geom(simplified):
            if not part.is_empty:
                out.append(part)
        return out

    # Compatibility alias for users following the referenced Scala naming.
    simplifyGeometry = simplify_geometry

    def merge(self, other: "IntermediateVectorTile") -> None:
        """Merge another tile with the same z/x/y without re-simplifying."""
        self._check_same_location(other)
        for feature in other._features_by_priority():
            self._add_tile_feature(
                _TileFeature(
                    priority=feature.priority,
                    seq=self._next_seq(),
                    geometry=feature.geometry,
                    properties=dict(feature.properties),
                    coordinate_count=feature.coordinate_count,
                )
            )

    def combined(self, other: "IntermediateVectorTile") -> "IntermediateVectorTile":
        """Return a new tile that contains the sampled union of two tiles."""
        self._check_same_location(other)
        out = IntermediateVectorTile(
            self.z,
            self.x,
            self.y,
            coordinate_capacity=self.coordinate_capacity,
            extent=self.extent,
            buffer=self.buffer,
            simplify_tolerance=self.simplify_tolerance,
            layer_name=self.layer_name,
            rng=self.rng,
        )
        out.merge(self)
        out.merge(other)
        return out

    def features(self) -> list[dict[str, Any]]:
        """Return retained MVT feature dictionaries."""
        return [
            {
                "geometry": mapping(feature.geometry),
                "properties": dict(feature.properties),
            }
            for feature in self._features_by_priority()
        ]

    def encode(self) -> bytes:
        """Encode the retained features as an MVT binary payload."""
        layer = {
            "name": self.layer_name,
            "features": self.features(),
            "extent": self.extent,
        }
        return mapbox_vector_tile.encode([layer])

    def _add_tile_feature(self, feature: _TileFeature) -> bool:
        heapq.heappush(self._heap, (feature.priority, feature.seq, feature))
        self._coordinate_count += feature.coordinate_count

        retained = True
        while self._coordinate_count > self.coordinate_capacity and self._heap:
            _, _, removed = heapq.heappop(self._heap)
            self._coordinate_count -= removed.coordinate_count
            if removed is feature:
                retained = False
        return retained

    def _can_skip_without_processing(self, priority: float) -> bool:
        return (
            self._coordinate_count >= self.coordinate_capacity
            and bool(self._heap)
            and priority <= self._heap[0][0]
        )

    def _features_by_priority(self) -> list[_TileFeature]:
        return [
            feature
            for _, _, feature in sorted(
                self._heap,
                key=lambda entry: (entry[0], entry[1]),
                reverse=True,
            )
        ]

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _check_same_location(self, other: "IntermediateVectorTile") -> None:
        if (self.z, self.x, self.y) != (other.z, other.x, other.y):
            raise ValueError("Cannot merge intermediate vector tiles with different tile IDs")

    def _simplify_geometry(self, geometry: Any) -> Any | None:
        geom_type = geometry.geom_type
        if geom_type == "Point":
            return self._simplify_point(geometry)
        if geom_type == "LineString":
            return self._simplify_line_string(list(geometry.coords))
        if geom_type == "LinearRing":
            return self._simplify_ring(list(geometry.coords))
        if geom_type == "Polygon":
            return self._simplify_polygon(geometry)
        if geom_type == "MultiPoint":
            return self._simplify_multi_point(geometry)
        if geom_type == "MultiLineString":
            return self._combine_geometries(
                self._simplify_geometry(part) for part in geometry.geoms
            )
        if geom_type in {"MultiPolygon", "GeometryCollection"}:
            return self._combine_geometries(
                self._simplify_geometry(part) for part in geometry.geoms
            )
        return None

    def _simplify_point(self, point: Any) -> Point | None:
        x, y = self._transform_xy(point.x, point.y)
        rx, ry = self._round_xy(x, y)
        return Point(rx, ry) if self._is_visible(rx, ry) else None

    def _simplify_multi_point(self, geometry: Any) -> Point | MultiPoint | None:
        points = []
        seen = set()
        for point in geometry.geoms:
            x, y = self._transform_xy(point.x, point.y)
            rounded = self._round_xy(x, y)
            if self._is_visible(*rounded) and rounded not in seen:
                seen.add(rounded)
                points.append(rounded)
        points.sort()
        if not points:
            return None
        if len(points) == 1:
            return Point(points[0])
        return MultiPoint(points)

    def _simplify_line_string(self, coords: list[tuple[float, ...]]) -> Any | None:
        if not coords:
            return None

        parts: list[Any] = []
        current: list[tuple[int, int]] = []

        def add_point(x: float, y: float) -> None:
            point = self._round_xy(x, y)
            if not current or current[-1] != point:
                current.append(point)

        def finish_current() -> None:
            if len(current) == 1:
                parts.append(Point(current[0]))
            elif len(current) > 1:
                parts.append(LineString(current))
            current.clear()

        x1, y1 = self._transform_coord(coords[0])
        start_visible = self._is_visible(x1, y1)
        if start_visible:
            add_point(x1, y1)

        for i, coord in enumerate(coords[1:], start=1):
            x2, y2 = self._transform_coord(coord)
            end_visible = self._is_visible(x2, y2)
            if i == 1 or self._round(x2) != self._round(x1) or self._round(y2) != self._round(y1):
                if start_visible and end_visible:
                    add_point(x2, y2)
                elif start_visible and not end_visible:
                    trimmed = self._trim_line_segment(x1, y1, x2, y2)
                    if trimmed is not None:
                        add_point(trimmed[2], trimmed[3])
                        finish_current()
                elif not start_visible and end_visible:
                    trimmed = self._trim_line_segment(x1, y1, x2, y2)
                    if trimmed is not None:
                        add_point(trimmed[0], trimmed[1])
                        add_point(trimmed[2], trimmed[3])
                else:
                    trimmed = self._trim_line_segment(x1, y1, x2, y2)
                    if trimmed is not None:
                        p1 = self._round_xy(trimmed[0], trimmed[1])
                        p2 = self._round_xy(trimmed[2], trimmed[3])
                        if p1 == p2:
                            parts.append(Point(p1))
                        else:
                            parts.append(LineString([p1, p2]))

                x1, y1 = x2, y2
                start_visible = end_visible

        finish_current()
        return self._combine_line_parts(parts)

    def _simplify_ring(self, coords: list[tuple[float, ...]]) -> Any | None:
        if not coords:
            return None

        points: list[tuple[int, int]] = []

        def add_point(x: float, y: float) -> None:
            point = self._round_xy(x, y)
            if not points or points[-1] != point:
                points.append(point)

        def trace_tile_edge(
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            cw_ordering: bool,
        ) -> None:
            current = (x1, y1)
            end = (x2, y2)
            guard = 0
            while current != end and guard < 8:
                current = (
                    self._next_point_cw(*current, *end)
                    if cw_ordering
                    else self._next_point_ccw(*current, *end)
                )
                add_point(*current)
                guard += 1

        x1, y1 = self._transform_coord(coords[0])
        start_visible = self._is_visible(x1, y1)
        sum_for_ordering = 0.0
        sum_for_ordering_first_part = 0.0
        if start_visible:
            add_point(x1, y1)

        for coord in coords[1:]:
            x2, y2 = self._transform_coord(coord)
            end_visible = self._is_visible(x2, y2)
            if start_visible and end_visible:
                add_point(x2, y2)
            elif start_visible and not end_visible:
                trimmed = self._trim_line_segment(x1, y1, x2, y2)
                if trimmed is not None:
                    add_point(trimmed[2], trimmed[3])
                    sum_for_ordering = (x2 - self._round(trimmed[2])) * (
                        y2 + self._round(trimmed[3])
                    )
            elif not start_visible and end_visible:
                trimmed = self._trim_line_segment(x1, y1, x2, y2)
                if trimmed is not None:
                    sum_for_ordering += (self._round(trimmed[0]) - x1) * (
                        self._round(trimmed[1]) + y1
                    )
                    if points:
                        sum_for_ordering += (
                            points[-1][0] - self._round(trimmed[0])
                        ) * (points[-1][1] + self._round(trimmed[1]))
                        trace_tile_edge(
                            *points[-1],
                            *self._round_xy(trimmed[0], trimmed[1]),
                            sum_for_ordering > 0,
                        )
                    else:
                        add_point(trimmed[0], trimmed[1])
                        sum_for_ordering_first_part = sum_for_ordering
                    add_point(x2, y2)
            else:
                trimmed = self._trim_line_segment(x1, y1, x2, y2)
                if trimmed is None:
                    sum_for_ordering += (x2 - x1) * (y2 + y1)
                else:
                    if points:
                        sum_for_ordering += (self._round(trimmed[0]) - x1) * (
                            self._round(trimmed[1]) + y1
                        )
                        trace_tile_edge(
                            *points[-1],
                            *self._round_xy(trimmed[0], trimmed[1]),
                            sum_for_ordering > 0,
                        )
                    else:
                        sum_for_ordering += (self._round(trimmed[0]) - x1) * (
                            self._round(trimmed[1]) + y1
                        )
                        sum_for_ordering_first_part = sum_for_ordering
                        add_point(trimmed[0], trimmed[1])
                    add_point(trimmed[2], trimmed[3])
                    sum_for_ordering = (x2 - self._round(trimmed[2])) * (
                        y2 + self._round(trimmed[3])
                    )
            x1, y1 = x2, y2
            start_visible = end_visible

        if not start_visible and len(points) == 1:
            points.clear()
            sum_for_ordering += sum_for_ordering_first_part
        elif not start_visible and len(points) > 1:
            sum_for_ordering += (points[-1][0] - points[0][0]) * (
                points[-1][1] + points[0][1]
            )
            trace_tile_edge(
                *points[-1],
                *points[0],
                (sum_for_ordering + sum_for_ordering_first_part) > 0,
            )

        if not points:
            return self._full_tile_ring_if_contains_origin(coords)
        if len(points) == 1:
            return Point(points[0])
        if points[0] != points[-1]:
            points.append(points[0])
        return LineString(points)

    def _simplify_polygon(self, polygon: Any) -> Any | None:
        exterior = self._simplify_ring(list(polygon.exterior.coords))
        if exterior is None:
            return None
        if exterior.geom_type == "Point":
            return exterior
        if exterior.geom_type == "LineString" and len(exterior.coords) <= 3:
            return exterior

        shell = list(exterior.coords)
        if self._is_ccw(shell):
            shell = list(reversed(shell))

        holes = []
        for interior in polygon.interiors:
            hole = self._simplify_ring(list(interior.coords))
            if hole is not None and hole.geom_type == "LineString" and len(hole.coords) > 3:
                hole_coords = list(hole.coords)
                if not self._is_ccw(hole_coords):
                    hole_coords = list(reversed(hole_coords))
                holes.append(hole_coords)
        return Polygon(shell, holes)

    def _combine_line_parts(self, parts: list[Any]) -> Any | None:
        if not parts:
            return None
        points = [part for part in parts if part.geom_type == "Point"]
        lines = [part for part in parts if part.geom_type == "LineString"]
        if not lines and len(points) == 1:
            return points[0]
        if not lines:
            return MultiPoint([point.coords[0] for point in points])
        if len(lines) == 1:
            return lines[0]
        return MultiLineString([list(line.coords) for line in lines])

    def _combine_geometries(self, geometries: Iterable[Any | None]) -> Any | None:
        parts = [part for part in geometries if part is not None and not part.is_empty]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        if all(part.geom_type == "Point" for part in parts):
            return MultiPoint([part.coords[0] for part in parts])
        if all(part.geom_type in {"LineString", "MultiLineString"} for part in parts):
            lines = []
            for part in parts:
                if part.geom_type == "LineString":
                    lines.append(list(part.coords))
                else:
                    lines.extend(list(line.coords) for line in part.geoms)
            return MultiLineString(lines)
        if all(part.geom_type in {"Polygon", "MultiPolygon"} for part in parts):
            polygons = []
            for part in parts:
                if part.geom_type == "Polygon":
                    polygons.append(part)
                else:
                    polygons.extend(part.geoms)
            return MultiPolygon(polygons)
        return GeometryCollection(parts)

    def _full_tile_ring_if_contains_origin(
        self,
        coords: list[tuple[float, ...]],
    ) -> LineString | None:
        x1, y1 = self._transform_coord(coords[0])
        num_intersections = 0
        for coord in coords[1:]:
            x2, y2 = self._transform_coord(coord)
            if (y1 < 0 < y2) or (y2 < 0 < y1):
                x_intersection = x1 - y1 * (x2 - x1) / (y2 - y1)
                if x_intersection < 0:
                    num_intersections += 1
            x1, y1 = x2, y2
        if num_intersections % 2 == 0:
            return None
        lo = self._min_pixel
        hi = self._max_pixel
        return LineString([(lo, lo), (lo, hi), (hi, hi), (hi, lo), (lo, lo)])

    def _trim_line_segment(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[float, float, float, float] | None:
        if self._is_visible(x1, y1) and self._is_visible(x2, y2):
            return (x1, y1, x2, y2)

        reversed_order = False

        def reverse() -> None:
            nonlocal x1, y1, x2, y2, reversed_order
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            reversed_order = not reversed_order

        lo = self._min_pixel
        hi = self._max_pixel
        if x1 > x2:
            reverse()
        if x2 < lo or x1 > hi:
            return None
        if x1 < lo:
            y1 = ((lo - x1) * y2 + (x2 - lo) * y1) / (x2 - x1)
            x1 = lo
        if x2 > hi:
            y2 = ((hi - x1) * y2 + (x2 - hi) * y1) / (x2 - x1)
            x2 = hi

        if y1 > y2:
            reverse()
        if y2 < lo or y1 > hi:
            return None
        if y1 < lo:
            x1 = ((lo - y1) * x2 + (y2 - lo) * x1) / (y2 - y1)
            y1 = lo
        if y2 > hi:
            x2 = ((hi - y1) * x2 + (y2 - hi) * x1) / (y2 - y1)
            y2 = hi

        if reversed_order:
            reverse()
        return (x1, y1, x2, y2)

    def _next_point_cw(self, x: int, y: int, x_end: int, y_end: int) -> tuple[int, int]:
        lo = self._min_pixel
        hi = self._max_pixel
        if x == x_end and y == y_end:
            return (x_end, y_end)
        if x == lo:
            if y == hi:
                return (x_end, y_end) if y == y_end else (hi, y)
            return (x_end, y_end) if x == x_end and y_end >= y else (x, hi)
        if x == hi:
            if y == lo:
                return (x_end, y_end) if y == y_end else (lo, y)
            return (x_end, y_end) if x == x_end and y_end <= y else (x, lo)
        if y == lo:
            if x == lo:
                return (x_end, y_end) if x == x_end else (x, hi)
            return (x_end, y_end) if y == y_end and x_end <= x else (lo, y)
        if x == hi:
            return (x_end, y_end) if x == x_end else (x, lo)
        return (x_end, y_end) if y == y_end and x_end >= x else (hi, y)

    def _next_point_ccw(self, x: int, y: int, x_end: int, y_end: int) -> tuple[int, int]:
        lo = self._min_pixel
        hi = self._max_pixel
        if x == x_end and y == y_end:
            return (x_end, y_end)
        if x == lo:
            if y == lo:
                return (x_end, y_end) if y == y_end else (hi, y)
            return (x_end, y_end) if x == x_end and y_end <= y else (x, lo)
        if x == hi:
            if y == hi:
                return (x_end, y_end) if y == y_end else (lo, y)
            return (x_end, y_end) if x == x_end and y_end >= y else (x, hi)
        if y == lo:
            if x == hi:
                return (x_end, y_end) if x == x_end else (x, hi)
            return (x_end, y_end) if y == y_end and x_end >= x else (hi, y)
        if x == lo:
            return (x_end, y_end) if x == x_end else (x, lo)
        return (x_end, y_end) if y == y_end and x_end <= x else (lo, y)

    def _transform_coord(self, coord: tuple[float, ...]) -> tuple[float, float]:
        return self._transform_xy(coord[0], coord[1])

    def _transform_xy(self, x: float, y: float) -> tuple[float, float]:
        a, b, d, e, xoff, yoff = self.affine_params
        return (a * x + b * y + xoff, d * x + e * y + yoff)

    def _is_visible(self, x: float, y: float) -> bool:
        return self._min_pixel <= x <= self._max_pixel and self._min_pixel <= y <= self._max_pixel

    @staticmethod
    def _round(value: float) -> int:
        return math.floor(value + 0.5)

    def _round_xy(self, x: float, y: float) -> tuple[int, int]:
        return (self._round(x), self._round(y))

    @staticmethod
    def _is_ccw(coords: list[tuple[float, ...]]) -> bool:
        area = 0.0
        for (x1, y1, *_), (x2, y2, *_) in zip(coords, coords[1:]):
            area += (x2 - x1) * (y2 + y1)
        return area < 0

    @staticmethod
    def count_coordinates(geometry: Any) -> int:
        """Count coordinates in a Shapely geometry."""
        try:
            return len(get_coordinates(geometry))
        except Exception:
            return sum(
                IntermediateVectorTile.count_coordinates(part)
                for part in _iter_parts(geometry)
            )


def _iter_parts(geometry: Any) -> Iterable[Any]:
    if hasattr(geometry, "geoms"):
        return geometry.geoms
    return ()
