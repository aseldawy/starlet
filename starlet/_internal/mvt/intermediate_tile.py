"""Standalone intermediate vector tile helper.

This module intentionally does not participate in the current MVT generation
pipeline. It provides a small in-memory tile object that can collect Web
Mercator geometries, reservoir-sample them by feature count, merge with another
intermediate tile, and simplify the retained features into tile pixel
coordinates only when encoding MVT bytes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import mapbox_vector_tile
import numpy as np
from shapely import get_coordinates
import shapely
from shapely.affinity import affine_transform
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

from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner

from .helpers import EXTENT, explode_geom, mercator_tile_bounds


DEFAULT_FEATURE_CAPACITY = 2_000


@dataclass(frozen=True)
class _TileFeature:
    geometry: Any
    properties: dict[str, Any]
    coordinate_count: int


class IntermediateVectorTile:
    """Collect sampled Web Mercator geometries before final MVT encoding."""

    def __init__(
        self,
        z: int,
        x: int,
        y: int,
        *,
        feature_capacity: int = DEFAULT_FEATURE_CAPACITY,
        extent: int = EXTENT,
        buffer: int = 256,
        rng: random.Random | None = None,
    ) -> None:
        self.z = int(z)
        self.x = int(x)
        self.y = int(y)
        self.feature_capacity = max(1, int(feature_capacity))
        self.extent = int(extent)
        self.buffer = int(buffer)
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

        self._features: list[_TileFeature] = []
        self._coordinate_count = 0
        self._features_seen = 0
        _pixel_area = (width * height) / (self.extent * self.extent) if self.extent > 0 else 0.0
        self._small_geometry_area = _pixel_area * 10.0

    @property
    def coordinate_count(self) -> int:
        """Number of raw geometry coordinates currently retained by this tile."""
        return self._coordinate_count
    
    @property
    def tile_id(self) -> int:
        """Unique tile ID for this z/x/y."""
        return PyramidPartitioner.encode_tile_id(self.z, self.x, self.y)

    @property
    def feature_count(self) -> int:
        """Number of retained raw features."""
        return len(self._features)

    def add_feature(
        self,
        geometry: Any,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Reservoir-sample a Web Mercator geometry by feature count."""
        if geometry is None or geometry.is_empty:
            return False

        self._features_seen += 1

        coordinate_count = shapely.count_coordinates(geometry)
        if coordinate_count == 0:
            return False

        if len(self._features) < self.feature_capacity:
            slot = len(self._features)
        else:
            if self.rng.randrange(self._features_seen) >= self.feature_capacity:
                return False
            slot = self.rng.randrange(self.feature_capacity)

        clean_properties = {
            key: value
            for key, value in (properties or {}).items()
            if value is not None
        }

        feature = _TileFeature(
            geometry=geometry,
            properties=clean_properties,
            coordinate_count=coordinate_count,
        )
        if slot == len(self._features):
            self._features.append(feature)
        else:
            replaced = self._features[slot]
            self._coordinate_count -= replaced.coordinate_count
            self._features[slot] = feature
        self._coordinate_count += coordinate_count
        return True

    def simplify_geometry(self, geometry: Any) -> list[Any]:
        """Return simplified tile-pixel geometries ready for MVT encoding."""
        geometry = affine_transform(
            geometry,
            (
                self.affine_params[0],
                0.0,
                0.0,
                self.affine_params[3],
                self.affine_params[4],
                self.affine_params[5],
            ),
        )

        minx, miny, maxx, maxy = geometry.bounds
        if (maxx - minx) * (maxy - miny) <= self._small_geometry_area:
            centroid = geometry.centroid
            geometry = Point(centroid.x, centroid.y)

        # Simplify the geometry to reduce the number of coordinates. Use tolerance of one pixel.
        if shapely.count_coordinates(geometry) > 10:
            geometry = shapely.simplify(geometry, 1.0, preserve_topology=False)
            geometry = shapely.clip_by_rect(
                geometry,
                -self.buffer, -self.buffer, self.extent + self.buffer, self.extent + self.buffer,
            )

        out = []
        for part in explode_geom(geometry):
            if not part.is_empty:
                out.append(part)
        return out

    def merge(self, other: "IntermediateVectorTile") -> None:
        """Merge another tile with the same z/x/y without simplifying."""
        # Verify that both tiles have the same location
        if (self.z, self.x, self.y) != (other.z, other.x, other.y):
            raise ValueError("Cannot merge intermediate vector tiles with different tile IDs")
        self._features_seen += other._features_seen
        for feature in other._features:
            if len(self._features) < self.feature_capacity:
                self._features.append(feature)
                self._coordinate_count += feature.coordinate_count
            else:
                slot = self.rng.randrange(self.feature_capacity)
                replaced = self._features[slot]
                self._coordinate_count += feature.coordinate_count - replaced.coordinate_count
                self._features[slot] = feature

    def encode(self, layer_name: str = "layer0") -> bytes:
        """Encode the retained features as an MVT binary payload."""
        layer = {
            "name": layer_name,
            "features": self._mvt_features(),
            "extent": self.extent,
        }
        result =  mapbox_vector_tile.encode([layer])
        return result

    def _mvt_features(self) -> list[dict[str, Any]]:
        out = []
        for feature in self._features:
            for geometry in self.simplify_geometry(feature.geometry):
                out.append(
                    {
                        "geometry": geometry,
                        "properties": dict(feature.properties),
                    }
                )
        return out
