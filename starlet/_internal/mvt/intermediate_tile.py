"""Standalone intermediate vector tile helper.

This module intentionally does not participate in the current MVT generation
pipeline. It provides a small in-memory tile object that can collect Web
Mercator geometries, reservoir-sample them by feature count, merge with another
intermediate tile, and simplify the retained features into tile pixel
coordinates only when encoding MVT bytes.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import mapbox_vector_tile
import numpy as np
import pyarrow as pa
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
        self._small_geometry_area = 10.0

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

        if len(self._features) < self.feature_capacity:
            # Have not yet filled the reservoir, so just append the new feature.
            slot = len(self._features)
        else:
            # Reservoir sampling: randomly replace an existing feature with the new one.
            slot = self.rng.randrange(self._features_seen) 
            if slot >= self.feature_capacity:
                # The new feature is not selected for retention, so skip it.
                return False

        coordinate_count = shapely.count_coordinates(geometry)
        if coordinate_count == 0:
            return False

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
            # Remove the feature in the slot to replace
            self.coordinate_count -= self._features[slot].coordinate_count
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
        if geometry.geom_type not in {"Point", "MultiPoint"}:
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
        assert (self.z, self.x, self.y) == (other.z, other.x, other.y), "Cannot merge intermediate tiles with different tile IDs"
        self._features_seen += other._features_seen
        for feature in other._features:
            if len(self._features) < self.feature_capacity:
                self._features.append(feature)
                self._coordinate_count += feature.coordinate_count
            else:
                slot = self.rng.randrange(self.feature_capacity)
                self._coordinate_count += feature.coordinate_count - self._features[slot].coordinate_count
                self._features[slot] = feature

    def write_features(self, path) -> None:
        """Write retained features to an Arrow IPC file."""
        table = pa.table(
            {
                "geometry": pa.array(
                    [feature.geometry.wkb for feature in self._features],
                    type=pa.binary(),
                ),
                "properties": pa.array(
                    [
                        json.dumps(feature.properties, separators=(",", ":"))
                        for feature in self._features
                    ],
                    type=pa.string(),
                ),
            }
        )
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    def load_features(self, path) -> None:
        """Load retained features from an Arrow IPC file without resampling."""
        with pa.memory_map(str(path), "r") as source:
            table = pa.ipc.open_file(source).read_all()

        geometries = table["geometry"].to_pylist()
        properties = table["properties"].to_pylist()
        for geometry_bytes, property_json in zip(geometries, properties):
            geometry = shapely.from_wkb(geometry_bytes)
            coordinate_count = shapely.count_coordinates(geometry)
            if coordinate_count == 0:
                continue
            self._features.append(
                _TileFeature(
                    geometry=geometry,
                    properties=json.loads(property_json),
                    coordinate_count=coordinate_count,
                )
            )
            self._coordinate_count += coordinate_count
            self._features_seen += 1

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
