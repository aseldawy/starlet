"""Standalone intermediate vector tile helper.

This module intentionally does not participate in the current MVT generation
pipeline. It provides a small in-memory tile object that can collect Web
Mercator geometries, reservoir-sample them by feature count, merge with another
intermediate tile, and simplify the retained features into tile pixel
coordinates only when encoding MVT bytes.
"""
from __future__ import annotations

import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mapbox_vector_tile
import pyarrow as pa
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Point

from starlet._internal.mvt.pyramid_partitioner import PyramidPartitioner

from .helpers import EXTENT, explode_geom, mercator_tile_bounds


DEFAULT_FEATURE_CAPACITY = 2_000
_FEATURES_SEEN_HEADER = struct.Struct("<Q")
_FEATURES_SEEN_PADDING = 0


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
        self._small_geometry_area = 30.0

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
            self._coordinate_count -= self._features[slot].coordinate_count
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
        """Merge another tile with the same z/x/y using an exact count split."""
        # Verify that both tiles have the same location
        assert (self.z, self.x, self.y) == (other.z, other.x, other.y), "Cannot merge intermediate tiles with different tile IDs"
        total_seen = self._features_seen + other._features_seen

        if total_seen <= self.feature_capacity:
            self._features.extend(other._features)
            self._coordinate_count += other._coordinate_count
            self._features_seen = total_seen
            return

        other_count = 0
        remaining_seen = total_seen
        remaining_other = other._features_seen
        remaining_draws = self.feature_capacity
        while remaining_draws > 0 and remaining_seen > 0 and remaining_other > 0:
            if self.rng.randrange(remaining_seen) < remaining_other:
                other_count += 1
                remaining_other -= 1
            remaining_seen -= 1
            remaining_draws -= 1

        self_count = self.feature_capacity - other_count
        self_count = min(self_count, len(self._features))
        other_count = min(other_count, len(other._features))
        if self_count + other_count < self.feature_capacity:
            deficit = self.feature_capacity - (self_count + other_count)
            if len(self._features) - self_count >= len(other._features) - other_count:
                self_count += min(deficit, len(self._features) - self_count)
            else:
                other_count += min(deficit, len(other._features) - other_count)

        def _sample_without_replacement(items: list[_TileFeature], count: int) -> list[_TileFeature]:
            if count <= 0:
                return []
            if count >= len(items):
                return list(items)
            pool = list(items)
            chosen: list[_TileFeature] = []
            for _ in range(count):
                index = self.rng.randrange(len(pool))
                chosen.append(pool.pop(index))
            return chosen

        self_sample = _sample_without_replacement(self._features, self_count)
        other_sample = _sample_without_replacement(other._features, other_count)

        self._features = self_sample + other_sample
        self._coordinate_count = sum(feature.coordinate_count for feature in self._features)
        self._features_seen = total_seen

    def write_features(self, path) -> None:
        """Write retained features and features-seen count to disk."""
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
        payload_sink = pa.BufferOutputStream()
        with pa.ipc.new_file(payload_sink, table.schema) as writer:
            writer.write_table(table)
        payload = payload_sink.getvalue().to_pybytes()
        with pa.OSFile(str(path), "wb") as sink:
            sink.write(_FEATURES_SEEN_HEADER.pack(self._features_seen))
            sink.write(b"\x00" * _FEATURES_SEEN_PADDING)
            sink.write(payload)

    def load_features(self, path) -> None:
        """Load retained features from disk without resampling."""
        data = Path(path).read_bytes()
        self._features_seen = _FEATURES_SEEN_HEADER.unpack(data[:_FEATURES_SEEN_HEADER.size])[0]
        payload_offset = _FEATURES_SEEN_HEADER.size + _FEATURES_SEEN_PADDING
        table = pa.ipc.open_file(pa.BufferReader(data[payload_offset:])).read_all()

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
