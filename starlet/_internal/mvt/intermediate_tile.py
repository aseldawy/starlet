"""Intermediate vector tile used by the map/reduce MVT pipeline.

Provides a small tile object that collects Web Mercator geometries,
retains a bounded uniform sample of them, merges with partial tiles from
other mappers, and simplifies the retained features into tile pixel
coordinates only when encoding MVT bytes.

Sampling is **priority-based top-k** rather than an independent random
reservoir: every feature carries a priority (by default ``crc32`` of its
WKB — geometry-intrinsic and deterministic; the batch pipeline passes the
crc32 of the *source* WKB bytes computed before decode), and each tile
keeps the ``feature_capacity`` features with the highest priority. Because
the same geometry has the same priority in every tile (and zoom level) it
touches, adjacent tiles make consistent keep/drop decisions — no seam
popping — and merging partial tiles from different mappers is a
deterministic top-k union instead of a statistical resample. Hash
priorities are uniformly distributed, so the retained set is still a
uniform sample of everything seen.
"""
from __future__ import annotations

import heapq
import json
import random
import struct
import zlib
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

# Features whose transformed bbox fits inside this many tile pixels in BOTH
# dimensions collapse to their centroid Point (sub-pixel clutter). Judged
# per-dimension, not by bbox area: a long straight line has bbox area 0 but
# must keep its type.
_SMALL_GEOMETRY_EXTENT_PX = 5.5


def feature_priority(wkb_bytes: bytes) -> int:
    """Deterministic, geometry-intrinsic sampling priority for a feature."""
    return zlib.crc32(wkb_bytes)


@dataclass(frozen=True)
class _TileFeature:
    geometry: Any
    properties: dict[str, Any]


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
        # Accepted for backward compatibility; sampling is deterministic
        # (priority top-k) and no longer draws from an RNG.
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

        # Min-heap of (priority, seq, _TileFeature); holds the top-k by
        # priority. seq is an insertion tiebreaker so heap comparisons never
        # fall through to comparing feature objects.
        self._heap: list[tuple[int, int, _TileFeature]] = []
        self._seq = 0
        self._features_seen = 0

    @property
    def tile_id(self) -> int:
        """Unique tile ID for this z/x/y."""
        return PyramidPartitioner.encode_tile_id(self.z, self.x, self.y)

    @property
    def feature_count(self) -> int:
        """Number of retained raw features."""
        return len(self._heap)

    @property
    def _features(self) -> list[_TileFeature]:
        """Retained features (arbitrary order); kept for introspection."""
        return [entry[2] for entry in self._heap]

    def add_feature(
        self,
        geometry: Any,
        properties: dict[str, Any] | None = None,
        priority: int | None = None,
    ) -> bool:
        """Offer a Web Mercator geometry; keep it if it ranks in the top-k.

        ``priority`` should be :func:`feature_priority` of the feature's
        canonical (source) WKB bytes so that every tile the feature touches
        ranks it identically. When omitted it is derived from the current
        geometry's WKB.
        """
        if geometry is None or geometry.is_empty:
            return False

        self._features_seen += 1

        if priority is None:
            priority = feature_priority(shapely.to_wkb(geometry))
        priority = int(priority)

        if len(self._heap) >= self.feature_capacity and priority <= self._heap[0][0]:
            return False

        clean_properties = {
            key: value
            for key, value in (properties or {}).items()
            if value is not None
        }
        entry = (priority, self._seq, _TileFeature(geometry, clean_properties))
        self._seq += 1

        if len(self._heap) < self.feature_capacity:
            heapq.heappush(self._heap, entry)
        else:
            heapq.heapreplace(self._heap, entry)
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
        threshold = _SMALL_GEOMETRY_EXTENT_PX * (self.extent / EXTENT)
        if (maxx - minx) <= threshold and (maxy - miny) <= threshold:
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
        """Merge another partial tile for the same z/x/y: top-k union.

        Every entry keeps the priority it was offered with, so the merged
        result is exactly the top ``feature_capacity`` features by priority
        across both partials — deterministic and independent of merge order.
        """
        if (self.z, self.x, self.y) != (other.z, other.x, other.y):
            raise ValueError("Cannot merge intermediate tiles with different tile IDs")

        combined = [(p, s) for (p, _, s) in self._heap]
        combined.extend((p, s) for (p, _, s) in other._heap)
        # Sort by priority (stable: self's entries win ties deterministically).
        combined.sort(key=lambda item: item[0], reverse=True)
        kept = combined[: self.feature_capacity]

        self._heap = []
        self._seq = 0
        for priority, feature in kept:
            heapq.heappush(self._heap, (priority, self._seq, feature))
            self._seq += 1
        self._features_seen += other._features_seen

    def write_features(self, path) -> None:
        """Write retained features, priorities, and seen count to disk."""
        entries = list(self._heap)
        table = pa.table(
            {
                "geometry": pa.array(
                    [entry[2].geometry.wkb for entry in entries],
                    type=pa.binary(),
                ),
                "properties": pa.array(
                    [
                        json.dumps(entry[2].properties, separators=(",", ":"))
                        for entry in entries
                    ],
                    type=pa.string(),
                ),
                "priority": pa.array(
                    [entry[0] for entry in entries],
                    type=pa.uint64(),
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
        """Load retained features (with their priorities) from disk."""
        data = Path(path).read_bytes()
        self._features_seen = _FEATURES_SEEN_HEADER.unpack(data[:_FEATURES_SEEN_HEADER.size])[0]
        payload_offset = _FEATURES_SEEN_HEADER.size + _FEATURES_SEEN_PADDING
        table = pa.ipc.open_file(pa.BufferReader(data[payload_offset:])).read_all()

        geometries = table["geometry"].to_pylist()
        properties = table["properties"].to_pylist()
        if "priority" in table.column_names:
            priorities = table["priority"].to_pylist()
        else:
            # Files written before the priority column: recompute from WKB.
            priorities = [feature_priority(geometry_bytes) for geometry_bytes in geometries]

        for geometry_bytes, property_json, priority in zip(geometries, properties, priorities):
            geometry = shapely.from_wkb(geometry_bytes)
            entry = (int(priority), self._seq, _TileFeature(geometry, json.loads(property_json)))
            self._seq += 1
            heapq.heappush(self._heap, entry)

    def encode(self, layer_name: str = "layer0") -> bytes:
        """Encode the retained features as an MVT binary payload."""
        layer = {
            "name": layer_name,
            "features": self._mvt_features(),
            "extent": self.extent,
        }
        result = mapbox_vector_tile.encode(
            [layer],
            default_options={"extents": self.extent},
        )
        return result

    def _mvt_features(self) -> list[dict[str, Any]]:
        out = []
        for _, _, feature in self._heap:
            for geometry in self.simplify_geometry(feature.geometry):
                out.append(
                    {
                        "geometry": geometry,
                        "properties": _mvt_properties(feature.properties),
                    }
                )
        return out


def _mvt_properties(properties: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in properties.items():
        if value is not None:
            for key, flattened_value in _flatten_property_items(name, value):
                out[key] = _mvt_property_value(flattened_value)
    return out


def _flatten_property_items(name: str, value: Any):
    if isinstance(value, dict):
        for key, item_value in value.items():
            if item_value is not None:
                yield from _flatten_property_items(f"{name}.{key}", item_value)
        return

    if _is_map_entries(value):
        for key, item_value in value:
            if item_value is not None:
                yield from _flatten_property_items(f"{name}.{key}", item_value)
        return

    yield name, value


def _is_map_entries(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    )


def _mvt_property_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
