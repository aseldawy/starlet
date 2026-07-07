"""Minimal pyramid tile partitioner for MVT-style tile assignment.

This helper is intentionally not wired into the current MVT pipeline yet.  It
maps geometry bounds in an arbitrary projected space to all overlapping tiles
from the deepest configured zoom back to zoom 0.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


TileCoord = Tuple[int, int, int]

_ZOOM_BITS = 5
_COORD_BITS = (64 - _ZOOM_BITS - 1) // 2
_COORD_MASK = (1 << _COORD_BITS) - 1


@dataclass(frozen=True)
class Bounds:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.minx, self.miny, self.maxx, self.maxy)):
            raise ValueError("bounds must contain finite values")
        if self.maxx <= self.minx or self.maxy <= self.miny:
            raise ValueError("bounds must have positive width and height")

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.minx, self.miny, self.maxx, self.maxy

    def expand(self, dx: float, dy: float) -> "Bounds":
        return Bounds(self.minx - dx, self.miny - dy, self.maxx + dx, self.maxy + dy)


class PyramidPartitioner:
    """Assign projected geometry MBRs to overlapping pyramid tiles.

    Parameters
    ----------
    bounds:
        Full pyramid extent as ``(minx, miny, maxx, maxy)``.  The input
        geometry must already be projected into this coordinate space.
    num_zoom_levels:
        Number of pyramid levels.  ``1`` means only zoom 0, ``2`` means zooms
        0 and 1, and so on.
    prefix_histogram:
        Optional 2D prefix-sum histogram.  When provided, tiles whose estimated
        count is lower than or equal to ``size_threshold`` are filtered out.
    size_threshold:
        Strict threshold for histogram filtering.  A tile is returned only when
        ``histogram_value > size_threshold``.
    buffer:
        Ratio of tile size by which geometry bounds are expanded at each level.
        For example, ``0.01`` expands by 1% of the tile width/height for the
        zoom level currently being processed.
    cache_size:
        Maximum number of accepted encoded tile IDs to keep in the LRU cache.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        num_zoom_levels: int,
        *,
        prefix_histogram: Optional[np.ndarray] = None,
        size_threshold: float = 0.0,
        buffer: float = 0.0,
        cache_size: int = 65_536,
    ) -> None:
        if num_zoom_levels <= 0:
            raise ValueError("num_zoom_levels must be positive")
        if num_zoom_levels > (1 << _ZOOM_BITS):
            raise ValueError("num_zoom_levels cannot exceed 32 with 5-bit zoom encoding")
        if buffer < 0:
            raise ValueError("buffer must be non-negative")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")

        if len(bounds) != 4:
            raise ValueError("bounds must contain four values")
        self.bounds = Bounds(*map(float, bounds))
        self.num_zoom_levels = int(num_zoom_levels)
        self.max_zoom = self.num_zoom_levels - 1
        self.prefix_histogram = self._validate_prefix(prefix_histogram)
        self.size_threshold = float(size_threshold)
        if self.prefix_histogram is not None and self.size_threshold > 0:
            self.max_zoom = min(
                self.max_zoom,
                self.estimate_deepest_level(self.prefix_histogram, self.size_threshold),
            )
            self.num_zoom_levels = self.max_zoom + 1
        self.buffer = float(buffer)
        self.cache_size = int(cache_size)
        self._accepted_cache: OrderedDict[int, None] = OrderedDict()

    @staticmethod
    def encode_tile_id(z: int, x: int, y: int) -> int:
        """Encode ``(z, x, y)`` into a compact positive integer."""
        return (int(z) << (_COORD_BITS * 2)) | (int(x) << _COORD_BITS) | int(y)

    @staticmethod
    def decode_tile_id(tile_id: int) -> TileCoord:
        """Decode an integer tile ID back into ``(z, x, y)``."""
        z = int(tile_id >> (_COORD_BITS * 2))
        x = int((tile_id >> _COORD_BITS) & _COORD_MASK)
        y = int(tile_id & _COORD_MASK)
        return z, x, y

    @property
    def cached_tile_ids(self) -> frozenset[int]:
        """Accepted tile IDs currently retained by the LRU cache."""
        return frozenset(self._accepted_cache.keys())

    def overlapping_tile_ids(self, geometry_or_bounds) -> list[int]:
        """Return accepted encoded tile IDs for a geometry or bounds tuple."""
        geom_bounds = self._coerce_geometry_bounds(geometry_or_bounds)
        if not all(math.isfinite(v) for v in geom_bounds.as_tuple()):
            return []
        if geom_bounds.maxx < geom_bounds.minx or geom_bounds.maxy < geom_bounds.miny:
            raise ValueError("geometry bounds are inverted")

        if self.buffer != 0.0:
            return self._overlapping_tile_ids_with_buffer(geom_bounds)

        current = self._tile_range_for_bounds(self.max_zoom, geom_bounds)
        if current is None:
            return []

        tx0, ty0, tx1, ty1 = current
        accepted: list[int] = []

        for z in range(self.max_zoom, -1, -1):
            for x in range(tx0, tx1 + 1):
                for y in range(ty0, ty1 + 1):
                    tile_id = self.encode_tile_id(z, x, y)
                    if self._tile_passes_filter(tile_id, z, x, y):
                        accepted.append(tile_id)
            tx0 //= 2
            ty0 //= 2
            tx1 //= 2
            ty1 //= 2

        return accepted

    def overlapping_tiles(self, geometry_or_bounds) -> list[TileCoord]:
        """Return accepted tiles as ``(z, x, y)`` tuples."""
        return [self.decode_tile_id(tile_id) for tile_id in self.overlapping_tile_ids(geometry_or_bounds)]

    @staticmethod
    def estimate_deepest_level(prefix_histogram: np.ndarray, size_threshold: float) -> int:
        """Estimate the deepest useful zoom level from a prefix histogram.

        The histogram stores counts at ``hist_zoom = log2(side_length)``.  If a
        histogram cell is already above the threshold, estimate how many deeper
        levels it can support by quartering its count at each level.  Otherwise,
        walk up the pyramid until a parent tile exceeds the threshold and return
        one level deeper than that parent.  If even the full histogram is at or
        below the threshold, return zoom 0.
        """
        if size_threshold <= 0:
            raise ValueError("size_threshold must be positive")

        prefix = PyramidPartitioner._validate_prefix_array(prefix_histogram)
        hist_size = prefix.shape[0]
        hist_zoom = int(math.log2(hist_size))

        padded = np.pad(prefix, ((1, 0), (1, 0)), mode="constant")
        raw_hist = padded[1:, 1:] - padded[:-1, 1:] - padded[1:, :-1] + padded[:-1, :-1]
        max_cell_value = float(np.max(raw_hist))
        max_y, max_x = np.unravel_index(int(np.argmax(raw_hist)), raw_hist.shape)

        if max_cell_value > size_threshold:
            level = hist_zoom
            value = max_cell_value
            while value / 4.0 > size_threshold:
                value /= 4.0
                level += 1
            return level

        for z in range(hist_zoom - 1, -1, -1):
            scale = 1 << (hist_zoom - z)
            parent_x = int(max_x) // scale
            parent_y = int(max_y) // scale
            x0 = parent_x * scale
            y0 = parent_y * scale
            x1 = x0 + scale - 1
            y1 = y0 + scale - 1
            if PyramidPartitioner._prefix_sum(prefix, x0, y0, x1, y1) > size_threshold:
                return z + 1

        return 0

    def _validate_prefix(self, prefix: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if prefix is None:
            return None
        return self._validate_prefix_array(prefix)

    @staticmethod
    def _validate_prefix_array(prefix: np.ndarray) -> np.ndarray:
        arr = np.asarray(prefix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("prefix_histogram must be a non-empty 2D array")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("prefix_histogram must be square")
        hist_zoom = math.log2(arr.shape[0])
        if int(hist_zoom) != hist_zoom:
            raise ValueError("prefix_histogram side length must be a power of two")
        return arr

    def _overlapping_tile_ids_with_buffer(self, geom_bounds: Bounds) -> list[int]:
        accepted: list[int] = []
        for z in range(self.max_zoom, -1, -1):
            tile_w, tile_h = self._tile_size(z)
            expanded = geom_bounds.expand(tile_w * self.buffer, tile_h * self.buffer)
            current = self._tile_range_for_bounds(z, expanded)
            if current is None:
                continue
            tx0, ty0, tx1, ty1 = current
            for x in range(tx0, tx1 + 1):
                for y in range(ty0, ty1 + 1):
                    tile_id = self.encode_tile_id(z, x, y)
                    if self._tile_passes_filter(tile_id, z, x, y):
                        accepted.append(tile_id)
        return accepted

    def _coerce_geometry_bounds(self, geometry_or_bounds) -> Bounds:
        if hasattr(geometry_or_bounds, "bounds"):
            raw = geometry_or_bounds.bounds
        else:
            raw = geometry_or_bounds
        if len(raw) != 4:
            raise ValueError("bounds must contain four values")
        return Bounds(*map(float, raw))

    def _tile_range_for_bounds(
        self,
        z: int,
        query: Bounds,
    ) -> Optional[Tuple[int, int, int, int]]:
        b = self.bounds
        if query.maxx < b.minx or query.minx > b.maxx or query.maxy < b.miny or query.miny > b.maxy:
            return None

        n = 1 << z
        tile_w, tile_h = self._tile_size(z)

        q_minx = max(query.minx, b.minx)
        q_maxx = min(query.maxx, b.maxx)
        q_miny = max(query.miny, b.miny)
        q_maxy = min(query.maxy, b.maxy)

        tx0 = self._coord_to_tile_floor(q_minx, b.minx, tile_w, n)
        tx1 = self._coord_to_tile_floor(q_maxx, b.minx, tile_w, n)
        ty0 = self._coord_to_tile_floor(b.maxy - q_maxy, 0.0, tile_h, n)
        ty1 = self._coord_to_tile_floor(b.maxy - q_miny, 0.0, tile_h, n)
        return tx0, ty0, tx1, ty1

    def _tile_size(self, z: int) -> Tuple[float, float]:
        n = 1 << z
        return (self.bounds.maxx - self.bounds.minx) / n, (self.bounds.maxy - self.bounds.miny) / n

    @staticmethod
    def _coord_to_tile_floor(value: float, origin: float, tile_size: float, n: int) -> int:
        return max(0, min(int((value - origin) // tile_size), n - 1))

    def _tile_passes_filter(self, tile_id: int, z: int, x: int, y: int) -> bool:
        if tile_id in self._accepted_cache:
            self._accepted_cache.move_to_end(tile_id)
            return True

        if self.prefix_histogram is not None:
            value = self._histogram_value(z, x, y)
            if value <= self.size_threshold:
                return False

        if self.cache_size > 0:
            self._accepted_cache[tile_id] = None
            self._accepted_cache.move_to_end(tile_id)
            while len(self._accepted_cache) > self.cache_size:
                self._accepted_cache.popitem(last=False)
        return True

    def _histogram_value(self, z: int, x: int, y: int) -> float:
        prefix = self.prefix_histogram
        if prefix is None:
            return math.inf

        hist_zoom = int(round(math.log2(prefix.shape[0])))
        if z == hist_zoom:
            return self._prefix_sum(prefix, x, y, x, y)

        if z < hist_zoom:
            scale = 1 << (hist_zoom - z)
            x0 = x * scale
            y0 = y * scale
            x1 = (x + 1) * scale - 1
            y1 = (y + 1) * scale - 1
            return self._prefix_sum(prefix, x0, y0, x1, y1)

        scale = 1 << (z - hist_zoom)
        parent_x = x // scale
        parent_y = y // scale
        return self._prefix_sum(prefix, parent_x, parent_y, parent_x, parent_y) / (scale * scale)

    @staticmethod
    def _prefix_sum(prefix: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
        h, w = prefix.shape
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            return 0.0
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h - 1))
        a = prefix[y1, x1]
        b = prefix[y0 - 1, x1] if y0 > 0 else 0.0
        c = prefix[y1, x0 - 1] if x0 > 0 else 0.0
        d = prefix[y0 - 1, x0 - 1] if y0 > 0 and x0 > 0 else 0.0
        return float(a - b - c + d)
