"""Clip, transform, and encode geometries into Mapbox Vector Tile format."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import shapely
from shapely import make_valid, ops
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


def explode_collections(geom: Any) -> List[Any]:
    """Flatten GeometryCollections into a list of encodable simple/multi parts."""
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, GeometryCollection):
        out: List[Any] = []
        for g in geom.geoms:
            out.extend(explode_collections(g))
        return out

    if isinstance(geom, (Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint)):
        return [geom]

    return []


class MVTEncoder:
    """Clips a set of geometries to a tile and encodes features to MVT."""

    def __init__(self, bbox_3857: Tuple[float, float, float, float], tile_poly_3857: Any, extent: int = 4096) -> None:
        self.bbox_3857 = bbox_3857
        self.tile_poly_3857 = tile_poly_3857
        self.extent = extent

    def prepare_features(self, geoms: Any, attrs_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Vectorised make_valid → clip → simplify → scale-to-tile-coords.

        Array-at-a-time shapely 2 operations replace the previous per-feature
        Python loop (``iterrows`` + a per-coordinate ``ops.transform``
        callback), which dominated on-the-fly generation time on dense tiles.
        Simplification uses the same sub-pixel tolerance rule as the batch
        renderer (0.05% of tile width) so on-the-fly output matches the
        pre-generated pyramid.
        """
        minx, miny, maxx, maxy = self.bbox_3857
        width = (maxx - minx) or 1.0
        height = (maxy - miny) or 1.0

        garr = np.array(geoms, dtype=object)
        garr = shapely.make_valid(garr)
        garr = shapely.intersection(garr, self.tile_poly_3857)

        simplify_tol = width * 0.0005
        garr = shapely.simplify(garr, simplify_tol, preserve_topology=True)
        invalid = ~shapely.is_valid(garr)
        if invalid.any():
            garr[invalid] = shapely.make_valid(garr[invalid])

        def _scale(coords: np.ndarray) -> np.ndarray:
            out = np.empty_like(coords)
            out[:, 0] = (coords[:, 0] - minx) / width * self.extent
            out[:, 1] = (coords[:, 1] - miny) / height * self.extent
            return out

        garr = shapely.transform(garr, _scale)

        # Geometries are passed to the encoder as shapely objects:
        # mapbox_vector_tile parses GeoJSON dicts back into shapely
        # internally, so a mapping() round-trip would only add cost.
        features: List[Dict[str, Any]] = []
        for geom, attrs in zip(garr, attrs_list):
            if geom is None or geom.is_empty:
                continue
            for part in explode_collections(geom):
                if part.is_empty:
                    continue
                features.append({"geometry": part, "properties": attrs})
        return features

    # ── legacy per-feature helpers (kept for compatibility) ──────────────

    def clip_to_tile(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(make_valid)
        gdf = gdf[~gdf.geometry.is_empty]
        return gdf.clip(self.tile_poly_3857)

    def transform_geom(self, geom: Any, scale_func: Callable) -> Any:
        return ops.transform(scale_func, geom)

    # ── encoding ──────────────────────────────────────────────────────────

    def encode(self, features: List[Dict[str, Any]]) -> bytes:
        layer = {
            "name": "layer0",
            "features": features,
            "extent": self.extent
        }
        return mapbox_vector_tile.encode([layer])

    @staticmethod
    def empty_tile(extent: int = 4096) -> bytes:
        layer = {"name": "layer0", "features": [], "extent": extent}
        return mapbox_vector_tile.encode([layer])
