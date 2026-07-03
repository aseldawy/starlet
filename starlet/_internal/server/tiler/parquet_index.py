"""Filename-based spatial index for GeoParquet tiles.

Parquet tiles are named ``tile_XXXXXX__minx_miny_maxx_maxy.parquet``.  The
bounding box is parsed from the filename to enable fast MBR intersection
filtering without reading file metadata.

On-the-fly tile generation only needs the geometries that fall inside the
requested tile.  Two layers of pruning make that cheap:

  1. **Partition pruning** — the filename bbox selects which partitions can
     intersect a tile (``find_intersecting_files``).
  2. **Row-group + row pruning** — tiles written by the current tiling stage
     carry per-row bbox "covering" columns (``_bbox_*``) and are split into
     spatially-coherent row groups.  ``load_and_reproject`` then uses pyarrow
     predicate pushdown to read only the row groups and rows whose bbox
     overlaps the tile, decoding a handful of geometries instead of the whole
     partition.

Older tiles without the ``_bbox_*`` columns fall back to a cached full read +
in-memory bbox pre-filter, so the server stays correct on legacy datasets.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyproj import Transformer

from starlet._internal.tiling.crs import WGS84_CRS, WEB_MERCATOR_CRS, geoparquet_crs

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]

# Per-row bbox covering columns written by the tiling stage (see writer_pool).
BBOX_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")


def parse_parquet_bbox(fname: str) -> Optional[BBox]:
    """Parse ``(minx, miny, maxx, maxy)`` from a tile filename, or ``None``.

    Expected format: ``tile_XXXXXX__minx_miny_maxx_maxy.parquet`` where each
    coordinate is encoded as an ``int_decimal`` pair (e.g. ``-97_123`` →
    ``-97.123``).
    """
    try:
        coord = fname.replace(".parquet", "").split("__")[1].split("_")
    except IndexError:
        return None
    nums: List[float] = []
    pair: List[str] = []
    for p in coord:
        pair.append(p)
        if len(pair) == 2:
            try:
                nums.append(float(pair[0] + "." + pair[1]))
            except ValueError:
                return None
            pair = []
    if len(nums) != 4:
        return None
    return (nums[0], nums[1], nums[2], nums[3])


def bbox_intersects(a: BBox, b: BBox) -> bool:
    """Whether two ``(minx, miny, maxx, maxy)`` boxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


class ParquetIndex:
    """Spatial index over GeoParquet tiles with read-time pruning.

    Filename bounding boxes are parsed once at construction.  For legacy tiles
    (no bbox columns) decoded partitions are kept in a bounded LRU cache
    (``partition_cache_size``) so panning/zooming in one region does not
    re-read the file.
    """

    def __init__(self, folder: Path, partition_cache_size: int = 4) -> None:
        self.folder = Path(folder)
        self._entries: List[Tuple[Path, BBox]] = []
        if self.folder.exists():
            for pf in sorted(self.folder.glob("*.parquet")):
                bbox = parse_parquet_bbox(pf.name)
                if bbox is not None:
                    self._entries.append((pf, bbox))
        self._partition_cache_size = partition_cache_size
        # key -> (native GeoDataFrame, cached per-geometry bounds DataFrame)
        self._partition_cache: "OrderedDict[str, Tuple[gpd.GeoDataFrame, object]]" = OrderedDict()
        # key -> (column_names, geometry_column, has_bbox_columns, crs)
        self._schema_cache: dict = {}
        self._transformer_cache: dict = {}

    # Kept for backward compatibility with callers that used the static helper.
    parse_parquet_bbox = staticmethod(parse_parquet_bbox)

    def find_intersecting_files(self, bbox_4326: BBox) -> List[Path]:
        """Partitions whose filename bbox overlaps ``bbox_4326``.

        Partition filenames store bboxes in the partition's native CRS, so the
        requested WGS84 tile bounds are transformed before comparison.
        """
        matches = []
        for pf, pbbox in self._entries:
            _, _, _, crs = self._schema_info(pf)
            query_bbox = self._transform_bbox(bbox_4326, WGS84_CRS, crs)
            if bbox_intersects(pbbox, query_bbox):
                matches.append(pf)
        return matches

    # ------------------------------------------------------------------ schema

    def _schema_info(self, path: Path):
        """Return ``(names, geometry_column, has_bbox)`` for a partition (cached)."""
        key = str(path)
        info = self._schema_cache.get(key)
        if info is not None:
            return info
        schema = pq.ParquetFile(path).schema_arrow
        names = list(schema.names)
        has_bbox = all(c in names for c in BBOX_COLS)
        geom_col = "geometry"
        if geom_col not in names:
            geom_col = names[-1] if names else "geometry"
            meta = schema.metadata or {}
            raw = meta.get(b"geo")
            if raw:
                try:
                    geom_col = json.loads(raw).get("primary_column", geom_col)
                except Exception:
                    pass
        crs = geoparquet_crs(schema, geom_col) or WGS84_CRS
        info = (names, geom_col, has_bbox, crs)
        self._schema_cache[key] = info
        return info

    def _transformer(self, src_crs, dst_crs):
        key = (str(src_crs or WGS84_CRS), str(dst_crs or WGS84_CRS))
        transformer = self._transformer_cache.get(key)
        if transformer is None:
            transformer = Transformer.from_crs(
                src_crs or WGS84_CRS,
                dst_crs or WGS84_CRS,
                always_xy=True,
            )
            self._transformer_cache[key] = transformer
        return transformer

    def _transform_bbox(self, bbox: BBox, src_crs, dst_crs) -> BBox:
        if str(src_crs or WGS84_CRS) == str(dst_crs or WGS84_CRS):
            return bbox
        transformer = self._transformer(src_crs, dst_crs)
        return tuple(transformer.transform_bounds(*bbox, densify_pts=21))

    # ------------------------------------------------------------------ reads

    def _read_native(self, path: Path):
        """Read a partition in its native CRS (defaulting to EPSG:4326).

        Returns ``(gdf, bounds_df)`` with a per-geometry envelope for spatial
        pre-filtering.  Used for legacy tiles; results are LRU-cached.
        """
        if self._partition_cache_size <= 0:
            _, _, _, crs = self._schema_info(path)
            gdf = gpd.read_parquet(path)
            if gdf.crs is None:
                gdf = gdf.set_crs(crs)
            return gdf, gdf.geometry.bounds

        key = str(path)
        cached = self._partition_cache.get(key)
        if cached is not None:
            self._partition_cache.move_to_end(key)
            return cached

        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            _, _, _, crs = self._schema_info(path)
            gdf = gdf.set_crs(crs)
        entry = (gdf, gdf.geometry.bounds)
        self._partition_cache[key] = entry
        self._partition_cache.move_to_end(key)
        while len(self._partition_cache) > self._partition_cache_size:
            self._partition_cache.popitem(last=False)
        return entry

    def _pushdown_read(self, path: Path, geom_col: str, bbox_native: BBox, crs) -> gpd.GeoDataFrame:
        """Read only rows whose bbox overlaps ``bbox_native``, reprojected to 3857.

        Uses pyarrow predicate pushdown on the ``_bbox_*`` columns; row groups
        whose statistics miss the tile are skipped entirely.
        """
        minx, miny, maxx, maxy = bbox_native
        flt = (
            (pc.field("_bbox_xmax") >= minx)
            & (pc.field("_bbox_xmin") <= maxx)
            & (pc.field("_bbox_ymax") >= miny)
            & (pc.field("_bbox_ymin") <= maxy)
        )
        table = pq.read_table(path, filters=flt)
        drop = [c for c in BBOX_COLS if c in table.column_names]
        if drop:
            table = table.drop(drop)
        df = table.to_pandas()
        if geom_col not in df.columns or len(df) == 0:
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=WEB_MERCATOR_CRS))
        geom = gpd.GeoSeries.from_wkb(df[geom_col].to_numpy(), crs=crs)
        gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs=crs)
        return gdf.to_crs(WEB_MERCATOR_CRS)

    def load_and_reproject(self, path: Path, bbox_4326: Optional[BBox] = None) -> gpd.GeoDataFrame:
        """Load a partition in EPSG:3857, pruned to ``bbox_4326`` when given.

        When the tile carries ``_bbox_*`` columns this uses pyarrow row-group +
        row pushdown (cost ~ geometries in the tile).  Otherwise it falls back
        to a cached full read plus an in-memory bbox pre-filter.  Both paths
        produce identical geometry sets — the downstream clip is exact.
        """
        if bbox_4326 is not None:
            try:
                _, geom_col, has_bbox, crs = self._schema_info(path)
                bbox_native = self._transform_bbox(bbox_4326, WGS84_CRS, crs)
            except Exception:
                has_bbox = False
                geom_col = "geometry"
                crs = WGS84_CRS
                bbox_native = bbox_4326
            if has_bbox:
                return self._pushdown_read(path, geom_col, bbox_native, crs)

        gdf, bounds = self._read_native(path)
        if bbox_4326 is not None and len(gdf):
            bbox_native = self._transform_bbox(bbox_4326, WGS84_CRS, gdf.crs or WGS84_CRS)
            minx, miny, maxx, maxy = bbox_native
            mask = ~(
                (bounds["maxx"] < minx)
                | (bounds["minx"] > maxx)
                | (bounds["maxy"] < miny)
                | (bounds["miny"] > maxy)
            )
            gdf = gdf.loc[mask]
        drop = [c for c in BBOX_COLS if c in gdf.columns]
        if drop:
            gdf = gdf.drop(columns=drop)
        if len(gdf) and gdf.crs is not None and gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(WEB_MERCATOR_CRS)
        return gdf
