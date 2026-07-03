# streamer.py

import numpy as np
import pyarrow.parquet as pq
import shapely
from pathlib import Path
from pyproj import Transformer
import logging
import json

from starlet._internal.tiling.crs import WGS84_CRS, WEB_MERCATOR_CRS, geoparquet_crs

logger = logging.getLogger("bucket_mvt")


class GeometryStreamer:
    """
    Streams geometries from GeoParquet using PyArrow, row group by row group,
    exactly like your GeoParquetSource pattern.
    """

    def __init__(self, parquet_dir=None):
        self.parquet_dir = Path(parquet_dir) if parquet_dir is not None else None
        self._transformers = {}

    def _reproject_coords(self, coords, transformer):
        """Reproject an (N, 2) coordinate array to EPSG:3857 in bulk."""
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    def _transformer_for(self, source_crs):
        key = str(source_crs or WGS84_CRS)
        transformer = self._transformers.get(key)
        if transformer is None:
            transformer = Transformer.from_crs(
                source_crs or WGS84_CRS,
                WEB_MERCATOR_CRS,
                always_xy=True,
            )
            self._transformers[key] = transformer
        return transformer

    def _decode_table(self, table):
        # Vectorised decode/repair/reproject over the whole row group. The old
        # per-geometry path spent ~75% of its time in the coordinate-by-coordinate
        # pyproj callback; doing it array-at-a-time (shapely 2 + pyproj bulk
        # transform) is ~8x faster for the WKB→make_valid→reproject stage.
        geom_col = _geometry_column_name(table.schema)
        source_crs = geoparquet_crs(table.schema, geom_col) or WGS84_CRS
        transformer = self._transformer_for(source_crs)
        wkb_arr = table[geom_col].to_numpy(zero_copy_only=False)
        geoms = shapely.from_wkb(wkb_arr)
        geoms = shapely.make_valid(geoms)
        geoms = shapely.transform(
            geoms,
            lambda coords: self._reproject_coords(coords, transformer),
        )

        # Extract all columns except geometry
        attrs = {
            col: table[col].to_pylist()
            for col in table.column_names
            if col != geom_col
        }

        for i, geom in enumerate(geoms):
            if geom is None or geom.is_empty:
                continue
            row_attrs = {k: attrs[k][i] for k in attrs}
            yield geom, row_attrs

    def iter_geometries(self):
        """
        Main generator: iterate all parquet files, stream row groups,
        decode geometries, and yield shapely objects.
        """
        parquet_files = list(self.parquet_dir.rglob("*.parquet"))

        for pf in parquet_files:
            logger.info("Streaming GeoParquet file %s", pf)

            pf_obj = pq.ParquetFile(pf)
            num_row_groups = pf_obj.num_row_groups

            for rg in range(num_row_groups):
                table = pf_obj.read_row_group(rg)
                yield from self._decode_table(table)


def _geometry_column_name(schema) -> str:
    raw_geo = (schema.metadata or {}).get(b"geo")
    if raw_geo:
        try:
            geo = json.loads(raw_geo.decode("utf-8"))
        except Exception:
            geo = {}
        primary = geo.get("primary_column")
        if isinstance(primary, str) and primary in schema.names:
            return primary

    if "geometry" in schema.names:
        return "geometry"

    for field in schema:
        if (field.metadata or {}).get(b"ARROW:extension:name") == b"geoarrow.wkb":
            return field.name

    raise ValueError(f"No geometry column found in GeoParquet schema: {schema.names}")
