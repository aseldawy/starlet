"""Tests for the shared on-the-fly tile index used by server and API paths."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkb
from shapely.geometry import box

from starlet._internal.server.tiler.parquet_index import (
    ParquetIndex,
    parse_parquet_bbox,
)


def test_parse_parquet_bbox_preserves_negative_fractional_coordinates():
    bbox = parse_parquet_bbox("tile_000001__-0_500000_0_000000_1_000000_2_000000.parquet")

    assert bbox == (-0.5, 0.0, 1.0, 2.0)


def test_load_and_reproject_pushdown_drops_internal_columns(temp_dir):
    tiles_dir = temp_dir / "parquet_tiles"
    tiles_dir.mkdir()
    tile_path = tiles_dir / "tile_000000__0_000000_0_000000_1_000000_1_000000.parquet"
    geom = wkb.dumps(box(0, 0, 1, 1))
    table = pa.table(
        {
            "geometry": [geom],
            "id": [7],
            "_tile_id": [3],
            "_bbox_xmin": [0.0],
            "_bbox_ymin": [0.0],
            "_bbox_xmax": [1.0],
            "_bbox_ymax": [1.0],
        }
    )
    pq.write_table(table, tile_path)

    index = ParquetIndex(tiles_dir)
    files = index.find_intersecting_files((0.25, 0.25, 0.75, 0.75))
    gdf = index.load_and_reproject(files[0], (0.25, 0.25, 0.75, 0.75))

    assert len(files) == 1
    assert len(gdf) == 1
    assert "id" in gdf.columns
    assert "_tile_id" not in gdf.columns
    assert "_bbox_xmin" not in gdf.columns
    assert "_bbox_ymin" not in gdf.columns
    assert "_bbox_xmax" not in gdf.columns
    assert "_bbox_ymax" not in gdf.columns
