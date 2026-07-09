import json

import pyarrow as pa
import pytest
from pyproj import Transformer
from shapely import wkb
from shapely.geometry import Point

from starlet._internal.mvt.streamer import GeometryStreamer


def test_streamer_reprojects_from_geoparquet_crs_to_web_mercator():
    lon, lat = -118.25, 34.05
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:3857"}},
    }
    table = pa.table({
        "geometry": [wkb.dumps(Point(x, y))],
        "id": [1],
    }).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})

    geom, attrs = next(GeometryStreamer()._decode_table(table))

    assert geom.x == pytest.approx(x)
    assert geom.y == pytest.approx(y)
    assert attrs == {"id": 1}


def test_streamer_uses_geoparquet_primary_geometry_column():
    x, y = 10.0, 20.0
    geo = {
        "version": "1.1.0",
        "primary_column": "SHAPE",
        "columns": {"SHAPE": {"encoding": "WKB", "crs": "EPSG:3857"}},
    }
    table = pa.table({
        "SHAPE": [wkb.dumps(Point(x, y))],
        "id": [1],
    }).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})

    geom, attrs = next(GeometryStreamer()._decode_table(table))

    assert geom.x == pytest.approx(x)
    assert geom.y == pytest.approx(y)
    assert attrs == {"id": 1}
