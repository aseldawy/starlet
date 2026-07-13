from shapely.geometry import Point

from starlet._internal.tiling.crs import (
    WEB_MERCATOR_CRS,
    WGS84_CRS,
    reproject_geometries,
)


def test_reproject_geometries_reprojects_between_distinct_crs():
    transformed, changed = reproject_geometries(
        Point(0, 0),
        WGS84_CRS,
        WEB_MERCATOR_CRS,
    )

    assert changed
    assert transformed.x == 0
    assert transformed.y == 0


def test_reproject_geometries_returns_original_for_matching_crs():
    point = Point(1, 2)

    transformed, changed = reproject_geometries(
        point,
        WEB_MERCATOR_CRS,
        WEB_MERCATOR_CRS,
    )

    assert transformed is point
    assert not changed
