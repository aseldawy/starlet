from shapely.geometry import Point

from starlet._internal.tiling.crs import (
    WEB_MERCATOR_CRS,
    WGS84_CRS,
    get_transformer,
    reproject_geometries,
)


def test_get_transformer_returns_none_for_matching_crs():
    assert get_transformer(WEB_MERCATOR_CRS, WEB_MERCATOR_CRS) is None


def test_reproject_geometries_uses_provided_transformer():
    transformer = get_transformer(WGS84_CRS, WEB_MERCATOR_CRS)

    transformed, changed = reproject_geometries(
        Point(0, 0),
        WGS84_CRS,
        WEB_MERCATOR_CRS,
        transformer=transformer,
    )

    assert changed
    assert transformed.x == 0
    assert transformed.y == 0


def test_reproject_geometries_accepts_cached_noop_transformer():
    point = Point(1, 2)

    transformed, changed = reproject_geometries(
        point,
        WGS84_CRS,
        WEB_MERCATOR_CRS,
        transformer=None,
    )

    assert transformed is point
    assert not changed
