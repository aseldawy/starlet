"""Tests for the standalone intermediate vector tile helper."""

import mapbox_vector_tile
import pytest
from shapely.geometry import LineString, Point, Polygon

from starlet._internal.mvt.intermediate_tile import IntermediateVectorTile


def _mercator_from_tile_pixel(tile, x, y):
    x_scale, _, _, y_scale, xoff, yoff = tile.affine_params
    return ((x - xoff) / x_scale, (y - yoff) / y_scale)


def test_initializes_web_mercator_to_tile_pixel_transform():
    tile = IntermediateVectorTile(0, 0, 0)

    transformed = tile.simplify_geometry(Point(0, 0))

    assert len(transformed) == 1
    assert transformed[0].x == pytest.approx(2048.0)
    assert transformed[0].y == pytest.approx(2048.0)


def test_simplify_geometry_trims_lines_to_buffered_tile_bounds():
    tile = IntermediateVectorTile(0, 0, 0)
    left = _mercator_from_tile_pixel(tile, -1000, 2048)
    right = _mercator_from_tile_pixel(tile, 5000, 2048)

    transformed = tile.simplify_geometry(LineString([left, right]))

    assert len(transformed) == 1
    assert list(transformed[0].coords) == [(-256.0, 2048.0), (4352.0, 2048.0)]


def test_simplify_geometry_clips_containing_polygon_to_tile_ring():
    tile = IntermediateVectorTile(0, 0, 0)
    corners = [
        _mercator_from_tile_pixel(tile, -1000, -1000),
        _mercator_from_tile_pixel(tile, -1000, 5000),
        _mercator_from_tile_pixel(tile, 5000, 5000),
        _mercator_from_tile_pixel(tile, 5000, -1000),
        _mercator_from_tile_pixel(tile, -1000, -1000),
    ]

    transformed = tile.simplify_geometry(Polygon(corners))

    assert len(transformed) == 1
    assert transformed[0].bounds == (-256.0, -256.0, 4352.0, 4352.0)


def test_add_feature_filters_null_properties_and_tracks_coordinates():
    tile = IntermediateVectorTile(0, 0, 0, coordinate_capacity=10)

    assert tile.add_feature(Point(0, 0), {"id": 1, "name": None}, priority=0.5)

    features = tile.features()
    assert tile.coordinate_count == 1
    assert tile.feature_count == 1
    assert features[0]["properties"] == {"id": 1}


def test_low_priority_feature_is_skipped_before_processing_when_full():
    tile = IntermediateVectorTile(0, 0, 0, coordinate_capacity=1)
    assert tile.add_feature(Point(0, 0), {"id": 1}, priority=0.5)

    called = False

    def fail_if_called(geometry):
        nonlocal called
        called = True
        raise AssertionError("low-priority feature should be skipped early")

    tile.simplify_geometry = fail_if_called

    assert not tile.add_feature(Point(1000, 0), {"id": 2}, priority=0.1)
    assert not called


def test_coordinate_capacity_keeps_highest_priority_features():
    tile = IntermediateVectorTile(0, 0, 0, coordinate_capacity=2)

    assert tile.add_feature(Point(-1000, 0), {"id": 1}, priority=0.1)
    assert tile.add_feature(Point(0, 0), {"id": 2}, priority=0.2)
    assert tile.add_feature(Point(1000, 0), {"id": 3}, priority=0.3)

    retained_ids = {feature["properties"]["id"] for feature in tile.features()}
    assert retained_ids == {2, 3}
    assert tile.coordinate_count == 2


def test_merge_combines_same_tile_without_simplifying_again():
    left = IntermediateVectorTile(0, 0, 0, coordinate_capacity=2)
    right = IntermediateVectorTile(0, 0, 0, coordinate_capacity=2)
    left.add_feature(Point(-1000, 0), {"id": 1}, priority=0.1)
    right.add_feature(Point(0, 0), {"id": 2}, priority=0.8)
    right.add_feature(Point(1000, 0), {"id": 3}, priority=0.9)

    def fail_if_called(geometry):
        raise AssertionError("merge should not simplify geometries")

    left.simplify_geometry = fail_if_called

    left.merge(right)

    retained_ids = {feature["properties"]["id"] for feature in left.features()}
    assert retained_ids == {2, 3}
    assert left.coordinate_count == 2


def test_merge_rejects_different_tile_ids():
    left = IntermediateVectorTile(0, 0, 0)
    right = IntermediateVectorTile(1, 0, 0)

    with pytest.raises(ValueError):
        left.merge(right)


def test_encode_returns_valid_mvt_binary():
    tile = IntermediateVectorTile(0, 0, 0, coordinate_capacity=10)
    tile.add_feature(Point(0, 0), {"id": 1}, priority=0.5)

    decoded = mapbox_vector_tile.decode(tile.encode())

    assert "layer0" in decoded
    assert len(decoded["layer0"]["features"]) == 1
    assert decoded["layer0"]["features"][0]["properties"]["id"] == 1
