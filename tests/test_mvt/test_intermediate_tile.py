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
    left = _mercator_from_tile_pixel(tile, -1000, -1000)
    right = _mercator_from_tile_pixel(tile, 5000, 5000)

    transformed = tile.simplify_geometry(LineString([left, right]))

    assert len(transformed) == 1
    assert list(transformed[0].coords) == [(-256.0, -256.0), (4352.0, 4352.0)]


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


def test_add_feature_filters_null_properties():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)

    assert tile.add_feature(Point(0, 0), {"id": 1, "name": None})

    assert tile.feature_count == 1
    assert tile._features[0].properties == {"id": 1}


def test_add_feature_delays_simplification_until_features_are_requested():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    called = False
    original_simplify_geometry = tile.simplify_geometry

    def fail_if_called(geometry):
        nonlocal called
        called = True
        raise AssertionError("add_feature should only sample raw geometries")

    tile.simplify_geometry = fail_if_called
    assert tile.add_feature(Point(0, 0), {"id": 1})
    assert tile._features[0].properties == {"id": 1}
    assert not called

    tile.simplify_geometry = original_simplify_geometry
    assert mapbox_vector_tile.decode(tile.encode())["layer0"]["features"][0]["properties"] == {"id": 1}


def test_encode_flattens_nested_properties_for_mvt_only():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    properties = {
        "id": 1,
        "tagsMap": {"name": "Oak Hill", "landuse": "cemetery"},
        "metadata": {"source": "osm", "nested": {"rank": 2}},
    }

    assert tile.add_feature(Point(0, 0), properties)
    assert tile._features[0].properties == properties

    decoded = mapbox_vector_tile.decode(tile.encode())["layer0"]["features"][0]["properties"]
    assert decoded == {
        "id": 1,
        "tagsMap.name": "Oak Hill",
        "tagsMap.landuse": "cemetery",
        "metadata.source": "osm",
        "metadata.nested.rank": 2,
    }


def test_encode_flattens_loaded_arrow_map_properties_for_mvt_only(tmp_path):
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    properties = {
        "tagsMap": [("name", "Oak Hill"), ("landuse", "cemetery")],
    }

    assert tile.add_feature(Point(0, 0), properties)
    path = tmp_path / "tile.arrow"
    tile.write_features(path)

    loaded = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    loaded.load_features(path)
    assert loaded._features[0].properties == {
        "tagsMap": [["name", "Oak Hill"], ["landuse", "cemetery"]]
    }

    decoded = mapbox_vector_tile.decode(loaded.encode())["layer0"]["features"][0]["properties"]
    assert decoded == {
        "tagsMap.name": "Oak Hill",
        "tagsMap.landuse": "cemetery",
    }


def test_feature_can_be_skipped_by_lower_priority():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    assert tile.add_feature(Point(0, 0), {"id": 1}, priority=10)

    called = False

    def fail_if_called(geometry):
        nonlocal called
        called = True
        raise AssertionError("unsampled feature should skip processing")

    tile.simplify_geometry = fail_if_called

    assert not tile.add_feature(Point(1000, 0), {"id": 2}, priority=5)
    assert not called


def test_feature_capacity_evicts_lowest_priority_to_make_room():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=2)

    assert tile.add_feature(Point(-1000, 0), {"id": 1}, priority=1)
    assert tile.add_feature(Point(0, 0), {"id": 2}, priority=5)
    assert tile.add_feature(Point(1000, 0), {"id": 3}, priority=9)

    retained_ids = {feature.properties["id"] for feature in tile._features}
    assert retained_ids == {2, 3}
    assert tile.feature_count == 2
    assert tile._features_seen == 3


def test_merge_combines_same_tile_without_simplifying_again():
    left = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    right = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    left.add_feature(Point(-1000, 0), {"id": 1}, priority=1)
    right.add_feature(Point(0, 0), {"id": 2}, priority=5)
    right.add_feature(Point(1000, 0), {"id": 3}, priority=9)

    def fail_if_called(geometry):
        raise AssertionError("merge should not simplify geometries")

    original_simplify_geometry = left.simplify_geometry
    original_add_feature = left.add_feature
    left.simplify_geometry = fail_if_called
    left.add_feature = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("merge should combine feature lists directly")
    )

    left.merge(right)

    left.simplify_geometry = original_simplify_geometry
    left.add_feature = original_add_feature
    retained_ids = {feature.properties["id"] for feature in left._features}
    assert retained_ids == {2, 3}
    assert left.feature_count == 2
    assert left._features_seen == 3


def test_merge_rejects_different_tile_ids():
    left = IntermediateVectorTile(0, 0, 0)
    right = IntermediateVectorTile(1, 0, 0)

    with pytest.raises(ValueError):
        left.merge(right)


def test_encode_returns_valid_mvt_binary():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    tile.add_feature(Point(0, 0), {"id": 1})

    decoded = mapbox_vector_tile.decode(tile.encode())

    assert "layer0" in decoded
    assert len(decoded["layer0"]["features"]) == 1
    assert decoded["layer0"]["features"][0]["properties"]["id"] == 1


def test_encode_accepts_layer_name():
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    tile.add_feature(Point(0, 0), {"id": 1})

    decoded = mapbox_vector_tile.decode(tile.encode(layer_name="custom"))

    assert "custom" in decoded


def test_feature_arrow_roundtrip_populates_tile_state(tmp_path):
    path = tmp_path / "0-0-0.pyarrow"
    tile = IntermediateVectorTile(0, 0, 0, feature_capacity=1)
    tile.add_feature(Point(0, 0), {"id": 1}, priority=9)
    tile.add_feature(Point(1, 1), {"id": 2}, priority=1)
    tile.write_features(path)

    loaded = IntermediateVectorTile(0, 0, 0, feature_capacity=10)
    loaded.load_features(path)

    assert loaded.feature_count == 1
    assert loaded._features_seen == 2
    assert loaded._features[0].properties == {"id": 1}
    # Priorities survive the roundtrip so reduce-side merges rank correctly.
    assert loaded._heap[0][0] == 9


def test_same_geometry_gets_same_default_priority_in_every_tile():
    """A geometry offered to two different tiles must win or lose in both."""
    from starlet._internal.mvt.intermediate_tile import feature_priority

    shared = Point(0, 0).buffer(10.0)
    assert feature_priority(shared.wkb) == feature_priority(shared.wkb)

    left = IntermediateVectorTile(1, 0, 0, feature_capacity=1)
    right = IntermediateVectorTile(1, 1, 0, feature_capacity=1)
    for tile in (left, right):
        tile.add_feature(shared, {"id": "shared"})

    competitor = Point(1, 1).buffer(5.0)
    left_kept = left.add_feature(competitor, {"id": "competitor"})
    right_kept = right.add_feature(competitor, {"id": "competitor"})

    # Identical candidates -> identical decision in both tiles (no seams).
    assert left_kept == right_kept
    left_ids = {feature.properties["id"] for feature in left._features}
    right_ids = {feature.properties["id"] for feature in right._features}
    assert left_ids == right_ids


def test_merge_is_order_independent():
    a = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    b = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    c = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
    a.add_feature(Point(-1000, 0), {"id": 1}, priority=3)
    b.add_feature(Point(0, 0), {"id": 2}, priority=7)
    c.add_feature(Point(1000, 0), {"id": 3}, priority=5)

    def merged_ids(order):
        first = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
        for src in order:
            clone = IntermediateVectorTile(0, 0, 0, feature_capacity=2)
            for prio, _, feat in src._heap:
                clone.add_feature(feat.geometry, dict(feat.properties), priority=prio)
            first.merge(clone)
        return {feature.properties["id"] for feature in first._features}

    assert merged_ids([a, b, c]) == merged_ids([c, b, a]) == {2, 3}
