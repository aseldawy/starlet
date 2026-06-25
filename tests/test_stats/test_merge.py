"""Tests for mergeable attribute-statistics sketches.

The two-stage tiling orchestrator collects partial statistics in parallel
assignment workers and merges them in the parent process. These tests verify
that merging partial collectors yields the same result as a single pass.
"""
import random

import pyarrow as pa
from shapely import wkb
from shapely.geometry import Point

from starlet._internal.stats.collector import AttributeStatsCollector


def _make_table(n, seed):
    random.seed(seed)
    return pa.table({
        "geometry": [
            wkb.dumps(Point(random.uniform(-120, -118), random.uniform(33, 35)))
            for _ in range(n)
        ],
        "num": [random.randint(0, 500) for _ in range(n)],
        "cat": [random.choice(["a", "b", "c", "d"]) for _ in range(n)],
        "name": [random.choice(["alpha", "beta", "gamma"]) for _ in range(n)],
    })


def _by_name(result):
    return {a["name"]: a["stats"] for a in result["attributes"]}


def test_merge_matches_single_pass():
    tbl = _make_table(4000, seed=7)
    half = tbl.num_rows // 2

    single = AttributeStatsCollector(tbl.schema)
    single.consume_table(tbl)
    expected = _by_name(single.finalize())

    a = AttributeStatsCollector(tbl.schema)
    a.consume_table(tbl.slice(0, half))
    b = AttributeStatsCollector(tbl.schema)
    b.consume_table(tbl.slice(half))
    a.merge(b)
    merged = _by_name(a.finalize())

    assert set(merged) == set(expected)

    # Exactly-mergeable quantities must be identical.
    assert merged["num"]["non_null_count"] == expected["num"]["non_null_count"]
    assert merged["num"]["min"] == expected["num"]["min"]
    assert merged["num"]["max"] == expected["num"]["max"]
    assert abs(merged["num"]["mean"] - expected["num"]["mean"]) < 1e-9
    assert merged["num"]["approx_distinct"] == expected["num"]["approx_distinct"]
    assert merged["cat"]["non_null_count"] == expected["cat"]["non_null_count"]
    assert merged["cat"]["approx_distinct"] == expected["cat"]["approx_distinct"]
    assert merged["geometry"]["mbr"] == expected["geometry"]["mbr"]
    assert merged["geometry"]["total_points"] == expected["geometry"]["total_points"]


def test_merge_into_empty_adopts_other():
    tbl = _make_table(100, seed=1)
    empty = AttributeStatsCollector(tbl.schema)
    other = AttributeStatsCollector(tbl.schema)
    other.consume_table(tbl)
    empty.merge(other)
    merged = _by_name(empty.finalize())
    assert merged["num"]["non_null_count"] == 100


def test_merge_three_partials():
    full = _make_table(900, seed=3)
    single = AttributeStatsCollector(full.schema)
    single.consume_table(full)
    expected = _by_name(single.finalize())

    acc = AttributeStatsCollector(full.schema)
    for i in range(3):
        part = AttributeStatsCollector(full.schema)
        part.consume_table(full.slice(i * 300, 300))
        acc.merge(part)
    merged = _by_name(acc.finalize())

    assert merged["num"]["min"] == expected["num"]["min"]
    assert merged["num"]["max"] == expected["num"]["max"]
    assert merged["num"]["non_null_count"] == expected["num"]["non_null_count"]
    assert merged["geometry"]["mbr"] == expected["geometry"]["mbr"]
