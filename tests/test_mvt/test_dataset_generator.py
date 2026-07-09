"""Tests for the two-stage dataset MVT generator helpers."""

import json
from types import SimpleNamespace

import mapbox_vector_tile
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkb
from shapely.geometry import Point

from starlet._internal.mvt.mvt_generator import (
    _TableBatch,
    _bucket_tile_ids,
    _group_splits,
    _group_table_batches,
    _positive_bounds_tuple,
    _property_value,
    _single_tile_index_cache,
    _single_tile_parquet_index,
    generate_single_mvt_tile,
)
from starlet._internal.tiling.geoparquet_source import GeoParquetSplit


def test_group_splits_round_robins_to_requested_groups():
    splits = [
        GeoParquetSplit(path=f"part-{index}.parquet", row_groups=(0,))
        for index in range(5)
    ]

    groups = _group_splits(splits, 2)

    assert groups == [
        [splits[0], splits[2], splits[4]],
        [splits[1], splits[3]],
    ]


def test_group_table_batches_splits_rows_across_requested_groups():
    table = pa.table({"value": list(range(10))})

    groups = _group_table_batches(table, 3)

    assert len(groups) == 3
    assert all(isinstance(group[0], _TableBatch) for group in groups)
    assert [group[0].table.num_rows for group in groups] == [4, 4, 2]


def test_bucket_tile_ids_uses_mod_hash():
    assert _bucket_tile_ids([1, 2, 3, 4, 5], 3) == [
        [3],
        [1, 4],
        [2, 5],
    ]


def test_positive_bounds_tuple_expands_zero_sized_bounds():
    minx, miny, maxx, maxy = _positive_bounds_tuple((1.0, 2.0, 1.0, 2.0))

    assert minx == 1.0
    assert miny == 2.0
    assert maxx > minx
    assert maxy > miny


def test_property_value_keeps_simple_scalars_and_stringifies_complex_values():
    assert _property_value("x") == "x"
    assert _property_value(10) == 10
    assert _property_value(1.5) == 1.5
    assert _property_value(("a", "b")) == "('a', 'b')"


def test_generate_single_mvt_tile_uses_partition_and_row_bbox_pruning(tmp_path):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [
                wkb.dumps(Point(-100.0, 80.0)),
                wkb.dumps(Point(100.0, -80.0)),
            ],
            "id": [1, 2],
            "_bbox_xmin": [-100.0, 100.0],
            "_bbox_ymin": [80.0, -80.0],
            "_bbox_xmax": [-100.0, 100.0],
            "_bbox_ymax": [80.0, -80.0],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__-100_0_-80_0_100_0_80_0.parquet",
    )

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (1, 0, 0),
        feature_capacity=10,
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    features = decoded["layer0"]["features"]
    assert len(features) == 1
    assert features[0]["properties"] == {"id": 1}


def test_generate_single_mvt_tile_queries_buffered_tile_bounds(tmp_path):
    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB", "crs": "EPSG:4326"}},
    }
    table = pa.table(
        {
            "geometry": [wkb.dumps(Point(1.0, 45.0))],
            "id": [9],
            "_bbox_xmin": [1.0],
            "_bbox_ymin": [45.0],
            "_bbox_xmax": [1.0],
            "_bbox_ymax": [45.0],
        }
    ).replace_schema_metadata({b"geo": json.dumps(geo).encode("utf-8")})
    pq.write_table(
        table,
        parquet_dir / "tile_000000__-1_0_40_0_2_0_50_0.parquet",
    )

    tile_bytes = generate_single_mvt_tile(
        str(dataset_dir),
        (1, 0, 0),
        feature_capacity=10,
        extent=4096,
        buffer=256,
    )
    decoded = mapbox_vector_tile.decode(tile_bytes)

    features = decoded["layer0"]["features"]
    assert len(features) == 1
    assert features[0]["properties"] == {"id": 9}


def test_single_tile_parquet_index_is_cached_by_path(tmp_path):
    parquet_dir = tmp_path / "dataset" / "parquet_tiles"
    parquet_dir.mkdir(parents=True)
    _single_tile_index_cache.clear()

    first = _single_tile_parquet_index(parquet_dir)
    second = _single_tile_parquet_index(parquet_dir)

    assert first is second


def test_generate_mvt_forwards_temp_dir(monkeypatch, tmp_path):
    import starlet

    captured = {}

    class FakeDatasetMVTGenerator:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.DatasetMVTGenerator",
        FakeDatasetMVTGenerator,
    )
    outdir = tmp_path / "mvt"
    outdir.mkdir()
    temp_dir = tmp_path / "scratch"

    starlet.generate_mvt(
        str(tmp_path / "dataset"),
        outdir=str(outdir),
        temp_dir=str(temp_dir),
    )

    assert captured["temp_dir"] == str(temp_dir)


def test_generate_mvt_forwards_pmtiles_settings(monkeypatch, tmp_path):
    import starlet

    captured = {}

    class FakeDatasetMVTGenerator:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def run(self):
            return SimpleNamespace(
                pmtiles_path=str(tmp_path / "dataset" / "tiles.pmtiles"),
                tile_count=3,
                tile_counts_by_zoom=[1, 2],
                zoom_levels=[0, 1],
            )

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.DatasetMVTGenerator",
        FakeDatasetMVTGenerator,
    )
    outdir = tmp_path / "mvt"
    outdir.mkdir()

    starlet.generate_mvt(
        str(tmp_path / "dataset"),
        outdir=str(outdir),
        pmtiles=True,
        pmtiles_compression="brotli",
    )

    assert captured["output_format"] == "pmtiles"
    assert captured["pmtiles_compression"] == "brotli"


def test_dataset_generator_removes_mvt_dir_after_pmtiles_export(monkeypatch, tmp_path):
    from starlet._internal.mvt.mvt_generator import DatasetMVTGenerator, _MapStageResult

    dataset_dir = tmp_path / "dataset"
    parquet_dir = dataset_dir / "parquet_tiles"
    hist_dir = dataset_dir / "histograms"
    parquet_dir.mkdir(parents=True)
    hist_dir.mkdir(parents=True)
    (hist_dir / "global_prefix.npy").write_bytes(b"fake")

    generator = DatasetMVTGenerator(
        str(dataset_dir),
        num_zoom_levels=2,
        threshold=0,
        output_format="pmtiles",
    )
    tile_path = generator.outdir / "0" / "0"
    tile_path.mkdir(parents=True)
    (tile_path / "0.mvt").write_bytes(b"tile")

    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.GeoParquetSource",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator._create_map_groups",
        lambda source, workers: [[object()]],
    )
    monkeypatch.setattr(
        DatasetMVTGenerator,
        "_run_map_stage",
        lambda self, groups, source, temp_root: [_MapStageResult("tmp", [1])],
    )
    monkeypatch.setattr(
        DatasetMVTGenerator,
        "_run_reduce_stage",
        lambda self, map_results: None,
    )
    exported = {}
    monkeypatch.setattr(
        "starlet._internal.mvt.mvt_generator.export_to_pmtiles",
        lambda **kwargs: exported.update(kwargs),
    )

    result = generator.run()

    assert result.tile_counts_by_zoom == [1]
    assert result.tile_count == 1
    assert result.pmtiles_path == str(dataset_dir / "tiles.pmtiles")
    assert exported["output_path"] == str(dataset_dir / "tiles.pmtiles")
    assert not generator.outdir.exists()
