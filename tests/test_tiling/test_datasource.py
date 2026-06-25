"""Unit tests for data source readers.

Tests cover:
- GeoParquetSource reading and iteration
- GeoJSONSource reading and iteration
- Column detection (geometry column)
- Error handling for missing files
- Schema validation
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point
from shapely import wkb

from starlet._internal.tiling.datasource import (
    GeoJSONSource,
    GeoJSONSplit,
    GeoParquetSource,
    GeoParquetSplit,
    read_spatial_sample,
    _iter_geojson_xy,
)
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader


class TestGeoParquetSource:
    """Test GeoParquet data source."""

    def test_init_with_file(self, sample_parquet_file):
        """Test initializing with a valid Parquet file."""
        source = GeoParquetSource(str(sample_parquet_file))
        # Source should be initialized successfully
        assert source is not None
        assert source.schema() is not None

    def test_init_with_missing_file(self, temp_dir):
        """Test that missing file raises appropriate error."""
        missing_file = temp_dir / "nonexistent.parquet"
        with pytest.raises(Exception):  # FileNotFoundError or similar
            GeoParquetSource(str(missing_file))

    def test_detect_geometry_column(self, sample_parquet_file):
        """Test that geometry column is in schema."""
        source = GeoParquetSource(str(sample_parquet_file))
        schema = source.schema()
        # Should have geometry column
        assert 'geometry' in schema.names

    def test_read_geometries(self, sample_parquet_file):
        """Test reading and decoding geometries."""
        source = GeoParquetSource(str(sample_parquet_file))

        # Use iter_tables() to read data
        for table in source.iter_tables():
            geoms_wkb = table['geometry'].to_pylist()
            # Decode first geometry
            geom = wkb.loads(geoms_wkb[0])
            assert geom is not None
            assert geom.geom_type == 'Polygon'
            break  # Only test first batch

    def test_read_all_columns(self, sample_parquet_file):
        """Test reading all columns including attributes."""
        source = GeoParquetSource(str(sample_parquet_file))

        for table in source.iter_tables():
            assert 'geometry' in table.column_names
            assert 'id' in table.column_names
            assert 'name' in table.column_names
            break  # Only test first batch

    def test_geometry_only_reads_only_geometry_column(self, sample_parquet_file):
        """Test geometry-only projection avoids reading attribute columns."""
        source = GeoParquetSource(str(sample_parquet_file), geometry_only=True)

        for table in source.iter_tables():
            assert table.column_names == ["geometry"]
            break

    def test_multiple_row_groups(self, temp_dir, sample_polygons):
        """Test reading Parquet file with multiple row groups."""
        # Create file with multiple row groups
        geoms = [wkb.dumps(g) for g in sample_polygons]
        table = pa.table({
            'geometry': geoms,
            'id': list(range(len(geoms)))
        })

        file_path = temp_dir / "multi_rg.parquet"
        pq.write_table(table, str(file_path), row_group_size=2)

        source = GeoParquetSource(str(file_path))
        # Check that we can iterate through all row groups
        tables = list(source.iter_tables())
        assert len(tables) > 1

    def test_geoparquet_splits_read_independently(self, temp_dir, sample_polygons):
        """Test row-group splits can be read independently."""
        geoms = [wkb.dumps(g) for g in sample_polygons]
        table = pa.table({
            'geometry': geoms,
            'id': list(range(len(geoms)))
        })

        file_path = temp_dir / "split_rg.parquet"
        pq.write_table(table, str(file_path), row_group_size=2)

        source = GeoParquetSource(str(file_path))
        splits = source.create_splits()
        assert len(splits) > 1
        assert all(isinstance(split, GeoParquetSplit) for split in splits)

        row_count = sum(
            table.num_rows
            for split in splits
            for table in source.iter_tables(split)
        )
        assert row_count == len(geoms)

    def test_directory_source_splits_all_geoparquet_files(self, temp_dir):
        """Test a directory source includes row groups from every Parquet file."""
        data_dir = temp_dir / "parquet_parts"
        data_dir.mkdir()
        for part, start in enumerate((0, 2)):
            pq.write_table(
                pa.table({
                    "geometry": [wkb.dumps(Point(start, start))],
                    "id": [start],
                }),
                str(data_dir / f"part-{part}.parquet"),
            )

        source = GeoParquetSource(str(data_dir))
        splits = source.create_splits()

        assert {Path(split.path).name for split in splits} == {"part-0.parquet", "part-1.parquet"}
        assert sum(
            table.num_rows
            for split in splits
            for table in source.iter_tables(split)
        ) == 2

    def test_schema_validation(self, sample_parquet_file):
        """Test that schema is accessible and valid."""
        source = GeoParquetSource(str(sample_parquet_file))
        schema = source.schema()

        assert 'geometry' in schema.names
        # Geometry should be binary type
        geom_field = schema.field('geometry')
        assert pa.types.is_binary(geom_field.type)


class TestGeoJSONSource:
    """Test GeoJSON data source.

    Note: GeoJSONSource implementation may vary. These are placeholder tests
    that should be adapted based on the actual implementation.
    """

    def test_read_geojson_feature_collection(self, temp_dir):
        """Test reading a GeoJSON FeatureCollection."""
        # Create a simple GeoJSON file
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0.0, 0.0]
                    },
                    "properties": {"id": 1}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [10.0, 10.0]
                    },
                    "properties": {"id": 2}
                }
            ]
        }

        json_path = temp_dir / "test.geojson"
        with open(json_path, 'w') as f:
            json.dump(geojson, f)

        source = GeoJSONSource(str(json_path))
        tables = list(source.iter_tables())
        ids = sorted(
            row_id
            for table in tables
            for row_id in table["id"].to_pylist()
        )

        assert sum(table.num_rows for table in tables) == 2
        assert ids == [1, 2]
        assert "geometry" in tables[0].column_names

    def test_read_empty_geojson(self, temp_dir):
        """Test reading empty GeoJSON file."""
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        json_path = temp_dir / "empty.geojson"
        with open(json_path, 'w') as f:
            json.dump(geojson, f)

        source = GeoJSONSource(str(json_path))
        assert list(source.iter_tables()) == []
        assert "geometry" in source.schema().names

    def test_partition_reader_returns_each_feature_once(self, temp_dir):
        """Test byte partitions align to complete Feature objects."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i, "name": f"feature-{i}"},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i * 2)]},
                }
                for i in range(12)
            ],
        }

        json_path = temp_dir / "partitioned.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        file_size = json_path.stat().st_size
        partition_size = max(1, file_size // 4)
        decoded = []

        for offset in range(0, file_size, partition_size):
            reader = GeoJSONPartitionReader(
                json_path,
                offset,
                min(partition_size, file_size - offset),
                batch_size=2,
            )
            for batch in reader:
                decoded.extend(json.loads(feature) for feature in batch)

        ids = sorted(feature["properties"]["id"] for feature in decoded)
        assert ids == list(range(12))

    def test_geojson_source_reads_feature_collection_in_parallel(self, temp_dir):
        """Test GeoJSONSource uses partitioned FeatureCollection reads."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
                }
                for i in range(20)
            ],
        }

        json_path = temp_dir / "parallel.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        source = GeoJSONSource(str(json_path), batch_rows=3)
        tables = list(source.iter_tables())
        ids = sorted(
            row_id
            for table in tables
            for row_id in table["id"].to_pylist()
        )

        assert ids == list(range(20))
        assert sum(table.num_rows for table in tables) == 20

    def test_geojson_nested_properties_have_stable_schema_across_batches(self, temp_dir):
        """Test dynamic JSON object properties do not infer different struct schemas."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1, "tagsMap": {"a": "1"}},
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"id": 2, "tagsMap": {"b": "2"}},
                    "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                },
            ],
        }

        json_path = temp_dir / "dynamic_tags.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        source = GeoJSONSource(str(json_path), batch_rows=1)
        source.schema()
        tables = list(source.iter_tables())

        _tags_types = {table.schema.field("tagsMap").type for table in tables}
        # nested dict properties are coerced to a stable string type across batches
        # (string or large_string depending on the pyarrow version) — the key
        # property is a single, consistent string type, never a struct.
        assert len(_tags_types) == 1
        assert all(
            pa.types.is_large_string(t) or pa.types.is_string(t) for t in _tags_types
        )
        assert [value for table in tables for value in table["tagsMap"].to_pylist()] == [
            '{"a":"1"}',
            '{"b":"2"}',
        ]

    def test_geojson_splits_read_independently_from_threads(self, temp_dir):
        """Test GeoJSON byte splits can be read independently by threads."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
                }
                for i in range(24)
            ],
        }

        json_path = temp_dir / "threaded_splits.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        source = GeoJSONSource(str(json_path), batch_rows=4)
        file_size = json_path.stat().st_size
        split_size = max(1, (file_size + 3) // 4)
        splits = [
            GeoJSONSplit(
                path=str(json_path),
                offset=offset,
                length=min(split_size, file_size - offset),
            )
            for offset in range(0, file_size, split_size)
        ]
        assert len(splits) == 4
        assert all(isinstance(split, GeoJSONSplit) for split in splits)

        def read_split(split):
            return [
                row_id
                for table in source.iter_tables(split)
                for row_id in table["id"].to_pylist()
            ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            ids = sorted(
                row_id
                for part in executor.map(read_split, splits)
                for row_id in part
            )

        assert ids == list(range(24))

    def test_directory_source_splits_all_geojson_files(self, temp_dir):
        """Test a directory source includes splits from every GeoJSON file."""
        data_dir = temp_dir / "geojson_parts"
        data_dir.mkdir()
        for part, feature_id in enumerate((1, 2)):
            (data_dir / f"part-{part}.geojson").write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"id": feature_id},
                    "geometry": {"type": "Point", "coordinates": [feature_id, feature_id]},
                }],
            }))

        source = GeoJSONSource(str(data_dir))
        splits = source.create_splits()
        ids = sorted(
            row_id
            for split in splits
            for table in source.iter_tables(split)
            for row_id in table["id"].to_pylist()
        )

        assert {Path(split.path).name for split in splits} == {"part-0.geojson", "part-1.geojson"}
        assert ids == [1, 2]

    def test_read_spatial_sample_returns_mbr_and_sample(self, temp_dir):
        """Test standalone sampling reads centroids and global MBR from a file."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
                    "geometry": {"type": "Point", "coordinates": [0.0, 2.0]},
                },
                {
                    "type": "Feature",
                    "properties": {"id": 2},
                    "geometry": {"type": "Point", "coordinates": [10.0, 12.0]},
                },
            ],
        }

        json_path = temp_dir / "sample.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)

        spatial_sample = read_spatial_sample(
            str(json_path),
            sample_cap=None,
            sample_ratio=1.0,
            seed=42,
            geojson_workers=1,
            geojson_executor="thread",
        )

        assert spatial_sample.total_seen == 2
        assert spatial_sample.total_sampled == 2
        assert spatial_sample.sample_points.shape == (2, 2)
        assert spatial_sample.mbr.mins.tolist() == [0.0, 2.0]
        assert spatial_sample.mbr.maxs.tolist() == [10.0, 12.0]

    def test_read_spatial_sample_splits_geojson_sample_cap(self, temp_dir):
        """Test GeoJSON partition sampling respects the total requested cap."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": i},
                    "geometry": {"type": "Point", "coordinates": [float(i), float(i * 2)]},
                }
                for i in range(24)
            ],
        }

        json_path = temp_dir / "capped.geojson"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        spatial_sample = read_spatial_sample(
            str(json_path),
            sample_cap=4,
            sample_ratio=1.0,
            seed=42,
            geojson_workers=4,
            geojson_executor="thread",
        )

        assert spatial_sample.total_seen == 24
        assert spatial_sample.total_sampled <= 4
        assert spatial_sample.mbr.mins.tolist() == [0.0, 0.0]
        assert spatial_sample.mbr.maxs.tolist() == [23.0, 46.0]

    def test_read_spatial_sample_geoparquet_uses_parallel_splits(self, temp_dir):
        """Test GeoParquet sampling merges row-group splits under one cap."""
        points = [Point(float(i), float(i + 1)) for i in range(10)]
        table = pa.table({
            "geometry": [wkb.dumps(point) for point in points],
            "id": list(range(10)),
        })
        parquet_path = temp_dir / "points.parquet"
        pq.write_table(table, str(parquet_path), row_group_size=2)

        spatial_sample = read_spatial_sample(
            str(parquet_path),
            sample_cap=3,
            sample_ratio=1.0,
            seed=42,
            geoparquet_workers=2,
        )

        assert spatial_sample.total_seen == 10
        assert spatial_sample.total_sampled == 3
        assert spatial_sample.batches_read == 5
        assert spatial_sample.sample_points.shape == (2, 3)
        assert spatial_sample.mbr.mins.tolist() == [0.0, 1.0]
        assert spatial_sample.mbr.maxs.tolist() == [9.0, 10.0]

    def test_iter_geojson_xy_walks_geometry_collections(self):
        """Test stack traversal over geometry objects and nested coordinates."""
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Point", "coordinates": [1.0, 2.0]},
                    {
                        "type": "LineString",
                        "coordinates": [[3.0, 4.0], [5.0, 6.0]],
                    },
                    {
                        "type": "GeometryCollection",
                        "geometries": [
                            {
                                "type": "Polygon",
                                "coordinates": [[
                                    [7.0, 8.0],
                                    [9.0, 10.0],
                                    [7.0, 8.0],
                                ]],
                            }
                        ],
                    },
                ],
            },
        }

        assert list(_iter_geojson_xy(json.dumps(feature))) == [
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 6.0),
            (7.0, 8.0),
            (9.0, 10.0),
            (7.0, 8.0),
        ]


class TestDataSourceIntegration:
    """Integration tests across data sources."""

    def test_consistent_geometry_reading(self, sample_parquet_file, sample_polygons):
        """Test that geometries are read correctly and match original data."""
        source = GeoParquetSource(str(sample_parquet_file))

        # Read all geometries using iter_tables()
        all_geoms = []
        for table in source.iter_tables():
            geoms_wkb = table['geometry'].to_pylist()
            all_geoms.extend([wkb.loads(g) for g in geoms_wkb])

        # Decode and compare bounds
        for i, geom in enumerate(all_geoms):
            original = sample_polygons[i]
            # Compare bounds (should be identical)
            assert geom.bounds == original.bounds
