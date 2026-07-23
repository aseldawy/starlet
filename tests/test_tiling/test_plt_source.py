import json
from pathlib import Path

import pyarrow as pa
import pytest
from shapely import wkb
from shapely.geometry import Point

from starlet._internal.tiling.datasource import (
    PLTSource,
    PLTSplit,
    read_spatial_sample,
    source_for_path,
)


_PLT_HEADER = (
    "Geolife trajectory\n"
    "WGS 84\n"
    "Altitude is in Feet\n"
    "Reserved 3\n"
    "0,2,255,My Track,0,0,2,8421376\n"
    "0\n"
)


def _write_plt(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PLT_HEADER + "".join(f"{row}\n" for row in rows))


def test_plt_source_reads_files_recursively_and_groups_points(temp_dir):
    _write_plt(
        temp_dir / "user-a" / "track.plt",
        [
            "40.008241,116.319894,0,219,39979.2472569444,2009-06-15,05:56:03",
            "40.008185,116.319814,0,184,39979.2473263889,2009-06-15,05:56:09",
        ],
    )
    _write_plt(
        temp_dir / "user-b" / "track.PLT",
        ["34.050000,-118.250000,0,-777,39980.0,2009-06-16,00:00:00"],
    )

    source = PLTSource(str(temp_dir), batch_rows=1)
    splits = source.create_splits(num_splits=20)
    tables = list(source.iter_tables())
    table = pa.concat_tables(tables)

    assert all(isinstance(split, PLTSplit) for split in splits)
    assert len(splits) == 2
    assert len(tables) == 3
    assert table.num_rows == 3
    assert table["filename"].to_pylist() == ["track.plt", "track.plt", "track.PLT"]
    assert table["altitude"].to_pylist() == [219.0, 184.0, -777.0]
    assert table["date"].to_pylist() == ["2009-06-15", "2009-06-15", "2009-06-16"]
    assert table["time"].to_pylist() == ["05:56:03", "05:56:09", "00:00:00"]

    first_point = wkb.loads(table["geometry"][0].as_py())
    assert first_point.equals(Point(116.319894, 40.008241))


def test_source_for_path_detects_plt_directory_and_custom_geometry(temp_dir):
    _write_plt(
        temp_dir / "track.plt",
        ["40.0,116.0,0,10,39979.0,2009-06-15,00:00:00"],
    )

    source = source_for_path(str(temp_dir), geom_col="geom")
    table = next(source.iter_tables())
    geo = json.loads(table.schema.metadata[b"geo"].decode("utf-8"))

    assert isinstance(source, PLTSource)
    assert table.column_names[-1] == "geom"
    assert geo["primary_column"] == "geom"
    assert geo["columns"]["geom"]["crs"] == "EPSG:4326"


def test_plt_source_reads_a_single_file_with_basename_id(temp_dir):
    path = temp_dir / "track.plt"
    _write_plt(path, ["40.0,116.0,0,10,39979.0,2009-06-15,00:00:00"])

    source = source_for_path(str(path))
    table = next(source.iter_tables())

    assert isinstance(source, PLTSource)
    assert len(source.create_splits()) == 1
    assert table["filename"].to_pylist() == ["track.plt"]


def test_plt_spatial_sample_uses_all_points(temp_dir):
    _write_plt(
        temp_dir / "nested" / "track.plt",
        [
            "40.0,116.0,0,10,39979.0,2009-06-15,00:00:00",
            "41.0,118.0,0,20,39980.0,2009-06-16,00:00:00",
        ],
    )

    sample = read_spatial_sample(
        str(temp_dir),
        sample_ratio=1.0,
        source_workers=1,
    )

    assert sample.total_seen == 2
    assert sample.total_sampled == 2
    assert sample.mbr.getMinCoord(0) == pytest.approx(116.0)
    assert sample.mbr.getMaxCoord(0) == pytest.approx(118.0)
    assert sample.mbr.getMinCoord(1) == pytest.approx(40.0)
    assert sample.mbr.getMaxCoord(1) == pytest.approx(41.0)


def test_plt_source_reports_file_and_line_for_invalid_records(temp_dir):
    _write_plt(temp_dir / "broken.plt", ["not,a,valid,point"])
    source = PLTSource(str(temp_dir))

    with pytest.raises(ValueError, match=r"broken\.plt at line 7: expected 7 fields"):
        list(source.iter_tables())
