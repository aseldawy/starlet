import json
from pathlib import Path

import pyarrow as pa
import pytest
from shapely import wkb
from shapely.geometry import Point

from starlet._internal.tiling.datasource import (
    GPXSource,
    GPXSplit,
    read_spatial_sample,
    source_for_path,
)


def _write_gpx(path: Path, body: str, *, version: str = "1.1") -> None:
    namespace = f"http://www.topografix.com/GPX/{version}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f"<gpx xmlns=\"{namespace}\" version=\"{version}\" creator=\"pytest\">\n"
        f"{body}\n"
        "</gpx>\n",
        encoding="utf-8",
    )


def test_gpx_source_reads_tracks_recursively_and_preserves_hierarchy(temp_dir):
    _write_gpx(
        temp_dir / "user-a" / "track.gpx",
        """
  <metadata>
    <name>Root name</name>
    <desc>Root description</desc>
    <time>2024-01-01T00:00:00Z</time>
    <keywords>walk,test</keywords>
    <bounds minlat="40.0" minlon="-118.0" maxlat="41.0" maxlon="-117.0" />
    <extensions><dataset>alpha</dataset></extensions>
  </metadata>
  <trk>
    <name>Morning track</name>
    <cmt>track comment</cmt>
    <desc>track description</desc>
    <src>watch</src>
    <link href="https://example.com/track"><text>track link</text></link>
    <number>7</number>
    <type>run</type>
    <extensions><trackCustom>yes</trackCustom></extensions>
    <trkseg>
      <extensions><segmentCustom>one</segmentCustom></extensions>
      <trkpt lat="40.0" lon="-118.0">
        <ele>10.5</ele>
        <time>2024-01-01T00:00:01Z</time>
        <name>start</name>
        <cmt>point comment</cmt>
        <desc>point description</desc>
        <src>gps</src>
        <sym>Dot</sym>
        <type>trail</type>
        <fix>3d</fix>
        <sat>8</sat>
        <hdop>1.25</hdop>
        <vdop>2.5</vdop>
        <pdop>3.75</pdop>
        <ageofdgpsdata>4.5</ageofdgpsdata>
        <dgpsid>12</dgpsid>
        <link href="https://example.com/point"><text>point link</text></link>
        <extensions><speed>5</speed></extensions>
      </trkpt>
      <trkpt lat="41.0" lon="-117.0">
        <ele>11.5</ele>
        <time>2024-01-01T00:00:02Z</time>
      </trkpt>
    </trkseg>
  </trk>
""",
    )
    _write_gpx(
        temp_dir / "user-b" / "route.GPX",
        """
  <rte>
    <name>Route name</name>
    <number>3</number>
    <type>bike</type>
    <rtept lat="35.0" lon="-120.0">
      <ele>15.0</ele>
      <time>2024-01-02T00:00:00Z</time>
    </rtept>
  </rte>
""",
    )

    source = GPXSource(str(temp_dir), batch_rows=1)
    splits = source.create_splits(num_splits=20)
    tables = list(source.iter_tables())
    table = pa.concat_tables(tables)

    assert all(isinstance(split, GPXSplit) for split in splits)
    assert len(splits) == 2
    assert len(tables) == 3
    assert table.num_rows == 3
    assert table["filename"].to_pylist() == ["track.gpx", "track.gpx", "route.GPX"]
    assert table["gpx_version"].to_pylist() == ["1.1", "1.1", "1.1"]
    assert table["gpx_creator"].to_pylist() == ["pytest", "pytest", "pytest"]
    assert table["gpx_name"].to_pylist()[0] == "Root name"
    assert json.loads(table["gpx_bounds"].to_pylist()[0]) == {
        "maxlat": "41.0",
        "maxlon": "-117.0",
        "minlat": "40.0",
        "minlon": "-118.0",
    }
    assert table["point_kind"].to_pylist() == ["track", "track", "route"]
    assert table["track_index"].to_pylist() == [0, 0, None]
    assert table["segment_index"].to_pylist() == [0, 0, None]
    assert table["route_index"].to_pylist() == [None, None, 0]
    assert table["point_index"].to_pylist() == [0, 1, 0]
    assert table["track_name"].to_pylist()[:2] == ["Morning track", "Morning track"]
    assert table["track_number"].to_pylist()[:2] == [7, 7]
    assert table["route_name"].to_pylist()[2] == "Route name"
    assert table["route_number"].to_pylist()[2] == 3
    assert table["latitude"].to_pylist() == [40.0, 41.0, 35.0]
    assert table["longitude"].to_pylist() == [-118.0, -117.0, -120.0]
    assert table["elevation"].to_pylist() == [10.5, 11.5, 15.0]
    assert table["point_time"].to_pylist() == [
        "2024-01-01T00:00:01Z",
        "2024-01-01T00:00:02Z",
        "2024-01-02T00:00:00Z",
    ]
    assert table["point_name"].to_pylist()[0] == "start"
    assert table["point_satellites"].to_pylist()[0] == 8
    assert table["point_hdop"].to_pylist()[0] == pytest.approx(1.25)
    assert "speed" in table["point_extensions_xml"].to_pylist()[0]
    assert "segmentCustom" in table["segment_extensions_xml"].to_pylist()[0]
    assert json.loads(table["point_links"].to_pylist()[0]) == [
        {"href": "https://example.com/point", "text": "point link"}
    ]

    first_point = wkb.loads(table["geometry"][0].as_py())
    assert first_point.equals(Point(-118.0, 40.0))


def test_gpx_source_reads_waypoints_and_single_file_with_basename_id(temp_dir):
    path = temp_dir / "points.gpx"
    _write_gpx(
        path,
        """
  <wpt lat="34.0" lon="-118.25">
    <name>Standalone waypoint</name>
    <ele>100.0</ele>
  </wpt>
""",
    )

    source = source_for_path(str(path))
    table = next(source.iter_tables())

    assert isinstance(source, GPXSource)
    assert len(source.create_splits()) == 1
    assert table["filename"].to_pylist() == ["points.gpx"]
    assert table["point_kind"].to_pylist() == ["waypoint"]
    assert table["point_index"].to_pylist() == [0]
    assert table["point_name"].to_pylist() == ["Standalone waypoint"]


def test_source_for_path_detects_gpx_directory_and_custom_geometry(temp_dir):
    _write_gpx(
        temp_dir / "track.gpx",
        """
  <trk>
    <trkseg>
      <trkpt lat="40.0" lon="-118.0" />
    </trkseg>
  </trk>
""",
    )

    source = source_for_path(str(temp_dir), geom_col="geom")
    table = next(source.iter_tables())
    geo = json.loads(table.schema.metadata[b"geo"].decode("utf-8"))

    assert isinstance(source, GPXSource)
    assert table.column_names[-1] == "geom"
    assert geo["primary_column"] == "geom"
    assert geo["columns"]["geom"]["crs"] == "EPSG:4326"


def test_gpx_spatial_sample_uses_all_points_and_infers_compact_schema(temp_dir):
    _write_gpx(
        temp_dir / "track.gpx",
        """
  <trk>
    <name>Track name</name>
    <number>4</number>
    <trkseg>
      <trkpt lat="40.0" lon="-118.0">
        <ele>10.0</ele>
        <time>2024-01-01T00:00:00Z</time>
      </trkpt>
      <trkpt lat="41.0" lon="-117.0">
        <ele>20.0</ele>
        <time>2024-01-01T00:00:10Z</time>
      </trkpt>
    </trkseg>
  </trk>
""",
    )

    sample = read_spatial_sample(
        str(temp_dir),
        sample_ratio=1.0,
        source_workers=1,
    )

    assert sample.total_seen == 2
    assert sample.total_sampled == 2
    assert sample.schema is not None
    assert sample.schema.names == [
        "filename",
        "gpx_version",
        "gpx_creator",
        "point_kind",
        "track_index",
        "track_name",
        "track_number",
        "segment_index",
        "point_index",
        "latitude",
        "longitude",
        "elevation",
        "point_time",
        "geometry",
    ]
    assert sample.mbr.getMinCoord(0) == pytest.approx(-118.0)
    assert sample.mbr.getMaxCoord(0) == pytest.approx(-117.0)
    assert sample.mbr.getMinCoord(1) == pytest.approx(40.0)
    assert sample.mbr.getMaxCoord(1) == pytest.approx(41.0)

    source = source_for_path(str(temp_dir))
    source.set_schema(sample.schema)
    table = next(source.iter_tables())

    assert table.column_names == sample.schema.names
    assert "gpx_name" not in table.column_names
    assert "track_comment" not in table.column_names
    assert "route_index" not in table.column_names
    assert "point_name" not in table.column_names


def test_gpx_source_reports_context_for_invalid_points(temp_dir):
    _write_gpx(
        temp_dir / "broken.gpx",
        """
  <trk>
    <trkseg>
      <trkpt lat="40.0" />
    </trkseg>
  </trk>
""",
    )
    source = GPXSource(str(temp_dir))

    with pytest.raises(ValueError, match=r"broken\.gpx .*missing 'lon'"):
        list(source.iter_tables())
