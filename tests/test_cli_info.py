from __future__ import annotations

import gzip

from click.testing import CliRunner

from starlet._cli import main


def test_info_prints_parquet_histogram_and_mvt_details(sample_dataset_dir):
    for z, count in ((0, 1), (1, 2), (3, 1)):
        for x in range(count):
            mvt_path = sample_dataset_dir / "mvt" / str(z) / str(x)
            mvt_path.mkdir(parents=True, exist_ok=True)
            (mvt_path / "0.mvt").write_bytes(b"tile")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "info",
            "--dir",
            str(sample_dataset_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Parquet bbox:no" in result.output
    assert "Parquet CRS: (unknown)" in result.output
    assert "Hist res:    64" in result.output
    assert "Zoom levels: [1, 2, 0, 1] total=4" in result.output
    assert "MVT count:   4" in result.output


def test_info_reads_zoom_counts_from_pmtiles(sample_dataset_dir):
    from pmtiles.tile import Compression, TileType, zxy_to_tileid
    from pmtiles.writer import Writer

    pmtiles_path = sample_dataset_dir / "tiles.pmtiles"
    with open(pmtiles_path, "wb") as handle:
        writer = Writer(handle)
        for z, x, y in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 3, 1)):
            writer.write_tile(zxy_to_tileid(z, x, y), gzip.compress(b"tile"))
        writer.finalize(
            {
                "tile_compression": Compression.GZIP,
                "tile_type": TileType.MVT,
                "min_zoom": 0,
                "max_zoom": 2,
            },
            {},
        )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "info",
            "--dir",
            str(sample_dataset_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Zoom levels: [1, 2, 1] total=4" in result.output
    assert "MVT count:   4" in result.output
