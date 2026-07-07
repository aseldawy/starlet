from __future__ import annotations

from click.testing import CliRunner

from starlet._cli import main


def test_info_prints_parquet_histogram_and_mvt_details(sample_dataset_dir):
    mvt_path = sample_dataset_dir / "mvt" / "0" / "0"
    mvt_path.mkdir(parents=True)
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
    assert "MVT count:   1" in result.output
