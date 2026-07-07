from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from starlet._cli import main


def test_tile_command_uses_config_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "starlet.toml"
    config_path.write_text(
        """
[global]
temp_dir = "/tmp/starlet"
parallelism = 6
log_level = "DEBUG"

[tile]
partition_size = "256mb"
sort = "hilbert"
compression = "gzip"
sample_cap = 123
sample_ratio = 0.5
csv_split_size = "64mb"
max_parallel_files = 11
grid_size = 1024
dtype = "float32"
sfc_bits = 20
orchestrator = "two-stage"
""".strip()
    )

    captured = {}

    def fake_tile(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            num_files=1,
            total_rows=2,
            outdir=kwargs["outdir"],
            histogram_path="hist.npy",
        )

    monkeypatch.setattr("starlet.tile", fake_tile)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(config_path),
            "tile",
            "--input",
            "input.geojson",
            "--outdir",
            "dataset",
        ],
    )

    assert result.exit_code == 0
    assert captured["partition_size"] == 256 * 1024 * 1024
    assert captured["sort"] == "hilbert"
    assert captured["compression"] == "gzip"
    assert captured["sample_cap"] == 123
    assert captured["sample_ratio"] == 0.5
    assert captured["csv_split_size"] == 64 * 1024 * 1024
    assert captured["parallelism"] == 6
    assert captured["max_parallel_files"] == 11
    assert captured["grid_size"] == 1024
    assert captured["histogram_dtype"] == "float32"
    assert captured["sfc_bits"] == 20
    assert captured["temp_dir"] == "/tmp/starlet"


def test_mvt_command_prefers_cli_zoom_and_uses_config_threshold(tmp_path, monkeypatch):
    config_path = tmp_path / "starlet.toml"
    config_path.write_text(
        """
[global]
parallelism = 4

[mvt]
zoom = 5
threshold = 777
feature_capacity = 333
extent = 2048
buffer = 64
partition_buffer = 0.25
""".strip()
    )

    captured = {}

    def fake_generate_mvt(*, tile_dir, zoom, threshold, outdir, parallelism, temp_dir, feature_capacity, extent, buffer, partition_buffer):
        captured.update(
            tile_dir=tile_dir,
            zoom=zoom,
            threshold=threshold,
            outdir=outdir,
            parallelism=parallelism,
            temp_dir=temp_dir,
            feature_capacity=feature_capacity,
            extent=extent,
            buffer=buffer,
            partition_buffer=partition_buffer,
        )
        return SimpleNamespace(tile_count=1, outdir=outdir or "mvt", zoom_levels=[0, 1])

    monkeypatch.setattr("starlet.generate_mvt", fake_generate_mvt)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(config_path),
            "mvt",
            "--dir",
            "dataset",
            "--zoom",
            "9",
        ],
    )

    assert result.exit_code == 0
    assert captured["zoom"] == 9
    assert captured["threshold"] == 777.0
    assert captured["parallelism"] == 4
    assert captured["feature_capacity"] == 333
    assert captured["extent"] == 2048
    assert captured["buffer"] == 64
    assert captured["partition_buffer"] == 0.25


def test_serve_command_uses_config_host_port_and_cache(tmp_path, monkeypatch):
    config_path = tmp_path / "starlet.toml"
    config_path.write_text(
        """
[serve]
host = "127.0.0.1"
port = 9001
cache_size = 99
""".strip()
    )

    captured = {}

    class FakeApp:
        def run(self, host, port, debug, use_reloader, threaded):
            captured.update(
                host=host,
                port=port,
                debug=debug,
                use_reloader=use_reloader,
                threaded=threaded,
            )

    def fake_create_app(*, data_dir, cache_size):
        captured.update(data_dir=data_dir, cache_size=cache_size)
        return FakeApp()

    monkeypatch.setattr("starlet.create_app", fake_create_app)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            str(config_path),
            "serve",
            "--dir",
            "datasets",
        ],
    )

    assert result.exit_code == 0
    assert captured["data_dir"] == "datasets"
    assert captured["cache_size"] == 99
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9001
