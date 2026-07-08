"""Click CLI for starlet: spatial tiling, MVT generation, and tile serving."""
from __future__ import annotations

import logging
import sys

import click

from starlet._internal.config import (
    command_parallelism,
    load_config,
    parse_size_value,
    resolve_command_value,
    set_loaded_config,
)


_LOG_RECORD_FACTORY_INSTALLED = False


def _install_relative_seconds_factory() -> None:
    global _LOG_RECORD_FACTORY_INSTALLED
    if _LOG_RECORD_FACTORY_INSTALLED:
        return

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.relativeSeconds = record.relativeCreated / 1000.0
        return record

    logging.setLogRecordFactory(record_factory)
    _LOG_RECORD_FACTORY_INSTALLED = True


def _setup_logging(log_level: str) -> None:
    _install_relative_seconds_factory()
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(relativeSeconds).3fs] %(levelname)s %(name)s: %(message)s",
    )


def _resolved_log_level(command: str, explicit: str | None) -> str:
    return str(resolve_command_value(command, "log_level", explicit, default="INFO"))


@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=str),
    help="Path to a Starlet TOML config file.",
)
@click.pass_context
@click.version_option(package_name="starlet")
def main(ctx: click.Context, config_path: str | None):
    """starlet — spatial tiling, MVT generation, and tile serving."""
    config = load_config(config_path)
    set_loaded_config(config, config_path)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@main.command()
@click.option("--input", "input_path", required=True, help="Path to a supported geospatial source.")
@click.option("--outdir", required=True, help="Output dataset directory.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for partitioner.")
@click.option("--geom-col", default="geometry", show_default=True, help="Geometry column name.")
@click.option("--csv-x-col", default=None, help="CSV x-coordinate column. Use with --csv-y-col.")
@click.option("--csv-y-col", default=None, help="CSV y-coordinate column. Use with --csv-x-col.")
@click.option("--csv-wkt-col", default=None, help="CSV WKT geometry column.")
@click.option("--src-crs", default="EPSG:4326", show_default=True, help="Source CRS hint for CSV inputs.")
@click.option("--covering-bbox/--no-covering-bbox", default=True, show_default=True,
              help="Write per-row bbox covering columns + bounded row groups for "
                   "fast on-demand serving. Use --no-covering-bbox to disable.")
@click.option("--log-level", default=None, help="Logging level.")
def tile(
    input_path,
    outdir,
    seed,
    geom_col,
    csv_x_col,
    csv_y_col,
    csv_wkt_col,
    src_crs,
    covering_bbox,
    log_level,
):
    """Partition a geospatial dataset into spatially-tiled Parquet files."""
    _setup_logging(_resolved_log_level("tile", log_level))
    import starlet

    parallelism = command_parallelism("tile")
    result = starlet.tile(
        input=input_path,
        outdir=outdir,
        parallelism=parallelism,
        partition_size=parse_size_value(resolve_command_value("tile", "partition_size", None, default=None)),
        sort=str(resolve_command_value("tile", "sort", None, default="zorder")),
        compression=str(resolve_command_value("tile", "compression", None, default="zstd")),
        sample_cap=resolve_command_value("tile", "sample_cap", None, default=10_000),
        sample_ratio=float(resolve_command_value("tile", "sample_ratio", None, default=1.0)),
        seed=seed,
        geom_col=geom_col,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_split_size=parse_size_value(resolve_command_value("tile", "csv_split_size", None, default="32mb")),
        src_crs=src_crs,
        sfc_bits=int(resolve_command_value("tile", "sfc_bits", None, default=16)),
        covering_bbox=covering_bbox,
        temp_dir=resolve_command_value("tile", "temp_dir", None, default=None),
        grid_size=int(resolve_command_value("tile", "grid_size", None, default=4096)),
        histogram_dtype=str(resolve_command_value("tile", "dtype", None, default="float64")),
    )
    click.echo(f"Tiling complete: {result.num_files} tiles, {result.total_rows} rows")
    click.echo(f"  Output: {result.outdir}")
    click.echo(f"  Histogram: {result.histogram_path}")


@main.command()
@click.option("--dir", "tile_dir", required=True, help="Dataset directory with parquet_tiles/ and histograms/.")
@click.option("--zoom", type=int, default=None, help="Maximum zoom level.")
@click.option("--outdir", default=None, help="MVT output directory (default: <dir>/mvt/).")
@click.option("--pmtiles/--no-pmtiles", default=None, help="Export generated tiles to a PMTiles archive.")
@click.option("--log-level", default=None, help="Logging level.")
def mvt(tile_dir, zoom, outdir, pmtiles, log_level):
    """Generate Mapbox Vector Tiles from a tiled dataset."""
    _setup_logging(_resolved_log_level("mvt", log_level))
    import starlet

    result = starlet.generate_mvt(
        tile_dir=tile_dir,
        zoom=int(resolve_command_value("mvt", "zoom", zoom, default=7)),
        threshold=float(resolve_command_value("mvt", "threshold", None, default=100_000)),
        pmtiles=bool(resolve_command_value("mvt", "pmtiles", pmtiles, default=False)),
        pmtiles_compression=str(resolve_command_value("mvt", "pmtiles_compression", None, default="gzip")),
        outdir=outdir,
        temp_dir=resolve_command_value("mvt", "temp_dir", None, default=None),
        parallelism=command_parallelism("mvt"),
        feature_capacity=int(resolve_command_value("mvt", "feature_capacity", None, default=10_000)),
        extent=int(resolve_command_value("mvt", "extent", None, default=4096)),
        buffer=int(resolve_command_value("mvt", "buffer", None, default=256)),
    )
    click.echo(f"MVT generation complete: {result.tile_count} tiles")
    click.echo(f"  Output: {result.outdir}")
    click.echo(f"  Zoom levels: {result.zoom_levels}")
    if result.pmtiles_path:
        click.echo(f"  PMTiles: {result.pmtiles_path}")


@main.command()
@click.option("--input", "input_path", required=True, help="Path to a supported geospatial source.")
@click.option("--outdir", required=True, help="Output dataset directory.")
@click.option("--zoom", type=int, default=None, help="Maximum zoom level.")
@click.option("--covering-bbox/--no-covering-bbox", default=True, show_default=True,
              help="Write per-row bbox covering columns for fast on-demand serving. "
                   "Use --no-covering-bbox to disable.")
@click.option("--csv-x-col", default=None, help="CSV x-coordinate column. Use with --csv-y-col.")
@click.option("--csv-y-col", default=None, help="CSV y-coordinate column. Use with --csv-x-col.")
@click.option("--csv-wkt-col", default=None, help="CSV WKT geometry column.")
@click.option("--src-crs", default="EPSG:4326", show_default=True, help="Source CRS hint for CSV inputs.")
@click.option("--pmtiles/--no-pmtiles", default=None, help="Export MVT tiles to a PMTiles archive after generation.")
@click.option("--log-level", default=None, help="Logging level.")
def build(
    input_path,
    outdir,
    zoom,
    covering_bbox,
    csv_x_col,
    csv_y_col,
    csv_wkt_col,
    src_crs,
    pmtiles,
    log_level,
):
    """Run the full pipeline: tile then generate MVTs."""
    _setup_logging(_resolved_log_level("build", log_level))
    import starlet

    parallelism = command_parallelism("build", fallback_sections=("tile", "mvt"))
    tile_result, mvt_result, pmtiles_path = starlet.build(
        input=input_path,
        outdir=outdir,
        zoom=int(resolve_command_value("build", "zoom", zoom, fallback_sections=("mvt",), default=7)),
        partition_size=parse_size_value(resolve_command_value("build", "partition_size", None, fallback_sections=("tile",), default=None)),
        threshold=float(resolve_command_value("build", "threshold", None, fallback_sections=("mvt",), default=100_000)),
        pmtiles=bool(resolve_command_value("build", "pmtiles", pmtiles, fallback_sections=("mvt",), default=False)),
        pmtiles_compression=str(resolve_command_value("build", "pmtiles_compression", None, fallback_sections=("mvt",), default="gzip")),
        temp_dir=resolve_command_value("build", "temp_dir", None, default=None),
        parallelism=parallelism,
        feature_capacity=int(resolve_command_value("build", "feature_capacity", None, fallback_sections=("mvt",), default=10_000)),
        extent=int(resolve_command_value("build", "extent", None, fallback_sections=("mvt",), default=4096)),
        buffer=int(resolve_command_value("build", "buffer", None, fallback_sections=("mvt",), default=256)),
        sort=str(resolve_command_value("build", "sort", None, fallback_sections=("tile",), default="zorder")),
        compression=str(resolve_command_value("build", "compression", None, fallback_sections=("tile",), default="zstd")),
        sample_cap=resolve_command_value("build", "sample_cap", None, fallback_sections=("tile",), default=10_000),
        sample_ratio=float(resolve_command_value("build", "sample_ratio", None, fallback_sections=("tile",), default=1.0)),
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_split_size=parse_size_value(resolve_command_value("build", "csv_split_size", None, fallback_sections=("tile",), default="32mb")),
        src_crs=src_crs,
        sfc_bits=int(resolve_command_value("build", "sfc_bits", None, fallback_sections=("tile",), default=16)),
        covering_bbox=covering_bbox,
        grid_size=int(resolve_command_value("build", "grid_size", None, fallback_sections=("tile",), default=4096)),
        histogram_dtype=str(resolve_command_value("build", "dtype", None, fallback_sections=("tile",), default="float64")),
    )
    click.echo("Build complete:")
    click.echo(f"  Tiles: {tile_result.num_files} files, {tile_result.total_rows} rows")
    click.echo(f"  MVTs: {mvt_result.tile_count} tiles across zoom levels {mvt_result.zoom_levels}")
    if pmtiles_path:
        click.echo(f"  PMTiles: {pmtiles_path}")


@main.command()
@click.option("--dir", "data_dir", required=True, help="Root directory containing dataset subdirectories.")
@click.option("--log-level", default=None, help="Logging level.")
def serve(data_dir, log_level):
    """Launch the tile server."""
    _setup_logging(_resolved_log_level("serve", log_level))
    import starlet

    host = str(resolve_command_value("serve", "host", None, default="0.0.0.0"))
    port = int(resolve_command_value("serve", "port", None, default=8765))
    cache_size = int(resolve_command_value("serve", "cache_size", None, default=256))
    app = starlet.create_app(
        data_dir=data_dir,
        cache_size=cache_size,
    )
    click.echo(f"Starting starlet server on {host}:{port}")
    click.echo(f"  Data root: {data_dir}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


@main.command()
@click.option("--dir", "data_dir", required=True, help="Dataset directory to inspect.")
def info(data_dir):
    """Print dataset metadata summary."""
    import starlet
    from pathlib import Path

    try:
        ds = starlet.Dataset(data_dir)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Dataset: {Path(data_dir).name}")
    click.echo(f"  Path:        {ds.path}")
    click.echo(f"  Tiles:       {ds.num_tiles}")
    click.echo(f"  Parquet bbox:{'yes' if ds.parquet_has_bbox else 'no'}")
    click.echo(f"  Parquet CRS: {ds.parquet_crs or '(unknown)'}")
    click.echo(f"  BBox:        {ds.bbox}")
    click.echo(f"  Zoom levels: {ds.zoom_levels or '(no MVTs)'}")
    click.echo(f"  Histograms:  {'yes' if ds.has_histograms else 'no'}")
    if ds.has_histograms:
        click.echo(f"  Hist res:    {ds.histogram_resolution or '(unknown)'}")
    click.echo(f"  MVTs:        {'yes' if ds.has_mvt else 'no'}")
    if ds.has_mvt:
        click.echo(f"  MVT count:   {ds.mvt_tile_count}")
    click.echo(f"  Stats:       {'yes' if ds.has_stats else 'no'}")

    total_bytes = sum(f.stat().st_size for f in Path(data_dir).rglob("*") if f.is_file())
    if total_bytes < 1024 ** 2:
        size_str = f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 ** 3:
        size_str = f"{total_bytes / 1024 ** 2:.1f} MB"
    else:
        size_str = f"{total_bytes / 1024 ** 3:.2f} GB"
    click.echo(f"  Total size:  {size_str}")
