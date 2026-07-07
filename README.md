# Starlet

Spatial tiling, Mapbox Vector Tile (MVT) generation, and on-demand tile serving
for large geospatial datasets (GeoParquet / GeoJSON).

The pipeline is: **partition a dataset into spatial tiles → build density
histograms → (optionally) pre-generate MVTs → serve them over HTTP.**

## Install

```bash
pip install starlet
```

Requires Python 3.10+. This installs the `starlet` command-line tool.

> Want to work on Starlet itself (run from a clone, run the tests)? See
> [DEVELOPMENT.md](DEVELOPMENT.md).

## Quick start

Turn a GeoParquet or GeoJSON file into a running tile server in two commands:

```bash
# 1. Build a dataset: partition into tiles + pre-generate vector tiles
starlet build --input data.parquet --outdir datasets/mydata

# 2. Serve it
starlet serve --dir datasets
```

Then open <http://localhost:8765> and pick your dataset to explore it on a map.

## Commands

Everything is available through the `starlet` CLI (`starlet --help`).

## Configuration

Starlet supports a TOML configuration file for settings you typically set once
and reuse, such as temporary directories, parallelism, compression, tile
thresholds, and server defaults.

By default, Starlet looks for:

1. `starlet.toml`
2. `.starlet.toml`
3. `pyproject.toml` under `[tool.starlet]`

You can also pass a file explicitly:

```bash
starlet --config /path/to/starlet.toml build --input data.parquet --outdir datasets/mydata
```

Start from [starlet.toml.example](starlet.toml.example). Full details are in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### `starlet build` — full pipeline (tile + MVT)

```bash
starlet build --input data.parquet --outdir datasets/mydata --zoom 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Path to a GeoParquet or GeoJSON file |
| `--outdir` | (required) | Output dataset directory |
| `--zoom` | config / 7 | Maximum MVT zoom level |
| `--covering-bbox` | on | Write per-row bbox columns for faster on-demand serving |
| `--pmtiles` | config / off | Also export a single `.pmtiles` archive |

Most performance and output-tuning settings for `build` now come from config.

### `starlet tile` — partition a dataset only

```bash
starlet tile --input data.parquet --outdir datasets/mydata
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Path to a GeoParquet or GeoJSON file |
| `--outdir` | (required) | Output dataset directory |
| `--covering-bbox` | on | Write per-row bbox columns for faster on-demand serving |
| `--geom-col` | geometry | Geometry column name (e.g. `wkb_geometry` for OGR exports) |
| `--seed` | 42 | Random seed for partitioning |

Partition sizing, compression, sort order, and histogram settings now come from
config. Shared process parallelism comes from `[global].parallelism`.

### `starlet mvt` — generate vector tiles from a tiled dataset

```bash
starlet mvt --dir datasets/mydata --zoom 7
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | (required) | Dataset directory (contains `parquet_tiles/` and `histograms/`) |
| `--zoom` | config / 7 | Maximum zoom level |
| `--outdir` | `<dir>/mvt/` | MVT output directory |

Threshold, feature reservoir size, extent, buffer, and partition overlap now
come from config. Shared process parallelism comes from `[global].parallelism`.

### `starlet serve` — launch the tile server

```bash
starlet serve --dir datasets
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | (required) | Root directory containing dataset subdirectories |

Host, port, and cache size now come from config.

### `starlet info` — inspect a dataset

```bash
starlet info --dir datasets/mydata
```

## Server API

Once `starlet serve` is running:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive dataset selector |
| `GET` | `/api/datasets` | List all datasets |
| `GET` | `/datasets/<dataset>.json` | Dataset metadata |
| `GET` | `/<dataset>/<z>/<x>/<y>.mvt` | Mapbox Vector Tile |
| `GET`/`POST` | `/datasets/<dataset>/features.<csv\|geojson>` | Download features (optional geometry filter) |
| `GET` | `/api/datasets/<dataset>/stats` | Attribute statistics |

## Notes & tips

- **Geometry column** not named `geometry` (common with OGR/`pyogrio`
  exports)? Pass `--geom-col wkb_geometry`.
- **Serving tiles on the fly** (zooming past the pre-generated levels)? The
  default tiling output includes covering bbox columns so the server can prune
  row groups at read time. Use `--no-covering-bbox` only when optimizing for
  smaller Parquet files and batch generation speed.
- **One-file distribution:** `starlet build --pmtiles` writes a single
  `datasets/mydata.pmtiles` archive alongside the dataset.

## Deploying a server

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for a step-by-step guide to standing
up a production tile server, including a no-root recipe behind an existing
Apache install.

## License

See the repository for license details.
