# Starlet

Spatial tiling, Mapbox Vector Tile (MVT) generation, and on-demand tile serving
for large geospatial datasets (GeoParquet, GeoJSON, CSV, vector files, and
GeoLife PLT trajectories).

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

Turn a supported geospatial source into a running tile server in two commands:

```bash
# 1. Build a dataset: partition into tiles + pre-generate vector tiles
starlet build --input data.parquet --outdir datasets/mydata

# 2. Serve it
starlet serve --dir datasets --port 8765
```

Then open <http://localhost:8765> and pick your dataset to explore it on a map.

## Commands

Everything is available through the `starlet` CLI (`starlet --help`).

## Configuration

Starlet supports a TOML configuration file for settings you typically set once and reuse.
Make a copy of the file [starlet.toml.example](starlet.toml.example) and named it `starlet.toml`.
Full details are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).
The configurations in that file will be loaded by default when Starlet runs.
You can also pass a file explicitly:

```bash
starlet --config /path/to/starlet.toml build --input data.parquet --outdir datasets/mydata
```

## Starlet CLI tools
### `starlet build` — full pipeline (tile + MVT)

```bash
starlet build --input data.parquet --outdir datasets/mydata --zoom 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Path to a supported source file or directory |
| `--outdir` | (required) | Output dataset directory |
| `--zoom` | 7 | Maximum MVT zoom level |
| `--partition-size` | 512mb (GeoJSON) / 128mb (GeoParquet) | Target partition size, e.g. `256mb`, `1gb` |
| `--threshold` | 0 | Minimum feature count per MVT tile |
| `--pmtiles` | off | Also export a single `.pmtiles` archive |
| `--covering-bbox` | on | Write per-row bbox columns for faster on-demand serving |

### `starlet tile` — partition a dataset only

```bash
starlet tile --input data.parquet --outdir datasets/mydata
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Path to a supported source file or directory |
| `--outdir` | (required) | Output dataset directory |
| `--partition-size` | 512mb (GeoJSON) / 128mb (GeoParquet) | Target partition size; the number of tiles is derived from the input size |
| `--sort` | zorder | Within-tile row order: `zorder`, `hilbert`, `columns`, `none` |
| `--covering-bbox` | on | Write per-row bbox columns for faster on-demand serving |
| `--geom-col` | geometry | Geometry column name (e.g. `wkb_geometry` for OGR exports) |
| `--compression` | zstd | Parquet compression codec |
| `--seed` | 42 | Random seed for partitioning |

### `starlet mvt` — generate vector tiles from a tiled dataset

```bash
starlet mvt --dir datasets/mydata --zoom 7 --threshold 100000  # threshold optional; default 0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | (required) | Dataset directory (contains `parquet_tiles/` and `histograms/`) |
| `--zoom` | 7 | Maximum zoom level |
| `--threshold` | 0 | Minimum feature count per tile |
| `--outdir` | `<dir>/mvt/` | MVT output directory |
| `--pmtiles` | config / off | Combinen all tiles in a single `.pmtiles` archive |

### `starlet serve` — launch the tile server

```bash
starlet serve --dir datasets --port 8765
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | (required) | Root directory containing dataset subdirectories |
| `--host` | 0.0.0.0 | Host to bind |
| `--port` | 8765 | Port to bind |
| `--cache-size` | 256 | In-memory tile cache size |

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
- **GeoLife PLT trajectories:** pass one `.plt` file or a directory. Starlet
  reads every trajectory record as a WGS 84 point and repeats its file ID in
  the `trajectory_id` field.
- **Serving tiles on the fly** (zooming past the pre-generated levels)? The
  default tiling output includes covering bbox columns so the server can prune
  row groups at read time. Use `--no-covering-bbox` only when optimizing for
  smaller Parquet files and batch generation speed.
- **One-file distribution:** `starlet mvt --pmtiles` writes a single
  `datasets/mydata/tiles.pmtiles` archive inside the dataset directory. `starlet build`
  forwards the same setting to its MVT stage.

## Deploying a server

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for a step-by-step guide to standing
up a production tile server, including a no-root recipe behind an existing
Apache install.

## License

See the repository for license details.
