# Starlet

Spatial tiling, MVT generation, and tile serving for geospatial data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI

All commands are available through the `starlet` CLI.

```bash
starlet --help
```

### `starlet tile` — Partition a dataset

```bash
starlet tile --input data.parquet --outdir datasets/mydata --num-tiles 40
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Path to GeoParquet or GeoJSON file |
| `--outdir` | (required) | Output dataset directory |
| `--num-tiles` | 40 | Target number of spatial partitions |
| `--partition-size` | 1gb | Target partition size (e.g. 512mb, 1gb) |
| `--sort` | zorder | Sort order: zorder, hilbert, columns, none |
| `--sample-cap` | 10000 | Reservoir sampling cap for centroids |
| `--compression` | zstd | Parquet compression codec |

### `starlet mvt` — Generate vector tiles

```bash
starlet mvt --dir datasets/mydata --zoom 7 --threshold 100000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | (required) | Dataset directory with parquet_tiles/ and histograms/ |
| `--zoom` | 7 | Maximum zoom level |
| `--threshold` | 0 | Minimum feature count per tile |
| `--outdir` | `<dir>/mvt/` | MVT output directory |

### `starlet build` — Full pipeline (tile + MVT)

```bash
starlet build --input data.parquet --outdir datasets/mydata
```

### `starlet serve` — Launch the tile server

```bash
starlet serve --dir datasets --port 8765
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | (required) | Root directory containing dataset subdirectories |
| `--host` | 0.0.0.0 | Host to bind |
| `--port` | 8765 | Port to bind |
| `--cache-size` | 256 | In-memory tile cache size |

### `starlet info` — Inspect a dataset

```bash
starlet info --dir datasets/mydata
```

## Make Targets

Convenience wrappers around the CLI:

```bash
make tiles INPUT=path/to/data.parquet
make mvt   INPUT=path/to/data.parquet
make build INPUT=path/to/data.parquet   # tiles + mvt
make server                              # starts on port 8765
make clean                               # removes datasets/*
```

## API Endpoints

Once the server is running:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive dataset selector |
| `GET` | `/api/datasets` | List all datasets |
| `GET` | `/datasets.json` | Search datasets by name |
| `GET` | `/datasets/<dataset>.json` | Dataset metadata |
| `GET` | `/datasets/<dataset>.html` | Dataset detail page |
| `GET` | `/<dataset>/<z>/<x>/<y>.mvt` | Mapbox Vector Tile |
| `GET` | `/datasets/<dataset>/features.<fmt>` | Download features (csv/geojson) |
| `POST` | `/datasets/<dataset>/features.<fmt>` | Download with geometry filter |
| `GET` | `/datasets/<dataset>/features/sample.json` | Sample attributes |
| `GET` | `/datasets/<dataset>/features/sample.geojson` | Sample record with geometry |
| `GET` | `/api/datasets/<dataset>/stats` | Attribute statistics |

## LLM Styling (Optional)

LLM-powered map styling suggestions have been extracted to a separate package for optional installation:

**[starlet-styles](https://github.com/rohanbennur43/starlet-styles)**

Install it separately if you need AI-generated styling rules based on dataset attributes.

## Example

```bash
# Full pipeline
starlet build --input ../data/TIGER2018_COUNTY.parquet --outdir datasets/TIGER2018_COUNTY

# Or via Make
make build INPUT=../data/TIGER2018_COUNTY.parquet

# Start the server
make server
```

Then open http://localhost:8765 and select a dataset to visualize.

## Prerequisites

- Python 3.10+
- `make` (optional, for convenience targets)

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Creating a Release

See [RELEASE.md](RELEASE.md) for the complete release process, including:
- Where to update version numbers
- How to create and push release tags
- Automated CI/CD workflow details
- Troubleshooting guide
