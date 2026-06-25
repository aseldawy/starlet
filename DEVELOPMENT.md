# Developing Starlet

This guide is for working on Starlet itself — running it from a clone, running
the tests, and cutting releases. If you just want to *use* Starlet, see
[README.md](README.md) (`pip install starlet`).

## Setup (from source)

```bash
git clone https://github.com/ucr-bdlab/starlet.git
cd starlet

python -m venv .venv
source .venv/bin/activate

# editable install with the dev/test extras
pip install -e ".[dev]"
```

The editable install exposes the `starlet` console script and points it at your
working tree, so code changes take effect without reinstalling.

```bash
starlet --help          # sanity check
```

## Running the tests

```bash
pytest tests/ -v
```

A coverage run (matches CI):

```bash
pytest tests/ --cov=starlet --cov-report=term
```

Some standalone scripts under the repo root exercise flows that need external
services (e.g. a live LLM) and are **not** part of the `pytest` suite — run them
directly only when needed.

## Project layout

The public API surface is deliberately small. Everything user-facing is
re-exported from `starlet/__init__.py` (`tile`, `generate_mvt`, `build`,
`export_pmtiles`, `create_app`) and `starlet/_types.py` (`TileResult`,
`MVTResult`, `Dataset`). Everything under `starlet/_internal/` is private.

```
starlet/
  __init__.py            # public API (lazy imports keep CLI startup fast)
  _types.py              # frozen result/dataset dataclasses
  _cli.py                # Click CLI (one subcommand per public function)
  _internal/
    tiling/              # partitioning: datasource, RSGrove partitioner,
                         # two-stage + round orchestrators, writer pool
    histogram/           # density-histogram pyramid
    mvt/                 # streaming MVT generation
    pmtiles/             # PMTiles export
    server/              # Flask tile server, on-demand tiler, catalog search
    stats/               # attribute-statistics sketches
```

A *dataset* on disk is a directory; subsystems communicate through it:

```
datasets/<name>/
  parquet_tiles/         # spatially-partitioned GeoParquet (one file per tile)
  histograms/            # density histograms (global.npy, global_prefix.npy)
  stats/attributes.json  # per-attribute statistics
  mvt/<z>/<x>/<y>.mvt    # pre-generated vector tiles (optional)
```

### Conventions

- Source data is EPSG:4326 (lon/lat); MVT/tile math is EPSG:3857 (Web Mercator).
  Reprojection happens in the streaming/rendering stages.
- The internal tile-partition column is `geo_parquet_tile_num`.
- Keep new public surface minimal — add to `__init__.py`/`_types.py` only what
  callers need; treat `_internal/` as private.

## Continuous integration

`.github/workflows/publish.yml` runs on every `v*.*.*` tag push:

1. **test** — `pytest` on Python 3.10 / 3.11 / 3.12 (gates everything below).
2. **build** — builds the sdist/wheel.
3. **benchmark** — runs `.github/scripts/run_benchmark.py` on a 1 GB dataset and
   attaches the results to the GitHub Release. (A benchmark failure does **not**
   block the PyPI publish.)
4. **publish** — uploads to PyPI via trusted publishing (OIDC, `pypi`
   environment).

## Cutting a release

See [RELEASE.md](RELEASE.md) for the full process. In short:

1. Bump `version` in `pyproject.toml`.
2. Commit and push to `master`.
3. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z` — the tag push runs
   the workflow above and publishes to PyPI.

> PyPI rejects re-uploading an existing version, so make sure `pyproject.toml`'s
> version is bumped before tagging.

## Deployment

For standing up a live tile server (including a no-root Apache/CGI recipe), see
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
