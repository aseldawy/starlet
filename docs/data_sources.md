# Starlet Public API

## Supported Input Sources

`starlet.tile()` and `starlet.build()` accept these source types:

| Source | Accepted input path | Geometry configuration | Notes |
| --- | --- | --- | --- |
| GeoParquet | `.parquet`, `.geoparquet`, or a directory containing only GeoParquet files | `geom_col="geometry"` by default | Reads Parquet row groups as splits. Geometry-only sampling reads only the geometry column. |
| GeoJSON | `.geojson`, `.geojsonl`, `.json`, `.jsonl`, or a directory containing only GeoJSON files | Geometry comes from GeoJSON feature geometry | FeatureCollection inputs are byte-partitioned; GeoJSONL is streamed by feature records. |
| Shapefile | `.shp`, `.zip` containing shapefile sidecars, or a directory containing `.shp` and/or `.zip` files | Geometry comes from the Shapefile geometry | Uses `pyogrio`. Feature-range splits are used when feature counts are available. Geometry-only sampling reads geometry without attributes. |
| CSV | `.csv` or a directory containing only CSV files | Use either `csv_x_col` + `csv_y_col`, or `csv_wkt_col` | CSV files are read in row chunks. `src_crs` provides the CRS hint. |
| File Geodatabase | `.gdb` directory, or a directory containing `.gdb` directories | Geometry comes from each GDB layer | Uses `pyogrio`. Multiple layers are read as separate splits. |

Directories must contain one supported source type. For example, a directory
containing both CSV and GeoJSON files is rejected to avoid ambiguous ingestion.

## Basic Tiling

```python
import starlet

result = starlet.tile(
    input="data/roads.parquet",
    outdir="datasets/roads",
)
```

Returns a `TileResult` with output path, file count, row count, bounds, and
histogram path.

## GeoParquet

```python
result = starlet.tile(
    input="data/buildings.geoparquet",
    outdir="datasets/buildings",
    geom_col="geometry",
)
```

GeoParquet inputs are split by row group. If a directory is passed, Starlet
recursively reads `.parquet` and `.geoparquet` files in that directory.

## GeoJSON

```python
result = starlet.tile(
    input="data/places.geojson",
    outdir="datasets/places",
    geojson_executor="process",  # or "thread"
)
```

GeoJSON FeatureCollections are partitioned by byte range while preserving
complete feature objects. GeoJSON Lines inputs are streamed by feature record.

## Shapefile

Use a `.shp` file:

```python
result = starlet.tile(
    input="data/roads/roads.shp",
    outdir="datasets/roads",
)
```

Use a zipped Shapefile:

```python
result = starlet.tile(
    input="data/roads.zip",
    outdir="datasets/roads",
)
```

Use a directory containing many Shapefiles or zipped Shapefiles:

```python
result = starlet.tile(
    input="data/shapefiles",
    outdir="datasets/shapefiles",
)
```

Starlet uses `pyogrio` for Shapefile reads. When building the spatial sample,
it requests geometry only so attribute columns are not read unnecessarily.

## CSV

CSV inputs need explicit geometry column configuration.

For x/y coordinate columns:

```python
result = starlet.tile(
    input="data/stops.csv",
    outdir="datasets/stops",
    csv_x_col="longitude",
    csv_y_col="latitude",
    src_crs="EPSG:4326",
)
```

For WKT geometry:

```python
result = starlet.tile(
    input="data/parcels.csv",
    outdir="datasets/parcels",
    csv_wkt_col="wkt",
    src_crs="EPSG:4326",
)
```

Useful CSV options:

```python
result = starlet.tile(
    input="data/points.csv",
    outdir="datasets/points",
    csv_x_col="x",
    csv_y_col="y",
    csv_split_size=64 * 1024 * 1024,
)
```

`csv_split_size` controls the target byte length for each CSV source split. The
default is 32 MiB. CSV splits follow Hadoop-style line ownership: a split starts
at a byte offset, skips to the next newline if needed, and reads every complete
line whose starting byte falls inside `[offset, offset + length)`.

## File Geodatabase

Use a `.gdb` directory directly:

```python
result = starlet.tile(
    input="data/city.gdb",
    outdir="datasets/city",
)
```

Use a parent directory containing one or more `.gdb` directories:

```python
result = starlet.tile(
    input="data/geodatabases",
    outdir="datasets/geodatabases",
)
```

Starlet reads all layers discovered by `pyogrio`. Layers are handled as separate
source splits, and feature-range splits are used when feature counts are
available.

## Build Pipeline

`starlet.build()` accepts the same source inputs and source-specific options as
`starlet.tile()`.

```python
tile_result, mvt_result, pmtiles_path = starlet.build(
    input="data/stops.csv",
    outdir="datasets/stops",
    csv_x_col="longitude",
    csv_y_col="latitude",
    zoom=10,
)
```

## CLI Examples

GeoParquet:

```bash
starlet tile --input data/buildings.parquet --outdir datasets/buildings
```

CSV with x/y columns:

```bash
starlet tile \
  --input data/stops.csv \
  --outdir datasets/stops \
  --csv-x-col longitude \
  --csv-y-col latitude
```

CSV with WKT:

```bash
starlet tile \
  --input data/parcels.csv \
  --outdir datasets/parcels \
  --csv-wkt-col wkt
```

Shapefile:

```bash
starlet tile --input data/roads.zip --outdir datasets/roads
```

File Geodatabase:

```bash
starlet tile --input data/city.gdb --outdir datasets/city
```

The `starlet build` command supports the same input options.
