# On-Demand Tile Serving: Read-Time Spatial Pruning

This note documents a design change to Starlet's **on-the-fly tile generation**
path (the server's third tier, used when a requested tile is neither in the
in-memory cache nor pre-generated on disk). It is intended as source material
for the evaluation/serving section of the paper.

## Problem

For a tile request `(z, x, y)` outside the pre-generated pyramid, the server
generated the tile directly from the partitioned GeoParquet. The original
procedure was:

1. **Partition selection** — parse each partition's filename bounding box and
   keep those intersecting the tile (`ParquetIndex.find_intersecting_files`).
2. For each selected partition, **read the entire file** (`geopandas.read_parquet`),
   **reproject every geometry** EPSG:4326 → EPSG:3857, run `make_valid` and
   **clip every geometry** to the tile, then encode the survivors.

The cost of step 2 is proportional to the **size of the partition**, not the
**area of the tile**. Partitions produced by RSGrove are large (≈ 100 MB,
~10⁵ geometries) and their minimum bounding rectangles overlap heavily (one
OSM-parks partition spans ≈ 120° × 133°), so a single deep-zoom tile whose
output is a few kilobytes triggered a full-partition reproject + validity
repair. Measured on a 16-core server (OSM-parks, 9.96 M polygons), one
on-the-fly tile took **17–39 s**, of which ≈ 6 s was reprojection and ≈ 7 s was
`make_valid` — almost all of it spent on geometries that the clip discarded.

## Design change

The fix introduces **read-time spatial pruning** so that on-the-fly generation
reads and processes work proportional to the geometries actually inside the
tile. It has two layers.

### 1. Geometry-level pre-filter (query side)

Before reprojection/validation/clipping, geometries whose envelope does not
overlap the tile bounding box are dropped. Because the Web-Mercator projection
is monotone per axis, filtering on the WGS-84 tile bbox is exact — it never
removes a geometry that the clip would have kept. The reprojection and
`make_valid` calls then run on a handful of candidate geometries instead of the
whole partition. A small LRU cache of decoded partitions lets panning/zooming
within one area reuse a partition instead of re-reading it.

### 2. Covering bbox columns + spatially-coherent row groups (storage side)

The pre-filter above still had to *read and decode* the whole partition before
it could discard rows, because Parquet decodes a column chunk in full. The
storage layout is therefore changed at tiling time so the reader can skip data
it never needs:

- **Per-row covering columns.** Each tile's Parquet file gains four
  `double` columns — `_bbox_xmin, _bbox_ymin, _bbox_xmax, _bbox_ymax` — holding
  every geometry's envelope. They are computed for free during the sort step,
  which already decodes WKB to derive the per-tile bbox and Z-order keys.
- **Bounded, spatially-coherent row groups.** Rows within a tile are already
  Z-order sorted; the writer now caps the row-group size
  (`DEFAULT_ROW_GROUP_SIZE = 16384`) so a partition is split into several row
  groups, each covering a compact sub-region. Each row group's covering-column
  min/max **statistics** therefore form a tight spatial filter.

At read time the server issues a pyarrow predicate-pushdown query on the
covering columns:

```
_bbox_xmax >= tile.minx AND _bbox_xmin <= tile.maxx AND
_bbox_ymax >= tile.miny AND _bbox_ymin <= tile.maxy
```

pyarrow uses the per-row-group statistics to **skip entire row groups** that do
not intersect the tile, decodes only the surviving row groups, and returns only
the matching rows. The remaining reproject/clip then operates on that small
set. This yields a three-level pruning hierarchy:

```
filename bbox        → which partitions can intersect the tile
row-group statistics → which row groups within a partition to decode   (NEW)
covering-column rows  → which rows to return                            (NEW)
exact geometry clip  → final tile contents
```

### Compatibility

The change is backward compatible. Tiles written before the change have no
covering columns; the reader detects their absence and falls back to the
cached full-read + in-memory pre-filter, which is still correct. The covering
columns are internal: they are excluded from both on-the-fly tile attributes
and batch-MVT properties, so tile *contents are byte-for-byte identical* to the
previous implementation (verified across point, line, and polygon datasets).

## Effect

- **Correctness:** generated tiles are byte-identical to the original path
  (0 mismatches over point/line/polygon datasets, including legacy tiles).
- **Compute:** the geometry-level pre-filter alone removed the reproject +
  `make_valid` blow-up — per-tile **7.3× median** speed-up on the server
  (13–35 s → 1.9–6.7 s), with cached / pre-generated serving unchanged at ~2 ms.
- **I/O:** the covering columns + row groups let the reader skip whole row
  groups (e.g. 9 of 10 in a controlled test), removing the remaining
  read-and-decode-the-whole-partition floor that the pre-filter could not.

*(Server-scale scalability numbers for the rebuilt datasets are reported in the
campaign results that accompany this change.)*

## Where it lives

| Concern | File |
|---|---|
| Covering columns + row-group sizing (write) | `starlet/_internal/tiling/writer_pool.py` |
| Predicate pushdown + legacy fallback (read) | `starlet/_internal/server/tiler/parquet_index.py` |
| Exclude covering columns from batch MVT | `starlet/_internal/mvt/streamer.py` |
| Tile-bbox pre-filter in the generate loop | `starlet/_internal/server/tiler/tiler.py` |
| Correctness + speed regression harness | `bench/verify_tile_fix.py` |
