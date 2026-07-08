import logging
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np
import shapely

from .tiler_bounds import TileBounds
from .parquet_index import ParquetIndex
from .mvt_encoder import MVTEncoder, explode_collections
from .tile_cache import TileCache

logger = logging.getLogger(__name__)

# Display budget for on-the-fly tiles. A 256-512px tile cannot usefully show
# more features than this; the batch pipeline caps tiles the same way
# (assigner.MAX_GEOMS_PER_TILE) via priority sampling. Without a cap, a dense
# z9+ tile can pull tens of thousands of geometries through clip/transform/
# encode and take tens of seconds.
MAX_OTF_FEATURES = 4096

# Per-row bbox covering columns written by the tiling stage; used for read
# pruning, never useful as display attributes.
_INTERNAL_COLS = ("_bbox_xmin", "_bbox_ymin", "_bbox_xmax", "_bbox_ymax")


class VectorTiler:
    """On-demand MVT tile server with a three-tier lookup.

    For each tile request the lookup order is:
      1. **Memory** — LRU cache (``TileCache``, default 256 entries)
      2. **Disk** — pre-generated ``.mvt`` files under ``<dataset>/mvt/z/x/y.mvt``
      3. **Generate** — reads intersecting GeoParquet tiles, clips/transforms
         geometries, and encodes a fresh MVT on the fly

    On-the-fly generation is bounded: when a tile contains more than
    ``MAX_OTF_FEATURES`` candidate geometries, a deterministic top-k sample
    is taken using a per-geometry priority derived from the geometry bytes
    (``crc32(wkb)``). Because the priority is intrinsic to the geometry, the
    same feature wins or loses in every tile it touches — adjacent
    on-the-fly tiles stay seam-consistent, the same guarantee the batch
    pipeline gets from its shared-priority reservoir sampling.
    """

    def __init__(self, dataset_root: str, memory_cache_size: int = 256) -> None:
        self.dataset_root = Path(dataset_root)
        self.parquet_dir = self.dataset_root / "parquet_tiles"
        self.mvt_dir = self.dataset_root / "mvt"
        self.index = ParquetIndex(self.parquet_dir)

        self.cache = TileCache(memory_cache_size)

    def tile_path(self, z: int, x: int, y: int) -> Path:
        return self.mvt_dir / str(z) / str(x) / f"{y}.mvt"

    def generate(self, z: int, x: int, y: int) -> bytes:
        t0 = perf_counter()
        bounds = TileBounds(z, x, y)
        encoder = MVTEncoder(bounds.bbox_3857, bounds.tile_poly_3857)

        try:
            intersecting = self.index.find_intersecting_files(bounds.bbox_4326)
        except Exception as e:
            logger.error("[TileGen] z=%d x=%d y=%d index error: %s", z, x, y, e)
            return encoder.empty_tile()

        if not intersecting:
            logger.debug("[TileGen] z=%d x=%d y=%d no intersecting files", z, x, y)
            return encoder.empty_tile()

        geoms: list = []
        attr_src: list = []  # (col_arrays, attr_cols, row_idx) — dicts built post-sample

        for pf in intersecting:
            try:
                gdf = self.index.load_and_reproject(pf, bounds.bbox_4326)
            except Exception as e:
                logger.error("[TileGen] z=%d x=%d y=%d load failed %s: %s", z, x, y, pf, e)
                continue

            if gdf.empty:
                continue

            attr_cols = [
                c for c in gdf.columns
                if c != "geometry" and c not in _INTERNAL_COLS
            ]
            col_arrays = {c: gdf[c].to_numpy() for c in attr_cols}
            for i, geom in enumerate(gdf.geometry.values):
                if geom is None or geom.is_empty:
                    continue
                geoms.append(geom)
                attr_src.append((col_arrays, attr_cols, i))

        if not geoms:
            return encoder.empty_tile()

        total_candidates = len(geoms)
        if total_candidates > MAX_OTF_FEATURES:
            # Deterministic, geometry-intrinsic priority: crc32 of the WKB.
            # Stable across processes/restarts and across neighbouring tiles.
            wkbs = shapely.to_wkb(np.array(geoms, dtype=object))
            prio = np.fromiter(
                (zlib.crc32(w) for w in wkbs),
                dtype=np.uint32,
                count=total_candidates,
            )
            keep = np.argpartition(prio, total_candidates - MAX_OTF_FEATURES)[
                total_candidates - MAX_OTF_FEATURES:
            ]
            geoms = [geoms[i] for i in keep]
            attr_src = [attr_src[i] for i in keep]
            logger.info(
                "[TileGen] z=%d x=%d y=%d sampled %d of %d candidates (budget=%d)",
                z, x, y, len(geoms), total_candidates, MAX_OTF_FEATURES,
            )

        # Attribute dicts only for the features that survived sampling.
        attrs_list = [
            {c: ca[c][i] for c in cols if ca[c][i] is not None}
            for (ca, cols, i) in attr_src
        ]

        try:
            features = encoder.prepare_features(geoms, attrs_list)
        except Exception as e:
            logger.error("[TileGen] z=%d x=%d y=%d prepare failed: %s", z, x, y, e)
            return encoder.empty_tile()

        if not features:
            return encoder.empty_tile()

        try:
            elapsed_ms = (perf_counter() - t0) * 1000
            logger.info("[TileGen] z=%d x=%d y=%d features=%d elapsed=%.1fms",
                        z, x, y, len(features), elapsed_ms)
            return encoder.encode(features)
        except Exception as e:
            logger.error("[TileGen] z=%d x=%d y=%d encode failed: %s", z, x, y, e)
            return encoder.empty_tile()

    def get_tile(self, z: int, x: int, y: int) -> bytes:
        key = (z, x, y)

        cached = self.cache.get(key)
        if cached is not None:
            logger.debug("[Cache] HIT memory z=%d x=%d y=%d", z, x, y)
            return cached

        path = self.tile_path(z, x, y)

        if path.exists():
            t0 = perf_counter()
            data = path.read_bytes()
            elapsed_ms = (perf_counter() - t0) * 1000
            logger.debug("[Cache] HIT disk z=%d x=%d y=%d elapsed=%.1fms", z, x, y, elapsed_ms)
            self.cache.put(key, data)
            return data

        logger.info("[Cache] MISS z=%d x=%d y=%d — generating (memory cache only)", z, x, y)

        tile_bytes = self.generate(z, x, y)

        # On-demand tiles are cached in memory only; we deliberately do NOT
        # persist them to disk. Writing every generated tile would incrementally
        # materialise the full pyramid on disk over time — exactly what lazy
        # serving exists to avoid. Callers who want a durable on-disk pyramid
        # should pre-generate it explicitly (`starlet mvt` / `generate_mvt`,
        # e.g. with threshold=0 to materialise every non-empty tile).
        self.cache.put(key, tile_bytes)
        return tile_bytes
