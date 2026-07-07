import logging
from pathlib import Path
from time import perf_counter
from typing import Any, MutableMapping

from .tile_cache import TileCache
logger = logging.getLogger(__name__)


TileInfo = MutableMapping[str, Any]


def _update_output(output: TileInfo | None, **values: Any) -> None:
    if output is not None:
        output.update(values)


class VectorTiler:
    """On-demand MVT tile server with a three-tier lookup.

    For each tile request the lookup order is:
      1. **Memory** — LRU cache (``TileCache``, default 256 entries)
      2. **Disk** — pre-generated ``.mvt`` files under ``<dataset>/mvt/z/x/y.mvt``
      3. **Generate** — reads intersecting GeoParquet tiles, clips/transforms
         geometries, and encodes a fresh MVT on the fly

    Generated tiles are promoted into the memory cache but are not persisted
    to disk.
    """

    def __init__(
        self,
        dataset_root: str,
        memory_cache_size: int = 256,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.mvt_dir = self.dataset_root / "mvt"

        self.cache = TileCache(memory_cache_size)

    def tile_path(self, z: int, x: int, y: int) -> Path:
        return self.mvt_dir / str(z) / str(x) / f"{y}.mvt"

    def get_tile(self, z: int, x: int, y: int, output: TileInfo | None = None) -> bytes:
        t0 = perf_counter()
        key = (z, x, y)

        cached = self.cache.get(key)
        if cached is not None:
            logger.debug("[Cache] HIT memory z=%d x=%d y=%d", z, x, y)
            _update_output(
                output,
                source="memory",
                generation="read_from_memory_cache",
                elapsed_ms=(perf_counter() - t0) * 1000,
            )
            return cached

        path = self.tile_path(z, x, y)

        if path.exists():
            t0 = perf_counter()
            data = path.read_bytes()
            elapsed_ms = (perf_counter() - t0) * 1000
            logger.debug("[Cache] HIT disk z=%d x=%d y=%d elapsed=%.1fms", z, x, y, elapsed_ms)
            self.cache.put(key, data)
            _update_output(
                output,
                source="disk",
                generation="read_from_disk",
                path=str(path),
                elapsed_ms=elapsed_ms,
            )
            return data

        logger.info("[Cache] MISS z=%d x=%d y=%d — generating (memory cache only)", z, x, y)
        from starlet._internal.mvt.dataset_generator import generate_single_mvt_tile

        t0 = perf_counter()
        tile_bytes = generate_single_mvt_tile(str(self.dataset_root), (z, x, y))
        _update_output(
            output,
            source="generated",
            generation="generated_on_the_fly",
            elapsed_ms=(perf_counter() - t0) * 1000,
        )

        # On-demand tiles are cached in memory only; we deliberately do NOT
        # persist them to disk. Writing every generated tile would incrementally
        # materialise the full pyramid on disk over time — exactly what lazy
        # serving exists to avoid. Callers who want a durable on-disk pyramid
        # should pre-generate it explicitly (`starlet mvt` / `generate_mvt`,
        # e.g. with threshold=0 to materialise every non-empty tile).
        self.cache.put(key, tile_bytes)
        return tile_bytes
