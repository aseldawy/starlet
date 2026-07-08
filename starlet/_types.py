"""Public result types and dataset introspection for starlet."""
from __future__ import annotations

import json
import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from starlet._internal.server.tiler.parquet_index import BBOX_COLS
from starlet._internal.tiling.crs import geoparquet_crs


@dataclasses.dataclass(frozen=True)
class TileResult:
    """Result returned by :func:`starlet.tile`."""
    outdir: str
    num_files: int
    total_rows: int
    bbox: Tuple[float, float, float, float]
    histogram_path: str


@dataclasses.dataclass(frozen=True)
class MVTResult:
    """Result returned by :func:`starlet.generate_mvt`."""
    outdir: str
    zoom_levels: List[int]
    tile_count: int
    pmtiles_path: Optional[str] = None


class Dataset:
    """Read-only introspection object for a starlet dataset directory.

    A dataset directory is expected to contain at least ``parquet_tiles/``
    and optionally ``histograms/``, ``mvt/``, and ``stats/``.

    Parameters
    ----------
    path : str
        Path to the dataset root directory.
    """

    def __init__(self, path: str) -> None:
        self._root = Path(path)
        if not self._root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        self._parquet_info: tuple[bool, str | None] | None = None

    @property
    def path(self) -> str:
        return str(self._root)

    @property
    def num_tiles(self) -> int:
        tiles_dir = self._root / "parquet_tiles"
        if not tiles_dir.exists():
            return 0
        return len(list(tiles_dir.glob("*.parquet")))

    @property
    def bbox(self) -> Optional[Tuple[float, float, float, float]]:
        stats_path = self._root / "stats" / "attributes.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                for attr in stats.get("attributes", []):
                    if attr["name"] == "geometry":
                        mbr = attr["stats"].get("mbr")
                        if mbr and len(mbr) == 4:
                            return tuple(mbr)
            except Exception:
                pass
        return None

    @property
    def zoom_levels(self) -> List[int]:
        mvt_dir = self._root / "mvt"
        if not mvt_dir.exists():
            return []
        levels = []
        for child in mvt_dir.iterdir():
            if child.is_dir():
                try:
                    levels.append(int(child.name))
                except ValueError:
                    pass
        return sorted(levels)

    @property
    def mvt_tile_count(self) -> int:
        mvt_dir = self._root / "mvt"
        if not mvt_dir.exists():
            return 0
        return len(list(mvt_dir.rglob("*.mvt")))

    @property
    def has_histograms(self) -> bool:
        return (self._root / "histograms" / "global_prefix.npy").exists()

    @property
    def has_mvt(self) -> bool:
        return (self._root / "mvt").is_dir()

    @property
    def has_stats(self) -> bool:
        return (self._root / "stats" / "attributes.json").exists()

    @property
    def parquet_has_bbox(self) -> bool:
        has_bbox, _ = self._get_parquet_info()
        return has_bbox

    @property
    def parquet_crs(self) -> str | None:
        _, crs = self._get_parquet_info()
        return crs

    @property
    def histogram_resolution(self) -> int | None:
        hist_dir = self._root / "histograms"
        for metadata_path in (hist_dir / "global_prefix.json", hist_dir / "global.json"):
            if metadata_path.exists():
                try:
                    with open(metadata_path) as handle:
                        metadata = json.load(handle)
                    grid_size = metadata.get("grid_size")
                    if grid_size is not None:
                        return int(grid_size)
                except Exception:
                    pass

        for array_path in (hist_dir / "global_prefix.npy", hist_dir / "global.npy"):
            if array_path.exists():
                try:
                    arr = np.load(array_path, allow_pickle=False)
                    if arr.ndim >= 2:
                        return int(arr.shape[0])
                except Exception:
                    pass
        return None

    def _get_parquet_info(self) -> tuple[bool, str | None]:
        if self._parquet_info is not None:
            return self._parquet_info

        tiles_dir = self._root / "parquet_tiles"
        parquet_files = sorted(tiles_dir.glob("*.parquet"))
        if not parquet_files:
            self._parquet_info = (False, None)
            return self._parquet_info

        has_bbox = True
        crs: str | None = None
        for parquet_file in parquet_files:
            schema = pq.ParquetFile(parquet_file).schema_arrow
            names = list(schema.names)
            has_bbox = has_bbox and all(column in names for column in BBOX_COLS)
            if crs is None:
                geom_col = "geometry"
                if geom_col not in names and names:
                    geom_col = names[-1]
                raw_crs = geoparquet_crs(schema, geom_col)
                if raw_crs is not None:
                    crs = str(raw_crs)

        self._parquet_info = (has_bbox, crs)
        return self._parquet_info

    def __repr__(self) -> str:
        return f"Dataset({self._root!s}, tiles={self.num_tiles})"
