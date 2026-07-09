"""Shared PMTiles path conventions."""

from __future__ import annotations

from pathlib import Path


def default_pmtiles_path(dataset_root: str | Path) -> Path:
    """Return the canonical PMTiles location for a dataset."""
    root = Path(dataset_root)
    return root / "tiles.pmtiles"


def pmtiles_path_candidates(dataset_root: str | Path) -> tuple[Path, ...]:
    """Return canonical then legacy PMTiles locations for a dataset."""
    root = Path(dataset_root)
    return (
        default_pmtiles_path(root),
        root.with_suffix(".pmtiles"),
        root / f"{root.name}.pmtiles",
    )


def discover_pmtiles_path(dataset_root: str | Path) -> Path | None:
    """Find an existing PMTiles archive for a dataset.

    Preference is given to the canonical ``tiles.pmtiles`` location, but older
    archive names are still recognized for compatibility.
    """
    candidates = pmtiles_path_candidates(dataset_root)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]
