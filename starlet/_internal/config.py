"""Process-wide configuration shared across Starlet internals."""
from __future__ import annotations

from pathlib import Path
import tempfile


_temp_dir: Path | None = None


def set_temp_dir(path: str | Path | None) -> Path | None:
    """Set the process-wide parent directory for temporary Starlet files."""
    global _temp_dir
    if path is None:
        _temp_dir = None
        return None
    _temp_dir = Path(path)
    _temp_dir.mkdir(parents=True, exist_ok=True)
    return _temp_dir


def get_temp_dir() -> Path | None:
    """Return the configured temp directory, if one was set."""
    return _temp_dir


def resolve_temp_dir(
    explicit: str | Path | None = None,
    default: str | Path | None = None,
) -> Path:
    """Return the temp parent directory for a step.

    Explicit step-level values win, then the process-wide setting, then the
    caller's default. If no default is provided, use Python's system temp dir.
    """
    if explicit is not None:
        temp_dir = Path(explicit)
    elif _temp_dir is not None:
        temp_dir = _temp_dir
    elif default is not None:
        temp_dir = Path(default)
    else:
        temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir
