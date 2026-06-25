"""Shared helpers for process-pool parallelism."""
from __future__ import annotations

import os
import sys
import multiprocessing


def pool_context():
    """Return a multiprocessing context that keeps pool memory bounded on macOS.

    macOS defaults to the ``spawn`` start method: every pool worker is a brand-new
    interpreter that re-imports the whole library stack (pyarrow / numpy / shapely,
    ~300 MB each) with no sharing, so even a handful of workers add several GB of
    RSS. ``fork`` shares the parent's pages copy-on-write (as on Linux), keeping
    pool overhead near zero. Returns ``None`` on other platforms so callers fall
    back to the default context.
    """
    # Optional override: STARLET_MP_START=fork|spawn|default
    override = os.environ.get("STARLET_MP_START", "").strip().lower()
    if override in ("fork", "spawn"):
        try:
            return multiprocessing.get_context(override)
        except ValueError:
            return None
    if override == "default":
        return None
    if sys.platform == "darwin":
        try:
            return multiprocessing.get_context("fork")
        except ValueError:
            return None
    return None
