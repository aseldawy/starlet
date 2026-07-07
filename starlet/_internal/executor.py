from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging


def create_process_executor(
    max_workers: int | None = None,
    *,
    logger: logging.Logger | None = None,
    context: str = "parallel work",
):
    """Prefer a process pool and fall back only when the runtime forbids it."""
    try:
        return ProcessPoolExecutor(max_workers=max_workers)
    except (NotImplementedError, PermissionError, OSError) as exc:
        active_logger = logger or logging.getLogger(__name__)
        active_logger.warning(
            "ProcessPoolExecutor unavailable for %s; falling back to ThreadPoolExecutor: %s",
            context,
            exc,
        )
        return ThreadPoolExecutor(max_workers=max_workers)
