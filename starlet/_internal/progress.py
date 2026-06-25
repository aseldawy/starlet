from __future__ import annotations

from collections.abc import Iterable, Iterator
import logging
from typing import TypeVar


T = TypeVar("T")


def _log_progress(
    logger: logging.Logger,
    label: str,
    completed: int,
    total: int,
    next_percent: int,
) -> int:
    if total <= 0:
        return next_percent

    percent = min(100, int(completed * 100 / total))
    bucket = min(100, (percent // 10) * 10)
    if bucket >= next_percent:
        logger.info("%s progress: %d%% (%d/%d)", label, bucket, completed, total)
        return bucket + 10
    return next_percent


def iter_with_progress(
    iterable: Iterable[T],
    *,
    total: int,
    logger: logging.Logger,
    label: str,
) -> Iterator[T]:
    """Yield items from an iterable and log progress as items are consumed."""
    next_percent = 10
    for completed, item in enumerate(iterable, start=1):
        yield item
        next_percent = _log_progress(
            logger,
            label,
            completed,
            total,
            next_percent,
        )
