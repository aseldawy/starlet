from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import errno
import gc
import heapq
import logging
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

from starlet._internal.tiling.datasource import DataSource
from starlet._internal.tiling.writer_pool import (
    SortKey,
    SortMode,
    _WriterPoolConfig,
    _finalize_one_tile,
)
from starlet._internal.stats import write_attribute_stats, AttributeStatsCollector
from starlet._internal.progress import iter_with_progress

logger = logging.getLogger(__name__)

_TILE_COL = "_tile_id"
_MIN_MERGE_FAN_IN = 16
_OPEN_FILE_RESERVE = 32


@dataclass(frozen=True)
class _ShardManifest:
    """Summary returned by one assignment-stage worker.

    The parent process uses these manifests to stitch mapper output files into
    reducer work queues and to aggregate assignment-stage accounting.
    """
    # Index of the source split this mapper processed; useful for tracing files
    # back to the split directory that produced them.
    split_index: int
    # Input accounting reported to the assignment-stage log line.
    rows_read: int
    rows_assigned: int
    batches_read: int
    # Reducer id -> sorted parquet file for this split. Each file contains only
    # rows owned by that reducer and is already sorted by tile id for k-way merge.
    intermediate_by_reducer: Dict[int, str]
    # Optional per-split attribute stats; the parent merges these after all
    # mapper workers finish so stats are collected without re-reading input.
    stats: Optional[AttributeStatsCollector] = None


def _executor_class(kind: str):
    normalized = kind.strip().lower()
    if normalized in {"process", "processes", "multiprocessing"}:
        return ProcessPoolExecutor
    if normalized in {"thread", "threads", "threading"}:
        return ThreadPoolExecutor
    raise ValueError(f"executor must be 'process' or 'thread' (got {kind!r})")


def _table_with_partition_column(original_table: pa.Table, partition_table: pa.Table) -> pa.Table:
    if partition_table.num_columns != 1:
        raise ValueError("partition table must have exactly one column")
    if partition_table.num_rows != original_table.num_rows:
        raise ValueError(
            "partition assignment row count does not match input table: "
            f"{partition_table.num_rows} != {original_table.num_rows}"
        )
    if _TILE_COL in original_table.column_names:
        raise ValueError(f"input table already contains internal column {_TILE_COL!r}")
    return original_table.append_column(
        _TILE_COL,
        partition_table.column(0).cast(pa.int64()),
    )


def _group_table_by_reducer(table: pa.Table, num_reducers: int) -> Dict[int, pa.Table]:
    pid_values = table[_TILE_COL].to_pylist()
    buckets: Dict[int, List[int]] = defaultdict(list)
    for row_index, partition_id in enumerate(pid_values):
        reducer_id = int(partition_id) % num_reducers
        buckets[reducer_id].append(row_index)

    grouped: Dict[int, pa.Table] = {}
    for reducer_id, indices in buckets.items():
        grouped[reducer_id] = table.take(pa.array(indices, type=pa.int64()))
    return grouped


def _sort_by_partition_id(table: pa.Table) -> pa.Table:
    if table.num_rows == 0:
        return table
    return table.sort_by([(_TILE_COL, "ascending")])


def _iter_partition_groups(path: str, batch_size: int = 64_000) -> Iterator[Tuple[int, pa.Table]]:
    parquet_file = pq.ParquetFile(path)
    current_partition: Optional[int] = None
    current_tables: List[pa.Table] = []

    def flush_current() -> Optional[Tuple[int, pa.Table]]:
        nonlocal current_partition, current_tables
        if current_partition is None or not current_tables:
            return None
        table = pa.concat_tables(current_tables, promote_options="default")
        result = (current_partition, table.combine_chunks())
        current_partition = None
        current_tables = []
        return result

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            partition_ids = table[_TILE_COL].to_pylist()
            if not partition_ids:
                continue

            run_start = 0
            while run_start < len(partition_ids):
                partition_id = int(partition_ids[run_start])
                run_end = run_start + 1
                while run_end < len(partition_ids) and int(partition_ids[run_end]) == partition_id:
                    run_end += 1

                if current_partition is not None and partition_id != current_partition:
                    flushed = flush_current()
                    if flushed is not None:
                        yield flushed

                current_partition = partition_id
                current_tables.append(
                    table.slice(run_start, run_end - run_start)
                )
                run_start = run_end

        flushed = flush_current()
        if flushed is not None:
            yield flushed
    finally:
        close = getattr(parquet_file, "close", None)
        if close is not None:
            close()


def _is_too_many_open_files_error(error: BaseException) -> bool:
    if not isinstance(error, OSError):
        return False
    if error.errno == errno.EMFILE:
        return True
    return "Too many open files" in str(error) or "errno 24" in str(error)


def _open_file_soft_limit() -> Optional[int]:
    try:
        import resource

        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, OSError, ValueError):
        return None

    if soft_limit == resource.RLIM_INFINITY:
        return None
    return int(soft_limit)


def _current_open_file_count() -> int:
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError:
            continue
    return 0


def _default_merge_fan_in(num_inputs: int) -> int:
    if num_inputs <= 1:
        return max(0, num_inputs)

    soft_limit = _open_file_soft_limit()
    if soft_limit is None:
        return min(num_inputs, 128)

    available = soft_limit - _current_open_file_count() - _OPEN_FILE_RESERVE
    if available <= 0:
        return min(num_inputs, 2)

    fan_in = int(available * 0.75)
    if available >= _MIN_MERGE_FAN_IN:
        fan_in = max(_MIN_MERGE_FAN_IN, fan_in)
    return max(2, min(num_inputs, fan_in))


def _merge_sorted_partition_files(
    input_paths: Sequence[str],
    output_path: str,
    compression: Optional[str],
) -> Optional[str]:
    iterators = [iter(_iter_partition_groups(path)) for path in input_paths]
    heap: List[Tuple[int, int, pa.Table]] = []
    writer: Optional[pq.ParquetWriter] = None

    for iterator_id, iterator in enumerate(iterators):
        try:
            partition_id, table = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (partition_id, iterator_id, table))

    if not heap:
        return None

    try:
        while heap:
            partition_id, iterator_id, table = heapq.heappop(heap)
            if writer is None:
                writer = pq.ParquetWriter(
                    where=output_path,
                    schema=table.schema,
                    compression=compression,
                )
            writer.write_table(table)

            try:
                next_partition_id, next_table = next(iterators[iterator_id])
            except StopIteration:
                continue
            heapq.heappush(heap, (next_partition_id, iterator_id, next_table))
    finally:
        if writer is not None:
            writer.close()

    return output_path


def _iter_merged_partition_groups(input_paths: Sequence[str]) -> Iterator[Tuple[int, List[pa.Table]]]:
    iterators = [iter(_iter_partition_groups(path)) for path in input_paths]
    heap: List[Tuple[int, int, pa.Table]] = []

    for iterator_id, iterator in enumerate(iterators):
        try:
            partition_id, table = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (partition_id, iterator_id, table))

    current_partition: Optional[int] = None
    current_tables: List[pa.Table] = []

    while heap:
        partition_id, iterator_id, table = heapq.heappop(heap)
        if current_partition is not None and partition_id != current_partition:
            yield current_partition, current_tables
            current_tables = []

        current_partition = partition_id
        current_tables.append(table)

        try:
            next_partition_id, next_table = next(iterators[iterator_id])
        except StopIteration:
            continue
        heapq.heappush(heap, (next_partition_id, iterator_id, next_table))

    if current_partition is not None and current_tables:
        yield current_partition, current_tables


def _merge_sorted_partition_files_to_fan_in(
    input_paths: Sequence[str],
    compression: Optional[str],
    temp_dir: str,
    max_fan_in: Optional[int] = None,
) -> List[str]:
    paths = list(input_paths)
    if not paths:
        return []

    fan_in = max_fan_in if max_fan_in is not None else _default_merge_fan_in(len(paths))
    fan_in = max(2, min(len(paths), fan_in))
    if len(paths) <= fan_in:
        return paths

    merge_dir = Path(temp_dir)
    merge_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Merging %d sorted partition files with fan-in %d",
        len(paths),
        fan_in,
    )

    attempt = 0
    while True:
        level = 0
        level_paths = paths
        attempt_dir = merge_dir / f"attempt_{attempt:02d}_fan_in_{fan_in:04d}"
        try:
            while len(level_paths) > fan_in:
                next_paths: List[str] = []
                level_dir = attempt_dir / f"level_{level:02d}"
                level_dir.mkdir(parents=True, exist_ok=True)

                for chunk_start in range(0, len(level_paths), fan_in):
                    chunk = level_paths[chunk_start:chunk_start + fan_in]
                    chunk_path = level_dir / f"run_{len(next_paths):06d}.parquet"
                    merged = _merge_sorted_partition_files(chunk, str(chunk_path), compression)
                    if merged is not None:
                        next_paths.append(merged)

                logger.info(
                    "Merge level %d reduced %d files to %d files",
                    level,
                    len(level_paths),
                    len(next_paths),
                )
                level_paths = next_paths
                level += 1

            if not level_paths:
                return []
            return level_paths
        except OSError as error:
            if not _is_too_many_open_files_error(error) or fan_in <= 2:
                raise
            next_fan_in = max(2, fan_in // 2)
            logger.warning(
                "Merge fan-in %d exceeded open file limit; retrying with fan-in %d",
                fan_in,
                next_fan_in,
            )
            fan_in = next_fan_in
            attempt += 1
            gc.collect()


def _assignment_stage_worker(
    source: DataSource,
    split,
    split_index: int,
    assigner,
    num_reducers: int,
    temp_dir: str,
    compression: Optional[str],
    collect_stats: bool = False,
    geom_col: str = "geometry",
) -> _ShardManifest:
    split_dir = Path(temp_dir) / f"split_{split_index:06d}"
    split_dir.mkdir(parents=True, exist_ok=True)

    reducer_run_paths: Dict[int, List[str]] = defaultdict(list)
    rows_read = 0
    rows_assigned = 0
    batches_read = 0
    # Partial attribute statistics for this split. Collected here (where every
    # input row passes through) and merged across workers in the parent so the
    # two-stage path produces stats/attributes.json just like the round
    # orchestrator does. Each worker computes a partial MBR; merge combines them.
    stats_collector: Optional[AttributeStatsCollector] = None

    for batch_index, table in enumerate(source.iter_tables(split)):
        batches_read += 1
        rows_read += table.num_rows
        if table.num_rows == 0:
            continue

        if collect_stats:
            if stats_collector is None:
                stats_collector = AttributeStatsCollector(
                    table.schema, geometry_column=geom_col
                )
            stats_collector.consume_table(table)

        partition_table = assigner.partition_by_tile(table)
        assigned_table = _table_with_partition_column(table, partition_table)
        grouped = _group_table_by_reducer(assigned_table, num_reducers)

        for reducer_id, reducer_table in grouped.items():
            if reducer_table.num_rows == 0:
                continue
            sorted_table = _sort_by_partition_id(reducer_table)
            run_path = split_dir / f"reducer_{reducer_id:06d}_run_{batch_index:06d}.parquet"
            pq.write_table(sorted_table, str(run_path), compression=compression)
            reducer_run_paths[reducer_id].append(str(run_path))
            rows_assigned += sorted_table.num_rows

    intermediate_by_reducer: Dict[int, str] = {}
    for reducer_id, run_paths in reducer_run_paths.items():
        merged_path = split_dir / f"mapper_{split_index:06d}_reducer_{reducer_id:06d}.parquet"
        merged = _merge_sorted_partition_files(run_paths, str(merged_path), compression)
        if merged is not None:
            intermediate_by_reducer[reducer_id] = merged

    return _ShardManifest(
        split_index=split_index,
        rows_read=rows_read,
        rows_assigned=rows_assigned,
        batches_read=batches_read,
        intermediate_by_reducer=intermediate_by_reducer,
        stats=stats_collector,
    )


def _reduce_stage_worker(
    reducer_id: int,
    intermediate_paths: Sequence[str],
    config: _WriterPoolConfig,
    temp_dir: str,
) -> List[str]:
    written: List[str] = []
    reducer_dir = Path(temp_dir) / f"reducer_{reducer_id:06d}"
    reducer_dir.mkdir(parents=True, exist_ok=True)
    merge_runs_dir = reducer_dir / "merge_runs"

    fan_in = _default_merge_fan_in(len(intermediate_paths))
    attempt = 0
    while True:
        merge_inputs = _merge_sorted_partition_files_to_fan_in(
            intermediate_paths,
            config.compression,
            str(merge_runs_dir / f"final_attempt_{attempt:02d}"),
            max_fan_in=fan_in,
        )
        if not merge_inputs:
            return written

        try:
            for partition_id, tables in _iter_merged_partition_groups(merge_inputs):
                written.append(_finalize_one_tile(partition_id, tables, config))
            return written
        except OSError as error:
            if (
                not _is_too_many_open_files_error(error)
                or fan_in <= 2
                or written
            ):
                raise
            next_fan_in = max(2, fan_in // 2)
            logger.warning(
                "Reducer %d final merge fan-in %d exceeded open file limit; "
                "retrying with fan-in %d",
                reducer_id,
                fan_in,
                next_fan_in,
            )
            fan_in = next_fan_in
            attempt += 1
            gc.collect()



class TwoStageOrchestrator:
    """Run tiling as one two-stage multiprocessing pass.

    Stage 1 reads each source split independently, assigns records to
    partitions, hash-shuffles rows into reducer files, and writes each mapper
    output sorted by partition ID. Stage 2 runs one task per reducer, performs a
    k-way merge over mapper outputs sorted by partition ID, and writes final
    GeoParquet tiles partition by partition.
    """

    def __init__(
        self,
        *,
        source: DataSource,
        assigner,
        outdir: str,
        geom_col: str = "geometry",
        assignment_workers: Optional[int] = None,
        write_workers: Optional[int] = None,
        num_reducers: Optional[int] = None,
        executor: str = "process",
        compression: Optional[str] = "zstd",
        sort_mode: str = SortMode.ZORDER,
        sort_keys: Optional[Sequence[Union[SortKey, Tuple[str, bool], str]]] = None,
        sfc_bits: int = 16,
        global_extent: Optional[Tuple[float, float, float, float]] = None,
        temp_dir: Optional[str] = None,
        keep_temp: bool = False,
        pq_args: Optional[dict[str, Any]] = None,
        covering_bbox: bool = False,
        collect_stats: bool = True,
    ) -> None:
        self.source = source
        self.assigner = assigner
        self.outdir = outdir
        self.geom_col = geom_col
        self.assignment_workers = assignment_workers
        self.write_workers = write_workers
        self.num_reducers = num_reducers
        self.executor = executor
        self.compression = compression
        self.sort_mode = sort_mode
        self.sort_keys = list(sort_keys or [])
        self.sfc_bits = int(sfc_bits)
        self.global_extent = global_extent
        self.temp_dir = temp_dir
        self.keep_temp = bool(keep_temp)
        self._pq_args = dict(pq_args or {})
        # Opt-in read-time pruning (fast-branch feature): write per-row bbox
        # covering columns + bounded row groups so the on-demand server can skip
        # row groups/rows. Off by default; plumbed through to _finalize_one_tile.
        self.covering_bbox = bool(covering_bbox)
        # Collect attribute statistics during the assignment stage so the
        # two-stage path writes stats/attributes.json (used by the tile server's
        # /stats endpoint and style suggestions), matching the round orchestrator.
        self.collect_stats = bool(collect_stats)

    def run(self) -> None:
        total_start = perf_counter()
        temp_parent = Path(self.temp_dir) if self.temp_dir is not None else Path.cwd() / "tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="starlet_two_stage_", dir=str(temp_parent)))
        logger.info("TwoStageOrchestrator temp dir: %s", temp_root)

        try:
            # Lock the source schema in the parent before spawning mapper workers.
            # GeoJSON batches can expose different property/tag keys, and process
            # workers receive independent source copies. Inferring once here makes
            # every mapper coerce to the same Arrow schema before shuffle writes.
            self.source.schema()
            splits = list(self.source.create_splits())
            if not splits:
                logger.info("TwoStageOrchestrator: source returned no splits")
                return

            intermediate_by_reducer = self._run_assignment_stage(splits, temp_root)
            self._run_reduce_stage(intermediate_by_reducer, temp_root)
            self._write_stats()
            logger.info(
                "TwoStageOrchestrator finished in %.2fs",
                perf_counter() - total_start,
            )
        finally:
            if self.keep_temp:
                logger.info("Keeping TwoStageOrchestrator temp dir: %s", temp_root)
            else:
                shutil.rmtree(temp_root, ignore_errors=True)

    def _write_stats(self) -> None:
        merged = getattr(self, "_merged_stats", None)
        if not self.collect_stats or merged is None:
            return
        try:
            dataset_root = Path(self.outdir).parent
            write_attribute_stats(dataset_root, merged.finalize())
            logger.info(
                "Attribute statistics written to %s/stats/attributes.json",
                dataset_root,
            )
        except Exception:
            logger.warning("Failed to write attribute statistics", exc_info=True)

    def _effective_num_reducers(self, splits: Sequence[Any]) -> int:
        if self.num_reducers is not None:
            if self.num_reducers <= 0:
                raise ValueError("num_reducers must be positive")
            return int(self.num_reducers)
        if self.write_workers is not None:
            if self.write_workers <= 0:
                raise ValueError("write_workers must be positive")
            return int(self.write_workers)
        return max(1, len(splits))

    def _run_assignment_stage(self, splits: Sequence[Any], temp_root: Path) -> Dict[int, List[str]]:
        executor_cls = _executor_class(self.executor)
        max_workers = self.assignment_workers
        num_reducers = self._effective_num_reducers(splits)
        logger.info(
            "TwoStageOrchestrator assignment stage: splits=%d reducers=%d workers=%s executor=%s",
            len(splits),
            num_reducers,
            max_workers or "auto",
            self.executor,
        )

        stage_start = perf_counter()
        self._merged_stats: Optional[AttributeStatsCollector] = None
        manifests: List[_ShardManifest] = []
        with executor_cls(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _assignment_stage_worker,
                    self.source,
                    split,
                    split_index,
                    self.assigner,
                    num_reducers,
                    str(temp_root),
                    self.compression,
                    self.collect_stats,
                    self.geom_col,
                )
                for split_index, split in enumerate(splits)
            ]
            for future in iter_with_progress(
                as_completed(futures),
                total=len(futures),
                logger=logger,
                label="two-stage-orchestrator: map",
            ):
                manifests.append(future.result())

        intermediate_by_reducer: Dict[int, List[str]] = defaultdict(list)
        rows_read = 0
        rows_assigned = 0
        batches_read = 0
        for manifest in manifests:
            rows_read += manifest.rows_read
            rows_assigned += manifest.rows_assigned
            batches_read += manifest.batches_read
            for reducer_id, path in manifest.intermediate_by_reducer.items():
                intermediate_by_reducer[reducer_id].append(path)
            if manifest.stats is not None:
                if self._merged_stats is None:
                    self._merged_stats = manifest.stats
                else:
                    self._merged_stats.merge(manifest.stats)

        logger.info(
            "TwoStageOrchestrator assignment stage finished in %.2fs: "
            "rows_read=%d rows_assigned=%d batches=%d reducers_with_data=%d",
            perf_counter() - stage_start,
            rows_read,
            rows_assigned,
            batches_read,
            len(intermediate_by_reducer),
        )
        return dict(intermediate_by_reducer)

    def _run_reduce_stage(self, intermediate_by_reducer: Dict[int, List[str]], temp_root: Path) -> None:
        if not intermediate_by_reducer:
            logger.info("TwoStageOrchestrator reduce stage: no intermediate files to write")
            return

        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        executor_cls = _executor_class(self.executor)
        max_workers = self.write_workers
        config = _WriterPoolConfig(
            geom_col=self.geom_col,
            sort_mode=self.sort_mode,
            sort_keys=list(self.sort_keys),
            sfc_bits=self.sfc_bits,
            global_extent=self.global_extent,
            compression=self.compression,
            pq_args=dict(self._pq_args),
            outdir=self.outdir,
            covering_bbox=self.covering_bbox,
        )

        logger.info(
            "TwoStageOrchestrator reduce stage: reducers=%d workers=%s executor=%s",
            len(intermediate_by_reducer),
            max_workers or "auto",
            self.executor,
        )
        stage_start = perf_counter()

        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_reduce_stage_worker, reducer_id, paths, config, str(temp_root)): reducer_id
                for reducer_id, paths in intermediate_by_reducer.items()
            }
            for future in iter_with_progress(
                as_completed(futures),
                total=len(futures),
                logger=logger,
                label="two-stage-orchestrator: reduce",
            ):
                reducer_id = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception("TwoStageOrchestrator failed reducing reducer %s", reducer_id)
                    raise

        logger.info(
            "TwoStageOrchestrator reduce stage finished in %.2fs",
            perf_counter() - stage_start,
        )
