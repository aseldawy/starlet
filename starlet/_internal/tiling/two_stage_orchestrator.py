from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import heapq
import logging
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

logger = logging.getLogger(__name__)

_TILE_COL = "_tile_id"


@dataclass(frozen=True)
class _ShardManifest:
    split_index: int
    rows_read: int
    rows_assigned: int
    batches_read: int
    intermediate_by_reducer: Dict[int, str]
    stats: Optional[AttributeStatsCollector] = None


def _executor_class(kind: str):
    normalized = kind.strip().lower()
    if normalized in {"process", "processes", "multiprocessing"}:
        return ProcessPoolExecutor
    if normalized in {"thread", "threads", "threading"}:
        return ThreadPoolExecutor
    raise ValueError(f"executor must be 'process' or 'thread' (got {kind!r})")


def _partition_table_to_values(partition_table) -> List[int]:
    if isinstance(partition_table, pa.Table):
        if partition_table.num_columns != 1:
            raise ValueError("partition table must have exactly one column")
        return [int(value) for value in partition_table.column(0).to_pylist()]
    return [int(value) for value in partition_table.to_pylist()]


def _table_with_partition_column(original_table: pa.Table, partition_table) -> pa.Table:
    pid_values = _partition_table_to_values(partition_table)
    if len(pid_values) != original_table.num_rows:
        raise ValueError(
            "partition assignment row count does not match input table: "
            f"{len(pid_values)} != {original_table.num_rows}"
        )

    if _TILE_COL in original_table.column_names:
        original_table = original_table.drop([_TILE_COL])
    return original_table.append_column(
        _TILE_COL,
        pa.array(pid_values, type=pa.int64()),
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
    merged_path = reducer_dir / f"reducer_{reducer_id:06d}_merged.parquet"

    merged = _merge_sorted_partition_files(intermediate_paths, str(merged_path), config.compression)
    if merged is None:
        return written

    for partition_id, table in _iter_partition_groups(merged):
        written.append(_finalize_one_tile(partition_id, [table], config))

    return written


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
            for future in as_completed(futures):
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
            for future in as_completed(futures):
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
