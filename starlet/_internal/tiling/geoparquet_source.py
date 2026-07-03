from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from starlet._internal.tiling.datasource import DataSource, _GEOPARQUET_SUFFIXES, _source_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoParquetSplit:
    """Row groups to read from one GeoParquet file."""

    path: str
    row_groups: Tuple[int, ...]


class GeoParquetSource(DataSource):
    def __init__(
        self,
        path: str,
        *,
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ):
        self.path = str(path)
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self._files = _source_files(self.path, _GEOPARQUET_SUFFIXES)
        if not self._files:
            raise ValueError(f"No GeoParquet files found in {self.path}")

        pf = pq.ParquetFile(str(self._files[0]))
        self._schema = pf.schema_arrow
        self._row_group_counts = {
            str(file_path): pq.ParquetFile(str(file_path)).num_row_groups
            for file_path in self._files
        }
        self._num_row_groups = sum(self._row_group_counts.values())
        if self.geometry_only and self.geom_col not in self._schema.names:
            raise ValueError(
                f"Geometry column {self.geom_col!r} was not found in {self.path}"
            )
        logger.info(
            "GeoParquetSource opened %s with %d files and %d row groups (geometry_only=%s)",
            path,
            len(self._files),
            self._num_row_groups,
            self.geometry_only,
        )

    def schema(self) -> pa.Schema:
        logger.debug("GeoParquet source schema metadata: %s", self._schema.metadata)
        return self._schema

    def input_size_bytes(self) -> int:
        return sum(file_path.stat().st_size for file_path in self._files)

    def create_splits(self, num_splits: Optional[int] = None) -> List[GeoParquetSplit]:
        row_groups = [
            (str(file_path), row_group)
            for file_path in self._files
            for row_group in range(self._row_group_counts[str(file_path)])
        ]
        if num_splits is None:
            return [
                GeoParquetSplit(path=path, row_groups=(row_group,))
                for path, row_group in row_groups
            ]

        split_count = max(1, min(int(num_splits), max(1, len(row_groups))))
        chunk_size = max(1, (len(row_groups) + split_count - 1) // split_count)
        splits: List[GeoParquetSplit] = []
        for file_path in self._files:
            groups = list(range(self._row_group_counts[str(file_path)]))
            for index in range(0, len(groups), chunk_size):
                splits.append(
                    GeoParquetSplit(
                        path=str(file_path),
                        row_groups=tuple(groups[index:index + chunk_size]),
                    )
                )
        return splits

    def iter_tables(
        self,
        split: Optional[GeoParquetSplit] = None,
        columns: Optional[List[str]] = None,
    ) -> Iterable[pa.Table]:
        selected_columns = [self.geom_col] if self.geometry_only else columns
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            pf = pq.ParquetFile(source_split.path)
            num_row_groups = self._row_group_counts.get(source_split.path, pf.num_row_groups)
            for row_group in source_split.row_groups:
                logger.debug(
                    "Reading row group %d/%d from %s",
                    row_group,
                    num_row_groups,
                    source_split.path,
                )
                yield pf.read_row_group(row_group, columns=selected_columns)


