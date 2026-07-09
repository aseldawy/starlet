from __future__ import annotations

from dataclasses import dataclass
import io
import logging
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow as pa
from shapely import from_wkt, points, to_wkb

from starlet._internal.tiling.datasource import (
    DataSource,
    _CSV_SUFFIXES,
    _attach_geoparquet_metadata,
    _normalize_decimal_columns,
    _source_files,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CSVSplit:
    path: str
    offset: int
    length: int


class CSVSource(DataSource):
    def __init__(
        self,
        path: str,
        *,
        x_col: str | None = None,
        y_col: str | None = None,
        wkt_col: str | None = None,
        split_size: int = 32 * 1024 * 1024,
        batch_rows: int | None = None,
        src_crs: str = "EPSG:4326",
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ) -> None:
        if bool(wkt_col) == bool(x_col and y_col):
            raise ValueError("CSV input requires either wkt_col or both x_col and y_col")

        self.path = str(path)
        self.x_col = x_col
        self.y_col = y_col
        self.wkt_col = wkt_col
        self.split_size = int(batch_rows if batch_rows is not None else split_size)
        self.src_crs = src_crs
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self._files = _source_files(self.path, _CSV_SUFFIXES)
        if not self._files:
            raise ValueError(f"No CSV files found in {self.path}")
        self._schema: pa.Schema | None = None

    def schema(self) -> pa.Schema:
        if self._schema is None:
            first = next(self.iter_tables(), None)
            self._schema = first.schema if first is not None else _attach_geoparquet_metadata(
                pa.schema([(self.geom_col, pa.binary())]),
                self.src_crs,
            )
        return self._schema

    def input_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._files)

    def create_splits(self, num_splits: Optional[int] = None) -> List[CSVSplit]:
        if num_splits is None:
            target_split_size = max(1, self.split_size)
        else:
            total_bytes = max(1, self.input_size_bytes())
            target_split_size = max(1, (total_bytes + max(1, int(num_splits)) - 1) // max(1, int(num_splits)))

        splits: List[CSVSplit] = []
        for path in self._files:
            file_size = path.stat().st_size
            if file_size <= 0:
                continue
            for offset in range(0, file_size, target_split_size):
                splits.append(
                    CSVSplit(
                        path=str(path),
                        offset=offset,
                        length=min(target_split_size, file_size - offset),
                    )
                )
        return splits

    def iter_tables(self, split: Optional[CSVSplit] = None) -> Iterable[pa.Table]:
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            df = self._read_split(source_split)
            if df.empty:
                continue
            yield self._dataframe_to_table(df)

    def _read_split(self, split: CSVSplit) -> pd.DataFrame:
        usecols = self._geometry_columns() if self.geometry_only else None
        data = _read_csv_split_bytes(split)
        if data is None:
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(data), usecols=usecols)

    def _geometry_columns(self) -> List[str]:
        if self.wkt_col:
            return [self.wkt_col]
        return [self.x_col, self.y_col]  # type: ignore[list-item]

    def _dataframe_to_table(self, df: pd.DataFrame) -> pa.Table:
        if self.wkt_col:
            if self.wkt_col not in df.columns:
                raise ValueError(f"CSV missing WKT column {self.wkt_col!r}")
            geoms = from_wkt(df[self.wkt_col].astype("string").to_numpy())
        else:
            if self.x_col not in df.columns or self.y_col not in df.columns:
                raise ValueError(f"CSV missing x/y columns {self.x_col!r}, {self.y_col!r}")
            geoms = points(
                pd.to_numeric(df[self.x_col], errors="coerce").to_numpy(),
                pd.to_numeric(df[self.y_col], errors="coerce").to_numpy(),
            )

        geometry_col = pa.array(to_wkb(geoms, hex=False).tolist(), type=pa.binary())
        props_df = pd.DataFrame(index=df.index) if self.geometry_only else df.copy()
        props_df = _normalize_decimal_columns(props_df)
        props_table = pa.Table.from_pandas(props_df, preserve_index=False)
        table = (
            pa.table([geometry_col], names=[self.geom_col])
            if props_table.num_columns == 0
            else props_table.append_column(self.geom_col, geometry_col)
        )
        schema_with_geo = _attach_geoparquet_metadata(table.schema, self.src_crs)
        return table.replace_schema_metadata(schema_with_geo.metadata).combine_chunks()


def _read_csv_split_bytes(split: CSVSplit) -> bytes | None:
    split_start = split.offset
    split_end = split.offset + split.length
    with open(split.path, "rb") as f:
        header = f.readline()
        data_start = f.tell()
        start = max(split_start, data_start)

        if start > data_start:
            f.seek(start - 1)
            previous = f.read(1)
            if previous != b"\n":
                f.readline()
                start = f.tell()
            else:
                f.seek(start)
        else:
            f.seek(start)

        rows = bytearray()
        while True:
            line_start = f.tell()
            if line_start >= split_end:
                break
            line = f.readline()
            if not line:
                break
            rows.extend(line)

    if not rows:
        return None
    return header + bytes(rows)
