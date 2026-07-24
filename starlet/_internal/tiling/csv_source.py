from __future__ import annotations

import bz2
from dataclasses import dataclass
import io
import logging
import os
import re
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow as pa
from shapely import from_wkt, points, to_wkb

from starlet._internal.tiling.datasource import (
    DataSource,
    _CSV_SUFFIXES,
    _attach_geoparquet_metadata,
    _source_files,
    _unify_tabular_schemas,
)

logger = logging.getLogger(__name__)
_BZ2_BLOCK_MAGIC = bytes.fromhex("314159265359")
_BZ2_STREAM_HEADER_LEN = 4
_HEADERLESS_COLUMN_PREFIX = "column_"


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
        x_col: str | int | None = None,
        y_col: str | int | None = None,
        wkt_col: str | int | None = None,
        split_size: int = 32 * 1024 * 1024,
        batch_rows: int | None = None,
        src_crs: str = "EPSG:4326",
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ) -> None:
        if bool(wkt_col) == bool(x_col and y_col):
            raise ValueError("CSV input requires either wkt_col or both x_col and y_col")
        if (x_col is None) != (y_col is None):
            raise ValueError("CSV x/y geometry requires both x_col and y_col")

        self.path = str(path)
        self.x_col = x_col
        self.y_col = y_col
        self.wkt_col = wkt_col
        self.split_size = int(batch_rows if batch_rows is not None else split_size)
        self.src_crs = src_crs
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self.has_header = _csv_uses_header(x_col=x_col, y_col=y_col, wkt_col=wkt_col)
        self._files = _source_files(self.path, _CSV_SUFFIXES)
        if not self._files:
            raise ValueError(f"No CSV files found in {self.path}")
        self._schema: pa.Schema | None = None

    def schema(self) -> pa.Schema:
        if self._schema is None:
            schemas = [
                table.schema
                for table in self.iter_tables_for_schema_inference()
            ]
            self._schema = (
                _unify_tabular_schemas(schemas)
                if schemas
                else _attach_geoparquet_metadata(
                    pa.schema([(self.geom_col, pa.binary())]),
                    self.src_crs,
                )
            )
        return self._schema

    def set_schema(self, schema: pa.Schema) -> None:
        if self.geom_col not in schema.names:
            raise ValueError(f"CSV schema must contain geometry column {self.geom_col!r}")
        self._schema = schema

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
        schema = self.schema()
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            df = self._read_split(source_split)
            if df.empty:
                continue
            yield self._dataframe_to_table(df, schema=schema)

    def iter_tables_for_schema_inference(
        self,
        split: Optional[CSVSplit] = None,
    ) -> Iterable[pa.Table]:
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            df = self._read_split(source_split)
            if df.empty:
                continue
            yield self._dataframe_to_table(df)

    def _read_split(self, split: CSVSplit) -> pd.DataFrame:
        usecols = self._geometry_columns() if self.geometry_only else None
        data = _read_csv_split_bytes(split, has_header=self.has_header)
        if data is None:
            return pd.DataFrame()
        read_kwargs = {
            "usecols": usecols,
            "dtype": str,
        }
        if not self.has_header:
            read_kwargs["header"] = None
        if split.path.lower().endswith(".txt"):
            read_kwargs["sep"] = r"\s+"
        df = pd.read_csv(io.BytesIO(data), **read_kwargs)
        if not self.has_header:
            df = df.rename(columns=lambda name: _headerless_column_name(int(name)))
        return df

    def _geometry_columns(self) -> List[str | int]:
        if self.wkt_col:
            return [self.wkt_col]
        return [self.x_col, self.y_col]  # type: ignore[list-item]

    def _dataframe_to_table(
        self,
        df: pd.DataFrame,
        schema: pa.Schema | None = None,
    ) -> pa.Table:
        wkt_column = _csv_column_name(self.wkt_col)
        x_column = _csv_column_name(self.x_col)
        y_column = _csv_column_name(self.y_col)
        if self.wkt_col:
            if wkt_column not in df.columns:
                raise ValueError(f"CSV missing WKT column {self.wkt_col!r}")
            geoms = from_wkt(df[wkt_column].astype("string").to_numpy())
        else:
            if x_column not in df.columns or y_column not in df.columns:
                raise ValueError(f"CSV missing x/y columns {self.x_col!r}, {self.y_col!r}")
            geoms = points(
                pd.to_numeric(df[x_column], errors="coerce").to_numpy(),
                pd.to_numeric(df[y_column], errors="coerce").to_numpy(),
            )

        geometry_col = pa.array(to_wkb(geoms, hex=False).tolist(), type=pa.binary())
        props_df = pd.DataFrame(index=df.index) if self.geometry_only else df.copy()
        properties_schema = (
            pa.schema([field for field in schema if field.name != self.geom_col])
            if schema is not None
            else None
        )
        props_table = _csv_dataframe_to_arrow_table(
            props_df,
            schema=properties_schema,
        )
        table = (
            pa.table([geometry_col], names=[self.geom_col])
            if props_table.num_columns == 0
            else props_table.append_column(self.geom_col, geometry_col)
        )
        schema_with_geo = _attach_geoparquet_metadata(table.schema, self.src_crs)
        table = table.replace_schema_metadata(schema_with_geo.metadata)
        if schema is not None:
            table = table.cast(schema)
        return table.combine_chunks()


_CSV_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_CSV_BOOLEAN_VALUES = {"true": True, "false": False}


def _csv_dataframe_to_arrow_table(
    df: pd.DataFrame,
    schema: pa.Schema | None = None,
) -> pa.Table:
    fields = list(schema) if schema is not None else [
        pa.field(str(name), _infer_csv_column_type(df[name].tolist()))
        for name in df.columns
    ]
    arrays = []
    for field in fields:
        values = df[field.name].tolist() if field.name in df.columns else [None] * len(df.index)
        arrays.append(pa.array(
            [_parse_csv_value(value, field.type) for value in values],
            type=field.type,
        ))
    return pa.table(arrays, schema=pa.schema(fields))


def _infer_csv_column_type(values: List[object]) -> pa.DataType:
    strings = [str(value) for value in values if not pd.isna(value)]
    if not strings:
        return pa.null()
    if all(value.lower() in _CSV_BOOLEAN_VALUES for value in strings):
        return pa.bool_()
    if all(_CSV_INTEGER_PATTERN.fullmatch(value) for value in strings):
        if all(_is_csv_integer(value) for value in strings):
            try:
                pa.array([int(value) for value in strings], type=pa.int64())
                return pa.int64()
            except (OverflowError, pa.ArrowInvalid):
                pass
        return pa.string()
    try:
        for value in strings:
            float(value)
        return pa.float64()
    except ValueError:
        return pa.string()


def _is_csv_integer(value: str) -> bool:
    if not _CSV_INTEGER_PATTERN.fullmatch(value):
        return False
    digits = value.lstrip("+-")
    return len(digits) == 1 or not digits.startswith("0")


def _parse_csv_value(value: object, arrow_type: pa.DataType):
    if pd.isna(value):
        return None
    text = str(value)
    if pa.types.is_boolean(arrow_type):
        return _CSV_BOOLEAN_VALUES[text.lower()]
    if pa.types.is_integer(arrow_type):
        return int(text)
    if pa.types.is_floating(arrow_type):
        return float(text)
    return text


def _csv_uses_header(
    *,
    x_col: str | int | None,
    y_col: str | int | None,
    wkt_col: str | int | None,
) -> bool:
    refs = [ref for ref in (x_col, y_col, wkt_col) if ref is not None]
    if not refs:
        return True
    has_named = any(isinstance(ref, str) for ref in refs)
    has_indexed = any(isinstance(ref, int) for ref in refs)
    if has_named and has_indexed:
        raise ValueError("CSV geometry columns must be specified either all by name or all by index")
    return has_named


def _csv_column_name(column: str | int | None) -> str | None:
    if column is None:
        return None
    if isinstance(column, int):
        return _headerless_column_name(column)
    return column


def _headerless_column_name(index: int) -> str:
    return f"{_HEADERLESS_COLUMN_PREFIX}{index}"


def _read_csv_split_bytes(split: CSVSplit, *, has_header: bool) -> bytes | None:
    if split.path.lower().endswith(".bz2"):
        return _read_bz2_csv_split_bytes(split, has_header=has_header)

    split_start = split.offset
    split_end = split.offset + split.length
    with open(split.path, "rb") as f:
        if has_header:
            header = f.readline()
            data_start = f.tell()
        else:
            header = b""
            data_start = 0
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


def _read_bz2_csv_split_bytes(split: CSVSplit, *, has_header: bool) -> bytes | None:
    block_starts, file_size = _bz2_block_starts(split.path)
    split_end = min(split.offset + split.length, file_size)
    first_owned = next((start for start in block_starts if start >= split.offset), None)
    if first_owned is None or first_owned >= split_end:
        return None

    stop_before = next((start for start in block_starts if start >= split_end), file_size)
    payload, previous_output_ended_with_newline, owned_output_len = _decompress_bz2_owned_blocks(
        split.path,
        block_starts=block_starts,
        first_owned=first_owned,
        stop_before=stop_before,
    )
    if not payload:
        return None

    if split.offset > 0 and not previous_output_ended_with_newline:
        first_newline = payload.find(b"\n")
        if first_newline == -1:
            return None
        payload = payload[first_newline + 1 :]
        owned_output_len = max(0, owned_output_len - (first_newline + 1))

    if stop_before < file_size:
        if owned_output_len <= 0:
            return None
        if payload[:owned_output_len].endswith(b"\n"):
            payload = payload[:owned_output_len]
        else:
            trailing_newline = payload.find(b"\n", owned_output_len)
            if trailing_newline == -1:
                return None
            payload = payload[: trailing_newline + 1]

    if not payload:
        return None
    if not has_header or split.offset == 0:
        return payload
    return _read_bz2_header_line(split.path) + payload


def _read_bz2_header_line(path: str) -> bytes:
    with bz2.open(path, "rb") as stream:
        return stream.readline()


def _bz2_block_starts(path: str) -> tuple[list[int], int]:
    with open(path, "rb") as stream:
        data = stream.read()
    starts = [match.start() for match in re.finditer(re.escape(_BZ2_BLOCK_MAGIC), data)]
    if not starts:
        starts = [_BZ2_STREAM_HEADER_LEN]
    return starts, len(data)


def _decompress_bz2_owned_blocks(
    path: str,
    *,
    block_starts: list[int],
    first_owned: int,
    stop_before: int,
) -> tuple[bytes, bool, int]:
    with open(path, "rb") as stream:
        header = stream.read(_BZ2_STREAM_HEADER_LEN)
        decompressor = bz2.BZ2Decompressor()
        header_output = decompressor.decompress(header)

        owned_output = bytearray()
        last_discarded_byte = header_output[-1:] if header_output else b"\n"
        boundaries = [start for start in block_starts if start >= _BZ2_STREAM_HEADER_LEN]
        if not boundaries or boundaries[0] != _BZ2_STREAM_HEADER_LEN:
            boundaries.insert(0, _BZ2_STREAM_HEADER_LEN)
        lookahead_stop = next((start for start in boundaries if start > stop_before), None)
        boundaries.append(lookahead_stop if lookahead_stop is not None else os.path.getsize(path))
        owned_output_len = 0

        for segment_start, segment_end in zip(boundaries, boundaries[1:]):
            if segment_start > stop_before:
                break
            if segment_end <= segment_start:
                continue
            stream.seek(segment_start)
            chunk = stream.read(segment_end - segment_start)
            decoded = decompressor.decompress(chunk)
            if segment_start >= first_owned:
                owned_output.extend(decoded)
                if segment_start < stop_before:
                    owned_output_len = len(owned_output)
            elif decoded:
                last_discarded_byte = decoded[-1:]

    return bytes(owned_output), last_discarded_byte == b"\n", owned_output_len
