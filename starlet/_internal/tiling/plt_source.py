from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
from shapely import points, to_wkb

from starlet._internal.tiling.datasource import (
    DataSource,
    _PLT_SUFFIXES,
    _attach_geoparquet_metadata,
    _source_files,
)


@dataclass(frozen=True)
class PLTSplit:
    """One GeoLife trajectory file, used as a natural source split."""

    path: str


class PLTSource(DataSource):
    """Read GeoLife ``.plt`` trajectories as points.

    GeoLife files have six header lines followed by seven comma-separated
    values per point: latitude, longitude, a reserved value, altitude, a date
    serial, date, and time. Each output row also carries the source filename
    so points from one file can be grouped back together.
    """

    def __init__(
        self,
        path: str,
        *,
        batch_rows: int = 65_536,
        geometry_only: bool = False,
        geom_col: str = "geometry",
    ) -> None:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {path}")
        if source_path.is_file() and source_path.suffix.lower() not in _PLT_SUFFIXES:
            raise ValueError(f"PLT source file must have a .plt extension: {path}")
        if batch_rows <= 0:
            raise ValueError("batch_rows must be greater than zero")

        self.path = str(source_path)
        self.batch_rows = int(batch_rows)
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self._files = _source_files(self.path, _PLT_SUFFIXES)
        if not self._files:
            raise ValueError(f"No PLT files found in {self.path}")
        self._schema = self._default_schema()

    def _default_schema(self) -> pa.Schema:
        fields = [] if self.geometry_only else [
            pa.field("filename", pa.string()),
            pa.field("latitude", pa.float64()),
            pa.field("longitude", pa.float64()),
            pa.field("reserved", pa.int64()),
            pa.field("altitude", pa.float64()),
            pa.field("date_days", pa.float64()),
            pa.field("date", pa.string()),
            pa.field("time", pa.string()),
        ]
        fields.append(pa.field(self.geom_col, pa.binary()))
        return _attach_geoparquet_metadata(
            pa.schema(fields),
            "EPSG:4326",
            geom_col=self.geom_col,
        )

    def schema(self) -> pa.Schema:
        return self._schema

    def set_schema(self, schema: pa.Schema) -> None:
        if self.geom_col not in schema.names:
            raise ValueError(f"PLT schema must contain geometry column {self.geom_col!r}")
        self._schema = schema

    def input_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._files)

    def create_splits(self, num_splits: Optional[int] = None) -> List[PLTSplit]:
        # A file is the indivisible unit because its first six lines contain
        # trajectory-level metadata.
        return [PLTSplit(path=str(path)) for path in self._files]

    def iter_tables(self, split: Optional[PLTSplit] = None) -> Iterable[pa.Table]:
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            yield from self._iter_file_tables(Path(source_split.path))

    def _iter_file_tables(self, path: Path) -> Iterable[pa.Table]:
        rows: List[Tuple[float, float, int, float, float, str, str]] = []

        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            for header_line in range(1, 7):
                try:
                    next(reader)
                except StopIteration as exc:
                    raise ValueError(
                        f"Invalid PLT file {path}: expected six header lines, "
                        f"stopped at line {header_line}"
                    ) from exc

            for line_number, values in enumerate(reader, start=7):
                if not values or all(not value.strip() for value in values):
                    continue
                rows.append(_parse_plt_point(values, path=path, line_number=line_number))
                if len(rows) >= self.batch_rows:
                    yield self._rows_to_table(rows, path.name)
                    rows = []

        if rows:
            yield self._rows_to_table(rows, path.name)

    def _rows_to_table(
        self,
        rows: Sequence[Tuple[float, float, int, float, float, str, str]],
        filename: str,
    ) -> pa.Table:
        latitudes = [row[0] for row in rows]
        longitudes = [row[1] for row in rows]
        geometry = pa.array(
            to_wkb(points(longitudes, latitudes), hex=False).tolist(),
            type=pa.binary(),
        )

        if self.geometry_only:
            table = pa.table([geometry], names=[self.geom_col])
        else:
            row_count = len(rows)
            table = pa.table(
                {
                    "filename": pa.array([filename] * row_count, type=pa.string()),
                    "latitude": pa.array(latitudes, type=pa.float64()),
                    "longitude": pa.array(longitudes, type=pa.float64()),
                    "reserved": pa.array([row[2] for row in rows], type=pa.int64()),
                    "altitude": pa.array([row[3] for row in rows], type=pa.float64()),
                    "date_days": pa.array([row[4] for row in rows], type=pa.float64()),
                    "date": pa.array([row[5] for row in rows], type=pa.string()),
                    "time": pa.array([row[6] for row in rows], type=pa.string()),
                    self.geom_col: geometry,
                },
                schema=self._default_schema(),
            )

        table = table.replace_schema_metadata(self._schema.metadata)
        if not table.schema.equals(self._schema, check_metadata=False):
            table = table.cast(self._schema)
        return table.combine_chunks()


def _parse_plt_point(
    values: Sequence[str],
    *,
    path: Path,
    line_number: int,
) -> Tuple[float, float, int, float, float, str, str]:
    if len(values) != 7:
        raise ValueError(
            f"Invalid PLT record in {path} at line {line_number}: "
            f"expected 7 fields, found {len(values)}"
        )

    stripped = [value.strip() for value in values]
    try:
        return (
            float(stripped[0]),
            float(stripped[1]),
            int(stripped[2]),
            float(stripped[3]),
            float(stripped[4]),
            stripped[5],
            stripped[6],
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid PLT record in {path} at line {line_number}: {exc}"
        ) from exc
