"""Feature-aligned, byte-range reader for conventional GeoJSON FeatureCollections."""

from __future__ import annotations

import mmap
from pathlib import Path
import re
from typing import Iterator


WHITESPACE = b" \t\r\n"
# A Feature's required type member can appear anywhere in its object. The
# match identifies that member; _object_start_before finds the enclosing `{`.
FEATURE_TYPE = re.compile(rb'"type"\s*:\s*"Feature"')


class GeoJSONPartitionReader:
    """Yield complete GeoJSON Feature strings from one nominal byte partition.

    ``offset`` and ``length`` describe a fixed byte partition. The reader
    advances both ends to Feature boundaries: a feature belongs to the
    partition containing its starting boundary, so adjacent partitions neither
    duplicate nor omit features. Feature JSON is returned verbatim as UTF-8;
    this class performs no JSON decoding.
    """

    def __init__(self, path: str | Path, offset: int, length: int, batch_size: int = 1_024):
        self.path = Path(path)
        self.offset = offset
        self.length = length
        self.batch_size = batch_size
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if length < 1:
            raise ValueError("length must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

    def batches(self) -> Iterator[list[str]]:
        """Return complete Feature JSON strings in batches of ``batch_size``."""
        file_size = self.path.stat().st_size
        if self.offset >= file_size:
            return

        with self.path.open("rb") as file_obj, mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ) as data:
            start = self._next_feature_start(data, self.offset, self.offset)
            if start is None:
                return

            nominal_end = min(self.offset + self.length, file_size)
            end = self._next_feature_start(data, nominal_end, nominal_end)
            if end is None:
                end = nominal_end
            if start >= end:
                return

            batch: list[str] = []
            current_start = start
            while current_start < end:
                next_start = self._next_feature_start(data, current_start + 1, current_start + 1)
                if next_start is None:
                    # This is the last Feature in the file. The nominal range
                    # extends through the enclosing FeatureCollection's `]}`;
                    # finish at this object's closing brace instead.
                    current_end = self._find_json_object_end(data, current_start)
                else:
                    current_end = end if next_start >= end else next_start
                    current_end = self._trim_feature_end(data, current_end)
                batch.append(bytes(data[current_start:current_end]).decode("utf-8"))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                if next_start is None or next_start >= end:
                    break
                current_start = next_start

            if batch:
                yield batch

    def __iter__(self) -> Iterator[list[str]]:
        return self.batches()


    @staticmethod
    def _next_feature_start(data: mmap.mmap, search_from: int, minimum_start: int) -> int | None:
        """Find the next Feature object beginning at or after ``minimum_start``.

        We remember the location of each candidate's ``"type": "Feature"``
        member, then walk backwards to the enclosing object start. This is the
        same essential alignment strategy as a token reader that records an
        object's start before it later encounters its type member.
        """
        candidate = FEATURE_TYPE.search(data, search_from)
        while candidate is not None:
            object_start = GeoJSONPartitionReader._object_start_before(data, candidate.start())
            if object_start >= minimum_start:
                return object_start
            candidate = FEATURE_TYPE.search(data, candidate.end())
        return None

    @staticmethod
    def _object_start_before(data: mmap.mmap, position: int) -> int:
        """Return the opening brace of the JSON object containing ``position``."""
        nesting = 0
        in_string = False
        position -= 1
        while position >= 0:
            byte = data[position]
            if byte == ord('"') and not GeoJSONPartitionReader._is_escaped(data, position):
                in_string = not in_string
            elif not in_string:
                if byte == ord("}"):
                    nesting += 1
                elif byte == ord("{"):
                    if nesting == 0:
                        return position
                    nesting -= 1
            position -= 1
        raise ValueError('Could not find the object containing a "type": "Feature" member')

    @staticmethod
    def _is_escaped(data: mmap.mmap, position: int) -> bool:
        backslashes = 0
        position -= 1
        while position >= 0 and data[position] == ord("\\"):
            backslashes += 1
            position -= 1
        return backslashes % 2 == 1

    @staticmethod
    def _trim_feature_end(data: mmap.mmap, end: int) -> int:
        """Remove the comma and whitespace separating adjacent features."""
        while end > 0 and data[end - 1] in WHITESPACE:
            end -= 1
        if end > 0 and data[end - 1] == ord(","):
            end -= 1
        while end > 0 and data[end - 1] in WHITESPACE:
            end -= 1
        return end

    @staticmethod
    def _find_json_object_end(data: mmap.mmap, start: int) -> int:
        """Return the exclusive end of the JSON object beginning at ``start``."""
        depth = 0
        in_string = False
        position = start
        while position < len(data):
            byte = data[position]
            if byte == ord('"') and not GeoJSONPartitionReader._is_escaped(data, position):
                in_string = not in_string
            elif not in_string:
                if byte == ord("{"):
                    depth += 1
                elif byte == ord("}"):
                    depth -= 1
                    if depth == 0:
                        return position + 1
            position += 1
        raise ValueError(f"Unterminated JSON object beginning at byte {start}")
