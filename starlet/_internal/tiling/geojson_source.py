from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pyarrow as pa

from starlet._internal.tiling.datasource import (
    DataSource,
    _GEOJSON_SUFFIXES,
    _attach_geoparquet_metadata,
    _extract_feature_collection_crs_hint,
    _geometries_to_wkb,
    _geojson_partition_ranges,
    _normalize_decimal_columns,
    _source_files,
)
from starlet._internal.tiling.partition_reader import GeoJSONPartitionReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoJSONSplit:
    """Byte range to read from one GeoJSON source."""

    path: str
    offset: int
    length: int


class GeoJSONSource(DataSource):
    """
    Streams GeoJSON / GeoJSONL as Arrow Tables, converting geometry to WKB.

    - For standard FeatureCollection GeoJSON, reads byte partitions in parallel.
    - For GeoJSON Lines (one Feature per line), reads and batches by line.
    - Geometry dicts → shapely.shape → WKB bytes (binary Arrow column 'geometry').
    - Attaches minimal GeoParquet metadata (version, primary_column, encoding, crs hint).
    """

    def __init__(
        self,
        path: str,
        batch_rows: int = 1_000,
        src_crs: str = "EPSG:4326",
        target_crs: Optional[str] = None,
        keep_null_geoms: bool = False,
    ):
        self.path = str(path)
        self._files = _source_files(self.path, _GEOJSON_SUFFIXES)
        if not self._files:
            raise ValueError(f"No GeoJSON files found in {self.path}")
        self.batch_rows = int(batch_rows)
        self.src_crs = src_crs
        self.target_crs = target_crs  # informational only here
        self.keep_null_geoms = keep_null_geoms

        if target_crs:
            logger.warning(
                "target_crs requested (%s) but GeoJSON reader does not reproject on the fly; data will be read as-is.",
                target_crs,
            )

        self._schema: Optional[pa.Schema] = None
        self._crs_hint: Optional[str] = _extract_feature_collection_crs_hint(str(self._files[0]))

        logger.info(
            "GeoJSONSource opened %s with %d files (batch_rows=%d, src_crs=%s)",
            path, len(self._files), self.batch_rows, self.src_crs
        )

    # ---------------- schema ---------------- #
    def schema(self) -> pa.Schema:
        if self._schema is None:
            first = self._read_first_batch()
            if first is None or first.num_rows == 0:
                # Empty input file. Create a minimal schema with geometry column.
                base = pa.schema([("geometry", pa.binary())])
                self._schema = _attach_geoparquet_metadata(
                    base, self._crs_hint or self.target_crs or self.src_crs
                )
            else:
                # Lock schema with GeoParquet metadata
                self._schema = _attach_geoparquet_metadata(
                    first.schema, self._crs_hint or self.target_crs or self.src_crs
                )

        return self._schema

    def input_size_bytes(self) -> int:
        return sum(file_path.stat().st_size for file_path in self._files)

    # ---------------- iterator ---------------- #
    def create_splits(self) -> List[GeoJSONSplit]:
        target_partition_size = 32 * 1024 * 1024
        splits: List[GeoJSONSplit] = []
        for file_path in self._files:
            file_size = file_path.stat().st_size
            num_splits = max(1, (file_size + target_partition_size - 1) // target_partition_size)
            splits.extend(
                GeoJSONSplit(path=str(file_path), offset=offset, length=length)
                for offset, length in _geojson_partition_ranges(file_size, int(num_splits))
            )
        return splits

    def iter_tables(self, split: Optional[GeoJSONSplit] = None) -> Iterable[pa.Table]:
        batch_index = 0
        crs_value = self._crs_hint or self.target_crs or self.src_crs

        import geopandas as gpd

        for features in self._iter_feature_batches_for_split(split):
            if not features:
                continue

            gdf = gpd.GeoDataFrame.from_features(features, crs=crs_value)
            geometry_col = pa.array(gdf.geometry.to_wkb(), type=pa.binary())

            props_df = gdf.drop(columns="geometry")
            props_df = _normalize_decimal_columns(props_df)
            props_table = pa.Table.from_pandas(props_df, preserve_index=False)

            table = (
                pa.table([geometry_col], names=["geometry"])
                if props_table.num_columns == 0
                else props_table.append_column("geometry", geometry_col)
            )

            # Attach GeoParquet metadata with CRS
            schema_with_geo = _attach_geoparquet_metadata(table.schema, crs_value)
            table = table.replace_schema_metadata(schema_with_geo.metadata)

            if split is None and not self._schema:
                self._schema = schema_with_geo

            if self._schema is not None:
                table = self._coerce_to_schema(table, self._schema)
            table = table.combine_chunks()

            logger.debug(
                "GeoJSON batch %d (%d rows) -> %d columns (including 'geometry')",
                batch_index,
                table.num_rows,
                len(table.column_names),
            )
            batch_index += 1
            yield table

    def _iter_feature_batches_for_split(
        self,
        split: Optional[GeoJSONSplit],
    ) -> Iterable[List[Dict[str, Any]]]:
        if split is None:
            for source_split in self.create_splits():
                yield from self._iter_feature_batches_for_split(source_split)
            return

        reader = GeoJSONPartitionReader(split.path, split.offset, split.length, batch_size=self.batch_rows)
        for feature_batch in reader.batches():
            yield [json.loads(feature) for feature in feature_batch]

    # ---------------- internal helpers ---------------- #
    def _read_first_batch(self) -> Optional[pa.Table]:
        """Read the first batch of features to establish the schema."""
        first_path = self._files[0]
        file_size = first_path.stat().st_size
        batches = GeoJSONPartitionReader(first_path, 0, file_size, batch_size=max(1, self.batch_rows)).batches()
        try:
            first_batch = next(batches)
            features = [json.loads(feature_str) for feature_str in first_batch]
        except StopIteration:
            logger.info("GeoJSON read returned 0 rows when inferring schema")
            return None
        finally:
            batches.close()

        rows_props: List[Dict[str, Any]] = []
        geometries: List[Any] = []

        for feat in features:
            rows_props.append(feat.get("properties") or {})
            geometries.append(feat.get("geometry", None))

        props_df = pd.DataFrame.from_records(rows_props)
        props_df = _normalize_decimal_columns(props_df)
        props_table = pa.Table.from_pandas(props_df, preserve_index=False)

        wkb_list = _geometries_to_wkb(geometries)
        geometry_col = pa.array(wkb_list, type=pa.binary())

        if props_table.num_columns == 0:
            return pa.table([geometry_col], names=["geometry"])

        return props_table.append_column("geometry", geometry_col)

    def _coerce_to_schema(self, t: pa.Table, schema: pa.Schema) -> pa.Table:
        if t.schema.equals(schema):
            return t

        out_cols = []
        for fld in schema:
            name = fld.name
            if name in t.column_names:
                col = t[name]
                if not col.type.equals(fld.type):
                    try:
                        col = col.cast(fld.type)
                    except Exception:
                        logger.warning(
                            "Type mismatch for column '%s': %s -> %s (kept original)",
                            name, col.type, fld.type
                        )
                out_cols.append(col)
            else:
                out_cols.append(pa.nulls(t.num_rows, type=fld.type))

        return pa.table(out_cols, names=[f.name for f in schema])

