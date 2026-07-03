from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
from shapely import to_wkb

from starlet._internal.tiling.datasource import (
    DataSource,
    _GDB_SUFFIXES,
    _SHAPEFILE_SUFFIXES,
    _ZIP_SUFFIXES,
    _attach_geoparquet_metadata,
    _normalize_decimal_columns,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorLayerSplit:
    path: str
    layer: str | None
    skip_features: int
    max_features: int | None


@dataclass(frozen=True)
class _VectorLayer:
    path: str
    layer: str | None
    feature_count: int | None


class _OGRVectorSource(DataSource):
    def __init__(
        self,
        path: str,
        *,
        geometry_only: bool = False,
        geom_col: str = "geometry",
        batch_features: int = 65_536,
    ) -> None:
        self.path = str(path)
        self.geometry_only = bool(geometry_only)
        self.geom_col = geom_col
        self.batch_features = int(batch_features)
        self._layers = self._discover_layers(self.path)
        if not self._layers:
            raise ValueError(f"No vector layers found in {self.path}")
        self._schema: pa.Schema | None = None

    def schema(self) -> pa.Schema:
        if self._schema is None:
            first = next(self.iter_tables(), None)
            self._schema = first.schema if first is not None else _attach_geoparquet_metadata(
                pa.schema([(self.geom_col, pa.binary())]),
                "EPSG:4326",
            )
        return self._schema

    def input_size_bytes(self) -> int:
        total = 0
        for path in self._source_paths_for_size():
            p = Path(path)
            if p.is_file():
                total += p.stat().st_size
            elif p.is_dir():
                total += sum(child.stat().st_size for child in p.rglob("*") if child.is_file())
        return total

    def create_splits(self, num_splits: Optional[int] = None) -> List[VectorLayerSplit]:
        if num_splits is None:
            chunk_size = max(1, self.batch_features)
        else:
            known_rows = sum(layer.feature_count or 0 for layer in self._layers)
            chunk_size = max(1, (max(1, known_rows) + max(1, int(num_splits)) - 1) // max(1, int(num_splits)))

        splits: List[VectorLayerSplit] = []
        for layer in self._layers:
            if layer.feature_count is None or layer.feature_count < 0:
                splits.append(VectorLayerSplit(layer.path, layer.layer, 0, None))
                continue
            for offset in range(0, layer.feature_count, chunk_size):
                splits.append(
                    VectorLayerSplit(
                        path=layer.path,
                        layer=layer.layer,
                        skip_features=offset,
                        max_features=min(chunk_size, layer.feature_count - offset),
                    )
                )
        return splits

    def iter_tables(self, split: Optional[VectorLayerSplit] = None) -> Iterable[pa.Table]:
        splits = [split] if split is not None else self.create_splits()
        for source_split in splits:
            gdf = self._read_split(source_split)
            if gdf.empty:
                continue
            yield self._geodataframe_to_table(gdf)

    def _read_split(self, split: VectorLayerSplit):
        import pyogrio

        columns = [] if self.geometry_only else None
        return pyogrio.read_dataframe(
            split.path,
            layer=split.layer,
            columns=columns,
            read_geometry=True,
            skip_features=split.skip_features,
            max_features=split.max_features,
            use_arrow=True,
        )

    def _geodataframe_to_table(self, gdf) -> pa.Table:
        geometry_col = pa.array(to_wkb(gdf.geometry.array, hex=False).tolist(), type=pa.binary())
        props_df = pd.DataFrame(index=gdf.index) if self.geometry_only else gdf.drop(columns=gdf.geometry.name)
        props_df = _normalize_decimal_columns(props_df)
        props_table = pa.Table.from_pandas(props_df, preserve_index=False)
        table = (
            pa.table([geometry_col], names=[self.geom_col])
            if props_table.num_columns == 0
            else props_table.append_column(self.geom_col, geometry_col)
        )
        crs_hint = gdf.crs.to_json() if gdf.crs is not None else None
        schema_with_geo = _attach_geoparquet_metadata(table.schema, crs_hint)
        return table.replace_schema_metadata(schema_with_geo.metadata).combine_chunks()

    def _discover_layers(self, path: str) -> List[_VectorLayer]:
        raise NotImplementedError

    def _source_paths_for_size(self) -> List[str]:
        return sorted({layer.path for layer in self._layers})

    @staticmethod
    def _layers_for_dataset(path: str) -> List[_VectorLayer]:
        import pyogrio

        layers = []
        try:
            layer_info = pyogrio.list_layers(path)
        except Exception:
            layer_info = []
        layer_names = [str(row[0]) for row in layer_info] if len(layer_info) else [None]
        for layer_name in layer_names:
            try:
                info = pyogrio.read_info(path, layer=layer_name, force_feature_count=True)
                feature_count = info.get("features")
            except Exception:
                feature_count = None
            layers.append(_VectorLayer(path=path, layer=layer_name, feature_count=feature_count))
        return layers


class ShapefileSource(_OGRVectorSource):
    def _discover_layers(self, path: str) -> List[_VectorLayer]:
        source = Path(path)
        if source.is_file():
            if source.suffix.lower() == ".zip":
                return self._layers_for_dataset(_zip_vsi_path(source))
            if source.suffix.lower() == ".shp":
                return self._layers_for_dataset(str(source))
            raise ValueError(f"Not a Shapefile source: {path}")

        if source.is_dir():
            layers: List[_VectorLayer] = []
            for shp in sorted(source.rglob("*.shp")):
                if ".gdb" not in {part.lower() for part in shp.parts}:
                    layers.extend(self._layers_for_dataset(str(shp)))
            for zip_path in sorted(source.rglob("*.zip")):
                layers.extend(self._layers_for_dataset(_zip_vsi_path(zip_path)))
            return layers

        raise FileNotFoundError(f"Source path does not exist: {path}")


class GDBSource(_OGRVectorSource):
    def _discover_layers(self, path: str) -> List[_VectorLayer]:
        source = Path(path)
        gdb_paths: List[Path]
        if source.suffix.lower() in _GDB_SUFFIXES:
            gdb_paths = [source]
        elif source.is_dir():
            gdb_paths = sorted(child for child in source.rglob("*.gdb") if child.is_dir())
        else:
            raise FileNotFoundError(f"Source path does not exist: {path}")

        layers: List[_VectorLayer] = []
        for gdb_path in gdb_paths:
            layers.extend(self._layers_for_dataset(str(gdb_path)))
        return layers


def _zip_vsi_path(path: Path) -> str:
    return f"/vsizip/{path}"
