import pyarrow as pa
from collections import OrderedDict
from .sketches import (
    NumericSketch,
    CategoricalSketch,
    TextSketch,
    TemporalSketch,
    GeometrySketch,
)


class AttributeStatsCollector:
    def __init__(self, schema: pa.Schema, geometry_column="geometry", global_mbr=None):
        """
        Initialize AttributeStatsCollector with optional pre-computed global MBR.

        Args:
            schema: PyArrow schema
            geometry_column: Name of geometry column
            global_mbr: Optional tuple of (minx, miny, maxx, maxy) to avoid redundant MBR computation
        """
        self.schema = schema
        self.geometry_column = geometry_column
        self.sketches = OrderedDict()

        for field in schema:
            name = field.name
            if name == geometry_column:
                self.sketches[name] = GeometrySketch(global_mbr=global_mbr)
                continue

            self.sketches[name] = _sketch_for_type(field.type)

    def consume_table(self, table: pa.Table):
        for col_name, sketch in self.sketches.items():
            if col_name not in table.column_names:
                continue

            col = table[col_name]

            if isinstance(sketch, GeometrySketch):
                # geometry is already decoded upstream in orchestrator
                geoms = col.to_pylist()
                sketch.update(geoms)
                continue

            desired_sketch = _sketch_for_type(col.type)
            if type(sketch) is not type(desired_sketch) and _is_empty_sketch(sketch):
                sketch = desired_sketch
                self.sketches[col_name] = sketch

            # fast path for primitive columns
            arr = col.combine_chunks()
            values = arr.to_pylist()
            sketch.update(values)

    def merge(self, other: "AttributeStatsCollector"):
        """Merge another collector's sketches into this one (for parallel
        stats collection across workers). Sketches present in ``other`` but not
        here are adopted as-is."""
        for name, other_sketch in other.sketches.items():
            mine = self.sketches.get(name)
            if mine is None:
                self.sketches[name] = other_sketch
            elif type(mine) is not type(other_sketch) and _is_empty_sketch(mine):
                self.sketches[name] = other_sketch
            elif type(mine) is not type(other_sketch) and _is_empty_sketch(other_sketch):
                continue
            else:
                mine.merge(other_sketch)
        return self

    def finalize(self):
        out = []

        for name, sketch in self.sketches.items():
            entry = {
                "name": name,
                "stats": sketch.finalize(),
            }
            out.append(entry)

        return {"attributes": out}


def _sketch_for_type(t: pa.DataType):
    if pa.types.is_integer(t) or pa.types.is_floating(t):
        return NumericSketch()
    if pa.types.is_boolean(t):
        return CategoricalSketch()
    if pa.types.is_timestamp(t) or pa.types.is_date(t) or pa.types.is_time(t):
        return TemporalSketch()
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return TextSketch()
    return CategoricalSketch()


def _is_empty_sketch(sketch) -> bool:
    if isinstance(sketch, GeometrySketch):
        return sketch.total_points == 0
    if isinstance(sketch, NumericSketch):
        return sketch.count == 0
    return getattr(sketch, "non_null", 0) == 0
