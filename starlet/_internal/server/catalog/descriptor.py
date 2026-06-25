from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_stats(attr: Dict[str, Any]) -> Dict[str, Any]:
    return attr.get("stats") or {}


def _top_k_values(stats: Dict[str, Any], max_items: int = 8) -> List[str]:
    items = stats.get("top_k") or []
    out: List[str] = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            value = item.get("value")
            count = item.get("count")
            if count is None:
                out.append(str(value))
            else:
                out.append(f"{value} ({count})")
        else:
            out.append(str(item))
    return out


def infer_attribute_role(attr: Dict[str, Any]) -> str:
    stats = _safe_stats(attr)

    if "geom_types" in stats or "mbr" in stats:
        return "geometry"

    if any(k in stats for k in ("min", "max", "mean", "stddev")):
        return "numeric"

    if any(k in stats for k in ("avg_length", "min_length", "max_length")):
        approx_distinct = stats.get("approx_distinct")
        if isinstance(approx_distinct, int) and approx_distinct <= 50:
            return "categorical_text"
        return "text"

    return "categorical"


def _attribute_summary(attr: Dict[str, Any]) -> Dict[str, Any]:
    name = attr.get("name")
    stats = _safe_stats(attr)
    role = infer_attribute_role(attr)

    summary: Dict[str, Any] = {
        "name": name,
        "role": role,
    }

    if role == "geometry":
        summary["geom_types"] = stats.get("geom_types") or {}
        summary["mbr"] = stats.get("mbr")
        summary["total_points"] = stats.get("total_points")
        return summary

    summary["approx_distinct"] = stats.get("approx_distinct")
    summary["non_null_count"] = stats.get("non_null_count")

    if role == "numeric":
        summary["min"] = stats.get("min")
        summary["max"] = stats.get("max")
        summary["mean"] = stats.get("mean")
        summary["stddev"] = stats.get("stddev")
        summary["top_k"] = stats.get("top_k") or []
        return summary

    if role in ("text", "categorical_text"):
        summary["avg_length"] = stats.get("avg_length")
        summary["min_length"] = stats.get("min_length")
        summary["max_length"] = stats.get("max_length")
        summary["top_k"] = stats.get("top_k") or []
        return summary

    summary["top_k"] = stats.get("top_k") or []
    return summary


def _attribute_text(attr: Dict[str, Any]) -> str:
    name = attr.get("name", "unknown")
    stats = _safe_stats(attr)
    role = infer_attribute_role(attr)

    if role == "geometry":
        geom_types = stats.get("geom_types") or {}
        mbr = stats.get("mbr")
        total_points = stats.get("total_points")
        return (
            f"- {name}: geometry; "
            f"geom_types={geom_types}; "
            f"mbr={mbr}; "
            f"total_points={total_points}"
        )

    approx_distinct = stats.get("approx_distinct")
    non_null_count = stats.get("non_null_count")
    top_values = _top_k_values(stats)

    if role == "numeric":
        return (
            f"- {name}: numeric; "
            f"non_null={non_null_count}; "
            f"approx_distinct={approx_distinct}; "
            f"min={stats.get('min')}; "
            f"max={stats.get('max')}; "
            f"mean={stats.get('mean')}; "
            f"stddev={stats.get('stddev')}; "
            f"top_values={top_values}"
        )

    if role in ("text", "categorical_text"):
        return (
            f"- {name}: {role}; "
            f"non_null={non_null_count}; "
            f"approx_distinct={approx_distinct}; "
            f"avg_length={stats.get('avg_length')}; "
            f"min_length={stats.get('min_length')}; "
            f"max_length={stats.get('max_length')}; "
            f"top_values={top_values}"
        )

    return (
        f"- {name}: categorical; "
        f"non_null={non_null_count}; "
        f"approx_distinct={approx_distinct}; "
        f"top_values={top_values}"
    )


def build_dataset_descriptor(
    dataset_name: str,
    stats_json: Dict[str, Any],
    dataset_description: Optional[str] = None,
    max_attributes_for_summary: int = 120,
) -> Dict[str, Any]:
    attributes = stats_json.get("attributes") or []

    geometry_attrs = []
    non_geometry_attrs = []
    for attr in attributes:
        if infer_attribute_role(attr) == "geometry":
            geometry_attrs.append(attr)
        else:
            non_geometry_attrs.append(attr)

    summary_attributes = [
        _attribute_summary(attr)
        for attr in non_geometry_attrs[:max_attributes_for_summary]
    ]

    geometry_summary = [
        _attribute_summary(attr)
        for attr in geometry_attrs
    ]

    lines: List[str] = []
    lines.append(f"Dataset: {dataset_name}")
    if dataset_description:
        lines.append(f"Description: {dataset_description}")

    if geometry_attrs:
        lines.append("Geometry:")
        for attr in geometry_attrs:
            lines.append(_attribute_text(attr))

    lines.append("Attributes:")
    for attr in non_geometry_attrs:
        lines.append(_attribute_text(attr))

    descriptor_text = "\n".join(lines)

    return {
        "dataset": dataset_name,
        "text": descriptor_text,
        "summary": {
            "dataset": dataset_name,
            "description": dataset_description,
            "geometry": geometry_summary,
            "attributes": summary_attributes,
            "attribute_count": len(non_geometry_attrs),
            "geometry_attribute_count": len(geometry_attrs),
        },
    }