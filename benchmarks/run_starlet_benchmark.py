#!/usr/bin/env python3
"""Benchmark Starlet's full pipeline (tiling + MVT generation) across dataset sizes."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PERCENTAGES = [25, 50, 75, 100]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STARLET_BIN = (
    os.environ.get("STARLET_BIN")
    or shutil.which("starlet")
    or str(SCRIPT_DIR.parent / ".venv" / "bin" / "starlet")
)


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly identifier."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def _is_parquet(dataset: Path) -> bool:
    return dataset.is_file() and dataset.suffix.lower() in {".parquet", ".geoparquet"}


def _list_vector_layers(dataset: Path) -> list[dict]:
    """Return vector layers with feature counts, or an empty list for non-vector inputs."""
    try:
        import pyogrio
    except Exception:
        return []

    try:
        raw_layers = pyogrio.list_layers(dataset)
    except Exception:
        return []

    layers = []
    for row in raw_layers:
        name = str(row[0])
        geometry_type = str(row[1]) if len(row) > 1 else None
        features = None
        try:
            info = pyogrio.read_info(dataset, layer=name)
            features = info.get("features")
            geometry_type = info.get("geometry_type") or geometry_type
        except Exception:
            pass
        layers.append({
            "name": name,
            "geometry_type": geometry_type,
            "features": int(features) if features is not None and features >= 0 else None,
        })
    return layers


def resolve_dataset_info(dataset: Path, requested_layer: str | None) -> tuple[int, str | None]:
    """Resolve the benchmarked row count and optional vector layer name."""
    if _is_parquet(dataset):
        import pyarrow.parquet as pq

        return int(pq.read_metadata(dataset).num_rows), requested_layer

    layers = _list_vector_layers(dataset)
    if not layers:
        raise RuntimeError(
            "Could not determine row count. For non-Parquet inputs, install pyogrio/GDAL "
            "or pass a supported vector dataset."
        )

    if requested_layer:
        matches = [layer for layer in layers if layer["name"] == requested_layer]
        if not matches:
            available = ", ".join(layer["name"] for layer in layers)
            raise RuntimeError(
                f"Layer {requested_layer!r} not found. Available layers: {available}"
            )
        layer = matches[0]
    else:
        layer = max(layers, key=lambda item: item["features"] or 0)
        print(
            "Auto-selected layer: "
            f"{layer['name']} ({layer['features']:,} rows, {layer['geometry_type']})"
        )

    if layer["features"] is None:
        raise RuntimeError(f"Could not determine feature count for layer {layer['name']!r}")
    return layer["features"], layer["name"]


def row_counts_for_total(total_rows: int) -> dict[int, int]:
    return {
        25: total_rows // 4,
        50: total_rows // 2,
        75: (total_rows * 3) // 4,
        100: total_rows,
    }


def dir_size_bytes(path: Path) -> int:
    """Recursively compute directory size in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def generate_subset(
    dataset: Path,
    output: Path,
    pct: int,
    row_counts: dict[int, int],
    layer_name: str | None,
) -> Path | None:
    """Generate a subset Parquet file using ogr2ogr and return its path on success."""
    if pct == 100 and _is_parquet(dataset) and layer_name is None:
        print(f"  [100%] Using original Parquet dataset: {dataset}")
        return dataset

    if output.exists() and output.stat().st_size > 0:
        print(f"  [{pct}%] Subset already exists: {output} ({output.stat().st_size / 1e6:.1f} MB)")
        return output

    limit = row_counts[pct]
    cmd = [
        "ogr2ogr",
        "-overwrite",
        "-f",
        "Parquet",
        "-limit",
        str(limit),
        str(output),
        str(dataset),
    ]
    if layer_name:
        cmd.append(layer_name)
    print(f"  [{pct}%] Generating subset ({limit:,} rows)...")
    print(f"         cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{pct}%] ERROR: ogr2ogr failed:\n{result.stderr}")
        return None

    print(f"  [{pct}%] Done. Size: {output.stat().st_size / 1e6:.1f} MB")
    return output


def run_pipeline(
    subset: Path,
    results_dir: Path,
    pct: int,
    row_counts: dict[int, int],
    starlet_bin: Path,
) -> dict:
    """Run starlet tile + mvt on a subset and collect metrics."""
    metrics = {
        "pct": pct,
        "input_rows": row_counts[pct],
        "input_size_mb": round(subset.stat().st_size / (1024 * 1024), 2),
        "indexing_time_s": None,
        "mvt_time_s": None,
        "total_time_s": None,
        "num_parquet_tiles": 0,
        "num_mvt_tiles": 0,
        "output_size_mb": 0,
        "error": None,
    }

    outdir = results_dir / f"pct_{pct}"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- starlet tile ---
    tile_cmd = [
        str(starlet_bin), "tile",
        "--input", str(subset),
        "--outdir", str(outdir),
    ]
    print(f"  [{pct}%] Running: {' '.join(tile_cmd)}")
    t0 = time.perf_counter()
    tile_result = subprocess.run(tile_cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    metrics["indexing_time_s"] = round(t1 - t0, 3)

    if tile_result.returncode != 0:
        msg = tile_result.stderr.strip() or tile_result.stdout.strip()
        print(f"  [{pct}%] ERROR: starlet tile failed:\n{msg}")
        metrics["error"] = f"tile failed: {msg[:500]}"
        return metrics

    print(f"  [{pct}%] Tiling done in {metrics['indexing_time_s']:.1f}s")

    # --- starlet mvt ---
    mvt_cmd = [
        str(starlet_bin), "mvt",
        "--dir", str(outdir),
    ]
    print(f"  [{pct}%] Running: {' '.join(mvt_cmd)}")
    t0 = time.perf_counter()
    mvt_result = subprocess.run(mvt_cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    metrics["mvt_time_s"] = round(t1 - t0, 3)

    if mvt_result.returncode != 0:
        msg = mvt_result.stderr.strip() or mvt_result.stdout.strip()
        print(f"  [{pct}%] ERROR: starlet mvt failed:\n{msg}")
        metrics["error"] = f"mvt failed: {msg[:500]}"
        return metrics

    print(f"  [{pct}%] MVT done in {metrics['mvt_time_s']:.1f}s")

    metrics["total_time_s"] = round(metrics["indexing_time_s"] + metrics["mvt_time_s"], 3)

    # --- count outputs ---
    parquet_tiles_dir = outdir / "parquet_tiles"
    if parquet_tiles_dir.exists():
        metrics["num_parquet_tiles"] = len(list(parquet_tiles_dir.glob("*.parquet")))

    mvt_dir = outdir / "mvt"
    if mvt_dir.exists():
        metrics["num_mvt_tiles"] = (
            len(list(mvt_dir.rglob("*.mvt")))
            + len(list(mvt_dir.rglob("*.pbf")))
        )

    metrics["output_size_mb"] = round(dir_size_bytes(outdir) / (1024 * 1024), 2)

    return metrics


def write_results(results: list[dict], output_dir: Path):
    """Write benchmark results to JSON and CSV."""
    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_results.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {json_path}")

    fieldnames = [
        "pct", "input_rows", "input_size_mb",
        "indexing_time_s", "mvt_time_s", "total_time_s",
        "num_parquet_tiles", "num_mvt_tiles", "output_size_mb", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {csv_path}")


def print_summary(results: list[dict]):
    """Print a summary table to stdout."""
    header = f"{'%':>5} | {'Rows':>12} | {'Size(MB)':>10} | {'Tile(s)':>10} | {'MVT(s)':>10} | {'Total(s)':>10} | {'#Parq':>6} | {'#MVT':>6} | {'Out(MB)':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        tile_t = f"{r['indexing_time_s']:.1f}" if r["indexing_time_s"] is not None else "ERR"
        mvt_t = f"{r['mvt_time_s']:.1f}" if r["mvt_time_s"] is not None else "ERR"
        total_t = f"{r['total_time_s']:.1f}" if r["total_time_s"] is not None else "ERR"
        print(
            f"{r['pct']:>5} | {r['input_rows']:>12,} | {r['input_size_mb']:>10.1f} | "
            f"{tile_t:>10} | {mvt_t:>10} | {total_t:>10} | "
            f"{r['num_parquet_tiles']:>6} | {r['num_mvt_tiles']:>6} | {r['output_size_mb']:>10.1f}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Starlet pipeline across dataset sizes.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a Starlet-supported source dataset",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for subsets, results, and reports",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Vector layer to benchmark. Defaults to the largest layer.",
    )
    parser.add_argument(
        "--starlet-bin",
        default=DEFAULT_STARLET_BIN,
        help="Path to the starlet CLI executable.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    starlet_bin = Path(args.starlet_bin).resolve()

    if not dataset.exists():
        print(f"ERROR: Dataset not found: {dataset}")
        sys.exit(1)

    if not starlet_bin.exists():
        print(f"ERROR: Starlet binary not found: {starlet_bin}")
        sys.exit(1)

    datasets_dir = output_dir / "datasets"
    results_dir = output_dir / "results"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Read actual row count
    print(f"Dataset: {dataset}")
    try:
        total, layer_name = resolve_dataset_info(dataset, args.layer)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if layer_name:
        print(f"Layer: {layer_name}")
    print(f"Total rows: {total:,}")
    row_counts = row_counts_for_total(total)

    # Generate subsets
    print("\n=== Generating subsets ===")
    subsets = {}
    dataset_name = _safe_name(layer_name or dataset.stem)
    for pct in PERCENTAGES:
        subset_path = datasets_dir / f"{dataset_name}_{pct}pct.parquet"
        subset = generate_subset(dataset, subset_path, pct, row_counts, layer_name)
        if subset:
            subsets[pct] = subset
        else:
            print(f"  Skipping {pct}% due to subset generation failure.")

    # Run pipeline for each subset
    print("\n=== Running pipeline ===")
    all_results = []
    for pct in PERCENTAGES:
        if pct not in subsets:
            continue
        print(f"\n--- {pct}% ({row_counts[pct]:,} rows) ---")
        metrics = run_pipeline(subsets[pct], results_dir, pct, row_counts, starlet_bin)
        all_results.append(metrics)

    # Write and display results
    write_results(all_results, output_dir)
    print_summary(all_results)


if __name__ == "__main__":
    main()
