from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import argparse
import json
import logging
import os

import numpy as np

from .descriptor import build_dataset_descriptor
from .embedder import GeminiTextEmbedder, TextEmbedder
from .pgvector_store import PgVectorConfig, PgVectorStore

logger = logging.getLogger(__name__)

CATALOG_DIRNAME = "_catalog"
CATALOG_FILENAME = "catalog_index.json"
CATALOG_EMBEDDINGS_FILENAME = "embeddings.npy"


def _iter_dataset_dirs(data_root: Path):
    if not data_root.exists():
        return
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        if path.name == CATALOG_DIRNAME:
            continue
        yield path


def _load_stats(stats_path: Path) -> Optional[Dict[str, Any]]:
    if not stats_path.exists():
        return None
    try:
        with open(stats_path, "r") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load stats from %s", stats_path)
        return None


def _pgvector_enabled() -> bool:
    value = os.environ.get("CATALOG_PGVECTOR_ENABLED", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def build_catalog_index(
    data_root: Path | str,
    out_dir: Optional[Path | str] = None,
    embedder: Optional[TextEmbedder] = None,
    sync_pgvector: Optional[bool] = None,
) -> Dict[str, Any]:
    data_root = Path(data_root)
    out_dir = Path(out_dir) if out_dir is not None else (data_root / CATALOG_DIRNAME)
    out_dir.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        embedder = GeminiTextEmbedder()

    if sync_pgvector is None:
        sync_pgvector = _pgvector_enabled()

    metadata_entries: List[Dict[str, Any]] = []
    pgvector_rows: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []

    for dataset_dir in _iter_dataset_dirs(data_root):
        stats_path = dataset_dir / "stats" / "attributes.json"
        stats_json = _load_stats(stats_path)
        if not stats_json:
            logger.info("Skipping %s because stats file is missing or unreadable", dataset_dir.name)
            continue

        descriptor = build_dataset_descriptor(dataset_dir.name, stats_json)

        try:
            embedding = embedder.embed_document(
                descriptor["text"],
                title=dataset_dir.name,
            )
        except Exception:
            logger.exception("Failed to embed dataset descriptor for %s", dataset_dir.name)
            continue

        entry = {
            "dataset": dataset_dir.name,
            "stats_path": str(stats_path.relative_to(data_root)),
            "descriptor_text": descriptor["text"],
            "summary": descriptor["summary"],
        }
        metadata_entries.append(entry)
        embeddings.append([float(v) for v in embedding])

        pgvector_rows.append({
            "dataset": dataset_dir.name,
            "stats_path": str(stats_path.relative_to(data_root)),
            "descriptor_text": descriptor["text"],
            "summary": descriptor["summary"],
            "embedding": [float(v) for v in embedding],
        })

        logger.info("Indexed dataset %s", dataset_dir.name)

    emb_array = np.asarray(embeddings, dtype=np.float32)
    emb_path = out_dir / CATALOG_EMBEDDINGS_FILENAME
    np.save(emb_path, emb_array, allow_pickle=False)

    catalog = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "embedding_model": getattr(embedder, "model_name", None),
        "embedding_file": CATALOG_EMBEDDINGS_FILENAME,
        "entry_count": len(metadata_entries),
        "entries": metadata_entries,
    }

    out_path = out_dir / CATALOG_FILENAME
    with open(out_path, "w") as f:
        json.dump(catalog, f, indent=2)

    logger.info("Wrote catalogue metadata to %s", out_path)
    logger.info("Wrote catalogue embeddings to %s", emb_path)

    if sync_pgvector and pgvector_rows:
        try:
            store = PgVectorStore(PgVectorConfig())
            store.upsert_many(pgvector_rows)
            logger.info(
                "Synced %d catalogue embeddings to pgvector table %s",
                len(pgvector_rows),
                store.config.table_name,
            )
        except Exception:
            logger.exception("Failed to sync catalogue to pgvector")

    return catalog


def load_catalog_index(index_dir_or_file: Path | str) -> Dict[str, Any]:
    path = Path(index_dir_or_file)
    if path.is_dir():
        path = path / CATALOG_FILENAME

    if not path.exists():
        raise FileNotFoundError(f"Catalogue index not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


def load_catalog_embeddings(index_dir_or_file: Path | str) -> np.ndarray:
    path = Path(index_dir_or_file)

    if path.is_file() and path.name == CATALOG_FILENAME:
        base = path.parent
    elif path.is_dir():
        base = path
    else:
        base = path.parent

    emb_path = base / CATALOG_EMBEDDINGS_FILENAME
    if not emb_path.exists():
        raise FileNotFoundError(f"Catalogue embeddings file not found: {emb_path}")

    return np.load(emb_path, allow_pickle=False)


def _main():
    parser = argparse.ArgumentParser(description="Build dataset embedding catalogue index")
    parser.add_argument("--data-dir", required=True, help="Root directory containing dataset folders")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for the catalogue index (default: <data-dir>/_catalog)",
    )
    parser.add_argument(
        "--sync-pgvector",
        action="store_true",
        help="Also sync the generated embeddings into PostgreSQL/pgvector",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    build_catalog_index(
        data_root=Path(args.data_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        sync_pgvector=bool(args.sync_pgvector),
    )


if __name__ == "__main__":
    _main()