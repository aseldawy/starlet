from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .embedder import TextEmbedder
from .index import load_catalog_embeddings, load_catalog_index
from .pgvector_store import PgVectorStore


class SearchBackend(str, Enum):
    AUTO = "auto"
    NPY = "npy"
    PGVECTOR = "pgvector"


@dataclass
class CatalogSearchResult:
    dataset: str
    score: float
    stats_path: str
    descriptor_text: str
    summary: Dict[str, Any]


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray([], dtype=np.float32)

    q = query_vec.astype(np.float32, copy=False)
    m = matrix.astype(np.float32, copy=False)

    qnorm = np.linalg.norm(q)
    mnorm = np.linalg.norm(m, axis=1)

    denom = (mnorm * qnorm)
    denom = np.where(denom == 0.0, 1e-12, denom)

    return (m @ q) / denom


class CatalogRouter:
    def __init__(
        self,
        index_dir_or_file: str,
        embedder: TextEmbedder,
        backend: SearchBackend = SearchBackend.AUTO,
        pgvector_store: Optional[PgVectorStore] = None,
    ):
        self.embedder = embedder
        self.backend = backend
        self.pgvector_store = pgvector_store

        self.catalog_index = load_catalog_index(index_dir_or_file)
        self.catalog_embeddings = load_catalog_embeddings(index_dir_or_file)

    def _resolve_backend(self) -> SearchBackend:
        if self.backend == SearchBackend.NPY:
            return SearchBackend.NPY
        if self.backend == SearchBackend.PGVECTOR:
            return SearchBackend.PGVECTOR

        # AUTO: prefer pgvector if configured and reachable, else fallback to npy.
        if self.pgvector_store is not None and self.pgvector_store.ping():
            return SearchBackend.PGVECTOR
        return SearchBackend.NPY

    def search(self, query: str, k: int = 5) -> List[CatalogSearchResult]:
        query_embedding = self.embedder.embed_query(query)
        backend = self._resolve_backend()

        if backend == SearchBackend.PGVECTOR:
            rows = self.pgvector_store.search(query_embedding, k=k) if self.pgvector_store else []
            return [
                CatalogSearchResult(
                    dataset=row["dataset"],
                    score=float(row["score"]),
                    stats_path=row["stats_path"],
                    descriptor_text=row["descriptor_text"],
                    summary=row["summary"],
                )
                for row in rows
            ]

        entries = self.catalog_index.get("entries") or []
        if not entries:
            return []

        emb = self.catalog_embeddings
        q = np.asarray(query_embedding, dtype=np.float32)
        scores = cosine_similarity_matrix(q, emb)

        if scores.size == 0:
            return []

        k = max(1, min(int(k), len(entries)))
        top_idx = np.argsort(scores)[::-1][:k]

        results: List[CatalogSearchResult] = []
        for idx in top_idx:
            entry = entries[int(idx)]
            results.append(
                CatalogSearchResult(
                    dataset=entry["dataset"],
                    score=float(scores[int(idx)]),
                    stats_path=entry["stats_path"],
                    descriptor_text=entry["descriptor_text"],
                    summary=entry["summary"],
                )
            )
        return results