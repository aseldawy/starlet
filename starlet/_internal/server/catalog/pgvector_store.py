from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class PgVectorConfig:
    host: str = os.environ.get("PGVECTOR_HOST", "localhost")
    port: int = int(os.environ.get("PGVECTOR_PORT", "5432"))
    dbname: str = os.environ.get("PGVECTOR_DB", "postgres")
    user: str = os.environ.get("PGVECTOR_USER", "postgres")
    password: str = os.environ.get("PGVECTOR_PASSWORD", "")
    table_name: str = os.environ.get("PGVECTOR_TABLE", "dataset_catalog_embeddings")
    sslmode: str = os.environ.get("PGVECTOR_SSLMODE", "prefer")


class PgVectorStore:
    """
    Minimal pgvector-backed store for dataset catalogue search.

    Requires:
      pip install psycopg[binary] pgvector
    and a PostgreSQL database with the pgvector extension available.
    """

    def __init__(self, config: Optional[PgVectorConfig] = None):
        self.config = config or PgVectorConfig()
        self._conn = None

    def _connect(self):
        if self._conn is not None:
            return self._conn

        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "pgvector backend requires psycopg. Install with: pip install psycopg[binary]"
            ) from exc

        try:
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError(
                "pgvector backend requires pgvector python package. Install with: pip install pgvector"
            ) from exc

        self._conn = psycopg.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.dbname,
            user=self.config.user,
            password=self.config.password,
            sslmode=self.config.sslmode,
            autocommit=True,
        )
        register_vector(self._conn)
        return self._conn

    def ensure_schema(self, embedding_dim: int):
        conn = self._connect()
        table = self.config.table_name

        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    dataset TEXT PRIMARY KEY,
                    stats_path TEXT NOT NULL,
                    descriptor_text TEXT NOT NULL,
                    summary_json JSONB NOT NULL,
                    embedding vector({int(embedding_dim)}) NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx "
                f"ON {table} USING hnsw (embedding vector_cosine_ops);"
            )

    def upsert_many(self, rows: List[Dict[str, Any]]):
        if not rows:
            return

        first_embedding = rows[0].get("embedding") or []
        if not first_embedding:
            raise ValueError("Cannot initialize pgvector table with empty embeddings")

        self.ensure_schema(len(first_embedding))
        conn = self._connect()
        table = self.config.table_name

        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    f"""
                    INSERT INTO {table}
                        (dataset, stats_path, descriptor_text, summary_json, embedding, updated_at)
                    VALUES
                        (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (dataset) DO UPDATE SET
                        stats_path = EXCLUDED.stats_path,
                        descriptor_text = EXCLUDED.descriptor_text,
                        summary_json = EXCLUDED.summary_json,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW();
                    """,
                    (
                        row["dataset"],
                        row["stats_path"],
                        row["descriptor_text"],
                        json.dumps(row["summary"]),
                        row["embedding"],
                    ),
                )

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not query_embedding:
            return []

        conn = self._connect()
        table = self.config.table_name

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    dataset,
                    stats_path,
                    descriptor_text,
                    summary_json,
                    1 - (embedding <=> %s::vector) AS score
                FROM {table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, query_embedding, int(k)),
            )
            rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for dataset, stats_path, descriptor_text, summary_json, score in rows:
            if isinstance(summary_json, str):
                summary_json = json.loads(summary_json)
            results.append({
                "dataset": dataset,
                "score": float(score),
                "stats_path": stats_path,
                "descriptor_text": descriptor_text,
                "summary": summary_json,
            })
        return results

    def ping(self) -> bool:
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
            return True
        except Exception as exc:
            logger.warning("pgvector ping failed: %s", exc)
            return False