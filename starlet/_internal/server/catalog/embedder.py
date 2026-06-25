from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence
import json
import logging
import math
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_ENV_KEY = "GEMINI_API_KEY"
_DEFAULT_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
_GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent?key={key}"
)


def normalize_vector(values: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm == 0:
        return [0.0 for _ in values]
    return [float(v) / norm for v in values]


class TextEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_document(self, text: str, title: Optional[str] = None) -> List[float]:
        raise NotImplementedError

    def embed_documents(
        self,
        texts: Iterable[str],
        titles: Optional[Iterable[Optional[str]]] = None,
    ) -> List[List[float]]:
        title_list = list(titles) if titles is not None else None
        out: List[List[float]] = []
        for i, text in enumerate(texts):
            title = title_list[i] if title_list is not None and i < len(title_list) else None
            out.append(self.embed_document(text, title=title))
        return out


class GeminiTextEmbedder(TextEmbedder):
    """
    Gemini text embedder using the official embedContent REST endpoint.

    Defaults:
      - model: gemini-embedding-001
      - query task type: RETRIEVAL_QUERY
      - document task type: RETRIEVAL_DOCUMENT
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        output_dimensionality: Optional[int] = 1536, #None default is : 3072. Google recommends 768, 1536, or 3072 as output sizes.
    ):
        self._api_key = os.environ.get(_ENV_KEY)
        if not self._api_key:
            raise RuntimeError(
                f"Environment variable {_ENV_KEY} is not set. "
                "It is required for catalogue embeddings."
            )
        self._model = model
        self._output_dimensionality = output_dimensionality

    @property
    def model_name(self) -> str:
        return self._model

    def embed_query(self, text: str) -> List[float]:
        return self._embed(
            text=text,
            task_type="RETRIEVAL_QUERY",
            title=None,
        )

    def embed_document(self, text: str, title: Optional[str] = None) -> List[float]:
        return self._embed(
            text=text,
            task_type="RETRIEVAL_DOCUMENT",
            title=title,
        )

    def _embed(
        self,
        text: str,
        task_type: str,
        title: Optional[str],
    ) -> List[float]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        url = _GEMINI_EMBED_URL.format(model=self._model, key=self._api_key)

        payload = {
            "model": f"models/{self._model}",
            "content": {
                "parts": [
                    {"text": text}
                ]
            },
            "taskType": task_type,
        }

        if title and task_type == "RETRIEVAL_DOCUMENT":
            payload["title"] = title

        if self._output_dimensionality is not None:
            payload["outputDimensionality"] = int(self._output_dimensionality)

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            logger.error("Gemini embedding HTTP %s: %s", exc.code, detail)
            raise RuntimeError(
                f"Gemini embedding API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.error("Gemini embedding network error: %s", exc.reason)
            raise RuntimeError(
                f"Gemini embedding network error: {exc.reason}"
            ) from exc

        try:
            values = data["embedding"]["values"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Gemini embedding response shape: %s", data)
            raise RuntimeError(
                f"Could not parse Gemini embedding response: {exc}"
            ) from exc

        return normalize_vector([float(v) for v in values])