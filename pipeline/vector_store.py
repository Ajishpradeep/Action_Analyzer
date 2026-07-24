"""A tiny persistent vector store for submission fingerprints.

Intentionally NOT ChromaDB. SentrySearch uses Chroma because it searches thousands
of video chunks; here we do a linear cosine scan over a handful of per-user
submissions, so a numpy + JSON store keeps the PoC small (CLAUDE.md convention) and
dependency-light. The interface (add / most_similar) mirrors SentrySearch's
SentryStore closely enough to swap in Chroma later behind the same methods.

Cosine math is adapted from SentrySearch's highlights.py (Apache License 2.0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_STORE_PATH = Path.home() / ".eco_reward" / "submissions.json"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


@dataclass
class Match:
    id: str
    score: float           # cosine similarity 0..1
    metadata: dict


class VectorStore:
    """Persistent list of (id, embedding, metadata), searched by cosine similarity."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_STORE_PATH)
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []
        self._metas: list[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for row in data.get("items", []):
            self._ids.append(row["id"])
            self._vecs.append(row["embedding"])
            self._metas.append(row.get("metadata", {}))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        items = [{"id": i, "embedding": v, "metadata": m}
                 for i, v, m in zip(self._ids, self._vecs, self._metas)]
        self.path.write_text(json.dumps({"items": items}))

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, id: str, embedding: list[float], metadata: dict, *, persist: bool = True) -> None:
        self._ids.append(id)
        self._vecs.append(list(embedding))
        self._metas.append(dict(metadata))
        if persist:
            self.save()

    def most_similar(self, embedding: list[float], *,
                     where: Optional[dict] = None) -> Optional[Match]:
        """Return the highest-cosine stored item, optionally filtered by metadata equality."""
        if not self._vecs:
            return None
        idxs = range(len(self._vecs))
        if where:
            idxs = [i for i in idxs
                    if all(self._metas[i].get(k) == v for k, v in where.items())]
            if not idxs:
                return None
        idxs = list(idxs)
        mat = _normalize(np.asarray([self._vecs[i] for i in idxs], dtype=np.float32))
        q = _normalize(np.asarray(embedding, dtype=np.float32))
        sims = mat @ q
        best = int(np.argmax(sims))
        gi = idxs[best]
        return Match(id=self._ids[gi], score=float(sims[best]), metadata=dict(self._metas[gi]))
