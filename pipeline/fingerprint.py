"""Semantic replay / duplicate-farming detection, built on embeddings.py + vector_store.py.

Embed the submission's representative frame and compare (cosine) against this user's
accepted history. Catches the same action resubmitted for a second reward — even
re-cropped, re-compressed, or shot from a new angle, which the average-hash in fraud.py
misses. Action-agnostic: it fingerprints whatever is in the frame, with no notion of
predefined action types.

No-op when no embedder is configured (no GEMINI_API_KEY) — the caller keeps the local-only
average-hash path.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .embeddings import BaseEmbedder
from .vector_store import DEFAULT_STORE_PATH, Match, VectorStore

# Cosine-similarity ceiling above which two submissions are "the same event".
# Gemini semantic-similarity puts near-identical shots >0.95 and same-scene/new-angle
# ~0.88-0.95; 0.90 flags re-submissions without tripping on genuinely different actions.
REPLAY_THRESHOLD = float(os.environ.get("ECO_REPLAY_THRESHOLD", "0.90"))


@dataclass
class ReplayResult:
    checked: bool = False          # did we actually run (embedder available)?
    is_duplicate: bool = False
    score: float = 0.0             # similarity to the closest prior submission
    prior: Optional[dict] = None   # metadata of the matched prior submission


def _submission_id(embedding: list[float]) -> str:
    return hashlib.sha256(np.asarray(embedding, dtype=np.float32).tobytes()).hexdigest()[:16]


def replay_check(store: VectorStore, embedding: list[float], user_id: str,
                 threshold: float = REPLAY_THRESHOLD) -> ReplayResult:
    """Compare against this user's prior accepted submissions. Does not record."""
    match: Optional[Match] = store.most_similar(embedding, where={"user_id": user_id})
    if match is None:
        return ReplayResult(checked=True, is_duplicate=False, score=0.0)
    return ReplayResult(checked=True, is_duplicate=match.score >= threshold,
                        score=match.score, prior=match.metadata)


def record_submission(store: VectorStore, embedding: list[float], user_id: str,
                      action_label: str) -> str:
    """Persist an accepted submission's fingerprint so future replays are caught."""
    sid = _submission_id(embedding)
    store.add(sid, embedding, {
        "user_id": user_id, "action_label": action_label,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return sid


def embed_frame(embedder: BaseEmbedder, frame_paths: list[str]) -> Optional[list[float]]:
    """Embed the representative (first/sharpest) frame; None on failure."""
    if not frame_paths:
        return None
    try:
        return embedder.embed_image(frame_paths[0])
    except Exception:  # noqa: BLE001 - never let fingerprinting break the main verdict
        return None


def default_store() -> VectorStore:
    return VectorStore(DEFAULT_STORE_PATH)
