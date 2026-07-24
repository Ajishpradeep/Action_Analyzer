"""Multimodal embedding backends for the anti-fraud / verification layer.

This is an *opt-in* stage. It powers two features (see pipeline/fingerprint.py):
  1. Semantic replay/duplicate detection — a far stronger fingerprint than the
     average-hash in fraud.py (which a re-crop or a new camera angle defeats).
  2. An independent "second opinion" on the verdict — an embedding nearest-centroid
     check against labeled exemplars, instead of trusting the VLM's self-reported
     confidence.

Backend: Google's Gemini Embedding (cloud). This RELAXES the project's fully-local
hard requirement, so it is strictly opt-in: with no GEMINI_API_KEY set, `get_embedder()`
returns None and the pipeline falls back to the local-only average-hash path.

Attribution: the GeminiEmbedder (rate limiter, retry/backoff, embed_content calls) is
adapted from SentrySearch (github.com/ssrajadh/sentrysearch), Apache License 2.0.
We use image + text embeddings only; the original also embeds video chunks.
"""

from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from collections import deque

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001 - dotenv is optional
    pass

# gemini-embedding-2 supports Matryoshka dims; 768 matches SentrySearch and is plenty.
EMBED_MODEL = os.environ.get("ECO_EMBED_MODEL", "gemini-embedding-2-preview")
DIMENSIONS = 768
DEFAULT_RPM = 55
# Symmetric task type: we compare image-to-image and image-to-exemplar, not query→doc.
SIMILARITY_TASK = "SEMANTIC_SIMILARITY"


class BaseEmbedder(ABC):
    """Minimal embedding interface (adapted from SentrySearch's BaseEmbedder)."""

    @abstractmethod
    def embed_image(self, image_path: str) -> list[float]:
        ...

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def dimensions(self) -> int:
        ...


class _RateLimiter:
    """Sliding-window request rate limiter (from SentrySearch)."""

    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


class EmbeddingError(RuntimeError):
    """Any failure talking to the embedding backend."""


def _retry(fn, *, max_retries: int = 4, initial_delay: float = 2.0, max_delay: float = 30.0):
    """Exponential back-off on transient (429/503) errors (from SentrySearch)."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status in (429, 503) or "resource exhausted" in msg \
                or "503" in msg or "429" in msg
            if not retryable or attempt == max_retries:
                raise EmbeddingError(str(exc)) from exc
            wait = min(delay, max_delay)
            print(f"  [embed] retryable error (attempt {attempt + 1}), waiting {wait:.0f}s: {exc}",
                  file=sys.stderr)
            time.sleep(wait)
            delay *= 2


_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/heic", ".heif": "image/heif",
}


class GeminiEmbedder(BaseEmbedder):
    """Gemini Embedding backend (cloud). Adapted from SentrySearch's GeminiEmbedder."""

    def __init__(self):
        from google import genai  # imported lazily so the dep is optional

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        self._limiter = _RateLimiter()

    def embed_image(self, image_path: str) -> list[float]:
        from google.genai import types

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        ext = os.path.splitext(image_path)[1].lower()
        mime = _MIME.get(ext)
        if mime is None:
            raise ValueError(f"Unsupported image type {ext!r}. Accepts: {', '.join(sorted(_MIME))}")
        with open(image_path, "rb") as f:
            data = f.read()
        if hasattr(types.Part, "from_bytes"):
            part = types.Part.from_bytes(data=data, mime_type=mime)
        else:
            part = types.Part(inline_data=types.Blob(data=data, mime_type=mime))

        self._limiter.wait()
        resp = _retry(lambda: self._client.models.embed_content(
            model=EMBED_MODEL,
            contents=types.Content(parts=[part]),
            config=types.EmbedContentConfig(
                task_type=SIMILARITY_TASK, output_dimensionality=DIMENSIONS),
        ))
        return list(resp.embeddings[0].values)

    def embed_text(self, text: str) -> list[float]:
        from google.genai import types

        self._limiter.wait()
        resp = _retry(lambda: self._client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=SIMILARITY_TASK, output_dimensionality=DIMENSIONS),
        ))
        return list(resp.embeddings[0].values)

    def dimensions(self) -> int:
        return DIMENSIONS


def get_embedder() -> BaseEmbedder | None:
    """Return a GeminiEmbedder if a key is configured, else None (local-only fallback).

    This is the single switch that keeps the cloud dependency opt-in: no key → the
    whole embedding layer is skipped and fraud.py's average-hash path is used instead.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        return GeminiEmbedder()
    except Exception as e:  # noqa: BLE001
        print(f"  [embed] embedder unavailable, falling back to local-only: {e}", file=sys.stderr)
        return None
