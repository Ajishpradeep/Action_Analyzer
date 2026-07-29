"""Perception stage: call Qwen2.5-VL through Ollama and return a validated dict.

We ask for JSON and parse defensively — local models occasionally wrap JSON in
prose or code fences despite instructions.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

import ollama

from .prompts import (
    EXTRACTION_SCHEMA_KEYS,
    SYSTEM_PROMPT,
    build_user_prompt,
)

# Verified against the live Ollama library (see docs/BLUEPRINT.md §2).
# qwen3-vl:8b — 6.1GB, 256K ctx, best OCR (32 langs incl. Chinese) + native video.
# Requires Ollama >= 0.12.7. Swap freely; every model here is model-agnostic.
DEFAULT_MODEL = "qwen3-vl:8b"

# Ollama defaults to a 4096-token context, which a single image (~3.5-4k image
# tokens) plus our prompt already overflows -> "exceed_context_size_error".
# We size num_ctx from the number of frames, and retry once (bigger) if the
# server still says the prompt doesn't fit. Override with ECO_NUM_CTX.
_CTX_PER_FRAME = 4096
_CTX_OVERHEAD = 2048     # system + user prompt + JSON answer headroom
_CTX_MIN = 8192
_CTX_MAX = 32768


def _round_ctx(n: int) -> int:
    """Round up to a 2048 multiple inside [_CTX_MIN, _CTX_MAX] (keeps the KV
    cache no bigger than it needs to be — matters on a 16 GB Mac)."""
    return max(_CTX_MIN, min(_CTX_MAX, ((n + 2047) // 2048) * 2048))


def num_ctx_for(n_frames: int, min_tokens: int = 0) -> int:
    """Context window to request for `n_frames` images (env override wins)."""
    override = os.getenv("ECO_NUM_CTX")
    if override:
        try:
            return max(1024, int(override))
        except ValueError:
            pass
    need = _CTX_OVERHEAD + _CTX_PER_FRAME * max(1, n_frames)
    return _round_ctx(max(need, min_tokens))


def _needed_tokens(err: str) -> int:
    """Pull `n_prompt_tokens` out of an Ollama context-overflow error, if present."""
    m = re.search(r'"?n_prompt_tokens"?\s*[:=]\s*(\d+)', err)
    if not m:
        m = re.search(r"\((\d+) tokens\) exceeds", err)
    return int(m.group(1)) if m else 0


def _is_ctx_error(err: str) -> bool:
    e = err.lower()
    return "exceed_context_size" in e or "context size" in e or "context length" in e


_DEFAULTS = {
    # what the action is
    "scene_description": "",
    "action_label": "",
    # how eco-friendly it is
    "is_eco_action": False,
    "eco_relevance": 0.0,
    "eco_rationale": "",
    # reward suggestion (clamped in Python)
    "suggested_points": 0,
    "suggested_reasoning": "",
    # evidence
    "detected_objects": [],
    "location_markers": [],
    "ocr_text": [],
    "item_count": None,
    "cleanliness": "unknown",
    "challenge_code_visible": None,
    "confidence": 0.0,
    "notes": "",
}


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a possibly-messy model response."""
    text = text.strip()
    # Strip code fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Otherwise grab the outermost {...}.
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    return json.loads(text)


def _coerce(raw: dict) -> dict:
    """Fill missing keys and coerce obvious type issues."""
    out = dict(_DEFAULTS)
    if isinstance(raw, dict):
        for k in EXTRACTION_SCHEMA_KEYS:
            if k in raw and raw[k] is not None:
                out[k] = raw[k]
    # Light coercions.
    for list_key in ("detected_objects", "location_markers", "ocr_text"):
        v = out[list_key]
        if isinstance(v, str):
            out[list_key] = [v]
        elif not isinstance(v, list):
            out[list_key] = []
        out[list_key] = [str(x).lower().strip() for x in out[list_key]]
    for float_key in ("confidence", "eco_relevance"):
        try:
            out[float_key] = max(0.0, min(1.0, float(out[float_key])))
        except (TypeError, ValueError):
            out[float_key] = 0.0
    if out["item_count"] is not None:
        try:
            out["item_count"] = int(out["item_count"])
        except (TypeError, ValueError):
            out["item_count"] = None
    if out["cleanliness"] not in ("clean", "dirty", "leaking", "unknown"):
        out["cleanliness"] = "unknown"

    # Open-set brain fields.
    out["is_eco_action"] = bool(out["is_eco_action"])
    try:
        out["suggested_points"] = max(0, int(round(float(out["suggested_points"]))))
    except (TypeError, ValueError):
        out["suggested_points"] = 0
    for str_key in ("scene_description", "action_label", "suggested_reasoning", "eco_rationale"):
        out[str_key] = "" if out[str_key] is None else str(out[str_key])
    return out


def perceive(frame_paths: List[str], challenge_code: str,
             model: str = DEFAULT_MODEL) -> dict:
    """Run the VLM over one or more frames and return the coerced extraction dict.

    Fully open-set: no action is supplied; the brain decides what the action is and how
    eco-friendly it is. Adds `_raw` (the model text) and `_error` (str or None).
    """
    user_prompt = build_user_prompt(challenge_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt, "images": frame_paths},
    ]
    n_ctx = num_ctx_for(len(frame_paths))

    def _call(ctx: int):
        return ollama.chat(
            model=model,
            messages=messages,
            format="json",          # ask Ollama to constrain to JSON
            options={"temperature": 0.1, "num_ctx": ctx},
        )

    try:
        try:
            resp = _call(n_ctx)
        except Exception as e:  # noqa: BLE001 - one retry if the prompt still didn't fit
            err = str(e)
            retry_ctx = num_ctx_for(len(frame_paths),
                                    min_tokens=_needed_tokens(err) + _CTX_OVERHEAD)
            if not _is_ctx_error(err) or retry_ctx <= n_ctx:
                raise
            n_ctx = retry_ctx
            resp = _call(n_ctx)
        text = resp["message"]["content"]
    except Exception as e:  # noqa: BLE001 - surface any Ollama/model error to UI
        result = dict(_DEFAULTS)
        result["_raw"] = ""
        result["_error"] = f"VLM call failed (num_ctx={n_ctx}): {e}"
        return result

    try:
        parsed = _extract_json(text)
        result = _coerce(parsed)
        result["_raw"] = text
        result["_error"] = None
    except Exception as e:  # noqa: BLE001
        result = dict(_DEFAULTS)
        result["_raw"] = text
        result["_error"] = f"Could not parse model JSON: {e}"
    return result
