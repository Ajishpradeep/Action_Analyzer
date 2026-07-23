"""Perception stage: call Qwen2.5-VL through Ollama and return a validated dict.

We ask for JSON and parse defensively — local models occasionally wrap JSON in
prose or code fences despite instructions.
"""

from __future__ import annotations

import json
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

_DEFAULTS = {
    "detected_objects": [],
    "primary_action": None,
    "location_markers": [],
    "ocr_text": [],
    "scale_reading_kg": None,
    "item_count": None,
    "cleanliness": "unknown",
    "lithium_terminal_taped": None,
    "reusable_cup_present": False,
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
    try:
        out["confidence"] = float(out["confidence"])
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    for num_key in ("scale_reading_kg", "item_count"):
        if out[num_key] is not None:
            try:
                out[num_key] = float(out[num_key]) if num_key == "scale_reading_kg" else int(out[num_key])
            except (TypeError, ValueError):
                out[num_key] = None
    if out["cleanliness"] not in ("clean", "dirty", "leaking", "unknown"):
        out["cleanliness"] = "unknown"
    return out


def perceive(frame_paths: List[str], action_hint: str, challenge_code: str,
             model: str = DEFAULT_MODEL) -> dict:
    """Run the VLM over one or more frames and return the coerced extraction dict.

    Adds two bookkeeping keys: `_raw` (the model text) and `_error` (str or None).
    """
    user_prompt = build_user_prompt(action_hint, challenge_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt, "images": frame_paths},
    ]
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            format="json",          # ask Ollama to constrain to JSON
            options={"temperature": 0.1},
        )
        text = resp["message"]["content"]
    except Exception as e:  # noqa: BLE001 - surface any Ollama/model error to UI
        result = dict(_DEFAULTS)
        result["_raw"] = ""
        result["_error"] = f"VLM call failed: {e}"
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
