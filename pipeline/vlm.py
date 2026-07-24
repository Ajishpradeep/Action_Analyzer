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
