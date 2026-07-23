"""Rules engine + deterministic reward calculator.

The LLM never computes reward or renders the final verdict. It only observes.
This module turns the observation into a pass/fail decision and a number, using
the flat `eco_rules.json` knowledge base.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rules", "eco_rules.json")


@dataclass
class Decision:
    action_id: str
    display_name: str
    verified: bool = False
    points: int = 0
    ntd: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)   # why pass/fail
    educational: str = ""
    needs_review: bool = False


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _any_match(candidates: List[str], targets: List[str]) -> bool:
    """True if any target string appears within any candidate string (substring, both ways)."""
    cand = [c.lower() for c in candidates]
    for t in targets:
        t = t.lower()
        for c in cand:
            if t in c or c in t:
                return True
    return False


def _compute_reward(rule: dict, obs: dict) -> tuple[int, float, Optional[str]]:
    r = rule["reward"]
    kind = r["type"]

    if kind == "flat":
        return int(r["points"]), float(r.get("ntd", 0)), None

    if kind == "per_item":
        count = obs.get(r.get("count_field", "item_count"))
        if not count or count < 1:
            count = r.get("default_count", 1)
        count = int(count)
        return count * int(r["points_per_item"]), count * float(r.get("ntd_per_item", 0)), \
            f"{count} item(s) counted"

    if kind == "per_weight":
        w = obs.get(rule.get("weight_field", "scale_reading_kg"))
        if w is None:
            # Reverse-vending machines (e.g. ecoco) weigh internally and never show
            # a scale to the camera. If the rule declares a fallback drop-off reward,
            # honor the verified action with the base reward instead of zero.
            fb = r.get("fallback_flat_points")
            if fb is not None:
                return int(fb), float(r.get("fallback_ntd", 0)), \
                    "no scale weight visible — awarding base drop-off reward"
            return 0, 0.0, "no scale weight was readable, so no weight-based reward"
        unit = float(r["unit_kg"])
        units = w / unit
        units = math.floor(units) if r.get("round") == "down" else round(units)
        units = int(units)
        pts = units * int(r["points_per_unit"])
        ntd = units * float(r.get("ntd_per_unit", 0))
        return pts, ntd, f"{w} kg → {units} × {unit} kg increments"

    return 0, 0.0, "unknown reward type"


def evaluate(action_id: str, obs: dict, rules: dict) -> Decision:
    """Evaluate one observation against one action's rule."""
    if action_id not in rules:
        return Decision(action_id, action_id, verified=False,
                        reasons=[f"No rule defined for '{action_id}'."])

    rule = rules[action_id]
    dec = Decision(action_id=action_id, display_name=rule.get("display_name", action_id),
                   confidence=float(obs.get("confidence", 0.0)),
                   educational=rule.get("educational", ""))

    objs = obs.get("detected_objects", [])
    markers = obs.get("location_markers", [])

    # 1. Required object present?
    req_obj = rule.get("required_objects_any", [])
    if req_obj and not _any_match(objs, req_obj):
        dec.verified = False
        dec.reasons.append(
            f"Expected to see one of {req_obj} but detected {objs or 'nothing relevant'}."
        )
        return dec
    dec.reasons.append(f"Recognized a valid item ({', '.join(objs) or 'ok'}).")

    # 2. Location marker (usually optional in the PoC).
    req_mark = rule.get("required_markers_any", [])
    if req_mark:
        if _any_match(markers, req_mark):
            dec.reasons.append(f"Location cue matched ({', '.join(markers)}).")
        elif rule.get("markers_optional", True):
            dec.reasons.append("No location cue seen (optional) — proceeding.")
        else:
            dec.verified = False
            dec.reasons.append(f"Required location cue {req_mark} not visible.")
            return dec

    # 3. Reject conditions.
    for cond in rule.get("reject_if", []):
        if str(obs.get(cond["field"])).lower() == str(cond["equals"]).lower():
            dec.verified = False
            dec.reasons.append(cond.get("message", f"Rejected: {cond['field']} = {cond['equals']}."))
            return dec

    # 4. Safety checks (conditional on object type).
    for chk in rule.get("safety_checks", []):
        only_if = chk.get("only_if_object_any")
        applies = True if not only_if else _any_match(objs, only_if)
        if applies:
            val = obs.get(chk["field"])
            if val != chk["must_be"]:
                dec.verified = False
                dec.reasons.append(chk.get("fail_message",
                                           f"Safety check failed: {chk['field']} must be {chk['must_be']}."))
                return dec
            dec.reasons.append("Safety check passed.")

    # 5. Confidence floor.
    min_conf = float(rule.get("min_confidence", 0.5))
    if dec.confidence < min_conf:
        dec.verified = False
        dec.needs_review = True
        dec.reasons.append(
            f"Model confidence {dec.confidence:.2f} below threshold {min_conf:.2f} — routed to review."
        )
        return dec

    # 6. Passed → reward.
    pts, ntd, note = _compute_reward(rule, obs)
    if note:
        dec.reasons.append(note)
    dec.points = pts
    dec.ntd = ntd
    dec.verified = pts > 0 or rule["reward"]["type"] == "flat"
    if not dec.verified:
        dec.reasons.append("Verified action but reward computed to zero.")
    return dec
