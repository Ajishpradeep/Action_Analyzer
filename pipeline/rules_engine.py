"""Deterministic scorer for the fully open-set eco brain.

There are NO predefined actions. The VLM judges any action on its own merits and reports
`is_eco_action`, `eco_relevance` (0-1, how eco-friendly it is), and a *suggested* point
value. This module turns that into the final reward deterministically:

    points = clamp(suggested_points, min, max) x eco_relevance x quality_multiplier

so the same observation always yields the same number and the model never sets the payout
(CLAUDE.md #6). All knobs live in `rules/eco_rules.json` -> "scoring".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rules", "eco_rules.json")


@dataclass
class Decision:
    action_label: str = ""
    scene_description: str = ""
    verified: bool = False
    points: int = 0
    ntd: float = 0.0
    confidence: float = 0.0
    eco_relevance: float = 0.0
    reasons: List[str] = field(default_factory=list)
    educational: str = ""
    needs_review: bool = False


def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_action(obs: dict, rules: dict) -> Decision:
    """Score any observed action open-set. `rules` is the loaded eco_rules.json."""
    cfg = rules.get("scoring", {})
    label = obs.get("action_label") or "this action"
    dec = Decision(
        action_label=label,
        scene_description=obs.get("scene_description", ""),
        confidence=float(obs.get("confidence", 0.0)),
        eco_relevance=float(obs.get("eco_relevance", 0.0)),
        educational="Any genuine action that helps the environment earns Green Points — "
                    "the more deliberate and higher-impact, the more it's worth.",
    )

    # 1. Is it an eco action at all, and how close to one?
    min_rel = float(cfg.get("min_eco_relevance", 0.4))
    if not obs.get("is_eco_action") or dec.eco_relevance < min_rel:
        dec.verified = False
        rationale = obs.get("eco_rationale") or "no clear environmental benefit"
        dec.reasons.append(
            f"Not eco-friendly enough: eco-relevance {dec.eco_relevance:.2f} is below the "
            f"{min_rel:.2f} threshold ({rationale})."
        )
        return dec

    # 2. Hazard/contamination gate: leaking items pay nothing (quality multiplier 0).
    quality = float(cfg.get("quality_multipliers", {}).get(str(obs.get("cleanliness")), 1.0))
    if quality <= 0.0:
        dec.verified = False
        dec.reasons.append("Item appears leaking/hazardous — hand it to a safe drop-off, no reward.")
        return dec

    # 3. Confidence floor -> route to review.
    min_conf = float(cfg.get("min_confidence", 0.5))
    if dec.confidence < min_conf:
        dec.verified = False
        dec.needs_review = True
        dec.reasons.append(
            f"Model confidence {dec.confidence:.2f} below threshold {min_conf:.2f} — routed to review."
        )
        return dec

    # 4. Deterministic reward: clamp the suggestion, scale by eco-relevance and quality.
    lo, hi = int(cfg.get("min_points", 50)), int(cfg.get("max_points", 600))
    suggested = int(obs.get("suggested_points") or 0)
    clamped = _clamp(suggested, lo, hi)
    pts = int(round(clamped * dec.eco_relevance * quality))
    dec.points = pts
    dec.ntd = pts / float(cfg.get("points_to_ntd", 100))
    dec.verified = pts > 0

    dec.reasons.append(f"Recognized an eco-friendly action: {label}.")
    dec.reasons.append(
        f"Eco-relevance {dec.eco_relevance:.2f}; brain suggested {suggested} pts → "
        f"clamped to [{lo}-{hi}] = {clamped:.0f}, ×{dec.eco_relevance:.2f} relevance"
        + (f" ×{quality:.2f} quality" if quality != 1.0 else "") + f" = {pts} pts."
    )
    if obs.get("suggested_reasoning"):
        dec.reasons.append(f"Brain's rationale: {obs['suggested_reasoning']}")
    if not dec.verified:
        dec.reasons.append("Recognized an eco-action but the reward computed to zero.")
    return dec
