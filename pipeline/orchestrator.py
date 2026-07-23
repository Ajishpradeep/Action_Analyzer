"""Ties the pipeline together: media -> VLM -> rules -> reward -> fraud -> result."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Set

from . import media as media_mod
from . import vlm as vlm_mod
from . import fraud as fraud_mod
from .rules_engine import Decision, evaluate, load_rules


@dataclass
class PipelineResult:
    decision: Optional[Decision]
    fraud: Optional[fraud_mod.FraudReport]
    observation: dict = field(default_factory=dict)
    error: Optional[str] = None
    frame_paths: List[str] = field(default_factory=list)


def run(media_path: str, action_id: str, action_hint: str, challenge_code: str,
        seen_hashes: Set[str], model: str = vlm_mod.DEFAULT_MODEL,
        rules: Optional[dict] = None) -> PipelineResult:
    if rules is None:
        rules = load_rules()

    tmp = tempfile.mkdtemp(prefix="ecoreward_")

    # 1. Media prep.
    try:
        prepared = media_mod.prepare(media_path, tmp)
    except Exception as e:  # noqa: BLE001
        return PipelineResult(decision=None, fraud=None, error=f"Media error: {e}")

    # 2. Perception.
    obs = vlm_mod.perceive(prepared.frame_paths, action_hint, challenge_code, model=model)
    if obs.get("_error"):
        return PipelineResult(decision=None, fraud=None, observation=obs,
                              error=obs["_error"], frame_paths=prepared.frame_paths)

    # 3. Fraud checks (independent of the rules verdict).
    fraud = fraud_mod.check(
        prepared.frame_paths, obs, challenge_code,
        is_video=prepared.is_video, motion_score=prepared.motion_score,
        seen_hashes=seen_hashes,
    )

    # 4. Rules + reward.
    decision = evaluate(action_id, obs, rules)

    # Fraud overrides a positive verdict.
    if not fraud.passed and decision.verified:
        decision.verified = False
        decision.points = 0
        decision.ntd = 0.0
        decision.reasons.append("Reward withheld: failed anti-fraud checks.")

    return PipelineResult(decision=decision, fraud=fraud, observation=obs,
                          frame_paths=prepared.frame_paths)
