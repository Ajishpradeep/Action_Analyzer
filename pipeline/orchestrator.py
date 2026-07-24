"""Ties the pipeline together: media -> open-set VLM -> deterministic score -> fraud -> result.

No action is selected anywhere — the VLM decides what the action is and how eco-friendly
it is. `run_stream` yields human-readable stage labels as it works (so the UI can show live
progress), then the final result. `run` is a thin wrapper that just returns the result.

Anti-replay (semantic history check + session duplicate) is gated by `use_replay`: when
off, no history is consulted and nothing is recorded.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Set, Tuple

from . import media as media_mod
from . import vlm as vlm_mod
from . import fraud as fraud_mod
from . import fingerprint as fp_mod
from .embeddings import BaseEmbedder
from .rules_engine import Decision, load_rules, score_action
from .vector_store import VectorStore


@dataclass
class PipelineResult:
    decision: Optional[Decision]
    fraud: Optional[fraud_mod.FraudReport]
    observation: dict = field(default_factory=dict)
    error: Optional[str] = None
    frame_paths: List[str] = field(default_factory=list)
    replay: Optional[fp_mod.ReplayResult] = None


def run_stream(media_path: str, seen_hashes: Set[str],
               model: str = vlm_mod.DEFAULT_MODEL, rules: Optional[dict] = None,
               embedder: Optional[BaseEmbedder] = None,
               store: Optional[VectorStore] = None,
               user_id: str = "demo", use_replay: bool = True
               ) -> Iterator[Tuple[str, object]]:
    """Yield ("stage", label) updates as work proceeds, then ("result", PipelineResult)."""
    if rules is None:
        rules = load_rules()
    tmp = tempfile.mkdtemp(prefix="ecoreward_")

    # 1. Media prep.
    yield "stage", "Reading your photo / video"
    try:
        prepared = media_mod.prepare(media_path, tmp)
    except Exception as e:  # noqa: BLE001
        yield "result", PipelineResult(decision=None, fraud=None, error=f"Media error: {e}")
        return

    # 2. Perception (open-set brain).
    yield "stage", "Understanding what you're doing"
    obs = vlm_mod.perceive(prepared.frame_paths, "", model=model)
    if obs.get("_error"):
        yield "result", PipelineResult(decision=None, fraud=None, observation=obs,
                                       error=obs["_error"], frame_paths=prepared.frame_paths)
        return

    # 3. Anti-replay (only when the toggle is on and an embedder is available).
    replay: Optional[fp_mod.ReplayResult] = None
    embedding: Optional[list] = None
    replay_active = use_replay and embedder is not None and store is not None
    if replay_active:
        yield "stage", "Anti-replay check"
        embedding = fp_mod.embed_frame(embedder, prepared.frame_paths)
        if embedding is not None:
            replay = fp_mod.replay_check(store, embedding, user_id)

    # 4. Fraud + deterministic score.
    yield "stage", "Scoring your eco-action"
    # When anti-replay is off, use a throwaway seen-set so no session history is consulted.
    active_seen = seen_hashes if use_replay else set()
    fraud = fraud_mod.check(
        prepared.frame_paths, obs, "",
        is_video=prepared.is_video, motion_score=prepared.motion_score,
        seen_hashes=active_seen,
    )
    if replay is not None and replay.is_duplicate:
        fraud.passed = False
        prior = replay.prior or {}
        fraud.flags.append(
            f"Anti-replay: {replay.score:.0%} similar to a previous submission "
            f"({prior.get('action_label', '?')} on {prior.get('recorded_at', '?')}) "
            f"— looks like the same action submitted again."
        )

    decision = score_action(obs, rules)

    if not fraud.passed and decision.verified:
        decision.verified = False
        decision.points = 0
        decision.ntd = 0.0
        decision.reasons.append("Reward withheld: failed anti-fraud checks.")

    # 5. Record accepted submissions for future replay checks (only when enabled).
    if replay_active and decision.verified and embedding is not None:
        fp_mod.record_submission(store, embedding, user_id, decision.action_label)

    yield "result", PipelineResult(decision=decision, fraud=fraud, observation=obs,
                                   frame_paths=prepared.frame_paths, replay=replay)


def run(media_path: str, seen_hashes: Set[str], model: str = vlm_mod.DEFAULT_MODEL,
        rules: Optional[dict] = None, embedder: Optional[BaseEmbedder] = None,
        store: Optional[VectorStore] = None, user_id: str = "demo",
        use_replay: bool = True) -> PipelineResult:
    """Run the whole pipeline and return the final result (drains run_stream)."""
    result: Optional[PipelineResult] = None
    for kind, payload in run_stream(media_path, seen_hashes, model=model, rules=rules,
                                    embedder=embedder, store=store, user_id=user_id,
                                    use_replay=use_replay):
        if kind == "result":
            result = payload  # type: ignore[assignment]
    return result  # type: ignore[return-value]
