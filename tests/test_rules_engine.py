"""Offline tests for the open-set deterministic scorer. No GPU / Ollama needed.

Run:  python tests/test_rules_engine.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rules_engine import load_rules, score_action

RULES = load_rules()


def obs(**kw):
    base = {
        "action_label": "an action", "scene_description": "",
        "is_eco_action": True, "eco_relevance": 1.0, "eco_rationale": "",
        "suggested_points": 300, "suggested_reasoning": "",
        "detected_objects": [], "location_markers": [], "ocr_text": [],
        "item_count": None, "cleanliness": "clean", "confidence": 0.9,
    }
    base.update(kw)
    return base


def test_clean_strong_action_rewards():
    d = score_action(obs(eco_relevance=1.0, suggested_points=300, cleanliness="clean"), RULES)
    assert d.verified and d.points == 300 and abs(d.ntd - 3.0) < 1e-6


def test_suggestion_clamped_to_max():
    # 999 suggested → clamped to max_points (600) before scaling.
    d = score_action(obs(eco_relevance=1.0, suggested_points=999), RULES)
    assert d.verified and d.points == 600


def test_suggestion_clamped_to_min_floor():
    # 10 suggested → floored to min_points (50).
    d = score_action(obs(eco_relevance=1.0, suggested_points=10), RULES)
    assert d.verified and d.points == 50


def test_eco_relevance_scales_reward():
    # A weakly eco-friendly action earns proportionally less.
    d = score_action(obs(eco_relevance=0.5, suggested_points=400, cleanliness="clean"), RULES)
    assert d.verified and d.points == 200  # clamp(400)=400 * 0.5


def test_quality_multiplier_applies():
    # dirty → ×0.6.
    d = score_action(obs(eco_relevance=1.0, suggested_points=300, cleanliness="dirty"), RULES)
    assert d.verified and d.points == 180


def test_non_eco_action_not_rewarded():
    d = score_action(obs(is_eco_action=False, eco_relevance=0.0, suggested_points=500), RULES)
    assert not d.verified and d.points == 0


def test_low_relevance_below_threshold_rejected():
    d = score_action(obs(is_eco_action=True, eco_relevance=0.3, suggested_points=300), RULES)
    assert not d.verified and d.points == 0


def test_leaking_item_gets_no_reward():
    d = score_action(obs(eco_relevance=1.0, suggested_points=300, cleanliness="leaking"), RULES)
    assert not d.verified and d.points == 0


def test_low_confidence_needs_review():
    d = score_action(obs(eco_relevance=1.0, suggested_points=300, confidence=0.2), RULES)
    assert d.needs_review and not d.verified


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
