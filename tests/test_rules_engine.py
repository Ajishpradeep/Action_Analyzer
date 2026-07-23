"""Offline tests for the rules engine + reward calculator. No GPU / Ollama needed.

Run:  python -m pytest tests/ -q     (or)     python tests/test_rules_engine.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rules_engine import evaluate, load_rules

RULES = load_rules()


def obs(**kw):
    base = {
        "detected_objects": [], "location_markers": [], "ocr_text": [],
        "scale_reading_kg": None, "item_count": None, "cleanliness": "unknown",
        "lithium_terminal_taped": None, "reusable_cup_present": False,
        "confidence": 0.9,
    }
    base.update(kw)
    return base


def test_battery_weight_reward_rounds_down():
    d = evaluate("battery_recycling",
                 obs(detected_objects=["dry battery"], location_markers=["7-eleven"],
                     scale_reading_kg=1.3), RULES)
    assert d.verified
    # 1.3 kg / 0.5 = 2.6 -> floor 2 units * 1100 pts = 2200
    assert d.points == 2200
    assert abs(d.ntd - 22.0) < 1e-6


def test_battery_no_weight_uses_fallback_reward():
    # Reverse-vending machine (ecoco): a valid battery drop-off with no visible scale.
    # Should still verify and pay the JSON-declared fallback flat reward, not zero.
    d = evaluate("battery_recycling",
                 obs(detected_objects=["battery recycling machine"],
                     location_markers=["ecoco"], scale_reading_kg=None), RULES)
    assert d.verified
    assert d.points == 500
    assert abs(d.ntd - 5.0) < 1e-6


def test_battery_weight_beats_fallback():
    # When a scale reading IS present, per-weight reward is used (not the fallback).
    d = evaluate("battery_recycling",
                 obs(detected_objects=["dry battery"], scale_reading_kg=1.3), RULES)
    assert d.verified and d.points == 2200


def test_lithium_requires_tape():
    d = evaluate("battery_recycling",
                 obs(detected_objects=["lithium battery"], scale_reading_kg=0.5,
                     lithium_terminal_taped=False), RULES)
    assert not d.verified
    assert any("tape" in r.lower() for r in d.reasons)


def test_dirty_paper_container_rejected():
    d = evaluate("paper_container_css",
                 obs(detected_objects=["bento box"], cleanliness="dirty"), RULES)
    assert not d.verified


def test_clean_paper_container_flat_reward():
    d = evaluate("paper_container_css",
                 obs(detected_objects=["paper container"], cleanliness="clean"), RULES)
    assert d.verified and d.points == 300


def test_pet_bottle_count():
    d = evaluate("pet_bottle_recycling",
                 obs(detected_objects=["pet bottle"], location_markers=["recycling bin"],
                     item_count=3, cleanliness="clean"), RULES)
    assert d.verified and d.points == 600


def test_low_confidence_needs_review():
    d = evaluate("reusable_cup",
                 obs(detected_objects=["reusable cup"], location_markers=["counter"],
                     confidence=0.2), RULES)
    assert d.needs_review and not d.verified


def test_wrong_object_fails():
    d = evaluate("reusable_cup",
                 obs(detected_objects=["pet bottle"], confidence=0.9), RULES)
    assert not d.verified


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
