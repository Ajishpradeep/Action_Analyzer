"""Empirically compare VLMs on YOUR data, so model choice isn't a matter of trust.

It runs each candidate Ollama model over a folder of sample images you label once,
then scores how well each model extracts the fields the rules engine actually needs
(object recognition, OCR of scale/stickers, cleanliness, counts, confidence) plus
latency. Prints a ranked table.

Usage
-----
1. Put sample photos in  samples/<action_id>/*.jpg   e.g. samples/battery_recycling/1.jpg
2. Create samples/labels.json (see samples/labels.example.json) with the ground truth
   for each file: the expected action, expected objects, and any expected scale/count.
3. Pull the models you want to compare, then:

     ollama pull qwen3-vl:8b
     ollama pull qwen3-vl:4b
     ollama pull gemma3:12b
     ollama pull minicpm-v4.5:8b

     uv run python scripts/benchmark_models.py --models qwen3-vl:8b qwen3-vl:4b gemma3:12b minicpm-v4.5:8b

No GPU here in CI — this is meant to run on your Mac where Ollama lives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import vlm as vlm_mod
from pipeline.rules_engine import evaluate, load_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT, "samples")
LABELS_PATH = os.path.join(SAMPLES_DIR, "labels.json")


def _score_one(obs: dict, truth: dict) -> Dict[str, float]:
    """Field-level correctness for one image. Each sub-score is 0..1."""
    s = {}

    # Object recognition: did any expected object appear?
    exp_objs = [o.lower() for o in truth.get("expected_objects", [])]
    got_objs = [o.lower() for o in obs.get("detected_objects", [])]
    if exp_objs:
        hit = any(any(e in g or g in e for g in got_objs) for e in exp_objs)
        s["objects"] = 1.0 if hit else 0.0

    # OCR scale reading (if the sample has a scale).
    if truth.get("expected_scale_kg") is not None:
        got = obs.get("scale_reading_kg")
        s["scale_ocr"] = 1.0 if (got is not None and
                                 abs(float(got) - float(truth["expected_scale_kg"])) <= 0.1) else 0.0

    # Count.
    if truth.get("expected_count") is not None:
        s["count"] = 1.0 if obs.get("item_count") == truth["expected_count"] else 0.0

    # Cleanliness.
    if truth.get("expected_cleanliness"):
        s["cleanliness"] = 1.0 if obs.get("cleanliness") == truth["expected_cleanliness"] else 0.0

    # End-to-end verdict: does the pipeline reach the expected verified state?
    rules = load_rules()
    dec = evaluate(truth["action_id"], obs, rules)
    s["verdict"] = 1.0 if dec.verified == truth.get("expected_verified", True) else 0.0

    return s


def run_model(model: str, labels: List[dict]) -> dict:
    per_field: Dict[str, List[float]] = {}
    latencies = []
    errors = 0

    for item in labels:
        path = os.path.join(SAMPLES_DIR, item["action_id"], item["file"])
        if not os.path.exists(path):
            print(f"  ! missing sample: {path}")
            continue
        t0 = time.time()
        obs = vlm_mod.perceive([path], item.get("action_hint", item["action_id"]),
                               item.get("challenge_code", ""), model=model)
        latencies.append(time.time() - t0)
        if obs.get("_error"):
            errors += 1
            print(f"  ! {model} error on {item['file']}: {obs['_error']}")
            continue
        for k, v in _score_one(obs, item).items():
            per_field.setdefault(k, []).append(v)

    summary = {k: (sum(v) / len(v) if v else None) for k, v in per_field.items()}
    scored = [x for v in per_field.values() for x in v]
    summary["_overall"] = sum(scored) / len(scored) if scored else 0.0
    summary["_avg_latency_s"] = sum(latencies) / len(latencies) if latencies else None
    summary["_errors"] = errors
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["qwen3-vl:8b", "qwen3-vl:4b", "gemma3:12b", "minicpm-v4.5:8b"])
    args = ap.parse_args()

    if not os.path.exists(LABELS_PATH):
        print(f"No labels file at {LABELS_PATH}. See samples/labels.example.json.")
        sys.exit(1)
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    results = {}
    for m in args.models:
        print(f"\n=== {m} ===")
        results[m] = run_model(m, labels)

    # Ranked table.
    fields = sorted({k for r in results.values() for k in r if not k.startswith("_")})
    header = ["model", "overall", "latency_s", "errors"] + fields
    print("\n" + " | ".join(f"{h:>12}" for h in header))
    print("-" * (15 * len(header)))
    for m in sorted(results, key=lambda x: results[x]["_overall"], reverse=True):
        r = results[m]
        row = [m, f"{r['_overall']:.2f}",
               f"{r['_avg_latency_s']:.1f}" if r["_avg_latency_s"] else "-",
               str(r["_errors"])]
        row += [f"{r[f]:.2f}" if r.get(f) is not None else "-" for f in fields]
        print(" | ".join(f"{c:>12}" for c in row))

    print("\nWinner = highest 'overall' with acceptable latency for your UX.")


if __name__ == "__main__":
    main()
