"""Empirically compare VLMs on YOUR data — open-set, no predefined actions.

Runs each candidate Ollama model over a folder of sample media you label once, then
scores how well each model does the open-set job the app needs: correctly deciding
whether something is an eco-action, judging eco-relevance, and reading the scene — plus
latency. Prints a ranked table.

Usage
-----
1. Put sample photos/videos in  samples/  (flat, any filename), e.g. samples/1.jpg
2. Create samples/labels.json (see samples/labels.example.json): for each file, the
   expected_is_eco flag and a short note.
3. Pull the models you want to compare, then:

     ollama pull qwen3-vl:8b
     ollama pull qwen3-vl:4b

     uv run python scripts/benchmark_models.py --models qwen3-vl:8b qwen3-vl:4b

No GPU here in CI — run this on your Mac where Ollama lives.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT, "samples")
LABELS_PATH = os.path.join(SAMPLES_DIR, "labels.json")


def _score_one(obs: dict, truth: dict) -> Dict[str, float]:
    """Open-set correctness for one file. Each sub-score is 0..1."""
    s = {}
    expected = bool(truth.get("expected_is_eco", True))
    # Did the brain get the eco / not-eco call right?
    s["is_eco"] = 1.0 if bool(obs.get("is_eco_action")) == expected else 0.0
    # For genuine eco-actions, reward a decisive eco-relevance; for non-eco, a low one.
    rel = float(obs.get("eco_relevance", 0.0))
    s["relevance"] = (rel if expected else (1.0 - rel))
    # Did it describe the scene at all?
    s["described"] = 1.0 if (obs.get("scene_description") or obs.get("action_label")) else 0.0
    return s


def run_model(model: str, labels: List[dict]) -> dict:
    per_field: Dict[str, List[float]] = {}
    latencies = []
    errors = 0

    for item in labels:
        path = os.path.join(SAMPLES_DIR, item["file"])
        if not os.path.exists(path):
            print(f"  ! missing sample: {path}")
            continue
        t0 = time.time()
        obs = vlm_mod.perceive([path], item.get("challenge_code", ""), model=model)
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
    ap.add_argument("--models", nargs="+", default=["qwen3-vl:8b", "qwen3-vl:4b"])
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
