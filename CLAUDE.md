# CLAUDE.md — Eco-Reward PoC (Taiwan)

Context for Claude Code working in this repo. Read this first, then `docs/BLUEPRINT.md`
(architecture + rationale) and `docs/AGENT_TASKS.md` (the ordered build plan).

## What this is

A **fully-local, fully open-set** proof of concept: a Gradio web UI where a user uploads a
photo or short video of **any** action. There are **no predefined actions** — a **local
Vision-Language Model served by Ollama** works out what the action is on its own, judges
how eco-friendly it is (`eco_relevance` 0–1), and *suggests* a reward; a small Python
scorer turns that into a deterministic number. The user never selects an action.

Solo-engineer scope. No cloud APIs (except the opt-in embedding layer, see #1), no model
training, no YOLO. The VLM does open-set perception + judgment; Python does the arithmetic
(the model never sets the final payout).

## Hard requirements (do not drift from these)

0. **Package manager is `uv`.** Use `uv venv`, `uv pip install -r requirements.txt`, and
   `uv run python ...` — never plain `pip`.
1. **Everything runs locally on an Apple Silicon Mac (16 GB+).** No hosted inference.
   *Exception (opt-in, off by default):* the semantic anti-fraud layer
   (`pipeline/embeddings.py` + `fingerprint.py`) uses Gemini embeddings and is enabled
   only when `GEMINI_API_KEY` is set. With no key the app stays 100% local. Don't wire
   any cloud call into the default path.
2. **The VLM must run on the Mac's GPU (Metal), not the CPU.** Ollama uses Metal
   automatically on Apple Silicon — verify with `ollama ps` (should show `100% GPU`).
   If it falls back to CPU, that's a bug to fix (usually model too big for available
   unified memory → use a smaller tag like `qwen3-vl:4b`).
3. **Default model: `qwen3-vl:8b`** via Ollama (6.1 GB, best OCR incl. Chinese + native
   video). Requires Ollama ≥ 0.12.7. Model name must stay configurable (env/UI/arg).
4. **Input: images AND short video.** Video is handled by sampling frames (`pipeline/media.py`).
5. **The deliverable is a working Gradio app** that: takes an upload → runs the local VLM →
   returns a plain-language description of what it sees as an eco-action → a verdict and a
   numeric reward, with the reasoning shown.
6. **Scoring stays decoupled** in `rules/eco_rules.json` (`scoring` block) +
   `pipeline/rules_engine.py`. Tuning the reward = editing JSON, not code.
7. **Fully open-set — no predefined actions anywhere.** Do NOT reintroduce named actions,
   an action taxonomy/categories, or an action picker into prompts, rules, or the UI. The
   VLM judges any action open-set and reports `is_eco_action`, `eco_relevance` (0–1), and a
   *suggested* point value. `rules_engine.score_action` computes the reward deterministically:
   `points = clamp(suggested_points, min, max) × eco_relevance × quality_multiplier`. The
   LLM proposes; Python disposes — the model never sets the final number (keeps #6 intact).

## Architecture (already scaffolded — extend, don't rewrite)

```
app.py                      Gradio UI (no action picker — just upload)
rules/eco_rules.json        `scoring` config: global point band + thresholds + multipliers
pipeline/
  media.py                  image / video-frame sampling + motion (liveness) score
  prompts.py                open-set brain prompt (no predefined actions/categories)
  vlm.py                    Ollama call (DEFAULT_MODEL) + robust JSON parsing
  rules_engine.py           score_action(): deterministic eco_relevance-scaled clamp
  fraud.py                  challenge code · duplicate hash · liveness sanity
  embeddings.py             opt-in Gemini embedder (key-gated); adapted from SentrySearch (Apache-2.0)
  vector_store.py           tiny numpy cosine store for submission fingerprints
  fingerprint.py            semantic replay detection (action-agnostic)
  orchestrator.py           media -> vlm -> score -> fraud -> result
scripts/
  preflight.py              verify Ollama + model are ready (and GPU)
  benchmark_models.py       A/B test candidate VLMs on samples/ (open-set scoring)
samples/                    YOU add test media here, flat (see samples/labels.example.json)
tests/test_rules_engine.py  offline unit tests (no GPU needed)
docs/BLUEPRINT.md           original design + what was cut & why (pre open-set pivot)
docs/AGENT_TASKS.md         ordered action plan for you
```

Data flow: the VLM returns an open-set JSON schema (scene_description, action_label,
is_eco_action, eco_relevance, suggested_points + evidence: objects, OCR text, cleanliness,
count, confidence — see `pipeline/prompts.py`). `rules_engine.score_action()` computes the
deterministic reward from `eco_relevance` + the clamped suggestion. `orchestrator.run()`
wires it all, adds the opt-in embedding layer (semantic replay detection), and lets fraud
checks veto a positive verdict.

## Conventions

- Python 3.11+, standard library + the pinned deps in `requirements.txt`. Keep it small.
- Each pipeline stage is a pure-ish function so it can be unit-tested without a GPU.
- Never let the LLM compute the *final* reward. The model may *suggest* points, but
  `rules_engine.score_action` clamps and scales them deterministically. The model output is
  a suggestion, never the payout.
- Reward figures in `eco_rules.json` are **illustrative**; don't present them as authoritative.
- Don't reintroduce the cut complexity (YOLO, agentic verifiers, anti-spoof models, MoE
  serving) — see BLUEPRINT §1 — nor any predefined action list (see #7).

## Definition of done

`uv run python scripts/preflight.py` passes · `uv run python app.py` serves a Gradio UI ·
uploading any test image returns a sensible open-set description + eco-relevance + reward ·
`ollama ps` shows the model on GPU · `uv run python tests/test_rules_engine.py` passes.

## Test data

There are no bundled sample images — the user supplies their own. Drop media flat into
`samples/` (e.g. `samples/1.jpg`), then copy `samples/labels.example.json` to
`samples/labels.json` and fill in the open-set ground truth (`expected_is_eco` + a note) so
`benchmark_models.py` can score models. `app.py` needs no samples to run.

## Commands

```bash
ollama pull qwen3-vl:8b                       # once
uv venv                                        # create .venv
uv pip install -r requirements.txt
uv run python scripts/preflight.py             # env + GPU check
uv run python app.py                           # run the UI
uv run python scripts/benchmark_models.py --models qwen3-vl:8b gemma3:12b   # optional
uv run python tests/test_rules_engine.py       # offline tests
```
