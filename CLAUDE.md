# CLAUDE.md — Eco-Reward PoC (Taiwan)

Context for Claude Code working in this repo. Read this first, then `docs/BLUEPRINT.md`
(architecture + rationale) and `docs/AGENT_TASKS.md` (the ordered build plan).

## What this is

A **fully-local** proof of concept: a Gradio web UI where a user uploads a photo or short
video of an eco-friendly action in Taiwan (recycling a battery, PET bottle, paper container,
or using a reusable cup). A **local Vision-Language Model served by Ollama** looks at the
media, describes what it sees *in the context of the eco-action*, and the app produces a
**score/reward** based on a hand-curated Taiwan rules file.

Solo-engineer scope. No cloud APIs, no model training, no YOLO, no vector DB. The VLM does
perception; a small Python rules engine does the scoring (the model never does the math).

## Hard requirements (do not drift from these)

0. **Package manager is `uv`.** Use `uv venv`, `uv pip install -r requirements.txt`, and
   `uv run python ...` — never plain `pip`.
1. **Everything runs locally on an Apple Silicon Mac (16 GB+).** No hosted inference.
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
6. **Scoring stays decoupled** in `rules/eco_rules.json` + `pipeline/rules_engine.py`.
   Adding an action = editing JSON, not code.

## Architecture (already scaffolded — extend, don't rewrite)

```
app.py                      Gradio UI
rules/eco_rules.json        knowledge base: action -> conditions -> reward
pipeline/
  media.py                  image / video-frame sampling + motion (liveness) score
  prompts.py                forced-JSON extraction prompt (VLM only observes)
  vlm.py                    Ollama call (DEFAULT_MODEL) + robust JSON parsing
  rules_engine.py           condition matching + deterministic reward calculator
  fraud.py                  challenge code · duplicate hash · liveness sanity
  orchestrator.py           media -> vlm -> rules -> reward -> fraud -> result
scripts/
  preflight.py              verify Ollama + model are ready (and GPU)
  benchmark_models.py       A/B test candidate VLMs on samples/ (evidence-based choice)
samples/                    YOU add test images here (see samples/labels.example.json)
tests/test_rules_engine.py  offline unit tests (no GPU needed)
docs/BLUEPRINT.md           full design + what was cut & why
docs/AGENT_TASKS.md         ordered action plan for you
```

Data flow: the VLM returns a fixed JSON schema (objects, OCR text, scale weight,
cleanliness, counts, confidence — see `pipeline/prompts.py`). `rules_engine.evaluate()`
matches that against the action's rule and computes points. `orchestrator.run()` wires it
all and lets fraud checks veto a positive verdict.

## Conventions

- Python 3.11+, standard library + the pinned deps in `requirements.txt`. Keep it small.
- Each pipeline stage is a pure-ish function so it can be unit-tested without a GPU.
- Never let the LLM compute rewards — that's `rules_engine._compute_reward`.
- Reward figures in `eco_rules.json` are **illustrative** (from research docs); don't
  present them as authoritative.
- Don't reintroduce the cut complexity (YOLO, agentic verifiers, anti-spoof models,
  vector DB, MoE serving) — see BLUEPRINT §1.

## Definition of done

`uv run python scripts/preflight.py` passes · `uv run python app.py` serves a Gradio UI ·
uploading one of the user's own test images returns a sensible description + a reward ·
`ollama ps` shows the model on GPU · `uv run python tests/test_rules_engine.py` is 7/7.

## Test data

There are no bundled sample images — the user supplies their own. Drop images into
`samples/<action_id>/` (e.g. `samples/battery_recycling/`), then copy
`samples/labels.example.json` to `samples/labels.json` and fill in the ground truth so
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
