# ♻️ Eco-Reward PoC

A fully-local, **fully open-set** proof of concept: upload a photo or short video of
**any** action and a local Vision-Language Model works out what it is, judges how
eco-friendly it is, and rewards it. There is **no predefined list of actions** and nothing
to pick — the model decides.

Solo-engineer scope: one Apple Silicon Mac, Ollama, Gradio. No cloud (except an optional
embedding layer for anti-fraud), no training, no YOLO. See
[`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) for the original engineering rationale.

---

## How it works

```
Gradio upload → sample frames → qwen3-vl:8b (Ollama) open-set analysis
             → score_action → deterministic reward → fraud checks → verdict
```

It's an **open-set brain**, not a checklist. You don't tell it what you did — the VLM
works out the action on its own (recycling, sorting trash into separate bags, a bottle
into a recycle-labeled bin, a reusable cup, a battery drop-off, litter cleanup, composting,
carrying a cloth bag, … *anything*), rates **how eco-friendly** it is (`eco_relevance`
0–1), and **suggests** a reward.

Scoring stays deterministic Python — the model never has the final say on the number:

```
points = clamp(suggested_points, min, max) × eco_relevance × quality_multiplier
```

So a weakly-eco action scores proportionally less, a leaking/hazardous item scores zero,
and the same submission always yields the same number. All knobs live in the `scoring`
block of `rules/eco_rules.json` — tuning is a JSON edit, not a code change.

---

## Setup (macOS, Apple Silicon)

1. **Install Ollama** — https://ollama.com/download

2. **Pull the model** (6.1 GB, needs Ollama ≥ 0.12.7):
   ```bash
   ollama pull qwen3-vl:8b
   # lighter / faster:            ollama pull qwen3-vl:4b
   # alternatives to A/B test:    ollama pull gemma3:12b   |   ollama pull minicpm-v4.5:8b
   ```
   Make sure the server is running (`ollama serve`, or the menu-bar app).

3. **Python env (uv):**
   ```bash
   cd Action_Analyzer
   uv venv                                # creates .venv
   uv pip install -r requirements.txt
   ```

4. **Preflight (confirms Ollama + model are ready):**
   ```bash
   uv run python scripts/preflight.py
   ```

5. **Run:**
   ```bash
   uv run python app.py
   ```
   Open the local URL Gradio prints (usually http://127.0.0.1:7860).

### Which model? Decide on evidence, not a blog post

`qwen3-vl:8b` is the default because its card claims the best OCR (32 languages incl.
Chinese) and native video — the axes that matter most here. But verify on **your** photos:

```bash
# add your own media flat in samples/ and create samples/labels.json
# (copy samples/labels.example.json and fill in expected_is_eco + a note)
uv run python scripts/benchmark_models.py --models qwen3-vl:8b qwen3-vl:4b
```

It prints a ranked table (eco / not-eco accuracy, eco-relevance, scene description,
latency) so you pick the winner empirically. Set your choice in the Gradio "Ollama model"
box, or change `DEFAULT_MODEL` in `pipeline/vlm.py`.

---

## Using it

1. Just upload — there's nothing to select. The brain figures out the action itself.
2. **(Optional) challenge code** — click *New code* and show it somewhere in your shot
   (write it on paper / display on another device) to prove the submission is fresh.
   Leave it blank to score a pre-existing photo (the anti-fraud code check is then skipped).
3. Upload a photo, or record/upload a short video.
4. Hit **Analyze & reward**. You'll get the brain's read of the scene, a verdict + points,
   the reasoning (including how any suggested points were clamped), and — when rejected —
   an educational tip on what to fix.

There are no predefined actions — *any* genuine eco-friendly action is recognized and
rewarded, scaled by how eco-friendly the brain judges it to be.

---

## Semantic anti-fraud (Gemini embeddings) — on by default when a key is set

This layer strengthens the two weakest spots in the PoC and is **active by default
whenever a Gemini key is configured**. With no key it's skipped and the app falls back to
the local-only average-hash duplicate check (bare-minimum mode). Configure the key with:

```bash
cp .env.example .env          # then paste your key into GEMINI_API_KEY
# get one free at https://aistudio.google.com/apikey
```

**Semantic replay detection.** Each submission's representative frame is embedded once via
Gemini Embedding 2 (768-dim) and compared (cosine) against this user's accepted history. It
catches the same action resubmitted for a second reward *even re-cropped, re-compressed, or
shot from a new angle* — cases the `imagehash.average_hash` check silently misses. A match
≥ `ECO_REPLAY_THRESHOLD` (0.90) withholds the reward. It's action-agnostic: it fingerprints
whatever is in the frame, with no notion of predefined action types.

This deliberately relaxes the project's fully-local rule, so it stays gated behind the key.
State lives in `~/.eco_reward/` (submission fingerprints).

**Verified live** (fully open-set — no action supplied — + Gemini embeddings):

| Submission | Brain's open-set read | eco-rel. | Verdict | Guard |
|---|---|---|---|---|
| Sample 1 | "inserting a battery into a recycling machine" | 1.0 | ✅ 180 pts | — |
| Sample 2 | "recycling a plastic bottle in a smart recycling machine" | 1.0 | ✅ 200 pts | — |
| Sample 1 **resubmitted** | same scene | 1.0 | ❌ withheld | **replay 100% similar** |

> **Attribution:** the embedding backend (`pipeline/embeddings.py`) and the cosine store
> pattern are adapted from [SentrySearch](https://github.com/ssrajadh/sentrysearch)
> (Apache License 2.0). SentrySearch is a semantic *video-retrieval* tool; only its
> embedding + similarity primitive is reused here — its search/highlights/overlay
> features don't apply to a single-submission reward flow. We use a lightweight numpy
> vector store instead of its ChromaDB to keep the PoC dependency-light.

---

## Test without a GPU

The rules engine and reward math are pure Python:

```bash
uv run python tests/test_rules_engine.py    # 9/9 — open-set deterministic scorer
uv run python tests/test_fingerprint.py      # 3/3 — replay detection (fixed vectors)
```

---

## Project layout

```
app.py                     Gradio UI
rules/eco_rules.json       the `scoring` config (point band + thresholds + multipliers)
pipeline/
  media.py                 image / video-frame sampling + motion score
  vlm.py                   Ollama qwen3-vl call + JSON parsing
  prompts.py               the open-set brain prompt (no predefined actions)
  rules_engine.py          score_action(): deterministic eco_relevance-scaled clamp
  fraud.py                 challenge code · duplicate hash · liveness sanity
  embeddings.py            OPT-IN Gemini embedder (BaseEmbedder ABC) — cloud, key-gated
  vector_store.py          tiny numpy cosine store for submission fingerprints
  fingerprint.py           semantic replay detection (action-agnostic)
  orchestrator.py          wires the stages together
scripts/
  preflight.py             checks Ollama + model are ready
  benchmark_models.py      A/B test candidate VLMs on your own media (open-set)
samples/labels.example.json open-set ground-truth format for the benchmark
tests/test_rules_engine.py offline unit tests
docs/BLUEPRINT.md          engineering blueprint + what was cut & why
```

---

## Verified on this Mac (Apple Silicon)

End-to-end run on 2026-07-23, Ollama 0.32.0, `uv` 0.7.3, Python 3.12.

**GPU (Metal), not CPU** — `ollama ps` right after an inference with the default model:

```
NAME           ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen3-vl:8b    901cae732162    5.8 GB    100% GPU     4096       4 minutes from now
```

`PROCESSOR = 100% GPU` confirms the VLM runs on Metal. No `OLLAMA_NUM_GPU`/CPU-forcing
env var is set; the 5.8 GB model fits comfortably in unified memory.

`qwen3-vl:8b` is the default (best OCR incl. Chinese + native video); `qwen3-vl:4b` runs
~6× faster with weaker OCR/reasoning. Re-derive on your own media with
`scripts/benchmark_models.py`.

**One full open-set result** — no action supplied; the brain works it out
(`samples/1.jpg`, a boy at a yellow *ecoco* battery machine):

```
✅ Rewarded — inserting a battery into a recycling machine
Reward: 180 Green Points  (≈ NT$1.80)
Eco-relevance: 1.00  ·  Confidence: 1.00
Why:
  • Recognized an eco-friendly action: inserting a battery into a recycling machine.
  • Eco-relevance 1.00; brain suggested 180 pts → clamped to [50-600] = 180, ×1.00 relevance = 180 pts.
  • Brain's rationale: Single battery recycling is a strong eco-action with clear environmental benefit.
```

The brain named the action itself, rated its eco-relevance, and suggested points — with no
predefined action list anywhere. `samples/2.jpg` (a woman at an *iCIRCLE* bottle machine)
is likewise recognized as "recycling a plastic bottle in a smart recycling machine" and
rewarded 200 pts, reading the machine's full Traditional-Chinese signage correctly.

---

## Notes & honesty

- **Reward values are illustrative**, pulled from the research docs. Verify against
  current MOENV / 7-Eleven / FamilyMart policy before any real deployment.
- **Anti-fraud is PoC-level** (stops casual cheating, not a determined attacker).
  Production-grade liveness/anti-spoofing, EXIF/GPS checks, and a real reward gateway
  are listed as future work in the blueprint.
- **To go live later:** swap the static rules file for the MOENV/EPB open-data APIs
  behind the same `load_rules()` interface — the pipeline shape doesn't change.
