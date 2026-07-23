# Eco-Reward PoC — Engineering Blueprint (Streamlined)

A solo-engineer, fully-local proof of concept. One machine (Apple Silicon Mac, 16 GB+),
one VLM served by Ollama, a Gradio UI, and a hand-curated Taiwan rules file. No cloud,
no training, no org backing.

---

## 1. What we cut and why

The three research docs describe a production system with a lot of moving parts that make
no sense for a single-engineer PoC. Here's the honest triage.

| Dropped from the docs | Why it's wrong for this PoC | What we do instead |
|---|---|---|
| **YOLO / CSPDarknet object detection** | A separate detector you'd have to fine-tune on a Taiwan waste dataset, then still bolt reasoning on top. Redundant. | A single VLM (Qwen2.5-VL) detects, reads text, *and* reasons in one call. This was your call and it's the right one. |
| **Argos agentic verifier, AutoThinkRAG complexity router** | Research-grade RL/agent frameworks; weeks of work, no local weights. | One well-structured VLM call + a deterministic rules check. |
| **InstructFLIP anti-spoof, rPPG blood-pulse liveness** | Face-anti-spoofing PhD territory. Overkill and creepy for recycling. | Lightweight fraud heuristics (challenge code, duplicate hashing, multi-frame motion sanity). |
| **Kimi-VL MoE + DP/EP/PP/CP parallelism** | Datacenter serving concerns. You have one Mac. | Ollama serves one quantized model. Done. |
| **COAD online/continual learning** | On-device weight updates. Not a PoC concern. | Static model. |
| **TTGNN / HierFinRAG graph-neural RAG, Pinecone** | Graph NN over municipal tables + vector DB. Heavy. | A flat `eco_rules.json` the VLM output is matched against. |
| **Optical-flow homography stabilization** | Classical CV pipeline for shaky egocentric video. | Sample a few frames, keep the sharpest. Good enough. |
| **Symbolic-Neural Fusion Reasoning** | Fancy name for "don't let the LLM do the math." | A plain Python reward calculator. That *is* the symbolic layer. |
| **Live MOENV/gov open-data APIs** | Network-dependent, fragile in a demo. | Static rules file now; swap to APIs later — the interface is the same. |

**Kept, simplified:** VLM as the perception+reasoning engine, a local knowledge base
(rules file), a deterministic reward calculator, a confidence threshold, basic anti-fraud,
and a Gradio front end.

---

## 2. Model choice

All facts below were checked against the **live ollama.com library** (not memory).
Don't take the ranking on faith — `scripts/benchmark_models.py` re-derives it on your
own photos (see §7).

**Primary: `qwen3-vl:8b` via Ollama** — 6.1 GB (Q4_K_M, 8.77B params), 256K context,
Text+Image. **Requires Ollama ≥ 0.12.7.**

Why it wins for *this* task specifically:

- **OCR across 32 languages incl. Chinese** (up from 10 in the 2.5 generation) and
  "more reliable under poor lighting, blur, or tilted text" per its model card. Half our
  eco-actions are verified by *text on the object* — scale weight ("1.3 kg"), the
  "友善食光" sticker, receipts, store signage — so Chinese OCR robustness is the single
  most important axis here.
- **Native long-video understanding** (down to the second), which matters because you
  chose image**+video** input.
- **Better spatial understanding** (2D/3D grounding) — helps confirm an item actually
  entered a bin vs. just being held near it.
- Emits stable JSON, so we can force structured extraction instead of parsing prose.
- Same ~6 GB footprint as the older `qwen2.5vl:7b`, so no cost to upgrading.

**Lighter option: `qwen3-vl:4b`** (3.3 GB) — for speed / tighter memory, weaker OCR.
`qwen3-vl:2b` (1.9 GB) exists for very constrained runs.

**Worth A/B-testing on your data (the harness makes this one command):**

| Model | Size | Why consider it | Trade-off |
|---|---|---|---|
| `qwen3-vl:8b` | 6.1 GB | Best OCR + Chinese + video (**default**) | — |
| `gemma3:12b` (or `gemma3:12b-it-qat`) | 8.1 GB | Strong general vision, 140+ langs, QAT keeps quality at low RAM | Weaker on dense OCR than Qwen |
| `minicpm-v4.5:8b` | ~6 GB | Explicitly tuned for **high-FPS video** understanding | Structured-JSON slightly less consistent |
| `gemma3:4b` | 3.3 GB | Very light multimodal fallback | Coarser OCR/reasoning |

**Correction to my first pass:** I initially defaulted to `qwen2.5vl:7b` and wrongly
claimed Qwen 3 vision "isn't wired into Ollama." It is — `qwen3-vl` is an official library
model with proper vision support. The default is now `qwen3-vl:8b`. Note: there is **no
official `qwen3.5-vl`** in the Ollama library as of this writing; the current Qwen vision
line is `qwen3-vl`.

> Reward figures (NT$ per battery kg, etc.) in `eco_rules.json` come from the research docs
> and are **illustrative** — verify against current MOENV / 7-Eleven / FamilyMart policy
> before any real deployment.

---

## 3. Pipeline

```
        ┌──────────── Gradio UI ────────────┐
        │  action hint (optional) + upload  │
        │  image  OR  short video           │
        │  + shows a random challenge code   │
        └───────────────┬───────────────────┘
                        ▼
   1. MEDIA PREP  (pipeline/media.py)
      image → use as-is
      video → sample K frames (OpenCV), keep the sharpest N,
              compute inter-frame motion (liveness sanity)
                        ▼
   2. PERCEPTION  (pipeline/vlm.py)
      Qwen2.5-VL via Ollama, forced-JSON prompt →
      { objects, action, location_markers, ocr_text,
        scale_reading_kg, item_count, cleanliness,
        lithium_terminal_taped, reusable_cup_present,
        challenge_code_visible, confidence, notes }
                        ▼
   3. RULES ENGINE  (pipeline/rules_engine.py)
      load eco_rules.json → match action → check
      required objects / location markers / safety /
      reject conditions / min confidence
                        ▼
   4. REWARD CALC  (deterministic Python, no LLM)
      per_weight | per_item | flat  → points + NT$
                        ▼
   5. FRAUD CHECKS  (pipeline/fraud.py)
      duplicate-image hash · challenge-code match ·
      motion/liveness sanity · confidence floor
                        ▼
   6. RESULT
      verdict ✓/✗ · points · plain-language rationale ·
      educational tip when rejected
```

Every stage is a small pure function so you can unit-test it without a GPU.

---

## 4. The knowledge base (`rules/eco_rules.json`)

A flat dictionary keyed by action id. Each action declares what the VLM output must satisfy
and how the reward is computed. Reward types:

- `per_weight` — points per unit weight, rounded down (batteries, optical discs).
- `per_item` — points per counted item (phones, cups).
- `flat` — fixed points for a verified action (reusable cup, correct sort).

Adding a new eco-action = adding one JSON object. No code change. That's the whole point of
keeping perception and policy decoupled.

PoC covers four actions:

1. **battery_recycling** — battery at a conv-store, OCR the scale, reward per 0.5 kg,
   safety check that lithium terminals are taped.
2. **pet_bottle_recycling** — clean PET bottle into a recycling bin.
3. **paper_container_css** — paper bento box, "Clean · Sort · Stack" (清、分、疊), reward
   only if visibly clean.
4. **reusable_cup** — personal cup handed over at a drink shop.

---

## 5. Anti-fraud (PoC-level, honest about limits)

- **Challenge–response:** UI shows a random 4-digit code; the user must have it visible
  (written on paper / shown on a second device) in the shot. The VLM reports whether it
  saw the code; we string-match. Defeats stock photos and reused clips.
- **Duplicate detection:** perceptual hash (average-hash) of submitted frames; reject if
  we've seen it before this session.
- **Liveness sanity for video:** require non-trivial inter-frame motion — a static photo
  filmed off a screen tends to be too still or too uniformly shaky.
- **Confidence floor:** VLM confidence below the action's `min_confidence` → route to
  "needs review" rather than auto-reward.

Deliberately **not** in the PoC (document as future work): EXIF/GPS validation, in-app
secure camera capture, injection-attack defense, real face-anti-spoofing.

---

## 6. Path to production (later, not now)

Swap the static rules file for the MOENV/EPB open-data APIs behind the same
`load_rules()` interface · add GPS/EXIF checks · move fraud to a real liveness model ·
add real Green Points / EasyCard reward gateway calls · batch the VLM behind a queue if
you ever get concurrency. None of these change the core pipeline shape.

---

## 7. Verify it yourself (don't trust the model ranking)

Two tools so the claims in this doc are reproducible on your machine, not taken on faith:

- **`scripts/preflight.py`** — confirms Ollama is running, the model is pulled, and it
  actually responds. Run before `app.py`.
- **`scripts/benchmark_models.py`** — runs several candidate models over *your own*
  labeled sample photos (`samples/<action_id>/*.jpg` + `samples/labels.json`) and prints
  a ranked table scoring object recognition, scale/sticker OCR, cleanliness, counts, the
  end-to-end verdict, and latency. This is how you decide `qwen3-vl:8b` vs `gemma3:12b`
  vs `minicpm-v4.5:8b` on real evidence instead of a blog benchmark.

**What was verified without a GPU (and what wasn't):** the rules engine + reward math have
unit tests (`tests/test_rules_engine.py`, 7/7) and all modules compile. The *model* stages
(`vlm.py`, `media.py`) can't be exercised in a CI sandbox with no Ollama — that's exactly
what preflight + benchmark are for, on your Mac.
