# ♻️ Eco-Reward PoC — Taiwan

A fully-local proof of concept that scores and rewards eco-friendly civic actions
(recycling, reusable cups, etc.) from a photo or short video — verified by a local
Vision-Language Model, with rewards decided by a hand-curated Taiwan rules file.

Solo-engineer scope: one Apple Silicon Mac, Ollama, Gradio. No cloud, no training,
no YOLO, no vector DB. See [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) for the full
engineering rationale and what was cut from the original research docs.

---

## How it works

```
Gradio upload → sample frames → qwen3-vl:8b (Ollama) structured JSON
             → rules engine → deterministic reward → fraud checks → verdict
```

The VLM only *observes* (objects, OCR text, cleanliness, scale weight, confidence).
A plain Python rules engine decides pass/fail and computes the reward — the model
never does the math. Adding a new eco-action is a JSON edit, not a code change.

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
# add your own images to samples/<action_id>/ and create samples/labels.json
# (copy samples/labels.example.json and fill in the ground truth)
uv run python scripts/benchmark_models.py --models qwen3-vl:8b qwen3-vl:4b gemma3:12b minicpm-v4.5:8b
```

It prints a ranked table (object recognition, scale/sticker OCR, cleanliness, counts,
end-to-end verdict, latency) so you pick the winner empirically. Set your choice in the
Gradio "Ollama model" box, or change `DEFAULT_MODEL` in `pipeline/vlm.py`.

---

## Using it

1. Pick the eco-action you're claiming.
2. **(Optional) challenge code** — click *New code* and show it somewhere in your shot
   (write it on paper / display on another device) to prove the submission is fresh.
   Leave it blank to score a pre-existing photo (the anti-fraud code check is then skipped).
3. Upload a photo, or record/upload a short video.
4. Hit **Verify & reward**. You'll get a verdict, points, the reasoning, and — when the
   action is rejected — an educational tip on what to fix.

Actions covered in the PoC: battery recycling · PET bottle · paper container
(Clean·Sort·Stack 清分疊) · reusable cup.

---

## Test without a GPU

The rules engine and reward math are pure Python:

```bash
uv run python tests/test_rules_engine.py    # 9/9 should pass
```

---

## Project layout

```
app.py                     Gradio UI
rules/eco_rules.json       the "knowledge base" (edit to add actions)
pipeline/
  media.py                 image / video-frame sampling + motion score
  vlm.py                   Ollama qwen3-vl call + JSON parsing
  prompts.py               the forced-JSON extraction prompt
  rules_engine.py          rule matching + deterministic reward calculator
  fraud.py                 challenge code · duplicate hash · liveness sanity
  orchestrator.py          wires the stages together
scripts/
  preflight.py             checks Ollama + model are ready
  benchmark_models.py      A/B test candidate VLMs on your own photos
samples/labels.example.json ground-truth format for the benchmark
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

**Model choice, re-derived on the two test photos** (`scripts/benchmark_models.py`):

| model                | overall | latency_s | objects | count | cleanliness | verdict |
|----------------------|:-------:|:---------:|:-------:|:-----:|:-----------:|:-------:|
| **qwen3-vl:8b**      | **1.00**| 70.7\*    | 1.00    | 1.00  | 1.00        | 1.00    |
| qwen3-vl:4b-instruct | 0.86    | 10.9      | 1.00    | 1.00  | 1.00        | 0.50    |

\* The 8b latency includes the one-time cold model load (~6 GB into GPU) on the first
image; warm per-image inference is faster. The 4b is ~6× quicker but gets a verdict wrong
(weaker OCR/reasoning on the battery machine), so `qwen3-vl:8b` stays the `DEFAULT_MODEL`
in `pipeline/vlm.py`.

**One full sample result** — battery photo (a boy at a yellow *ecoco* reverse-vending
battery machine, `samples/battery_recycling/1.jpg`):

```
✅ Verified — Battery recycling at a convenience store
Reward: 500 Green Points  (≈ NT$5.00)
Confidence: 0.95
Why:
  • Recognized a valid item (battery recycling machine).
  • Location cue matched (ecoco, 新竹市政府).
  • no scale weight visible — awarding base drop-off reward
```

Raw VLM JSON for that image (note the Chinese OCR and correct object/marker read):

```json
{
  "detected_objects": ["battery recycling machine"],
  "primary_action": "recycling batteries",
  "location_markers": ["ecoco", "新竹市政府"],
  "ocr_text": ["battery", "ecoco", "join to play and have fun"],
  "scale_reading_kg": null,
  "cleanliness": "clean",
  "confidence": 0.95
}
```

The PET-bottle photo (`samples/pet_bottle_recycling/1.jpg`, a woman inserting a clear
bottle into an *iCIRCLE* machine) verifies at **200 Green Points (NT$2)**, item_count 1,
with the machine's full Traditional-Chinese signage read correctly.

> **Fixes the real run exposed** (see git diff): (1) reverse-vending battery machines show
> no scale to the camera, so a `per_weight` reward computed to zero — added a JSON-declared
> `fallback_flat_points` so a verified drop-off still pays a base reward (kept in the rules
> file, not hardcoded); (2) made the challenge code opt-in so pre-existing photos aren't
> auto-rejected by the freshness check.

---

## Notes & honesty

- **Reward values are illustrative**, pulled from the research docs. Verify against
  current MOENV / 7-Eleven / FamilyMart policy before any real deployment.
- **Anti-fraud is PoC-level** (stops casual cheating, not a determined attacker).
  Production-grade liveness/anti-spoofing, EXIF/GPS checks, and a real reward gateway
  are listed as future work in the blueprint.
- **To go live later:** swap the static rules file for the MOENV/EPB open-data APIs
  behind the same `load_rules()` interface — the pipeline shape doesn't change.
