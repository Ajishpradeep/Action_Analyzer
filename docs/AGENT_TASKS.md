# AGENT_TASKS.md — ordered build plan

For Claude Code. The repo is already scaffolded and the rules engine is tested. Your job is
to get it running end-to-end on this Apple Silicon Mac with the VLM on the GPU, verify it,
and polish. Work top to bottom; check off as you go.

## 0. Orient
- [ ] Read `CLAUDE.md`, then `docs/BLUEPRINT.md` (esp. §1 what-was-cut, §2 model, §3 pipeline).
- [ ] Skim every file under `pipeline/` and `scripts/`. Don't rewrite; extend.

## 1. Environment (uv — not pip)
- [ ] Confirm Ollama is installed and ≥ 0.12.7 (`ollama --version`). If older, tell the user
      to update (qwen3-vl needs it).
- [ ] `uv venv` then `uv pip install -r requirements.txt`.
- [ ] `ollama pull qwen3-vl:8b`.
- [ ] Run `uv run python scripts/preflight.py`. It must report server reachable, model
      present, and a live response. Fix any failure before moving on.

## 2. GPU (Metal) — must NOT run on CPU
- [ ] Start the model and run `ollama ps`. Confirm the PROCESSOR column shows `100% GPU`
      (Metal). If it shows CPU or a split:
  - unified memory may be too tight — close apps, or switch default to `qwen3-vl:4b`;
  - confirm no `OLLAMA_NUM_GPU=0` / CPU-forcing env var is set.
- [ ] Document in the README what you observed (`ollama ps` output).

## 3. Test data (user supplies their own — none bundled)
- [ ] Ask the user to drop a few test images into `samples/<action_id>/`, e.g.
      `samples/battery_recycling/`, `samples/reusable_cup/`. They may also just upload
      directly in the Gradio UI — samples are only needed for the benchmark in §6.
- [ ] For the benchmark, copy `samples/labels.example.json` to `samples/labels.json` and
      fill in the ground truth (`expected_objects`, `expected_cleanliness`,
      `expected_verified`, etc.) to match those actual images.

## 4. Run the app end-to-end
- [ ] `uv run python app.py`, open the Gradio URL.
- [ ] Upload one of the user's battery photos. Expect a description mentioning the
      battery/scale, any weight reading picked up, and a battery reward.
- [ ] Upload a reusable-cup photo and a PET-bottle photo. Sanity-check the verdicts.
- [ ] Record or upload a short video (a few seconds). Confirm frames are sampled and a
      verdict returns. Watch latency; note it.

## 5. Fix what the real run reveals
Likely issues to expect and handle:
- [ ] VLM occasionally returns non-JSON → confirm `pipeline/vlm.py` parsing handles it;
      tighten the prompt in `pipeline/prompts.py` if needed.
- [ ] Object names from the model don't match rule keywords (e.g. "aa cell" vs "battery")
      → broaden `required_objects_any` in `rules/eco_rules.json`, don't hardcode in Python.
- [ ] Cleanliness/scale fields come back null on real photos → adjust prompt wording.
- [ ] Confidence threshold too strict/loose → tune `min_confidence` per action.

## 6. Verify
- [ ] `uv run python tests/test_rules_engine.py` → 7/7.
- [ ] `uv run python scripts/benchmark_models.py --models qwen3-vl:8b qwen3-vl:4b` on the
      user's `samples/`. Record the ranked table in the README and keep the best model as
      default in `pipeline/vlm.py`.
- [ ] Add 2–3 more unit tests for any new rules logic you introduce.

## 7. Polish (optional, if time)
- [ ] Show the sampled video frames and the raw VLM JSON in the UI (there's already a debug
      accordion) so the demo is legible.
- [ ] Add a 5th eco-action (e.g. `optical_disc_recycling`) purely by editing
      `eco_rules.json` — proves the decoupling.
- [ ] Optional: a `qwen3-vl` native-video path (send the clip directly instead of sampled
      frames) and benchmark it against frame sampling.

## Guardrails
- Keep it local, keep scoring in the rules file, keep the LLM out of arithmetic.
- Don't reintroduce cut complexity (BLUEPRINT §1).
- Treat reward NT$ values as illustrative.
