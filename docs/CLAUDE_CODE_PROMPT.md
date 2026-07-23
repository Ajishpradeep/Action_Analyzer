# Paste-ready Claude Code prompt

Open a terminal in the repo root (`Action_Analyzer/`) and run `claude`. Paste the block
below as your first message.

---

You are working in the `Action_Analyzer` repo — an "Eco-Reward" proof of concept. Start by
reading `CLAUDE.md`, then `docs/BLUEPRINT.md`, then `docs/AGENT_TASKS.md`. These fully
define the project; follow them and don't reintroduce the complexity listed as "cut" in
BLUEPRINT §1.

Goal: get a **working local demo** running on my Apple Silicon Mac and verify it end to end.

The end result must be:
- A **Gradio** web interface I can open in a browser.
- It accepts an **image or a short video** of an eco-friendly action in Taiwan.
- A **Vision-Language Model running locally via Ollama** analyzes the media, on the Mac's
  **GPU (Metal), not the CPU**, and describes what it sees *in the context of the eco-action*.
- The app then produces a **score/reward** for that action using the rules file, and shows
  the reasoning.

Use **uv** as the package manager (not pip). Do this in order:
1. Verify Ollama is installed and ≥ 0.12.7; `uv venv`; `uv pip install -r requirements.txt`;
   `ollama pull qwen3-vl:8b`; then run `uv run python scripts/preflight.py` and fix any failure.
2. Confirm the model runs on the **GPU**: start it and check `ollama ps` shows `100% GPU`.
   If it falls back to CPU, diagnose (unified-memory pressure → try `qwen3-vl:4b`; ensure no
   CPU-forcing env var). Record the `ollama ps` output in the README.
3. Test data: I will add my own images to `samples/<action_id>/` (or upload directly in the
   UI). Samples are only needed for the benchmark in step 6 — for that, copy
   `samples/labels.example.json` to `samples/labels.json` and fill in ground truth to match
   my images. Do NOT generate or download placeholder images.
4. Run `uv run python app.py`. I'll upload a battery photo (expect a description + any weight
   reading + a battery reward), a reusable-cup photo, a PET-bottle photo, and a short video;
   confirm each returns a sensible verdict.
5. Fix whatever the real run exposes (JSON parsing robustness in `pipeline/vlm.py`, prompt
   wording in `pipeline/prompts.py`, and — importantly — mismatches between the model's
   object words and the keyword lists in `rules/eco_rules.json`; broaden the JSON, don't
   hardcode in Python). Keep the LLM out of arithmetic; scoring stays in `rules_engine.py`.
6. Verify: `uv run python tests/test_rules_engine.py` (7/7), and run
   `uv run python scripts/benchmark_models.py --models qwen3-vl:8b qwen3-vl:4b` over my
   `samples/` to confirm the default model choice on my data. Put the ranked table in the
   README and set the winner as `DEFAULT_MODEL` in `pipeline/vlm.py`.
7. Update the README with exact run steps and what you verified (GPU status, sample results).

Definition of done: `preflight.py` passes, `uv run python app.py` serves the UI, uploading a
sample image returns a sensible eco-action description + reward, `ollama ps` shows the model
on GPU, and the unit tests pass. Show me the `ollama ps` output and one full sample result
when you're done.

---
