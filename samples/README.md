# samples/

No images are bundled — add your own.

- Put test photos/videos under `samples/<action_id>/`, e.g.
  `samples/battery_recycling/`, `samples/pet_bottle_recycling/`,
  `samples/paper_container_css/`, `samples/reusable_cup/`.
  (Action ids are the keys in `rules/eco_rules.json`.)
- The Gradio app (`app.py`) needs no samples — you can upload directly in the UI.
- Samples are only required for `scripts/benchmark_models.py`. For that, copy
  `labels.example.json` to `labels.json` and fill in the ground truth so each model's
  extraction can be scored.
