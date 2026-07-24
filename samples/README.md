# samples/

No images are bundled — add your own. There are **no predefined action categories**;
just drop any photos/videos of any actions here (flat, any filename).

- Put test media directly in `samples/`, e.g. `samples/1.jpg`, `samples/clip.mp4`.
- The Gradio app (`app.py`) needs no samples — you can upload directly in the UI.
- Samples are only required for `scripts/benchmark_models.py`. For that, copy
  `labels.example.json` to `labels.json` and fill in the open-set ground truth (whether
  each file is a genuine eco-action, and a short note) so each model can be scored.
