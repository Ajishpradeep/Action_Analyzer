"""Eco-Reward PoC — Gradio front end (fully open-set).

Upload any photo or short video of any action; the app works out what you're doing, judges
how eco-friendly it is, and rewards it. Nothing to pick, no model details to worry about.

Run:
    uv run python app.py
"""

from __future__ import annotations

import gradio as gr

from pipeline import fingerprint as fp_mod
from pipeline import orchestrator
from pipeline.embeddings import get_embedder
from pipeline.rules_engine import load_rules

RULES = load_rules()

# Session-scoped duplicate memory (average-hash fallback). Per-user in a real deployment.
SEEN_HASHES: set[str] = set()

# Semantic replay detection is available when a Gemini key is configured; otherwise the
# anti-replay toggle falls back to a local session-only duplicate check.
EMBEDDER = get_embedder()
STORE = fp_mod.default_store() if EMBEDDER is not None else None

_UPLOAD_TYPES = [
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic", ".heif",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
]


def _progress_md(done: list[str], active: str | None) -> str:
    lines = ["### ⏳ Working on it…", ""]
    for label in done:
        lines.append(f"- ✅ {label}")
    if active:
        lines.append(f"- 🔄 **{active}…**")
    return "\n".join(lines)


def _fmt_result(res: orchestrator.PipelineResult) -> str:
    if res is None or res.error:
        return f"### ⚠️ Error\n\n{res.error if res else 'Something went wrong.'}"

    d = res.decision
    f = res.fraud
    lines = []

    if d.verified:
        lines.append(f"## ✅ Rewarded — {d.action_label}")
        lines.append(f"**Reward:** {d.points} Green Points  (≈ NT${d.ntd:.2f})")
    elif d.needs_review:
        lines.append(f"## 🕵️ Needs review — {d.action_label}")
    else:
        lines.append(f"## ❌ Not rewarded — {d.action_label}")

    if d.scene_description:
        lines.append(f"\n> 🧠 {d.scene_description}")

    lines.append(f"\n**How eco-friendly:** {d.eco_relevance:.0%}")

    lines.append("\n**Why:**")
    for r in d.reasons:
        lines.append(f"- {r}")

    if f and f.flags:
        lines.append("\n**Anti-fraud:**")
        for flag in f.flags:
            lines.append(f"- {flag}")

    if not d.verified and d.educational:
        lines.append(f"\n💡 **Tip:** {d.educational}")

    return "\n".join(lines)


def analyze(media, use_replay):
    if not media:
        yield "Please upload a photo or short video first."
        return

    done: list[str] = []
    active: str | None = None
    res = None
    for kind, payload in orchestrator.run_stream(
        media_path=media,
        seen_hashes=SEEN_HASHES,
        rules=RULES,
        embedder=EMBEDDER,
        store=STORE,
        use_replay=bool(use_replay),
    ):
        if kind == "stage":
            if active:
                done.append(active)
            active = payload
            yield _progress_md(done, active)
        else:
            res = payload
    yield _fmt_result(res)


with gr.Blocks(title="Eco-Reward") as demo:
    gr.Markdown(
        "# ♻️ Eco-Reward\n"
        "Upload a photo or short video of **any** action. The app figures out what you're "
        "doing, judges how eco-friendly it is, and rewards it — nothing to pick."
    )

    with gr.Row():
        with gr.Column():
            media_in = gr.File(
                label="Upload a photo or short video",
                file_types=_UPLOAD_TYPES, file_count="single", type="filepath",
            )
            replay_toggle = gr.Checkbox(
                value=True, label="🛡️ Anti-replay check",
                info="When on, compares against your past submissions so the same action "
                     "can't be rewarded twice. When off, no history is checked or recorded.",
            )
            go = gr.Button("Analyze & reward", variant="primary")
        with gr.Column():
            out = gr.Markdown()

    go.click(analyze, inputs=[media_in, replay_toggle], outputs=[out])

    gr.Markdown("---\n*Reward values are illustrative.*")


if __name__ == "__main__":
    demo.launch()
