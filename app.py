"""Eco-Reward PoC — Gradio front end.

Run Ollama first:
    ollama serve            # (usually already running)
    ollama pull qwen3-vl:8b
Then:
    uv run python app.py
"""

from __future__ import annotations

import gradio as gr

from pipeline import fraud as fraud_mod
from pipeline import orchestrator
from pipeline.rules_engine import load_rules
from pipeline.vlm import DEFAULT_MODEL

RULES = load_rules()
ACTION_CHOICES = [
    (RULES[k]["display_name"], k)
    for k in RULES
    if not k.startswith("_")
]

# Session-scoped duplicate memory. For a multi-user deployment this would be per-user.
SEEN_HASHES: set[str] = set()


def _fmt_result(res: orchestrator.PipelineResult) -> str:
    if res.error:
        return f"### ⚠️ Error\n\n{res.error}"

    d = res.decision
    f = res.fraud
    lines = []

    if d.verified:
        lines.append(f"## ✅ Verified — {d.display_name}")
        lines.append(f"**Reward:** {d.points} Green Points  (≈ NT${d.ntd:.2f})")
    elif d.needs_review:
        lines.append(f"## 🕵️ Needs review — {d.display_name}")
    else:
        lines.append(f"## ❌ Not rewarded — {d.display_name}")

    lines.append(f"\n**Confidence:** {d.confidence:.2f}")

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


def issue_code() -> str:
    return fraud_mod.new_challenge_code()


def analyze(image, video, action_id, challenge_code, model):
    media_path = video or image
    if not media_path:
        return "Please upload an image or a short video first.", None
    # Challenge code is opt-in: only enforced when the user asks for one and shows
    # it in-frame. Leaving it blank skips that anti-fraud check (fraud.py handles this),
    # which is what you want when scoring pre-existing photos in a demo.
    challenge_code = (challenge_code or "").strip()

    action_hint = dict((v, k) for k, v in ACTION_CHOICES).get(action_id, action_id)

    res = orchestrator.run(
        media_path=media_path,
        action_id=action_id,
        action_hint=action_hint,
        challenge_code=challenge_code,
        seen_hashes=SEEN_HASHES,
        model=model,
        rules=RULES,
    )
    raw = res.observation.get("_raw", "") if res.observation else ""
    return _fmt_result(res), raw


with gr.Blocks(title="Eco-Reward PoC (Taiwan)") as demo:
    gr.Markdown(
        "# ♻️ Eco-Reward PoC — Taiwan\n"
        "Upload a photo or short video of an eco-action. A local VLM "
        f"(`{DEFAULT_MODEL}` via Ollama) inspects it; a rules file decides the reward.\n\n"
        "**Challenge code (optional):** click *New code* and show it somewhere in your "
        "shot to prove the submission is fresh. Leave it blank to score an existing photo."
    )

    with gr.Row():
        with gr.Column():
            action = gr.Dropdown(
                choices=ACTION_CHOICES, value=ACTION_CHOICES[0][1],
                label="Which eco-action are you claiming?",
            )
            with gr.Row():
                code = gr.Textbox(value="",
                                  label="Challenge code (optional — show in-frame)", scale=3)
                new_code_btn = gr.Button("New code", scale=1)
            image_in = gr.Image(type="filepath", label="Photo", sources=["upload", "webcam"])
            video_in = gr.Video(label="Short video (optional)", sources=["upload", "webcam"])
            model_in = gr.Textbox(value=DEFAULT_MODEL, label="Ollama model")
            go = gr.Button("Verify & reward", variant="primary")
        with gr.Column():
            out = gr.Markdown(label="Result")
            with gr.Accordion("Raw VLM output (debug)", open=False):
                raw_out = gr.Code(label="model JSON", language="json")

    new_code_btn.click(issue_code, outputs=code)
    go.click(analyze, inputs=[image_in, video_in, action, code, model_in],
             outputs=[out, raw_out])

    gr.Markdown(
        "---\n*Reward values are illustrative (from the research docs). "
        "Verify against current MOENV / retailer policy before real use.*"
    )


if __name__ == "__main__":
    demo.launch()
