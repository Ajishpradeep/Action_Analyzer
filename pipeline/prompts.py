"""Prompt templates for the VLM perception stage.

We force a single structured JSON extraction that works across every eco-action.
The rules engine (not the model) decides pass/fail and reward, so the prompt only
has to *observe* accurately, not judge.
"""

EXTRACTION_SCHEMA_KEYS = [
    "detected_objects",
    "primary_action",
    "location_markers",
    "ocr_text",
    "scale_reading_kg",
    "item_count",
    "cleanliness",
    "lithium_terminal_taped",
    "reusable_cup_present",
    "challenge_code_visible",
    "confidence",
    "notes",
]

SYSTEM_PROMPT = (
    "You are a careful visual inspector for an eco-recycling reward app in Taiwan. "
    "You only report what is actually visible. You never invent objects, text, or "
    "actions. If something is not visible or unclear, use null. Respond with JSON only."
)

# {action_hint} and {challenge_code} are filled in at call time.
USER_PROMPT_TEMPLATE = """Look at the image(s). They may be frames from a short video of one continuous action.

The user claims this eco-action: "{action_hint}"
A challenge code was issued for this submission: "{challenge_code}"

Return a SINGLE JSON object with EXACTLY these keys:

- "detected_objects": array of short lowercase strings for every relevant object you see
  (e.g. "lithium battery", "pet bottle", "paper container", "reusable cup", "digital scale",
  "recycling bin"). Empty array if none.
- "primary_action": one short phrase for what the person is doing, or null.
- "location_markers": array of visible brand/place cues, lowercase
  (e.g. "7-eleven", "familymart", "ecoco", "recycling bin", "counter"). Empty array if none.
- "ocr_text": array of any text strings you can read in the image(s). Empty array if none.
- "scale_reading_kg": the weight in kilograms if a scale/display shows one, else null.
- "item_count": integer count of the main recyclable items if clearly countable, else null.
- "cleanliness": one of "clean", "dirty", "leaking", or "unknown".
- "lithium_terminal_taped": true/false/null — only meaningful for lithium or button cells.
- "reusable_cup_present": true/false.
- "challenge_code_visible": the exact code string if you can read it in the image, else null.
- "confidence": your confidence 0.0-1.0 that the claimed action genuinely happened.
- "notes": one short sentence of anything important, else "".

JSON only. No markdown, no commentary."""


def build_user_prompt(action_hint: str, challenge_code: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        action_hint=action_hint or "unspecified",
        challenge_code=challenge_code or "none",
    )
