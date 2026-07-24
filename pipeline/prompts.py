"""Prompt for the fully open-set eco-action brain.

There is NO predefined list of actions. The VLM looks at the media, works out what the
person is doing (whatever it is), judges how much it is a genuine eco-friendly /
environmental action, and *suggests* a reward. The Python side (rules_engine) turns that
into a deterministic number — the model never has the final say (CLAUDE.md #6).
"""

EXTRACTION_SCHEMA_KEYS = [
    # what the action is
    "scene_description",     # plain-language: what is happening in the media
    "action_label",          # short free-text label of the specific action
    # how eco-friendly it is
    "is_eco_action",         # bool: is this an environmental action at all?
    "eco_relevance",         # 0.0-1.0: how strongly this is a genuine eco-friendly action
    "eco_rationale",         # one sentence: why, and how impactful
    # the brain's reward suggestion (validated/clamped in Python)
    "suggested_points",
    "suggested_reasoning",
    # supporting evidence
    "detected_objects",
    "location_markers",
    "ocr_text",
    "item_count",
    "cleanliness",
    "challenge_code_visible",
    "confidence",
    "notes",
]

SYSTEM_PROMPT = (
    "You are the analytical brain of an eco-reward app. You are shown a photo or video "
    "frames of a person doing something, and you must work out — with no predefined list "
    "of actions — what they are doing and how much it counts as a genuine eco-friendly or "
    "environmental action. Judge any action on its own merits, however common or unusual. "
    "Report only what is actually visible, never invent objects or text, and read any "
    "labels, logos, signage, or readings you can see. Respond with JSON only."
)

USER_PROMPT_TEMPLATE = """Analyze the image(s). They may be frames from a short video of one continuous action.

A challenge code was issued for this submission: "{challenge_code}"

Work out what the person is doing and how eco-friendly it is. There is NO fixed list of
allowed actions — evaluate whatever you actually see. Eco-friendly actions can be anything
with a positive environmental effect (examples, NOT an exhaustive list, do not limit
yourself to these): recycling or sorting waste, proper disposal of hazardous items,
reusing instead of using single-use, composting, cleaning up litter, saving energy or
water, choosing low-carbon transport, repairing instead of discarding, refusing plastic.
If what you see is not environmental at all, say so honestly.

Return a SINGLE JSON object with EXACTLY these keys:

- "scene_description": one or two sentences, plain language, describing what is happening.
- "action_label": a short lowercase label for the specific action you see
  (e.g. "putting a bottle into a recycling machine", "separating trash into sorted bags",
  "carrying groceries in a cloth bag" — but describe whatever the real action is).
- "is_eco_action": true if this is a genuine eco-friendly/environmental action, else false.
- "eco_relevance": a number 0.0-1.0 for HOW eco-friendly the action is — 0.0 = not
  environmental at all, ~0.5 = weakly/partially eco-friendly, 1.0 = clearly a strong,
  deliberate eco-friendly action. Judge how close this is to a real environmental action.
- "eco_rationale": one short sentence on why it is (or isn't) eco-friendly and how impactful.
- "suggested_points": integer Green Points you think it deserves (100 points = NT$1),
  weighing effort and environmental impact; typical range 50-600. A suggestion only.
- "suggested_reasoning": one short sentence justifying the amount.
- "detected_objects": array of short lowercase strings for every relevant object you see. [] if none.
- "location_markers": array of visible brand/place cues, lowercase. [] if none.
- "ocr_text": array of any text strings you can read (labels, signage, stickers). [] if none.
- "item_count": integer count of the main relevant items if clearly countable, else null.
- "cleanliness": one of "clean", "dirty", "leaking", or "unknown" (contamination/condition).
- "challenge_code_visible": the exact code string if you can read it, else null.
- "confidence": your confidence 0.0-1.0 that the action genuinely happened as described.
- "notes": one short sentence of anything important, else "".

JSON only. No markdown, no commentary."""


def build_user_prompt(challenge_code: str) -> str:
    return USER_PROMPT_TEMPLATE.format(challenge_code=challenge_code or "none")
