"""Lightweight anti-fraud for the PoC.

Honest scope: this stops casual cheating (reused photos, stock images, obviously
static replays). It is NOT production anti-spoofing — see BLUEPRINT.md section 5.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import List, Set

from PIL import Image
import imagehash


@dataclass
class FraudReport:
    passed: bool = True
    flags: List[str] = field(default_factory=list)


def new_challenge_code(n: int = 4) -> str:
    """A short numeric code the user must show in-frame."""
    return "".join(random.choices(string.digits, k=n))


def _phash(path: str) -> str:
    return str(imagehash.average_hash(Image.open(path)))


def check(frame_paths: List[str], obs: dict, challenge_code: str,
          is_video: bool, motion_score: float,
          seen_hashes: Set[str], min_motion: float = 1.5) -> FraudReport:
    report = FraudReport()

    # 1. Duplicate submission (perceptual hash of the first frame).
    try:
        h = _phash(frame_paths[0])
        if h in seen_hashes:
            report.passed = False
            report.flags.append("Duplicate image — this frame was already submitted this session.")
        else:
            seen_hashes.add(h)
    except Exception as e:  # noqa: BLE001
        report.flags.append(f"Could not hash image ({e}).")

    # 2. Challenge-code presence.
    seen_code = obs.get("challenge_code_visible")
    if challenge_code:
        if seen_code and str(seen_code).strip() == str(challenge_code).strip():
            report.flags.append("Challenge code matched ✓")
        else:
            report.passed = False
            report.flags.append(
                f"Challenge code '{challenge_code}' not clearly visible in the submission "
                f"(model read: {seen_code!r})."
            )

    # 3. Liveness sanity for video.
    if is_video and motion_score < min_motion:
        report.passed = False
        report.flags.append(
            f"Video too static (motion {motion_score:.2f} < {min_motion}) — possible replay of a still image."
        )

    return report
