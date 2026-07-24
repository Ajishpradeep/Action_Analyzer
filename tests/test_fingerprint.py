"""Offline tests for the embedding-powered replay-detection layer.

No API or GPU: we exercise the vector store and replay detection with fixed vectors,
without touching Gemini.

Run:  python tests/test_fingerprint.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import fingerprint as fp
from pipeline.vector_store import VectorStore


def _tmp_store():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)  # start empty
    return VectorStore(path)


def test_vector_store_cosine_and_persistence():
    s = _tmp_store()
    s.add("a", [1.0, 0.0, 0.0], {"user_id": "u1"})
    s.add("b", [0.0, 1.0, 0.0], {"user_id": "u1"})
    m = s.most_similar([0.9, 0.1, 0.0])
    assert m.id == "a" and m.score > 0.98
    # Reload from disk → same data.
    s2 = VectorStore(s.path)
    assert len(s2) == 2
    os.unlink(s.path)


def test_replay_flags_near_identical_but_not_different():
    s = _tmp_store()
    emb1 = [1.0, 0.0, 0.0, 0.0]
    fp.record_submission(s, emb1, user_id="u1", action_label="recycling a bottle")

    # Same event, slightly perturbed (new angle / recompression) → still flagged.
    near = [0.98, 0.02, 0.0, 0.0]
    r = fp.replay_check(s, near, user_id="u1")
    assert r.checked and r.is_duplicate and r.score >= 0.9

    # A genuinely different submission → not a duplicate.
    diff = [0.0, 0.0, 1.0, 0.0]
    r2 = fp.replay_check(s, diff, user_id="u1")
    assert r2.checked and not r2.is_duplicate
    os.unlink(s.path)


def test_replay_is_scoped_per_user():
    s = _tmp_store()
    fp.record_submission(s, [1.0, 0.0], user_id="u1", action_label="using a reusable cup")
    # Same fingerprint, different user → not a replay for that user.
    r = fp.replay_check(s, [1.0, 0.0], user_id="u2")
    assert not r.is_duplicate
    os.unlink(s.path)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
