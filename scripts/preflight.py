"""Quick environment check. Run before app.py to confirm Ollama + model are ready.

    uv run python scripts/preflight.py
    uv run python scripts/preflight.py --model qwen3-vl:8b
"""

from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:8b")
    args = ap.parse_args()

    try:
        import ollama
    except ImportError:
        print("✗ `ollama` package not installed. Run: uv pip install -r requirements.txt")
        sys.exit(1)

    # Is the server up?
    try:
        info = ollama.list()
    except Exception as e:  # noqa: BLE001
        print(f"✗ Ollama server not reachable ({e}).")
        print("  Start it: `ollama serve` (or launch the Ollama app).")
        sys.exit(1)
    print("✓ Ollama server reachable")

    # Is the model pulled?
    names = []
    for m in info.get("models", []):
        names.append(m.get("model") or m.get("name", ""))
    if any(args.model in n for n in names):
        print(f"✓ Model '{args.model}' is available")
    else:
        print(f"✗ Model '{args.model}' not found. Pull it: `ollama pull {args.model}`")
        print(f"  Installed: {', '.join(names) or '(none)'}")
        sys.exit(1)

    # Tiny smoke test that vision actually responds.
    try:
        r = ollama.chat(model=args.model,
                        messages=[{"role": "user", "content": "Reply with the single word OK."}],
                        options={"temperature": 0})
        print(f"✓ Model responds: {r['message']['content'].strip()[:40]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"✗ Model call failed: {e}")
        sys.exit(1)

    print("\nAll good — run: uv run python app.py")


if __name__ == "__main__":
    main()
