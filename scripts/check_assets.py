"""Diagram progress tracker — which Excalidraw PNGs still need drawing.

The checklist in docs/assets/README.md is the single source of truth: this script parses
the expected diagram filenames from it and reports which exist in docs/assets/ yet.

Run:
    uv run python scripts/check_assets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
CHECKLIST = ASSETS / "README.md"

# Matches lines like:  - [ ] `05_positional_encoding.png` — sin/cos curves ...
_ITEM = re.compile(r"^-\s*\[.\]\s*`([^`]+\.png)`\s*(?:—|-)?\s*(.*)$")


def main() -> None:
    if not CHECKLIST.exists():
        print(f"Checklist not found: {CHECKLIST}", file=sys.stderr)
        sys.exit(1)

    items: list[tuple[str, str]] = []
    for line in CHECKLIST.read_text(encoding="utf-8").splitlines():
        m = _ITEM.match(line.strip())
        if m:
            items.append((m.group(1), m.group(2).strip()))

    done = 0
    print(f"Diagram progress  (docs/assets/)\n{'-' * 60}")
    for name, desc in items:
        exists = (ASSETS / name).exists()
        done += exists
        mark = "[x]" if exists else "[ ]"
        print(f"{mark} {name:<32} {desc[:40]}")

    total = len(items)
    print("-" * 60)
    print(f"{done} / {total} diagrams done"
          + ("  —  all set!" if done == total else f"  ({total - done} to draw)"))


if __name__ == "__main__":
    main()
