"""Download the Tiny Shakespeare corpus into data/.

Tiny Shakespeare is ~1 MB of concatenated Shakespeare plays — the canonical nanoGPT starter
dataset. See docs/02_dataset.md.

Run:

    uv run python scripts/prepare_data.py

The file is saved to data/input.txt (gitignored). Re-running is a no-op if it already exists.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Use the OS certificate store so downloads work behind SSL-inspecting proxies / antivirus
# (the same "UnknownIssuer" issue we fixed for uv). No-op if truststore isn't installed.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path(__file__).resolve().parent.parent / "data" / "input.txt"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"Already present: {OUT} ({OUT.stat().st_size:,} bytes)")
        return

    print(f"Downloading Tiny Shakespeare from\n  {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=30) as resp:
            OUT.write_bytes(resp.read())
    except Exception as exc:  # noqa: BLE001 - surface a helpful message
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "If this is a TLS/certificate error, make sure 'truststore' is installed "
            "(uv add truststore) so the OS certificate store is used.",
            file=sys.stderr,
        )
        sys.exit(1)

    text = OUT.read_text(encoding="utf-8")
    print(f"Saved {OUT} ({len(text):,} chars, {len(set(text))} unique characters).")
    print(f"First 120 chars:\n{text[:120]!r}")


if __name__ == "__main__":
    main()
