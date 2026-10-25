#!/usr/bin/env python3
"""Compatibility wrapper for the HardScope CLI.

This engine ingests Markdown content, sends it to an OpenAI-compatible
reasoning endpoint in strict JSON mode, and enriches each finding with a
character-range location computed against the raw source text.
"""

from __future__ import annotations

from hardscope.cli import main


if __name__ == "__main__":
    raise SystemExit(main())