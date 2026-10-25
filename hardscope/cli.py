#!/usr/bin/env python3
"""Command-line interface for HardScope."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .hardscope import DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_SECONDS, analyze_target, save_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HardScope: local static security analysis for AI instructions and agent prompts."
    )
    parser.add_argument("target", help="Path to a Markdown file or a directory containing Markdown files.")
    parser.add_argument("--output", default="hardscope_report.json", help="Where to save the final JSON report.")
    parser.add_argument("--model", default=os.getenv("HARDSCOPE_MODEL", "qwen/qwen3-8b"), help="OpenAI-compatible model name (for Ollama, e.g. qwen3:8b).")
    parser.add_argument("--api-key", default=os.getenv("HARDSCOPE_API_KEY"), help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--api-base", default=os.getenv("HARDSCOPE_API_BASE"), help="Base URL for the OpenAI-compatible API (e.g. http://127.0.0.1:11434/v1).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Seconds to wait for each model response before failing (default: %(default)s).")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum tokens the model may generate per file (default: %(default)s).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        findings = analyze_target(
            args.target,
            args.model,
            args.api_key,
            args.api_base,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        output_file = save_report(findings, args.output)
        unlocated = sum(1 for f in findings if f.get("location", {}).get("start_char", -1) < 0)
        print(f"HardScope report saved to: {output_file}")
        print(f"Findings: {len(findings)}")
        if unlocated:
            print(f"Findings without a source location: {unlocated}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"HardScope error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
