# HardScope

HardScope is a local static security analysis tool for AI instructions and agent prompts.

It scans Markdown files, sends them to an OpenAI-compatible model, and identifies issues such as:

- ambiguous destructive actions
- unbounded scope
- implicit trust boundaries

## Installation

```bash
pip install git+https://github.com/<your-username>/Hard-Scope-CLI.git
```

## Usage

```bash
hardscope path/to/markdown --output hardscope_report.json --model qwen/qwen3-8b --api-key YOUR_KEY --api-base http://127.0.0.1:11434/v1
```

You can also run the project directly from a checkout:

```bash
python hardscope.py path/to/markdown --output hardscope_report.json
```
