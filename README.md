# Recsys - Resume Skills Extraction and Task Matching

A system for extracting skills from resumes and matching them to tasks using semantic similarity and dependency graphs.

## Setup

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

## Usage

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY='your-api-key-here'
```

Extract skills from resume:

```bash
uv run extract.py
```

Match skills to tasks:

```bash
uv run taskmatch.py
```

