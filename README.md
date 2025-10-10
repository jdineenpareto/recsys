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

### Option 1: Run Complete Pipeline (Recommended)

Run the entire workflow (extract + match):

```bash
uv run main.py all                                          # Use defaults
uv run main.py all --with-sort                              # Include skill ranking
uv run main.py all --task task_linux_monitoring.json        # Specify task
uv run main.py all --resume data/resume.txt --task task_linux_monitoring.json
```

### Option 2: Run Individual Steps

Extract skills from resume:

```bash
uv run main.py extract                            # Use default resume
uv run main.py extract --resume data/resume.txt
```

Match skills to tasks:

```bash
uv run main.py match                              # Use default task
uv run main.py match --task task_linux_monitoring.json
```

Sort/rank skills by evidence strength (TrueSkill):

```bash
uv run main.py sort                               # Use default settings
uv run main.py sort --rounds 3                    # Specify number of rounds
```

List available tasks:

```bash
uv run main.py list-tasks
```

### Option 3: Run Scripts Directly

You can also run the scripts directly:

```bash
uv run extract.py                                 # Extract skills
uv run taskmatch.py                               # Match to default task
uv run taskmatch.py task_linux_monitoring.json    # Match to specific task
uv run sort.py                                    # Rank skills by evidence
```

## Task Definitions

Tasks are defined in JSON files in the `data/` directory. Each task file specifies:
- Required skills (input nodes)
- A dependency graph showing how skills combine
- Logic operations: `min` (AND), `max` (OR), `avg` (average)

Example: `data/task_linux_monitoring.json`

To create a new task, copy an existing task file and modify the skills and dependency graph.

## Pipeline Architecture

The system works as a 2-3 stage DAG (Directed Acyclic Graph):

1. **Extract** → Transforms resume text into structured skills list
2. **Match** → Compares skills against task requirements using semantic similarity
3. **Sort** (optional) → Ranks skills by evidence strength using TrueSkill algorithm

See [PIPELINE.md](PIPELINE.md) for detailed architecture documentation.

## Output Files

All generated files are saved to the `output/` directory (created automatically):
- `output/extracted_skills.json` - Skills extracted from the resume
- `output/task_skill_matches.json` - Task matching results with compatibility scores
- `output/sorted_skills.json` - (Optional) Skills ranked by evidence strength using TrueSkill
- `output/cost_report_*.json` - API cost breakdown and token usage

These files are idempotent - you can safely delete the entire `output/` directory and regenerate them.

## Cost Tracking

The system automatically tracks OpenRouter API costs:
- **Token Usage**: Tracks prompt and completion tokens for each API call
- **Pricing**: Fetches real-time pricing from OpenRouter's models API
- **Cost Breakdown**: Shows costs by operation (extraction, comparison) and by model
- **Reports**: Saves detailed JSON reports with per-call breakdowns

Cost summary is printed at the end of each run and saved to:
- `output/cost_report_extract.json` - Extraction costs
- `output/cost_report_sort.json` - Sorting costs
- `output/cost_report_pipeline.json` - Full pipeline costs (when using `all` command)

