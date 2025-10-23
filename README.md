# escoSkill2

AI-powered skill extraction and matching system using LLMs, embeddings, and ontology-based fuzzy matching.

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt  # or use uv/poetry

# Set API key
export OPENROUTER_API_KEY="your-key-here"
```

### Basic Workflow

```bash
# 1. Extract skills from resume
python extract.py --resume resumes/resume.txt

# 2. Sort skills by evidence strength
python sort.py --resume resumes/resume.txt

# 3. Match skills to task requirements
python taskmatch.py --resume resumes/resume.txt
```

All outputs are automatically organized in `data/{resume_name}/`

## Scripts

### `extract.py` - Skill Extraction

Extract skills from a resume using iterative LLM prompting.

```bash
# Basic extraction
python extract.py --resume resumes/ebony_moore.txt

# With verification gate (filters inferred skills)
python extract.py --resume resumes/ebony_moore.txt --verify-evidence

# Custom model
python extract.py --resume resumes/ebony_moore.txt --model google/gemini-2.5-flash-lite
```

**Output:** `data/{resume_name}/extracted_skills.json`

### `sort.py` - Skill Ranking

Rank skills using TrueSkill algorithm with quadruple LLM verification.

```bash
# Sort skills by evidence strength
python sort.py --resume resumes/ebony_moore.txt

# Custom number of tournament rounds
python sort.py --resume resumes/ebony_moore.txt --rounds 5
```

**Input:** `data/{resume_name}/extracted_skills.json`
**Output:** `data/{resume_name}/sorted_skills.json`

### `taskmatch.py` - Task Matching

Match candidate skills to task requirements using embedding similarity and dependency graphs.

```bash
# Basic matching
python taskmatch.py --resume resumes/ebony_moore.txt

# With LLM gate (filters homonyms)
python taskmatch.py --resume resumes/ebony_moore.txt --use-llm-gate

# With ontology-based fuzzy matching
python taskmatch.py --resume resumes/ebony_moore.txt --use-ontologies --use-esco

# With both LLM gate and ontologies
python taskmatch.py --resume resumes/ebony_moore.txt --use-llm-gate --use-ontologies --use-esco
```

**Input:** `data/{resume_name}/extracted_skills.json`
**Output:** `data/{resume_name}/task_skill_matches.json`

## Features

### 🔄 Async Concurrency

All LLM API calls use `asyncio` and `aiohttp` for parallel execution:

- **extract.py**: All skill verification filters run concurrently
- **sort.py**: 4 verification questions per comparison run in parallel (4x speedup)
- **taskmatch.py**: All filters per skill pair run concurrently

### 🧠 LLM Gates

Configurable LLM-based verification filters to improve accuracy:

- **Extract filters** (`extract_filters.json`): Filter out inferred skills, keep only directly supported
- **Match filters** (`taskmatch_filters.json`): Detect homonyms vs synonyms

### 🔗 Ontology Integration

Fuzzy matching using domain ontologies:

- **ESCO**: Skills, competences, occupations
- **Gene Ontology**: Biological processes, molecular functions
- **ChEBI**: Chemical entities (WARNING: 2-3GB RAM)
- **MeSH**: Medical terminology
- **Cell Ontology**: Cell types and lineages

See [[docs/ontology-fuzzing|Ontology Fuzzy Matching]] for details.

## Project Structure

```
/resumes/               # Resume files (gitignored except resume.txt)
  resume.txt           # Example resume (tracked in git)

/data/                 # Output artifacts (gitignored)
  {resume_name}/
    extracted_skills.json
    sorted_skills.json
    task_skill_matches.json

/docs/                 # Documentation
  claude.md            # Claude-specific notes
  data-structure.md    # Data organization details
  ontology-fuzzing.md  # Ontology fuzzy matching guide
  summary.md           # Project summary

/ontologies/           # Ontology data files (gitignored, download locally)
  esco/
  gene_ontology/
  ...

extract_filters.json   # LLM verification filters for extraction
taskmatch_filters.json # LLM verification filters for matching
```

See [[docs/data-structure|Data Structure]] for more details.

## Configuration

### LLM Models

Default models can be changed via CLI arguments:

```bash
# Extract with different model
python extract.py --model mistralai/mistral-7b-instruct:free

# All scripts support --model parameter
```

### Verification Filters

Edit `extract_filters.json` and `taskmatch_filters.json` to customize LLM verification logic:

```json
{
  "model": "google/gemini-flash-1.5",
  "cache_results": true,
  "filters": [
    {
      "name": "direct_evidence_check",
      "enabled": true,
      "system_prompt": "...",
      "user_prompt_template": "...",
      "pass_condition": "YES",
      "temperature": 0.1,
      "max_tokens": 10
    }
  ]
}
```

## Documentation

- [[docs/claude|Claude Notes]] - Development notes and context
- [[docs/data-structure|Data Structure]] - File organization and paths
- [[docs/ontology-fuzzing|Ontology Fuzzy Matching]] - Using ontologies for matching
- [[docs/summary|Project Summary]] - High-level overview

## Advanced Usage

### Custom Paths

Override automatic path generation:

```bash
# Extract to custom location
python extract.py --resume resumes/custom.txt --output my_output.json

# Sort with custom input/output
python sort.py --input my_skills.json --output my_sorted.json

# Match with custom paths
python taskmatch.py --skills-file my_skills.json --output-file my_matches.json
```

### Multiple Resumes

Process different resumes without conflicts:

```bash
# Resume 1
python extract.py --resume resumes/candidate_a.txt
python sort.py --resume resumes/candidate_a.txt
python taskmatch.py --resume resumes/candidate_a.txt

# Resume 2 (outputs go to different directory)
python extract.py --resume resumes/candidate_b.txt
python sort.py --resume resumes/candidate_b.txt
python taskmatch.py --resume resumes/candidate_b.txt
```

Outputs automatically organized in `data/candidate_a/` and `data/candidate_b/`

## Development

See [[docs/claude|claude.md]] for development notes and Claude Code integration.

## License

[Add license information]
