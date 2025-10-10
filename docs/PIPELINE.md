# Pipeline Architecture

## DAG Structure

The recsys pipeline follows a 2-3 stage DAG (Directed Acyclic Graph):

```
┌─────────────────┐
│  Input: Resume  │
│  (data/)        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  STEP 1: Skill Extraction       │
│  (extract.py)                   │
│                                 │
│  - Load resume text             │
│  - Use LLM to extract skills    │
│  - Iterative masking algorithm  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Output: Extracted Skills       │
│  (output/extracted_skills.json) │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Input: Task Definition         │
│  (data/task_*.json)             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  STEP 2: Task Matching          │
│  (taskmatch.py)                 │
│                                 │
│  - Compute embeddings           │
│  - Calculate similarities       │
│  - Evaluate dependency graph    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Output: Match Results          │
│  (output/task_skill_matches.json)│
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  STEP 3 (Optional): Sort/Rank   │
│  (sort.py)                      │
│                                 │
│  - TrueSkill pairwise ranking   │
│  - LLM-based comparisons        │
│  - Rank by evidence strength    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Output: Ranked Skills          │
│  (output/sorted_skills.json)    │
└─────────────────────────────────┘
```

## Execution Modes

### 1. Full Pipeline (DAG Mode)
```bash
uv run main.py all               # Extract + Match
uv run main.py all --with-sort   # Extract + Match + Sort
```
Runs steps in sequence, passing data between them in-memory.

### 2. Individual Steps (Isolated Mode)
```bash
uv run main.py extract  # Step 1 only
uv run main.py match    # Step 2 only (requires step 1 output)
uv run main.py sort     # Step 3 only (requires step 1 output)
```

### 3. Direct Script Execution
```bash
uv run extract.py       # Run extraction independently
uv run taskmatch.py     # Run matching independently
uv run sort.py          # Run sorting independently
```

## Data Flow

1. **Input Data** (`data/`)
   - `resume.txt` - Candidate resume
   - `task_*.json` - Task definitions with dependency graphs

2. **Processing**
   - Step 1 transforms resume text → structured skill list
   - Step 2 transforms (skills + task definition) → compatibility scores
   - Step 3 (optional) transforms (skills + resume) → ranked skills by evidence strength

3. **Output Data** (`output/`)
   - `extracted_skills.json` - List of extracted skills
   - `task_skill_matches.json` - Ranked compatibility scores
   - `sorted_skills.json` - (Optional) Skills ranked by evidence strength

## Dependencies Between Steps

- **Step 1 (Extract)** has no dependencies (can run standalone)
- **Step 2 (Match)** depends on Step 1 output (`extracted_skills.json`)
- **Step 3 (Sort)** depends on Step 1 output (`extracted_skills.json`) and original resume

When running `main.py all`, data flows in-memory between steps for efficiency.
When running steps individually, data is passed via JSON files.

## TrueSkill Ranking (Step 3)

The sorting step uses a **TrueSkill tournament** to rank skills:

1. Performs pairwise skill comparisons using LLM
2. Asks which skill is better supported by resume evidence
3. Uses quadruple verification for consistency
4. Updates TrueSkill ratings (μ = mean, σ = uncertainty)
5. Final ranking based on conservative estimate (μ - 3σ)

This provides a robust ranking of skills by evidence strength in the resume.

