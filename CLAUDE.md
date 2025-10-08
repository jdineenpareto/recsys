# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a skill extraction and matching system that uses LLMs and semantic embeddings to analyze resumes against job requirements. The system has three main components that work sequentially:

1. **extract.py** - Extract skills from resumes using iterative LLM prompting
2. **sort.py** - Rank skills by evidence strength using TrueSkill algorithm
3. **taskmatch.py** - Evaluate candidate compatibility using dependency graph matching

## Running the Scripts

### Setup
```bash
# Install dependencies
pip install sentence-transformers numpy requests transformers

# Set API key for extract.py and sort.py
export OPENROUTER_API_KEY=your_key_here
```

### Execution Order
```bash
# Step 1: Extract skills from resume
python extract.py
# Output: extracted_skills.json

# Step 2 (Optional): Rank skills by evidence
python sort.py
# Output: sorted_skills.json

# Step 3: Match candidate to task
python taskmatch.py
# Output: task_skill_matches.json + console output with final match score
```

## Architecture

### Data Flow
```
resume.txt → extract.py → extracted_skills.json → taskmatch.py → task_skill_matches.json
                                ↓
                            sort.py → sorted_skills.json
```

### Key Architectural Patterns

#### 1. Adaptive Masking in extract.py
The skill extractor uses a counter-based masking strategy:
- Counter increases when new skill found, decreases on failure
- Masks N random skills (where N = counter) to force exploration
- Stops after 10 consecutive failures at counter=0
- Randomizes skill order each prompt to prevent pattern bias

**Why this matters:** Don't simplify the masking logic - it prevents the LLM from just confirming existing skills without searching the resume.

#### 2. TrueSkill with Quadruple Verification in sort.py
For each skill comparison, asks 4 logically-related questions:
- "Is A > B?"
- "Is B > A?"
- "Is NOT (A > B)?"
- "Is NOT (B > A)?"

Verifies answers match expected patterns: `(T,F,F,T)` for A wins, `(F,T,T,F)` for B wins.

**Why this matters:** The verification pattern is critical for catching LLM inconsistencies. Don't remove or simplify the 4-question structure.

#### 3. Dependency Graph Evaluation in taskmatch.py
The task matching uses a graph with three operation types:
- `MIN` (AND gate): Both skills required - takes minimum score
- `MAX` (OR gate): Either skill sufficient - takes maximum score
- `AVG`: Both skills important - takes arithmetic mean

**Critical design decision:** Uses AVG instead of MULTIPLY because multiplying decimals <1 causes scores to drop too rapidly. If you see multiply operations, they were intentionally removed.

Graph structure in `TASK_SKILL_GRAPH`:
```python
{
    "input_nodes": [...],  # Base skills that get similarity scores [0,1]
    "graph": {             # Operations combining nodes
        "node_name": {
            "op": "min"|"max"|"avg",
            "inputs": [...]
        }
    },
    "root": "final_score"  # Entry point for evaluation
}
```

The graph evaluates recursively from root → leaves, with memoization to avoid recomputation.

### Semantic Similarity
taskmatch.py uses sentence-transformers (`all-MiniLM-L6-v2`) for embeddings:
- Computes embeddings for input nodes and extracted skills
- Dot products between normalized embeddings = cosine similarity
- Maps to [0,1] range: `(similarity + 1) / 2`

**Why transformers:** Captures semantic similarity, not just keyword matching. "Linux Administration" matches well with "System Administration" despite different words.

## Modifying Task Definitions

To change the task in taskmatch.py:

1. Update `input_nodes` list with required base skills
2. Define graph operations combining those nodes
3. Set `root` to the final scoring node
4. The script will automatically:
   - Compute embeddings for new input nodes
   - Evaluate the graph structure
   - Output compatibility scores

Example node types:
```python
# AND gate - both required
"core_skills": {"op": "min", "inputs": ["skill_a", "skill_b"]}

# OR gate - either sufficient
"any_scripting": {"op": "max", "inputs": ["bash", "python"]}

# Weighted combination
"combined": {"op": "avg", "inputs": ["skill_a", "skill_b"]}
```

## File Locations

- **Input:** `resume.txt` - Resume to analyze
- **Intermediate:** `extracted_skills.json` - Skills extracted from resume (599+ skills)
- **Optional:** `sorted_skills.json` - Skills ranked by evidence (from sort.py)
- **Output:** `task_skill_matches.json` - Full compatibility analysis with rankings
- **Reference:** `summary.md` - Detailed explanation of each script and algorithm

## Important Constraints

- Python 3.12+ required
- OpenRouter API key required for extract.py and sort.py
- taskmatch.py does NOT require API key (uses local transformer models)
- First run of taskmatch.py downloads ~90MB model from HuggingFace
- sort.py is computationally expensive - scales O(n²) with number of skills

## Score Interpretation

taskmatch.py final scores:
- 0.7+ = Excellent match
- 0.5-0.7 = Good match
- 0.3-0.5 = Moderate match
- 0.15-0.3 = Partial match
- <0.15 = Low match

The final score is the maximum compatibility across all extracted skills (not an average).
