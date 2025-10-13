# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a skill extraction and matching system that uses LLMs and semantic embeddings to analyze resumes against job requirements. The system has four main components:

1. **extract.py** - Extract skills from resumes using iterative LLM prompting
2. **sort.py** - Rank skills by evidence strength using TrueSkill algorithm
3. **taskmatch.py** - Evaluate candidate compatibility using dependency graph matching
4. **ontologies/api/** - Python APIs for accessing biomedical and skills ontologies with RAG-based matching

## Running the Scripts

### Setup
```bash
# IMPORTANT: This project uses uv for dependency management
# NEVER use pip install - always modify pyproject.toml and run uv sync

# Install dependencies (Python 3.12+ required)
# Note: Configured for Linux/WSL with PyTorch CUDA 12.8 for RTX 5070 Ti (sm_120)
uv sync

# Set API key for extract.py, sort.py, and taskmatch.py LLM gate
export OPENROUTER_API_KEY=your_key_here
```

### Platform-Specific Notes

**Linux/WSL (Recommended for GPU):**
- PyTorch configured with CUDA 12.8 for RTX 5070 Ti support (sm_120 compute capability)
- FAISS-GPU enabled for accelerated vector search
- Requires NVIDIA driver ≥570 on Windows host for WSL2

**Windows (CPU Only):**
- Current pyproject.toml uses Linux markers - GPU support requires WSL2
- CPU mode works but embedding computation is slower

### Dependency Management with uv

**CRITICAL: Always use uv, never pip**

```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Remove dependency
uv remove package-name

# Update dependencies
uv sync

# WRONG - Never do this:
# pip install package-name
# uv pip install package-name
```

**To change PyTorch version or other complex dependencies:**
1. Edit `pyproject.toml` directly
2. Update the `[[tool.uv.index]]` sections if needed
3. Run `uv sync` to apply changes

### Execution Order

#### Core Skill Matching Pipeline
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

# Step 3 (Advanced): Match with ontology fuzzy matching
python taskmatch.py --use-ontologies --use-esco
# Uses ESCO ontology to recognize related skills (e.g., "DevOps" partially matches "CI/CD")

# Step 3 (Advanced): Match with LLM gate to filter homonyms
python taskmatch.py --use-llm-gate
# Requires OPENROUTER_API_KEY - filters out false matches like "Java" vs "JavaScript"

# Step 3 (Advanced): Combine ontology + LLM gate
python taskmatch.py --use-ontologies --use-esco --use-llm-gate
```

#### Ontology API Usage
```bash
# Test ontology APIs
python test_ontology_api.py

# Run basic examples
python ontologies/api/examples/example_basic_usage.py

# Run unified manager examples
python ontologies/api/examples/example_unified_manager.py
```

## Architecture

### Overall System Architecture

The system consists of two major subsystems:

1. **Skill Extraction & Matching Pipeline** (extract.py, sort.py, taskmatch.py)
2. **Ontology Knowledge Base** (ontologies/api/)

```
                    ┌─────────────────────────────────────┐
                    │   Skill Extraction Pipeline         │
                    │                                     │
resume.txt ──────> extract.py ──> extracted_skills.json ──> taskmatch.py ──> task_skill_matches.json
                       │                                         │
                       │                                         │
                       └────> sort.py ──> sorted_skills.json    │
                                                                 │
                                                                 │
                    ┌─────────────────────────────────────┐     │
                    │   Ontology Knowledge Base           │     │
                    │                                     │     │
                    │  ontologies/                        │<────┘
                    │  ├── gene_ontology/                 │  (can be used
                    │  ├── chebi/                         │   to enhance
                    │  ├── esco/                          │   matching)
                    │  ├── cell_ontology/                 │
                    │  ├── mesh/                          │
                    │  └── api/                           │
                    │      ├── GeneOntologyAPI            │
                    │      ├── ChEBIAPI                   │
                    │      ├── ESCOAPI                    │
                    │      ├── CellOntologyAPI            │
                    │      ├── MeSHAPI                    │
                    │      └── OntologyManager            │
                    └─────────────────────────────────────┘
```

### Key Architectural Patterns

#### 1. Adaptive Masking in extract.py
The skill extractor uses a counter-based masking strategy:
- Counter increases when new skill found, decreases on failure
- Masks N random skills (where N = counter) to force exploration
- Stops after 10 consecutive failures at counter=0
- Randomizes skill order each prompt to prevent pattern bias

**Why this matters:** Don't simplify the masking logic - it prevents the LLM from just confirming existing skills without searching the resume.

**Implementation details:**
- Lines 45-58: `mask_random_skills()` randomly removes N skills from the list
- Line 68: Skills are masked before each prompt using current counter value
- Lines 86-95: Counter decrements when "NONE" returned, increments on success
- Lines 97-106: Counter decrements when skill already exists (duplicate detection)
- Lines 64-66: Halts after `zero_counter_streak >= 10` consecutive failures

#### 2. TrueSkill with Quadruple Verification in sort.py
For each skill comparison, asks 4 logically-related questions:
- "Is A > B?"
- "Is B > A?"
- "Is NOT (A > B)?"
- "Is NOT (B > A)?"

Verifies answers match expected patterns: `(T,F,F,T)` for A wins, `(F,T,T,F)` for B wins.

**Why this matters:** The verification pattern is critical for catching LLM inconsistencies. Don't remove or simplify the 4-question structure.

**Implementation details:**
- Lines 65-85: `ask_comparison()` handles all 4 question types
- Lines 87-119: `compare_skills_verified()` enforces consistency checks
- Lines 103-109: A wins pattern check
- Lines 111-116: B wins pattern check
- Line 118: Returns None if verification fails (match is skipped)
- Lines 129-164: TrueSkill rating updates use Bayesian skill estimation with mu (mean) and sigma (uncertainty)

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

**Implementation details:**
- Lines 14-106: `TASK_SKILL_GRAPH` defines current task
- Lines 127-157: `compute_input_node_scores()` creates embedding similarity matrix
- Lines 160-205: `evaluate_graph_node()` recursively evaluates with memoization
- Lines 196-203: Operations applied via numpy (min/max/reduce/mean)
- Lines 355-357: Final score is MAX across all skills (not average)

### Semantic Similarity
taskmatch.py uses sentence-transformers (`all-MiniLM-L6-v2`) for embeddings:
- Computes embeddings for input nodes and extracted skills
- Dot products between normalized embeddings = cosine similarity
- Maps to [0,1] range: `(similarity + 1) / 2`

**Why transformers:** Captures semantic similarity, not just keyword matching. "Linux Administration" matches well with "System Administration" despite different words.

#### 4. Ontology-Based Fuzzy Matching in taskmatch.py (NEW)

**Purpose:** Recognize related skills even when exact terms don't match (e.g., "Physics" partially matches "Quantum Field Theory").

**Three Matching Modes:**

1. **Expand Mode** (`--ontology-mode expand`):
   - Expands task input nodes with ontologically-related concepts
   - Parent concepts (superset) get weight × 0.7
   - Child concepts (subset) get weight × 0.9
   - Takes maximum score across original and related concepts

2. **Boost Mode** (`--ontology-mode boost`):
   - Computes base embedding similarity first
   - Boosts scores for pairs found related in ontology
   - Boost formula: `new_score = base + (1 - base) × boost_factor`

3. **Hybrid Mode** (`--ontology-mode hybrid`):
   - Combines both expansion and boosting
   - Most aggressive fuzzy matching

**Implementation details:**
- Lines 561-633: `expand_task_nodes_with_ontology()` - Node expansion logic
- Lines 636-724: `compute_ontology_boosted_scores()` - Relationship-based score boosting
- Lines 514-558: `initialize_ontology_manager()` - Loads selected ontologies
- Configurable weights: `--relation-weight` (default 0.5), `--sibling-weight` (default 0.3)

**See also:** `ontologyfuzz.md` for detailed principles and examples

#### 5. LLM Gate for Homonym Filtering in taskmatch.py (NEW)

**Purpose:** Distinguish true synonyms from homonyms using LLM verification to prevent false matches.

**How it works:**
- Applied only to high-similarity matches (above `--llm-gate-threshold`, default 0.7)
- Sends pair to OpenRouter API with TRUE/FALSE prompt
- Caches results to avoid redundant API calls
- Penalizes homonym scores by 50% (reduces by half)

**Examples:**
- "Python Programming" vs "Python Development" → SYNONYM (accepted)
- "Java" vs "JavaScript" → HOMONYM (penalized)
- "Machine Learning" vs "ML" → SYNONYM (accepted)

**Implementation details:**
- Lines 10-106: `LLM_Gate` class with caching and error handling
- Lines 260-280: Applied in `compute_input_node_scores()` after similarity matrix computation
- Temperature 0.1 for consistent answers
- Graceful fallback if API key missing or error occurs

**Command-line options:**
- `--use-llm-gate`: Enable LLM-based verification
- `--llm-gate-threshold 0.7`: Only check matches above this similarity
- `--llm-gate-model`: Choose OpenRouter model (default: mistral-7b)

#### 6. Ontology API Architecture (ontologies/api/)

The ontology APIs provide structured access to 5 major ontologies with consistent interfaces:

**Class Hierarchy:**
```
BaseOntologyAPI (abstract base)
├── GeneOntologyAPI (OBO parser)
│   └── CellOntologyAPI (reuses OBO parser)
├── ChEBIAPI (OBO parser)
├── ESCOAPI (RDF parser)
└── MeSHAPI (XML/RDF parsers)

OntologyManager (facade pattern)
└── Manages all APIs + cross-ontology search
```

**Core Features:**
- **Node Navigation**: `get_node()`, `get_neighbors()`, `get_node_context()`
- **RAG Matching**: `match_term_to_nodes()` using sentence-transformers
- **Three-level Caching**: Nodes, neighbors, embeddings
- **Lazy Loading**: Embedding models loaded on first RAG query

**Key Implementation Details:**
- `base_ontology.py` lines 103-180: RAG matching pipeline
  - Exact/synonym matching first (score: 1.0/0.95)
  - Falls back to semantic search with cosine similarity
  - Maps similarity from [-1,1] to [0,1]
- `ontology_manager.py` lines 98-132: Cross-ontology term matching
  - Searches across multiple ontologies simultaneously
  - Returns best match globally or per-ontology results
- All APIs implement same interface defined in `BaseOntologyAPI`

**Performance Characteristics:**
- First RAG query: Slow (computes embeddings for all nodes)
- Subsequent queries: Fast (uses cached embeddings)
- Large ontologies: ChEBI ~2-3GB RAM, GO ~500MB, others <500MB

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

### Core Pipeline Files
- **Input:** `resume.txt` - Resume to analyze
- **Intermediate:** `extracted_skills.json` - Skills extracted from resume (599+ skills in current example)
- **Optional:** `sorted_skills.json` - Skills ranked by evidence (from sort.py)
- **Output:** `task_skill_matches.json` - Full compatibility analysis with rankings
- **Reference:** `summary.md` - Detailed explanation of each script and algorithm

### Ontology Files
- **Data:** `ontologies/*/` - Downloaded ontology files (1.6GB total)
  - `gene_ontology/` - GO terms (biological processes, molecular functions)
  - `chebi/` - Chemical entities (247MB OBO, 773MB OWL)
  - `esco/` - Skills, competences, occupations
  - `cell_ontology/` - Cell types and lineages
  - `mesh/` - Medical terminology
- **APIs:** `ontologies/api/` - Python APIs for ontology access
  - `base_ontology.py` - Abstract base with RAG matching
  - `*_api.py` - Individual ontology implementations
  - `ontology_manager.py` - Unified cross-ontology interface
  - `examples/` - Usage examples
  - `README.md` - Complete API documentation

## Important Constraints

- Python 3.12+ required (specified in pyproject.toml)
- OpenRouter API key required for extract.py, sort.py, and taskmatch.py LLM gate (uses Mistral-7B model)
- taskmatch.py basic mode does NOT require API key (uses local transformer models)
- First run of taskmatch.py downloads ~90MB model from HuggingFace
- sort.py is computationally expensive - scales O(n²) with number of skills
- Uses uv for dependency management (uv.lock present)
- GPU acceleration requires Linux/WSL with CUDA 12.8 and RTX 5070 Ti (sm_120) or newer
- numpy<2.0 constraint due to faiss-gpu-cu12 compatibility

## Score Interpretation

taskmatch.py final scores:
- 0.7+ = Excellent match
- 0.5-0.7 = Good match
- 0.3-0.5 = Moderate match
- 0.15-0.3 = Partial match
- <0.15 = Low match

The final score is the maximum compatibility across all extracted skills (not an average).

## Using Ontology APIs

### Quick Start
```python
from ontologies.api import OntologyManager

# Load ontologies
manager = OntologyManager()
manager.load_go(use_basic=True)  # Gene Ontology
manager.load_esco()              # Skills/competences

# Match a term across all loaded ontologies
matches = manager.match_term_across_ontologies(
    "protein synthesis",
    top_k_per_ontology=3,
    min_similarity=0.3
)

# Get best match globally
best = manager.get_best_match_across_ontologies("machine learning")
```

### Integration Points with Core Pipeline

**Option 1: Enhance extract.py**
```python
from ontologies.api import ESCOAPI
esco = ESCOAPI()

# Validate extracted skills against ESCO ontology
for skill in extracted_skills:
    matches = esco.match_term_to_nodes(skill, top_k=1)
    if matches and matches[0].similarity_score > 0.8:
        validated_skill = matches[0].node.id
```

**Option 2: Enhance taskmatch.py**
```python
from ontologies.api import OntologyManager

# Replace manual input_nodes with ontology-validated concepts
manager = OntologyManager()
manager.load_esco()
manager.load_go()

for skill in TASK_SKILL_GRAPH["input_nodes"]:
    match = manager.get_best_match_across_ontologies(skill)
    # Use match.node.definition and relationships in graph
```

### Performance Tips
- Use `load_all(skip_large=True)` to skip ChEBI (saves 2-3GB RAM)
- Use `go-basic.obo` instead of full GO (5x smaller)
- First RAG query per ontology is slow (computes embeddings)
- Load ontologies selectively - only what you need

## Common Pitfalls

### Core Pipeline

1. **Don't simplify the masking in extract.py** - The counter-based randomization is essential
2. **Don't reduce the 4-question verification in sort.py** - It catches LLM inconsistencies
3. **Don't replace AVG with multiplication in taskmatch.py** - Causes score collapse
4. **taskmatch.py line 258** - Cosine similarity is mapped from [-1,1] to [0,1] range
5. **sort.py requires resume.txt as system prompt** - This provides context for all comparisons

### Ontology Fuzzy Matching

1. **Start with conservative weights** - Default `--relation-weight 0.5` is a good starting point
2. **Expand mode is fastest** - Boost mode requires many ontology lookups
3. **Don't combine with LLM gate on first try** - Test ontology matching alone first to tune weights
4. **ESCO is best for general skills** - GO/MeSH/ChEBI are domain-specific (biology/medicine)
5. **Ontology depth affects performance** - Higher depth (>2) causes exponential slowdown

### LLM Gate

1. **Only use for high-similarity matches** - Default threshold 0.7 is optimal
2. **LLM gate is expensive** - Each verification is an API call (use caching)
3. **Cache persists within session** - Results cached in memory, not disk
4. **Graceful degradation** - Missing API key disables gate, doesn't error
5. **Can produce false negatives** - LLM may incorrectly mark synonyms as homonyms

### Ontology APIs

1. **Don't load ChEBI unless necessary** - It's 2-3GB in memory (use skip_large=True)
2. **First RAG query is always slow** - Embeddings are computed lazily, then cached via FAISS
3. **All APIs use same base interface** - Don't duplicate code, extend BaseOntologyAPI
4. **Ontology files must be downloaded** - See ontologies/README.md for instructions
5. **Cross-ontology matching is powerful** - Use OntologyManager for best results
6. **FAISS accelerates vector search** - 10-100x faster than naive dot products after first query
