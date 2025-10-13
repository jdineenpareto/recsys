# Ontology-Based Fuzzy Matching for Task-Skill Compatibility

## Overview

This document describes the high-level principles for combining skill decomposition from resumes with task input nodes using ontology-based fuzzy matching. The goal is to enable **partial activation** of task requirements based on ontological relationships between concepts.

## Core Principle: Ontological Proximity

**Key Insight**: Skills and concepts exist in a structured knowledge space. Ontologies encode these relationships, allowing us to:

1. **Recognize related concepts** even when exact terms don't match
2. **Apply weighted partial credit** based on relationship type and distance
3. **Expand task requirements** to include semantically related concepts

## Relationship Types and Activation Rules

### 1. Hierarchical Relationships (Parent-Child)

#### Parent → Child (Superset → Subset)
- **Example**: "Physics" (task requirement) → "Quantum Field Theory" (candidate skill)
- **Principle**: If a task requires a broad concept, having deep expertise in a specific area provides **partial activation**
- **Activation Weight**: `relation_weight × 0.7` (default: 0.35)
- **Rationale**: The candidate has specialized knowledge that is predictive of understanding the broader domain

```
Task Requirement: "Physics"
Candidate Has: "Quantum Field Theory"
Ontology: Physics ──is_a──> Quantum Mechanics ──is_a──> Quantum Field Theory
Result: Partial match (0.35 × similarity score)
```

#### Child → Parent (Subset → Superset)
- **Example**: "Machine Learning" (task requirement) → "Artificial Intelligence" (candidate skill)
- **Principle**: If a task requires specific knowledge, having broader expertise provides **stronger partial activation**
- **Activation Weight**: `relation_weight × 0.9` (default: 0.45)
- **Rationale**: Broader knowledge likely encompasses the specific requirement, though may lack depth

```
Task Requirement: "Machine Learning"
Candidate Has: "Artificial Intelligence"
Ontology: AI ──has_part──> Machine Learning
Result: Strong partial match (0.45 × similarity score)
```

### 2. Sibling Relationships (Shared Parent)

- **Example**: "Python" (task requirement) → "JavaScript" (candidate skill)
- **Principle**: Skills sharing a common parent category are **weakly related**
- **Activation Weight**: `sibling_weight` (default: 0.3)
- **Rationale**: Transferable skills and mental models, but not direct equivalence

```
Task Requirement: "Python"
Candidate Has: "JavaScript"
Ontology: Programming Languages ──has_part──> Python
          Programming Languages ──has_part──> JavaScript
Result: Weak partial match (0.3 × similarity score)
```

### 3. Distance Decay

All relationship weights decay with ontological distance:

```
final_weight = base_weight × (1.0 / distance)
```

Where `distance` is the number of hops in the ontology graph.

**Example**:
- Distance 1: Full relationship weight
- Distance 2: 50% of relationship weight
- Distance 3: 33% of relationship weight

## Matching Modes

### Mode 1: Expand (Task Node Expansion)

**Strategy**: Expand each task input node with ontologically-related concepts before matching.

**Process**:
1. For each task requirement (e.g., "Linux Administration"):
   - Find best match in loaded ontologies
   - Traverse relationships up to `max_depth` (default: 2)
   - Collect related concepts with weights
2. Compute embeddings for both original and expanded concepts
3. For each candidate skill:
   - Match against original requirement (weight: 1.0)
   - Match against related concepts (weighted)
   - Take maximum weighted score

**Use Case**: When task requirements are under-specified and you want to recognize broader/narrower expertise.

**Example**:
```
Task Node: "Monitoring Tools"
Expands to:
  - "Monitoring Tools" (weight: 1.0)
  - "Nagios" (weight: 0.63) [child concept]
  - "Zabbix" (weight: 0.63) [child concept]
  - "Prometheus" (weight: 0.63) [sibling concept]
  - "System Monitoring" (weight: 0.49) [parent concept]

Candidate Skill: "Nagios Experience"
Match: Hits "Nagios" with high similarity × 0.63 weight
Result: Boosted score beyond pure embedding similarity
```

### Mode 2: Boost (Relationship-Based Score Enhancement)

**Strategy**: Use ontology to boost similarity scores for related concepts.

**Process**:
1. Compute base embedding similarity between task nodes and candidate skills
2. For each (task_node, candidate_skill) pair:
   - Find both in ontologies
   - Check if they're related via ontology relationships
   - If related, boost the base score using: `new_score = base + (1 - base) × boost`
3. Boost is calculated from relationship type and distance

**Use Case**: When you want to preserve base embedding scores but enhance them for ontologically-verified relationships.

**Example**:
```
Task Node: "System Administration"
Candidate Skill: "Linux Administration"
Base Embedding Similarity: 0.65

Ontology Lookup:
  - "System Administration" ──has_part──> "Linux Administration"
  - Relationship: parent→child, Distance: 1
  - Boost: 0.5 × 0.9 × (1/1) = 0.45

Boosted Score: 0.65 + (1 - 0.65) × 0.45 = 0.81
```

### Mode 3: Hybrid (Combined Expansion + Boosting)

**Strategy**: Apply both expansion and boosting for maximum fuzzy matching.

**Process**:
1. Expand task nodes with related concepts
2. Compute base similarities
3. Apply relationship-based boosting
4. Take maximum across all matching pathways

**Use Case**: Aggressive fuzzy matching when you want to recognize any plausible skill overlap.

**Caution**: Can produce false positives if weights are too high. Requires careful tuning.

## Scoring Formula

### Base Embedding Similarity
```
similarity = cosine_similarity(candidate_skill_embedding, task_node_embedding)
normalized_similarity = (similarity + 1) / 2  # Map [-1,1] to [0,1]
```

### Ontology Boost (Mode: boost or hybrid)
```
boost_factor = relationship_weight × (1.0 / distance)

where relationship_weight = {
    0.7 × relation_weight  if parent→child
    0.9 × relation_weight  if child→parent
    sibling_weight         if sibling
}

boosted_score = base_score + (1 - base_score) × boost_factor
```

### Expansion Score (Mode: expand or hybrid)
```
For each related_concept with weight w:
    concept_score = cosine_similarity(candidate_skill, related_concept)
    weighted_score = concept_score × w

final_score = max(base_score, max(weighted_scores))
```

## Parameter Tuning Guidelines

### `--relation-weight` (default: 0.5)
- **Range**: 0.0 to 1.0
- **Low (0.2-0.4)**: Conservative fuzzy matching, only strong relationships
- **Medium (0.4-0.6)**: Balanced recognition of related concepts
- **High (0.6-0.9)**: Aggressive matching, recognizes distant relationships

### `--sibling-weight` (default: 0.3)
- **Range**: 0.0 to 1.0
- **Low (0.1-0.3)**: Sibling concepts provide weak evidence
- **Medium (0.3-0.5)**: Moderate transferability between siblings
- **High (0.5-0.7)**: Strong assumption of skill transferability

### `--min-ontology-sim` (default: 0.4)
- **Range**: 0.0 to 1.0
- **Low (0.2-0.3)**: Include loosely-related ontology matches
- **Medium (0.4-0.6)**: Only include moderately similar concepts
- **High (0.6-0.9)**: Strict ontology matching, near-exact terms only

### `--ontology-depth` (default: 2)
- **Range**: 1 to 5
- **Depth 1**: Only immediate neighbors (direct parents/children)
- **Depth 2**: Two hops (parents of parents, siblings of parents, etc.)
- **Depth 3+**: Potentially noisy, includes distant relationships

## Real-World Examples

### Example 1: DevOps Task

```bash
python taskmatch.py --use-ontologies --use-esco \
  --ontology-mode boost --relation-weight 0.6
```

**Task Requirements**:
- "Kubernetes"
- "Docker"
- "CI/CD"

**Candidate Skills**:
- "Container Orchestration" → Matches "Kubernetes" via parent→child
- "Containerization" → Matches "Docker" via parent→child
- "Jenkins Pipeline" → Matches "CI/CD" via child→parent

**Result**: Candidate recognized as qualified despite not using exact terminology.

### Example 2: Bioinformatics Task

```bash
python taskmatch.py --use-ontologies --use-go --use-mesh \
  --ontology-mode expand --relation-weight 0.7
```

**Task Requirements**:
- "Gene Expression Analysis"
- "RNA Sequencing"

**Candidate Skills**:
- "Transcriptomics" → Expands to include RNA-Seq
- "Differential Gene Expression" → Matches via specialization

**Result**: Ontology recognizes that transcriptomics expertise covers RNA-Seq requirements.

### Example 3: Aggressive Fuzzy Matching

```bash
python taskmatch.py --use-ontologies --use-all-ontologies \
  --ontology-mode hybrid --relation-weight 0.8 --sibling-weight 0.5
```

**Use Case**: Exploratory matching to find candidates with any plausible skill overlap, accepting higher false positive rate in exchange for finding hidden gems.

## Integration with Task Dependency Graph

The ontology fuzzy matching operates at the **input node level**:

1. **Input Node Scores** are computed with ontology-enhanced matching
2. **Graph Operations** (MIN/MAX/AVG) remain unchanged
3. **Final Score** reflects both graph structure and ontology relationships

This means:
- A candidate with ontologically-related skills can partially satisfy input nodes
- The dependency graph still enforces task requirements (AND/OR gates)
- The final score represents true task compatibility, not just skill overlap

## Ontology Selection by Domain

### ESCO (Skills/Competences)
- **Best for**: General professional skills, IT skills, business competences
- **Examples**: "Project Management", "Python Programming", "Data Analysis"

### Gene Ontology (GO)
- **Best for**: Biological processes, molecular functions, cellular components
- **Examples**: "DNA Replication", "Protein Synthesis", "Cell Division"

### MeSH (Medical Terminology)
- **Best for**: Medical concepts, diseases, treatments
- **Examples**: "Cardiovascular Disease", "Clinical Trials", "Patient Care"

### ChEBI (Chemical Entities)
- **Best for**: Chemical compounds, molecular structures
- **Examples**: "Small Molecule Drug Design", "Organic Synthesis"
- **Warning**: Uses 2-3GB RAM

### Cell Ontology
- **Best for**: Cell types, lineages, cellular contexts
- **Examples**: "Stem Cell Biology", "T Cell Function"

## Performance Considerations

### Speed vs. Accuracy Tradeoffs

**Without Ontologies** (baseline):
- Fast: ~30 seconds for 600 skills
- Relies purely on semantic embeddings

**With Ontologies - Boost Mode**:
- Moderate: ~2-5 minutes (depends on ontology size)
- Most accurate for verified relationships

**With Ontologies - Expand Mode**:
- Slower: ~3-8 minutes (many embedding computations)
- Best coverage for under-specified requirements

**With Ontologies - Hybrid Mode**:
- Slowest: ~5-15 minutes
- Maximum fuzzy matching capability

### Memory Usage

- **Base System**: ~500MB (embeddings + skill list)
- **+ ESCO**: +200MB
- **+ GO**: +500MB
- **+ MeSH**: +300MB
- **+ ChEBI**: +2-3GB (only use if needed!)

## Future Extensions

### 1. Ontology Weight Learning
Learn optimal `relation_weight` and `sibling_weight` from historical hiring data.

### 2. Multi-Hop Reasoning
Use graph neural networks to reason over multiple ontology hops simultaneously.

### 3. Custom Ontology Integration
Allow users to provide domain-specific ontologies in OWL/RDF format.

### 4. Confidence Intervals
Provide uncertainty estimates for fuzzy matches based on ontology coverage and match quality.

## Conclusion

Ontology-based fuzzy matching bridges the gap between exact skill matching and pure semantic similarity. By encoding domain knowledge about concept relationships, we can:

- Recognize qualified candidates who use different terminology
- Apply principled partial credit for related expertise
- Reduce false negatives from under-specified requirements
- Maintain interpretability through explicit relationship types

The key is **tuning parameters to match your tolerance for false positives vs. false negatives** in your specific hiring or matching context.
