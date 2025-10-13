import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Union, Any, Optional
import argparse
from ontologies.api import OntologyManager
import requests
import os
import hashlib
import pickle
from pathlib import Path

class LLM_Gate:
    """
    LLM-based verification to distinguish between true synonyms and homonyms.
    Uses OpenRouter API to verify if high embedding similarity indicates actual semantic equivalence.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/mistral-7b-instruct:free"):
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            print("WARNING: No OPENROUTER_API_KEY found. LLM gate will be disabled.")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.cache = {}  # Cache results to avoid redundant API calls

    def is_enabled(self) -> bool:
        """Check if LLM gate is enabled (API key is available)"""
        return self.api_key is not None

    def verify_synonym(self, input_node: str, extracted_skill: str, similarity_score: float) -> bool:
        """
        Verify if two terms with high embedding similarity are true synonyms.

        Args:
            input_node: The task requirement term
            extracted_skill: The candidate's skill term
            similarity_score: The embedding similarity score [0,1]

        Returns:
            True if they are true synonyms, False if they are homonyms (unrelated despite high similarity)
        """
        if not self.is_enabled():
            # If no API key, default to accepting high similarity matches
            return True

        # Check cache
        cache_key = f"{input_node}||{extracted_skill}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Construct verification prompt
        system_prompt = """You are a semantic verification system. Your job is to determine if two terms mean the same thing (synonyms) or are unrelated despite sounding similar (homonyms).

Examples:
- "Python Programming" and "Python Development" -> TRUE (synonyms)
- "Java" and "JavaScript" -> FALSE (homonyms - different languages)
- "Machine Learning" and "ML" -> TRUE (synonyms)
- "Windows" and "Windows Operating System" -> TRUE (synonyms)
- "Apple" (fruit) and "Apple" (company) -> FALSE (homonyms - different domains)

Respond with ONLY "TRUE" if they are synonyms or "FALSE" if they are homonyms."""

        user_prompt = f"""Are these two terms synonyms (mean the same thing)?

Term 1: "{input_node}"
Term 2: "{extracted_skill}"

Embedding similarity score: {similarity_score:.3f}

Answer TRUE or FALSE:"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/yourusername/yourrepo",
                "X-Title": "Task Skill Matcher - LLM Gate"
            }

            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistent answers
                "max_tokens": 10
            }

            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            answer = result['choices'][0]['message']['content'].strip().upper()

            # Parse answer
            is_synonym = "TRUE" in answer

            # Cache result
            self.cache[cache_key] = is_synonym

            print(f"LLM Gate: '{input_node}' vs '{extracted_skill}' -> {'SYNONYM' if is_synonym else 'HOMONYM'} (sim={similarity_score:.3f})")

            return is_synonym

        except Exception as e:
            print(f"LLM Gate error: {e}. Defaulting to accepting match.")
            # On error, default to accepting the match
            return True

# Define a task suitable for Ebony Moore: "Deploy and Configure a Monitoring System for Linux Servers"
# This task involves setting up Nagios/Zabbix monitoring across multiple Linux servers

# Dependency graph structure:
# - Input nodes are skill strings (leaf nodes)
# - Internal nodes are operations: {"op": "min"|"max"|"avg", "inputs": [...]}
# - The graph evaluates to a compatibility score between 0 and 1

TASK_SKILL_GRAPH = {
    "task_name": "Deploy and Configure a Monitoring System for Linux Servers",

    # Input nodes - these are the skills that will get dot product values [0,1] from embeddings
    "input_nodes": [
        "Linux Administration",
        "System Administration",
        "Nagios",
        "Monitoring Tools",
        "RHEL",
        "VMware",
        "Shell Scripting",
        "Bash",
        "Troubleshooting",
        "User Account Management",
        "File System Management",
        "Networking",
        "Automation",
    ],

    # Dependency graph with operations
    # The root node represents the final compatibility score
    "graph": {
        # Core Linux skills (must have both)
        "core_linux": {
            "op": "min",  # AND gate
            "inputs": ["Linux Administration", "System Administration"]
        },

        # Monitoring capability (Nagios OR general monitoring tools)
        "monitoring_capability": {
            "op": "max",  # OR gate
            "inputs": ["Nagios", "Monitoring Tools"]
        },

        # Platform knowledge (RHEL AND VMware)
        "platform_knowledge": {
            "op": "min",  # AND gate
            "inputs": ["RHEL", "VMware"]
        },

        # Scripting ability (Shell OR Bash)
        "scripting_ability": {
            "op": "max",  # OR gate
            "inputs": ["Shell Scripting", "Bash"]
        },

        # Automation skills (scripting AND automation knowledge)
        "automation_skills": {
            "op": "min",  # AND gate
            "inputs": ["scripting_ability", "Automation"]
        },

        # System management (user management OR file system management)
        "system_management": {
            "op": "max",  # OR gate
            "inputs": ["User Account Management", "File System Management"]
        },

        # Technical foundation (troubleshooting and networking both important)
        "technical_foundation": {
            "op": "avg",
            "inputs": ["Troubleshooting", "Networking"]
        },

        # Core competency (must have core Linux AND monitoring)
        "core_competency": {
            "op": "min",  # AND gate
            "inputs": ["core_linux", "monitoring_capability"]
        },

        # Platform skills (platform knowledge and automation both important)
        "platform_skills": {
            "op": "avg",
            "inputs": ["platform_knowledge", "automation_skills"]
        },

        # Supporting skills (system management AND technical foundation)
        "supporting_skills": {
            "op": "min",
            "inputs": ["system_management", "technical_foundation"]
        },

        # Final compatibility (core competency AND platform skills AND supporting skills)
        "final_score": {
            "op": "avg",
            "inputs": ["core_competency", "platform_skills", "supporting_skills"]
        }
    },

    # The root node that gives the final compatibility score
    "root": "final_score"
}

def load_extracted_skills(file_path: str) -> List[str]:
    """Load skills from extracted_skills.json"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['skills']

def get_cache_key(texts: List[str], model_name: str) -> str:
    """Generate cache key from texts and model name"""
    # Sort texts to ensure consistent ordering
    sorted_texts = sorted(texts)
    content = f"{model_name}||{'||'.join(sorted_texts)}"
    return hashlib.sha256(content.encode()).hexdigest()


def load_embedding_cache(cache_key: str, cache_dir: str = '.embedding_cache') -> Optional[np.ndarray]:
    """Load embeddings from cache if available"""
    cache_path = Path(cache_dir) / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                print(f"Loaded embeddings from cache ({cache_path.name})")
                return cached_data['embeddings']
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
    return None


def save_embedding_cache(cache_key: str, embeddings: np.ndarray, cache_dir: str = '.embedding_cache') -> None:
    """Save embeddings to cache"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    try:
        with open(cache_path / f"{cache_key}.pkl", 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'shape': embeddings.shape
            }, f)
        print(f"Saved embeddings to cache ({cache_key[:16]}...)")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def compute_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2', use_cache: bool = True) -> np.ndarray:
    """
    Compute sentence embeddings using transformer model with caching support.
    Using all-MiniLM-L6-v2: Fast, efficient, and good for semantic similarity

    Args:
        texts: List of text strings to embed
        model_name: Name of sentence-transformers model
        use_cache: Whether to use disk cache for embeddings

    Returns:
        Normalized embedding vectors as numpy array
    """
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(texts, model_name)
        cached_embeddings = load_embedding_cache(cache_key)
        if cached_embeddings is not None:
            return cached_embeddings

    # Compute embeddings
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model: {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True, device=device)

    # Save to cache
    if use_cache:
        save_embedding_cache(cache_key, embeddings)

    return embeddings

def compute_input_node_scores(
    input_nodes: List[str],
    extracted_skills: List[str],
    llm_gate: Optional['LLM_Gate'] = None,
    llm_gate_threshold: float = 0.7,
    match_threshold: float = 0.0,
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute dot product scores for each input node against all extracted skills.
    Optionally uses LLM gate to verify high-similarity matches.
    Applies global match threshold to filter low-similarity matches.

    Args:
        input_nodes: Task requirement terms
        extracted_skills: Candidate's extracted skills
        llm_gate: Optional LLM gate for homonym filtering
        llm_gate_threshold: Similarity threshold for applying LLM gate
        match_threshold: Global minimum similarity threshold (scores below this are zeroed)
        use_cache: Whether to use embedding cache

    Returns: Dict mapping input_node -> array of scores [0,1] for each extracted skill
    """
    all_texts = input_nodes + extracted_skills

    # Compute embeddings
    embeddings = compute_embeddings(all_texts, use_cache=use_cache)

    # Split embeddings
    input_embeddings = embeddings[:len(input_nodes)]
    extracted_embeddings = embeddings[len(input_nodes):]

    # Compute similarity matrix (dot products since embeddings are normalized)
    # Result shape: (num_extracted_skills, num_input_nodes)
    similarity_matrix = np.dot(extracted_embeddings, input_embeddings.T)

    # Ensure values are in [0, 1] range (they should be since embeddings are normalized)
    # Cosine similarity is [-1, 1], so we'll map to [0, 1]
    similarity_matrix = (similarity_matrix + 1) / 2

    # Apply global match threshold (zero out scores below threshold)
    if match_threshold > 0:
        print(f"\nApplying global match threshold: {match_threshold}")
        below_threshold = similarity_matrix < match_threshold
        num_filtered = np.sum(below_threshold)
        similarity_matrix[below_threshold] = 0.0
        print(f"Filtered {num_filtered} matches below threshold")

    # Apply LLM gate if enabled - process per input node in descending order
    if llm_gate and llm_gate.is_enabled():
        print(f"\nApplying LLM gate to high-similarity matches (threshold: {llm_gate_threshold})...")
        gate_checks = 0
        gate_rejections = 0

        for i, input_node in enumerate(input_nodes):
            # Get scores for this input node
            node_scores = similarity_matrix[:, i]

            # Find candidates above LLM gate threshold, sorted descending
            candidates = [(j, node_scores[j]) for j in range(len(extracted_skills))
                         if node_scores[j] >= llm_gate_threshold]
            candidates.sort(key=lambda x: x[1], reverse=True)

            if candidates:
                print(f"\n  Input node: '{input_node}' - checking {len(candidates)} high-similarity matches")

            # Process in descending order with early stopping
            for j, score in candidates:
                extracted_skill = extracted_skills[j]
                gate_checks += 1

                is_valid = llm_gate.verify_synonym(input_node, extracted_skill, score)

                if is_valid:
                    # Found a valid match - stop checking this input node
                    print(f"    [OK] Confirmed match '{extracted_skill}' - stopping checks for this node")
                    break
                else:
                    # Zero out homonym score completely
                    similarity_matrix[j, i] = 0.0
                    gate_rejections += 1

        print(f"\nLLM gate summary: checked {gate_checks} pairs, rejected {gate_rejections} as homonyms")

    # Create dict mapping each input node to its scores across all extracted skills
    input_scores = {}
    for i, input_node in enumerate(input_nodes):
        input_scores[input_node] = similarity_matrix[:, i]

    return input_scores


def evaluate_graph_node(
    node: Union[str, Dict],
    graph: Dict,
    input_scores: Dict[str, np.ndarray],
    memo: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Recursively evaluate a node in the dependency graph.
    Returns an array of scores [0,1] for each extracted skill.
    """
    # If it's a string, it's either an input node or a reference to another graph node
    if isinstance(node, str):
        if node in memo:
            return memo[node]
        if node in input_scores:
            # It's an input node - return its scores
            return input_scores[node]
        elif node in graph:
            # It's a reference to another graph node - evaluate it
            result = evaluate_graph_node(graph[node], graph, input_scores, memo)
            memo[node] = result
            return result
        else:
            raise ValueError(f"Unknown node: {node}")

    # It's a dict with an operation
    op = node["op"]
    inputs = node["inputs"]

    # Recursively evaluate all inputs
    evaluated_inputs = [
        evaluate_graph_node(inp, graph, input_scores, memo)
        for inp in inputs
    ]

    # Apply the operation
    if op == "min":
        result = np.minimum.reduce(evaluated_inputs)
    elif op == "max":
        result = np.maximum.reduce(evaluated_inputs)
    elif op == "avg":
        result = np.mean(evaluated_inputs, axis=0)
    else:
        raise ValueError(f"Unknown operation: {op}")

    return result


def evaluate_task_compatibility(
    task_graph: Dict,
    extracted_skills: List[str],
    ontology_manager: Optional[OntologyManager] = None,
    ontology_config: Optional[Dict[str, Any]] = None,
    llm_gate: Optional[LLM_Gate] = None,
    llm_gate_threshold: float = 0.7,
    match_threshold: float = 0.0,
    use_cache: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluate the task dependency graph for all extracted skills.
    Returns: (final_scores, debug_info)
    """
    input_nodes = task_graph["input_nodes"]
    graph = task_graph["graph"]
    root = task_graph["root"]

    # Compute scores for all input nodes
    if ontology_manager and ontology_config:
        mode = ontology_config['mode']

        if mode == 'expand':
            # Expand input nodes with ontology-related concepts
            print("Expanding task nodes with ontology relationships...")
            expanded = expand_task_nodes_with_ontology(
                input_nodes,
                ontology_manager,
                ontology_config
            )

            # Compute embeddings for all concepts
            all_texts = input_nodes + extracted_skills
            embeddings = compute_embeddings(all_texts, use_cache=use_cache)
            input_embeddings = embeddings[:len(input_nodes)]
            extracted_embeddings = embeddings[len(input_nodes):]

            # Compute base similarity
            similarity_matrix = np.dot(extracted_embeddings, input_embeddings.T)
            similarity_matrix = (similarity_matrix + 1) / 2

            # Apply expansion weights
            input_scores = {}
            for i, input_node in enumerate(input_nodes):
                base_scores = similarity_matrix[:, i].copy()

                # Get expanded concepts for this node
                concepts = expanded.get(input_node, [(input_node, 1.0)])

                if len(concepts) > 1:
                    # Compute scores for related concepts
                    concept_texts = [c[0] for c in concepts[1:]]  # Skip original
                    concept_weights = [c[1] for c in concepts[1:]]

                    if concept_texts:
                        concept_embeddings = compute_embeddings(concept_texts, use_cache=use_cache)
                        concept_sims = np.dot(extracted_embeddings, concept_embeddings.T)
                        concept_sims = (concept_sims + 1) / 2

                        # Weighted average with base scores
                        for j in range(len(extracted_skills)):
                            weighted_concept_scores = concept_sims[j] * concept_weights
                            max_concept_score = np.max(weighted_concept_scores)
                            # Boost base score if concepts match better
                            base_scores[j] = max(base_scores[j], max_concept_score)

                input_scores[input_node] = base_scores

        elif mode == 'boost':
            # Compute base embeddings
            all_texts = input_nodes + extracted_skills
            embeddings = compute_embeddings(all_texts, use_cache=use_cache)
            input_embeddings = embeddings[:len(input_nodes)]
            extracted_embeddings = embeddings[len(input_nodes):]

            # Boost scores using ontology relationships
            print("Boosting scores with ontology relationships...")
            input_scores = compute_ontology_boosted_scores(
                input_nodes,
                extracted_skills,
                extracted_embeddings,
                input_embeddings,
                ontology_manager,
                ontology_config
            )

        else:  # hybrid mode
            # First expand, then boost
            print("Hybrid mode: Expanding and boosting with ontology...")
            expanded = expand_task_nodes_with_ontology(
                input_nodes,
                ontology_manager,
                ontology_config
            )

            all_texts = input_nodes + extracted_skills
            embeddings = compute_embeddings(all_texts, use_cache=use_cache)
            input_embeddings = embeddings[:len(input_nodes)]
            extracted_embeddings = embeddings[len(input_nodes):]

            # First apply boost
            input_scores = compute_ontology_boosted_scores(
                input_nodes,
                extracted_skills,
                extracted_embeddings,
                input_embeddings,
                ontology_manager,
                ontology_config
            )

            # Then apply expansion (not implemented in this hybrid mode for simplicity)
            # In practice, you'd combine both approaches here

    else:
        # Standard mode without ontologies
        input_scores = compute_input_node_scores(
            input_nodes,
            extracted_skills,
            llm_gate=llm_gate,
            llm_gate_threshold=llm_gate_threshold,
            match_threshold=match_threshold,
            use_cache=use_cache
        )

    # Evaluate the graph starting from the root
    memo = {}
    final_scores = evaluate_graph_node(root, graph, input_scores, memo)

    # Prepare debug info
    debug_info = {
        "input_scores": input_scores,
        "intermediate_nodes": memo
    }

    return final_scores, debug_info

def print_graph_structure(graph: Dict, indent: int = 0):
    """Pretty print the dependency graph structure"""
    for node_name, node_def in graph.items():
        if isinstance(node_def, dict):
            op = node_def["op"]
            op_name = {"min": "MIN (AND)", "max": "MAX (OR)", "avg": "AVG"}[op]
            print("  " * indent + f"{node_name}: {op_name}")
            for inp in node_def["inputs"]:
                if isinstance(inp, str):
                    print("  " * (indent + 1) + f"- {inp}")
                else:
                    print("  " * (indent + 1) + f"- (inline operation)")
                    print_graph_structure({f"inline_{op}": inp}, indent + 2)


def print_task_description():
    """Print the task description and skill graph"""
    print("=" * 80)
    print(f"TASK: {TASK_SKILL_GRAPH['task_name']}")
    print("=" * 80)
    print("\nTask Description:")
    print("Deploy Nagios monitoring system across multiple RHEL/Linux servers in a")
    print("VMware virtualized environment. Configure monitoring checks, set up alerts,")
    print("automate deployment with shell scripts, and manage user access.")
    print("\nThis task is ideal for Ebony Moore based on her resume showing:")
    print("  - 3+ years Linux Administration experience")
    print("  - Experience with Nagios Monitoring tool")
    print("  - VMware experience")
    print("  - RHEL and Linux server management")
    print("  - User account management")
    print("  - Troubleshooting and automation experience")
    print("\n" + "=" * 80)
    print("INPUT NODES (Skills with dot product similarity [0,1])")
    print("=" * 80)
    for i, skill in enumerate(TASK_SKILL_GRAPH["input_nodes"], 1):
        print(f"  {i:2}. {skill}")
    print("\n" + "=" * 80)
    print("DEPENDENCY GRAPH STRUCTURE")
    print("=" * 80)
    print_graph_structure(TASK_SKILL_GRAPH["graph"])
    print(f"\nRoot node: {TASK_SKILL_GRAPH['root']}")
    print()


def initialize_ontology_manager(args) -> OntologyManager:
    """Initialize and load ontology manager based on command-line arguments"""
    manager = OntologyManager()

    # Determine which ontologies to load
    use_go = args.use_go or args.use_all_ontologies
    use_chebi = args.use_chebi  # Never auto-enabled due to RAM usage
    use_esco = args.use_esco or args.use_all_ontologies
    use_mesh = args.use_mesh or args.use_all_ontologies
    use_cell = args.use_cell or args.use_all_ontologies

    loaded = []

    if use_go:
        print("Loading Gene Ontology (GO)...")
        manager.load_go(use_basic=True)  # Use go-basic for smaller memory footprint
        loaded.append("GO")

    if use_chebi:
        print("Loading ChEBI (WARNING: This uses 2-3GB RAM)...")
        manager.load_chebi()
        loaded.append("ChEBI")

    if use_esco:
        print("Loading ESCO (Skills/Competences)...")
        manager.load_esco()
        loaded.append("ESCO")

    if use_mesh:
        print("Loading MeSH (Medical Terminology)...")
        manager.load_mesh()
        loaded.append("MeSH")

    if use_cell:
        print("Loading Cell Ontology...")
        manager.load_cell_ontology()
        loaded.append("Cell Ontology")

    if not loaded:
        print("WARNING: No ontologies specified. Use --use-esco, --use-go, etc.")
        print("Ontology system will be disabled.")
        return None

    print(f"[OK] Loaded ontologies: {', '.join(loaded)}")
    return manager


def expand_task_nodes_with_ontology(
    input_nodes: List[str],
    ontology_manager: OntologyManager,
    config: Dict[str, Any]
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Expand each task input node with related ontology concepts.

    Returns: Dict mapping original_node -> [(related_concept, weight), ...]
    """
    expanded_nodes = {}

    for node in input_nodes:
        # Find best match in ontologies
        best_match = ontology_manager.get_best_match_across_ontologies(
            node,
            min_similarity=config['min_similarity']
        )

        if not best_match:
            # No ontology match, keep original node only
            expanded_nodes[node] = [(node, 1.0)]
            continue

        related_concepts = [(node, 1.0)]  # Original node with full weight

        # Get the ontology API for the best match
        ontology_name = None
        for name, loaded in ontology_manager._loaded.items():
            if loaded:
                api = getattr(ontology_manager, f"{name}_api", None)
                if api:
                    # Check if this API has the matched node
                    test_node = api.get_node(best_match.node.id)
                    if test_node:
                        ontology_name = name
                        break

        if ontology_name is None:
            expanded_nodes[node] = [(node, 1.0)]
            continue

        ontology_api = getattr(ontology_manager, f"{ontology_name}_api")

        # Get neighbors (parents, children, siblings)
        matched_node_id = best_match.node.id
        neighbors = ontology_api.get_neighbors(
            matched_node_id,
            direction='both'
        )

        # Process relationships
        for neighbor_node, relation_type in neighbors[:config['max_depth'] * 3]:
            weight = 0.0

            # Parent relationships (superset concepts)
            if relation_type in ['is_a', 'part_of', 'broader']:
                weight = config['relation_weight'] * 0.7  # Partial activation

            # Child relationships (subset concepts)
            elif relation_type in ['has_part', 'narrower']:
                weight = config['relation_weight'] * 0.9  # Strong activation

            # Sibling relationships
            elif relation_type in ['related_to']:
                weight = config['sibling_weight']

            if weight > 0:
                related_concepts.append((neighbor_node.name, weight))

        expanded_nodes[node] = related_concepts[:10]  # Limit to top 10 concepts

    return expanded_nodes


def compute_ontology_boosted_scores(
    input_nodes: List[str],
    extracted_skills: List[str],
    embeddings: np.ndarray,
    input_embeddings: np.ndarray,
    ontology_manager: OntologyManager,
    config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Boost similarity scores using ontology relationships.

    For each extracted skill, check if it's related to input nodes via ontology.
    If related, boost the similarity score based on relationship type and strength.
    """
    input_scores = {}
    similarity_matrix = np.dot(embeddings, input_embeddings.T)
    similarity_matrix = (similarity_matrix + 1) / 2  # Map to [0,1]

    for i, input_node in enumerate(input_nodes):
        base_scores = similarity_matrix[:, i].copy()

        # For each extracted skill, check ontology relationships
        for j, skill in enumerate(extracted_skills):
            # Find both in ontology
            node_match = ontology_manager.get_best_match_across_ontologies(
                input_node,
                min_similarity=config['min_similarity']
            )
            skill_match = ontology_manager.get_best_match_across_ontologies(
                skill,
                min_similarity=config['min_similarity']
            )

            if not node_match or not skill_match:
                continue

            # Find which ontology both are in
            ontology_name = None
            for name, loaded in ontology_manager._loaded.items():
                if loaded:
                    api = getattr(ontology_manager, f"{name}_api", None)
                    if api:
                        node_test = api.get_node(node_match.node.id)
                        skill_test = api.get_node(skill_match.node.id)
                        if node_test and skill_test:
                            ontology_name = name
                            break

            if ontology_name is None:
                continue

            ontology_api = getattr(ontology_manager, f"{ontology_name}_api")

            # Get context around node match
            node_context = ontology_api.get_node_context(
                node_match.node.id,
                depth=config['max_depth']
            )

            # Check if skill node is in the context
            skill_node_id = skill_match.node.id
            found_relation = None
            distance = None

            for level_key, level_neighbors in node_context.get('neighbors', {}).items():
                for neighbor_info in level_neighbors:
                    if neighbor_info['node'].id == skill_node_id:
                        found_relation = neighbor_info['relation']
                        distance = neighbor_info['distance']
                        break
                if found_relation:
                    break

            if found_relation and distance:
                # Apply boost based on relationship
                boost = 0.0
                if found_relation in ['is_a', 'part_of', 'broader']:
                    boost = config['relation_weight'] * (1.0 / distance)
                elif found_relation in ['has_part', 'narrower']:
                    boost = config['relation_weight'] * 0.9 * (1.0 / distance)
                elif found_relation in ['related_to']:
                    boost = config['sibling_weight'] * (1.0 / distance)

                # Apply boost: new_score = base + (1 - base) * boost
                base_scores[j] = min(1.0, base_scores[j] + (1 - base_scores[j]) * boost)

        input_scores[input_node] = base_scores

    return input_scores


def parse_arguments():
    """Parse command-line arguments for ontology configuration"""
    parser = argparse.ArgumentParser(
        description='Match candidate skills to task requirements with optional ontology-based fuzzy matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ontology Options:
  --use-ontologies     Enable ontology-based fuzzy matching
  --ontology-mode      How to use ontologies for matching:
                       'expand': Expand task nodes with related concepts (default)
                       'boost': Boost scores for ontologically-related skills
                       'hybrid': Combine both expansion and boosting

  Individual Ontology Toggles:
  --use-go             Use Gene Ontology (biological processes, molecular functions)
  --use-chebi          Use ChEBI (chemical entities) - WARNING: Uses 2-3GB RAM
  --use-esco           Use ESCO (skills, competences, occupations)
  --use-mesh           Use MeSH (medical terminology)
  --use-cell           Use Cell Ontology (cell types and lineages)

  Fuzzy Matching Parameters:
  --relation-weight    Weight for hierarchical relationships (0-1, default: 0.5)
                       - Superset concepts (e.g., "Physics" -> "Quantum Field Theory")
                       - Subset concepts (e.g., "Machine Learning" -> "AI")
  --sibling-weight     Weight for sibling concepts sharing parent (0-1, default: 0.3)
  --min-ontology-sim   Minimum similarity for ontology matches (0-1, default: 0.4)
  --ontology-depth     Max depth for relationship traversal (default: 2)

Examples:
  # Basic usage without ontologies
  python taskmatch.py

  # Use ESCO ontology with default settings
  python taskmatch.py --use-ontologies --use-esco

  # Use multiple ontologies with custom weights
  python taskmatch.py --use-ontologies --use-esco --use-go --relation-weight 0.7

  # Hybrid mode with aggressive fuzzy matching
  python taskmatch.py --use-ontologies --use-esco --ontology-mode hybrid --relation-weight 0.8 --sibling-weight 0.5

  # Use LLM gate to filter homonyms (requires OPENROUTER_API_KEY)
  python taskmatch.py --use-llm-gate --llm-gate-threshold 0.75

  # Combine ontology fuzzy matching with LLM gate
  python taskmatch.py --use-ontologies --use-esco --use-llm-gate
        """
    )

    # Input/output files
    parser.add_argument('--skills-file', type=str, default='extracted_skills.json',
                       help='Path to extracted skills JSON file (default: extracted_skills.json)')
    parser.add_argument('--output-file', type=str, default='task_skill_matches.json',
                       help='Path to output results JSON file (default: task_skill_matches.json)')

    # Ontology flags
    parser.add_argument('--use-ontologies', action='store_true',
                       help='Enable ontology-based fuzzy matching')
    parser.add_argument('--ontology-mode', type=str, default='expand',
                       choices=['expand', 'boost', 'hybrid'],
                       help='How to use ontologies for matching (default: expand)')

    # Individual ontology toggles
    parser.add_argument('--use-go', action='store_true',
                       help='Use Gene Ontology')
    parser.add_argument('--use-chebi', action='store_true',
                       help='Use ChEBI (WARNING: 2-3GB RAM)')
    parser.add_argument('--use-esco', action='store_true',
                       help='Use ESCO (skills/competences)')
    parser.add_argument('--use-mesh', action='store_true',
                       help='Use MeSH (medical terminology)')
    parser.add_argument('--use-cell', action='store_true',
                       help='Use Cell Ontology')
    parser.add_argument('--use-all-ontologies', action='store_true',
                       help='Use all available ontologies (except ChEBI unless --use-chebi specified)')

    # Fuzzy matching parameters
    parser.add_argument('--relation-weight', type=float, default=0.5,
                       help='Weight for hierarchical relationships (0-1, default: 0.5)')
    parser.add_argument('--sibling-weight', type=float, default=0.3,
                       help='Weight for sibling concepts (0-1, default: 0.3)')
    parser.add_argument('--min-ontology-sim', type=float, default=0.4,
                       help='Minimum similarity for ontology matches (0-1, default: 0.4)')
    parser.add_argument('--ontology-depth', type=int, default=2,
                       help='Max depth for relationship traversal (default: 2)')

    # LLM gate parameters
    parser.add_argument('--use-llm-gate', action='store_true',
                       help='Enable LLM-based verification to filter homonyms from synonyms')
    parser.add_argument('--llm-gate-threshold', type=float, default=0.7,
                       help='Similarity threshold above which to use LLM gate (0-1, default: 0.7)')
    parser.add_argument('--llm-gate-model', type=str, default='mistralai/mistral-7b-instruct:free',
                       help='OpenRouter model for LLM gate (default: mistralai/mistral-7b-instruct:free)')

    # Caching parameters
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable embedding cache (recompute all embeddings)')
    parser.add_argument('--cache-dir', type=str, default='.embedding_cache',
                       help='Directory for embedding cache (default: .embedding_cache)')

    # Global matching threshold
    parser.add_argument('--match-threshold', type=float, default=0.0,
                       help='Global minimum similarity threshold for all matches (0-1, default: 0.0)')

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Print task information
    print_task_description()

    # Load extracted skills
    print("=" * 80)
    print(f"Loading extracted skills from {args.skills_file}...")
    extracted_skills = load_extracted_skills(args.skills_file)
    print(f"Loaded {len(extracted_skills)} skills")

    # Initialize ontology manager if requested
    ontology_manager = None
    if args.use_ontologies:
        print("\n" + "=" * 80)
        print("INITIALIZING ONTOLOGY SYSTEM")
        print("=" * 80)
        ontology_manager = initialize_ontology_manager(args)
        print(f"Ontology mode: {args.ontology_mode}")
        print(f"Relation weight: {args.relation_weight}")
        print(f"Sibling weight: {args.sibling_weight}")
        print(f"Min similarity: {args.min_ontology_sim}")
        print(f"Max depth: {args.ontology_depth}")

    # Initialize LLM gate if requested
    llm_gate = None
    if args.use_llm_gate:
        print("\n" + "=" * 80)
        print("INITIALIZING LLM GATE")
        print("=" * 80)
        llm_gate = LLM_Gate(model=args.llm_gate_model)
        if llm_gate.is_enabled():
            print(f"LLM gate enabled with model: {args.llm_gate_model}")
            print(f"Threshold: {args.llm_gate_threshold}")
            print("Will verify high-similarity matches to filter homonyms")
        else:
            print("LLM gate disabled (no API key found)")
            llm_gate = None

    # Evaluate task compatibility
    print("\n" + "=" * 80)
    print("Computing embeddings and evaluating dependency graph...")
    if not args.no_cache:
        print(f"Using embedding cache: {args.cache_dir}/")
    else:
        print("Embedding cache disabled")

    final_scores, debug_info = evaluate_task_compatibility(
        TASK_SKILL_GRAPH,
        extracted_skills,
        ontology_manager=ontology_manager,
        ontology_config={
            'mode': args.ontology_mode,
            'relation_weight': args.relation_weight,
            'sibling_weight': args.sibling_weight,
            'min_similarity': args.min_ontology_sim,
            'max_depth': args.ontology_depth
        } if ontology_manager else None,
        llm_gate=llm_gate,
        llm_gate_threshold=args.llm_gate_threshold,
        match_threshold=args.match_threshold,
        use_cache=not args.no_cache
    )

    # Create results with skill names and scores
    results = list(zip(extracted_skills, final_scores))
    results.sort(key=lambda x: x[1], reverse=True)

    # Display top results
    print("\n" + "=" * 80)
    print("TOP 30 SKILLS BY COMPATIBILITY SCORE")
    print("=" * 80)
    print(f"{'Rank':<6} {'Skill':<50} {'Score':<10}")
    print("-" * 80)

    for rank, (skill, score) in enumerate(results[:30], 1):
        print(f"{rank:<6} {skill:<50} {score:.4f}")

    # Show detailed breakdown for top skill
    print("\n" + "=" * 80)
    print(f"DETAILED BREAKDOWN FOR TOP SKILL: {results[0][0]}")
    print("=" * 80)
    top_skill_idx = extracted_skills.index(results[0][0])

    print("\nInput Node Scores:")
    for input_node in TASK_SKILL_GRAPH["input_nodes"]:
        score = debug_info["input_scores"][input_node][top_skill_idx]
        bar = "#" * int(score * 40)
        print(f"  {input_node:.<40} {score:.4f} {bar}")

    print("\nIntermediate Node Scores:")
    for node_name, node_scores in debug_info["intermediate_nodes"].items():
        if node_name != TASK_SKILL_GRAPH["root"]:
            score = node_scores[top_skill_idx]
            bar = "#" * int(score * 40)
            print(f"  {node_name:.<40} {score:.4f} {bar}")

    # Save detailed results
    output_data = {
        "task": TASK_SKILL_GRAPH,
        "results": [
            {
                "rank": rank,
                "skill": skill,
                "compatibility_score": float(score)
            }
            for rank, (skill, score) in enumerate(results, 1)
        ]
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Full results saved to {args.output_file}")
    print("=" * 80)

    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total extracted skills analyzed: {len(extracted_skills)}")
    print(f"  Input nodes: {len(TASK_SKILL_GRAPH['input_nodes'])}")
    print(f"  Graph nodes: {len(TASK_SKILL_GRAPH['graph'])}")
    print(f"  Skills with score > 0.5: {sum(1 for _, score in results if score > 0.5)}")
    print(f"  Skills with score > 0.3: {sum(1 for _, score in results if score > 0.3)}")
    print(f"  Skills with score > 0.1: {sum(1 for _, score in results if score > 0.1)}")

    # Calculate final task match score
    # Take the maximum score across all extracted skills as the candidate's match
    final_match_score = max(final_scores)
    best_matching_skill = results[0][0]

    print("\n" + "=" * 80)
    print("FINAL TASK MATCH FOR CANDIDATE (Ebony Moore)")
    print("=" * 80)
    print(f"\nTask: {TASK_SKILL_GRAPH['task_name']}")
    print(f"\nCompatibility Score: {final_match_score:.4f} / 1.0000")

    # Visual representation
    bar_length = int(final_match_score * 50)
    bar = "#" * bar_length + "-" * (50 - bar_length)
    percentage = final_match_score * 100
    print(f"[{bar}] {percentage:.1f}%")

    print(f"\nBest Matching Skill: {best_matching_skill}")

    # Interpretation
    if final_match_score >= 0.7:
        interpretation = "EXCELLENT MATCH - Candidate has strong alignment with task requirements"
    elif final_match_score >= 0.5:
        interpretation = "GOOD MATCH - Candidate meets most task requirements"
    elif final_match_score >= 0.3:
        interpretation = "MODERATE MATCH - Candidate has relevant skills but may need training"
    elif final_match_score >= 0.15:
        interpretation = "PARTIAL MATCH - Candidate has some relevant skills"
    else:
        interpretation = "LOW MATCH - Significant skill gaps for this task"

    print(f"\nInterpretation: {interpretation}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
