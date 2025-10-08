import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Union, Any

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
            "op": "min",
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

def compute_embeddings(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Compute sentence embeddings using transformer model
    Using all-MiniLM-L6-v2: Fast, efficient, and good for semantic similarity
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    return embeddings

def compute_input_node_scores(
    input_nodes: List[str],
    extracted_skills: List[str]
) -> Dict[str, np.ndarray]:
    """
    Compute dot product scores for each input node against all extracted skills.
    Returns: Dict mapping input_node -> array of scores [0,1] for each extracted skill
    """
    all_texts = input_nodes + extracted_skills

    # Compute embeddings
    embeddings = compute_embeddings(all_texts)

    # Split embeddings
    input_embeddings = embeddings[:len(input_nodes)]
    extracted_embeddings = embeddings[len(input_nodes):]

    # Compute similarity matrix (dot products since embeddings are normalized)
    # Result shape: (num_extracted_skills, num_input_nodes)
    similarity_matrix = np.dot(extracted_embeddings, input_embeddings.T)

    # Ensure values are in [0, 1] range (they should be since embeddings are normalized)
    # Cosine similarity is [-1, 1], so we'll map to [0, 1]
    similarity_matrix = (similarity_matrix + 1) / 2

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
    extracted_skills: List[str]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluate the task dependency graph for all extracted skills.
    Returns: (final_scores, debug_info)
    """
    input_nodes = task_graph["input_nodes"]
    graph = task_graph["graph"]
    root = task_graph["root"]

    # Compute scores for all input nodes
    input_scores = compute_input_node_scores(input_nodes, extracted_skills)

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


def main():
    # Print task information
    print_task_description()

    # Load extracted skills
    print("=" * 80)
    print("Loading extracted skills from extracted_skills.json...")
    extracted_skills = load_extracted_skills('extracted_skills.json')
    print(f"Loaded {len(extracted_skills)} skills")

    # Evaluate task compatibility
    print("\nComputing embeddings and evaluating dependency graph...")
    final_scores, debug_info = evaluate_task_compatibility(TASK_SKILL_GRAPH, extracted_skills)

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

    with open('task_skill_matches.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 80)
    print("Full results saved to task_skill_matches.json")
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
