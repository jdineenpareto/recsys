import json
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Union, Any

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")


class TaskMatcher:
    def __init__(self, task_file: str, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.task_file = task_file
        self.embedding_model = embedding_model
        self.task_graph = self.load_task_definition(task_file)
    
    def load_task_definition(self, task_file: str) -> Dict:
        """Load task definition from a JSON file in the data directory."""
        task_path = DATA_DIR / task_file
        if not task_path.exists():
            raise FileNotFoundError(f"Task file not found: {task_path}")
        
        with open(task_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_extracted_skills(self, file_path: str) -> List[str]:
        """Load skills from extracted_skills.json"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['skills']
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings using transformer model
        Using all-MiniLM-L6-v2: Fast, efficient, and good for semantic similarity
        """
        print(f"Loading embedding model: {self.embedding_model}...")
        model = SentenceTransformer(self.embedding_model)

        print(f"Computing embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        return embeddings
    
    def compute_input_node_scores(
        self,
        input_nodes: List[str],
        extracted_skills: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Compute dot product scores for each input node against all extracted skills.
        Returns: Dict mapping input_node -> array of scores [0,1] for each extracted skill
        """
        all_texts = input_nodes + extracted_skills

        # Compute embeddings
        embeddings = self.compute_embeddings(all_texts)

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
        self,
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
                result = self.evaluate_graph_node(graph[node], graph, input_scores, memo)
                memo[node] = result
                return result
            else:
                raise ValueError(f"Unknown node: {node}")

        # It's a dict with an operation
        op = node["op"]
        inputs = node["inputs"]

        # Recursively evaluate all inputs
        evaluated_inputs = [
            self.evaluate_graph_node(inp, graph, input_scores, memo)
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
        self,
        extracted_skills: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Evaluate the task dependency graph for all extracted skills.
        Returns: (final_scores, debug_info)
        """
        input_nodes = self.task_graph["input_nodes"]
        graph = self.task_graph["graph"]
        root = self.task_graph["root"]

        # Compute scores for all input nodes
        input_scores = self.compute_input_node_scores(input_nodes, extracted_skills)

        # Evaluate the graph starting from the root
        memo = {}
        final_scores = self.evaluate_graph_node(root, graph, input_scores, memo)

        # Prepare debug info
        debug_info = {
            "input_scores": input_scores,
            "intermediate_nodes": memo
        }

        return final_scores, debug_info
    
    def print_task_description(self):
        """Print the task description and skill graph"""
        print("=" * 80)
        print(f"TASK: {self.task_graph['task_name']}")
        print("=" * 80)
        
        if 'task_description' in self.task_graph:
            print("\nTask Description:")
            print(self.task_graph['task_description'])
        
        print("\n" + "=" * 80)
        print("INPUT NODES (Skills with dot product similarity [0,1])")
        print("=" * 80)
        for i, skill in enumerate(self.task_graph["input_nodes"], 1):
            print(f"  {i:2}. {skill}")
        print("\n" + "=" * 80)
        print("DEPENDENCY GRAPH STRUCTURE")
        print("=" * 80)
        self.print_graph_structure(self.task_graph["graph"])
        print(f"\nRoot node: {self.task_graph['root']}")
        print()
    
    def print_graph_structure(self, graph: Dict, indent: int = 0):
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
                        print("  " * (indent + 1) + "- (inline operation)")
                        self.print_graph_structure({f"inline_{op}": inp}, indent + 2)
    
    def run(self, skills: List[str] = None, skills_file: str = None, 
            output_dir: Path = OUTPUT_DIR) -> Dict:
        """Main execution method for task matching."""
        # Print task information
        self.print_task_description()

        # Load extracted skills
        print("=" * 80)
        if skills is None:
            if skills_file is None:
                skills_file = str(output_dir / "extracted_skills.json")
            print(f"Loading extracted skills from {skills_file}...")
            extracted_skills = self.load_extracted_skills(skills_file)
        else:
            extracted_skills = skills
            print(f"Using {len(extracted_skills)} skills from previous step")
        
        print(f"Loaded {len(extracted_skills)} skills")

        # Evaluate task compatibility
        print("\nComputing embeddings and evaluating dependency graph...")
        final_scores, debug_info = self.evaluate_task_compatibility(extracted_skills)

        # Create results with skill names and scores
        results = list(zip(extracted_skills, final_scores))
        results.sort(key=lambda x: x[1], reverse=True)

        # Display top results
        print("\n" + "=" * 80)
        print("TOP 10 SKILLS BY COMPATIBILITY SCORE")
        print("=" * 80)
        print(f"{'Rank':<6} {'Skill':<50} {'Score':<10}")
        print("-" * 80)

        for rank, (skill, score) in enumerate(results[:10], 1):
            print(f"{rank:<6} {skill:<50} {score:.4f}")

        # Show detailed breakdown for top skill
        if results:
            print("\n" + "=" * 80)
            print(f"DETAILED BREAKDOWN FOR TOP SKILL: {results[0][0]}")
            print("=" * 80)
            top_skill_idx = extracted_skills.index(results[0][0])

            print("\nInput Node Scores:")
            for input_node in self.task_graph["input_nodes"]:
                score = debug_info["input_scores"][input_node][top_skill_idx]
                bar = "#" * int(score * 40)
                print(f"  {input_node:.<40} {score:.4f} {bar}")

            print("\nIntermediate Node Scores:")
            for node_name, node_scores in debug_info["intermediate_nodes"].items():
                if node_name != self.task_graph["root"]:
                    score = node_scores[top_skill_idx]
                    bar = "#" * int(score * 40)
                    print(f"  {node_name:.<40} {score:.4f} {bar}")

        # Save detailed results
        output_data = {
            "task": self.task_graph,
            "results": [
                {
                    "rank": rank,
                    "skill": skill,
                    "compatibility_score": float(score)
                }
                for rank, (skill, score) in enumerate(results, 1)
            ]
        }

        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "task_skill_matches.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print("\n" + "=" * 80)
        print(f"Full results saved to {output_file}")
        print("=" * 80)

        # Summary statistics
        print("\nSummary:")
        print(f"  Total extracted skills analyzed: {len(extracted_skills)}")
        print(f"  Input nodes: {len(self.task_graph['input_nodes'])}")
        print(f"  Graph nodes: {len(self.task_graph['graph'])}")
        print(f"  Skills with score > 0.5: {sum(1 for _, score in results if score > 0.5)}")
        print(f"  Skills with score > 0.3: {sum(1 for _, score in results if score > 0.3)}")
        print(f"  Skills with score > 0.1: {sum(1 for _, score in results if score > 0.1)}")

        # Calculate final task match score
        if results:
            final_match_score = max(final_scores)
            best_matching_skill = results[0][0]

            print("\n" + "=" * 80)
            print("FINAL TASK MATCH FOR CANDIDATE")
            print("=" * 80)
            print(f"\nTask: {self.task_graph['task_name']}")
            print(f"\nCompatibility Score: {final_match_score:.4f} / 1.0000")

            # Visual representation
            bar_length = int(final_match_score * 50)
            bar = "#" * bar_length + "-" * (50 - bar_length)
            percentage = final_match_score * 100
            print(f"[{bar}] {percentage:.1f}%")

            print(f"\nBest Matching Skill: {best_matching_skill}")

            # Interpretation
            if final_match_score >= 0.7:
                interpretation = "EXCELLENT MATCH - Strong alignment with task requirements"
            elif final_match_score >= 0.5:
                interpretation = "GOOD MATCH - Meets most task requirements"
            elif final_match_score >= 0.3:
                interpretation = "MODERATE MATCH - Relevant skills but may need training"
            elif final_match_score >= 0.15:
                interpretation = "PARTIAL MATCH - Some relevant skills"
            else:
                interpretation = "LOW MATCH - Significant skill gaps for this task"

            print(f"\nInterpretation: {interpretation}")
            print("\n" + "=" * 80 + "\n")
        
        return output_data


def list_tasks():
    """List all available task files."""
    print("\nAvailable tasks in data/ directory:")
    print("-" * 80)
    
    task_files = sorted(DATA_DIR.glob("task_*.json"))
    if not task_files:
        print("No task files found (should start with 'task_' and end with '.json')")
        return
    
    for task_file in task_files:
        # Try to load and show task name
        try:
            with open(task_file) as f:
                task_data = json.load(f)
                task_name = task_data.get('task_name', 'Unknown')
                print(f"  {task_file.name:<40} - {task_name}")
        except Exception as e:
            print(f"  {task_file.name:<40} - (Error loading: {e})")
    
    print()


def main():
    """Standalone entry point for taskmatch.py"""
    task_file = sys.argv[1] if len(sys.argv) > 1 else "task_linux_monitoring.json"
    print(f"Loading task definition from: {DATA_DIR / task_file}")
    
    try:
        matcher = TaskMatcher(task_file)
        matcher.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nAvailable task files in {DATA_DIR}:")
        for f in sorted(DATA_DIR.glob("task_*.json")):
            print(f"  - {f.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
