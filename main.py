#!/usr/bin/env python3
"""
Main entrypoint for the recsys pipeline.
Can run individual steps or the entire pipeline as a DAG.
"""

import argparse
import sys

# Import the main functions from each script
from extract import SkillExtractor
from taskmatch import TaskMatcher, DATA_DIR, OUTPUT_DIR, list_tasks
from sort import SkillSorter
from cost_tracker import CostTracker
import os

def run_extract(resume_path: str = "data/resume.txt", model: str = "anthropic/claude-3.5-sonnet", 
                cost_tracker: CostTracker = None) -> tuple:
    """Run skill extraction step."""
    print("\n" + "="*80)
    print("STEP 1: SKILL EXTRACTION")
    print("="*80 + "\n")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Use provided cost tracker or create new one
    if cost_tracker is None:
        cost_tracker = CostTracker(api_key)
    
    extractor = SkillExtractor(api_key, model=model, cost_tracker=cost_tracker)
    skills, cost_tracker = extractor.run(resume_path, OUTPUT_DIR)
    
    return skills, cost_tracker


def run_match(task_file: str = "task_linux_monitoring.json", skills: list = None, 
              embedding_model: str = 'all-MiniLM-L6-v2') -> dict:
    """Run task matching step."""
    print("\n" + "="*80)
    print("STEP 2: TASK MATCHING")
    print("="*80 + "\n")
    
    try:
        matcher = TaskMatcher(task_file, embedding_model=embedding_model)
        output_data = matcher.run(skills=skills, output_dir=OUTPUT_DIR)
        return output_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nAvailable task files in {DATA_DIR}:")
        for f in sorted(DATA_DIR.glob("task_*.json")):
            print(f"  - {f.name}")
        sys.exit(1)


def run_sort(resume_path: str = "data/resume.txt", skills: list = None, num_rounds: int = None, 
             model: str = "anthropic/claude-3.5-sonnet", cost_tracker: CostTracker = None) -> tuple:
    """Run skill sorting step using TrueSkill ranking."""
    print("\n" + "="*80)
    print("STEP 3: SKILL RANKING (TrueSkill)")
    print("="*80 + "\n")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    # Use provided cost tracker or create new one
    if cost_tracker is None:
        cost_tracker = CostTracker(api_key)
    
    sorter = SkillSorter(api_key, model=model, cost_tracker=cost_tracker)
    
    result, cost_tracker = sorter.run(resume_path, skills=skills, num_rounds=num_rounds, output_dir=OUTPUT_DIR)
    return result, cost_tracker


def run_pipeline(resume_path: str = "data/resume.txt", task_file: str = "task_linux_monitoring.json", 
                 include_sort: bool = False, sort_rounds: int = None,
                 extract_model: str = "anthropic/claude-3.5-sonnet",
                 sort_model: str = "anthropic/claude-3.5-sonnet",
                 embedding_model: str = 'all-MiniLM-L6-v2'):
    """Run the complete pipeline: extract -> match -> (optional) sort."""
    print("\n" + "="*80)
    print("RUNNING COMPLETE PIPELINE")
    print("="*80 + "\n")
    
    # Create a single cost tracker for the entire pipeline
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    cost_tracker = CostTracker(api_key)
    
    # Step 1: Extract skills
    skills, cost_tracker = run_extract(resume_path, extract_model, cost_tracker)
    
    # Step 2: Match to task
    run_match(task_file, skills, embedding_model)
    
    # Step 3 (optional): Sort skills by evidence strength
    if include_sort:
        _, cost_tracker = run_sort(resume_path, skills, sort_rounds, sort_model, cost_tracker)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {OUTPUT_DIR / 'extracted_skills.json'}")
    print(f"  - {OUTPUT_DIR / 'task_skill_matches.json'}")
    if include_sort:
        print(f"  - {OUTPUT_DIR / 'sorted_skills.json'}")
    
    # Print and save consolidated cost report
    cost_tracker.print_summary()
    cost_tracker.save_report(OUTPUT_DIR / "cost_report_pipeline.json")


# list_tasks is now imported from taskmatch


def main():
    parser = argparse.ArgumentParser(
        description="Resume Skills Extraction and Task Matching Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (extract + match)
  python main.py all
  python main.py all --with-sort                              # Include skill ranking
  python main.py all --resume data/resume.txt --task task_linux_monitoring.json
  
  # Run individual steps
  python main.py extract
  python main.py extract --resume data/resume.txt
  python main.py match
  python main.py match --task task_linux_monitoring.json
  python main.py sort
  python main.py sort --rounds 3
  
  # List available tasks
  python main.py list-tasks
        """
    )
    
    parser.add_argument(
        'command',
        choices=['all', 'extract', 'match', 'sort', 'list-tasks'],
        help='Command to run: all (full pipeline), extract, match, sort, or list-tasks'
    )
    
    parser.add_argument(
        '--resume',
        default='data/resume.txt',
        help='Path to resume file (default: data/resume.txt)'
    )
    
    parser.add_argument(
        '--task',
        default='task_linux_monitoring.json',
        help='Task definition file in data/ directory (default: task_linux_monitoring.json)'
    )
    
    parser.add_argument(
        '--with-sort',
        action='store_true',
        help='Include skill sorting step in pipeline (uses TrueSkill ranking)'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=None,
        help='Number of TrueSkill rounds (default: auto-calculated based on skill count)'
    )
    
    parser.add_argument(
        '--extract-model',
        default='anthropic/claude-3.5-sonnet',
        help='Model for skill extraction (default: anthropic/claude-3.5-sonnet)'
    )
    
    parser.add_argument(
        '--sort-model',
        default='anthropic/claude-3.5-sonnet',
        help='Model for skill sorting (default: anthropic/claude-3.5-sonnet)'
    )
    
    parser.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='Embedding model for task matching (default: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'all':
        run_pipeline(
            args.resume, args.task, args.with_sort, args.rounds,
            args.extract_model, args.sort_model, args.embedding_model
        )
    elif args.command == 'extract':
        skills, cost_tracker = run_extract(args.resume, args.extract_model)
        cost_tracker.print_summary()
        cost_tracker.save_report(OUTPUT_DIR / "cost_report_extract.json")
    elif args.command == 'match':
        run_match(args.task, embedding_model=args.embedding_model)
    elif args.command == 'sort':
        _, cost_tracker = run_sort(args.resume, num_rounds=args.rounds, model=args.sort_model)
        cost_tracker.print_summary()
        cost_tracker.save_report(OUTPUT_DIR / "cost_report_sort.json")
    elif args.command == 'list-tasks':
        list_tasks()


if __name__ == "__main__":
    main()
