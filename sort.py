import json
import requests
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, deque

class SkillSorter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "google/gemini-2.5-flash"
        self.resume_content = ""

    def load_resume(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.resume_content = content
            return content

    def load_skills(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('skills', [])

    def make_api_request(self, user_prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/yourrepo",
            "X-Title": "Skill Sorter"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.resume_content},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def ask_comparison(self, skill_a: str, skill_b: str, question_type: str) -> bool:
        """
        Ask one of four question types:
        - 'a_over_b': "Is skill A more supported by the resume than skill B?"
        - 'b_over_a': "Is skill B more supported by the resume than skill A?"
        - 'not_a_over_b': "Is it false that skill A is more supported by the resume than skill B?"
        - 'not_b_over_a': "Is it false that skill B is more supported by the resume than skill A?"
        """

        if question_type == 'a_over_b':
            prompt = f'Is the skill "{skill_a}" more supported by evidence in the resume than the skill "{skill_b}"? Answer only "true" or "false".'
        elif question_type == 'b_over_a':
            prompt = f'Is the skill "{skill_b}" more supported by evidence in the resume than the skill "{skill_a}"? Answer only "true" or "false".'
        elif question_type == 'not_a_over_b':
            prompt = f'Is it false that the skill "{skill_a}" is more supported by evidence in the resume than the skill "{skill_b}"? Answer only "true" or "false".'
        elif question_type == 'not_b_over_a':
            prompt = f'Is it false that the skill "{skill_b}" is more supported by evidence in the resume than the skill "{skill_a}"? Answer only "true" or "false".'
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        try:
            response = self.make_api_request(prompt)
            content = response['choices'][0]['message']['content'].strip().lower()

            if content == 'true':
                return True
            elif content == 'false':
                return False
            else:
                print(f"Unexpected response: {content}")
                return False

        except Exception as e:
            print(f"Error in comparison: {e}")
            return False

    def compare_skills_verified(self, skill_a: str, skill_b: str) -> Optional[bool]:
        """
        Compare two skills with quadruple verification.
        Returns True if A > B, False if B > A, None if verification fails.
        """
        print(f"Comparing: {skill_a} vs {skill_b}")

        # Ask all four ways
        a_over_b = self.ask_comparison(skill_a, skill_b, 'a_over_b')
        b_over_a = self.ask_comparison(skill_a, skill_b, 'b_over_a')
        not_a_over_b = self.ask_comparison(skill_a, skill_b, 'not_a_over_b')
        not_b_over_a = self.ask_comparison(skill_a, skill_b, 'not_b_over_a')

        print(f"  A>B: {a_over_b}, B>A: {b_over_a}, !(A>B): {not_a_over_b}, !(B>A): {not_b_over_a}")

        # Check consistency
        # If A>B is true, then B>A should be false, !(A>B) should be false, !(B>A) should be true
        # If A>B is false, then B>A should be true, !(A>B) should be true, !(B>A) should be false

        if a_over_b:
            # A > B case
            expected = (a_over_b == True, b_over_a == False, not_a_over_b == False, not_b_over_a == True)
            actual = (a_over_b, b_over_a, not_a_over_b, not_b_over_a)
            if expected == actual:
                print(f"   Verified: {skill_a} > {skill_b}")
                return True
        else:
            # B > A case (since A > B is false)
            expected = (a_over_b == False, b_over_a == True, not_a_over_b == True, not_b_over_a == False)
            actual = (a_over_b, b_over_a, not_a_over_b, not_b_over_a)
            if expected == actual:
                print(f"   Verified: {skill_b} > {skill_a}")
                return False

        print(f"   Verification failed - inconsistent answers")
        return None

    def topological_sort(self, skills: List[str]) -> List[str]:
        """
        Perform topological sort using verified comparisons.
        """
        print(f"Starting topological sort of {len(skills)} skills...")

        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize all skills
        for skill in skills:
            in_degree[skill] = 0

        # Compare all pairs
        total_comparisons = len(skills) * (len(skills) - 1) // 2
        comparison_count = 0

        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                comparison_count += 1
                print(f"Comparison {comparison_count}/{total_comparisons}")

                skill_a = skills[i]
                skill_b = skills[j]

                result = self.compare_skills_verified(skill_a, skill_b)

                if result is True:  # A > B
                    graph[skill_a].append(skill_b)
                    in_degree[skill_b] += 1
                elif result is False:  # B > A
                    graph[skill_b].append(skill_a)
                    in_degree[skill_a] += 1
                # If result is None, skip this edge (verification failed)

        # Kahn's algorithm for topological sort
        queue = deque([skill for skill in skills if in_degree[skill] == 0])
        sorted_skills = []

        while queue:
            current = queue.popleft()
            sorted_skills.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if we have a valid topological sort
        if len(sorted_skills) != len(skills):
            print("Warning: Cycle detected or verification failures. Some skills may be missing from sorted result.")
            # Add remaining skills to the end
            for skill in skills:
                if skill not in sorted_skills:
                    sorted_skills.append(skill)

        return sorted_skills

def main():
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    sorter = SkillSorter(api_key)

    # Load resume and skills
    print("Loading resume and skills...")
    sorter.load_resume("resume.txt")
    skills = sorter.load_skills("extracted_skills.json")

    print(f"Loaded {len(skills)} skills: {skills}")

    # Perform topological sort
    sorted_skills = sorter.topological_sort(skills)

    print(f"\nFinal sorted skills (most supported to least supported):")
    for i, skill in enumerate(sorted_skills, 1):
        print(f"{i}. {skill}")

    # Save results
    result = {
        "original_skills": skills,
        "sorted_skills": sorted_skills,
        "sort_order": "most_supported_to_least_supported"
    }

    with open("sorted_skills.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nResults saved to sorted_skills.json")

if __name__ == "__main__":
    main()