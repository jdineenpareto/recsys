import json
import random
import requests
import os
from pathlib import Path
from typing import List, Dict, Any
from utils import CostTracker

OUTPUT_DIR = Path("output")

class SkillExtractor:
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet", 
                 temperature: float = 0.0, cost_tracker: CostTracker = None):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.extracted_skills = []
        self.counter = 0
        self.zero_counter_streak = 0
        self.cost_tracker = cost_tracker or CostTracker(api_key)

    def load_resume(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def make_api_request(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/yourrepo",
            "X-Title": "Skill Extractor"
        }

        data = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def randomize_list(self, skills: List[str]) -> List[str]:
        randomized = skills.copy()
        random.shuffle(randomized)
        return randomized

    def mask_random_skills(self, skills: List[str], n: int) -> List[str]:
        if n <= 0 or len(skills) == 0:
            return skills

        mask_count = min(n, len(skills))
        remaining_skills = skills.copy()

        for _ in range(mask_count):
            if remaining_skills:
                masked_skill = random.choice(remaining_skills)
                remaining_skills.remove(masked_skill)
                print(f"Temporarily masking skill: {masked_skill}")

        return remaining_skills

    def extract_skills(self, resume_path: str) -> List[str]:
        system_prompt = self.load_resume(resume_path)
        
        print(f"Loaded resume ({len(system_prompt)} characters)")
        print(f"Using model: {self.model}")
        print("Starting with empty skills list\n")

        while True:
            if self.zero_counter_streak >= 10:
                print(f"Counter stuck at zero for {self.zero_counter_streak} prompts. Halting script.")
                break

            current_list = self.mask_random_skills(self.extracted_skills, self.counter)
            current_list = self.randomize_list(current_list)

            user_prompt = f"Extend the following list with the most obvious and clear skill that is not already on this list {current_list}. Respond with JSON in the format: {{\"skill\": \"new_skill_name\"}} or {{\"skill\": \"NONE\"}} if no new skills can be identified."

            try:
                response = self.make_api_request(system_prompt, user_prompt)
                
                # Track API cost
                self.cost_tracker.track_api_call(self.model, response, operation="skill_extraction")
                
                content = response['choices'][0]['message']['content']
                print(f"API Response: {content[:100]}...")  # Show first 100 chars for debugging

                # Clean up markdown code blocks if present
                if content.startswith('```json'):
                    content = content.replace('```json\n', '').replace('```', '').strip()
                elif content.startswith('```'):
                    content = content.replace('```\n', '').replace('```', '').strip()

                result = json.loads(content)
                new_skill = result.get('skill', '').strip()

                if new_skill.upper() == 'NONE' or not new_skill:
                    if self.counter > 0:
                        self.counter -= 1
                    print(f"Model indicated no more skills can be extracted. Counter: {self.counter}")

                    if self.counter == 0:
                        self.zero_counter_streak += 1
                    else:
                        self.zero_counter_streak = 0
                    continue

                if new_skill in self.extracted_skills:
                    if self.counter > 0:
                        self.counter -= 1
                    print(f"Model repeated skill: {new_skill}. Counter: {self.counter}")

                    if self.counter == 0:
                        self.zero_counter_streak += 1
                    else:
                        self.zero_counter_streak = 0
                    continue

                self.extracted_skills.append(new_skill)
                self.counter += 1
                self.zero_counter_streak = 0
                print(f"Added skill: {new_skill}")
                print(f"Current skills list (randomized for next prompt): {self.randomize_list(self.extracted_skills)}")
                print(f"Counter: {self.counter}")

            except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
                if self.counter > 0:
                    self.counter -= 1
                print(f"Error processing response: {e}. Counter: {self.counter}")

                if self.counter == 0:
                    self.zero_counter_streak += 1
                else:
                    self.zero_counter_streak = 0
                continue

        return self.extracted_skills

    def run(self, resume_path: str, output_dir: Path = OUTPUT_DIR) -> tuple:
        """Main execution method for skill extraction."""
        print("Starting skill extraction...")
        print(f"Using model: {self.model} (temperature: {self.temperature})")
        
        skills = self.extract_skills(resume_path)

        print(f"\nFinal extracted skills ({len(skills)}):")
        for i, skill in enumerate(skills, 1):
            print(f"{i}. {skill}")

        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "extracted_skills.json"
        with open(output_file, "w") as f:
            json.dump({"skills": skills}, f, indent=2)

        print(f"\nSkills saved to {output_file}")
        
        return skills, self.cost_tracker


def main():
    """Standalone entry point for extract.py"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    cost_tracker = CostTracker(api_key)
    extractor = SkillExtractor(api_key, cost_tracker=cost_tracker)
    
    skills, cost_tracker = extractor.run("data/resume.txt")
    
    # Print and save cost report
    cost_tracker.print_summary()
    cost_tracker.save_report(OUTPUT_DIR / "cost_report_extract.json")

if __name__ == "__main__":
    main()