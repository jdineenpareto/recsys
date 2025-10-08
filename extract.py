import json
import random
import requests
import os
from typing import List, Dict, Any

class SkillExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "mistralai/mistral-7b-instruct:free"
        self.extracted_skills = []
        self.counter = 0
        self.zero_counter_streak = 0

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

        while True:
            if self.zero_counter_streak >= 10:
                print(f"Counter stuck at zero for {self.zero_counter_streak} prompts. Halting script.")
                break

            current_list = self.mask_random_skills(self.extracted_skills, self.counter)
            current_list = self.randomize_list(current_list)

            user_prompt = f"Extend the following list with the most obvious and clear skill that is not already on this list {current_list}. Respond with JSON in the format: {{\"skill\": \"new_skill_name\"}} or {{\"skill\": \"NONE\"}} if no new skills can be identified."

            try:
                response = self.make_api_request(system_prompt, user_prompt)
                content = response['choices'][0]['message']['content']

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

def main():
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    extractor = SkillExtractor(api_key)
    resume_path = "resume.txt"

    print("Starting skill extraction...")
    skills = extractor.extract_skills(resume_path)

    print(f"\nFinal extracted skills ({len(skills)}):")
    for i, skill in enumerate(skills, 1):
        print(f"{i}. {skill}")

    with open("extracted_skills.json", "w") as f:
        json.dump({"skills": skills}, f, indent=2)

    print("\nSkills saved to extracted_skills.json")

if __name__ == "__main__":
    main()