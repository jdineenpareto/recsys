import json
import random
import requests
import os
from typing import List, Dict, Any

class SkillExtractor:
    def __init__(self, api_key: str, use_verification_gate: bool = False):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "mistralai/mistral-7b-instruct:free"
        self.extracted_skills = []
        self.counter = 0
        self.zero_counter_streak = 0
        self.use_verification_gate = use_verification_gate
        self.resume_text = ""

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

    def verify_skill_directly_supported(self, skill: str) -> bool:
        """
        Verification gate: Check if skill is directly supported by resume evidence.
        Returns True if directly supported, False if only inferred.
        """
        if not self.use_verification_gate:
            return True

        verification_prompt = f"""Does the resume contain DIRECT evidence that the candidate has the skill "{skill}"?

Direct evidence means:
- Explicit mention of the skill/technology/tool by name
- Clear description of using or working with this specific skill
- Projects, tasks, or responsibilities that explicitly involve this skill

NOT direct evidence:
- Skills that could be inferred but aren't explicitly mentioned
- General statements that might imply the skill
- Related skills that aren't the same thing

Respond with ONLY "YES" if there is direct evidence, or "NO" if the skill is only inferred."""

        try:
            response = self.make_api_request(self.resume_text, verification_prompt)
            content = response['choices'][0]['message']['content'].strip().upper()

            is_supported = "YES" in content

            if not is_supported:
                print(f"  [REJECTED] Skill '{skill}' - only inferred, not directly supported")
            else:
                print(f"  [VERIFIED] Skill '{skill}' - directly supported by resume")

            return is_supported

        except Exception as e:
            print(f"  [ERROR] Verification failed for '{skill}': {e}. Defaulting to accept.")
            return True

    def extract_skills(self, resume_path: str) -> List[str]:
        self.resume_text = self.load_resume(resume_path)

        # Validate resume content
        if not self.resume_text or len(self.resume_text) < 50:
            print(f"[ERROR] Resume file '{resume_path}' is empty or too short (< 50 characters)")
            print(f"[ERROR] Resume length: {len(self.resume_text)} characters")
            return []

        print(f"[INFO] Using resume as system prompt: {len(self.resume_text)} chars")
        print(f"[INFO] First 200 chars of resume: {self.resume_text[:200]}")

        system_prompt = self.resume_text

        while True:
            if self.zero_counter_streak >= 10:
                print(f"[HALT] Counter stuck at zero for {self.zero_counter_streak} prompts. Halting script.")
                break

            current_list = self.mask_random_skills(self.extracted_skills, self.counter)
            current_list = self.randomize_list(current_list)

            user_prompt = f"Extend the following list with the most obvious and clear skill that is not already on this list {current_list}. Respond with JSON in the format: {{\"skill\": \"new_skill_name\"}} or {{\"skill\": \"NONE\"}} if no new skills can be identified."

            try:
                print(f"\n[DEBUG] Sending request to API...")
                print(f"[DEBUG] Current list size: {len(current_list)} skills")
                print(f"[DEBUG] Counter: {self.counter}, Zero streak: {self.zero_counter_streak}")

                response = self.make_api_request(system_prompt, user_prompt)
                content = response['choices'][0]['message']['content']

                print(f"[DEBUG] Raw API response: {content[:200]}...")

                # Clean up markdown code blocks if present
                if content.startswith('```json'):
                    content = content.replace('```json\n', '').replace('```', '').strip()
                elif content.startswith('```'):
                    content = content.replace('```\n', '').replace('```', '').strip()

                print(f"[DEBUG] Cleaned response: {content[:200]}...")

                result = json.loads(content)
                new_skill = result.get('skill', '').strip()

                print(f"[DEBUG] Parsed skill: '{new_skill}'")

                if new_skill.upper() == 'NONE' or not new_skill:
                    if self.counter > 0:
                        self.counter -= 1
                    print(f"[WARN] Model indicated no more skills can be extracted. Counter: {self.counter}")

                    if self.counter == 0:
                        self.zero_counter_streak += 1
                    else:
                        self.zero_counter_streak = 0
                    continue

                if new_skill in self.extracted_skills:
                    if self.counter > 0:
                        self.counter -= 1
                    print(f"[WARN] Model repeated skill: {new_skill}. Counter: {self.counter}")

                    if self.counter == 0:
                        self.zero_counter_streak += 1
                    else:
                        self.zero_counter_streak = 0
                    continue

                self.extracted_skills.append(new_skill)
                self.counter += 1
                self.zero_counter_streak = 0
                print(f"[SUCCESS] Added skill: {new_skill}")
                print(f"[INFO] Total skills extracted so far: {len(self.extracted_skills)}")
                print(f"[INFO] Counter: {self.counter}")

            except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
                if self.counter > 0:
                    self.counter -= 1
                print(f"[ERROR] Error processing response: {e}. Counter: {self.counter}")
                print(f"[ERROR] This might be a malformed API response or network issue")

                if self.counter == 0:
                    self.zero_counter_streak += 1
                else:
                    self.zero_counter_streak = 0
                continue

        # Post-processing: Apply verification gate if enabled
        if self.use_verification_gate:
            print(f"\n{'='*60}")
            print(f"Starting verification gate: checking {len(self.extracted_skills)} skills")
            print(f"{'='*60}\n")

            verified_skills = []
            rejected_count = 0

            for skill in self.extracted_skills:
                if self.verify_skill_directly_supported(skill):
                    verified_skills.append(skill)
                else:
                    rejected_count += 1

            print(f"\n{'='*60}")
            print(f"Verification complete: {len(verified_skills)} verified, {rejected_count} rejected")
            print(f"{'='*60}\n")

            return verified_skills

        return self.extracted_skills

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract skills from resume using iterative LLM prompting")
    parser.add_argument('--verify-evidence', action='store_true',
                       help='Enable verification gate to check if skills are directly supported by resume (not just inferred)')
    parser.add_argument('--resume', type=str, default='resume.txt',
                       help='Path to resume file (default: resume.txt)')
    parser.add_argument('--output', type=str, default='extracted_skills.json',
                       help='Output JSON file (default: extracted_skills.json)')

    args = parser.parse_args()

    # Check if resume file exists
    if not os.path.exists(args.resume):
        print(f"ERROR: Resume file '{args.resume}' not found")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please ensure the resume file exists at the specified path")
        return

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    extractor = SkillExtractor(api_key, use_verification_gate=args.verify_evidence)

    # Load and validate resume
    with open(args.resume, 'r', encoding='utf-8') as f:
        resume_content = f.read()
    print(f"Loaded resume from '{args.resume}': {len(resume_content)} characters")

    if args.verify_evidence:
        print("Verification gate ENABLED - will check if skills are directly supported by resume")
    else:
        print("Verification gate DISABLED - accepting all extracted skills")

    print("Starting skill extraction...")
    skills = extractor.extract_skills(args.resume)

    print(f"\nFinal extracted skills ({len(skills)}):")
    for i, skill in enumerate(skills, 1):
        print(f"{i}. {skill}")

    with open(args.output, "w") as f:
        json.dump({"skills": skills}, f, indent=2)

    print(f"\nSkills saved to {args.output}")

if __name__ == "__main__":
    main()