import json
import random
import requests
import os
from typing import List, Dict, Any
import asyncio
import aiohttp

class SkillExtractor:
    def __init__(self, api_key: str, use_verification_gate: bool = False, model: str = "google/gemini-2.5-flash-lite",
                 filters_config_path: str = "extract_filters.json"):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.extracted_skills = []
        self.counter = 0
        self.zero_counter_streak = 0
        self.use_verification_gate = use_verification_gate
        self.resume_text = ""
        self.filters_config = self._load_filters_config(filters_config_path)
        self.verification_cache = {}

    def _load_filters_config(self, config_path: str) -> Dict[str, Any]:
        """Load filters configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[WARN] Filters config '{config_path}' not found. Using default verification behavior.")
            return {"filters": [], "model": "google/gemini-2.5-flash-lite", "cache_results": False}
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse filters config: {e}. Using default verification behavior.")
            return {"filters": [], "model": "google/gemini-2.5-flash-lite", "cache_results": False}

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

    async def _run_filter_async(self, session: aiohttp.ClientSession, filter_config: dict, skill: str, model: str) -> tuple[bool, str]:
        """
        Run a single filter check asynchronously.
        Returns (is_pass, filter_name)
        """
        filter_name = filter_config.get("name", "unnamed_filter")

        # Build system prompt (replace resume_text placeholder)
        system_prompt = filter_config.get("system_prompt", "").replace("{resume_text}", self.resume_text)

        # Build user prompt from template
        user_prompt_template = filter_config.get("user_prompt_template", "")
        user_prompt = user_prompt_template.replace("{skill}", skill)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/yourusername/yourrepo",
                "X-Title": "Skill Extractor"
            }

            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": filter_config.get("temperature", 0.1),
                "max_tokens": filter_config.get("max_tokens", 100)
            }

            async with session.post(self.base_url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                result = await response.json()

            content = result['choices'][0]['message']['content'].strip().upper()
            pass_condition = filter_config.get("pass_condition", "YES").upper()

            is_pass = pass_condition in content
            return is_pass, filter_name

        except Exception as e:
            print(f"  [ERROR] Filter '{filter_name}' failed for '{skill}': {e}. Defaulting to accept.")
            return True, filter_name

    async def verify_skill_directly_supported_async(self, skill: str) -> bool:
        """
        Verification gate: Check if skill is directly supported by resume evidence.
        Returns True if directly supported, False if only inferred.
        Uses filters configuration from extract_filters.json with async concurrency.
        """
        if not self.use_verification_gate:
            return True

        # Check cache if enabled
        if self.filters_config.get("cache_results", False) and skill in self.verification_cache:
            return self.verification_cache[skill]

        # Get enabled filters
        enabled_filters = [f for f in self.filters_config.get("filters", []) if f.get("enabled", False)]

        if not enabled_filters:
            print(f"  [WARN] No enabled filters found. Accepting skill by default.")
            return True

        # Use model from config or fallback to instance model
        model = self.filters_config.get("model", self.model)

        # Run all filters concurrently
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._run_filter_async(session, filter_config, skill, model)
                for filter_config in enabled_filters
            ]
            results = await asyncio.gather(*tasks)

        # Check results
        for is_pass, filter_name in results:
            if not is_pass:
                print(f"  [REJECTED by {filter_name}] Skill '{skill}'")
                if self.filters_config.get("cache_results", False):
                    self.verification_cache[skill] = False
                return False
            else:
                print(f"  [PASSED {filter_name}] Skill '{skill}'")

        # All filters passed
        if self.filters_config.get("cache_results", False):
            self.verification_cache[skill] = True
        return True

    def verify_skill_directly_supported(self, skill: str) -> bool:
        """
        Synchronous wrapper for verify_skill_directly_supported_async.
        """
        return asyncio.run(self.verify_skill_directly_supported_async(skill))

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

                # Always add skill during extraction - filtering happens post-extraction only
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

        # Post-extraction filtering: Apply verification gate if enabled
        if self.use_verification_gate:
            print(f"\n{'='*60}")
            print(f"Starting post-extraction verification: checking {len(self.extracted_skills)} skills")
            print(f"{'='*60}\n")

            # Run verification in parallel for all skills
            async def verify_all_skills():
                tasks = [self.verify_skill_directly_supported_async(skill) for skill in self.extracted_skills]
                return await asyncio.gather(*tasks)

            verification_results = asyncio.run(verify_all_skills())

            verified_skills = []
            rejected_count = 0

            for skill, is_verified in zip(self.extracted_skills, verification_results):
                if is_verified:
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
    parser.add_argument('--model', type=str, default='google/gemini-2.5-flash-lite',
                       help='Model to use for extraction (default: google/gemini-2.5-flash-lite)')
    parser.add_argument('--filters-config', type=str, default='extract_filters.json',
                       help='Path to filters configuration JSON (default: extract_filters.json)')

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

    extractor = SkillExtractor(api_key, use_verification_gate=args.verify_evidence,
                              model=args.model, filters_config_path=args.filters_config)

    # Load and validate resume
    with open(args.resume, 'r', encoding='utf-8') as f:
        resume_content = f.read()
    print(f"Loaded resume from '{args.resume}': {len(resume_content)} characters")
    print(f"Using model: {args.model}")

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