import json
import requests
import os
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from utils import CostTracker

OUTPUT_DIR = Path("output")

@dataclass
class TrueSkillRating:
    mu: float = 25.0      # Mean skill level
    sigma: float = 25.0/3  # Standard deviation (uncertainty)

    @property
    def conservative_rating(self) -> float:
        """Conservative skill estimate (mu - 3*sigma)"""
        return self.mu - 3 * self.sigma

class SkillSorter:
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet", 
                 temperature: float = 0.5, cost_tracker: CostTracker = None):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.resume_content = ""
        self.ratings: Dict[str, TrueSkillRating] = {}
        self.beta = 25.0/6    # Skill difference factor
        self.tau = 25.0/300   # Additive dynamics factor
        self.cost_tracker = cost_tracker or CostTracker(api_key)

    def load_resume(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.resume_content = content
            return content

    def load_skills(self, file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('skills', [])

    def initialize_ratings(self, skills: List[str]) -> None:
        """Initialize all skills with default TrueSkill ratings"""
        for skill in skills:
            self.ratings[skill] = TrueSkillRating()

    def make_api_request(self, user_prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/yourrepo",
            "X-Title": "Skill Sorter"
        }

        data = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.resume_content},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def ask_comparison(self, skill_a: str, skill_b: str, question_type: str) -> bool:
        """Ask one of four question types for verification"""

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
            
            # Track API cost
            self.cost_tracker.track_api_call(self.model, response, operation="skill_comparison")
            
            content = response['choices'][0]['message']['content'].strip().lower()
            return content == 'true'
        except Exception as e:
            print(f"Error in comparison: {e}")
            return False

    def compare_skills_verified(self, skill_a: str, skill_b: str) -> Optional[bool]:
        """
        Compare two skills with quadruple verification.
        Returns True if A wins, False if B wins, None if verification fails.
        """
        print(f"Comparing: {skill_a} vs {skill_b}")

        # Ask all four ways
        a_over_b = self.ask_comparison(skill_a, skill_b, 'a_over_b')
        b_over_a = self.ask_comparison(skill_a, skill_b, 'b_over_a')
        not_a_over_b = self.ask_comparison(skill_a, skill_b, 'not_a_over_b')
        not_b_over_a = self.ask_comparison(skill_a, skill_b, 'not_b_over_a')

        print(f"  A>B: {a_over_b}, B>A: {b_over_a}, !(A>B): {not_a_over_b}, !(B>A): {not_b_over_a}")

        # Check consistency
        if a_over_b:
            # A > B case
            expected = (True, False, False, True)
            actual = (a_over_b, b_over_a, not_a_over_b, not_b_over_a)
            if expected == actual:
                print(f"  ✓ Verified: {skill_a} wins")
                return True
        else:
            # B > A case
            expected = (False, True, True, False)
            actual = (a_over_b, b_over_a, not_a_over_b, not_b_over_a)
            if expected == actual:
                print(f"  ✓ Verified: {skill_b} wins")
                return False

        print("  ✗ Verification failed - inconsistent answers")
        return None

    def gaussian_cdf(self, x: float) -> float:
        """Cumulative distribution function of standard normal distribution"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def gaussian_pdf(self, x: float) -> float:
        """Probability density function of standard normal distribution"""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def update_ratings(self, winner: str, loser: str) -> None:
        """Update TrueSkill ratings based on match outcome"""
        winner_rating = self.ratings[winner]
        loser_rating = self.ratings[loser]

        # Calculate match quality parameters
        c = math.sqrt(winner_rating.sigma**2 + loser_rating.sigma**2 + 2 * self.beta**2)

        # Performance difference
        delta_mu = winner_rating.mu - loser_rating.mu

        # TrueSkill update factors
        v = self.gaussian_pdf(delta_mu / c) / self.gaussian_cdf(delta_mu / c)
        w = v * (v + delta_mu / c)

        # Update winner
        winner_mu_delta = (winner_rating.sigma**2 / c) * v
        winner_sigma_multiplier = 1 - (winner_rating.sigma**2 / c**2) * w

        self.ratings[winner] = TrueSkillRating(
            mu=winner_rating.mu + winner_mu_delta,
            sigma=winner_rating.sigma * math.sqrt(max(winner_sigma_multiplier, 0.01))
        )

        # Update loser
        loser_mu_delta = -(loser_rating.sigma**2 / c) * v
        loser_sigma_multiplier = 1 - (loser_rating.sigma**2 / c**2) * w

        self.ratings[loser] = TrueSkillRating(
            mu=loser_rating.mu + loser_mu_delta,
            sigma=loser_rating.sigma * math.sqrt(max(loser_sigma_multiplier, 0.01))
        )

        print("  Updated ratings:")
        print(f"    {winner}: μ={self.ratings[winner].mu:.2f}, σ={self.ratings[winner].sigma:.2f}")
        print(f"    {loser}: μ={self.ratings[loser].mu:.2f}, σ={self.ratings[loser].sigma:.2f}")

    def run_trueskill_tournament(self, skills: List[str], num_rounds: int = 3) -> List[str]:
        """
        Run a stochastic TrueSkill tournament with multiple rounds.
        Each round performs random pairwise comparisons.
        """
        self.initialize_ratings(skills)

        print(f"Starting TrueSkill tournament with {len(skills)} skills for {num_rounds} rounds...")

        for round_num in range(num_rounds):
            print(f"\n=== Round {round_num + 1} ===")

            # Create random pairs for this round
            skills_copy = skills.copy()
            random.shuffle(skills_copy)

            pairs = []
            for i in range(0, len(skills_copy) - 1, 2):
                pairs.append((skills_copy[i], skills_copy[i + 1]))

            # If odd number of skills, add a random pairing with the last skill
            if len(skills_copy) % 2 == 1:
                last_skill = skills_copy[-1]
                random_opponent = random.choice(skills_copy[:-1])
                pairs.append((last_skill, random_opponent))

            print(f"Round {round_num + 1}: {len(pairs)} matches")

            # Process each match
            for match_num, (skill_a, skill_b) in enumerate(pairs, 1):
                print(f"\nMatch {match_num}/{len(pairs)}")

                result = self.compare_skills_verified(skill_a, skill_b)

                if result is True:
                    self.update_ratings(skill_a, skill_b)
                elif result is False:
                    self.update_ratings(skill_b, skill_a)
                else:
                    print("  Skipping match due to verification failure")

            # Show current standings
            print(f"\nStandings after round {round_num + 1}:")
            current_standings = self.get_current_standings()
            for i, (skill, rating) in enumerate(current_standings, 1):
                print(f"  {i}. {skill}: μ={rating.mu:.2f}, σ={rating.sigma:.2f}, conservative={rating.conservative_rating:.2f}")

        return [skill for skill, _ in self.get_current_standings()]

    def get_current_standings(self) -> List[Tuple[str, TrueSkillRating]]:
        """Get current standings sorted by conservative rating (mu - 3*sigma)"""
        return sorted(self.ratings.items(),
                     key=lambda x: x[1].conservative_rating,
                     reverse=True)

    def run(self, resume_path: str, skills: List[str] = None, skills_file: str = None, 
            num_rounds: int = None, output_dir: Path = OUTPUT_DIR) -> tuple:
        """Main execution method for skill sorting."""
        print("Starting TrueSkill skill sorting...")
        print(f"Using model: {self.model} (temperature: {self.temperature})")
        
        # Load resume
        self.load_resume(resume_path)
        
        # Load skills
        if skills is None:
            if skills_file is None:
                skills_file = str(output_dir / "extracted_skills.json")
            skills = self.load_skills(skills_file)
        
        print(f"Loaded {len(skills)} skills")
        
        # Determine rounds
        if num_rounds is None:
            num_rounds = max(2, len(skills) // 2)
        
        # Run tournament
        final_rankings = self.run_trueskill_tournament(skills, num_rounds)
        
        print("\n🏆 Final TrueSkill Rankings:")
        final_standings = self.get_current_standings()
        for i, (skill, rating) in enumerate(final_standings, 1):
            print(f"{i}. {skill}")
            print(f"   μ={rating.mu:.2f}, σ={rating.sigma:.2f}, conservative={rating.conservative_rating:.2f}")
        
        # Save results
        result = {
            "original_skills": skills,
            "final_rankings": final_rankings,
            "detailed_ratings": {
                skill: {
                    "mu": rating.mu,
                    "sigma": rating.sigma,
                    "conservative_rating": rating.conservative_rating
                }
                for skill, rating in self.ratings.items()
            },
            "algorithm": "TrueSkill",
            "rounds": num_rounds
        }
        
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "sorted_skills.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        return result, self.cost_tracker


def main():
    """Standalone entry point for sort.py"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        return

    cost_tracker = CostTracker(api_key)
    sorter = SkillSorter(api_key, cost_tracker=cost_tracker)
    
    _, cost_tracker = sorter.run("data/resume.txt")
    
    # Print and save cost report
    cost_tracker.print_summary()
    cost_tracker.save_report(OUTPUT_DIR / "cost_report_sort.json")

if __name__ == "__main__":
    main()