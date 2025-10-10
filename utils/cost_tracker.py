"""Simple cost tracking for OpenRouter API usage."""

import requests
from typing import Dict

class CostTracker:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
        self._pricing_cache = {}
    
    def _get_pricing(self, model: str) -> Dict:
        """Fetch and cache model pricing."""
        if model in self._pricing_cache:
            return self._pricing_cache[model]
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
            response.raise_for_status()
            
            for m in response.json().get('data', []):
                if m.get('id') == model:
                    pricing = m.get('pricing', {})
                    self._pricing_cache[model] = {
                        'prompt': float(pricing.get('prompt', 0)),
                        'completion': float(pricing.get('completion', 0))
                    }
                    return self._pricing_cache[model]
        except:
            pass
        
        self._pricing_cache[model] = {'prompt': 0.0, 'completion': 0.0}
        return self._pricing_cache[model]
    
    def track_api_call(self, model: str, response: Dict, operation: str = "api_call"):
        """Track an API call and calculate costs."""
        usage = response.get('usage', {})
        prompt_tok = usage.get('prompt_tokens', 0)
        completion_tok = usage.get('completion_tokens', 0)
        
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.calls += 1
        
        # Calculate cost (pricing is per million tokens)
        pricing = self._get_pricing(model)
        self.total_cost += (prompt_tok / 1_000_000) * pricing['prompt']
        self.total_cost += (completion_tok / 1_000_000) * pricing['completion']
    
    def print_summary(self):
        """Print cost summary."""
        print(f"\nAPI Usage:")
        print(f"  Calls: {self.calls}")
        print(f"  Tokens In: {self.prompt_tokens:,}")
        print(f"  Tokens Out: {self.completion_tokens:,}")
        print(f"  Total Cost: ${self.total_cost:.6f}")
    
    def save_report(self, output_path):
        """No-op for compatibility."""
        pass

