"""
Cost tracking for OpenRouter API usage.
Tracks tokens and calculates costs based on model pricing.
"""

import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

@dataclass
class APICall:
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cost: float
    completion_cost: float
    total_cost: float
    operation: str  # e.g., "skill_extraction", "skill_comparison"

class CostTracker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls: List[APICall] = []
        self.model_pricing: Dict[str, Dict] = {}
        
    def fetch_model_pricing(self, model: str) -> Optional[Dict]:
        """Fetch pricing information for a specific model from OpenRouter."""
        if model in self.model_pricing:
            return self.model_pricing[model]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
            response.raise_for_status()
            
            models_data = response.json()
            
            # Find the specific model
            for model_info in models_data.get('data', []):
                if model_info.get('id') == model:
                    pricing = model_info.get('pricing', {})
                    self.model_pricing[model] = {
                        'prompt': float(pricing.get('prompt', 0)),
                        'completion': float(pricing.get('completion', 0)),
                        'name': model_info.get('name', model),
                    }
                    return self.model_pricing[model]
            
            # Default to zero if not found
            print(f"Warning: Pricing not found for model {model}, using $0")
            self.model_pricing[model] = {'prompt': 0.0, 'completion': 0.0, 'name': model}
            return self.model_pricing[model]
            
        except Exception as e:
            print(f"Error fetching model pricing: {e}")
            # Use defaults
            self.model_pricing[model] = {'prompt': 0.0, 'completion': 0.0, 'name': model}
            return self.model_pricing[model]
    
    def track_api_call(self, model: str, response: Dict, operation: str = "api_call") -> APICall:
        """Track an API call and calculate costs."""
        usage = response.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # Get pricing
        pricing = self.fetch_model_pricing(model)
        
        # Calculate costs (pricing is per million tokens)
        prompt_cost = (prompt_tokens / 1_000_000) * pricing['prompt']
        completion_cost = (completion_tokens / 1_000_000) * pricing['completion']
        total_cost = prompt_cost + completion_cost
        
        # Create call record
        call = APICall(
            timestamp=datetime.now().isoformat(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost,
            operation=operation
        )
        
        self.calls.append(call)
        return call
    
    def get_summary(self) -> Dict:
        """Get cost summary statistics."""
        if not self.calls:
            return {
                'total_calls': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'by_operation': {},
                'by_model': {}
            }
        
        total_tokens = sum(call.total_tokens for call in self.calls)
        total_cost = sum(call.total_cost for call in self.calls)
        
        # Group by operation
        by_operation = {}
        for call in self.calls:
            if call.operation not in by_operation:
                by_operation[call.operation] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost': 0.0
                }
            by_operation[call.operation]['calls'] += 1
            by_operation[call.operation]['tokens'] += call.total_tokens
            by_operation[call.operation]['cost'] += call.total_cost
        
        # Group by model
        by_model = {}
        for call in self.calls:
            if call.model not in by_model:
                by_model[call.model] = {
                    'calls': 0,
                    'tokens': 0,
                    'cost': 0.0
                }
            by_model[call.model]['calls'] += 1
            by_model[call.model]['tokens'] += call.total_tokens
            by_model[call.model]['cost'] += call.total_cost
        
        return {
            'total_calls': len(self.calls),
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'by_operation': by_operation,
            'by_model': by_model
        }
    
    def print_summary(self):
        """Print a formatted cost summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("API COST SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal API Calls: {summary['total_calls']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost']:.6f}")
        
        if summary['by_model']:
            print("\n" + "-" * 80)
            print("Cost by Model:")
            print("-" * 80)
            for model, stats in summary['by_model'].items():
                pricing = self.model_pricing.get(model, {})
                model_name = pricing.get('name', model)
                print(f"\n{model_name}:")
                print(f"  Calls: {stats['calls']}")
                print(f"  Tokens: {stats['tokens']:,}")
                print(f"  Cost: ${stats['cost']:.6f}")
                if pricing:
                    print(f"  Pricing: ${pricing.get('prompt', 0):.2f}/M prompt, ${pricing.get('completion', 0):.2f}/M completion")
        
        if summary['by_operation']:
            print("\n" + "-" * 80)
            print("Cost by Operation:")
            print("-" * 80)
            for operation, stats in summary['by_operation'].items():
                print(f"\n{operation}:")
                print(f"  Calls: {stats['calls']}")
                print(f"  Tokens: {stats['tokens']:,}")
                print(f"  Cost: ${stats['cost']:.6f}")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_path: Path):
        """Save detailed cost report to JSON."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'model_pricing': self.model_pricing,
            'detailed_calls': [asdict(call) for call in self.calls]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed cost report saved to {output_path}")

