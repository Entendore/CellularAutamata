# rulesets.py
"""Advanced ruleset system."""
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional

@dataclass
class TotalisticRule:
    birth: Set[int] = field(default_factory=lambda: {3})
    survive: Set[int] = field(default_factory=lambda: {2, 3})
    
    def to_string(self) -> str:
        b = ''.join(str(n) for n in sorted(self.birth))
        s = ''.join(str(n) for n in sorted(self.survive))
        return f"B{b}/S{s}"
    
    @classmethod
    def from_string(cls, rs: str) -> 'TotalisticRule':
        m = re.match(r'B(\d*)/S(\d*)', rs.upper())
        if not m:
            raise ValueError(f"Invalid rule format: {rs}. Expected B/S notation like B3/S23")
        birth = {int(c) for c in m.group(1)} if m.group(1) else set()
        survive = {int(c) for c in m.group(2)} if m.group(2) else set()
        return cls(birth, survive)
    
    def get_lookups(self) -> Tuple[np.ndarray, np.ndarray]:
        b = np.zeros(9, dtype=np.bool_)
        s = np.zeros(9, dtype=np.bool_)
        for n in self.birth:
            if 0 <= n <= 8:
                b[n] = True
        for n in self.survive:
            if 0 <= n <= 8:
                s[n] = True
        return b, s
    
    def get_description(self) -> Optional[str]:
        return RULE_DESCRIPTIONS.get(self.to_string())


RULE_DESCRIPTIONS = {
    "B3/S23": "Conway's Life - The classic rules",
    "B36/S23": "HighLife - Like Life but with 6-cell birth",
    "B3678/S34678": "Day & Night - Symmetric rule",
    "B1357/S1357": "Replicator - Creates perfect copies",
    "B2/S": "Seeds - Explosive growth pattern",
    "B368/S245": "Morley (Move) - Complex behavior",
    "B3/S012345678": "Life without Death - Cells never die",
    "B368/S245": "Morley - Complex dynamics",
    "B1/S12": "Gnarl - Chaotic growth",
    "B2/S345": "Maze - Generates maze-like structures",
    "B3/S1234": "Mazectric - Another maze generator",
}

class RuleAnalyzer:
    """Analyzes ruleset behavior by running simulation."""
    
    def __init__(self, size: int = 100):
        self.size = size
    
    def analyze(self, rule_str: str, generations: int = 200) -> dict:
        """Analyze a ruleset by running simulation and measuring properties."""
        try:
            rule = TotalisticRule.from_string(rule_str)
        except ValueError as e:
            return {"error": str(e)}
        
        b, s = rule.get_lookups()
        grid = (np.random.random((self.size, self.size)) > 0.7).astype(np.int32)
        initial_pop = int(np.sum(grid > 0))
        pops = [initial_pop]
        
        for _ in range(generations):
            padded = np.pad(grid, 1, mode='wrap')
            a = (padded > 0).astype(np.int32)
            n = (a[:-2, :-2] + a[:-2, 1:-1] + a[:-2, 2:] +
                 a[1:-1, :-2] + a[1:-1, 2:] +
                 a[2:, :-2] + a[2:, 1:-1] + a[2:, 2:])
            
            is_al = grid > 0
            new_g = np.zeros_like(grid)
            new_g[~is_al & b[n]] = 1
            new_g[is_al & s[n]] = grid[is_al & s[n]]
            grid = new_g
            pops.append(int(np.sum(grid > 0)))
            
            # Early termination if grid is empty
            if pops[-1] == 0:
                break
        
        final_pop = pops[-1]
        
        # Calculate metrics
        diffs = np.diff(pops[1:])
        expansion = (final_pop - initial_pop) / max(1, initial_pop)
        
        # Stability - how constant is the population in later generations
        if len(pops) > 20:
            late_pops = pops[-20:]
            stability = 1.0 / (1.0 + np.std(late_pops) / max(1, np.mean(late_pops)))
        else:
            stability = 0.5
        
        # Chaos - population variance
        if len(diffs) > 0:
            chaos = min(1.0, np.std(diffs[-min(50, len(diffs)):]) / max(1, np.mean(pops)))
        else:
            chaos = 0.0
        
        # Growth rate
        growth_rates = []
        for i in range(1, min(20, len(pops))):
            if pops[i-1] > 0:
                growth_rates.append((pops[i] - pops[i-1]) / pops[i-1])
        avg_growth = np.mean(growth_rates) if growth_rates else 0
        
        # Categorize
        if final_pop == 0:
            category = "Dying"
        elif expansion > 5:
            category = "Explosive"
        elif expansion > 2:
            category = "Expanding"
        elif chaos > 0.5:
            category = "Chaotic"
        elif stability > 0.9:
            category = "Stable/Oscillating"
        elif abs(expansion) < 0.1:
            category = "Dynamic"
        else:
            category = "Complex"
        
        desc = rule.get_description()
        
        return {
            "rule": rule_str,
            "description": desc,
            "category": category,
            "initial_pop": initial_pop,
            "final_pop": final_pop,
            "expansion": float(expansion),
            "stability": float(stability),
            "chaos": float(chaos),
            "avg_growth": float(avg_growth),
            "generations_ran": len(pops) - 1
        }


def get_suggested_rules() -> list:
    """Get a list of interesting rules to try."""
    return [
        ("B3/S23", "Conway's Life"),
        ("B36/S23", "HighLife"),
        ("B3678/S34678", "Day & Night"),
        ("B1357/S1357", "Replicator"),
        ("B2/S", "Seeds"),
        ("B368/S245", "Morley"),
        ("B1/S12", "Gnarl"),
        ("B2/S345", "Maze"),
        ("B3/S1234", "Mazectric"),
        ("B3/S012345678", "Life without Death"),
    ]