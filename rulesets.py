"""Advanced ruleset system."""
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

@dataclass
class TotalisticRule:
    birth: Set[int] = field(default_factory=lambda: {3})
    survive: Set[int] = field(default_factory=lambda: {2, 3})
    def to_string(self) -> str:
        b = ''.join(str(n) for n in sorted(self.birth))
        s = ''.join(str(n) for n in sorted(self.survive))
        return f"B{b}/S{s}"
    @classmethod
    def from_string(cls, rs):
        m = re.match(r'B(\d*)/S(\d*)', rs.upper())
        if not m: raise ValueError("Invalid rule")
        return cls({int(c) for c in m.group(1)}, {int(c) for c in m.group(2)})
    def get_lookups(self):
        b = np.zeros(9, dtype=np.bool_); s = np.zeros(9, dtype=np.bool_)
        for n in self.birth: b[n] = True
        for n in self.survive: s[n] = True
        return b, s

RULE_DESCRIPTIONS = {
    "B3/S23": "Conway's Life", "B36/S23": "HighLife", "B3678/S34678": "Day & Night",
    "B1357/S1357": "Replicator", "B2/S": "Seeds", "B368/S245": "Morley",
}

class RuleAnalyzer:
    def __init__(self, size=100): self.size = size
    def analyze(self, rule_str: str) -> dict:
        try:
            rule = TotalisticRule.from_string(rule_str)
        except: return {"error": "Invalid format"}
        b, s = rule.get_lookups()
        grid = (np.random.random((self.size, self.size)) > 0.7).astype(np.int32)
        pops = [np.sum(grid > 0)]
        for _ in range(100):
            padded = np.pad(grid, 1, mode='wrap')
            a = (padded > 0).astype(np.int32)
            n = a[:-2, :-2] + a[:-2, 1:-1] + a[:-2, 2:] + a[1:-1, :-2] + a[1:-1, 2:] + a[2:, :-2] + a[2:, 1:-1] + a[2:, 2:]
            is_al = grid > 0
            new_g = np.zeros_like(grid)
            new_g[~is_al & b[n]] = 1
            new_g[is_al & s[n]] = grid[is_al & s[n]]
            grid = new_g; pops.append(np.sum(grid > 0))
        
        diffs = np.diff(pops[1:])
        exp = (pops[-1] - pops[0]) / (pops[0] + 1)
        stability = 1.0 / (1.0 + np.std(diffs) / (np.mean(pops) + 1))
        chaos = min(1.0, np.std(diffs[-20:]) / (np.mean(pops[-20:]) + 1)) if len(pops)>20 else 0
        
        cat = "Expanding" if exp > 2 else "Chaotic" if chaos > 0.5 else "Stable" if stability > 0.8 else "Dynamic"
        return {"rule": rule_str, "category": cat, "expansion": float(exp), "stability": float(stability), "chaos": float(chaos)}