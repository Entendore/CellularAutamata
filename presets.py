"""Preset patterns for cellular automata."""
from typing import Dict, List, Tuple

Pattern = List[Tuple[int, int]]

PRESETS: Dict[str, Pattern] = {
    "Clear": [],
    "Block": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "Beehive": [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)],
    "Blinker": [(0, 0), (0, 1), (0, 2)],
    "Toad": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
    "Beacon": [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2), (3, 3)],
    "Pulsar": [
        (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10),
        (2, 0), (2, 5), (2, 7), (2, 12), (3, 0), (3, 5), (3, 7), (3, 12),
        (4, 0), (4, 5), (4, 7), (4, 12), (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (5, 10),
        (7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10), (8, 0), (8, 5), (8, 7), (8, 12),
        (9, 0), (9, 5), (9, 7), (9, 12), (10, 0), (10, 5), (10, 7), (10, 12),
        (12, 2), (12, 3), (12, 4), (12, 8), (12, 9), (12, 10)
    ],
    "Glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "LWSS": [(0, 1), (0, 4), (1, 0), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3)],
    "R-pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "Diehard": [(0, 6), (1, 0), (1, 1), (2, 1), (2, 5), (2, 6), (2, 7)],
    "Acorn": [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)],
    "Gosper Glider Gun": [
        (0, 24), (1, 22), (1, 24), (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
        (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35), (4, 0), (4, 1), (4, 10),
        (4, 16), (4, 20), (4, 21), (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17),
        (5, 22), (5, 24), (6, 10), (6, 16), (6, 24), (7, 11), (7, 15), (8, 12), (8, 13)
    ],
}

PRESET_CATEGORIES: Dict[str, List[str]] = {
    "Clear": ["Clear"], "Still Lifes": ["Block", "Beehive"], "Oscillators": ["Blinker", "Toad", "Beacon", "Pulsar"],
    "Spaceships": ["Glider", "LWSS"], "Methuselahs": ["R-pentomino", "Diehard", "Acorn"], "Guns": ["Gosper Glider Gun"],
}

def rotate_pattern(pattern: Pattern, degrees: int) -> Pattern:
    if not pattern or degrees == 0: return pattern.copy()
    max_r = max(r for r, c in pattern); max_c = max(c for r, c in pattern)
    rotated = []
    for r, c in pattern:
        if degrees == 90: rotated.append((c, max_r - r))
        elif degrees == 180: rotated.append((max_r - r, max_c - c))
        elif degrees == 270: rotated.append((max_c - c, r))
    min_r = min(r for r, c in rotated); min_c = min(c for r, c in rotated)
    return [(r - min_r, c - min_c) for r, c in rotated]

def parse_rle(rle_text: str) -> Pattern:
    pattern, row, col, count = [], 0, 0, ''
    lines = [l for l in rle_text.split('\n') if not l.startswith('#')]
    rle_text = ''.join(lines)
    for i, char in enumerate(rle_text):
        if char in ('.', 'o', 'b', '$', '!'): rle_text = rle_text[i:]; break
    for char in rle_text:
        if char.isdigit(): count += char
        elif char in ('o', 'b', '.'):
            n = int(count) if count else 1; count = ''
            if char == 'o':
                for _ in range(n): pattern.append((row, col)); col += 1
            else: col += n
        elif char == '$': n = int(count) if count else 1; count = ''; row += n; col = 0
        elif char == '!': break
    return pattern