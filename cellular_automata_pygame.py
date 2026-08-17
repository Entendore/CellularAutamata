#!/usr/bin/env python3
"""Cellular Automata Studio & Explorer v3.0 - Pygame Implementation (Part 1)

Core modules: Color/Palette system, Presets, Rulesets, Backends,
Visual Effects, Themes, UI Widgets, CAEngine, UndoStack.
"""

# ============================================================
# SECTION 1: Imports
# ============================================================
import sys
import os
import json
import random
import math
import time
import logging
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import numpy as np
import pygame
import pygame.gfxdraw

# ============================================================
# SECTION 2: Color, Palette, PaletteManager
# ============================================================

@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int = 255

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    def to_array(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b], dtype=np.uint8)

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    @classmethod
    def from_hex(cls, hex_str: str) -> 'Color':
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 3:
            r = int(hex_str[0] * 2, 16)
            g = int(hex_str[1] * 2, 16)
            b = int(hex_str[2] * 2, 16)
        else:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
        return cls(r, g, b)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float) -> 'Color':
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        return cls(int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    def lerp(self, other: 'Color', t: float) -> 'Color':
        t = max(0.0, min(1.0, t))
        return Color(
            int(self.r + (other.r - self.r) * t),
            int(self.g + (other.g - self.g) * t),
            int(self.b + (other.b - self.b) * t),
            int(self.a + (other.a - self.a) * t),
        )

    def brightness(self) -> float:
        return 0.299 * self.r + 0.587 * self.g + 0.114 * self.b


class PaletteType(Enum):
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"
    CYCLIC = "cyclic"


@dataclass
class Palette:
    name: str
    colors: List[Color]
    palette_type: PaletteType = PaletteType.CATEGORICAL
    background_color: Color = field(default_factory=lambda: Color(30, 30, 30))
    description: str = ""

    def to_lut(self, max_state: int, interpolate: bool = True) -> np.ndarray:
        if max_state <= 1:
            return np.zeros((1, 3), dtype=np.uint8)
        lut = np.zeros((max_state, 3), dtype=np.uint8)
        n = len(self.colors)
        if n == 0:
            return lut
        if not interpolate or max_state <= n:
            for i in range(min(max_state, n)):
                c = self.colors[i]
                lut[i] = [c.r, c.g, c.b]
            if max_state > n and n > 0:
                for i in range(n, max_state):
                    lut[i] = [self.colors[-1].r, self.colors[-1].g, self.colors[-1].b]
        else:
            for i in range(max_state):
                t = i / max(1, max_state - 1) * (n - 1)
                idx = int(t)
                frac = t - idx
                if idx >= n - 1:
                    c = self.colors[-1]
                else:
                    c = self.colors[idx].lerp(self.colors[idx + 1], frac)
                lut[i] = [c.r, c.g, c.b]
        return lut

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "colors": [c.to_hex() for c in self.colors],
            "palette_type": self.palette_type.value,
            "background_color": self.background_color.to_hex(),
            "description": self.description,
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Palette':
        colors = [Color.from_hex(h) for h in data.get("colors", [])]
        bg = Color.from_hex(data["background_color"]) if "background_color" in data else Color(30, 30, 30)
        pt = PaletteType(data.get("palette_type", "categorical"))
        return cls(
            name=data.get("name", "Custom"),
            colors=colors,
            palette_type=pt,
            background_color=bg,
            description=data.get("description", ""),
        )


# ---- Built-in Palettes ----

BUILTIN_PALETTES: Dict[str, Palette] = {
    "Standard": Palette(
        "Standard",
        [Color(30, 30, 30), Color(0, 212, 168)],
        PaletteType.CATEGORICAL,
        description="Classic dark background with teal alive cells",
    ),
    "Grayscale": Palette(
        "Grayscale",
        [Color(0, 0, 0), Color(255, 255, 255)],
        PaletteType.SEQUENTIAL,
        description="Simple black and white",
    ),
    "Plasma": Palette(
        "Plasma",
        [Color(13, 8, 135), Color(126, 3, 168), Color(204, 71, 120),
         Color(248, 149, 64), Color(240, 249, 33)],
        PaletteType.SEQUENTIAL,
        description="Plasma colormap from matplotlib",
    ),
    "Viridis": Palette(
        "Viridis",
        [Color(68, 1, 84), Color(59, 82, 139), Color(33, 145, 140),
         Color(94, 201, 98), Color(253, 231, 37)],
        PaletteType.SEQUENTIAL,
        description="Viridis colormap, perceptually uniform",
    ),
    "Inferno": Palette(
        "Inferno",
        [Color(0, 0, 4), Color(40, 11, 84), Color(101, 21, 110),
         Color(159, 42, 99), Color(212, 72, 66), Color(245, 125, 21),
         Color(250, 193, 39), Color(252, 255, 164)],
        PaletteType.SEQUENTIAL,
        description="Inferno colormap, perceptually uniform",
    ),
    "Neon": Palette(
        "Neon",
        [Color(10, 10, 20), Color(57, 255, 20), Color(255, 0, 255),
         Color(0, 255, 255), Color(255, 255, 0)],
        PaletteType.CYCLIC,
        description="Bright neon colors on dark background",
    ),
    "Pastel": Palette(
        "Pastel",
        [Color(240, 240, 240), Color(255, 179, 186), Color(255, 223, 186),
         Color(255, 255, 186), Color(186, 255, 201), Color(186, 225, 255)],
        PaletteType.CATEGORICAL,
        description="Soft pastel colors",
    ),
    "Earth Tones": Palette(
        "Earth Tones",
        [Color(30, 25, 20), Color(139, 90, 43), Color(160, 120, 60),
         Color(85, 107, 47), Color(107, 142, 35), Color(189, 183, 107)],
        PaletteType.CATEGORICAL,
        description="Natural earth tones palette",
    ),
    "Ocean": Palette(
        "Ocean",
        [Color(0, 7, 20), Color(0, 40, 80), Color(0, 80, 130),
         Color(0, 130, 180), Color(60, 180, 220), Color(140, 220, 240)],
        PaletteType.SEQUENTIAL,
        description="Deep ocean blue gradient",
    ),
    "Fire": Palette(
        "Fire",
        [Color(10, 2, 2), Color(80, 0, 0), Color(180, 30, 0),
         Color(220, 80, 0), Color(255, 160, 0), Color(255, 230, 80),
         Color(255, 255, 200)],
        PaletteType.SEQUENTIAL,
        description="Hot fire gradient from dark to bright",
    ),
    "Matrix": Palette(
        "Matrix",
        [Color(0, 0, 0), Color(0, 40, 0), Color(0, 100, 0),
         Color(0, 180, 0), Color(0, 255, 0)],
        PaletteType.SEQUENTIAL,
        description="Matrix digital rain green",
    ),
    "Cyberpunk": Palette(
        "Cyberpunk",
        [Color(10, 5, 20), Color(255, 0, 110), Color(0, 255, 200),
         Color(130, 0, 255), Color(255, 230, 0)],
        PaletteType.CATEGORICAL,
        description="Cyberpunk neon aesthetic",
    ),
    "Mono Blue": Palette(
        "Mono Blue",
        [Color(10, 10, 30), Color(20, 40, 100), Color(40, 80, 180),
         Color(80, 140, 230), Color(160, 200, 255)],
        PaletteType.SEQUENTIAL,
        description="Monochromatic blue gradient",
    ),
    "Terrain": Palette(
        "Terrain",
        [Color(0, 40, 100), Color(0, 100, 150), Color(50, 180, 100),
         Color(150, 200, 80), Color(200, 180, 100), Color(180, 140, 80),
         Color(220, 220, 220)],
        PaletteType.SEQUENTIAL,
        description="Terrain heightmap colors",
    ),
    "Rainbow": Palette(
        "Rainbow",
        [Color(255, 0, 0), Color(255, 127, 0), Color(255, 255, 0),
         Color(0, 255, 0), Color(0, 0, 255), Color(75, 0, 130),
         Color(148, 0, 211)],
        PaletteType.CYCLIC,
        description="Full rainbow spectrum",
    ),
    "Color Blind Safe": Palette(
        "Color Blind Safe",
        [Color(30, 30, 30), Color(0, 114, 178), Color(230, 159, 0),
         Color(0, 158, 115), Color(204, 121, 167), Color(86, 180, 233),
         Color(213, 94, 0), Color(240, 228, 66)],
        PaletteType.CATEGORICAL,
        description="Wong's color-blind safe palette",
    ),
    "Sepia": Palette(
        "Sepia",
        [Color(20, 15, 10), Color(60, 40, 20), Color(112, 66, 20),
         Color(150, 100, 50), Color(190, 150, 90), Color(220, 200, 160)],
        PaletteType.SEQUENTIAL,
        description="Vintage sepia tones",
    ),
    "Candy": Palette(
        "Candy",
        [Color(240, 230, 240), Color(255, 105, 180), Color(255, 182, 193),
         Color(255, 218, 185), Color(186, 85, 211), Color(100, 149, 237)],
        PaletteType.CATEGORICAL,
        description="Sweet candy colors",
    ),
    "Thermal": Palette(
        "Thermal",
        [Color(0, 0, 20), Color(20, 0, 80), Color(80, 0, 120),
         Color(160, 0, 80), Color(200, 50, 0), Color(230, 140, 0),
         Color(255, 230, 100)],
        PaletteType.SEQUENTIAL,
        description="Thermal imaging style heatmap",
    ),
    "Amber": Palette(
        "Amber",
        [Color(10, 8, 0), Color(40, 25, 0), Color(100, 60, 0),
         Color(180, 120, 20), Color(240, 190, 60), Color(255, 230, 140)],
        PaletteType.SEQUENTIAL,
        description="Warm amber monochrome",
    ),
    "Ice": Palette(
        "Ice",
        [Color(5, 5, 20), Color(10, 20, 60), Color(30, 60, 140),
         Color(80, 140, 210), Color(160, 210, 240), Color(220, 240, 255)],
        PaletteType.SEQUENTIAL,
        description="Cool ice crystal gradient",
    ),
}


class PaletteGenerator:
    """Generate palettes algorithmically."""

    @staticmethod
    def random_palette(n: int = 6) -> Palette:
        colors = [Color.from_hsv(random.uniform(0, 360),
                                  random.uniform(0.5, 1.0),
                                  random.uniform(0.5, 1.0)) for _ in range(n)]
        return Palette("Random", colors, PaletteType.CATEGORICAL)

    @staticmethod
    def analogous_palette(base_hue: float = None, n: int = 5) -> Palette:
        if base_hue is None:
            base_hue = random.uniform(0, 360)
        colors = [Color.from_hsv((base_hue + i * 30 - (n - 1) * 15) % 360, 0.7, 0.9)
                  for i in range(n)]
        return Palette("Analogous", colors, PaletteType.CATEGORICAL)

    @staticmethod
    def complementary_palette(base_hue: float = None, n: int = 4) -> Palette:
        if base_hue is None:
            base_hue = random.uniform(0, 360)
        comp = (base_hue + 180) % 360
        colors = []
        for i in range(n):
            h = base_hue if i < n // 2 else comp
            s = 0.5 + 0.4 * (i / max(1, n - 1))
            v = 0.5 + 0.4 * (i / max(1, n - 1))
            colors.append(Color.from_hsv(h, s, v))
        return Palette("Complementary", colors, PaletteType.DIVERGING)

    @staticmethod
    def triadic_palette(base_hue: float = None) -> Palette:
        if base_hue is None:
            base_hue = random.uniform(0, 360)
        hues = [(base_hue + i * 120) % 360 for i in range(3)]
        colors = [Color.from_hsv(h, 0.75, 0.9) for h in hues]
        return Palette("Triadic", colors, PaletteType.CATEGORICAL)

    @staticmethod
    def gradient_palette(start_hex: str, end_hex: str, steps: int = 8) -> Palette:
        c1 = Color.from_hex(start_hex)
        c2 = Color.from_hex(end_hex)
        colors = [c1.lerp(c2, i / max(1, steps - 1)) for i in range(steps)]
        return Palette("Gradient", colors, PaletteType.SEQUENTIAL)


class PaletteManager:
    """Manages built-in and custom palettes."""

    def __init__(self):
        self._custom: Dict[str, Palette] = {}

    def get_palette(self, name: str) -> Optional[Palette]:
        if name in BUILTIN_PALETTES:
            return BUILTIN_PALETTES[name]
        return self._custom.get(name)

    def add_custom(self, palette: Palette):
        self._custom[palette.name] = palette

    @staticmethod
    def get_names() -> List[str]:
        names = list(BUILTIN_PALETTES.keys())
        return names


# ============================================================
# SECTION 3: Presets
# ============================================================

PRESETS: Dict[str, List[Tuple[int, int]]] = {
    # Still lifes
    "Block": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "Beehive": [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)],
    "Loaf": [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 3), (3, 2)],
    "Boat": [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)],
    "Tub": [(0, 1), (1, 0), (1, 2), (2, 1)],
    # Oscillators
    "Blinker": [(0, 0), (0, 1), (0, 2)],
    "Toad": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
    "Beacon": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)],
    "Pulsar": [
        (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10),
        (2, 0), (2, 5), (2, 7), (2, 12),
        (3, 0), (3, 5), (3, 7), (3, 12),
        (4, 0), (4, 5), (4, 7), (4, 12),
        (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (5, 10),
        (7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10),
        (8, 0), (8, 5), (8, 7), (8, 12),
        (9, 0), (9, 5), (9, 7), (9, 12),
        (10, 0), (10, 5), (10, 7), (10, 12),
        (12, 2), (12, 3), (12, 4), (12, 8), (12, 9), (12, 10),
    ],
    "Pentadecathlon": [
        (0, 1), (1, 1), (2, 0), (2, 2), (3, 1), (4, 1),
        (5, 1), (6, 1), (7, 0), (7, 2), (8, 1), (9, 1),
    ],
    # Spaceships
    "Glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "Lightweight Spaceship": [(0, 1), (0, 4), (1, 0), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3)],
    "Middleweight Spaceship": [
        (0, 1), (0, 4), (1, 0), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2),
        (3, 3), (4, 0), (4, 4),
    ],
    "Heavyweight Spaceship": [
        (0, 2), (0, 3), (1, 0), (1, 5), (2, 6), (3, 0), (3, 6),
        (4, 1), (4, 2), (4, 5), (4, 6), (5, 2), (5, 3),
    ],
    # Methuselahs
    "R-pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "Diehard": [(0, 6), (1, 0), (1, 1), (2, 1), (2, 5), (2, 6), (2, 7)],
    "Acorn": [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)],
    # Guns
    "Gosper Glider Gun": [
        (0, 24),
        (1, 22), (1, 24),
        (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
        (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35),
        (4, 0), (4, 1), (4, 10), (4, 16), (4, 20), (4, 21),
        (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17), (5, 22), (5, 24),
        (6, 10), (6, 16), (6, 24),
        (7, 11), (7, 15),
        (8, 12), (8, 13),
    ],
    # Other patterns
    "LWSS": [(0, 1), (0, 4), (1, 0), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3)],
    "Pulsar (period 3)": [
        (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10),
        (2, 0), (2, 5), (2, 7), (2, 12),
        (3, 0), (3, 5), (3, 7), (3, 12),
        (4, 0), (4, 5), (4, 7), (4, 12),
        (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (5, 10),
        (7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10),
        (8, 0), (8, 5), (8, 7), (8, 12),
        (9, 0), (9, 5), (9, 7), (9, 12),
        (10, 0), (10, 5), (10, 7), (10, 12),
        (12, 2), (12, 3), (12, 4), (12, 8), (12, 9), (12, 10),
    ],
    "Block (2x2)": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "Clock": [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)],
    "Fountain": [(0, 0), (1, 0), (1, 2), (2, 1)],
    "Galaxy": [(0, 0), (0, 5), (1, 2), (1, 3), (2, 0), (2, 5), (3, 1), (3, 4), (4, 2), (4, 3)],
    "Herschel": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "Pi-heptomino": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 4), (2, 2)],
    "B-heptomino": [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0), (3, 0), (3, 1)],
    "Switch Engine": [
        (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0),
    ],
    "Rabbit": [
        (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2), (3, 3), (4, 4),
    ],
    "Century": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 2), (4, 2), (4, 3), (5, 2)],
}

PRESET_CATEGORIES: Dict[str, List[str]] = {
    "Still Lifes": ["Block", "Beehive", "Loaf", "Boat", "Tub"],
    "Oscillators": ["Blinker", "Toad", "Beacon", "Pulsar", "Pentadecathlon", "Clock"],
    "Spaceships": ["Glider", "Lightweight Spaceship", "Middleweight Spaceship",
                   "Heavyweight Spaceship", "LWSS"],
    "Methuselahs": ["R-pentomino", "Diehard", "Acorn", "Pi-heptomino",
                    "B-heptomino", "Herschel", "Rabbit", "Century"],
    "Guns": ["Gosper Glider Gun"],
    "Other": ["Fountain", "Galaxy", "Switch Engine", "Pulsar (period 3)",
              "Block (2x2)"],
}


def rotate_pattern(pattern: List[Tuple[int, int]], times: int = 1) -> List[Tuple[int, int]]:
    """Rotate a pattern 90 degrees clockwise `times` times."""
    result = pattern
    for _ in range(times % 4):
        result = [(c, -r) for r, c in result]
    min_r = min(r for r, c in result)
    min_c = min(c for r, c in result)
    return [(r - min_r, c - min_c) for r, c in result]


def flip_pattern(pattern: List[Tuple[int, int]], horizontal: bool = True) -> List[Tuple[int, int]]:
    """Flip a pattern horizontally or vertically."""
    if horizontal:
        result = [(r, -c) for r, c in pattern]
        min_c = min(c for r, c in result)
        return [(r, c - min_c) for r, c in result]
    else:
        result = [(-r, c) for r, c in pattern]
        min_r = min(r for r, c in result)
        return [(r - min_r, c) for r, c in result]


def parse_rle(rle_string: str) -> List[Tuple[int, int]]:
    """Parse a Run Length Encoded pattern string into coordinates."""
    lines = rle_string.strip().split('\n')
    cells = []
    row = 0
    col = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('!'):
            continue
        # Strip header line like "x = 3, y = 3, rule = B3/S23"
        # but keep any RLE data that follows on the same line
        if line.lower().startswith('x '):
            # Find where the RLE data begins (after the header)
            rest = re.sub(r'^x\s*=\s*\d+\s*,\s*y\s*=\s*\d+\s*(,\s*rule\s*=\s*[^\s]+)?\s*', '', line)
            if not rest:
                continue
            line = rest
        count_str = ''
        for ch in line:
            if ch.isdigit():
                count_str += ch
            elif ch == 'b':
                count = int(count_str) if count_str else 1
                col += count
                count_str = ''
            elif ch == 'o':
                count = int(count_str) if count_str else 1
                for _ in range(count):
                    cells.append((row, col))
                    col += 1
                count_str = ''
            elif ch == '$':
                row += int(count_str) if count_str else 1
                col = 0
                count_str = ''
            elif ch == '!':
                break
    return cells


def pattern_to_rle(pattern: List[Tuple[int, int]]) -> str:
    """Convert a list of cell coordinates to RLE format."""
    if not pattern:
        return "!"
    max_r = max(r for r, c in pattern)
    max_c = max(c for r, c in pattern)
    rows = max_r + 1
    cols = max_c + 1
    grid = [[False] * cols for _ in range(rows)]
    for r, c in pattern:
        grid[r][c] = True
    data_parts = []
    for r in range(rows):
        run = 0
        prev_state = None
        for c in range(cols + 1):
            state = grid[r][c] if c < cols else None
            if state == prev_state:
                run += 1
            else:
                if prev_state is not None:
                    prefix = str(run) if run > 1 else ''
                    data_parts.append(prefix + ('o' if prev_state else 'b'))
                run = 1
                prev_state = state
        data_parts.append('$')
    # Remove trailing $
    while data_parts and data_parts[-1] == '$':
        data_parts.pop()
    data_parts.append('!')
    rle = f"x = {cols}, y = {rows}, rule = B3/S23\n" + ''.join(data_parts)
    # Compress: "1o" -> "o", "1b" -> "b", "1$" -> "$"
    # But only standalone 1, not part of multi-digit like "11b"
    rle = re.sub(r'(?<!\d)1([ob$])', r'\1', rle)
    return rle


# ============================================================
# SECTION 4: Rulesets
# ============================================================

class TotalisticRule:
    """Represents a totalistic outer-totalistic CA rule like B3/S23."""

    def __init__(self, birth: List[int] = None, survive: List[int] = None):
        self.birth = birth or []
        self.survive = survive or []

    @classmethod
    def from_string(cls, rule_str: str) -> 'TotalisticRule':
        """Parse 'B3/S23' or 'B368/S245' format."""
        rule_str = rule_str.strip().upper()
        birth = []
        survive = []
        m = re.match(r'B(\d*)/?S(\d*)', rule_str)
        if m:
            birth_str = m.group(1)
            survive_str = m.group(2)
            birth = sorted(int(c) for c in birth_str) if birth_str else []
            survive = sorted(int(c) for c in survive_str) if survive_str else []
        return cls(birth, survive)

    def to_string(self) -> str:
        b = ''.join(str(n) for n in self.birth)
        s = ''.join(str(n) for n in self.survive)
        return f"B{b}/S{s}"

    def __repr__(self) -> str:
        return f"TotalisticRule({self.to_string()})"


RULE_DESCRIPTIONS: Dict[str, str] = {
    "B3/S23": "Conway's Game of Life - The most famous CA rule",
    "B36/S23": "HighLife - Like Life but with 6-neighbor birth",
    "B3678/S34678": "Day & Night - Symmetric rule, stable patterns",
    "B3/S012345678": "Maze - Expanding maze-like structures",
    "B3/S12345": "Maze2 - Alternative maze generator",
    "B368/S245": "Morley (Move) - Chaotic moving patterns",
    "B2/S": "Seeds - Every cell dies, only birth matters",
    "B3/S234": "Diamoeba - Diamond-shaped amoeba growth",
    "B378/S235678": "Replicator - Self-replicating structures",
    "B36/S125": "2x2 - Patterns made of 2x2 blocks",
    "B45678/S2345": "Walled Cities - Expanding city walls",
    "B3/S1234": "Coral - Coral-like growth patterns",
    "B1357/S1357": "Replicator - XOR replicator rule",
    "B1/S1": "Gnarl - Chaotic growth from single cell",
    "B1/S12": "Gnarl2 - Variant gnarl rule",
    "B2/S12": "Flock - Flocking behavior patterns",
    "B368/S245": "Morley - Chaotic moving patterns",
}


class RuleAnalyzer:
    """Analyze CA rule properties."""

    def __init__(self, rule: TotalisticRule):
        self.rule = rule

    def get_birth_count(self) -> int:
        return len(self.rule.birth)

    def get_survive_count(self) -> int:
        return len(self.rule.survive)

    def is_life_like(self) -> bool:
        return 3 in self.rule.birth and 2 in self.rule.survive and 3 in self.rule.survive

    def get_complexity_score(self) -> float:
        """Estimate complexity based on rule parameters."""
        b = len(self.rule.birth)
        s = len(self.rule.survive)
        total = b + s
        # Rules with moderate birth/survive counts tend to be complex
        if total < 2:
            return 0.1
        if total > 12:
            return 0.3
        return min(1.0, total / 10.0)

    def classify(self) -> str:
        """Classify the rule into a category."""
        if not self.rule.birth and not self.rule.survive:
            return "Empty"
        if not self.rule.survive:
            return "Explosive"
        if not self.rule.birth:
            return "Dying"
        if self.is_life_like():
            return "Life-like"
        if 0 in self.rule.birth:
            return "Expansive"
        if 8 in self.rule.survive:
            return "Dense"
        return "Other"

    def get_description(self) -> str:
        key = self.rule.to_string()
        return RULE_DESCRIPTIONS.get(key, f"Custom rule: {key}")


def get_suggested_rules() -> List[str]:
    """Return a list of suggested rule strings."""
    return [
        "B3/S23 - Life",
        "B36/S23 - HighLife",
        "B3678/S34678 - Day&Night",
        "B3/S012345678 - Maze",
        "B3/S12345 - Maze2",
        "B368/S245 - Morley",
        "B2/S - Seeds",
        "B3/S234 - Diamoeba",
        "B378/S235678 - Replicator",
        "B36/S125 - 2x2",
        "B45678/S2345 - Walled Cities",
        "B3/S1234 - Coral",
        "B1357/S1357 - Replicator",
        "B1/S1 - Gnarl",
        "Custom",
    ]


# ============================================================
# SECTION 5: Backends
# ============================================================

HAS_NUMBA = False
HAS_CUPY = False


def _evolve_numpy(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Evolve grid one step with wrapping using NumPy convolution."""
    rows, cols = grid.shape
    padded = np.pad(grid, 1, mode='wrap')
    neighbors = np.zeros((rows, cols), dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbors += padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    birth_mask = birth[neighbors]
    survive_mask = survive[neighbors]
    new_grid = np.where((grid == 0) & birth_mask, 1,
               np.where((grid > 0) & survive_mask, np.minimum(grid + 1, 255), 0)).astype(np.int32)
    return new_grid


def _evolve_numpy_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Evolve grid one step without wrapping using NumPy."""
    rows, cols = grid.shape
    padded = np.zeros((rows + 2, cols + 2), dtype=grid.dtype)
    padded[1:1 + rows, 1:1 + cols] = grid
    neighbors = np.zeros((rows, cols), dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbors += padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
    birth_mask = birth[neighbors]
    survive_mask = survive[neighbors]
    new_grid = np.where((grid == 0) & birth_mask, 1,
               np.where((grid > 0) & survive_mask, np.minimum(grid + 1, 255), 0)).astype(np.int32)
    return new_grid


def _evolve_numba(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    return _evolve_numpy(grid, birth, survive)


def _evolve_numba_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    return _evolve_numpy_nowrap(grid, birth, survive)


def _evolve_cupy(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    return _evolve_numpy(grid, birth, survive)


def _evolve_cupy_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    return _evolve_numpy_nowrap(grid, birth, survive)


class BackendManager:
    """Manages computation backend selection."""

    AVAILABLE_BACKENDS = ["Auto", "Python", "NumPy"]

    def __init__(self, preferred: str = "Auto"):
        self._backend = preferred
        self._effective = "NumPy"

    def evolve(self, grid: np.ndarray, birth: np.ndarray, survive: np.ndarray,
               wrap: bool = True) -> np.ndarray:
        if wrap:
            return _evolve_numpy(grid, birth, survive)
        else:
            return _evolve_numpy_nowrap(grid, birth, survive)

    def set_backend(self, backend: str):
        self._backend = backend
        if backend == "Auto":
            self._effective = "NumPy"
        elif backend in ("NumPy", "Python"):
            self._effective = backend
        else:
            self._effective = "NumPy"

    def get_effective_backend(self) -> str:
        return self._effective


# ============================================================
# SECTION 6: Visual Effects
# ============================================================

class VisualMode(Enum):
    STANDARD = "Standard"
    AGE = "Age"
    HEATMAP = "Heatmap"
    NEIGHBOR_COUNT = "Neighbor Count"
    OUTLINE = "Outline"
    GRADIENT = "Gradient"


@dataclass
class VisualSettings:
    mode: VisualMode = VisualMode.STANDARD
    glow_enabled: bool = False
    glow_radius: int = 2
    glow_intensity: float = 0.5
    trail_enabled: bool = False
    trail_length: int = 15
    vignette_enabled: bool = False
    vignette_intensity: float = 0.3
    birth_flash_enabled: bool = False
    death_flash_enabled: bool = False
    outline_thickness: int = 1
    heatmap_decay: float = 0.99


class AgeTracker:
    """Tracks cell age for age-based coloring."""

    def __init__(self, rows: int, cols: int):
        self.ages = np.zeros((rows, cols), dtype=np.int32)
        self.rows = rows
        self.cols = cols
        self._colormap = "plasma"

    def reset(self):
        self.ages.fill(0)

    def resize(self, rows: int, cols: int):
        new_ages = np.zeros((rows, cols), dtype=np.int32)
        mr = min(self.rows, rows)
        mc = min(self.cols, cols)
        new_ages[:mr, :mc] = self.ages[:mr, :mc]
        self.ages = new_ages
        self.rows = rows
        self.cols = cols

    def update(self, grid: np.ndarray, prev_grid: np.ndarray):
        # Cells that are alive and were alive: age++
        alive_now = grid > 0
        alive_before = prev_grid > 0
        self.ages[alive_now & alive_before] += 1
        self.ages[alive_now & ~alive_before] = 1
        self.ages[~alive_now] = 0

    def get_age_color_lut(self, max_age: int = 256) -> np.ndarray:
        """Return a (max_age+1, 3) colormap LUT."""
        if max_age <= 0:
            max_age = 256
        lut = np.zeros((max_age, 3), dtype=np.uint8)
        cmap_fn = {
            "plasma": self._plasma_colormap,
            "viridis": self._viridis_colormap,
            "inferno": self._inferno_colormap,
            "cool": self._cool_colormap,
            "hot": self._hot_colormap,
            "rainbow": self._rainbow_colormap,
            "amber": self._amber_colormap,
        }.get(self._colormap, self._plasma_colormap)
        for i in range(max_age):
            t = min(i / max(1, max_age - 1), 1.0)
            lut[i] = cmap_fn(t)
        return lut

    @staticmethod
    def _plasma_colormap(t: float) -> Tuple[int, int, int]:
        r = int(min(255, max(0, (0.050383 + t * (2.478005 + t * (-1.443748 + t * 0.510837))) * 255)))
        g = int(min(255, max(0, (0.019995 + t * (1.286049 + t * (-1.680284 + t * 1.994617))) * 255)))
        b = int(min(255, max(0, (0.529285 + t * (0.227669 + t * (-1.942103 + t * 2.793545))) * 255)))
        return (r, g, b)

    @staticmethod
    def _viridis_colormap(t: float) -> Tuple[int, int, int]:
        r = int(min(255, max(0, (0.267004 + t * (0.329415 + t * (-1.396245 + t * 2.248232))) * 255)))
        g = int(min(255, max(0, (0.004874 + t * (1.368563 + t * (-0.814832 + t * 1.452195))) * 255)))
        b = int(min(255, max(0, (0.329415 + t * (1.441754 + t * (-2.477044 + t * 2.716548))) * 255)))
        return (r, g, b)

    @staticmethod
    def _inferno_colormap(t: float) -> Tuple[int, int, int]:
        r = int(min(255, max(0, (0.001462 + t * (2.568412 + t * (-3.654737 + t * 3.126896))) * 255)))
        g = int(min(255, max(0, (0.000000 + t * (1.233621 + t * (-1.880101 + t * 2.590823))) * 255)))
        b = int(min(255, max(0, (0.014630 + t * (0.674059 + t * (-1.247536 + t * 1.686399))) * 255)))
        return (r, g, b)

    @staticmethod
    def _cool_colormap(t: float) -> Tuple[int, int, int]:
        r = int(t * 180)
        g = int(100 + t * 155)
        b = 255
        return (min(255, r), min(255, g), min(255, b))

    @staticmethod
    def _hot_colormap(t: float) -> Tuple[int, int, int]:
        if t < 0.33:
            r = int(t / 0.33 * 255)
            return (r, 0, 0)
        elif t < 0.66:
            r = 255
            g = int((t - 0.33) / 0.33 * 255)
            return (r, g, 0)
        else:
            r = 255
            g = 255
            b = int((t - 0.66) / 0.34 * 255)
            return (r, g, b)

    @staticmethod
    def _rainbow_colormap(t: float) -> Tuple[int, int, int]:
        h = t * 360
        c = Color.from_hsv(h, 1.0, 1.0)
        return (c.r, c.g, c.b)

    @staticmethod
    def _amber_colormap(t: float) -> Tuple[int, int, int]:
        r = int(min(255, 20 + t * 235))
        g = int(min(255, 10 + t * 170))
        b = int(min(255, t * 60))
        return (r, g, b)


class HeatmapTracker:
    """Tracks cell activity for heatmap visualization."""

    def __init__(self, rows: int, cols: int, decay: float = 0.995):
        self.heatmap = np.zeros((rows, cols), dtype=np.float64)
        self.rows = rows
        self.cols = cols
        self.decay = decay

    def reset(self):
        self.heatmap.fill(0)

    def resize(self, rows: int, cols: int):
        new_heat = np.zeros((rows, cols), dtype=np.float64)
        mr = min(self.rows, rows)
        mc = min(self.cols, cols)
        new_heat[:mr, :mc] = self.heatmap[:mr, :mc]
        self.heatmap = new_heat
        self.rows = rows
        self.cols = cols

    def update(self, grid: np.ndarray, prev_grid: np.ndarray):
        # Decay
        self.heatmap *= self.decay
        # Mark births
        births = (grid > 0) & (prev_grid == 0)
        deaths = (grid == 0) & (prev_grid > 0)
        self.heatmap[births] = np.minimum(self.heatmap[births] + 1.0, 5.0)
        self.heatmap[deaths] = np.minimum(self.heatmap[deaths] + 0.5, 5.0)

    def get_colors(self, bg_color: Tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
        """Return (rows, cols, 3) uint8 array based on heatmap intensity."""
        colors = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        colors[:, :] = bg_color
        normalized = np.clip(self.heatmap / 3.0, 0, 1)
        for c in range(3):
            channel = (normalized * 255).astype(np.uint8)
            colors[:, :, 0] = np.where(normalized < 0.5,
                                        (normalized * 2 * 255).astype(np.uint8),
                                        255).astype(np.uint8)
            colors[:, :, 1] = np.where(normalized < 0.5,
                                        0,
                                        ((normalized - 0.5) * 2 * 255).astype(np.uint8)).astype(np.uint8)
            colors[:, :, 2] = np.where(normalized > 0.75,
                                        ((normalized - 0.75) * 4 * 255).astype(np.uint8),
                                        0).astype(np.uint8)
        return colors


class BirthDeathTracker:
    """Tracks recent births and deaths for flash effects."""

    def __init__(self, rows: int, cols: int):
        self.births = np.zeros((rows, cols), dtype=np.int32)
        self.deaths = np.zeros((rows, cols), dtype=np.int32)
        self.rows = rows
        self.cols = cols
        self._max_frames = 8

    def reset(self):
        self.births.fill(0)
        self.deaths.fill(0)

    def resize(self, rows: int, cols: int):
        new_b = np.zeros((rows, cols), dtype=np.int32)
        new_d = np.zeros((rows, cols), dtype=np.int32)
        mr = min(self.rows, rows)
        mc = min(self.cols, cols)
        new_b[:mr, :mc] = self.births[:mr, :mc]
        new_d[:mr, :mc] = self.deaths[:mr, :mc]
        self.births = new_b
        self.deaths = new_d
        self.rows = rows
        self.cols = cols

    def update(self, grid: np.ndarray, prev_grid: np.ndarray):
        new_births = (grid > 0) & (prev_grid == 0)
        new_deaths = (grid == 0) & (prev_grid > 0)
        self.births = np.where(new_births, self._max_frames, np.maximum(self.births - 1, 0))
        self.deaths = np.where(new_deaths, self._max_frames, np.maximum(self.deaths - 1, 0))

    def get_overlay(self) -> np.ndarray:
        """Return (rows, cols, 3) float array with birth (green) / death (red) overlay."""
        overlay = np.zeros((self.rows, self.cols, 3), dtype=np.float64)
        if np.any(self.births > 0):
            intensity = self.births.astype(np.float64) / self._max_frames
            overlay[:, :, 1] = intensity * 200  # green
            overlay[:, :, 2] = intensity * 100  # some blue
        if np.any(self.deaths > 0):
            intensity = self.deaths.astype(np.float64) / self._max_frames
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], intensity * 200)  # red
        return overlay


class GlowEffect:
    """Applies a glow effect around alive cells. No scipy dependency."""

    def __init__(self, radius: int = 2, intensity: float = 0.5):
        self.radius = radius
        self.intensity = intensity

    def apply(self, colors: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Apply glow effect. Returns (rows, cols, 3) uint8 array."""
        if grid is None or not np.any(grid > 0):
            return colors
        rows, cols = grid.shape
        glow = np.zeros((rows, cols, 3), dtype=np.float64)
        alive = (grid > 0).astype(np.float64)
        # Simple box blur for glow - no scipy needed
        for dr in range(-self.radius, self.radius + 1):
            for dc in range(-self.radius, self.radius + 1):
                if dr == 0 and dc == 0:
                    continue
                dist = math.sqrt(dr * dr + dc * dc)
                if dist > self.radius + 0.5:
                    continue
                weight = self.intensity * (1.0 - dist / (self.radius + 1))
                # Shift the alive mask
                sr = slice(max(0, dr), min(rows, rows + dr))
                sc = slice(max(0, dc), min(cols, cols + dc))
                tr = slice(max(0, -dr), min(rows, rows - dr))
                tc = slice(max(0, -dc), min(cols, cols - dc))
                glow[tr, tc, :] += alive[sr, sc, np.newaxis] * weight
        # Add glow colors (use a teal/cyan tint)
        glow_color = np.array([0, 180, 200], dtype=np.float64)
        for c in range(3):
            glow[:, :, c] *= glow_color[c] / 255.0
        result = colors.astype(np.float64) + glow
        return np.clip(result, 0, 255).astype(np.uint8)


class OutlineRenderer:
    """Renders cells with outlines."""

    def __init__(self, color: Tuple[int, int, int] = (200, 200, 200),
                 thickness: int = 1):
        self.color = color
        self.thickness = thickness

    def render(self, grid: np.ndarray, bg_color: Tuple[int, int, int] = (30, 30, 30)) -> np.ndarray:
        """Return (rows, cols, 3) uint8 array with outlined cells."""
        rows, cols = grid.shape
        colors = np.zeros((rows, cols, 3), dtype=np.uint8)
        colors[:, :] = bg_color

        if not np.any(grid > 0):
            return colors

        # Fill alive cells with a color
        alive = grid > 0
        colors[alive] = [0, 160, 130]

        # Find edges (alive cells adjacent to dead cells)
        padded = np.pad(alive.astype(np.int32), 1, mode='constant', constant_values=0)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor_alive = padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols].astype(bool)
                edge = alive & ~neighbor_alive
                colors[edge] = self.color

        return colors


class NeighborCountVisualizer:
    """Visualizes neighbor count as colors."""

    def __init__(self):
        self._lut = self._build_lut()

    @staticmethod
    def _build_lut() -> np.ndarray:
        """Build a color LUT for neighbor counts 0-8."""
        lut = np.zeros((9, 3), dtype=np.uint8)
        # 0: dark, 1-2: blue, 3: green, 4-5: yellow, 6-7: orange, 8: red
        colors = [
            (20, 20, 30),    # 0
            (30, 50, 150),   # 1
            (40, 80, 200),   # 2
            (30, 180, 80),   # 3
            (180, 200, 40),  # 4
            (220, 180, 30),  # 5
            (230, 120, 20),  # 6
            (220, 60, 30),   # 7
            (200, 20, 20),   # 8
        ]
        for i, c in enumerate(colors):
            lut[i] = c
        return lut

    def render(self, grid: np.ndarray) -> np.ndarray:
        """Return (rows, cols, 3) uint8 array showing neighbor counts."""
        rows, cols = grid.shape
        padded = np.pad(grid, 1, mode='wrap')
        neighbors = np.zeros((rows, cols), dtype=np.int32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbors += (padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols] > 0).astype(np.int32)
        neighbors = np.clip(neighbors, 0, 8)
        return self._lut[neighbors]


class VignetteEffect:
    """Applies a vignette (darkened edges) to the rendered image."""

    def __init__(self, intensity: float = 0.3):
        self.intensity = intensity
        self._mask = None
        self._shape = None

    def _build_mask(self, rows: int, cols: int) -> np.ndarray:
        if self._mask is not None and self._shape == (rows, cols):
            return self._mask
        y = np.linspace(-1, 1, rows)
        x = np.linspace(-1, 1, cols)
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(xx ** 2 + yy ** 2)
        dist = np.clip(dist / 1.414, 0, 1)
        self._mask = 1.0 - dist * self.intensity
        self._shape = (rows, cols)
        return self._mask

    def apply(self, colors: np.ndarray) -> np.ndarray:
        """Apply vignette. Returns (rows, cols, 3) uint8 array."""
        rows, cols = colors.shape[:2]
        mask = self._build_mask(rows, cols)
        result = (colors.astype(np.float64) * mask[:, :, np.newaxis])
        return np.clip(result, 0, 255).astype(np.uint8)


class GradientOverlay:
    """Applies a gradient overlay to the rendered image."""

    def __init__(self, direction: str = "vertical",
                 color: Tuple[int, int, int] = (0, 0, 40),
                 intensity: float = 0.3):
        self.direction = direction
        self.color = color
        self.intensity = intensity

    def apply(self, colors: np.ndarray) -> np.ndarray:
        rows, cols = colors.shape[:2]
        if self.direction == "vertical":
            gradient = np.linspace(0, 1, rows)[:, np.newaxis]
        else:
            gradient = np.linspace(0, 1, cols)[np.newaxis, :]
        overlay = np.zeros((rows, cols, 3), dtype=np.float64)
        for c in range(3):
            overlay[:, :, c] = gradient * self.color[c] * self.intensity
        result = colors.astype(np.float64) + overlay
        return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# SECTION 7: Themes
# ============================================================

THEMES = {
    "light": {
        "bg": (240, 240, 240),
        "grid": (220, 220, 220),
        "states": [
            (240, 240, 240),
            (30, 30, 30),
            (0, 120, 100),
            (0, 80, 160),
            (180, 60, 0),
            (120, 0, 120),
            (0, 100, 0),
            (160, 120, 0),
        ],
    },
    "dark": {
        "bg": (30, 30, 30),
        "grid": (50, 50, 50),
        "states": [
            (30, 30, 30),
            (0, 212, 168),
            (255, 100, 100),
            (100, 150, 255),
            (255, 200, 50),
            (200, 100, 255),
            (100, 255, 100),
            (255, 150, 50),
        ],
    },
    "matrix": {
        "bg": (0, 5, 0),
        "grid": (0, 20, 0),
        "states": [
            (0, 5, 0),
            (0, 255, 0),
            (0, 180, 0),
            (0, 120, 0),
            (0, 80, 0),
            (0, 220, 80),
            (80, 255, 80),
            (0, 160, 40),
        ],
    },
    "ocean": {
        "bg": (5, 10, 30),
        "grid": (15, 25, 50),
        "states": [
            (5, 10, 30),
            (60, 180, 220),
            (0, 100, 180),
            (120, 220, 255),
            (30, 60, 140),
            (100, 200, 180),
            (0, 150, 200),
            (180, 240, 255),
        ],
    },
    "cyberpunk": {
        "bg": (10, 5, 20),
        "grid": (30, 15, 45),
        "states": [
            (10, 5, 20),
            (255, 0, 110),
            (0, 255, 200),
            (130, 0, 255),
            (255, 230, 0),
            (0, 150, 255),
            (255, 100, 200),
            (200, 255, 0),
        ],
    },
}


def get_theme_lut(theme_name: str, max_state: int) -> np.ndarray:
    """Get a (max_state, 3) LUT for the given theme."""
    theme = THEMES.get(theme_name, THEMES["dark"])
    states = theme["states"]
    lut = np.zeros((max_state, 3), dtype=np.uint8)
    n = len(states)
    for i in range(max_state):
        if i < n:
            lut[i] = states[i]
        else:
            # Cycle or repeat last
            lut[i] = states[i % n]
    return lut


def get_grid_color(theme_name: str) -> Tuple[int, int, int]:
    """Get the grid line color for a theme."""
    theme = THEMES.get(theme_name, THEMES["dark"])
    return theme["grid"]


def get_background_color(theme_name: str) -> Tuple[int, int, int]:
    """Get the background color for a theme."""
    theme = THEMES.get(theme_name, THEMES["dark"])
    return theme["bg"]


# ============================================================
# SECTION 8: UndoStack
# ============================================================

class UndoStack:
    """Manages undo/redo history for the grid state."""

    def __init__(self, max_size: int = 50):
        self._undo_stack: deque = deque(maxlen=max_size)
        self._redo_stack: deque = deque(maxlen=max_size)

    def push(self, state: np.ndarray):
        self._undo_stack.append(state.copy())
        self._redo_stack.clear()

    def undo(self, current: np.ndarray) -> Optional[np.ndarray]:
        if not self._undo_stack:
            return None
        prev = self._undo_stack.pop()
        self._redo_stack.append(current.copy())
        return prev

    def redo(self, current: np.ndarray) -> Optional[np.ndarray]:
        if not self._redo_stack:
            return None
        next_state = self._redo_stack.pop()
        self._undo_stack.append(current.copy())
        return next_state

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)


# ============================================================
# SECTION 9: UI Widget Library
# ============================================================

# UI Design Colors (dark theme)
_UI_BG = (26, 26, 46)
_UI_ACCENT = (0, 212, 168)
_UI_TEXT = (200, 200, 212)
_UI_TEXT_DIM = (120, 120, 150)
_UI_BTN = (46, 46, 80)
_UI_BTN_HOVER = (62, 62, 104)
_UI_BTN_PRESSED = (80, 80, 138)
_UI_BORDER = (46, 46, 72)
_UI_INPUT = (34, 34, 58)
_UI_CHECK = (0, 212, 168)

# Default font for widgets
_DEFAULT_FONT = None


def _get_default_font():
    global _DEFAULT_FONT
    if _DEFAULT_FONT is None:
        _DEFAULT_FONT = pygame.font.Font(None, 16)
    return _DEFAULT_FONT


class Button:
    """A clickable button widget."""

    def __init__(self, rect: pygame.Rect, text: str, callback=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self._hovered = False
        self._pressed = False
        self._is_active_tab = False

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._pressed = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._pressed and self.rect.collidepoint(event.pos):
                self._pressed = False
                if self.callback:
                    self.callback()
                return True
            self._pressed = False
        return False

    def draw(self, surface, font=None):
        if font is None:
            font = _get_default_font()
        # Determine color
        if self._is_active_tab:
            bg = _UI_BTN_HOVER
            border_color = _UI_ACCENT
            border_w = 2
        elif self._pressed:
            bg = _UI_BTN_PRESSED
            border_color = _UI_BORDER
            border_w = 1
        elif self._hovered:
            bg = _UI_BTN_HOVER
            border_color = _UI_BORDER
            border_w = 1
        else:
            bg = _UI_BTN
            border_color = _UI_BORDER
            border_w = 1

        pygame.draw.rect(surface, bg, self.rect, border_radius=5)
        pygame.draw.rect(surface, border_color, self.rect, border_w, border_radius=5)

        text_color = _UI_ACCENT if self._is_active_tab else _UI_TEXT
        text_surf = font.render(self.text, True, text_color)
        tx = self.rect.x + (self.rect.width - text_surf.get_width()) // 2
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))

    def update_position(self, x: int, y: int):
        self.rect.topleft = (x, y)


class Label:
    """A simple text label widget."""

    def __init__(self, rect: pygame.Rect, text: str, color=None, font_size='sm'):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.color = color if color is not None else _UI_TEXT
        self.font_size = font_size

    def handle_event(self, event) -> bool:
        return False

    def draw(self, surface, font=None):
        if font is None:
            font = _get_default_font()
        text_surf = font.render(self.text, True, self.color)
        surface.blit(text_surf, (self.rect.x, self.rect.y))


class Slider:
    """A horizontal slider widget."""

    def __init__(self, rect, min_val, max_val, value, callback=None, label=""):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.callback = callback
        self.label = label
        self._dragging = False

    @property
    def _ratio(self):
        if self.max_val == self.min_val:
            return 0.0
        return (self.value - self.min_val) / (self.max_val - self.min_val)

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if click is on the track area (expanded hit area)
            track_rect = pygame.Rect(self.rect.x - 4, self.rect.y - 4,
                                     self.rect.width + 8, self.rect.height + 8)
            if track_rect.collidepoint(event.pos):
                self._dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging:
                self._dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION:
            if self._dragging:
                self._update_value(event.pos[0])
                return True
        return False

    def _update_value(self, mouse_x: int):
        ratio = (mouse_x - self.rect.x) / max(1, self.rect.width)
        ratio = max(0.0, min(1.0, ratio))
        old = self.value
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        if self.callback and self.value != old:
            self.callback(self.value)

    def draw(self, surface, font=None):
        if font is None:
            font = _get_default_font()
        cy = self.rect.y + self.rect.height // 2
        # Track background
        track_rect = pygame.Rect(self.rect.x, cy - 2, self.rect.width, 4)
        pygame.draw.rect(surface, _UI_BORDER, track_rect, border_radius=2)
        # Filled portion
        fill_w = int(self._ratio * self.rect.width)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.rect.x, cy - 2, fill_w, 4)
            pygame.draw.rect(surface, _UI_ACCENT, fill_rect, border_radius=2)
        # Handle
        hx = self.rect.x + fill_w
        pygame.draw.circle(surface, _UI_ACCENT, (hx, cy), 6)
        pygame.draw.circle(surface, (255, 255, 255), (hx, cy), 3)


class SpinBox:
    """A numeric input box with +/- buttons."""

    def __init__(self, rect, min_val, max_val, value, step=1, callback=None):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.step = step
        self.callback = callback
        self._plus_rect = pygame.Rect(rect.right - 18, rect.y, 18, rect.height)
        self._minus_rect = pygame.Rect(rect.x, rect.y, 18, rect.height)
        self._plus_hovered = False
        self._minus_hovered = False

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._plus_hovered = self._plus_rect.collidepoint(event.pos)
            self._minus_hovered = self._minus_rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._plus_rect.collidepoint(event.pos):
                self.value = min(self.max_val, self.value + self.step)
                if self.callback:
                    self.callback(self.value)
                return True
            elif self._minus_rect.collidepoint(event.pos):
                self.value = max(self.min_val, self.value - self.step)
                if self.callback:
                    self.callback(self.value)
                return True
            elif self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface, font=None):
        if font is None:
            font = _get_default_font()
        # Input box background
        input_rect = pygame.Rect(self.rect.x + 18, self.rect.y,
                                 self.rect.width - 36, self.rect.height)
        pygame.draw.rect(surface, _UI_INPUT, input_rect, border_radius=3)
        pygame.draw.rect(surface, _UI_BORDER, input_rect, 1, border_radius=3)
        # Value text
        text_str = str(int(self.value))
        text_surf = font.render(text_str, True, _UI_TEXT)
        tx = input_rect.x + (input_rect.width - text_surf.get_width()) // 2
        ty = input_rect.y + (input_rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))
        # Minus button
        minus_color = _UI_BTN_HOVER if self._minus_hovered else _UI_BTN
        pygame.draw.rect(surface, minus_color, self._minus_rect, border_radius=3)
        pygame.draw.line(surface, _UI_TEXT,
                         (self._minus_rect.x + 5, self._minus_rect.centery),
                         (self._minus_rect.right - 5, self._minus_rect.centery), 2)
        # Plus button
        plus_color = _UI_BTN_HOVER if self._plus_hovered else _UI_BTN
        pygame.draw.rect(surface, plus_color, self._plus_rect, border_radius=3)
        cx, cy = self._plus_rect.centerx, self._plus_rect.centery
        pygame.draw.line(surface, _UI_TEXT, (cx - 4, cy), (cx + 4, cy), 2)
        pygame.draw.line(surface, _UI_TEXT, (cx, cy - 4), (cx, cy + 4), 2)


class CheckBox:
    """A checkbox with label text."""

    def __init__(self, rect, text, checked=False, callback=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.checked = checked
        self.callback = callback

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                if self.callback:
                    self.callback(self.checked)
                return True
        return False

    def draw(self, surface, font=None):
        if font is None:
            font = _get_default_font()
        # Checkbox square
        box_size = 14
        box_rect = pygame.Rect(self.rect.x, self.rect.y + (self.rect.height - box_size) // 2,
                               box_size, box_size)
        pygame.draw.rect(surface, _UI_INPUT, box_rect, border_radius=2)
        pygame.draw.rect(surface, _UI_BORDER, box_rect, 1, border_radius=2)
        if self.checked:
            # Draw checkmark
            pygame.draw.rect(surface, _UI_CHECK, box_rect.inflate(-4, -4), border_radius=1)
        # Label text
        text_surf = font.render(self.text, True, _UI_TEXT)
        tx = box_rect.right + 5
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))


class ComboBox:
    """A dropdown combo box widget."""

    def __init__(self, rect, items, selected_index=0, callback=None):
        self.rect = pygame.Rect(rect)
        self.items = list(items)
        self.selected_index = min(selected_index, len(items) - 1) if items else 0
        self.callback = callback
        self.is_open = False
        self._scroll_offset = 0
        self._hovered_index = -1
        self._item_height = 22

    @property
    def selected_text(self) -> str:
        return self.items[self.selected_index] if self.items else ""

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_open:
                # Check if clicking on an item in the dropdown
                dropdown_rect = self._get_dropdown_rect()
                if dropdown_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - dropdown_rect.y
                    idx = self._scroll_offset + rel_y // self._item_height
                    if 0 <= idx < len(self.items):
                        self.selected_index = idx
                        if self.callback:
                            self.callback(self.selected_index)
                    self.is_open = False
                    return True
                else:
                    self.is_open = False
                    return True
            else:
                if self.rect.collidepoint(event.pos):
                    self.is_open = True
                    self._scroll_offset = 0
                    self._hovered_index = -1
                    return True
        elif event.type == pygame.MOUSEMOTION:
            if self.is_open:
                dropdown_rect = self._get_dropdown_rect()
                if dropdown_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - dropdown_rect.y
                    self._hovered_index = self._scroll_offset + rel_y // self._item_height
                else:
                    self._hovered_index = -1
        elif event.type == pygame.MOUSEWHEEL:
            if self.is_open:
                self._scroll_offset -= event.y * 3
                self._scroll_offset = max(0, min(self._scroll_offset,
                                                  max(0, len(self.items) * self._item_height - 200)))
                return True
        return False

    def _get_dropdown_rect(self) -> pygame.Rect:
        max_h = min(len(self.items) * self._item_height, 200)
        return pygame.Rect(self.rect.x, self.rect.bottom, self.rect.width, max_h)

    def draw(self, surface, font=None):
        if font is None:
            font = _get_default_font()
        # Draw closed state
        pygame.draw.rect(surface, _UI_INPUT, self.rect, border_radius=4)
        pygame.draw.rect(surface, _UI_BORDER, self.rect, 1, border_radius=4)
        # Selected text (clipped)
        text_surf = font.render(self.selected_text, True, _UI_TEXT)
        clip = surface.get_clip()
        text_clip = pygame.Rect(self.rect.x + 5, self.rect.y,
                                self.rect.width - 22, self.rect.height)
        surface.set_clip(text_clip)
        tx = self.rect.x + 5
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))
        surface.set_clip(clip)
        # Down arrow
        ax = self.rect.right - 12
        ay = self.rect.centery
        arrow_color = _UI_ACCENT if self.is_open else _UI_TEXT_DIM
        pygame.draw.polygon(surface, arrow_color,
                            [(ax - 4, ay - 2), (ax + 4, ay - 2), (ax, ay + 3)])

    def draw_dropdown(self, surface):
        """Draw the dropdown list (called separately to be on top)."""
        if not self.is_open:
            return
        font = _get_default_font()
        dropdown_rect = self._get_dropdown_rect()
        # Background
        dd_surf = pygame.Surface((dropdown_rect.width, dropdown_rect.height), pygame.SRCALPHA)
        dd_surf.fill((*_UI_BG, 245))
        surface.blit(dd_surf, dropdown_rect.topleft)
        pygame.draw.rect(surface, _UI_ACCENT, dropdown_rect, 1, border_radius=2)

        # Items
        clip = surface.get_clip()
        surface.set_clip(dropdown_rect)
        for i in range(len(self.items)):
            y = dropdown_rect.y + (i - self._scroll_offset) * self._item_height
            if y + self._item_height < dropdown_rect.y:
                continue
            if y > dropdown_rect.bottom:
                break
            item_rect = pygame.Rect(dropdown_rect.x + 1, y,
                                    dropdown_rect.width - 2, self._item_height)
            if i == self.selected_index:
                pygame.draw.rect(surface, _UI_BTN_HOVER, item_rect)
            elif i == self._hovered_index:
                pygame.draw.rect(surface, (50, 50, 80), item_rect)
            # Item text
            color = _UI_ACCENT if i == self.selected_index else _UI_TEXT
            text_surf = font.render(self.items[i], True, color)
            surface.blit(text_surf, (item_rect.x + 5, item_rect.y + 3))
        surface.set_clip(clip)


# ============================================================
# SECTION 10: CAEngine
# ============================================================

class CAEngine:
    """Core Cellular Automata simulation engine."""

    def __init__(self, rows=100, cols=100, cell_size=6, rule="B3/S23"):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.max_state = 2
        self.generation = 0
        self.grid = np.zeros((rows, cols), dtype=np.int32)
        self.initial_grid = self.grid.copy()
        self.rule_string = rule
        self.birth_lookup = np.zeros(9, dtype=np.bool_)
        self.survive_lookup = np.zeros(9, dtype=np.bool_)
        self._parse_rule(rule)
        self.backend = BackendManager("Auto")
        self.undo_stack = UndoStack()

        # Visual settings
        self.current_theme = "dark"
        self.current_palette_name = "Standard"
        self.visual_mode = "Standard"
        self.wrap_mode = True
        self.symmetry_mode = "None"
        self.show_grid_lines = True
        self.trail_enabled = False
        self.trail_length = 15
        self.trail_grid = None
        self.glow_enabled = False
        self.vignette_enabled = False
        self.birth_death_enabled = False

        # Trackers
        self._age_tracker = AgeTracker(rows, cols)
        self._heatmap_tracker = HeatmapTracker(rows, cols)
        self._vignette = VignetteEffect(0.3)
        self._glow = GlowEffect(2, 0.5)
        self._outline_renderer = OutlineRenderer()
        self._neighbor_visualizer = NeighborCountVisualizer()
        self._birth_death_tracker = BirthDeathTracker(rows, cols)

        self._palette_manager = PaletteManager()

    def _parse_rule(self, rule: str):
        """Parse a B/S rule string into birth/survive lookup arrays."""
        rule = rule.strip().upper()
        self.birth_lookup.fill(False)
        self.survive_lookup.fill(False)
        m = re.match(r'B(\d*)/?S(\d*)', rule)
        if m:
            for c in m.group(1):
                if c.isdigit() and int(c) < 9:
                    self.birth_lookup[int(c)] = True
            for c in m.group(2):
                if c.isdigit() and int(c) < 9:
                    self.survive_lookup[int(c)] = True
        self.rule_string = rule

    def get_population(self) -> int:
        return int(np.sum(self.grid > 0))

    def push_undo(self):
        self.undo_stack.push(self.grid)

    def undo(self):
        s = self.undo_stack.undo(self.grid)
        if s is not None:
            self.grid = s

    def redo(self):
        s = self.undo_stack.redo(self.grid)
        if s is not None:
            self.grid = s

    def step(self):
        """Evolve one generation and update trackers."""
        prev_grid = self.grid.copy()
        self.grid = self.backend.evolve(self.grid, self.birth_lookup, self.survive_lookup,
                                        wrap=self.wrap_mode)
        self.grid = np.clip(self.grid, 0, self.max_state - 1)
        self.generation += 1

        # Update trackers
        self._age_tracker.update(self.grid, prev_grid)
        self._heatmap_tracker.update(self.grid, prev_grid)
        self._birth_death_tracker.update(self.grid, prev_grid)

        # Update trail grid
        if self.trail_enabled and self.trail_grid is not None:
            alive = self.grid > 0
            self.trail_grid[alive] = self.trail_length
            self.trail_grid = np.maximum(self.trail_grid - 1, 0)

    def clear(self):
        self.grid.fill(0)
        self.generation = 0
        self._reset_trackers()

    def reset(self):
        self.grid = self.initial_grid.copy()
        self.generation = 0
        self._reset_trackers()

    def randomize(self, density=0.3):
        self.grid = (np.random.random((self.rows, self.cols)) < density).astype(np.int32)
        self.generation = 0
        self._reset_trackers()

    def set_cell(self, r: int, c: int, v: int):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r, c] = v

    def set_visual_mode(self, mode: str):
        self.visual_mode = mode
        self._reset_trackers()

    def set_palette(self, name: str):
        self.current_palette_name = name

    def set_trails(self, v: bool):
        self.trail_enabled = v
        if v:
            self.trail_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        else:
            self.trail_grid = None

    def set_glow(self, v: bool):
        self.glow_enabled = v

    def set_vignette(self, v: bool):
        self.vignette_enabled = v

    def set_birth_effect(self, v: bool):
        self.birth_death_enabled = bool(v)

    def set_death_effect(self, v: bool):
        self.birth_death_enabled = self.birth_death_enabled or bool(v)

    def set_grid_lines(self, v: bool):
        self.show_grid_lines = v

    def set_wrap(self, v: bool):
        self.wrap_mode = v

    def set_symmetry(self, mode: str):
        self.symmetry_mode = mode

    def set_max_states(self, n: int):
        self.max_state = n
        self.grid = np.clip(self.grid, 0, n - 1)

    def set_rule(self, rule_str: str):
        rule = rule_str.split()[0] if ' ' in rule_str else rule_str
        self._parse_rule(rule)

    def resize(self, rows: int, cols: int):
        old = self.grid.copy()
        self.rows, self.cols = rows, cols
        self.grid = np.zeros((rows, cols), dtype=np.int32)
        self.initial_grid = np.zeros((rows, cols), dtype=np.int32)
        mr = min(old.shape[0], rows)
        mc = min(old.shape[1], cols)
        self.grid[:mr, :mc] = old[:mr, :mc]
        self._resize_trackers(rows, cols)
        self.generation = 0

    def get_state(self) -> dict:
        return {
            "grid": self.grid.tolist(),
            "rows": self.rows,
            "cols": self.cols,
            "rule": self.rule_string,
            "gen": self.generation,
            "wrap": self.wrap_mode,
        }

    def set_state(self, state: dict):
        self.grid = np.array(state["grid"], dtype=np.int32)
        self.rows, self.cols = self.grid.shape
        self.rule_string = state.get("rule", "B3/S23")
        self._parse_rule(self.rule_string)
        self.generation = state.get("gen", 0)
        self.wrap_mode = state.get("wrap", True)
        self._resize_trackers(self.rows, self.cols)

    def _reset_trackers(self):
        self._age_tracker.reset()
        self._heatmap_tracker.reset()
        self._birth_death_tracker.reset()
        if self.trail_grid is not None:
            self.trail_grid.fill(0)
        self._vignette._mask = None
        self._vignette._shape = None

    def _resize_trackers(self, rows: int, cols: int):
        self._age_tracker.resize(rows, cols)
        self._heatmap_tracker.resize(rows, cols)
        self._birth_death_tracker.resize(rows, cols)
        self._vignette._mask = None
        self._vignette._shape = None
        if self.trail_grid is not None:
            new_trail = np.zeros((rows, cols), dtype=np.int32)
            mr = min(self.trail_grid.shape[0], rows)
            mc = min(self.trail_grid.shape[1], cols)
            new_trail[:mr, :mc] = self.trail_grid[:mr, :mc]
            self.trail_grid = new_trail

    def render(self) -> np.ndarray:
        """Render the grid to an RGB numpy array (rows, cols, 3) uint8."""
        rows, cols = self.rows, self.cols
        bg_color = get_background_color(self.current_theme)

        # Build the base render grid
        if self.trail_enabled and self.trail_grid is not None:
            render_grid = np.where(self.trail_grid > 0, self.grid + self.trail_grid, self.grid)
        else:
            render_grid = self.grid

        # Render based on visual mode
        if self.visual_mode == "Standard":
            rgb = self._render_standard(render_grid, bg_color)
        elif self.visual_mode == "Age":
            rgb = self._render_age(render_grid, bg_color)
        elif self.visual_mode == "Heatmap":
            rgb = self._render_heatmap(bg_color)
        elif self.visual_mode == "Outline":
            rgb = self._render_outline(bg_color)
        elif self.visual_mode == "Neighbor Count":
            rgb = self._render_neighbor_count()
        elif self.visual_mode == "Gradient":
            rgb = self._render_standard(render_grid, bg_color)
        else:
            rgb = self._render_standard(render_grid, bg_color)

        # Apply birth/death flash overlay
        if self.birth_death_enabled:
            overlay = self._birth_death_tracker.get_overlay()
            rgb = np.clip(rgb.astype(np.float64) + overlay, 0, 255).astype(np.uint8)

        # Apply glow
        if self.glow_enabled:
            rgb = self._glow.apply(rgb, self.grid)

        # Apply vignette
        if self.vignette_enabled:
            rgb = self._vignette.apply(rgb)

        return rgb

    def _render_standard(self, render_grid: np.ndarray,
                         bg_color: Tuple[int, int, int]) -> np.ndarray:
        """Render using palette LUT."""
        palette = self._palette_manager.get_palette(self.current_palette_name)
        if palette is None:
            palette = BUILTIN_PALETTES.get("Standard", BUILTIN_PALETTES["Standard"])

        max_val = max(int(np.max(render_grid)) + 1, self.max_state)
        lut = palette.to_lut(max_val, interpolate=True)
        bg = palette.background_color

        # Start with background color
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        rgb[:, :] = [bg.r, bg.g, bg.b]

        # Map grid values to colors
        alive = render_grid > 0
        if np.any(alive):
            vals = np.clip(render_grid[alive], 0, lut.shape[0] - 1)
            rgb[alive] = lut[vals]

        return rgb

    def _render_age(self, render_grid: np.ndarray,
                    bg_color: Tuple[int, int, int]) -> np.ndarray:
        """Render using age-based coloring."""
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        rgb[:, :] = bg_color

        max_age = max(int(np.max(self._age_tracker.ages)) + 1, 2)
        lut = self._age_tracker.get_age_color_lut(min(max_age, 512))

        alive = self.grid > 0
        if np.any(alive):
            ages = np.clip(self._age_tracker.ages[alive], 0, lut.shape[0] - 1)
            rgb[alive] = lut[ages]

        return rgb

    def _render_heatmap(self, bg_color: Tuple[int, int, int]) -> np.ndarray:
        """Render using heatmap tracker."""
        return self._heatmap_tracker.get_colors(bg_color)

    def _render_outline(self, bg_color: Tuple[int, int, int]) -> np.ndarray:
        """Render with outlines."""
        return self._outline_renderer.render(self.grid, bg_color)

    def _render_neighbor_count(self) -> np.ndarray:
        """Render neighbor count visualization."""
        return self._neighbor_visualizer.render(self.grid)

# ============================================================
# PART 2: Main Application
# ============================================================

import pygame.gfxdraw

# Colors
BG_COLOR = (26, 26, 46)
PANEL_BG = (26, 26, 46)
CANVAS_BG = (16, 16, 28)
ACCENT = (0, 212, 168)
ACCENT_DIM = (0, 140, 110)
TEXT_COLOR = (200, 200, 212)
TEXT_DIM = (120, 120, 150)
TEXT_BRIGHT = (240, 240, 255)
BUTTON_BG = (46, 46, 80)
BUTTON_HOVER = (62, 62, 104)
BUTTON_PRESSED = (80, 80, 138)
BORDER_COLOR = (46, 46, 72)
INPUT_BG = (34, 34, 58)
SCROLLBAR_BG = (26, 26, 46)
SCROLLBAR_HANDLE = (62, 62, 96)
GRID_LINE_COLOR = (40, 40, 64)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
RED = (220, 60, 60)
GREEN = (60, 200, 80)
YELLOW = (255, 220, 50)
DARK_OVERLAY = (0, 0, 0, 180)

WINDOW_W, WINDOW_H = 1400, 900

TAB_BAR_H = 36
STATUS_BAR_H = 24
CANVAS_PANEL_W = 950
CTRL_PANEL_X = 960
CTRL_PANEL_W = 440
CTRL_CONTENT_X = CTRL_PANEL_X + 10
CTRL_CONTENT_W = CTRL_PANEL_W - 20

DOWNLOAD_DIR = "/home/z/my-project/download"

# Famous elementary CA rules
FAMOUS_RULES = [
    ("Rule 30 (Chaos)", 30),
    ("Rule 90 (Sierpinski)", 90),
    ("Rule 110 (Turing)", 110),
    ("Rule 184 (Traffic)", 184),
    ("Rule 0 (Blank)", 0),
    ("Rule 1", 1),
    ("Rule 18", 18),
    ("Rule 22", 22),
    ("Rule 54", 54),
    ("Rule 60", 60),
    ("Rule 73", 73),
    ("Rule 105", 105),
    ("Rule 150", 150),
    ("Rule 182", 182),
    ("Rule 250", 250),
    ("Rule 255 (Fill)", 255),
]

# Preset 2D rules for explorer
PRESET_2D_RULES = [
    "B3/S23 - Life",
    "B36/S23 - HighLife",
    "B3678/S34678 - Day&Night",
    "B3/S012345678 - Maze",
    "B3/S12345 - Maze2",
    "B368/S245 - Morley",
    "B2/S - Seeds",
    "B3/S234 - Diamoeba",
    "B378/S235678 - Replicator",
    "B36/S125 - 2x2",
    "B3678/S34678 - Day&Night",
    "B45678/S2345 - Walled Cities",
    "B3/S1234 - Coral",
    "B1357/S1357 - Replicator",
    "B1/S1 - Gnarl",
    "Custom",
]

# Preset patterns for explorer
PRESET_PATTERNS = [
    "Glider",
    "Blinker",
    "Toad",
    "Beacon",
    "Pulsar",
    "Pentadecathlon",
    "Lightweight Spaceship",
    "R-pentomino",
    "Diehard",
    "Acorn",
    "Gosper Glider Gun",
    "Block",
    "Beehive",
    "Loaf",
    "Boat",
    "Random 5%",
    "Random 15%",
    "Random 30%",
    "Random 50%",
]

# Pattern data (relative coordinates for each pattern)
PATTERN_DATA = {
    "Glider": [(0,1),(1,2),(2,0),(2,1),(2,2)],
    "Blinker": [(0,0),(0,1),(0,2)],
    "Toad": [(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)],
    "Beacon": [(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)],
    "Pulsar": [
        (0,2),(0,3),(0,4),(0,8),(0,9),(0,10),
        (2,0),(2,5),(2,7),(2,12),
        (3,0),(3,5),(3,7),(3,12),
        (4,0),(4,5),(4,7),(4,12),
        (5,2),(5,3),(5,4),(5,8),(5,9),(5,10),
        (7,2),(7,3),(7,4),(7,8),(7,9),(7,10),
        (8,0),(8,5),(8,7),(8,12),
        (9,0),(9,5),(9,7),(9,12),
        (10,0),(10,5),(10,7),(10,12),
        (12,2),(12,3),(12,4),(12,8),(12,9),(12,10),
    ],
    "Pentadecathlon": [
        (0,1),(1,1),(2,0),(2,2),(3,1),(4,1),(5,1),(6,1),(7,0),(7,2),(8,1),(9,1),
    ],
    "Lightweight Spaceship": [(0,1),(0,4),(1,0),(2,0),(2,4),(3,0),(3,1),(3,2),(3,3)],
    "R-pentomino": [(0,1),(0,2),(1,0),(1,1),(2,1)],
    "Diehard": [(0,6),(1,0),(1,1),(2,1),(2,5),(2,6),(2,7)],
    "Acorn": [(0,1),(1,3),(2,0),(2,1),(2,4),(2,5),(2,6)],
    "Gosper Glider Gun": [
        (0,24),
        (1,22),(1,24),
        (2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
        (3,11),(3,15),(3,20),(3,21),(3,34),(3,35),
        (4,0),(4,1),(4,10),(4,16),(4,20),(4,21),
        (5,0),(5,1),(5,10),(5,14),(5,16),(5,17),(5,22),(5,24),
        (6,10),(6,16),(6,24),
        (7,11),(7,15),
        (8,12),(8,13),
    ],
    "Block": [(0,0),(0,1),(1,0),(1,1)],
    "Beehive": [(0,1),(0,2),(1,0),(1,3),(2,1),(2,2)],
    "Loaf": [(0,1),(0,2),(1,0),(1,3),(2,1),(2,3),(3,2)],
    "Boat": [(0,0),(0,1),(1,0),(1,2),(2,1)],
}


# ============================================================
# SECTION 2: Text Input Dialog & Canvas Renderer
# ============================================================

class TextInputDialog:
    """A simple modal text input dialog for filename entry."""

    def __init__(self, prompt="Enter filename:", default_text="", width=400, height=130):
        self.prompt = prompt
        self.text = default_text
        self.width = width
        self.height = height
        self.active = False
        self.result = None
        self.cursor_visible = True
        self.cursor_blink = 0
        self.x = (WINDOW_W - width) // 2
        self.y = (WINDOW_H - height) // 2

    def show(self, prompt=None, default=""):
        if prompt is not None:
            self.prompt = prompt
        self.text = default
        self.active = True
        self.result = None
        self.cursor_blink = 0

    def handle_event(self, event):
        if not self.active:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.result = self.text
                self.active = False
                return True
            elif event.key == pygame.K_ESCAPE:
                self.result = None
                self.active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                return True
            elif event.key == pygame.K_v and (event.mod & pygame.KMOD_CTRL):
                try:
                    clip = pygame.scrap.get(pygame.SCRAP_TEXT)
                    if clip:
                        self.text += clip.decode('utf-8', errors='ignore').rstrip('\x00')
                except Exception:
                    pass
                return True
            elif event.unicode and event.unicode.isprintable():
                self.text += event.unicode
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if not (self.x <= mx <= self.x + self.width and self.y <= my <= self.y + self.height):
                self.result = None
                self.active = False
                return True
        return True  # Consume all events while active

    def draw(self, surface, fonts):
        if not self.active:
            return
        # Dim overlay
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        # Dialog box
        box_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, (36, 36, 66), box_rect, border_radius=8)
        pygame.draw.rect(surface, ACCENT, box_rect, 2, border_radius=8)

        # Prompt
        font = fonts.get('md', fonts.get('lg'))
        prompt_surf = font.render(self.prompt, True, TEXT_COLOR)
        surface.blit(prompt_surf, (self.x + 15, self.y + 15))

        # Input field
        input_rect = pygame.Rect(self.x + 15, self.y + 45, self.width - 30, 32)
        pygame.draw.rect(surface, INPUT_BG, input_rect, border_radius=4)
        pygame.draw.rect(surface, ACCENT, input_rect, 1, border_radius=4)

        # Text
        text_surf = font.render(self.text, True, TEXT_BRIGHT)
        surface.blit(text_surf, (input_rect.x + 8, input_rect.y + 7))

        # Cursor
        self.cursor_blink += 1
        if self.cursor_blink % 60 < 30:
            cx = input_rect.x + 8 + text_surf.get_width()
            pygame.draw.line(surface, TEXT_BRIGHT, (cx, input_rect.y + 6), (cx, input_rect.y + 26), 2)

        # Hint text
        hint_font = fonts.get('sm', font)
        hint = hint_font.render("Enter to confirm, Esc to cancel", True, TEXT_DIM)
        surface.blit(hint, (self.x + 15, self.y + self.height - 30))


class CanvasRenderer:
    """Renders a grid RGB numpy array to a pygame surface with zoom/pan."""

    def __init__(self, screen_area_rect):
        self.screen_rect = pygame.Rect(screen_area_rect)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._dragging_pan = False
        self._last_pan_pos = None
        self._min_zoom = 0.25
        self._max_zoom = 8.0

    def render(self, surface, rgb_array, grid_rows, grid_cols, cell_size,
               show_grid_lines, grid_color, hover_cell, symmetry_mode):
        """Render the grid RGB array to the pygame surface within screen_rect."""
        if rgb_array is None or rgb_array.size == 0:
            pygame.draw.rect(surface, CANVAS_BG, self.screen_rect)
            return

        eff_size = max(1, cell_size * self.zoom)
        img_w = int(grid_cols * eff_size)
        img_h = int(grid_rows * eff_size)

        if img_w <= 0 or img_h <= 0:
            pygame.draw.rect(surface, CANVAS_BG, self.screen_rect)
            return

        # Create small surface from RGB array
        small_surf = pygame.Surface((grid_cols, grid_rows))
        pygame.surfarray.blit_array(small_surf, rgb_array.transpose(1, 0, 2))

        # Scale up
        if eff_size == int(eff_size) and eff_size >= 1:
            scaled = pygame.transform.scale(small_surf, (img_w, img_h))
        else:
            scaled = pygame.transform.scale(small_surf, (max(1, img_w), max(1, img_h)))

        # Calculate blit position with pan
        blit_x = self.screen_rect.x + self.pan_x
        blit_y = self.screen_rect.y + self.pan_y

        # Clip to screen rect
        clip_rect = self.screen_rect.clip(pygame.Rect(blit_x, blit_y, img_w, img_h))
        if clip_rect.width > 0 and clip_rect.height > 0:
            # Fill canvas background
            pygame.draw.rect(surface, CANVAS_BG, self.screen_rect)
            # Blit the scaled surface
            src_x = int(clip_rect.x - blit_x)
            src_y = int(clip_rect.y - blit_y)
            src_rect = pygame.Rect(src_x, src_y, clip_rect.width, clip_rect.height)
            surface.blit(scaled, clip_rect.topleft, src_rect)
        else:
            pygame.draw.rect(surface, CANVAS_BG, self.screen_rect)

        # Draw grid lines
        if show_grid_lines and eff_size >= 6:
            self._draw_grid_lines(surface, grid_rows, grid_cols, eff_size,
                                  blit_x, blit_y, grid_color)

        # Draw hover highlight
        if hover_cell is not None:
            hr, hc = hover_cell
            hx = int(blit_x + hc * eff_size)
            hy = int(blit_y + hr * eff_size)
            hs = int(eff_size)
            hover_rect = pygame.Rect(hx, hy, hs, hs)
            if self.screen_rect.colliderect(hover_rect):
                hover_surf = pygame.Surface((hs, hs), pygame.SRCALPHA)
                hover_surf.fill((255, 255, 255, 60))
                surface.blit(hover_surf, hover_rect.topleft)
                pygame.draw.rect(surface, (255, 255, 255, 120), hover_rect, 1)

        # Draw symmetry lines
        if symmetry_mode and symmetry_mode != "None":
            self._draw_symmetry_lines(surface, grid_rows, grid_cols, eff_size,
                                      blit_x, blit_y, symmetry_mode)

        # Canvas border
        pygame.draw.rect(surface, BORDER_COLOR, self.screen_rect, 1)

    def _draw_grid_lines(self, surface, rows, cols, eff_size, bx, by, color):
        sr = self.screen_rect
        # Vertical lines
        for c in range(cols + 1):
            x = int(bx + c * eff_size)
            if sr.left <= x <= sr.right:
                pygame.draw.line(surface, color, (x, max(sr.top, int(by))),
                                 (x, min(sr.bottom, int(by + rows * eff_size))), 1)
        # Horizontal lines
        for r in range(rows + 1):
            y = int(by + r * eff_size)
            if sr.top <= y <= sr.bottom:
                pygame.draw.line(surface, color, (max(sr.left, int(bx)), y),
                                 (min(sr.right, int(bx + cols * eff_size)), y), 1)

    def _draw_symmetry_lines(self, surface, rows, cols, eff_size, bx, by, mode):
        sr = self.screen_rect
        color = (255, 100, 100, 150)
        if mode in ("Horizontal", "Both"):
            y = int(by + rows * eff_size / 2)
            if sr.top <= y <= sr.bottom:
                pygame.draw.line(surface, (255, 100, 100),
                                 (max(sr.left, int(bx)), y),
                                 (min(sr.right, int(bx + cols * eff_size)), y), 2)
        if mode in ("Vertical", "Both"):
            x = int(bx + cols * eff_size / 2)
            if sr.left <= x <= sr.right:
                pygame.draw.line(surface, (100, 100, 255),
                                 (x, max(sr.top, int(by))),
                                 (x, min(sr.bottom, int(by + rows * eff_size))), 2)
        if mode == "Rotational":
            cx = int(bx + cols * eff_size / 2)
            cy = int(by + rows * eff_size / 2)
            if sr.left <= cx <= sr.right:
                pygame.draw.line(surface, (255, 100, 100),
                                 (cx, max(sr.top, int(by))),
                                 (cx, min(sr.bottom, int(by + rows * eff_size))), 1)
            if sr.top <= cy <= sr.bottom:
                pygame.draw.line(surface, (100, 100, 255),
                                 (max(sr.left, int(bx)), cy),
                                 (min(sr.right, int(bx + cols * eff_size)), cy), 1)

    def screen_to_cell(self, mx, my, grid_rows, grid_cols, cell_size):
        """Convert screen coordinates to grid cell (row, col) or None."""
        eff_size = max(1, cell_size * self.zoom)
        lx = mx - self.screen_rect.x - self.pan_x
        ly = my - self.screen_rect.y - self.pan_y
        col = int(lx / eff_size)
        row = int(ly / eff_size)
        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            return (row, col)
        return None

    def handle_event(self, event, grid_rows=0, grid_cols=0, cell_size=6):
        """Handle zoom (wheel) and pan (middle mouse). Returns True if consumed."""
        if event.type == pygame.MOUSEWHEEL and self.screen_rect.collidepoint(pygame.mouse.get_pos()):
            mx, my = pygame.mouse.get_pos()
            # Get cell under mouse before zoom
            cell_before = self.screen_to_cell(mx, my, grid_rows, grid_cols, cell_size) if grid_rows else None

            factor = 1.15
            if event.y > 0:
                self.zoom = min(self.zoom * factor, self._max_zoom)
            elif event.y < 0:
                self.zoom = max(self.zoom / factor, self._min_zoom)

            # Adjust pan to zoom toward mouse position
            if cell_before is not None and grid_rows > 0:
                eff_size_new = max(1, cell_size * self.zoom)
                target_x = self.screen_rect.x + self.pan_x + cell_before[1] * eff_size_new + eff_size_new / 2
                target_y = self.screen_rect.y + self.pan_y + cell_before[0] * eff_size_new + eff_size_new / 2
                self.pan_x += mx - target_x
                self.pan_y += my - target_y
            return True

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            if self.screen_rect.collidepoint(event.pos):
                self._dragging_pan = True
                self._last_pan_pos = event.pos
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            self._dragging_pan = False
            self._last_pan_pos = None
            return True

        elif event.type == pygame.MOUSEMOTION and self._dragging_pan:
            if self._last_pan_pos:
                dx = event.pos[0] - self._last_pan_pos[0]
                dy = event.pos[1] - self._last_pan_pos[1]
                self.pan_x += dx
                self.pan_y += dy
                self._last_pan_pos = event.pos
            return True

        return False

    def reset_view(self, grid_rows, grid_cols, cell_size):
        """Reset zoom and pan to fit the grid in the screen rect."""
        if grid_rows <= 0 or grid_cols <= 0:
            return
        avail_w = self.screen_rect.width - 20
        avail_h = self.screen_rect.height - 20
        zoom_x = avail_w / (grid_cols * cell_size)
        zoom_y = avail_h / (grid_rows * cell_size)
        self.zoom = min(zoom_x, zoom_y, 4.0)
        self.zoom = max(self.zoom, self._min_zoom)
        # Center the grid
        total_w = grid_cols * cell_size * self.zoom
        total_h = grid_rows * cell_size * self.zoom
        self.pan_x = (self.screen_rect.width - total_w) / 2
        self.pan_y = (self.screen_rect.height - total_h) / 2


# ============================================================
# SECTION 3: Studio Tab (Main 2D CA)
# ============================================================

class StudioTab:
    """The main 2D Cellular Automata studio tab."""

    def __init__(self, screen, fonts, engine):
        self.screen = screen
        self.fonts = fonts
        self.engine = engine
        self.canvas_renderer = CanvasRenderer(pygame.Rect(0, TAB_BAR_H, CANVAS_PANEL_W, WINDOW_H - TAB_BAR_H - STATUS_BAR_H))

        # State
        self.is_playing = False
        self.sim_speed = 50
        self.steps_per_frame = 1
        self.last_step_time = 0
        self.population_history = []
        self.saved_states = []
        self.hover_cell = None
        self._drawing = False
        self._draw_value = 1
        self._last_draw_cell = None
        self._undo_pushed = False
        self.panel_scroll = 0
        self.max_scroll = 0

        # Build UI
        self.widgets = []
        self.all_widgets = []  # flat list for event routing
        self._build_controls()

        # Initial view reset
        try:
            cs = self.engine.cell_size if hasattr(self.engine, 'cell_size') else 6
            self.canvas_renderer.reset_view(self.engine.rows, self.engine.cols, cs)
        except Exception:
            pass

    def _build_controls(self):
        """Create all the control widgets for the Studio tab."""
        px = CTRL_CONTENT_X
        pw = CTRL_CONTENT_W
        y = TAB_BAR_H + 6
        h = 26
        gap = 4
        sgap = 8
        hw = (pw - gap) // 2
        tw = (pw - 2 * gap) // 3
        qw = (pw - 3 * gap) // 4

        # ---- PLAYBACK ----
        self.play_btn = Button(pygame.Rect(px, y, tw, h), "Play", self._on_play_toggle)
        self.step_btn = Button(pygame.Rect(px + tw + gap, y, tw, h), "Step", self._on_step)
        self.reset_btn = Button(pygame.Rect(px + 2 * (tw + gap), y, tw, h), "Reset", self._on_reset)
        self._add(self.play_btn, self.step_btn, self.reset_btn)
        y += h + gap

        # ---- STATS ----
        self.gen_label = Label(pygame.Rect(px, y, pw, 18), "Gen: 0  |  Pop: 0  |  FPS: 60")
        self._add(self.gen_label)
        y += 20 + gap

        # ---- SPEED ----
        self.speed_section = Label(pygame.Rect(px, y, pw, 16), "-- SPEED --", color=ACCENT)
        self._add(self.speed_section)
        y += 18

        self.speed_slider = Slider(pygame.Rect(px, y, pw - 70, h), 1, 500, 50,
                                   self._on_speed_change, label="")
        self.speed_val_label = Label(pygame.Rect(px + pw - 65, y, 65, h), "50 ms")
        self._add(self.speed_slider, self.speed_val_label)
        y += h + gap

        self.spf_label = Label(pygame.Rect(px, y, 90, h), "Steps/Frame:")
        self.spf_spin = SpinBox(pygame.Rect(px + 95, y, 60, h), 1, 50, 1, step=1,
                                callback=self._on_spf_change)
        self._add(self.spf_label, self.spf_spin)
        y += h + sgap

        # ---- GRID ----
        self.grid_section = Label(pygame.Rect(px, y, pw, 16), "-- GRID --", color=ACCENT)
        self._add(self.grid_section)
        y += 18

        self.rows_label = Label(pygame.Rect(px, y, 42, h), "Rows:")
        self.rows_spin = SpinBox(pygame.Rect(px + 44, y, 60, h), 10, 800, 100, step=1)
        self.cols_label = Label(pygame.Rect(px + 110, y, 42, h), "Cols:")
        self.cols_spin = SpinBox(pygame.Rect(px + 154, y, 60, h), 10, 800, 100, step=1)
        self.cs_label = Label(pygame.Rect(px + 220, y, 26, h), "CS:")
        self.cs_spin = SpinBox(pygame.Rect(px + 248, y, 50, h), 2, 30, 6, step=1)
        self.apply_grid_btn = Button(pygame.Rect(px + pw - 80, y, 80, h), "Apply", self._on_apply_grid)
        self._add(self.rows_label, self.rows_spin, self.cols_label, self.cols_spin,
                  self.cs_label, self.cs_spin, self.apply_grid_btn)
        y += h + sgap

        # ---- PATTERNS ----
        self.pat_section = Label(pygame.Rect(px, y, pw, 16), "-- PATTERNS --", color=ACCENT)
        self._add(self.pat_section)
        y += 18

        pattern_names = list(PATTERN_DATA.keys())
        self.pattern_combo = ComboBox(pygame.Rect(px, y, pw - 80, h), pattern_names, 0)
        self.inject_btn = Button(pygame.Rect(px + pw - 75, y, 75, h), "Inject", self._on_inject_pattern)
        self._add(self.pattern_combo, self.inject_btn)
        y += h + sgap

        # ---- ACTIONS ----
        self.act_section = Label(pygame.Rect(px, y, pw, 16), "-- ACTIONS --", color=ACCENT)
        self._add(self.act_section)
        y += 18

        self.random_btn = Button(pygame.Rect(px, y, hw, h), "Random Fill", self._on_random)
        self.density_label = Label(pygame.Rect(px + hw + gap, y, 55, h), "Dens:")
        self.density_spin = SpinBox(pygame.Rect(px + hw + gap + 58, y, 60, h), 1, 99, 30, step=1)
        self.density_pct = Label(pygame.Rect(px + hw + gap + 122, y, 20, h), "%")
        self._add(self.random_btn, self.density_label, self.density_spin, self.density_pct)
        y += h + gap

        self.save_state_btn = Button(pygame.Rect(px, y, hw, h), "Save State", self._on_save_state)
        self.restore_state_btn = Button(pygame.Rect(px + hw + gap, y, hw, h), "Restore", self._on_restore_state)
        self._add(self.save_state_btn, self.restore_state_btn)
        y += h + gap

        self.clear_btn = Button(pygame.Rect(px, y, hw, h), "Clear", self._on_clear)
        self.reset_all_btn = Button(pygame.Rect(px + hw + gap, y, hw, h), "Reset All", self._on_reset)
        self._add(self.clear_btn, self.reset_all_btn)
        y += h + gap

        self.save_file_btn = Button(pygame.Rect(px, y, qw, h), "Save", self._on_save_file)
        self.load_file_btn = Button(pygame.Rect(px + qw + gap, y, qw, h), "Load", self._on_load_file)
        self.export_btn = Button(pygame.Rect(px + 2 * (qw + gap), y, qw, h), "Export", self._on_export_png)
        self.fit_btn = Button(pygame.Rect(px + 3 * (qw + gap), y, qw, h), "Fit View", self._on_fit_view)
        self._add(self.save_file_btn, self.load_file_btn, self.export_btn, self.fit_btn)
        y += h + sgap

        # ---- VISUAL ----
        self.vis_section = Label(pygame.Rect(px, y, pw, 16), "-- VISUAL --", color=ACCENT)
        self._add(self.vis_section)
        y += 18

        self.vis_label = Label(pygame.Rect(px, y, 50, h), "Mode:")
        self.vis_combo = ComboBox(pygame.Rect(px + 54, y, pw - 54, h),
                                 ["Standard", "Age", "Heatmap", "Outline", "Neighbor Count"], 0,
                                 callback=self._on_visual_mode_change)
        self._add(self.vis_label, self.vis_combo)
        y += h + gap

        self.pal_label = Label(pygame.Rect(px, y, 50, h), "Palette:")
        self.pal_combo = ComboBox(pygame.Rect(px + 54, y, pw - 54, h),
                                 self._get_palette_names(), 0,
                                 callback=self._on_palette_change)
        self._add(self.pal_label, self.pal_combo)
        y += h + sgap

        # ---- EFFECTS ----
        self.eff_section = Label(pygame.Rect(px, y, pw, 16), "-- EFFECTS --", color=ACCENT)
        self._add(self.eff_section)
        y += 18

        ew = (pw - 2 * gap) // 3
        self.trails_cb = CheckBox(pygame.Rect(px, y, ew, h), "Trails", False, self._on_effect_change)
        self.glow_cb = CheckBox(pygame.Rect(px + ew + gap, y, ew, h), "Glow", False, self._on_effect_change)
        self.grid_cb = CheckBox(pygame.Rect(px + 2 * (ew + gap), y, ew, h), "Grid Lines", True, self._on_effect_change)
        self._add(self.trails_cb, self.glow_cb, self.grid_cb)
        y += h + gap

        self.vignette_cb = CheckBox(pygame.Rect(px, y, ew, h), "Vignette", False, self._on_effect_change)
        self.birth_cb = CheckBox(pygame.Rect(px + ew + gap, y, ew, h), "Birth FX", False, self._on_effect_change)
        self.death_cb = CheckBox(pygame.Rect(px + 2 * (ew + gap), y, ew, h), "Death FX", False, self._on_effect_change)
        self._add(self.vignette_cb, self.birth_cb, self.death_cb)
        y += h + gap

        self.wrap_cb = CheckBox(pygame.Rect(px, y, ew, h), "Wrap Edges", True, self._on_effect_change)
        self._add(self.wrap_cb)
        y += h + sgap

        # ---- RULES ----
        self.rule_section = Label(pygame.Rect(px, y, pw, 16), "-- RULES --", color=ACCENT)
        self._add(self.rule_section)
        y += 18

        self.rule_combo = ComboBox(pygame.Rect(px, y, pw - 100, h),
                                   ["B3/S23 - Life", "B36/S23 - HighLife", "B3678/S34678 - Day&Night",
                                    "B3/S012345678 - Maze", "B3/S12345 - Maze2", "B368/S245 - Morley",
                                    "B2/S - Seeds", "B3/S234 - Diamoeba",
                                    "B378/S235678 - Replicator", "B36/S125 - 2x2",
                                    "B45678/S2345 - Walled Cities", "B3/S1234 - Coral",
                                    "B1357/S1357 - Replicator", "B1/S1 - Gnarl"],
                                   0, callback=self._on_rule_change)
        self.max_states_label = Label(pygame.Rect(px + pw - 95, y, 32, h), "Max:")
        self.max_states_spin = SpinBox(pygame.Rect(px + pw - 60, y, 60, h), 2, 256, 2, step=1,
                                       callback=self._on_max_states_change)
        self._add(self.rule_combo, self.max_states_label, self.max_states_spin)
        y += h + sgap

        # ---- SYMMETRY ----
        self.sym_section = Label(pygame.Rect(px, y, pw, 16), "-- SYMMETRY --", color=ACCENT)
        self._add(self.sym_section)
        y += 18

        self.sym_combo = ComboBox(pygame.Rect(px, y, pw, h),
                                  ["None", "Horizontal", "Vertical", "Both", "Rotational"], 0,
                                  callback=self._on_symmetry_change)
        self._add(self.sym_combo)
        y += h + 10

        self.max_scroll = max(0, y - TAB_BAR_H - (WINDOW_H - TAB_BAR_H - STATUS_BAR_H) + 20)

    def _add(self, *widgets):
        for w in widgets:
            self.widgets.append(w)
            self.all_widgets.append(w)

    def _get_palette_names(self):
        try:
            if hasattr(PaletteManager, 'get_names'):
                return PaletteManager.get_names()
            return ["Default", "Ocean", "Fire", "Neon", "Grayscale", "Earth", "Pastel"]
        except NameError:
            return ["Default", "Ocean", "Fire", "Neon", "Grayscale", "Earth", "Pastel"]

    # ---- Callbacks ----

    def _on_play_toggle(self):
        self.is_playing = not self.is_playing
        self.play_btn.text = "Pause" if self.is_playing else "Play"
        if self.is_playing:
            self.last_step_time = pygame.time.get_ticks()

    def _on_step(self):
        self._do_step()

    def _on_reset(self):
        self.is_playing = False
        self.play_btn.text = "Play"
        self.population_history.clear()
        try:
            self.engine.reset()
        except Exception:
            pass

    def _on_clear(self):
        try:
            self.engine.push_undo()
            self.engine.clear()
        except Exception:
            pass

    def _on_random(self):
        density = self.density_spin.value / 100.0
        try:
            self.engine.push_undo()
            self.engine.randomize(density)
        except Exception:
            pass

    def _on_save_state(self):
        try:
            state = self.engine.get_state()
            self.saved_states.append(state)
        except Exception:
            pass

    def _on_restore_state(self):
        if self.saved_states:
            try:
                self.engine.push_undo()
                self.engine.set_state(self.saved_states[-1])
            except Exception:
                pass

    def _on_save_file(self):
        try:
            path = os.path.join(DOWNLOAD_DIR, "ca_save.json")
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            state = self.engine.get_state()
            with open(path, 'w') as f:
                json.dump(state, f)
            print(f"Saved to {path}")
        except Exception as e:
            print(f"Save failed: {e}")

    def _on_load_file(self):
        try:
            path = os.path.join(DOWNLOAD_DIR, "ca_save.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                self.engine.push_undo()
                self.engine.set_state(state)
                print(f"Loaded from {path}")
            else:
                print(f"No save file found at {path}")
        except Exception as e:
            print(f"Load failed: {e}")

    def _on_export_png(self):
        try:
            path = os.path.join(DOWNLOAD_DIR, "ca_export.png")
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            rgb = self.engine.render()
            if rgb is not None:
                surf = pygame.Surface((rgb.shape[1], rgb.shape[0]))
                pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
                pygame.image.save(surf, path)
                print(f"Exported to {path}")
        except Exception as e:
            print(f"Export failed: {e}")

    def _on_fit_view(self):
        try:
            cs = self.cs_spin.value
            self.canvas_renderer.reset_view(self.engine.rows, self.engine.cols, cs)
        except Exception:
            pass

    def _on_inject_pattern(self):
        try:
            name = self.pattern_combo.selected_text
            coords = PATTERN_DATA.get(name, [])
            if coords:
                self.engine.push_undo()
                cr, cc = self.engine.rows // 2, self.engine.cols // 2
                for dr, dc in coords:
                    r, c = cr + dr, cc + dc
                    if 0 <= r < self.engine.rows and 0 <= c < self.engine.cols:
                        self.engine.set_cell(r, c, 1)
        except Exception as e:
            print(f"Inject failed: {e}")

    def _on_apply_grid(self):
        try:
            new_rows = self.rows_spin.value
            new_cols = self.cols_spin.value
            new_cs = self.cs_spin.value
            self.engine.resize(new_rows, new_cols)
            if hasattr(self.engine, 'cell_size'):
                self.engine.cell_size = new_cs
            self.canvas_renderer.reset_view(new_rows, new_cols, new_cs)
        except Exception as e:
            print(f"Resize failed: {e}")

    def _on_speed_change(self, val):
        self.sim_speed = int(val)
        self.speed_val_label.text = f"{self.sim_speed} ms"

    def _on_spf_change(self, val):
        self.steps_per_frame = max(1, int(val))

    def _on_visual_mode_change(self, idx):
        try:
            mode = self.vis_combo.selected_text
            self.engine.set_visual_mode(mode)
        except Exception:
            pass

    def _on_palette_change(self, idx):
        try:
            name = self.pal_combo.selected_text
            self.engine.set_palette(name)
        except Exception:
            pass

    def _on_effect_change(self, val=None):
        try:
            self.engine.set_trails(self.trails_cb.checked)
            self.engine.set_glow(self.glow_cb.checked)
            self.engine.set_vignette(self.vignette_cb.checked)
            self.engine.set_birth_effect(self.birth_cb.checked)
            self.engine.set_death_effect(self.death_cb.checked)
            self.engine.set_grid_lines(self.grid_cb.checked)
            self.engine.set_wrap(self.wrap_cb.checked)
        except Exception:
            pass

    def _on_rule_change(self, idx):
        try:
            text = self.rule_combo.selected_text
            self.engine.set_rule(text)
        except Exception:
            pass

    def _on_max_states_change(self, val):
        try:
            self.engine.set_max_states(int(val))
        except Exception:
            pass

    def _on_symmetry_change(self, idx):
        try:
            mode = self.sym_combo.selected_text
            self.engine.set_symmetry(mode)
        except Exception:
            pass

    def _do_step(self):
        try:
            self.engine.push_undo()
            self.engine.step()
            pop = self.engine.get_population()
            self.population_history.append(pop)
            if len(self.population_history) > 500:
                self.population_history.pop(0)
        except Exception:
            pass

    # ---- Event Handling ----

    def handle_event(self, event):
        # Keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            mods = pygame.mod.get_mods()
            if mods & pygame.KMOD_CTRL:
                if event.key == pygame.K_z:
                    try:
                        self.engine.undo()
                    except Exception:
                        pass
                    return True
                elif event.key == pygame.K_y:
                    try:
                        self.engine.redo()
                    except Exception:
                        pass
                    return True
            else:
                if event.key == pygame.K_SPACE:
                    self._on_play_toggle()
                    return True
                elif event.key == pygame.K_RIGHT:
                    self._on_step()
                    return True
                elif event.key == pygame.K_c:
                    self._on_clear()
                    return True
                elif event.key == pygame.K_r:
                    self._on_random()
                    return True
                elif event.key == pygame.K_f:
                    self._on_fit_view()
                    return True

        # Check if mouse is over panel
        mx, my = pygame.mouse.get_pos() if event.type in (
            pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION,
            pygame.MOUSEWHEEL) else (0, 0)

        # If any combobox dropdown is open, route all events to widgets first
        any_dropdown_open = False
        for w in self.all_widgets:
            if isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open:
                any_dropdown_open = True
                break

        if any_dropdown_open or (mx >= CTRL_PANEL_X and event.type in (
                pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.MOUSEWHEEL)):
            for w in self.all_widgets:
                if w.handle_event(event):
                    return True
            if event.type == pygame.MOUSEWHEEL and mx >= CTRL_PANEL_X:
                return True  # Consume wheel over panel

        # Canvas events
        if self.canvas_renderer.screen_rect.collidepoint(mx, my) or self.canvas_renderer._dragging_pan:
            if self.canvas_renderer.handle_event(event, self.engine.rows, self.engine.cols,
                                                 self.cs_spin.value):
                return True

        # Drawing on canvas
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (1, 3) and self.canvas_renderer.screen_rect.collidepoint(event.pos):
                cell = self.canvas_renderer.screen_to_cell(event.pos[0], event.pos[1],
                                                            self.engine.rows, self.engine.cols,
                                                            self.cs_spin.value)
                if cell:
                    self._drawing = True
                    self._draw_value = 1 if event.button == 1 else 0
                    self._undo_pushed = False
                    if not self._undo_pushed:
                        try:
                            self.engine.push_undo()
                        except Exception:
                            pass
                        self._undo_pushed = True
                    self.engine.set_cell(cell[0], cell[1], self._draw_value)
                    self._last_draw_cell = cell
                    return True

        elif event.type == pygame.MOUSEMOTION and self._drawing:
            cell = self.canvas_renderer.screen_to_cell(event.pos[0], event.pos[1],
                                                        self.engine.rows, self.engine.cols,
                                                        self.cs_spin.value)
            if cell and cell != self._last_draw_cell:
                self.engine.set_cell(cell[0], cell[1], self._draw_value)
                # Apply symmetry
                sym = self.sym_combo.selected_text if self.sym_combo else "None"
                if sym in ("Horizontal", "Both"):
                    sr = cell[0]
                    sc = self.engine.cols - 1 - cell[1]
                    self.engine.set_cell(sr, sc, self._draw_value)
                if sym in ("Vertical", "Both"):
                    sr = self.engine.rows - 1 - cell[0]
                    sc = cell[1]
                    self.engine.set_cell(sr, sc, self._draw_value)
                if sym == "Rotational":
                    sr = self.engine.rows - 1 - cell[0]
                    sc = self.engine.cols - 1 - cell[1]
                    self.engine.set_cell(sr, sc, self._draw_value)
                self._last_draw_cell = cell
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3):
            self._drawing = False
            self._last_draw_cell = None

        # Update hover
        if event.type == pygame.MOUSEMOTION:
            if self.canvas_renderer.screen_rect.collidepoint(event.pos):
                self.hover_cell = self.canvas_renderer.screen_to_cell(
                    event.pos[0], event.pos[1], self.engine.rows, self.engine.cols, self.cs_spin.value)
            else:
                self.hover_cell = None

        return False

    # ---- Update ----

    def update(self):
        now = pygame.time.get_ticks()
        if self.is_playing and now - self.last_step_time >= self.sim_speed:
            for _ in range(self.steps_per_frame):
                self._do_step()
            self.last_step_time = now

    # ---- Draw ----

    def draw(self, fps):
        try:
            cs = self.cs_spin.value
        except Exception:
            cs = 6
        try:
            rgb = self.engine.render()
        except Exception:
            rgb = None

        sym = self.sym_combo.selected_text if self.sym_combo else "None"
        show_grid = self.grid_cb.checked

        self.canvas_renderer.render(
            self.screen, rgb, self.engine.rows, self.engine.cols, cs,
            show_grid, GRID_LINE_COLOR, self.hover_cell, sym
        )

        # Draw widgets
        panel_rect = pygame.Rect(CTRL_PANEL_X, TAB_BAR_H, CTRL_PANEL_W,
                                 WINDOW_H - TAB_BAR_H - STATUS_BAR_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, BORDER_COLOR,
                         (CTRL_PANEL_X, TAB_BAR_H), (CTRL_PANEL_X, WINDOW_H - STATUS_BAR_H), 1)

        # Clip widgets to panel
        clip = self.screen.get_clip()
        self.screen.set_clip(panel_rect)

        for w in self.all_widgets:
            w.draw(self.screen)

        self.screen.set_clip(clip)

        # Draw any open combobox dropdowns on top
        for w in self.all_widgets:
            if isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open:
                w.draw_dropdown(self.screen)

        # Update stats label
        try:
            gen = self.engine.generation if hasattr(self.engine, 'generation') else 0
            pop = self.engine.get_population() if hasattr(self.engine, 'get_population') else 0
            self.gen_label.text = f"Gen: {gen}  |  Pop: {pop}  |  FPS: {fps:.0f}"
        except Exception:
            pass

        return self.engine.get_population() if hasattr(self.engine, 'get_population') else 0


# ============================================================
# SECTION 4: 1D Elementary CA Tab
# ============================================================

class ElementaryCA:
    """A 1D elementary cellular automaton engine."""

    def __init__(self, width=201, rule=30, init_mode='center'):
        self.width = width
        self.rule = rule
        self.init_mode = init_mode
        self.state = np.zeros(width, dtype=np.int8)
        self.history = []
        self.generation = 0
        self.max_history = 800
        self.reset(rule=rule, width=width, init_mode=init_mode)

    def reset(self, rule=None, width=None, init_mode=None):
        if rule is not None:
            self.rule = max(0, min(255, int(rule)))
        if width is not None:
            self.width = max(3, int(width))
        if init_mode is not None:
            self.init_mode = init_mode
        self.state = np.zeros(self.width, dtype=np.int8)
        if self.init_mode == 'center':
            self.state[self.width // 2] = 1
        elif self.init_mode == 'random':
            self.state = np.random.randint(0, 2, self.width).astype(np.int8)
        elif self.init_mode == 'left':
            self.state[0] = 1
        elif self.init_mode == 'right':
            self.state[-1] = 1
        self.history = [self.state.copy()]
        self.generation = 0

    def step(self):
        new_state = np.zeros(self.width, dtype=np.int8)
        for i in range(self.width):
            left = self.state[(i - 1) % self.width]
            center = self.state[i]
            right = self.state[(i + 1) % self.width]
            idx = (left << 2) | (center << 1) | right
            new_state[i] = (self.rule >> idx) & 1
        self.state = new_state
        self.generation += 1
        self.history.append(new_state.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def step_n(self, n):
        for _ in range(n):
            self.step()

    def get_rule_bits(self):
        """Return list of 8 output bits (for patterns 111, 110, ..., 000)."""
        return [(self.rule >> i) & 1 for i in range(7, -1, -1)]

    def set_rule_bit(self, idx, value):
        """Set bit at index (0-7, where 0=pattern 111, 7=pattern 000)."""
        if value:
            self.rule |= (1 << (7 - idx))
        else:
            self.rule &= ~(1 << (7 - idx))

    def render_rgb(self, color_mode='teal', cell_size=2):
        """Render history as an RGB array (gen_count x width x 3)."""
        n = len(self.history)
        if n == 0:
            return np.zeros((1, self.width, 3), dtype=np.uint8)
        img = np.zeros((n, self.width, 3), dtype=np.uint8)
        for g in range(n):
            for x in range(self.width):
                if self.history[g][x]:
                    if color_mode == 'teal':
                        t = g / max(1, n - 1)
                        img[g, x] = (int(0 + 30 * t), int(212 - 100 * t), int(168 - 60 * t))
                    elif color_mode == 'rainbow':
                        hue = (g / max(1, n - 1)) * 360
                        img[g, x] = self._hsv_to_rgb(hue, 0.8, 1.0)
                    elif color_mode == 'bw':
                        img[g, x] = (255, 255, 255)
                else:
                    img[g, x] = (16, 16, 28)
        return img

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


class ElementaryCATab:
    """1D Elementary CA exploration tab."""

    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.ca = ElementaryCA(201, 30, 'center')
        self.is_playing = False
        self.sim_speed = 20
        self.last_step_time = 0
        self.cell_size = 2
        self.scroll_offset = 0
        self.auto_scroll = True
        self.widgets = []
        self.rule_viz_rect = pygame.Rect(0, 0, 0, 0)  # for rule visualizer clicks
        self._build_controls()

    def _build_controls(self):
        px = CTRL_CONTENT_X
        pw = CTRL_CONTENT_W
        y = TAB_BAR_H + 6
        h = 26
        gap = 4
        sgap = 8
        hw = (pw - gap) // 2
        tw = (pw - 2 * gap) // 3
        fw = (pw - 3 * gap) // 4

        # Playback
        self.play_btn = Button(pygame.Rect(px, y, tw, h), "Play", self._on_play_toggle)
        self.step_btn = Button(pygame.Rect(px + tw + gap, y, tw, h), "Step", self._on_step)
        self.reset_btn = Button(pygame.Rect(px + 2 * (tw + gap), y, tw, h), "Reset", self._on_reset)
        self._add(self.play_btn, self.step_btn, self.reset_btn)
        y += h + gap

        # Skip
        self.skip_label = Label(pygame.Rect(px, y, 42, h), "Skip:")
        self.skip_spin = SpinBox(pygame.Rect(px + 44, y, 60, h), 1, 500, 10, step=1)
        self.skip_btn = Button(pygame.Rect(px + 110, y, 60, h), "Go", self._on_skip)
        self._add(self.skip_label, self.skip_spin, self.skip_btn)
        y += h + sgap

        # Rule
        self.rule_section = Label(pygame.Rect(px, y, pw, 16), "-- RULE --", color=ACCENT)
        self._add(self.rule_section)
        y += 18

        self.rule_label = Label(pygame.Rect(px, y, 42, h), "Rule:")
        self.rule_spin = SpinBox(pygame.Rect(px + 44, y, 65, h), 0, 255, 30, step=1,
                                 callback=self._on_rule_change)
        self.rule_combo = ComboBox(pygame.Rect(px + 115, y, pw - 115, h),
                                   [n for n, r in FAMOUS_RULES], 0,
                                   callback=self._on_rule_preset)
        self._add(self.rule_label, self.rule_spin, self.rule_combo)
        y += h + gap

        # Rule Visualizer area
        self.rule_viz_y = y
        y += 50
        self.rule_desc = Label(pygame.Rect(px, y, pw, 18), "")
        self._add(self.rule_desc)
        y += 20 + sgap

        # Settings
        self.set_section = Label(pygame.Rect(px, y, pw, 16), "-- SETTINGS --", color=ACCENT)
        self._add(self.set_section)
        y += 18

        self.width_label = Label(pygame.Rect(px, y, 50, h), "Width:")
        self.width_spin = SpinBox(pygame.Rect(px + 54, y, 65, h), 11, 801, 201, step=2)
        self.cs_label = Label(pygame.Rect(px + 125, y, 26, h), "CS:")
        self.cs_spin = SpinBox(pygame.Rect(px + 153, y, 50, h), 1, 10, 2, step=1,
                               callback=self._on_cs_change)
        self._add(self.width_label, self.width_spin, self.cs_label, self.cs_spin)
        y += h + gap

        self.init_label = Label(pygame.Rect(px, y, 50, h), "Init:")
        self.init_combo = ComboBox(pygame.Rect(px + 54, y, hw - 54, h),
                                   ["center", "random", "left", "right"], 0)
        self.color_label = Label(pygame.Rect(px + hw + gap, y, 42, h), "Color:")
        self.color_combo = ComboBox(pygame.Rect(px + hw + gap + 44, y, pw - hw - gap - 44, h),
                                    ["teal", "rainbow", "bw"], 0)
        self._add(self.init_label, self.init_combo, self.color_label, self.color_combo)
        y += h + sgap

        # Speed
        self.speed_section = Label(pygame.Rect(px, y, pw, 16), "-- SPEED --", color=ACCENT)
        self._add(self.speed_section)
        y += 18

        self.speed_slider = Slider(pygame.Rect(px, y, pw, h), 1, 200, 20,
                                   self._on_speed_change)
        self._add(self.speed_slider)
        y += h + sgap

        # Auto scroll
        self.autoscroll_cb = CheckBox(pygame.Rect(px, y, pw, h), "Auto-Scroll", True,
                                      lambda v: setattr(self, 'auto_scroll', v))
        self._add(self.autoscroll_cb)
        y += h + gap

        # Stats
        self.gen_label = Label(pygame.Rect(px, y, pw, 18), "Gen: 0  |  Rule: 30")
        self._add(self.gen_label)

    def _add(self, *widgets):
        for w in widgets:
            self.widgets.append(w)

    def _on_play_toggle(self):
        self.is_playing = not self.is_playing
        self.play_btn.text = "Pause" if self.is_playing else "Play"
        if self.is_playing:
            self.last_step_time = pygame.time.get_ticks()

    def _on_step(self):
        self.ca.step()

    def _on_reset(self):
        self.is_playing = False
        self.play_btn.text = "Play"
        init = self.init_combo.selected_text if self.init_combo else 'center'
        self.ca.reset(rule=self.rule_spin.value, width=self.width_spin.value, init_mode=init)
        self.scroll_offset = 0

    def _on_skip(self):
        self.ca.step_n(self.skip_spin.value)

    def _on_rule_change(self, val):
        self.ca.rule = int(val)
        self._on_reset()

    def _on_rule_preset(self, idx):
        name = self.rule_combo.selected_text
        for n, r in FAMOUS_RULES:
            if n == name:
                self.rule_spin.value = r
                self.ca.rule = r
                self._on_reset()
                break

    def _on_cs_change(self, val):
        self.cell_size = max(1, int(val))

    def _on_speed_change(self, val):
        self.sim_speed = int(val)

    def handle_event(self, event):
        # Keyboard
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._on_play_toggle()
                return True
            elif event.key == pygame.K_RIGHT:
                self._on_step()
                return True
            elif event.key == pygame.K_r:
                self._on_reset()
                return True

        mx, my = pygame.mouse.get_pos() if event.type in (
            pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION,
            pygame.MOUSEWHEEL) else (0, 0)

        # Route to widgets first
        any_dropdown_open = any(isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open
                                for w in self.widgets)
        if any_dropdown_open or mx >= CTRL_PANEL_X:
            for w in self.widgets:
                if w.handle_event(event):
                    return True
            if event.type == pygame.MOUSEWHEEL and mx >= CTRL_PANEL_X:
                return True

        # Rule visualizer clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            rvr = self.rule_viz_rect
            if rvr.collidepoint(event.pos):
                # Determine which of the 8 output cells was clicked
                rel_x = event.pos[0] - rvr.x
                cell_total_w = rvr.width / 8
                idx = int(rel_x / cell_total_w)
                if 0 <= idx < 8:
                    bits = self.ca.get_rule_bits()
                    new_val = 1 - bits[idx]
                    self.ca.set_rule_bit(idx, new_val)
                    self.rule_spin.value = self.ca.rule
                    self._on_reset()
                    return True

        # Canvas scroll (wheel over canvas area)
        if event.type == pygame.MOUSEWHEEL and mx < CANVAS_PANEL_W:
            self.scroll_offset -= event.y * 20
            self.auto_scroll = False
            return True

        return False

    def update(self):
        now = pygame.time.get_ticks()
        if self.is_playing and now - self.last_step_time >= self.sim_speed:
            self.ca.step()
            self.last_step_time = now

    def draw(self, fps):
        canvas_rect = pygame.Rect(0, TAB_BAR_H, CANVAS_PANEL_W, WINDOW_H - TAB_BAR_H - STATUS_BAR_H)
        pygame.draw.rect(self.screen, CANVAS_BG, canvas_rect)

        # Render 1D CA
        color_mode = self.color_combo.selected_text if self.color_combo else 'teal'
        rgb = self.ca.render_rgb(color_mode, self.cell_size)

        n_gens = rgb.shape[0]
        n_cols = rgb.shape[1]
        cs = self.cell_size

        img_h = n_gens * cs
        img_w = n_cols * cs

        # Create surface
        if img_w > 0 and img_h > 0:
            # Scale up
            if cs > 1:
                scaled = np.repeat(np.repeat(rgb, cs, axis=0), cs, axis=1)
            else:
                scaled = rgb

            surf = pygame.Surface((scaled.shape[1], scaled.shape[0]))
            pygame.surfarray.blit_array(surf, scaled.transpose(1, 0, 2))

            # Auto-scroll to show latest generation
            if self.auto_scroll:
                visible_h = canvas_rect.height
                self.scroll_offset = max(0, img_h - visible_h)

            self.scroll_offset = max(0, min(self.scroll_offset, img_h - canvas_rect.height))

            # Blit
            clip_rect = canvas_rect.clip(pygame.Rect(0, self.scroll_offset, img_w, canvas_rect.height))
            if clip_rect.height > 0:
                src_rect = pygame.Rect(0, int(self.scroll_offset), img_w, clip_rect.height)
                self.screen.blit(surf, clip_rect.topleft, src_rect)

        pygame.draw.rect(self.screen, BORDER_COLOR, canvas_rect, 1)

        # Draw panel
        panel_rect = pygame.Rect(CTRL_PANEL_X, TAB_BAR_H, CTRL_PANEL_W,
                                 WINDOW_H - TAB_BAR_H - STATUS_BAR_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, BORDER_COLOR,
                         (CTRL_PANEL_X, TAB_BAR_H), (CTRL_PANEL_X, WINDOW_H - STATUS_BAR_H), 1)

        clip = self.screen.get_clip()
        self.screen.set_clip(panel_rect)
        for w in self.widgets:
            w.draw(self.screen)
        self.screen.set_clip(clip)

        # Draw dropdowns on top
        for w in self.widgets:
            if isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open:
                w.draw_dropdown(self.screen)

        # Draw rule visualizer
        self._draw_rule_viz()

        # Update labels
        self.gen_label.text = f"Gen: {self.ca.generation}  |  Rule: {self.ca.rule}"
        # Rule description
        rule = self.ca.rule
        b7 = (rule >> 7) & 1
        b0 = rule & 1
        total = bin(rule).count('1')
        if total == 0:
            desc = "Class I - All cells die"
        elif total == 8:
            desc = "All cells survive"
        elif rule in (0, 4, 32, 36, 108, 128, 160, 232):
            desc = "Class I - Evolves to uniform"
        elif rule in (90, 150, 182, 218, 222, 250):
            desc = "Class II/III - Sierpinski-like patterns"
        elif rule == 110:
            desc = "Class IV - Turing complete!"
        elif rule in (30, 45, 73, 75, 86, 89, 101, 135, 149, 169):
            desc = "Class III - Chaotic"
        elif rule == 184:
            desc = "Class II - Traffic flow model"
        else:
            desc = f"Complexity: {total}/8 bits set"
        self.rule_desc.text = desc

    def _draw_rule_viz(self):
        """Draw the rule visualizer: 8 groups of 3 input cells + 1 output cell."""
        px = CTRL_CONTENT_X
        y = self.rule_viz_y
        pw = CTRL_CONTENT_W
        font = self.fonts.get('sm', self.fonts.get('md'))
        bits = self.ca.get_rule_bits()
        cell_s = 12
        group_w = pw // 8
        group_h = 40

        # Store rect for click handling
        self.rule_viz_rect = pygame.Rect(px, y, pw, group_h)

        # Background
        pygame.draw.rect(self.screen, INPUT_BG, self.rule_viz_rect, border_radius=4)

        for i in range(8):
            gx = px + i * group_w + group_w // 2

            # 3 input cells (top row)
            for j in range(3):
                bit_val = (7 - i) >> (2 - j) & 1
                cx = gx + (j - 1) * (cell_s + 2) - cell_s // 2
                cy = y + 4
                color = TEXT_BRIGHT if bit_val else (40, 40, 60)
                pygame.draw.rect(self.screen, color, (cx, cy, cell_s, cell_s))
                pygame.draw.rect(self.screen, BORDER_COLOR, (cx, cy, cell_s, cell_s), 1)

            # Arrow
            arrow_x = gx
            arrow_y = y + cell_s + 6
            pygame.draw.polygon(self.screen, TEXT_DIM,
                                [(arrow_x - 3, arrow_y), (arrow_x + 3, arrow_y),
                                 (arrow_x, arrow_y + 4)])

            # 1 output cell (bottom row)
            cx = gx - cell_s // 2
            cy = y + cell_s + 12
            out_val = bits[i]
            color = ACCENT if out_val else (40, 40, 60)
            pygame.draw.rect(self.screen, color, (cx, cy, cell_s, cell_s))
            pygame.draw.rect(self.screen, BORDER_COLOR, (cx, cy, cell_s, cell_s), 1)

            # Bit index
            idx_text = font.render(str(7 - i), True, TEXT_DIM)
            self.screen.blit(idx_text, (gx - idx_text.get_width() // 2, y + group_h - 12))


# ============================================================
# SECTION 5: 1D Comparison Tab
# ============================================================

class ComparisonTab:
    """Compare 4 elementary CA rules side by side (Wolfram classification)."""

    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.cas = [
            ElementaryCA(151, 0, 'center'),    # Class I
            ElementaryCA(151, 4, 'center'),     # Class II
            ElementaryCA(151, 30, 'center'),    # Class III
            ElementaryCA(151, 110, 'center'),   # Class IV
        ]
        self.class_names = ["Class I", "Class II", "Class III", "Class IV"]
        self.is_playing = False
        self.sim_speed = 20
        self.last_step_time = 0
        self.cell_size = 2
        self.widgets = []
        self._build_controls()

    def _build_controls(self):
        px = CTRL_CONTENT_X
        pw = CTRL_CONTENT_W
        y = TAB_BAR_H + 6
        h = 26
        gap = 4
        sgap = 8
        tw = (pw - 2 * gap) // 3

        # Playback
        self.play_btn = Button(pygame.Rect(px, y, tw, h), "Play", self._on_play_toggle)
        self.step_btn = Button(pygame.Rect(px + tw + gap, y, tw, h), "Step", self._on_step)
        self.reset_btn = Button(pygame.Rect(px + 2 * (tw + gap), y, tw, h), "Reset", self._on_reset)
        self._add(self.play_btn, self.step_btn, self.reset_btn)
        y += h + sgap

        # 4 Rule spinboxes
        self.rule_spins = []
        self.rule_labels = []
        for i in range(4):
            self.rule_labels.append(Label(pygame.Rect(px, y, 60, h), f"{self.class_names[i]}:"))
            self.rule_spins.append(SpinBox(pygame.Rect(px + 64, y, 60, h), 0, 255,
                                           [0, 4, 30, 110][i], step=1,
                                           callback=lambda val, idx=i: self._on_rule_change(idx, val)))
            self._add(self.rule_labels[i], self.rule_spins[i])
            y += h + gap

        y += sgap - gap

        # Preset buttons
        preset_hw = (pw - 3 * gap) // 4
        presets = [("I:0", 0), ("II:4", 1), ("III:30", 2), ("IV:110", 3)]
        self.preset_btns = []
        for i, (name, _) in enumerate(presets):
            btn = Button(pygame.Rect(px + i * (preset_hw + gap), y, preset_hw, h),
                         name, lambda v, idx=i: self._on_preset(idx))
            self.preset_btns.append(btn)
            self._add(btn)
        y += h + sgap

        # Speed
        self.speed_slider = Slider(pygame.Rect(px, y, pw, h), 1, 200, 20,
                                   self._on_speed_change)
        self._add(self.speed_slider)
        y += h + gap

        self.cs_label = Label(pygame.Rect(px, y, 26, h), "CS:")
        self.cs_spin = SpinBox(pygame.Rect(px + 30, y, 50, h), 1, 8, 2, step=1,
                               callback=lambda v: setattr(self, 'cell_size', max(1, int(v))))
        self._add(self.cs_label, self.cs_spin)
        y += h + sgap

        # Gen label
        self.gen_label = Label(pygame.Rect(px, y, pw, 18), "Gen: 0")
        self._add(self.gen_label)

    def _add(self, *widgets):
        for w in widgets:
            self.widgets.append(w)

    def _on_play_toggle(self):
        self.is_playing = not self.is_playing
        self.play_btn.text = "Pause" if self.is_playing else "Play"
        if self.is_playing:
            self.last_step_time = pygame.time.get_ticks()

    def _on_step(self):
        for ca in self.cas:
            ca.step()

    def _on_reset(self):
        self.is_playing = False
        self.play_btn.text = "Play"
        for i, ca in enumerate(self.cas):
            ca.reset(rule=self.rule_spins[i].value, width=ca.width, init_mode='center')

    def _on_rule_change(self, idx, val):
        self.cas[idx].reset(rule=int(val), width=self.cas[idx].width, init_mode='center')

    def _on_preset(self, idx):
        rules = [0, 4, 30, 110]
        self.rule_spins[idx].value = rules[idx]
        self.cas[idx].reset(rule=rules[idx], width=self.cas[idx].width, init_mode='center')

    def _on_speed_change(self, val):
        self.sim_speed = int(val)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._on_play_toggle()
                return True
            elif event.key == pygame.K_RIGHT:
                self._on_step()
                return True

        mx, my = pygame.mouse.get_pos() if event.type in (
            pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION,
            pygame.MOUSEWHEEL) else (0, 0)

        any_dropdown_open = any(isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open
                                for w in self.widgets)
        if any_dropdown_open or mx >= CTRL_PANEL_X:
            for w in self.widgets:
                if w.handle_event(event):
                    return True
            if event.type == pygame.MOUSEWHEEL and mx >= CTRL_PANEL_X:
                return True
        return False

    def update(self):
        now = pygame.time.get_ticks()
        if self.is_playing and now - self.last_step_time >= self.sim_speed:
            for ca in self.cas:
                ca.step()
            self.last_step_time = now

    def draw(self, fps):
        canvas_rect = pygame.Rect(0, TAB_BAR_H, CANVAS_PANEL_W, WINDOW_H - TAB_BAR_H - STATUS_BAR_H)
        pygame.draw.rect(self.screen, CANVAS_BG, canvas_rect)

        # 2x2 grid
        margin = 4
        qw = (CANVAS_PANEL_W - 3 * margin) // 2
        qh = (canvas_rect.height - 3 * margin - 40) // 2
        font = self.fonts.get('md', self.fonts.get('lg'))
        sm_font = self.fonts.get('sm', font)

        for i in range(4):
            col = i % 2
            row = i // 2
            qx = canvas_rect.x + col * (qw + margin)
            qy = canvas_rect.y + row * (qh + margin) + 24

            # Label
            label = font.render(f"{self.class_names[i]} - Rule {self.rule_spins[i].value}", True, ACCENT)
            self.screen.blit(label, (qx + 4, qy - 18))

            # Sub-rect
            sub_rect = pygame.Rect(qx, qy, qw, qh)
            pygame.draw.rect(self.screen, (12, 12, 22), sub_rect)
            pygame.draw.rect(self.screen, BORDER_COLOR, sub_rect, 1)

            # Render CA
            ca = self.cas[i]
            rgb = ca.render_rgb('teal', self.cell_size)
            n_gens, n_cols = rgb.shape[0], rgb.shape[1]
            cs = self.cell_size

            img_h = n_gens * cs
            img_w = n_cols * cs

            if img_w > 0 and img_h > 0:
                if cs > 1:
                    scaled = np.repeat(np.repeat(rgb, cs, axis=0), cs, axis=1)
                else:
                    scaled = rgb
                surf = pygame.Surface((scaled.shape[1], scaled.shape[0]))
                pygame.surfarray.blit_array(surf, scaled.transpose(1, 0, 2))

                # Scale to fit sub-rect
                scale_x = qw / max(1, img_w)
                scale_y = qh / max(1, img_h)
                scale = min(scale_x, scale_y, 1.0)
                if scale < 1.0:
                    new_w = max(1, int(img_w * scale))
                    new_h = max(1, int(img_h * scale))
                    surf = pygame.transform.scale(surf, (new_w, new_h))

                # Center in sub-rect
                bx = qx + (qw - surf.get_width()) // 2
                by = qy + (qh - surf.get_height()) // 2
                self.screen.blit(surf, (bx, by))

        # Draw panel
        panel_rect = pygame.Rect(CTRL_PANEL_X, TAB_BAR_H, CTRL_PANEL_W,
                                 WINDOW_H - TAB_BAR_H - STATUS_BAR_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, BORDER_COLOR,
                         (CTRL_PANEL_X, TAB_BAR_H), (CTRL_PANEL_X, WINDOW_H - STATUS_BAR_H), 1)

        clip = self.screen.get_clip()
        self.screen.set_clip(panel_rect)
        for w in self.widgets:
            w.draw(self.screen)
        self.screen.set_clip(clip)

        for w in self.widgets:
            if isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open:
                w.draw_dropdown(self.screen)

        self.gen_label.text = f"Gen: {self.cas[0].generation}"


# ============================================================
# SECTION 6: 2D Explorer Tab (Simple Grid)
# ============================================================

class SimpleCA:
    """A simple 2D CA using Python lists for the grid."""

    def __init__(self, rows=80, cols=120, birth=None, survive=None):
        self.rows = rows
        self.cols = cols
        self.birth = birth if birth is not None else {3}
        self.survive = survive if survive is not None else {2, 3}
        self.grid = [[0] * cols for _ in range(rows)]
        self.age_grid = [[0] * cols for _ in range(rows)]
        self.heat_grid = [[0.0] * cols for _ in range(rows)]
        self.generation = 0
        self.wrap = True

    def step(self):
        new_grid = [[0] * self.cols for _ in range(self.rows)]
        new_age = [[0] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                n = self._count_neighbors(r, c)
                if self.grid[r][c]:
                    if n in self.survive:
                        new_grid[r][c] = 1
                        new_age[r][c] = self.age_grid[r][c] + 1
                    self.heat_grid[r][c] = min(1.0, self.heat_grid[r][c] + 0.15)
                else:
                    if n in self.birth:
                        new_grid[r][c] = 1
                        new_age[r][c] = 1
                    self.heat_grid[r][c] = max(0.0, self.heat_grid[r][c] - 0.02)
        self.grid = new_grid
        self.age_grid = new_age
        self.generation += 1

    def _count_neighbors(self, r, c):
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.wrap:
                    nr = nr % self.rows
                    nc = nc % self.cols
                elif 0 <= nr < self.rows and 0 <= nc < self.cols:
                    pass
                else:
                    continue
                if self.grid[nr][nc]:
                    count += 1
        return count

    def clear(self):
        self.grid = [[0] * self.cols for _ in range(self.rows)]
        self.age_grid = [[0] * self.cols for _ in range(self.rows)]
        self.heat_grid = [[0.0] * self.cols for _ in range(self.rows)]
        self.generation = 0

    def randomize(self, density=0.3):
        for r in range(self.rows):
            for c in range(self.cols):
                if random.random() < density:
                    self.grid[r][c] = 1
                    self.age_grid[r][c] = 1
                else:
                    self.grid[r][c] = 0
                    self.age_grid[r][c] = 0

    def invert(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = 1 - self.grid[r][c]
                if self.grid[r][c] and self.age_grid[r][c] == 0:
                    self.age_grid[r][c] = 1

    def get_population(self):
        return sum(self.grid[r][c] for r in range(self.rows) for c in range(self.cols))

    def set_cell(self, r, c, val):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r][c] = val
            if val and self.age_grid[r][c] == 0:
                self.age_grid[r][c] = 1

    def inject_pattern(self, pattern_name, center_r, center_c):
        coords = PATTERN_DATA.get(pattern_name, [])
        if not coords and "Random" in pattern_name:
            density = {"Random 5%": 0.05, "Random 15%": 0.15,
                       "Random 30%": 0.30, "Random 50%": 0.50}.get(pattern_name, 0.1)
            for r in range(self.rows):
                for c in range(self.cols):
                    if random.random() < density:
                        self.grid[r][c] = 1
                        self.age_grid[r][c] = 1
            return

        for dr, dc in coords:
            r, c = center_r + dr, center_c + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r][c] = 1
                self.age_grid[r][c] = 1

    def render_array(self, mode='standard', cell_size=6):
        """Render grid to RGB numpy array."""
        rows, cols = self.rows, self.cols
        img = np.zeros((rows, cols, 3), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                if self.grid[r][c]:
                    if mode == 'standard':
                        img[r, c] = (0, 212, 168)  # Teal
                    elif mode == 'age':
                        age = min(self.age_grid[r][c], 100)
                        hue = max(0, 160 - age * 3.2)  # green to red
                        img[r, c] = self._hue_to_rgb(hue, 0.85, 1.0)
                    elif mode == 'heat':
                        h = min(1.0, self.heat_grid[r][c])
                        img[r, c] = (int(255 * h), int(80 * h), int(20 * h))
                    else:
                        img[r, c] = (0, 212, 168)
                else:
                    if mode == 'heat':
                        h = min(1.0, self.heat_grid[r][c])
                        v = int(20 + 30 * h)
                        img[r, c] = (v, v // 2, v // 3)
                    else:
                        img[r, c] = (16, 16, 28)
        return img

    @staticmethod
    def _hue_to_rgb(h, s, v):
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    def resize(self, rows, cols):
        new_grid = [[0] * cols for _ in range(rows)]
        new_age = [[0] * cols for _ in range(rows)]
        new_heat = [[0.0] * cols for _ in range(rows)]
        for r in range(min(rows, self.rows)):
            for c in range(min(cols, self.cols)):
                new_grid[r][c] = self.grid[r][c]
                new_age[r][c] = self.age_grid[r][c]
                new_heat[r][c] = self.heat_grid[r][c]
        self.rows = rows
        self.cols = cols
        self.grid = new_grid
        self.age_grid = new_age
        self.heat_grid = new_heat


class ExplorerTab:
    """Simplified 2D CA Explorer with rule editor."""

    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts = fonts
        self.ca = SimpleCA(80, 120, {3}, {2, 3})
        self.canvas_renderer = CanvasRenderer(pygame.Rect(0, TAB_BAR_H, CANVAS_PANEL_W,
                                                          WINDOW_H - TAB_BAR_H - STATUS_BAR_H - 120))
        self.is_playing = False
        self.sim_speed = 50
        self.last_step_time = 0
        self.hover_cell = None
        self._drawing = False
        self._draw_value = 1
        self._last_draw_cell = None
        self.widgets = []
        self.birth_cbs = []
        self.survive_cbs = []
        self.pop_graph = PopulationGraph(pygame.Rect(0, WINDOW_H - STATUS_BAR_H - 118,
                                                      CANVAS_PANEL_W, 116))
        self._build_controls()
        self.canvas_renderer.reset_view(self.ca.rows, self.ca.cols, 6)

    def _build_controls(self):
        px = CTRL_CONTENT_X
        pw = CTRL_CONTENT_W
        y = TAB_BAR_H + 6
        h = 24
        gap = 3
        sgap = 6
        hw = (pw - gap) // 2
        tw = (pw - 2 * gap) // 3
        fw = (pw - 3 * gap) // 4
        qw = (pw - 4 * gap) // 5
        csz = 22  # checkbox size for rule editor

        # Playback
        self.play_btn = Button(pygame.Rect(px, y, tw, h), "Play", self._on_play_toggle)
        self.step_btn = Button(pygame.Rect(px + tw + gap, y, tw, h), "Step", self._on_step)
        self.reset_btn = Button(pygame.Rect(px + 2 * (tw + gap), y, tw, h), "Reset", self._on_reset)
        self._add(self.play_btn, self.step_btn, self.reset_btn)
        y += h + gap

        # Actions
        self.random_btn = Button(pygame.Rect(px, y, hw, h), "Random", self._on_random)
        self.invert_btn = Button(pygame.Rect(px + hw + gap, y, hw, h), "Invert", self._on_invert)
        self._add(self.random_btn, self.invert_btn)
        y += h + sgap

        # Rule combo
        self.rule_section = Label(pygame.Rect(px, y, pw, 16), "-- RULE --", color=ACCENT)
        self._add(self.rule_section)
        y += 18

        self.rule_combo = ComboBox(pygame.Rect(px, y, pw, h), PRESET_2D_RULES, 0,
                                   callback=self._on_rule_preset)
        self._add(self.rule_combo)
        y += h + sgap

        # Birth checkboxes
        self.birth_label = Label(pygame.Rect(px, y, 30, csz), "B:", color=ACCENT)
        self._add(self.birth_label)
        bx = px + 28
        for i in range(10):
            cb = CheckBox(pygame.Rect(bx + i * (csz + 1), y, csz, csz), str(i),
                          i in self.ca.birth, self._on_rule_cb_change)
            self.birth_cbs.append(cb)
            self._add(cb)
        y += csz + 2

        # Survive checkboxes
        self.survive_label = Label(pygame.Rect(px, y, 30, csz), "S:", color=ORANGE)
        self._add(self.survive_label)
        for i in range(10):
            cb = CheckBox(pygame.Rect(bx + i * (csz + 1), y, csz, csz), str(i),
                          i in self.ca.survive, self._on_rule_cb_change)
            self.survive_cbs.append(cb)
            self._add(cb)
        y += csz + sgap

        # Pattern combo
        self.pat_section = Label(pygame.Rect(px, y, pw, 16), "-- PATTERN --", color=ACCENT)
        self._add(self.pat_section)
        y += 18

        self.pat_combo = ComboBox(pygame.Rect(px, y, pw - 60, h), PRESET_PATTERNS, 0)
        self.inject_btn = Button(pygame.Rect(px + pw - 55, y, 55, h), "Add", self._on_inject)
        self._add(self.pat_combo, self.inject_btn)
        y += h + sgap

        # Display mode
        self.disp_section = Label(pygame.Rect(px, y, pw, 16), "-- DISPLAY --", color=ACCENT)
        self._add(self.disp_section)
        y += 18

        self.disp_combo = ComboBox(pygame.Rect(px, y, hw, h),
                                   ["standard", "age", "heat"], 0)
        self.speed_label = Label(pygame.Rect(px + hw + gap, y, 40, h), "Spd:")
        self.speed_slider = Slider(pygame.Rect(px + hw + gap + 44, y, pw - hw - gap - 44, h),
                                   1, 500, 50, self._on_speed_change)
        self._add(self.disp_combo, self.speed_label, self.speed_slider)
        y += h + sgap

        # Grid size + fit
        self.grid_label = Label(pygame.Rect(px, y, 36, h), "Grid:")
        self.grid_spin = SpinBox(pygame.Rect(px + 38, y, 55, h), 20, 300, 80, step=10)
        self.grid_label2 = Label(pygame.Rect(px + 97, y, 10, h), "x")
        self.grid_spin2 = SpinBox(pygame.Rect(px + 109, y, 55, h), 20, 400, 120, step=10)
        self.apply_btn = Button(pygame.Rect(px + 170, y, 50, h), "Apply", self._on_apply_grid)
        self.fit_btn = Button(pygame.Rect(px + 224, y, 60, h), "Fit", self._on_fit_view)
        self._add(self.grid_label, self.grid_spin, self.grid_label2, self.grid_spin2,
                  self.apply_btn, self.fit_btn)
        y += h + gap

        # Wrap
        self.wrap_cb = CheckBox(pygame.Rect(px, y, pw, h), "Wrap Edges", True,
                                lambda v: setattr(self.ca, 'wrap', v))
        self._add(self.wrap_cb)
        y += h + sgap

        # Stats
        self.gen_label = Label(pygame.Rect(px, y, pw, 18), "Gen: 0  |  Pop: 0")
        self._add(self.gen_label)

    def _add(self, *widgets):
        for w in widgets:
            self.widgets.append(w)

    def _on_play_toggle(self):
        self.is_playing = not self.is_playing
        self.play_btn.text = "Pause" if self.is_playing else "Play"
        if self.is_playing:
            self.last_step_time = pygame.time.get_ticks()

    def _on_step(self):
        self.ca.step()
        self.pop_graph.add(self.ca.get_population())

    def _on_reset(self):
        self.is_playing = False
        self.play_btn.text = "Play"
        self.ca.clear()
        self.pop_graph.clear_data()

    def _on_random(self):
        self.ca.randomize(0.3)
        self.pop_graph.clear_data()

    def _on_invert(self):
        self.ca.invert()

    def _on_inject(self):
        name = self.pat_combo.selected_text
        self.ca.inject_pattern(name, self.ca.rows // 2, self.ca.cols // 2)

    def _on_rule_preset(self, idx):
        text = self.rule_combo.selected_text
        if text == "Custom":
            return
        try:
            parts = text.split(" - ")[0]
            b_part, s_part = parts.split("/")

            birth = set()
            for ch in b_part[1:]:
                if ch.isdigit():
                    birth.add(int(ch))

            survive = set()
            for ch in s_part[1:]:
                if ch.isdigit():
                    survive.add(int(ch))

            self.ca.birth = birth
            self.ca.survive = survive

            # Update checkboxes
            for i, cb in enumerate(self.birth_cbs):
                cb.checked = i in birth
            for i, cb in enumerate(self.survive_cbs):
                cb.checked = i in survive
        except Exception:
            pass

    def _on_rule_cb_change(self, val=None):
        birth = set()
        for i, cb in enumerate(self.birth_cbs):
            if cb.checked:
                birth.add(i)
        survive = set()
        for i, cb in enumerate(self.survive_cbs):
            if cb.checked:
                survive.add(i)
        self.ca.birth = birth
        self.ca.survive = survive

    def _on_speed_change(self, val):
        self.sim_speed = int(val)

    def _on_apply_grid(self):
        r = self.grid_spin.value
        c = self.grid_spin2.value
        self.ca.resize(r, c)
        self.canvas_renderer.reset_view(r, c, 6)
        self.pop_graph.clear_data()

    def _on_fit_view(self):
        self.canvas_renderer.reset_view(self.ca.rows, self.ca.cols, 6)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._on_play_toggle()
                return True
            elif event.key == pygame.K_RIGHT:
                self._on_step()
                return True
            elif event.key == pygame.K_c:
                self._on_reset()
                return True
            elif event.key == pygame.K_r:
                self._on_random()
                return True

        mx, my = pygame.mouse.get_pos() if event.type in (
            pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION,
            pygame.MOUSEWHEEL) else (0, 0)

        any_dropdown_open = any(isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open
                                for w in self.widgets)
        if any_dropdown_open or mx >= CTRL_PANEL_X:
            for w in self.widgets:
                if w.handle_event(event):
                    return True
            if event.type == pygame.MOUSEWHEEL and mx >= CTRL_PANEL_X:
                return True

        # Canvas events
        cr = self.canvas_renderer.screen_rect
        if cr.collidepoint(mx, my) or self.canvas_renderer._dragging_pan:
            if self.canvas_renderer.handle_event(event, self.ca.rows, self.ca.cols, 6):
                return True

        # Drawing on canvas
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (1, 3) and cr.collidepoint(event.pos):
                cell = self.canvas_renderer.screen_to_cell(event.pos[0], event.pos[1],
                                                            self.ca.rows, self.ca.cols, 6)
                if cell:
                    self._drawing = True
                    self._draw_value = 1 if event.button == 1 else 0
                    self.ca.set_cell(cell[0], cell[1], self._draw_value)
                    self._last_draw_cell = cell
                    return True

        elif event.type == pygame.MOUSEMOTION and self._drawing:
            cell = self.canvas_renderer.screen_to_cell(event.pos[0], event.pos[1],
                                                        self.ca.rows, self.ca.cols, 6)
            if cell and cell != self._last_draw_cell:
                self.ca.set_cell(cell[0], cell[1], self._draw_value)
                self._last_draw_cell = cell
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3):
            self._drawing = False
            self._last_draw_cell = None

        if event.type == pygame.MOUSEMOTION:
            if cr.collidepoint(event.pos):
                self.hover_cell = self.canvas_renderer.screen_to_cell(
                    event.pos[0], event.pos[1], self.ca.rows, self.ca.cols, 6)
            else:
                self.hover_cell = None

        return False

    def update(self):
        now = pygame.time.get_ticks()
        if self.is_playing and now - self.last_step_time >= self.sim_speed:
            self.ca.step()
            self.pop_graph.add(self.ca.get_population())
            self.last_step_time = now

    def draw(self, fps):
        mode = self.disp_combo.selected_text if self.disp_combo else 'standard'

        try:
            rgb = self.ca.render_array(mode, 6)
        except Exception:
            rgb = np.full((self.ca.rows, self.ca.cols, 3), CANVAS_BG, dtype=np.uint8)

        self.canvas_renderer.render(
            self.screen, rgb, self.ca.rows, self.ca.cols, 6,
            True, GRID_LINE_COLOR, self.hover_cell, "None"
        )

        # Population graph
        self.pop_graph.draw(self.screen, self.fonts.get('sm', self.fonts.get('md')))

        # Panel
        panel_rect = pygame.Rect(CTRL_PANEL_X, TAB_BAR_H, CTRL_PANEL_W,
                                 WINDOW_H - TAB_BAR_H - STATUS_BAR_H)
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, BORDER_COLOR,
                         (CTRL_PANEL_X, TAB_BAR_H), (CTRL_PANEL_X, WINDOW_H - STATUS_BAR_H), 1)

        clip = self.screen.get_clip()
        self.screen.set_clip(panel_rect)
        for w in self.widgets:
            w.draw(self.screen)
        self.screen.set_clip(clip)

        for w in self.widgets:
            if isinstance(w, ComboBox) and hasattr(w, 'is_open') and w.is_open:
                w.draw_dropdown(self.screen)

        self.gen_label.text = f"Gen: {self.ca.generation}  |  Pop: {self.ca.get_population()}"

        return self.ca.get_population()


# ============================================================
# SECTION 7: Population Graph
# ============================================================

class PopulationGraph:
    """Mini line chart showing population over time."""

    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.data = []
        self.max_pts = 500

    def add(self, pop):
        self.data.append(pop)
        if len(self.data) > self.max_pts:
            self.data.pop(0)

    def clear_data(self):
        self.data.clear()

    def draw(self, surface, font):
        r = self.rect
        # Background
        pygame.draw.rect(surface, (20, 20, 36), r)
        pygame.draw.rect(surface, BORDER_COLOR, r, 1)

        if len(self.data) < 2:
            hint = font.render("Population Graph", True, TEXT_DIM)
            surface.blit(hint, (r.x + r.width // 2 - hint.get_width() // 2,
                                r.y + r.height // 2 - hint.get_height() // 2))
            return

        margin_l = 5
        margin_r = 50
        margin_t = 5
        margin_b = 5
        plot_x = r.x + margin_l
        plot_y = r.y + margin_t
        plot_w = r.width - margin_l - margin_r
        plot_h = r.height - margin_t - margin_b

        if plot_w <= 0 or plot_h <= 0:
            return

        max_val = max(self.data) if self.data else 1
        min_val = min(self.data) if self.data else 0
        val_range = max(max_val - min_val, 1)

        # Build point list
        n = len(self.data)
        points = []
        for i, val in enumerate(self.data):
            x = plot_x + int(i * plot_w / max(1, n - 1))
            y = plot_y + plot_h - int((val - min_val) / val_range * plot_h)
            points.append((x, y))

        # Filled area
        if len(points) >= 2:
            fill_points = list(points) + [(points[-1][0], plot_y + plot_h),
                                          (points[0][0], plot_y + plot_h)]
            fill_surf = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            adjusted = [(p[0] - r.x, p[1] - r.y) for p in fill_points]
            try:
                pygame.draw.polygon(fill_surf, (0, 212, 168, 40), adjusted)
                surface.blit(fill_surf, r.topleft)
            except Exception:
                pass

            # Line
            if len(points) >= 2:
                pygame.draw.lines(surface, ACCENT, False, points, 2)

        # Max value label
        max_text = font.render(f"Max: {max_val}", True, ACCENT)
        surface.blit(max_text, (r.x + r.width - margin_r + 5, r.y + 4))

        # Current value
        cur_text = font.render(f"Cur: {self.data[-1]}", True, TEXT_COLOR)
        surface.blit(cur_text, (r.x + r.width - margin_r + 5, r.y + 18))

        # Min value
        min_text = font.render(f"Min: {min_val}", True, TEXT_DIM)
        surface.blit(min_text, (r.x + r.width - margin_r + 5, r.y + 32))


# ============================================================
# SECTION 8: Main Application
# ============================================================

class CellularAutomataApp:
    """Main application with tab system and game loop."""

    def __init__(self):
        pygame.init()
        pygame.scrap.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Cellular Automata Studio & Explorer v3.0 (Pygame)")
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = 60.0

        # Fonts
        try:
            self.fonts = {
                'sm': pygame.font.SysFont("segoeui", 11),
                'md': pygame.font.SysFont("segoeui", 13),
                'lg': pygame.font.SysFont("segoeui", 16),
                'xl': pygame.font.SysFont("segoeui", 20),
                'title': pygame.font.SysFont("segoeui", 24),
            }
        except Exception:
            self.fonts = {
                'sm': pygame.font.Font(None, 14),
                'md': pygame.font.Font(None, 16),
                'lg': pygame.font.Font(None, 20),
                'xl': pygame.font.Font(None, 24),
                'title': pygame.font.Font(None, 28),
            }

        # Create shared engine for studio tab
        try:
            self.engine = CAEngine()
        except NameError:
            # Fallback if CAEngine not available
            self.engine = type('MockEngine', (), {
                'rows': 100, 'cols': 100,
                'cell_size': 6, 'generation': 0,
                'get_population': lambda self: 0,
                'push_undo': lambda self: None,
                'undo': lambda self: None,
                'redo': lambda self: None,
                'step': lambda self: None,
                'clear': lambda self: None,
                'reset': lambda self: None,
                'randomize': lambda self, d: None,
                'set_cell': lambda self, r, c, v: None,
                'set_visual_mode': lambda self, m: None,
                'set_palette': lambda self, n: None,
                'set_trails': lambda self, v: None,
                'set_glow': lambda self, v: None,
                'set_vignette': lambda self, v: None,
                'set_birth_effect': lambda self, v: None,
                'set_death_effect': lambda self, v: None,
                'set_grid_lines': lambda self, v: None,
                'set_wrap': lambda self, v: None,
                'set_symmetry': lambda self, m: None,
                'set_max_states': lambda self, n: None,
                'set_rule': lambda self, r: None,
                'resize': lambda self, r, c: None,
                'render': lambda self: np.full((100, 100, 3), CANVAS_BG, dtype=np.uint8),
                'get_state': lambda self: {},
                'set_state': lambda self, s: None,
            })()

        # Tab system
        self.tab_names = ["2D Studio", "1D Elementary", "1D Comparison", "2D Explorer"]
        self.active_tab = 0

        # Create tab instances
        self.studio_tab = StudioTab(self.screen, self.fonts, self.engine)
        self.elementary_tab = ElementaryCATab(self.screen, self.fonts)
        self.comparison_tab = ComparisonTab(self.screen, self.fonts)
        self.explorer_tab = ExplorerTab(self.screen, self.fonts)
        self.tabs = [self.studio_tab, self.elementary_tab, self.comparison_tab, self.explorer_tab]

        # Tab bar buttons
        self.tab_buttons = []
        self._build_tab_bar()

        # Text input dialog
        self.dialog = TextInputDialog()

        # Status info
        self.status_population = 0
        self.status_generation = 0

    def _build_tab_bar(self):
        self.tab_buttons = []
        x = 10
        for i, name in enumerate(self.tab_names):
            w = self.fonts['md'].size(name)[0] + 24
            btn = Button(pygame.Rect(x, 4, w, TAB_BAR_H - 8), name,
                         lambda idx=i: self._switch_tab(idx))
            self.tab_buttons.append(btn)
            x += w + 4

    def _switch_tab(self, idx):
        self.active_tab = idx

    def run(self):
        """Main loop."""
        while self.running:
            dt = self.clock.tick(60)
            self.fps = self.clock.get_fps()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if self.dialog.active:
                    self.dialog.handle_event(event)
                    continue
                self._handle_event(event)

            self._update(dt)
            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _handle_event(self, event):
        # Tab bar clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, btn in enumerate(self.tab_buttons):
                if btn.rect.collidepoint(event.pos):
                    self._switch_tab(i)
                    # Update button visual states
                    for j, b in enumerate(self.tab_buttons):
                        if j == i:
                            b._is_active_tab = True
                        else:
                            b._is_active_tab = False
                    return

        # Global keyboard
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.running = False
                return
            elif event.key == pygame.K_1:
                self._switch_tab(0)
                return
            elif event.key == pygame.K_2:
                self._switch_tab(1)
                return
            elif event.key == pygame.K_3:
                self._switch_tab(2)
                return
            elif event.key == pygame.K_4:
                self._switch_tab(3)
                return

        # Route to active tab
        self.tabs[self.active_tab].handle_event(event)

    def _update(self, dt):
        self.tabs[self.active_tab].update()

    def _draw(self):
        self.screen.fill(BG_COLOR)

        # Tab bar background
        tab_bar_rect = pygame.Rect(0, 0, WINDOW_W, TAB_BAR_H)
        pygame.draw.rect(self.screen, (20, 20, 38), tab_bar_rect)
        pygame.draw.line(self.screen, BORDER_COLOR, (0, TAB_BAR_H - 1), (WINDOW_W, TAB_BAR_H - 1))

        # Tab buttons
        for i, btn in enumerate(self.tab_buttons):
            is_active = (i == self.active_tab)
            r = btn.rect
            if is_active:
                pygame.draw.rect(self.screen, (40, 40, 68), r, border_radius=4)
                pygame.draw.rect(self.screen, ACCENT, r, 2, border_radius=4)
                text_color = ACCENT
            else:
                pygame.draw.rect(self.screen, BUTTON_BG, r, border_radius=4)
                text_color = TEXT_COLOR
            text_surf = self.fonts['md'].render(btn.text, True, text_color)
            tx = r.x + (r.width - text_surf.get_width()) // 2
            ty = r.y + (r.height - text_surf.get_height()) // 2
            self.screen.blit(text_surf, (tx, ty))

        # Active tab underline
        if 0 <= self.active_tab < len(self.tab_buttons):
            ar = self.tab_buttons[self.active_tab].rect
            pygame.draw.line(self.screen, ACCENT, (ar.x + 2, ar.bottom - 1),
                             (ar.right - 2, ar.bottom - 1), 2)

        # Draw active tab content
        pop = 0
        if self.active_tab == 0:
            pop = self.studio_tab.draw(self.fps)
        elif self.active_tab == 1:
            self.elementary_tab.draw(self.fps)
        elif self.active_tab == 2:
            self.comparison_tab.draw(self.fps)
        elif self.active_tab == 3:
            pop = self.explorer_tab.draw(self.fps)

        self.status_population = pop if isinstance(pop, (int, float)) else 0

        # Status bar
        status_rect = pygame.Rect(0, WINDOW_H - STATUS_BAR_H, WINDOW_W, STATUS_BAR_H)
        pygame.draw.rect(self.screen, (20, 20, 38), status_rect)
        pygame.draw.line(self.screen, BORDER_COLOR, (0, WINDOW_H - STATUS_BAR_H),
                         (WINDOW_W, WINDOW_H - STATUS_BAR_H))

        tab_name = self.tab_names[self.active_tab]
        status_text = f"  {tab_name}  |  FPS: {self.fps:.0f}  |  Pop: {int(self.status_population)}"
        try:
            if self.active_tab == 0:
                gen = self.engine.generation if hasattr(self.engine, 'generation') else 0
                status_text += f"  |  Gen: {gen}"
            elif self.active_tab == 1:
                status_text += f"  |  Gen: {self.elementary_tab.ca.generation}"
            elif self.active_tab == 2:
                status_text += f"  |  Gen: {self.comparison_tab.cas[0].generation}"
            elif self.active_tab == 3:
                status_text += f"  |  Gen: {self.explorer_tab.ca.generation}"
        except Exception:
            pass

        status_text += "  |  1-4: Switch Tabs  |  Space: Play/Pause  |  Ctrl+Q: Quit"
        status_surf = self.fonts['sm'].render(status_text, True, TEXT_DIM)
        self.screen.blit(status_surf, (8, WINDOW_H - STATUS_BAR_H + 5))

        # Draw dialog on top of everything
        if self.dialog.active:
            self.dialog.draw(self.screen, self.fonts)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    app = CellularAutomataApp()
    app.run()