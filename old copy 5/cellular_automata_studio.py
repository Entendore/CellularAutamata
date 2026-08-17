"""
Cellular Automata Studio v2.1
A comprehensive cellular automata simulator with multiple backends,
visual effects, color palettes, and rule analysis.

All modules combined into a single file.
Usage: python cellular_automata_studio.py
"""


# ========================================================
# SECTION: Color Palettes (from color_palettes.py)
# ========================================================

"""
Advanced color palette system for cellular automata.

Features:
- Multiple palette types (categorical, sequential, diverging, cyclic)
- Age-based color interpolation
- Custom palette creation
- Palette import/export
- Random palette generation
- Color blind friendly palettes
- Noise-based coloring
- State transition blending
"""

import numpy as np
import json
import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


# =====================================================================
# DATA STRUCTURES
# =====================================================================

class PaletteType(Enum):
    """Types of color palettes."""
    CATEGORICAL = "categorical"  
    SEQUENTIAL = "sequential"    
    DIVERGING = "diverging"      
    CYCLIC = "cyclic"           


@dataclass
class Color:
    """Represents an RGB color."""
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
        if len(hex_str) != 6:
            raise ValueError(f"Invalid hex color: {hex_str}")
        return cls(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
    
    @classmethod
    def from_hsv(cls, h: float, s: float, v: float) -> 'Color':
        """Create color from HSV values (h: 0-360, s: 0-1, v: 0-1)."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60: r, g, b = c, x, 0
        elif h < 120: r, g, b = x, c, 0
        elif h < 180: r, g, b = 0, c, x
        elif h < 240: r, g, b = 0, x, c
        elif h < 300: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        
        return cls(int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    def lerp(self, other: 'Color', t: float) -> 'Color':
        """Linear interpolation between two colors."""
        t = max(0.0, min(1.0, t))
        return Color(
            int(self.r + (other.r - self.r) * t),
            int(self.g + (other.g - self.g) * t),
            int(self.b + (other.b - self.b) * t),
            int(self.a + (other.a - self.a) * t)
        )
    
    def brightness(self) -> float:
        """Get perceived brightness (0-255)."""
        return (self.r * 299 + self.g * 587 + self.b * 114) / 1000


@dataclass
class Palette:
    """A color palette with metadata."""
    name: str
    colors: List[Color]
    palette_type: PaletteType = PaletteType.CATEGORICAL
    background_color: Color = field(default_factory=lambda: Color(30, 30, 30))
    description: str = ""
    
    def to_lut(self, max_state: int, interpolate: bool = True) -> np.ndarray:
        """Convert palette to color lookup table."""
        if max_state < 1:
            max_state = 1
        lut = np.zeros((max_state, 3), dtype=np.uint8)
        
        # State 0 is always background
        lut[0] = self.background_color.to_array()
        
        if not self.colors:
            return lut
            
        if not interpolate or max_state <= len(self.colors) + 1:
            # Direct mapping
            for i in range(1, min(max_state, len(self.colors) + 1)):
                lut[i] = self.colors[i - 1].to_array()
            # Repeat colors if we exceed defined colors
            if max_state > len(self.colors) + 1:
                for i in range(len(self.colors) + 1, max_state):
                    lut[i] = self.colors[(i - 1) % len(self.colors)].to_array()
        else:
            # Smooth interpolation across defined color stops
            n_colors = len(self.colors)
            for i in range(1, max_state):
                t = (i - 1) / max(1, (max_state - 2)) * (n_colors - 1)
                idx = int(t)
                frac = t - idx
                
                if idx >= n_colors - 1:
                    lut[i] = self.colors[-1].to_array()
                else:
                    lut[i] = self.colors[idx].lerp(self.colors[idx + 1], frac).to_array()
                    
        return lut
    
    def to_json(self) -> dict:
        """Serialize palette to JSON."""
        return {
            "name": self.name,
            "type": self.palette_type.value,
            "background": self.background_color.to_hex(),
            "colors": [c.to_hex() for c in self.colors],
            "description": self.description
        }
    
    @classmethod
    def from_json(cls, data: dict) -> 'Palette':
        """Deserialize palette from JSON."""
        return cls(
            name=data.get("name", "Unnamed"),
            colors=[Color.from_hex(c) for c in data.get("colors", [])],
            palette_type=PaletteType(data.get("type", "categorical")),
            background_color=Color.from_hex(data.get("background", "#1e1e1e")),
            description=data.get("description", "")
        )


# =====================================================================
# BUILT-IN PALETTES
# =====================================================================

BUILTIN_PALETTES: Dict[str, Palette] = {
    "Standard": Palette(
        name="Standard", description="Default blue/green/red/orange scheme",
        colors=[Color(100,149,237), Color(34,139,34), Color(220,20,60), Color(255,165,0)]
    ),
    "Grayscale": Palette(
        name="Grayscale", palette_type=PaletteType.SEQUENTIAL,
        colors=[Color(60,60,60), Color(120,120,120), Color(180,180,180), Color(240,240,240)],
        background_color=Color(0,0,0)
    ),
    "Plasma": Palette(
        name="Plasma", palette_type=PaletteType.SEQUENTIAL,
        description="Perceptually uniform plasma colormap",
        colors=[Color(13,8,135), Color(126,3,168), Color(204,71,120), Color(248,149,64), Color(240,249,33)],
        background_color=Color(0,0,0)
    ),
    "Viridis": Palette(
        name="Viridis", palette_type=PaletteType.SEQUENTIAL,
        description="Perceptually uniform viridis colormap",
        colors=[Color(68,1,84), Color(59,82,139), Color(33,145,140), Color(94,201,98), Color(253,231,37)],
        background_color=Color(0,0,0)
    ),
    "Inferno": Palette(
        name="Inferno", palette_type=PaletteType.SEQUENTIAL,
        description="Perceptually uniform inferno colormap",
        colors=[Color(0,0,4), Color(87,16,110), Color(188,55,84), Color(249,142,9), Color(252,255,164)],
        background_color=Color(0,0,0)
    ),
    "Neon": Palette(
        name="Neon", description="Bright neon colors on dark background",
        colors=[Color(57,255,20), Color(255,0,255), Color(0,255,255), Color(255,255,0)],
        background_color=Color(10,10,20)
    ),
    "Pastel": Palette(
        name="Pastel", description="Soft pastel colors",
        colors=[Color(255,182,193), Color(176,224,230), Color(255,228,181), Color(221,160,221), Color(144,238,144)],
        background_color=Color(250,245,240)
    ),
    "Earth Tones": Palette(
        name="Earth Tones", description="Natural earth tones",
        colors=[Color(139,90,43), Color(160,120,60), Color(85,107,47), Color(107,142,35), Color(189,183,107)],
        background_color=Color(245,235,220)
    ),
    "Ocean": Palette(
        name="Ocean", palette_type=PaletteType.SEQUENTIAL, description="Ocean depth gradient",
        colors=[Color(0,105,148), Color(0,150,199), Color(72,202,228), Color(144,224,239), Color(202,240,248)],
        background_color=Color(10,25,50)
    ),
    "Fire": Palette(
        name="Fire", palette_type=PaletteType.SEQUENTIAL, description="Fire gradient",
        colors=[Color(128,0,0), Color(255,0,0), Color(255,128,0), Color(255,200,0), Color(255,255,200)],
        background_color=Color(20,10,5)
    ),
    "Matrix": Palette(
        name="Matrix", description="Matrix-style green on black",
        colors=[Color(0,255,0), Color(0,200,0), Color(50,255,50), Color(0,150,0)],
        background_color=Color(0,10,0)
    ),
    "Cyberpunk": Palette(
        name="Cyberpunk", description="Cyberpunk aesthetic colors",
        colors=[Color(255,0,110), Color(0,255,255), Color(255,234,0), Color(131,56,236)],
        background_color=Color(20,10,30)
    ),
    "Mono Blue": Palette(
        name="Mono Blue", palette_type=PaletteType.SEQUENTIAL, description="Blue monochrome gradient",
        colors=[Color(100,149,237), Color(65,105,225), Color(30,60,180), Color(0,30,120)],
        background_color=Color(240,245,255)
    ),
    "Terrain": Palette(
        name="Terrain", palette_type=PaletteType.SEQUENTIAL, description="Geographic terrain colors",
        colors=[Color(0,0,128), Color(0,100,200), Color(194,178,128), Color(34,139,34), Color(0,100,0), Color(139,90,43), Color(255,255,255)],
        background_color=Color(0,0,80)
    ),
    "Rainbow": Palette(
        name="Rainbow", palette_type=PaletteType.CYCLIC, description="Full rainbow spectrum",
        colors=[Color(255,0,0), Color(255,127,0), Color(255,255,0), Color(0,255,0), Color(0,0,255), Color(75,0,130), Color(148,0,211)],
        background_color=Color(30,30,30)
    ),
    "Color Blind Safe": Palette(
        name="Color Blind Safe", description="Accessible palette for color vision deficiency",
        colors=[Color(230,159,0), Color(86,180,233), Color(0,158,115), Color(240,228,66), Color(0,114,178), Color(213,94,0)],
        background_color=Color(245,245,245)
    ),
    "Sepia": Palette(
        name="Sepia", palette_type=PaletteType.SEQUENTIAL, description="Vintage sepia tones",
        colors=[Color(112,66,20), Color(150,100,50), Color(188,143,95), Color(210,180,140)],
        background_color=Color(250,240,220)
    ),
    "Candy": Palette(
        name="Candy", description="Sweet candy colors",
        colors=[Color(255,105,180), Color(255,182,193), Color(100,149,237), Color(138,43,226), Color(255,215,0)],
        background_color=Color(255,240,245)
    ),
    "Thermal": Palette(
        name="Thermal", palette_type=PaletteType.SEQUENTIAL, description="Thermal camera style",
        colors=[Color(0,0,0), Color(33,0,100), Color(0,0,255), Color(0,255,255), Color(0,255,0), Color(255,255,0), Color(255,0,0), Color(255,255,255)],
        background_color=Color(0,0,0)
    ),
    "Amber": Palette(
        name="Amber", palette_type=PaletteType.SEQUENTIAL, description="Amber/retro terminal style",
        colors=[Color(50,30,0), Color(150,100,0), Color(200,150,50), Color(255,200,100), Color(255,235,180)],
        background_color=Color(10,5,0)
    ),
    "Ice": Palette(
        name="Ice", palette_type=PaletteType.SEQUENTIAL, description="Frozen ice colors",
        colors=[Color(100,150,200), Color(150,200,230), Color(200,230,250), Color(230,245,255), Color(250,252,255)],
        background_color=Color(20,30,50)
    ),
}


# =====================================================================
# PALETTE GENERATORS
# =====================================================================

class PaletteGenerator:
    """Generates color palettes programmatically."""
    
    @staticmethod
    def random_palette(n_colors: int = 4, min_brightness: int = 50, max_brightness: int = 230, seed: Optional[int] = None) -> Palette:
        """Generate a random palette with visually distinct colors."""
        if seed is not None: 
            random.seed(seed)
        colors = []
        base_hue = random.random() * 360
        for i in range(n_colors):
            hue = (base_hue + i * (360 / n_colors) + random.uniform(-20, 20)) % 360
            sat = random.uniform(0.5, 1.0)
            val = random.uniform(min_brightness / 255, max_brightness / 255)
            colors.append(Color.from_hsv(hue, sat, val))
        return Palette(name=f"Random_{seed if seed else id(colors)}", colors=colors)

    @staticmethod
    def analogous_palette(base_hue: float = 0, n_colors: int = 4, saturation: float = 0.7, value: float = 0.9) -> Palette:
        """Generate an analogous color palette."""
        colors = []
        spread = 30 
        for i in range(n_colors):
            hue = (base_hue + (i - n_colors//2) * spread / n_colors) % 360
            colors.append(Color.from_hsv(hue, saturation, value))
        return Palette(name=f"Analogous_{int(base_hue)}", colors=colors)

    @staticmethod
    def complementary_palette(base_hue: float = 0, n_colors: int = 4, saturation: float = 0.7, value: float = 0.9) -> Palette:
        """Generate a complementary color palette."""
        colors = []
        complement = (base_hue + 180) % 360
        for i in range(n_colors):
            hue = base_hue + i * 15 if i < n_colors // 2 else complement + (i - n_colors // 2) * 15
            colors.append(Color.from_hsv(hue % 360, saturation, value))
        return Palette(name=f"Complementary_{int(base_hue)}", colors=colors)

    @staticmethod
    def triadic_palette(base_hue: float = 0, saturation: float = 0.7, value: float = 0.9) -> Palette:
        """Generate a triadic color palette (3 hues 120° apart)."""
        colors = [
            Color.from_hsv(base_hue, saturation, value),
            Color.from_hsv((base_hue + 120) % 360, saturation, value),
            Color.from_hsv((base_hue + 240) % 360, saturation, value),
        ]
        return Palette(name=f"Triadic_{int(base_hue)}", colors=colors)

    @staticmethod
    def gradient_palette(color1: Color, color2: Color, n_colors: int = 4) -> Palette:
        """Generate a gradient between two colors."""
        colors = [color1.lerp(color2, i / max(1, n_colors - 1)) for i in range(n_colors)]
        return Palette(name="Gradient", colors=colors, palette_type=PaletteType.SEQUENTIAL)


# =====================================================================
# PALETTE MANAGER
# =====================================================================

class PaletteManager:
    """Manages palettes and palette operations."""
    
    def __init__(self):
        self._palettes: Dict[str, Palette] = BUILTIN_PALETTES.copy()
        self._custom_palettes: Dict[str, Palette] = {}

    def get_palette(self, name: str) -> Optional[Palette]:
        """Get a palette by name."""
        return self._palettes.get(name) or self._custom_palettes.get(name)

    def add_custom(self, palette: Palette) -> None:
        """Add a custom palette."""
        self._custom_palettes[palette.name] = palette

    def remove_custom(self, name: str) -> bool:
        """Remove a custom palette."""
        if name in self._custom_palettes:
            del self._custom_palettes[name]
            return True
        return False

    def get_names(self) -> List[str]:
        """Get all available palette names."""
        return list(self._palettes.keys()) + list(self._custom_palettes.keys())
        
    def export_palette(self, name: str, path: str) -> bool:
        """Export palette to JSON file."""
        palette = self.get_palette(name)
        if not palette: return False
        try:
            with open(path, 'w') as f: 
                json.dump(palette.to_json(), f, indent=2)
            return True
        except Exception: 
            return False

    def import_palette(self, path: str) -> Tuple[bool, str]:
        """Import palette from JSON file. Returns (success, message)."""
        try:
            with open(path, 'r') as f: 
                data = json.load(f)
            palette = Palette.from_json(data)
            self._custom_palettes[palette.name] = palette
            return True, f"Imported palette: {palette.name}"
        except Exception as e: 
            return False, f"Import failed: {str(e)}"


# =====================================================================
# SPECIALIZED COLORING SYSTEMS
# =====================================================================

class AgeColoringSystem:
    """Colors cells based on their age using scientific colormaps."""
    
    def __init__(self, palette_name: str = "inferno"):
        self.palette_name = palette_name
        self.max_age = 200
        self._lut = self._build_lut()

    def set_max_age(self, max_age: int) -> None:
        self.max_age = max(1, max_age)
        self._lut = self._build_lut()

    def _build_lut(self) -> np.ndarray:
        lut = np.zeros((self.max_age + 1, 3), dtype=np.uint8)
        lut[0] = [0, 0, 0]  # Dead
        for age in range(1, self.max_age + 1):
            t = age / self.max_age
            if self.palette_name == "plasma":
                r = int(255 * min(1, 1.5 * t**0.5))
                g = int(255 * max(0, min(1, 2.5 * t**2 - 0.3)))
                b = int(255 * max(0, min(1, 0.8 * math.sin(math.pi * t * 0.8) + 0.2)))
            elif self.palette_name == "viridis":
                r = int(255 * max(0, min(1, -1.87 * t**2 + 2.14 * t + 0.25)))
                g = int(255 * max(0, min(1, 0.07 * t**3 + 0.65 * t + 0.15)))
                b = int(255 * max(0, min(1, 0.35 * t**2 - 0.45 * t + 0.65)))
            else:  # Default inferno
                r = int(255 * min(1, 1.5 * t**0.5))
                g = int(255 * max(0, min(1, 2.5 * t**2 - 0.3)))
                b = int(255 * max(0, min(1, 0.8 * math.sin(math.pi * t * 0.8) + 0.2)))
            lut[age] = [min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b))]
        return lut

    def get_colors(self, age_grid: np.ndarray) -> np.ndarray:
        clamped = np.clip(age_grid, 0, self.max_age)
        return self._lut[clamped]


class NoiseColoring:
    """Colors cells using procedural noise for texture."""
    
    def __init__(self, scale: float = 0.1, seed: int = 42):
        self.scale = scale
        self.seed = seed
        self._noise_grid = None
        self._size = None

    def _generate_simple_noise(self, rows: int, cols: int) -> np.ndarray:
        """Simple value noise implementation avoiding scipy dependency."""
        rng = np.random.RandomState(self.seed)
        noise_rows = max(2, int(rows * self.scale))
        noise_cols = max(2, int(cols * self.scale))
        
        base = rng.rand(noise_rows, noise_cols)
        
        row_indices = np.linspace(0, noise_rows - 1, rows)
        col_indices = np.linspace(0, noise_cols - 1, cols)
        
        r0 = np.floor(row_indices).astype(int)
        c0 = np.floor(col_indices).astype(int)
        r1 = np.minimum(r0 + 1, noise_rows - 1)
        c1 = np.minimum(c0 + 1, noise_cols - 1)
        
        dr = (row_indices - r0)[:, np.newaxis]
        dc = (col_indices - c0)[np.newaxis, :]
        
        top = base[np.ix_(r0, c0)] * (1 - dc) + base[np.ix_(r0, c1)] * dc
        bottom = base[np.ix_(r1, c0)] * (1 - dc) + base[np.ix_(r1, c1)] * dc
        noise = top * (1 - dr) + bottom * dr
        
        mn, mx = noise.min(), noise.max()
        if mx - mn > 0: 
            noise = (noise - mn) / (mx - mn)
        return noise

    def apply_noise(self, grid: np.ndarray, palette: Palette) -> np.ndarray:
        rows, cols = grid.shape
        if self._size != (rows, cols) or self._noise_grid is None:
            self._noise_grid = self._generate_simple_noise(rows, cols)
            self._size = (rows, cols)
            
        lut = palette.to_lut(256, interpolate=True)
        noise_idx = (self._noise_grid * 255).astype(np.uint8)
        
        result = np.zeros((rows, cols, 3), dtype=np.uint8)
        alive_mask = grid > 0
        result[alive_mask] = lut[noise_idx[alive_mask]]
        return result


class TransitionColorSystem:
    """Handles smooth color blending during state transitions."""
    
    def __init__(self):
        self._old_grid: Optional[np.ndarray] = None
        self._progress = 1.0

    def start_transition(self, old_grid: np.ndarray) -> None:
        self._old_grid = old_grid.copy()
        self._progress = 0.0

    def update(self, dt: float) -> bool:
        self._progress += dt * 6.0
        if self._progress >= 1.0:
            self._progress = 1.0
            return True
        return False

    def get_blended_colors(self, current_grid: np.ndarray, color_lut: np.ndarray) -> np.ndarray:
        if self._old_grid is None or self._progress >= 1.0:
            return color_lut[current_grid]
        
        t = self._progress
        old_colors = color_lut[self._old_grid].astype(np.float32)
        new_colors = color_lut[current_grid].astype(np.float32)
        return (old_colors * (1 - t) + new_colors * t).astype(np.uint8)

# ========================================================
# SECTION: Presets (from presets.py)
# ========================================================

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
    "Pentadecathlon": [
        (0, 1), (1, 1), (2, 0), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 0), (7, 2), (8, 1), (9, 1)
    ],
    "Glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "LWSS": [(0, 1), (0, 4), (1, 0), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3)],
    "MWSS": [(0, 2), (1, 0), (1, 4), (2, 5), (3, 0), (3, 5), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)],
    "HWSS": [(0, 2), (1, 0), (1, 5), (2, 6), (3, 0), (3, 6), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)],
    "R-pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "Diehard": [(0, 6), (1, 0), (1, 1), (2, 1), (2, 5), (2, 6), (2, 7)],
    "Acorn": [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)],
    "B-heptomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (3, 1), (4, 1)],
    "Pi-heptomino": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 4), (2, 1), (2, 2), (2, 3)],
    "Gosper Glider Gun": [
        (0, 24), (1, 22), (1, 24), (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
        (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35), (4, 0), (4, 1), (4, 10),
        (4, 16), (4, 20), (4, 21), (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17),
        (5, 22), (5, 24), (6, 10), (6, 16), (6, 24), (7, 11), (7, 15), (8, 12), (8, 13)
    ],
    "Simkin Glider Gun": [
        (0, 0), (0, 1), (0, 8), (0, 9), (1, 0), (1, 1), (1, 8), (1, 9),
        (2, 2), (2, 3), (2, 6), (2, 7), (5, 2), (5, 3), (5, 6), (5, 7),
        (10, 4), (10, 5), (10, 12), (10, 13), (11, 4), (11, 5), (11, 12), (11, 13),
        (12, 6), (12, 7), (12, 10), (12, 11), (15, 6), (15, 7), (15, 10), (15, 11),
        (20, 8), (20, 9), (21, 8), (21, 9)
    ],
    "R2 Nut": [
        (0, 3), (1, 1), (1, 5), (2, 0), (2, 6), (3, 0), (3, 6),
        (4, 0), (4, 6), (5, 1), (5, 5), (6, 3)
    ],
    "Symmetrical R2": [
        (0, 5), (1, 3), (1, 7), (2, 1), (2, 5), (2, 9), (3, 0), (3, 2),
        (3, 4), (3, 6), (3, 8), (3, 10), (4, 0), (4, 2), (4, 4), (4, 6),
        (4, 8), (4, 10), (5, 1), (5, 5), (5, 9), (6, 3), (6, 7), (7, 5)
    ],
}

PRESET_CATEGORIES: Dict[str, List[str]] = {
    "Clear": ["Clear"], 
    "Still Lifes": ["Block", "Beehive"], 
    "Oscillators": ["Blinker", "Toad", "Beacon", "Pulsar", "Pentadecathlon"],
    "Spaceships": ["Glider", "LWSS", "MWSS", "HWSS"], 
    "Methuselahs": ["R-pentomino", "Diehard", "Acorn", "B-heptomino", "Pi-heptomino"], 
    "Guns": ["Gosper Glider Gun", "Simkin Glider Gun"],
    "Reflector": ["R2 Nut", "Symmetrical R2"],
}

def rotate_pattern(pattern: Pattern, degrees: int) -> Pattern:
    """Rotate a pattern by 0, 90, 180, or 270 degrees."""
    if not pattern or degrees == 0: 
        return pattern.copy()
    if degrees not in [90, 180, 270]:
        raise ValueError("Rotation must be 90, 180, or 270 degrees")
    
    max_r = max(r for r, c in pattern)
    max_c = max(c for r, c in pattern)
    rotated = []
    
    for r, c in pattern:
        if degrees == 90:
            rotated.append((c, max_r - r))
        elif degrees == 180:
            rotated.append((max_r - r, max_c - c))
        elif degrees == 270:
            rotated.append((max_c - c, r))
    
    min_r = min(r for r, c in rotated)
    min_c = min(c for r, c in rotated)
    return [(r - min_r, c - min_c) for r, c in rotated]

def flip_pattern(pattern: Pattern, horizontal: bool = True) -> Pattern:
    """Flip a pattern horizontally or vertically."""
    if not pattern:
        return pattern.copy()
    
    max_r = max(r for r, c in pattern)
    max_c = max(c for r, c in pattern)
    flipped = []
    
    for r, c in pattern:
        if horizontal:
            flipped.append((r, max_c - c))
        else:
            flipped.append((max_r - r, c))
    
    min_r = min(r for r, c in flipped)
    min_c = min(c for r, c in flipped)
    return [(r - min_r, c - min_c) for r, c in flipped]

def parse_rle(rle_text: str) -> Pattern:
    """Parse RLE (Run Length Encoded) pattern format."""
    pattern, row, col, count = [], 0, 0, ''
    lines = [l for l in rle_text.split('\n') if not l.startswith('#')]
    rle_text = ''.join(lines)
    
    # Skip to first pattern character
    for i, char in enumerate(rle_text):
        if char in ('.', 'o', 'b', '$', '!'):
            rle_text = rle_text[i:]
            break
    
    for char in rle_text:
        if char.isdigit():
            count += char
        elif char in ('o', 'b', '.'):
            n = int(count) if count else 1
            count = ''
            if char == 'o':
                for _ in range(n):
                    pattern.append((row, col))
                    col += 1
            else:
                col += n
        elif char == '$':
            n = int(count) if count else 1
            count = ''
            row += n
            col = 0
        elif char == '!':
            break
    
    return pattern

def pattern_to_rle(pattern: Pattern) -> str:
    """Convert a pattern to RLE format."""
    if not pattern:
        return "!"
    
    max_r = max(r for r, c in pattern) + 1
    max_c = max(c for r, c in pattern) + 1
    
    # Create grid
    grid = [[False] * max_c for _ in range(max_r)]
    for r, c in pattern:
        grid[r][c] = True
    
    rle_parts = []
    for r, row in enumerate(grid):
        run = 0
        for c, cell in enumerate(row):
            if cell:
                if run > 0:
                    if run > 1:
                        rle_parts.append(f"{run}b")
                    else:
                        rle_parts.append("b")
                    run = 0
                rle_parts.append("o")
            else:
                run += 1
        if run > 0:
            if run > 1:
                rle_parts.append(f"{run}b")
            else:
                rle_parts.append("b")
        if r < max_r - 1:
            rle_parts.append("$")
    
    rle_parts.append("!")
    return "".join(rle_parts)

# ========================================================
# SECTION: Rulesets (from rulesets.py)
# ========================================================

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

# ========================================================
# SECTION: Backends (from backends.py)
# ========================================================

"""Compute backends for cellular automata evolution."""
import numpy as np
import warnings
import logging

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

logger = logging.getLogger(__name__)


class BackendManager:
    AVAILABLE_BACKENDS = ["Auto", "Python", "NumPy"]
    if HAS_NUMBA: AVAILABLE_BACKENDS.append("Numba")
    if HAS_CUPY: AVAILABLE_BACKENDS.append("CuPy")

    def __init__(self, preferred: str = "Auto"):
        self.current_backend = preferred
        self._evolve_func = None
        self._evolve_func_nowrap = None  # For non-wrapping mode
        self._update_backend()

    def _update_backend(self) -> None:
        """Update the active evolve function based on current backend setting."""
        if self.current_backend == "CuPy" and HAS_CUPY:
            self._evolve_func = _evolve_cupy
            self._evolve_func_nowrap = _evolve_cupy_nowrap
        elif self.current_backend == "Numba" and HAS_NUMBA:
            self._evolve_func = _evolve_numba
            self._evolve_func_nowrap = _evolve_numba_nowrap
        elif self.current_backend == "NumPy":
            self._evolve_func = _evolve_numpy
            self._evolve_func_nowrap = _evolve_numpy_nowrap
        elif self.current_backend == "Auto":
            self._evolve_func = _evolve_numba if HAS_NUMBA else _evolve_numpy
            self._evolve_func_nowrap = _evolve_numba_nowrap if HAS_NUMBA else _evolve_numpy_nowrap
        else:
            self._evolve_func = _evolve_python
            self._evolve_func_nowrap = _evolve_python_nowrap
        logger.debug(f"Backend set to {self.get_effective_backend()}")

    def set_backend(self, backend: str) -> bool:
        """Set the computation backend. Returns True if successful."""
        if backend in self.AVAILABLE_BACKENDS:
            self.current_backend = backend
            self._update_backend()
            return True
        warnings.warn(f"Backend '{backend}' not available. Available: {self.AVAILABLE_BACKENDS}")
        return False

    def evolve(self, grid: np.ndarray, birth: np.ndarray, survive: np.ndarray, wrap: bool = True) -> np.ndarray:
        """Evolve the grid by one generation."""
        if wrap:
            return self._evolve_func(grid, birth, survive)
        else:
            return self._evolve_func_nowrap(grid, birth, survive)

    def get_effective_backend(self) -> str:
        """Get the actual backend being used (resolves 'Auto')."""
        if self.current_backend == "Auto":
            return "Numba" if HAS_NUMBA else "NumPy"
        return self.current_backend


def _evolve_python(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Pure Python implementation with wrapping (slow, for reference/testing)."""
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            alive = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if grid[(r + dr) % rows, (c + dc) % cols] > 0:
                        alive += 1
            if grid[r, c] > 0:
                if alive < len(survive) and survive[alive]:
                    new_grid[r, c] = grid[r, c]
            else:
                if alive < len(birth) and birth[alive]:
                    new_grid[r, c] = 1
    return new_grid


def _evolve_python_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Pure Python implementation without wrapping."""
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            alive = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0:
                        alive += 1
            if grid[r, c] > 0:
                if alive < len(survive) and survive[alive]:
                    new_grid[r, c] = grid[r, c]
            else:
                if alive < len(birth) and birth[alive]:
                    new_grid[r, c] = 1
    return new_grid


def _evolve_numpy(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Vectorized NumPy implementation with wrapping."""
    padded = np.pad(grid, 1, mode='wrap')
    alive = (padded > 0).astype(np.int32)
    neighbors = (alive[:-2, :-2] + alive[:-2, 1:-1] + alive[:-2, 2:] +
                 alive[1:-1, :-2] + alive[1:-1, 2:] +
                 alive[2:, :-2] + alive[2:, 1:-1] + alive[2:, 2:])
    
    is_alive = grid > 0
    
    # Clamp neighbor counts to valid lookup indices
    neighbors_clamped = np.clip(neighbors, 0, len(birth) - 1)
    
    birth_mask = ~is_alive & birth[neighbors_clamped]
    surv_mask = is_alive & survive[neighbors_clamped]
    
    new_grid = np.zeros_like(grid)
    new_grid[birth_mask] = 1
    new_grid[surv_mask] = grid[surv_mask]
    return new_grid


def _evolve_numpy_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Vectorized NumPy implementation without wrapping."""
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    alive = (padded > 0).astype(np.int32)
    neighbors = (alive[:-2, :-2] + alive[:-2, 1:-1] + alive[:-2, 2:] +
                 alive[1:-1, :-2] + alive[1:-1, 2:] +
                 alive[2:, :-2] + alive[2:, 1:-1] + alive[2:, 2:])
    
    is_alive = grid > 0
    neighbors_clamped = np.clip(neighbors, 0, len(birth) - 1)
    
    birth_mask = ~is_alive & birth[neighbors_clamped]
    surv_mask = is_alive & survive[neighbors_clamped]
    
    new_grid = np.zeros_like(grid)
    new_grid[birth_mask] = 1
    new_grid[surv_mask] = grid[surv_mask]
    return new_grid


if HAS_NUMBA:
    @numba.njit(cache=True, parallel=True)
    def _evolve_numba(grid: np.ndarray, birth_lookup: np.ndarray, survive_lookup: np.ndarray) -> np.ndarray:
        """Numba-accelerated implementation with wrapping."""
        rows, cols = grid.shape
        new_grid = np.zeros_like(grid)
        for r in numba.prange(rows):
            for c in range(cols):
                alive = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr = (r + dr) % rows
                        nc = (c + dc) % cols
                        if grid[nr, nc] > 0:
                            alive += 1
                if grid[r, c] > 0:
                    if alive < len(survive_lookup) and survive_lookup[alive]:
                        new_grid[r, c] = grid[r, c]
                else:
                    if alive < len(birth_lookup) and birth_lookup[alive]:
                        new_grid[r, c] = 1
        return new_grid

    @numba.njit(cache=True, parallel=True)
    def _evolve_numba_nowrap(grid: np.ndarray, birth_lookup: np.ndarray, survive_lookup: np.ndarray) -> np.ndarray:
        """Numba-accelerated implementation without wrapping."""
        rows, cols = grid.shape
        new_grid = np.zeros_like(grid)
        for r in numba.prange(rows):
            for c in range(cols):
                alive = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0:
                            alive += 1
                if grid[r, c] > 0:
                    if alive < len(survive_lookup) and survive_lookup[alive]:
                        new_grid[r, c] = grid[r, c]
                else:
                    if alive < len(birth_lookup) and birth_lookup[alive]:
                        new_grid[r, c] = 1
        return new_grid
else:
    def _evolve_numba(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy(grid, birth, survive)
    
    def _evolve_numba_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy_nowrap(grid, birth, survive)


if HAS_CUPY:
    def _evolve_cupy(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        """GPU-accelerated implementation using CuPy with wrapping."""
        if grid_np.size < 10000:
            return _evolve_numpy(grid_np, birth, survive)
        
        try:
            g = cp.asarray(grid_np)
            g_bool = g > 0
            
            # Use padding instead of multiple rolls for efficiency
            padded = cp.pad(g_bool, 1, mode='wrap')
            neighbors = (
                padded[:-2, :-2].astype(cp.int32) + padded[:-2, 1:-1].astype(cp.int32) +
                padded[:-2, 2:].astype(cp.int32) + padded[1:-1, :-2].astype(cp.int32) +
                padded[1:-1, 2:].astype(cp.int32) + padded[2:, :-2].astype(cp.int32) +
                padded[2:, 1:-1].astype(cp.int32) + padded[2:, 2:].astype(cp.int32)
            )
            
            cp_b, cp_s = cp.asarray(birth), cp.asarray(survive)
            neighbors_clamped = cp.clip(neighbors, 0, len(cp_b) - 1)
            
            new_g = cp.zeros_like(g)
            alive_mask = g > 0
            new_g[~alive_mask & cp_b[neighbors_clamped]] = 1
            new_g[alive_mask & cp_s[neighbors_clamped]] = g[alive_mask & cp_s[neighbors_clamped]]
            
            return cp.asnumpy(new_g)
        except Exception as e:
            logger.warning(f"CuPy error, falling back to NumPy: {e}")
            return _evolve_numpy(grid_np, birth, survive)

    def _evolve_cupy_nowrap(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        """GPU-accelerated implementation using CuPy without wrapping."""
        if grid_np.size < 10000:
            return _evolve_numpy_nowrap(grid_np, birth, survive)
        
        try:
            g = cp.asarray(grid_np)
            g_bool = g > 0
            
            padded = cp.pad(g_bool, 1, mode='constant', constant_values=False)
            neighbors = (
                padded[:-2, :-2].astype(cp.int32) + padded[:-2, 1:-1].astype(cp.int32) +
                padded[:-2, 2:].astype(cp.int32) + padded[1:-1, :-2].astype(cp.int32) +
                padded[1:-1, 2:].astype(cp.int32) + padded[2:, :-2].astype(cp.int32) +
                padded[2:, 1:-1].astype(cp.int32) + padded[2:, 2:].astype(cp.int32)
            )
            
            cp_b, cp_s = cp.asarray(birth), cp.asarray(survive)
            neighbors_clamped = cp.clip(neighbors, 0, len(cp_b) - 1)
            
            new_g = cp.zeros_like(g)
            alive_mask = g > 0
            new_g[~alive_mask & cp_b[neighbors_clamped]] = 1
            new_g[alive_mask & cp_s[neighbors_clamped]] = g[alive_mask & cp_s[neighbors_clamped]]
            
            return cp.asnumpy(new_g)
        except Exception as e:
            logger.warning(f"CuPy error, falling back to NumPy: {e}")
            return _evolve_numpy_nowrap(grid_np, birth, survive)
else:
    def _evolve_cupy(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy(grid_np, birth, survive)
    
    def _evolve_cupy_nowrap(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy_nowrap(grid_np, birth, survive)

# ========================================================
# SECTION: Visual Effects (from visual_effects.py)
# ========================================================

"""
Visual effects for cellular automata rendering.

Provides various visual enhancements including:
- Cell age-based coloring
- Glow effects
- Birth/death animations
- Heat map visualization
- Neighbor count visualization
- Smooth transitions
- Various overlay effects
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VisualMode(Enum):
    """Visual rendering modes."""
    STANDARD = "standard"
    AGE = "age"
    HEATMAP = "heatmap"
    NEIGHBOR_COUNT = "neighbor_count"
    OUTLINE = "outline"
    GRADIENT = "gradient"


class EffectIntensity(Enum):
    """Effect intensity levels."""
    NONE = 0
    SUBTLE = 1
    MEDIUM = 2
    STRONG = 3


@dataclass
class VisualSettings:
    """Container for all visual settings."""
    mode: VisualMode = VisualMode.STANDARD
    glow_enabled: bool = False
    glow_intensity: EffectIntensity = EffectIntensity.MEDIUM
    glow_radius: int = 2
    
    birth_flash: bool = False
    birth_flash_color: Tuple[int, int, int] = (255, 255, 255)
    birth_flash_duration: int = 3
    
    death_fade: bool = False
    death_fade_duration: int = 5
    
    age_coloring: bool = False
    max_age_colors: int = 100
    
    heatmap_enabled: bool = False
    heatmap_history_length: int = 100
    
    neighbor_count_enabled: bool = False
    
    outline_mode: bool = False
    outline_color: Tuple[int, int, int] = (255, 255, 255)
    outline_width: int = 1
    
    gradient_enabled: bool = False
    gradient_direction: str = "vertical"  # vertical, horizontal, radial
    
    vignette_enabled: bool = False
    vignette_strength: float = 0.3
    
    cell_border: bool = False
    cell_border_color: Tuple[int, int, int] = (0, 0, 0)
    cell_border_width: int = 1
    
    hover_highlight: bool = True
    hover_highlight_color: Tuple[int, int, int] = (255, 255, 0)
    hover_highlight_alpha: int = 100
    
    crosshair_enabled: bool = False
    crosshair_color: Tuple[int, int, int] = (200, 200, 200)
    
    symmetry_lines: List[str] = field(default_factory=list)  # horizontal, vertical, diagonal


class AgeTracker:
    """Tracks the age of each cell."""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.age_grid = np.zeros((rows, cols), dtype=np.int32)
    
    def reset(self) -> None:
        """Reset all ages to zero."""
        self.age_grid.fill(0)
    
    def update(self, grid: np.ndarray) -> None:
        """Update ages based on current grid state."""
        # Increase age for alive cells
        self.age_grid[grid > 0] += 1
        # Reset age for dead cells
        self.age_grid[grid == 0] = 0
    
    def resize(self, new_rows: int, new_cols: int) -> None:
        """Resize the age tracker."""
        old_age = self.age_grid.copy()
        self.rows = new_rows
        self.cols = new_cols
        self.age_grid = np.zeros((new_rows, new_cols), dtype=np.int32)
        
        min_r = min(old_age.shape[0], new_rows)
        min_c = min(old_age.shape[1], new_cols)
        self.age_grid[:min_r, :min_c] = old_age[:min_r, :min_c]
    
    def get_age_color_lut(self, max_age: int = 100, palette: str = "plasma") -> np.ndarray:
        """Generate age-based color lookup table."""
        max_age = max(1, max_age)
        lut = np.zeros((max_age + 1, 3), dtype=np.uint8)
        lut[0] = [0, 0, 0]  # Dead cells are black
        
        for age in range(1, max_age + 1):
            t = min(age / max_age, 1.0)
            
            if palette == "plasma":
                lut[age] = self._plasma_colormap(t)
            elif palette == "viridis":
                lut[age] = self._viridis_colormap(t)
            elif palette == "inferno":
                lut[age] = self._inferno_colormap(t)
            elif palette == "cool":
                lut[age] = self._cool_colormap(t)
            elif palette == "hot":
                lut[age] = self._hot_colormap(t)
            elif palette == "rainbow":
                lut[age] = self._rainbow_colormap(t)
            elif palette == "amber":
                lut[age] = self._amber_colormap(t)
            else:
                lut[age] = self._plasma_colormap(t)
        
        return lut
    
    @staticmethod
    def _plasma_colormap(t: float) -> Tuple[int, int, int]:
        """Plasma colormap approximation."""
        r = int(255 * (0.05 + 0.95 * (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.0)))))
        g = int(255 * (0.05 + 0.95 * (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.33)))))
        b = int(255 * (0.05 + 0.95 * (0.5 + 0.5 * np.cos(2 * np.pi * (t + 0.67)))))
        return (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))
    
    @staticmethod
    def _viridis_colormap(t: float) -> Tuple[int, int, int]:
        """Viridis colormap approximation."""
        r = int(255 * max(0, min(1, -1.87 * t**2 + 2.14 * t + 0.25)))
        g = int(255 * max(0, min(1, 0.07 * t**3 + 0.65 * t + 0.15)))
        b = int(255 * max(0, min(1, 0.35 * t**2 - 0.45 * t + 0.65)))
        return (r, g, b)
    
    @staticmethod
    def _inferno_colormap(t: float) -> Tuple[int, int, int]:
        """Inferno colormap approximation."""
        r = int(255 * min(1, 1.5 * t**0.5))
        g = int(255 * max(0, min(1, 2.5 * t**2 - 0.3)))
        b = int(255 * max(0, min(1, 0.8 * np.sin(np.pi * t * 0.8) + 0.2)))
        return (r, g, b)
    
    @staticmethod
    def _cool_colormap(t: float) -> Tuple[int, int, int]:
        """Cool colormap (cyan to magenta)."""
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 255
        return (r, g, b)
    
    @staticmethod
    def _hot_colormap(t: float) -> Tuple[int, int, int]:
        """Hot colormap (black to red to yellow to white)."""
        if t < 0.33:
            s = t / 0.33
            return (int(255 * s), 0, 0)
        elif t < 0.66:
            s = (t - 0.33) / 0.33
            return (255, int(255 * s), 0)
        else:
            s = (t - 0.66) / 0.34
            return (255, 255, int(255 * s))
    
    @staticmethod
    def _rainbow_colormap(t: float) -> Tuple[int, int, int]:
        """Rainbow colormap using HSV."""
        h = t * 270
        c = 1.0
        x = c * (1 - abs((h / 60) % 2 - 1))
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        else:
            r, g, b = x, 0, c
        
        return (int(255 * r), int(255 * g), int(255 * b))
    
    @staticmethod
    def _amber_colormap(t: float) -> Tuple[int, int, int]:
        """Amber colormap for retro terminal look."""
        r = int(255 * min(1, 0.2 + 0.8 * t**0.5))
        g = int(255 * min(1, 0.1 + 0.6 * t))
        b = int(255 * min(1, 0.05 * t))
        return (r, g, b)


class HeatmapTracker:
    """Tracks how often cells are alive for heatmap visualization."""
    
    def __init__(self, rows: int, cols: int, history_length: int = 100):
        self.rows = rows
        self.cols = cols
        self.history_length = max(1, history_length)
        self.history: List[np.ndarray] = []
    
    def update(self, grid: np.ndarray) -> None:
        """Add current grid state to history."""
        alive = (grid > 0).astype(np.float32)
        self.history.append(alive)
        
        # Keep only recent history
        while len(self.history) > self.history_length:
            self.history.pop(0)
    
    def get_heatmap(self) -> np.ndarray:
        """Get normalized heatmap (0-1) of cell activity."""
        if not self.history:
            return np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # Sum all history and normalize
        heatmap = np.sum(self.history, axis=0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        return heatmap
    
    def get_heatmap_color_lut(self) -> np.ndarray:
        """Generate heatmap color lookup table (256 entries)."""
        lut = np.zeros((256, 3), dtype=np.uint8)
        
        for i in range(256):
            t = i / 255.0
            if t < 0.25:
                s = t / 0.25
                lut[i] = [0, int(255 * s), 255]
            elif t < 0.5:
                s = (t - 0.25) / 0.25
                lut[i] = [0, 255, int(255 * (1 - s))]
            elif t < 0.75:
                s = (t - 0.5) / 0.25
                lut[i] = [int(255 * s), 255, 0]
            else:
                s = (t - 0.75) / 0.25
                lut[i] = [255, int(255 * (1 - s)), 0]
        
        return lut
    
    def resize(self, new_rows: int, new_cols: int) -> None:
        """Resize the heatmap tracker."""
        self.rows = new_rows
        self.cols = new_cols
        self.history.clear()
    
    def reset(self) -> None:
        """Clear history."""
        self.history.clear()
    
    def set_history_length(self, length: int) -> None:
        """Set maximum history length."""
        self.history_length = max(1, length)
        while len(self.history) > self.history_length:
            self.history.pop(0)


class BirthDeathTracker:
    """Tracks recent births and deaths for flash effects."""
    
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.birth_timers = np.zeros((rows, cols), dtype=np.int32)
        self.death_timers = np.zeros((rows, cols), dtype=np.int32)
        self._birth_flash_duration = 3
        self._death_fade_duration = 5
    
    def set_durations(self, birth_duration: int, death_duration: int) -> None:
        """Set flash/fade durations."""
        self._birth_flash_duration = max(1, birth_duration)
        self._death_fade_duration = max(1, death_duration)
    
    def update(self, old_grid: np.ndarray, new_grid: np.ndarray) -> None:
        """Update birth/death trackers based on grid changes."""
        # Detect births (dead -> alive)
        births = (old_grid == 0) & (new_grid > 0)
        self.birth_timers[births] = self._birth_flash_duration
        
        # Detect deaths (alive -> dead)
        deaths = (old_grid > 0) & (new_grid == 0)
        self.death_timers[deaths] = self._death_fade_duration
        
        # Decay timers
        self.birth_timers = np.maximum(0, self.birth_timers - 1)
        self.death_timers = np.maximum(0, self.death_timers - 1)
    
    def get_birth_overlay(self, color: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Get RGBA overlay for birth flash effect."""
        if np.max(self.birth_timers) == 0:
            return None
        
        overlay = np.zeros((self.rows, self.cols, 4), dtype=np.uint8)
        mask = self.birth_timers > 0
        alpha = (self.birth_timers / self._birth_flash_duration * 200).astype(np.uint8)
        
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = alpha[mask]
        
        return overlay
    
    def get_death_overlay(self, base_color: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Get RGBA overlay for death fade effect."""
        if np.max(self.death_timers) == 0:
            return None
        
        overlay = np.zeros((self.rows, self.cols, 4), dtype=np.uint8)
        mask = self.death_timers > 0
        alpha = (self.death_timers / self._death_fade_duration * 150).astype(np.uint8)
        
        overlay[mask, 0] = base_color[0]
        overlay[mask, 1] = base_color[1]
        overlay[mask, 2] = base_color[2]
        overlay[mask, 3] = alpha[mask]
        
        return overlay
    
    def resize(self, new_rows: int, new_cols: int) -> None:
        """Resize trackers."""
        self.rows = new_rows
        self.cols = new_cols
        self.birth_timers = np.zeros((new_rows, new_cols), dtype=np.int32)
        self.death_timers = np.zeros((new_rows, new_cols), dtype=np.int32)
    
    def reset(self) -> None:
        """Clear all timers."""
        self.birth_timers.fill(0)
        self.death_timers.fill(0)


class GlowEffect:
    """Applies glow effect to alive cells."""
    
    def __init__(self, radius: int = 2, intensity: float = 0.5):
        self.radius = radius
        self.intensity = intensity
        self._kernel: Optional[np.ndarray] = None
        self._has_scipy = False
        self._check_scipy()
        self._update_kernel()
    
    def _check_scipy(self) -> None:
        """Check if scipy is available."""
        try:
            from scipy import ndimage
            self._has_scipy = True
        except ImportError:
            self._has_scipy = False
            logger.debug("scipy not available, glow effect disabled")
    
    def _update_kernel(self) -> None:
        """Create glow kernel."""
        size = self.radius * 2 + 1
        self._kernel = np.zeros((size, size), dtype=np.float32)
        center = self.radius
        
        for r in range(size):
            for c in range(size):
                dist = np.sqrt((r - center) ** 2 + (c - center) ** 2)
                if dist <= self.radius:
                    self._kernel[r, c] = (1 - dist / self.radius) * self.intensity
    
    def apply(self, grid: np.ndarray, color_lut: np.ndarray) -> np.ndarray:
        """Apply glow effect to grid."""
        if not self._has_scipy or np.sum(grid > 0) == 0:
            return color_lut[grid]
        
        try:
            from scipy import ndimage
            
            alive = (grid > 0).astype(np.float32)
            glow = ndimage.convolve(alive, self._kernel, mode='wrap')
            
            # Get base colors
            rgb = color_lut[grid].astype(np.float32)
            
            # Get glow color (average of non-zero colors)
            nonzero_colors = color_lut[1:]
            if len(nonzero_colors) > 0:
                glow_color = np.mean(nonzero_colors, axis=0).astype(np.float32)
            else:
                glow_color = np.array([255, 255, 255], dtype=np.float32)
            
            # Add glow
            for i in range(3):
                rgb[:, :, i] = np.clip(
                    rgb[:, :, i] + glow * glow_color[i], 
                    0, 255
                )
            
            return rgb.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Glow effect error: {e}")
            return color_lut[grid]
    
    def set_radius(self, radius: int) -> None:
        """Set glow radius."""
        self.radius = max(1, min(10, radius))
        self._update_kernel()
    
    def set_intensity(self, intensity: float) -> None:
        """Set glow intensity."""
        self.intensity = max(0.0, min(1.0, intensity))
        self._update_kernel()
    
    @property
    def is_available(self) -> bool:
        """Check if glow effect is available."""
        return self._has_scipy


class OutlineRenderer:
    """Renders cells with outlines only."""
    
    def __init__(self, outline_color: Tuple[int, int, int] = (255, 255, 255)):
        self.outline_color = outline_color
    
    def render(self, grid: np.ndarray, background_color: Tuple[int, int, int]) -> np.ndarray:
        """Render grid with outlines only - vectorized version."""
        rows, cols = grid.shape
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        rgb[:] = background_color
        
        alive = grid > 0
        
        # Find edges using neighbor comparison (vectorized)
        padded = np.pad(alive, 1, mode='wrap').astype(np.int32)
        neighbor_sum = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                       padded[1:-1, :-2] + padded[1:-1, 2:] +
                       padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:])
        
        # Edge = alive cell with at least one dead neighbor
        is_edge = alive & (neighbor_sum < 8)
        rgb[is_edge] = self.outline_color
        
        return rgb


class NeighborCountVisualizer:
    """Visualizes neighbor counts for alive cells."""
    
    def __init__(self):
        # Colors for each neighbor count (0-8)
        self.count_colors = [
            (50, 50, 50),     # 0
            (30, 30, 200),    # 1
            (0, 100, 255),    # 2
            (0, 200, 200),    # 3
            (0, 200, 0),      # 4
            (200, 200, 0),    # 5
            (255, 150, 0),    # 6
            (255, 50, 0),     # 7
            (255, 0, 0),      # 8
        ]
    
    def get_neighbor_counts(self, grid: np.ndarray) -> np.ndarray:
        """Calculate neighbor counts for all cells."""
        alive = (grid > 0).astype(np.int32)
        
        counts = (
            np.roll(alive, 1, axis=0) + np.roll(alive, -1, axis=0) +
            np.roll(alive, 1, axis=1) + np.roll(alive, -1, axis=1) +
            np.roll(np.roll(alive, 1, axis=0), 1, axis=1) +
            np.roll(np.roll(alive, 1, axis=0), -1, axis=1) +
            np.roll(np.roll(alive, -1, axis=0), 1, axis=1) +
            np.roll(np.roll(alive, -1, axis=0), -1, axis=1)
        )
        
        return counts
    
    def render(self, grid: np.ndarray) -> np.ndarray:
        """Render grid with neighbor count colors - vectorized version."""
        counts = self.get_neighbor_counts(grid)
        rows, cols = grid.shape
        rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
        
        # Vectorized color assignment
        for count, color in enumerate(self.count_colors):
            mask = (counts == count) & (grid > 0)
            rgb[mask] = color
        
        return rgb


class VignetteEffect:
    """Applies vignette effect to the rendered image."""
    
    def __init__(self, strength: float = 0.3):
        self.strength = strength
        self._vignette_mask: Optional[np.ndarray] = None
        self._last_size: Optional[Tuple[int, int]] = None
    
    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply vignette effect."""
        h, w, _ = rgb.shape
        
        # Generate mask if size changed
        if self._vignette_mask is None or self._last_size != (h, w):
            self._generate_mask(h, w)
            self._last_size = (h, w)
        
        # Apply mask
        result = rgb.astype(np.float32) * self._vignette_mask[:, :, np.newaxis]
        return result.astype(np.uint8)
    
    def _generate_mask(self, h: int, w: int) -> None:
        """Generate vignette mask."""
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        dist = dist / max_dist
        
        self._vignette_mask = 1 - self.strength * dist ** 2
        self._vignette_mask = np.clip(self._vignette_mask, 0, 1).astype(np.float32)
    
    def set_strength(self, strength: float) -> None:
        """Set vignette strength."""
        self.strength = max(0.0, min(1.0, strength))
        self._vignette_mask = None  # Force regeneration


class GradientOverlay:
    """Applies gradient overlay to the grid."""
    
    def __init__(self, direction: str = "vertical"):
        self.direction = direction
    
    def apply(self, rgb: np.ndarray, color1: Tuple[int, int, int], 
              color2: Tuple[int, int, int], alpha: float = 0.2) -> np.ndarray:
        """Apply gradient overlay."""
        h, w, _ = rgb.shape
        
        if self.direction == "vertical":
            t = np.linspace(0, 1, h)[:, np.newaxis]
            t = np.broadcast_to(t, (h, w))
        elif self.direction == "horizontal":
            t = np.linspace(0, 1, w)[np.newaxis, :]
            t = np.broadcast_to(t, (h, w))
        else:  # radial
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - w/2) ** 2 + (y - h/2) ** 2)
            max_dist = np.sqrt((w/2) ** 2 + (h/2) ** 2)
            t = dist / max_dist
        
        gradient = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(3):
            gradient[:, :, i] = color1[i] * (1 - t) + color2[i] * t
        
        result = rgb.astype(np.float32) * (1 - alpha) + gradient * alpha
        return result.astype(np.uint8)

# ========================================================
# SECTION: Themes (from themes.py)
# ========================================================

"""Color themes and Qt styling."""
import numpy as np
import logging

logger = logging.getLogger(__name__)

THEMES = {
    "light": {
        "bg": (240, 240, 240), 
        "grid": (220, 220, 220), 
        "states": [(100,149,237), (34,139,34), (220,20,60), (255,165,0), (148,0,211)]
    },
    "dark": {
        "bg": (30, 30, 30), 
        "grid": (50, 50, 50),
        "states": [(0,255,127), (0,191,255), (255,0,127), (255,215,0), (138,43,226)]
    },
    "matrix": {
        "bg": (0, 10, 0), 
        "grid": (0, 30, 0),
        "states": [(0,255,0), (0,200,0), (50,255,50), (0,150,0), (100,255,100)]
    },
    "ocean": {
        "bg": (10, 25, 50), 
        "grid": (20, 40, 70),
        "states": [(0,150,255), (0,200,200), (100,200,255), (0,255,200), (50,100,255)]
    },
    "cyberpunk": {
        "bg": (20, 10, 30), 
        "grid": (40, 20, 50),
        "states": [(255,0,110), (0,255,255), (255,234,0), (131,56,236), (255,100,200)]
    },
}

def get_theme_lut(theme_name: str, max_state: int) -> np.ndarray:
    theme = THEMES.get(theme_name, THEMES["dark"])
    lut = np.zeros((max_state, 3), dtype=np.uint8)
    lut[0] = theme["bg"]
    for i in range(1, max_state):
        lut[i] = theme["states"][(i - 1) % len(theme["states"])]
    return lut

def get_grid_color(theme_name: str) -> tuple:
    return THEMES.get(theme_name, THEMES["dark"])["grid"]

def get_background_color(theme_name: str) -> tuple:
    return THEMES.get(theme_name, THEMES["dark"])["bg"]

def apply_theme(theme_name: str) -> None:
    """Apply Qt stylesheet based on theme."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if not app:
        return
    
    is_dark = theme_name != "light"
    
    if is_dark:
        if theme_name == "matrix":
            accent = "#00ff00"
            accent_dark = "#003300"
        elif theme_name == "ocean":
            accent = "#00aaff"
            accent_dark = "#001a33"
        elif theme_name == "cyberpunk":
            accent = "#ff006e"
            accent_dark = "#140a1e"
        else:
            accent = "#00ff7f"
            accent_dark = "#1e1e1e"
        
        app.setStyleSheet(f"""
            QWidget {{ 
                background-color: {accent_dark}; 
                color: #ddd; 
                font-size: 12px; 
            }}
            QGroupBox {{ 
                border: 1px solid #444; 
                border-radius: 4px; 
                margin-top: 8px; 
                padding-top: 16px; 
                font-weight: bold; 
            }}
            QGroupBox::title {{ 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
                color: {accent};
            }}
            QPushButton {{ 
                background-color: #333; 
                border: 1px solid #555; 
                border-radius: 4px; 
                padding: 5px 10px; 
                min-height: 20px;
            }}
            QPushButton:hover {{ 
                background-color: #444; 
                border-color: {accent};
            }}
            QPushButton:pressed {{ 
                background-color: {accent_dark}; 
            }}
            QComboBox {{ 
                background-color: #2a2a2a; 
                border: 1px solid #555; 
                padding: 3px; 
                min-height: 20px;
            }}
            QComboBox::drop-down {{ 
                border: none; 
            }}
            QComboBox QAbstractItemView {{
                background-color: #2a2a2a;
                selection-background-color: #444;
                border: 1px solid #555;
            }}
            QSpinBox, QDoubleSpinBox {{ 
                background-color: #2a2a2a; 
                border: 1px solid #555; 
                padding: 3px;
                min-height: 20px;
            }}
            QTextEdit {{ 
                background-color: #222; 
                border: 1px solid #555; 
                font-family: monospace;
                font-size: 11px;
            }}
            QSlider::groove:horizontal {{ 
                border: 1px solid #555; 
                height: 6px; 
                background: #333; 
                border-radius: 3px; 
            }}
            QSlider::handle:horizontal {{ 
                background: {accent}; 
                width: 14px; 
                margin: -5px 0; 
                border-radius: 7px; 
            }}
            QSlider::sub-page:horizontal {{
                background: {accent};
                border-radius: 3px;
            }}
            QScrollArea {{ 
                border: none; 
            }}
            QTabWidget::pane {{
                border: 1px solid #444;
            }}
            QTabBar::tab {{
                background-color: #2a2a2a;
                border: 1px solid #444;
                padding: 6px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: #333;
                border-bottom-color: #333;
                color: {accent};
            }}
            QTabBar::tab:hover {{
                background-color: #3a3a3a;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2a2a2a;
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent};
                border-color: {accent};
            }}
            QStatusBar {{
                background-color: #1a1a1a;
                color: #888;
            }}
            QLabel {{
                background: transparent;
            }}
        """)
    else:
        app.setStyleSheet("""
            QWidget { 
                background-color: #f0f0f0; 
                font-size: 12px; 
            }
            QGroupBox { 
                border: 1px solid #bbb; 
                border-radius: 4px; 
                margin-top: 8px; 
                padding-top: 16px; 
                font-weight: bold; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
            }
            QPushButton { 
                background-color: #e0e0e0; 
                border: 1px solid #aaa; 
                border-radius: 4px; 
                padding: 5px 10px;
                min-height: 20px;
            }
            QPushButton:hover { 
                background-color: #d0d0d0; 
            }
            QPushButton:pressed { 
                background-color: #c0c0c0; 
            }
            QComboBox { 
                background-color: white; 
                border: 1px solid #aaa; 
                padding: 3px;
                min-height: 20px;
            }
            QSpinBox, QDoubleSpinBox { 
                background-color: white; 
                border: 1px solid #aaa; 
                padding: 3px;
                min-height: 20px;
            }
            QTextEdit { 
                background-color: white; 
                border: 1px solid #aaa;
                font-family: monospace;
                font-size: 11px;
            }
            QSlider::groove:horizontal { 
                border: 1px solid #aaa; 
                height: 6px; 
                background: #ddd; 
                border-radius: 3px; 
            }
            QSlider::handle:horizontal { 
                background: #666; 
                width: 14px; 
                margin: -5px 0; 
                border-radius: 7px; 
            }
            QScrollArea { 
                border: none; 
            }
            QTabWidget::pane {
                border: 1px solid #bbb;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #bbb;
                padding: 6px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #f0f0f0;
                border-bottom-color: #f0f0f0;
            }
            QStatusBar {
                background-color: #e0e0e0;
                color: #666;
            }
            QLabel {
                background: transparent;
            }
        """)
    
    logger.debug(f"Applied theme: {theme_name}")

# ========================================================
# SECTION: Widgets (from widgets.py)
# ========================================================

"""Cellular automata rendering widget."""
import json
import random
import time
import logging
from typing import Optional, Tuple, List
from collections import deque

import numpy as np

from PySide6.QtWidgets import QWidget, QApplication, QToolTip
from PySide6.QtGui import QPainter, QColor, QImage, QPen
from PySide6.QtCore import QTimer, Qt, Signal, QPointF


logger = logging.getLogger(__name__)


class UndoStack:
    """Simple undo/redo stack for grid states."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self._undo: deque = deque(maxlen=max_size)
        self._redo: deque = deque(maxlen=max_size)
    
    def push(self, state: np.ndarray) -> None:
        self._undo.append(state.copy())
        self._redo.clear()
    
    def undo(self, current: np.ndarray) -> Optional[np.ndarray]:
        if not self._undo:
            return None
        self._redo.append(current.copy())
        return self._undo.pop()
    
    def redo(self, current: np.ndarray) -> Optional[np.ndarray]:
        if not self._redo:
            return None
        self._undo.append(current.copy())
        return self._redo.pop()
    
    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
    
    @property
    def undo_count(self) -> int:
        return len(self._undo)
    
    @property
    def redo_count(self) -> int:
        return len(self._redo)


class CellularAutomataWidget(QWidget):
    """Main cellular automata rendering and simulation widget."""
    
    generation_updated = Signal(int)
    population_updated = Signal(int)
    fps_updated = Signal(float)
    cell_hovered = Signal(int, int, int)  # row, col, state

    def __init__(self, rows: int = 150, cols: int = 150, cell_size: int = 5, 
                 max_state: int = 16, rule: str = "B3/S23", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.max_state = max_state
        self.zoom = 1.0
        
        self.grid = np.zeros((rows, cols), dtype=np.int32)
        self.initial_grid = np.zeros((rows, cols), dtype=np.int32)
        self.generation = 0
        self.undo_stack = UndoStack()
        
        self.backend_manager = BackendManager("Auto")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        
        self._frame_times: deque = deque(maxlen=30)
        self._last_time = time.perf_counter()
        self._steps_per_frame = 1
        
        # Rule
        self.rule_string = rule
        self.birth_lookup = np.zeros(9, dtype=np.bool_)
        self.survive_lookup = np.zeros(9, dtype=np.bool_)
        self._parse_rule(rule)
        
        # Wrap mode
        self.wrap_mode = True
        
        # Mouse state
        self._dragging = False
        self._panning = False
        self._pan_start: Optional[QPointF] = None
        self._last_btn: Optional[Qt.MouseButton] = None
        self._last_cell: Optional[Tuple[int, int]] = None
        self._draw_state: Optional[int] = None
        self._hover_cell: Optional[Tuple[int, int]] = None
        
        # Symmetry drawing
        self.symmetry_mode = "none"  # none, horizontal, vertical, both, rotational
        
        # Visual settings
        self.current_theme = "dark"
        self.palette_manager = PaletteManager()
        self.current_palette_name = "Standard"
        self.show_grid_lines = False
        
        # Visual modes
        self.visual_mode = "Standard"
        self.trail_enabled = False
        self.trail_grid: Optional[np.ndarray] = None
        self.trail_length = 15
        self.glow_enabled = False
        self.vignette_enabled = False
        self.birth_death_enabled = False
        
        # Visual effect trackers
        self._age_tracker = AgeTracker(rows, cols)
        self._heatmap_tracker = HeatmapTracker(rows, cols)
        self._vignette = VignetteEffect(0.3)
        self._glow = GlowEffect(2, 0.5)
        self._outline_renderer = OutlineRenderer()
        self._neighbor_visualizer = NeighborCountVisualizer()
        self._birth_death_tracker = BirthDeathTracker(rows, cols)
        
        # Image reference to prevent GC during paint
        self._paint_image: Optional[QImage] = None
        self._paint_rgb: Optional[np.ndarray] = None
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_size()

    def _update_size(self) -> None:
        w = int(self.cols * self.cell_size * self.zoom)
        h = int(self.rows * self.cell_size * self.zoom)
        self.setFixedSize(w, h)

    def _parse_rule(self, rule: str) -> None:
        self.rule_string = rule
        self.birth_lookup[:] = False
        self.survive_lookup[:] = False
        if "/" not in rule:
            return
        try:
            parts = rule.upper().split("/")
            b_part = parts[0].replace("B", "")
            s_part = parts[1].replace("S", "") if len(parts) > 1 else ""
            
            for c in b_part:
                if c.isdigit():
                    n = int(c)
                    if 0 <= n <= 8:
                        self.birth_lookup[n] = True
            for c in s_part:
                if c.isdigit():
                    n = int(c)
                    if 0 <= n <= 8:
                        self.survive_lookup[n] = True
        except Exception as e:
            logger.warning(f"Rule parse error: {e}")

    def set_rule(self, rule: str) -> None:
        self._parse_rule(rule)
    
    def set_theme(self, theme: str) -> None:
        self.current_theme = theme
    
    def set_palette(self, name: str) -> None:
        self.current_palette_name = name
    
    def set_visual_mode(self, mode: str) -> None:
        self.visual_mode = mode
        if mode == "Age":
            self._age_tracker.reset()
        elif mode == "Heatmap":
            self._heatmap_tracker.reset()
        self.update()

    def set_wrap_mode(self, enabled: bool) -> None:
        self.wrap_mode = enabled
    
    def set_symmetry(self, mode: str) -> None:
        self.symmetry_mode = mode
    
    def set_steps_per_frame(self, steps: int) -> None:
        self._steps_per_frame = max(1, min(100, steps))

    def enable_trail(self, length: int = 15) -> None:
        self.trail_enabled = True
        self.trail_length = max(1, min(50, length))
        if self.trail_grid is None or self.trail_grid.shape != (self.rows, self.cols):
            self.trail_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
    
    def disable_trail(self) -> None:
        self.trail_enabled = False
        self.trail_grid = None

    def _apply_symmetry(self, r: int, c: int, state: int) -> List[Tuple[int, int]]:
        """Get all cells that should be drawn due to symmetry."""
        cells = [(r, c)]
        
        if self.symmetry_mode == "horizontal":
            cells.append((r, self.cols - 1 - c))
        elif self.symmetry_mode == "vertical":
            cells.append((self.rows - 1 - r, c))
        elif self.symmetry_mode == "both":
            cells.append((r, self.cols - 1 - c))
            cells.append((self.rows - 1 - r, c))
            cells.append((self.rows - 1 - r, self.cols - 1 - c))
        elif self.symmetry_mode == "rotational":
            cells.append((self.rows - 1 - r, self.cols - 1 - c))
        
        # Filter to valid cells
        return [(rr, cc) for rr, cc in cells if 0 <= rr < self.rows and 0 <= cc < self.cols]

    def _tick(self) -> None:
        """Timer tick - run steps_per_frame generations."""
        for _ in range(self._steps_per_frame):
            self._do_generation()

    def _do_generation(self) -> None:
        """Perform one generation step."""
        if self.trail_enabled and self.trail_grid is not None:
            self.trail_grid = np.maximum(self.trail_grid, self.grid)
            self.trail_grid = np.where(self.trail_grid > 0, self.trail_grid - 1, 0)
            self.trail_grid[self.grid > 0] = self.trail_length
        
        old_grid = self.grid.copy()
        self.grid = self.backend_manager.evolve(
            self.grid, self.birth_lookup, self.survive_lookup, self.wrap_mode
        )
        
        if self.birth_death_enabled:
            self._birth_death_tracker.update(old_grid, self.grid)
        
        self.generation += 1
        self._age_tracker.update(self.grid)
        self._heatmap_tracker.update(self.grid)
        
        now = time.perf_counter()
        dt = now - self._last_time
        self._frame_times.append(dt)
        self._last_time = now
        
        if len(self._frame_times) >= 5:
            avg_time = sum(self._frame_times) / len(self._frame_times)
            if avg_time > 0:
                self.fps_updated.emit(1.0 / avg_time)
        
        self.generation_updated.emit(self.generation)
        self.population_updated.emit(int(np.sum(self.grid > 0)))
        self.update()

    def next_generation(self) -> None:
        """Public method to advance one generation."""
        self._do_generation()

    def set_speed(self, ms: int) -> None:
        self.timer.setInterval(max(1, ms))
    
    def start(self) -> None:
        self._last_time = time.perf_counter()
        self._frame_times.clear()
        self.timer.start(50)
    
    def stop(self) -> None:
        self.timer.stop()
        self.fps_updated.emit(0)
    
    def is_running(self) -> bool:
        return self.timer.isActive()
    
    def step(self) -> None:
        if not self.timer.isActive():
            self._do_generation()

    def save_undo(self) -> None:
        self.undo_stack.push(self.grid)
    
    def undo(self) -> None:
        s = self.undo_stack.undo(self.grid)
        if s is not None:
            self.grid = s
            self.update()
            self.population_updated.emit(int(np.sum(self.grid > 0)))
    
    def redo(self) -> None:
        s = self.undo_stack.redo(self.grid)
        if s is not None:
            self.grid = s
            self.update()
            self.population_updated.emit(int(np.sum(self.grid > 0)))

    def clear_grid(self, save: bool = True) -> None:
        if save:
            self.save_undo()
        self.grid.fill(0)
        self.generation = 0
        if self.trail_grid is not None:
            self.trail_grid.fill(0)
        self._age_tracker.reset()
        self._heatmap_tracker.reset()
        self._birth_death_tracker.reset()
        self.generation_updated.emit(0)
        self.population_updated.emit(0)
        self.update()

    def randomize(self, density: float = 0.3) -> None:
        self.save_undo()
        self.grid = (np.random.random((self.rows, self.cols)) < density).astype(np.int32)
        self.generation = 0
        self._age_tracker.reset()
        self._heatmap_tracker.reset()
        self._birth_death_tracker.reset()
        self.generation_updated.emit(0)
        self.population_updated.emit(int(np.sum(self.grid > 0)))
        self.update()

    def reset_to_initial(self) -> None:
        """Reset grid to the saved initial state."""
        self.save_undo()
        self.grid = self.initial_grid.copy()
        self.generation = 0
        if self.trail_grid is not None:
            self.trail_grid.fill(0)
        self._age_tracker.reset()
        self._heatmap_tracker.reset()
        self._birth_death_tracker.reset()
        self.generation_updated.emit(0)
        self.population_updated.emit(int(np.sum(self.grid > 0)))
        self.update()

    def resize_grid(self, r: int, c: int, preserve: bool = False) -> None:
        old = self.grid.copy() if preserve else None
        self.rows, self.cols = r, c
        self.grid = np.zeros((r, c), dtype=np.int32)
        self.initial_grid = np.zeros((r, c), dtype=np.int32)
        if preserve and old is not None:
            mr, mc = min(old.shape[0], r), min(old.shape[1], c)
            self.grid[:mr, :mc] = old[:mr, :mc]
        if self.trail_enabled:
            self.trail_grid = np.zeros((r, c), dtype=np.int32)
        self._age_tracker.resize(r, c)
        self._heatmap_tracker.resize(r, c)
        self._birth_death_tracker.resize(r, c)
        self.undo_stack = UndoStack()
        self.generation = 0
        self._update_size()
        self.update()

    def set_cell_size(self, s: int) -> None:
        self.cell_size = max(1, s)
        self._update_size()
        self.update()
    
    def set_zoom(self, z: float) -> None:
        self.zoom = max(0.1, min(10.0, z))
        self._update_size()
        self.update()
    
    def set_max_state(self, m: int) -> None:
        self.grid = np.clip(self.grid, 0, m - 1)
        self.max_state = m
        self.update()

    def inject_pattern(self, name: str, center: bool = True, save: bool = True, 
                       rotation: int = 0, flip_h: bool = False, flip_v: bool = False) -> bool:
        p = PRESETS.get(name)
        if p is None:
            return False
        if not p:
            self.clear_grid(save)
            return True
        if save:
            self.save_undo()
        
        # Apply transformations
        transformed = p.copy()
        if flip_h:
            transformed = flip_pattern(transformed, horizontal=True)
        if flip_v:
            transformed = flip_pattern(transformed, horizontal=False)
        if rotation != 0:
            transformed = rotate_pattern(transformed, rotation)
        
        if not transformed:
            return True
        
        mr = max(r for r, c in transformed)
        mc = max(c for r, c in transformed)
        sr = (self.rows - mr - 1) // 2 if center else 0
        sc = (self.cols - mc - 1) // 2 if center else 0
        
        for dr, dc in transformed:
            r, c = sr + dr, sc + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = 1
                # Apply symmetry if enabled
                if self.symmetry_mode != "none":
                    for sr2, sc2 in self._apply_symmetry(r, c, 1)[1:]:
                        self.grid[sr2, sc2] = 1
        
        self.population_updated.emit(int(np.sum(self.grid > 0)))
        self.update()
        return True

    def inject_pattern_at(self, name: str, row: int, col: int, 
                          rotation: int = 0, flip_h: bool = False, flip_v: bool = False) -> bool:
        """Inject pattern at specific position."""
        p = PRESETS.get(name)
        if p is None:
            return False
        if not p:
            return True
        
        self.save_undo()
        
        transformed = p.copy()
        if flip_h:
            transformed = flip_pattern(transformed, horizontal=True)
        if flip_v:
            transformed = flip_pattern(transformed, horizontal=False)
        if rotation != 0:
            transformed = rotate_pattern(transformed, rotation)
        
        for dr, dc in transformed:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = 1
        
        self.population_updated.emit(int(np.sum(self.grid > 0)))
        self.update()
        return True

    def import_rle(self, rle_text: str, center: bool = True) -> bool:
        """Import pattern from RLE string."""
        try:
            pattern = parse_rle(rle_text)
            if not pattern:
                return False
            self.save_undo()
            mr = max(r for r, c in pattern)
            mc = max(c for r, c in pattern)
            sr = (self.rows - mr - 1) // 2 if center else 0
            sc = (self.cols - mc - 1) // 2 if center else 0
            for dr, dc in pattern:
                r, c = sr + dr, sc + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.grid[r, c] = 1
            self.population_updated.emit(int(np.sum(self.grid > 0)))
            self.update()
            return True
        except Exception as e:
            logger.error(f"RLE import failed: {e}")
            return False

    def save_to_file(self, path: str, code: str = "") -> None:
        data = {
            "v": "2.1", 
            "grid": self.grid.tolist(), 
            "rows": self.rows, 
            "cols": self.cols, 
            "rule": self.rule_string, 
            "gen": self.generation,
            "wrap": self.wrap_mode,
            "custom_code": code
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_from_file(self, path: str) -> Tuple[bool, dict]:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.grid = np.array(data["grid"], dtype=np.int32)
            self.rows, self.cols = self.grid.shape
            self.rule_string = data.get("rule", "B3/S23")
            self._parse_rule(self.rule_string)
            self.generation = data.get("gen", 0)
            self.wrap_mode = data.get("wrap", True)
            self.initial_grid = self.grid.copy()
            self._age_tracker.resize(self.rows, self.cols)
            self._heatmap_tracker.resize(self.rows, self.cols)
            self._birth_death_tracker.resize(self.rows, self.cols)
            if self.trail_enabled:
                self.trail_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
            self._update_size()
            self.update()
            return True, data
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return False, {"error": str(e)}

    def export_to_png(self, path: str) -> Tuple[bool, str]:
        """Export current grid to PNG. Returns (success, message)."""
        try:
            from PIL import Image
            render_grid = np.maximum(self.grid, self.trail_grid) if self.trail_enabled and self.trail_grid is not None else self.grid
            rgb = self._get_render_rgb(render_grid)
            img = Image.fromarray(rgb, 'RGB')
            img.save(path)
            return True, f"Exported to {path}"
        except ImportError:
            return False, "Pillow is required for PNG export. Install with: pip install Pillow"
        except Exception as e:
            return False, f"Export failed: {str(e)}"

    def _get_lut(self) -> np.ndarray:
        """Get the color lookup table for current visual mode."""
        if self.visual_mode == "Age":
            return self._age_tracker.get_age_color_lut(200, "inferno")
        if self.visual_mode == "Heatmap":
            return self._heatmap_tracker.get_heatmap_color_lut()
        # Standard / Trail mode
        palette = self.palette_manager.get_palette(self.current_palette_name)
        if palette is None:
            palette = self.palette_manager.get_palette("Standard")
        return palette.to_lut(self.max_state) if palette else get_theme_lut(self.current_theme, self.max_state)

    def _get_render_rgb(self, render_grid: np.ndarray) -> np.ndarray:
        """Get RGB array for rendering."""
        # Special modes
        if self.visual_mode == "Outline":
            bg = get_background_color(self.current_theme)
            return self._outline_renderer.render(render_grid, bg)
        
        if self.visual_mode == "Neighbor Count":
            return self._neighbor_visualizer.render(render_grid)
        
        lut = self._get_lut()
        
        if self.visual_mode == "Age":
            max_age = lut.shape[0] - 1
            rgb = lut[np.clip(self._age_tracker.age_grid, 0, max_age)]
            rgb[render_grid == 0] = lut[0]
        elif self.visual_mode == "Heatmap":
            hm_idx = (self._heatmap_tracker.get_heatmap() * 255).astype(np.uint8)
            rgb = lut[hm_idx]
            rgb[render_grid == 0] = lut[0]
        else:
            rgb = lut[render_grid]
        
        if self.glow_enabled:
            rgb = self._glow.apply(render_grid, lut)
        if self.vignette_enabled:
            rgb = self._vignette.apply(rgb)
        
        return rgb

    def paintEvent(self, event) -> None:
        render_grid = np.maximum(self.grid, self.trail_grid) if self.trail_enabled and self.trail_grid is not None else self.grid
        rgb = self._get_render_rgb(render_grid)
        
        scaled_size = max(1, int(self.cell_size * self.zoom))
        if scaled_size > 1:
            rgb = np.repeat(np.repeat(rgb, scaled_size, axis=0), scaled_size, axis=1)
        
        # Keep reference to prevent garbage collection during paint
        self._paint_rgb = np.ascontiguousarray(rgb)
        h, w, ch = self._paint_rgb.shape
        
        self._paint_image = QImage(
            self._paint_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
        )
        
        painter = QPainter(self)
        painter.drawImage(0, 0, self._paint_image)
        
        # Draw grid lines
        if self.show_grid_lines and scaled_size >= 6:
            grid_color = get_grid_color(self.current_theme)
            painter.setPen(QPen(QColor(*grid_color), 1))
            for r in range(self.rows + 1):
                y = r * scaled_size
                painter.drawLine(0, y, w, y)
            for c in range(self.cols + 1):
                x = c * scaled_size
                painter.drawLine(x, 0, x, h)
        
        # Draw hover highlight
        if self._hover_cell is not None and not self._dragging:
            r, c = self._hover_cell
            if 0 <= r < self.rows and 0 <= c < self.cols:
                painter.setPen(QPen(QColor(255, 255, 0, 128), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(c * scaled_size, r * scaled_size, scaled_size, scaled_size)
        
        # Draw symmetry lines
        if self.symmetry_mode != "none" and scaled_size >= 2:
            painter.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.PenStyle.DashLine))
            if self.symmetry_mode in ("horizontal", "both"):
                x = w // 2
                painter.drawLine(x, 0, x, h)
            if self.symmetry_mode in ("vertical", "both"):
                y = h // 2
                painter.drawLine(0, y, w, y)
            if self.symmetry_mode == "rotational":
                cx, cy = w // 2, h // 2
                painter.drawLine(cx - 10, cy, cx + 10, cy)
                painter.drawLine(cx, cy - 10, cx, cy + 10)
        
        painter.end()

    def _get_cell_from_pos(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        """Convert mouse position to cell coordinates."""
        scaled = max(1, int(self.cell_size * self.zoom))
        c = int(pos.x() // scaled)
        r = int(pos.y() // scaled)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = e.position()
            return
        
        cell = self._get_cell_from_pos(e.position())
        if cell is None:
            return
            
        r, c = cell
        self._dragging = True
        self._last_btn = e.button()
        self._last_cell = None
        self.save_undo()
        
        if e.button() == Qt.MouseButton.LeftButton:
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._draw_state = (self.grid[r, c] + 1) % self.max_state
            else:
                self._draw_state = 1
        elif e.button() == Qt.MouseButton.RightButton:
            self._draw_state = 0
        else:
            self._draw_state = None
            
        self._handle_draw(e)

    def mouseMoveEvent(self, e) -> None:
        # Update hover cell
        cell = self._get_cell_from_pos(e.position())
        if cell != self._hover_cell:
            self._hover_cell = cell
            if cell:
                r, c = cell
                state = int(self.grid[r, c])
                self.cell_hovered.emit(r, c, state)
                QToolTip.showText(
                    e.globalPos().toPoint(),
                    f"({r}, {c}) State: {state}",
                    self
                )
            if not self._dragging:
                self.update()
        
        if self._panning and self._pan_start is not None:
            delta = e.position() - self._pan_start
            parent = self.parent()
            while parent:
                if hasattr(parent, 'horizontalScrollBar'):
                    parent.horizontalScrollBar().setValue(
                        parent.horizontalScrollBar().value() - int(delta.x())
                    )
                    parent.verticalScrollBar().setValue(
                        parent.verticalScrollBar().value() - int(delta.y())
                    )
                    break
                parent = parent.parent()
            self._pan_start = e.position()
            return
            
        if self._dragging:
            self._handle_draw(e)

    def mouseReleaseEvent(self, e) -> None:
        self._panning = False
        self._pan_start = None
        self._dragging = False
        self._last_cell = None
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover_cell = None
        self.update()
        super().leaveEvent(e)

    def _handle_draw(self, e) -> None:
        cell = self._get_cell_from_pos(e.position())
        if cell is None:
            return
            
        r, c = cell
        if (r, c) == self._last_cell:
            return
            
        self._last_cell = (r, c)
        if self._draw_state is not None:
            # Apply to all symmetry positions
            for sr, sc in self._apply_symmetry(r, c, self._draw_state):
                self.grid[sr, sc] = self._draw_state
            self.update()

    def wheelEvent(self, e) -> None:
        if e.angleDelta().y() > 0:
            self.set_zoom(self.zoom * 1.15)
        else:
            self.set_zoom(self.zoom / 1.15)

    def keyPressEvent(self, e) -> None:
        key = e.key()
        modifiers = e.modifiers()
        
        if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
        elif key == Qt.Key.Key_Y and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.redo()
        elif key == Qt.Key.Key_Space:
            if self.is_running():
                self.stop()
            else:
                self.start()
        elif key == Qt.Key.Key_Right:
            self.step()
        elif key == Qt.Key.Key_C and not (modifiers & Qt.KeyboardModifier.ControlModifier):
            self.clear_grid()
        elif key == Qt.Key.Key_R and not (modifiers & Qt.KeyboardModifier.ControlModifier):
            self.randomize(0.3)

# ========================================================
# SECTION: Window (from window.py)
# ========================================================

"""Main application window."""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QSpinBox, QLabel, QSlider, QComboBox, QGroupBox, QFormLayout, QScrollArea,
    QCheckBox, QMessageBox, QDoubleSpinBox, QStatusBar, QTabWidget, QSplitter,
    QTextEdit, QDialog, QLineEdit, QDialogButtonBox, QListWidget, QListWidgetItem,
    QMenuBar, QMenu
)
from PySide6.QtGui import QShortcut, QKeySequence, QAction
from PySide6.QtCore import Qt, QPointF

import logging

logger = logging.getLogger(__name__)


class RLEImportDialog(QDialog):
    """Dialog for importing RLE patterns."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import RLE Pattern")
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Paste RLE pattern below:"))
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "#N Glider\n#C A classic pattern\nx = 3, y = 3\nbo$2bo$3o!"
        )
        layout.addWidget(self.text_edit)
        
        btn_layout = QHBoxLayout()
        from_file_btn = QPushButton("From File...")
        from_file_btn.clicked.connect(self._load_from_file)
        ok_btn = QPushButton("Import")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(from_file_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
    
    def _load_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open RLE File", "", "RLE Files (*.rle *.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'r') as f:
                    self.text_edit.setPlainText(f.read())
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to read file: {e}")
    
    def get_rle_text(self) -> str:
        return self.text_edit.toPlainText()


class PaletteEditorDialog(QDialog):
    """Dialog for creating custom palettes."""
    
    def __init__(self, parent=None, initial_palette=None):
        super().__init__(parent)
        self.setWindowTitle("Palette Editor")
        self.setMinimumSize(450, 550)
        self._colors = []
        
        layout = QVBoxLayout(self)
        
        # Name
        form = QFormLayout()
        self.name_edit = QLineEdit("Custom Palette")
        form.addRow("Name:", self.name_edit)
        layout.addLayout(form)
        
        # Color preview list
        self.color_list = QListWidget()
        self.color_list.setMaximumHeight(150)
        layout.addWidget(QLabel("Colors:"))
        layout.addWidget(self.color_list)
        
        # Color input
        input_layout = QHBoxLayout()
        self.color_input = QLineEdit("#ff0000")
        self.color_input.setPlaceholderText("#RRGGBB")
        input_layout.addWidget(self.color_input)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_color)
        input_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_color)
        input_layout.addWidget(remove_btn)
        layout.addLayout(input_layout)
        
        # Quick add buttons
        quick_layout = QHBoxLayout()
        for name, hex_color in [("Random", None), ("Red", "#ff0000"), ("Green", "#00ff00"), 
                                  ("Blue", "#0000ff"), ("Yellow", "#ffff00"), ("Cyan", "#00ffff"),
                                  ("Magenta", "#ff00ff"), ("White", "#ffffff")]:
            btn = QPushButton(name)
            if hex_color:
                btn.clicked.connect(lambda checked, h=hex_color: self._add_hex(h))
            else:
                btn.clicked.connect(self._add_random)
            quick_layout.addWidget(btn)
        layout.addLayout(quick_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        generate_btn = QPushButton("Generate Random 5")
        generate_btn.clicked.connect(self._generate_random)
        ok_btn = QPushButton("Save Palette")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(generate_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        # Load initial palette if provided
        if initial_palette:
            self.name_edit.setText(initial_palette.name)
            self._colors = initial_palette.colors.copy()
            self._update_display()
    
    def _add_color(self) -> None:
        hex_str = self.color_input.text().strip()
        try:
            color = Color.from_hex(hex_str)
            self._colors.append(color)
            self._update_display()
            self.color_input.clear()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid hex color format. Use #RRGGBB")
    
    def _add_hex(self, hex_str: str) -> None:
        self.color_input.setText(hex_str)
        self._add_color()
    
    def _add_random(self) -> None:
        import random
        color = Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self._colors.append(color)
        self._update_display()
    
    def _remove_color(self) -> None:
        row = self.color_list.currentRow()
        if row >= 0:
            self._colors.pop(row)
            self._update_display()
    
    def _generate_random(self) -> None:
        self._colors.clear()
        palette = PaletteGenerator.random_palette(5, seed=None)
        self._colors = palette.colors.copy()
        self._update_display()
    
    def _update_display(self) -> None:
        self.color_list.clear()
        if not self._colors:
            self.color_list.addItem("No colors added")
            return
        for color in self._colors:
            item = QListWidgetItem(f"{color.to_hex()}  (rgb({color.r}, {color.g}, {color.b}))")
            item.setForeground(color.to_tuple())
            self.color_list.addItem(item)
    
    def get_palette(self) -> Palette:
        name = self.name_edit.text().strip() or "Custom"
        return Palette(name=name, colors=self._colors.copy())


class HelpDialog(QDialog):
    """Dialog showing keyboard shortcuts and help."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help - Keyboard Shortcuts")
        self.setMinimumSize(500, 650)
        
        layout = QVBoxLayout(self)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <h2>Keyboard Shortcuts</h2>
        <table cellpadding="5" cellspacing="0" style="width:100%;">
        <tr><td><b>Space</b></td><td>Toggle play/pause</td></tr>
        <tr><td><b>Right Arrow</b></td><td>Step one generation</td></tr>
        <tr><td><b>Ctrl+Z</b></td><td>Undo</td></tr>
        <tr><td><b>Ctrl+Y / Ctrl+Shift+Z</b></td><td>Redo</td></tr>
        <tr><td><b>Ctrl+S</b></td><td>Save file</td></tr>
        <tr><td><b>C</b></td><td>Clear grid</td></tr>
        <tr><td><b>R</b></td><td>Random fill (30%)</td></tr>
        </table>
        
        <h2>Mouse Controls</h2>
        <table cellpadding="5" cellspacing="0" style="width:100%;">
        <tr><td><b>Left Click / Drag</b></td><td>Draw cells</td></tr>
        <tr><td><b>Right Click / Drag</b></td><td>Erase cells</td></tr>
        <tr><td><b>Shift+Left Click</b></td><td>Cycle cell state</td></tr>
        <tr><td><b>Middle Click + Drag</b></td><td>Pan view (in scroll area)</td></tr>
        <tr><td><b>Scroll Wheel</b></td><td>Zoom in/out</td></tr>
        </table>
        
        <h2>Rule Format</h2>
        <p>Use B/S (Birth/Survive) notation: <code>B3/S23</code> means cells are born with 
        exactly 3 neighbors and survive with 2 or 3 neighbors.</p>
        
        <h2>RLE Format</h2>
        <p>RLE (Run Length Encoded) is a standard pattern format used by 
        Life-like cellular automata programs. You can import RLE patterns 
        via the "Rules & Analysis" tab.</p>
        
        <h2>Visual Modes</h2>
        <ul>
        <li><b>Standard</b> - Uses selected palette colors mapped to cell states.</li>
        <li><b>Age</b> - Colors based on how many generations a cell has been alive (Inferno palette).</li>
        <li><b>Heatmap</b> - Colors based on how frequently a cell has been alive over time.</li>
        <li><b>Outline</b> - Shows only the edges/boundaries of alive cells.</li>
        <li><b>Neighbor Count</b> - Colors alive cells based on their number of neighbors.</li>
        </ul>
        </html>
        """)
        layout.addWidget(help_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class MainWindow(QMainWindow):
    """Main application window containing all controls and the CA widget."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellular Automata Studio v2.1")
        self.resize(1300, 900)
        self.ca_widget = CellularAutomataWidget()
        self.palette_manager = PaletteManager()
        
        # Connect signals
        self.ca_widget.generation_updated.connect(self._upd_gen)
        self.ca_widget.population_updated.connect(self._upd_pop)
        self.ca_widget.fps_updated.connect(self._upd_fps)
        
        self._build_menu_bar()
        self._build_ui()
        self._build_shortcuts()
        self._update_status()

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("File")
        save_act = QAction("Save State...", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_file)
        file_menu.addAction(save_act)
        
        load_act = QAction("Load State...", self)
        load_act.triggered.connect(self._load_file)
        file_menu.addAction(load_act)
        
        file_menu.addSeparator()
        
        export_png_act = QAction("Export as PNG...", self)
        export_png_act.triggered.connect(self._export_png)
        file_menu.addAction(export_png_act)
        
        # Edit Menu
        edit_menu = menubar.addMenu("Edit")
        undo_act = QAction("Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self.ca_widget.undo)
        edit_menu.addAction(undo_act)
        
        redo_act = QAction("Redo", self)
        redo_act.setShortcut("Ctrl+Y")
        redo_act.triggered.connect(self.ca_widget.redo)
        edit_menu.addAction(redo_act)
        
        edit_menu.addSeparator()
        
        clear_act = QAction("Clear Grid", self)
        clear_act.triggered.connect(self.ca_widget.clear_grid)
        edit_menu.addAction(clear_act)
        
        random_act = QAction("Random Fill", self)
        random_act.triggered.connect(lambda: self.ca_widget.randomize(0.3))
        edit_menu.addAction(random_act)
        
        # Help Menu
        help_menu = menubar.addMenu("Help")
        help_act = QAction("Controls && Shortcuts", self)
        help_act.triggered.connect(self._show_help)
        help_menu.addAction(help_act)

    def _build_ui(self) -> None:
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.ca_widget)
        self.scroll.setWidgetResizable(False)
        self.scroll.setMinimumSize(400, 400)
        
        tabs = QTabWidget()
        tabs.addTab(self._build_playback_tab(), "Controls")
        tabs.addTab(self._build_visuals_tab(), "Visuals & Palettes")
        tabs.addTab(self._build_rules_tab(), "Rules & Analysis")
        tabs.setMaximumWidth(380)
        tabs.setMinimumWidth(320)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.scroll)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.splitter = splitter
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    def _build_playback_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        
        # Playback controls
        grp = QGroupBox("Playback")
        h = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.ca_widget.start)
        self.step_btn = QPushButton("⏭ Step")
        self.step_btn.clicked.connect(self.ca_widget.step)
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.ca_widget.stop)
        h.addWidget(self.play_btn)
        h.addWidget(self.step_btn)
        h.addWidget(self.pause_btn)
        grp.setLayout(h)
        lay.addWidget(grp)
        
        # Stats display
        grp2 = QGroupBox("Stats")
        f = QFormLayout()
        self.gen_lbl = QLabel("0")
        self.pop_lbl = QLabel("0")
        self.fps_lbl = QLabel("0.0")
        self.undo_lbl = QLabel("0")
        self.redo_lbl = QLabel("0")
        f.addRow("Generation:", self.gen_lbl)
        f.addRow("Population:", self.pop_lbl)
        f.addRow("FPS:", self.fps_lbl)
        f.addRow("Undo Stack:", self.undo_lbl)
        f.addRow("Redo Stack:", self.redo_lbl)
        grp2.setLayout(f)
        lay.addWidget(grp2)
        
        # Grid and backend settings
        grp3 = QGroupBox("Grid & Backend")
        f2 = QFormLayout()
        
        self.backend_combo = QComboBox()
        for b in BackendManager.AVAILABLE_BACKENDS:
            self.backend_combo.addItem(b, b)
        self.backend_combo.currentTextChanged.connect(
            lambda t: self.ca_widget.backend_manager.set_backend(t)
        )
        f2.addRow("Backend:", self.backend_combo)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 500)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(lambda v: self.ca_widget.set_speed(v))
        self.speed_lbl = QLabel("50 ms")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_lbl.setText(f"{v} ms"))
        f2.addRow("Speed:", self.speed_slider)
        f2.addRow("", self.speed_lbl)
        
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(1)
        self.steps_spin.setToolTip("Generations to compute per frame tick")
        self.steps_spin.valueChanged.connect(lambda v: self.ca_widget.set_steps_per_frame(v))
        f2.addRow("Steps/Frame:", self.steps_spin)
        
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(10, 2000)
        self.rows_spin.setValue(150)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(10, 2000)
        self.cols_spin.setValue(150)
        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(1, 50)
        self.cell_size_spin.setValue(5)
        self.cell_size_spin.valueChanged.connect(lambda v: self.ca_widget.set_cell_size(v))
        
        f2.addRow("Rows:", self.rows_spin)
        f2.addRow("Cols:", self.cols_spin)
        f2.addRow("Cell Size:", self.cell_size_spin)
        
        apply_btn = QPushButton("Apply Grid Size")
        apply_btn.clicked.connect(
            lambda: self.ca_widget.resize_grid(self.rows_spin.value(), self.cols_spin.value())
        )
        f2.addRow(apply_btn)
        grp3.setLayout(f2)
        lay.addWidget(grp3)
        
        # Pattern injection
        grp4 = QGroupBox("Patterns")
        h2 = QHBoxLayout()
        self.preset_combo = QComboBox()
        for cat, pats in PRESET_CATEGORIES.items():
            self.preset_combo.insertSeparator(self.preset_combo.count())
            for p in pats:
                self.preset_combo.addItem(f"  {p}", p)
        inj_btn = QPushButton("Inject")
        inj_btn.clicked.connect(
            lambda: self.ca_widget.inject_pattern(self.preset_combo.currentData())
        )
        h2.addWidget(self.preset_combo)
        h2.addWidget(inj_btn)
        grp4.setLayout(h2)
        lay.addWidget(grp4)
        
        # Action buttons
        grp5 = QGroupBox("Actions")
        v = QVBoxLayout()
        
        h3 = QHBoxLayout()
        rand_spin = QDoubleSpinBox()
        rand_spin.setRange(0.01, 0.99)
        rand_spin.setValue(0.3)
        rand_spin.setSingleStep(0.05)
        rand_btn = QPushButton("Random Fill")
        rand_btn.clicked.connect(lambda: self.ca_widget.randomize(rand_spin.value()))
        h3.addWidget(rand_spin)
        h3.addWidget(rand_btn)
        v.addLayout(h3)
        
        h4 = QHBoxLayout()
        save_st_btn = QPushButton("💾 Save St.")
        save_st_btn.setToolTip("Save current grid as initial state for Reset")
        save_st_btn.clicked.connect(
            lambda: setattr(self.ca_widget, 'initial_grid', self.ca_widget.grid.copy())
        )
        reset_btn = QPushButton("↩ Reset")
        reset_btn.setToolTip("Reset to saved initial state")
        reset_btn.clicked.connect(self.ca_widget.reset_to_initial)
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self.ca_widget.clear_grid)
        h4.addWidget(save_st_btn)
        h4.addWidget(reset_btn)
        h4.addWidget(clear_btn)
        v.addLayout(h4)
        
        h5 = QHBoxLayout()
        save_btn = QPushButton("Save File")
        save_btn.clicked.connect(self._save_file)
        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self._load_file)
        export_btn = QPushButton("Export PNG")
        export_btn.clicked.connect(self._export_png)
        h5.addWidget(save_btn)
        h5.addWidget(load_btn)
        h5.addWidget(export_btn)
        v.addLayout(h5)
        grp5.setLayout(v)
        lay.addWidget(grp5)
        
        lay.addStretch()
        return w

    def _build_visuals_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        
        # Visual mode selection
        grp = QGroupBox("Visual Modes")
        f = QFormLayout()
        self.visual_mode_combo = QComboBox()
        self.visual_mode_combo.addItems([
            "Standard", "Age (Inferno)", "Heatmap", "Outline", "Neighbor Count"
        ])
        self.visual_mode_combo.currentTextChanged.connect(self._on_visual_mode_changed)
        f.addRow("Mode:", self.visual_mode_combo)
        grp.setLayout(f)
        lay.addWidget(grp)
        
        # Palette selection
        grp2 = QGroupBox("Color Palettes")
        f2 = QFormLayout()
        self.palette_combo = QComboBox()
        for name in self.palette_manager.get_names():
            self.palette_combo.addItem(name)
        self.palette_combo.currentTextChanged.connect(lambda n: self.ca_widget.set_palette(n))
        f2.addRow("Palette:", self.palette_combo)
        
        pal_btn_layout = QHBoxLayout()
        edit_pal_btn = QPushButton("Edit/Create")
        edit_pal_btn.clicked.connect(self._edit_palette)
        import_pal_btn = QPushButton("Import")
        import_pal_btn.clicked.connect(self._import_palette)
        export_pal_btn = QPushButton("Export")
        export_pal_btn.clicked.connect(self._export_palette)
        pal_btn_layout.addWidget(edit_pal_btn)
        pal_btn_layout.addWidget(import_pal_btn)
        pal_btn_layout.addWidget(export_pal_btn)
        f2.addRow(pal_btn_layout)
        grp2.setLayout(f2)
        lay.addWidget(grp2)
        
        # Effect toggles
        grp3 = QGroupBox("Effects & Grid")
        v = QVBoxLayout()
        
        h_trail = QHBoxLayout()
        self.trail_chk = QCheckBox("Trails")
        self.trail_chk.toggled.connect(self._on_trail_toggled)
        self.trail_length_spin = QSpinBox()
        self.trail_length_spin.setRange(1, 50)
        self.trail_length_spin.setValue(15)
        self.trail_length_spin.setEnabled(False)
        self.trail_length_spin.valueChanged.connect(lambda v: self.ca_widget.enable_trail(v) if self.trail_chk.isChecked() else None)
        h_trail.addWidget(self.trail_chk)
        h_trail.addWidget(QLabel("Length:"))
        h_trail.addWidget(self.trail_length_spin)
        v.addLayout(h_trail)
        
        self.glow_chk = QCheckBox("Glow Effect (requires scipy)")
        self.glow_chk.toggled.connect(lambda c: setattr(self.ca_widget, 'glow_enabled', c))
        
        self.vignette_chk = QCheckBox("Vignette Effect")
        self.vignette_chk.toggled.connect(lambda c: setattr(self.ca_widget, 'vignette_enabled', c))
        
        self.birth_death_chk = QCheckBox("Birth/Death Flash")
        self.birth_death_chk.toggled.connect(lambda c: setattr(self.ca_widget, 'birth_death_enabled', c))
        
        self.gridlines_chk = QCheckBox("Show Grid Lines")
        self.gridlines_chk.setChecked(self.ca_widget.show_grid_lines)
        self.gridlines_chk.toggled.connect(self._on_gridlines_toggled)
        
        self.wrap_chk = QCheckBox("Toroidal Wrap (Edges wrap around)")
        self.wrap_chk.setChecked(True)
        self.wrap_chk.toggled.connect(self.ca_widget.set_wrap_mode)
        
        v.addWidget(self.glow_chk)
        v.addWidget(self.vignette_chk)
        v.addWidget(self.birth_death_chk)
        v.addWidget(self.gridlines_chk)
        v.addWidget(self.wrap_chk)
        grp3.setLayout(v)
        lay.addWidget(grp3)
        
        # Symmetry
        grp4 = QGroupBox("Drawing Symmetry")
        f3 = QFormLayout()
        self.symmetry_combo = QComboBox()
        self.symmetry_combo.addItems(["None", "Horizontal", "Vertical", "Both", "Rotational"])
        self.symmetry_combo.currentTextChanged.connect(
            lambda t: self.ca_widget.set_symmetry(t.lower())
        )
        f3.addRow("Mode:", self.symmetry_combo)
        grp4.setLayout(f3)
        lay.addWidget(grp4)
        
        lay.addStretch()
        return w

    def _build_rules_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        
        # Rule configuration
        grp = QGroupBox("Rule Configuration")
        f = QFormLayout()
        
        self.rule_combo = QComboBox()
        self.rule_combo.setEditable(True)
        self.rule_combo.setEditText("B3/S23")
        rules = [
            "B3/S23 (Conway's Life)", 
            "B36/S23 (HighLife)", 
            "B3678/S34678 (Day & Night)", 
            "B1357/S1357 (Replicator)", 
            "B2/S (Seeds)", 
            "B368/S245 (Morley)",
            "B3/S012345678 (Life w/o Death)",
            "B2/S345 (Maze)",
            "B3/S1234 (Mazectric)"
        ]
        self.rule_combo.addItems(rules)
        self.rule_combo.editTextChanged.connect(self._on_rule_changed)
        f.addRow("Rule (B/S):", self.rule_combo)
        
        self.max_state_spin = QSpinBox()
        self.max_state_spin.setRange(2, 256)
        self.max_state_spin.setValue(16)
        self.max_state_spin.valueChanged.connect(lambda v: self.ca_widget.set_max_state(v))
        f.addRow("Max States:", self.max_state_spin)
        
        import_rle_btn = QPushButton("Import RLE Pattern...")
        import_rle_btn.clicked.connect(self._import_rle)
        f.addRow(import_rle_btn)
        grp.setLayout(f)
        lay.addWidget(grp)
        
        # Rule analysis
        grp2 = QGroupBox("Rule Analysis")
        v2 = QVBoxLayout()
        ana_btn = QPushButton("Analyze Current Rule")
        ana_btn.clicked.connect(self._analyze_rule)
        v2.addWidget(ana_btn)
        
        self.ana_text = QTextEdit()
        self.ana_text.setReadOnly(True)
        self.ana_text.setMaximumHeight(180)
        v2.addWidget(self.ana_text)
        grp2.setLayout(v2)
        lay.addWidget(grp2)
        
        # Suggested rules
        grp3 = QGroupBox("Suggested Rules")
        v3 = QVBoxLayout()
        self.suggested_list = QListWidget()
        self.suggested_list.setMaximumHeight(150)
        for rule_str, desc in get_suggested_rules():
            self.suggested_list.addItem(f"{rule_str} - {desc}")
        self.suggested_list.itemDoubleClicked.connect(self._apply_suggested_rule)
        v3.addWidget(self.suggested_list)
        tip_lbl = QLabel("Double-click to apply")
        tip_lbl.setStyleSheet("color: gray; font-style: italic;")
        v3.addWidget(tip_lbl)
        grp3.setLayout(v3)
        lay.addWidget(grp3)
        
        lay.addStretch()
        return w

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.ca_widget.step)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._do_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.ca_widget.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.ca_widget.redo)

    def _do_undo(self) -> None:
        self.ca_widget.undo()
        self._upd_undo_redo()

    def _upd_gen(self, g: int) -> None:
        self.gen_lbl.setText(f"{g:,}")
    
    def _upd_pop(self, p: int) -> None:
        self.pop_lbl.setText(f"{p:,}")
    
    def _upd_fps(self, f: float) -> None:
        self.fps_lbl.setText(f"{f:.1f}")
        # Update undo/redo counts periodically to save overhead
        self._upd_undo_redo()
        
    def _upd_undo_redo(self) -> None:
        self.undo_lbl.setText(str(self.ca_widget.undo_stack.undo_count))
        self.redo_lbl.setText(str(self.ca_widget.undo_stack.redo_count))

    def _toggle_play(self) -> None:
        if self.ca_widget.is_running():
            self.ca_widget.stop()
        else:
            self.ca_widget.start()

    def _on_visual_mode_changed(self, text: str) -> None:
        if "Age" in text:
            self.ca_widget.set_visual_mode("Age")
        elif "Heat" in text:
            self.ca_widget.set_visual_mode("Heatmap")
        elif "Outline" in text:
            self.ca_widget.set_visual_mode("Outline")
        elif "Neighbor" in text:
            self.ca_widget.set_visual_mode("Neighbor Count")
        else:
            self.ca_widget.set_visual_mode("Standard")
    
    def _on_trail_toggled(self, checked: bool) -> None:
        self.trail_length_spin.setEnabled(checked)
        if checked:
            self.ca_widget.enable_trail(self.trail_length_spin.value())
        else:
            self.ca_widget.disable_trail()
    
    def _on_gridlines_toggled(self, checked: bool) -> None:
        self.ca_widget.show_grid_lines = checked
        self.ca_widget.update()
    
    def _on_rule_changed(self, text: str) -> None:
        rule = text.split()[0] if ' ' in text else text
        self.ca_widget.set_rule(rule)
        self._update_status()

    def _analyze_rule(self) -> None:
        r = self.rule_combo.currentText().split()[0]
        res = RuleAnalyzer().analyze(r)
        if "error" in res:
            self.ana_text.setText(res["error"])
            return
        desc = res.get("description", "N/A")
        self.ana_text.setText(
            f"Rule: {res['rule']}\n"
            f"Name: {desc}\n"
            f"Category: {res['category']}\n\n"
            f"Metrics (over {res['generations_ran']} gens):\n"
            f"  Initial Pop: {res['initial_pop']:,}\n"
            f"  Final Pop:   {res['final_pop']:,}\n"
            f"  Expansion:   {res['expansion']:.3f}\n"
            f"  Stability:   {res['stability']:.3f}\n"
            f"  Chaos:       {res['chaos']:.3f}\n"
            f"  Avg Growth:  {res['avg_growth']:.3f}"
        )

    def _apply_suggested_rule(self, item: QListWidgetItem) -> None:
        rule_str = item.text().split(" - ")[0]
        self.rule_combo.setEditText(rule_str)

    def _import_rle(self) -> None:
        dialog = RLEImportDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            rle_text = dialog.get_rle_text()
            if rle_text.strip():
                if self.ca_widget.import_rle(rle_text):
                    self.statusBar().showMessage("RLE pattern imported successfully", 3000)
                else:
                    QMessageBox.warning(self, "Import Failed", "Could not parse RLE pattern. Check format.")

    def _edit_palette(self) -> None:
        current_name = self.palette_combo.currentText()
        current_pal = self.palette_manager.get_palette(current_name)
        
        dialog = PaletteEditorDialog(self, initial_palette=current_pal)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_pal = dialog.get_palette()
            self.palette_manager.add_custom(new_pal)
            
            # Refresh combo box
            self.palette_combo.clear()
            for name in self.palette_manager.get_names():
                self.palette_combo.addItem(name)
            
            idx = self.palette_combo.findText(new_pal.name)
            if idx >= 0:
                self.palette_combo.setCurrentIndex(idx)
            
            self.statusBar().showMessage(f"Palette '{new_pal.name}' saved", 3000)

    def _import_palette(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Palette", "", "JSON (*.json)")
        if path:
            success, msg = self.palette_manager.import_palette(path)
            if success:
                self.palette_combo.clear()
                for name in self.palette_manager.get_names():
                    self.palette_combo.addItem(name)
                idx = self.palette_combo.findText(msg.split(": ")[-1])
                if idx >= 0:
                    self.palette_combo.setCurrentIndex(idx)
                self.statusBar().showMessage(msg, 3000)
            else:
                QMessageBox.warning(self, "Import Failed", msg)

    def _export_palette(self) -> None:
        name = self.palette_combo.currentText()
        path, _ = QFileDialog.getSaveFileName(self, "Export Palette", f"{name}.json", "JSON (*.json)")
        if path:
            if self.palette_manager.export_palette(name, path):
                self.statusBar().showMessage(f"Palette '{name}' exported", 3000)
            else:
                QMessageBox.warning(self, "Export Failed", f"Could not find palette '{name}'")

    def _show_help(self) -> None:
        dialog = HelpDialog(self)
        dialog.exec()

    def _save_file(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save State", "", "JSON (*.json)")
        if p:
            self.ca_widget.save_to_file(p)
            self.statusBar().showMessage(f"Saved to {p}", 3000)

    def _load_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Load State", "", "JSON (*.json)")
        if p:
            ok, d = self.ca_widget.load_from_file(p)
            if not ok:
                QMessageBox.critical(self, "Error", d.get("error", "Unknown error"))
            else:
                # Update UI to reflect loaded state
                self.rule_combo.setEditText(d.get("rule", "B3/S23"))
                self.rows_spin.setValue(d.get("rows", 150))
                self.cols_spin.setValue(d.get("cols", 150))
                self.wrap_chk.setChecked(d.get("wrap", True))
                self._upd_gen(d.get("gen", 0))
                self.statusBar().showMessage(f"Loaded from {p}", 3000)

    def _export_png(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG (*.png)")
        if p:
            success, msg = self.ca_widget.export_to_png(p)
            if not success:
                QMessageBox.warning(self, "Error", msg)
            else:
                self.statusBar().showMessage(msg, 3000)

    def set_theme(self, name: str) -> None:
        self.ca_widget.set_theme(name)
        # Auto select palette based on theme
        pal_map = {
            "dark": "Standard", 
            "matrix": "Matrix", 
            "ocean": "Ocean", 
            "light": "Pastel",
            "cyberpunk": "Cyberpunk"
        }
        pal = pal_map.get(name, "Standard")
        self.ca_widget.set_palette(pal)
        idx = self.palette_combo.findText(pal)
        if idx >= 0:
            self.palette_combo.setCurrentIndex(idx)

    def _update_status(self) -> None:
        b = self.ca_widget.backend_manager.get_effective_backend()
        deps = []
        if HAS_NUMBA: deps.append("Numba")
        if HAS_CUPY: deps.append("CuPy")
        dep_str = f" | Extras: {', '.join(deps)}" if deps else ""
        self.statusBar().showMessage(
            f"Backend: {b}{dep_str} | LMB:Draw, RMB:Erase, Shift+LMB:Cycle, MMB:Pan, Scroll:Zoom"
        )

# ========================================================
# SECTION: App (from app.py)
# ========================================================

"""
Entry point for Cellular Automata Studio.
Usage: python app.py
"""
import sys
import os
import argparse
import logging

# Absolute imports since all files are in the same directory

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cellular Automata Studio v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --rule B36/S23 --preset Glider
  python app.py --random-fill 0.3 --theme matrix --fast
  python app.py --rows 300 --cols 300 --cell-size 3
        """
    )
    parser.add_argument("--rows", type=int, default=150, help="Grid rows (default: 150)")
    parser.add_argument("--cols", type=int, default=150, help="Grid cols (default: 150)")
    parser.add_argument("--cell-size", type=int, default=5, help="Cell size in pixels (default: 5)")
    parser.add_argument("--rule", type=str, default="B3/S23", help="B/S notation rule")
    parser.add_argument("--max-state", type=int, default=16, help="Max cell state value")
    parser.add_argument("--preset", type=str, default=None, help="Initial pattern to inject")
    parser.add_argument("--random-fill", type=float, default=None, help="Random fill density (0.0-1.0)")
    parser.add_argument("--theme", type=str, choices=["light", "dark", "matrix", "ocean", "cyberpunk"], default="dark")
    parser.add_argument("--backend", type=str, choices=["Auto", "Python", "NumPy", "Numba", "CuPy"], default="Auto")
    parser.add_argument("--fast", action="store_true", help="Shortcut for --backend Numba")
    parser.add_argument("--speed", type=int, default=50, help="Simulation speed in ms (1-500)")
    parser.add_argument("--steps-per-frame", type=int, default=1, help="Generations per frame (1-100)")
    parser.add_argument("--no-grid-lines", action="store_true", help="Hide grid lines")
    parser.add_argument("--trail", action="store_true", help="Enable trail/fade effect")
    parser.add_argument("--trail-length", type=int, default=15, help="Trail fade length (1-50)")
    parser.add_argument("--no-wrap", action="store_true", help="Disable toroidal wrapping")
    parser.add_argument("--symmetry", type=str, choices=["none", "horizontal", "vertical", "both", "rotational"], default="none")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser.parse_args()

def check_dependencies() -> dict:
    status = {'numpy': False, 'pyside6': False, 'numba': HAS_NUMBA, 'cupy': HAS_CUPY, 'scipy': False, 'pillow': False}
    try:
        import numpy; status['numpy'] = True
    except ImportError: pass
    try:
        import PySide6; status['pyside6'] = True
    except ImportError: pass
    try:
        import scipy; status['scipy'] = True
    except ImportError: pass
    try:
        import PIL; status['pillow'] = True
    except ImportError: pass
    return status

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    deps = check_dependencies()
    
    if not deps['numpy']:
        print("ERROR: NumPy is required. Install with: pip install numpy")
        sys.exit(1)
    if not deps['pyside6']:
        print("ERROR: PySide6 is required. Install with: pip install PySide6")
        sys.exit(1)

    logging.info(f"Dependencies: {deps}")

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    apply_theme(args.theme)
    app.setApplicationName("Cellular Automata Studio")
    app.setApplicationVersion("2.1")

    window = MainWindow()
    ca = window.ca_widget

    # Apply command line arguments
    if args.rows != 150 or args.cols != 150: 
        ca.resize_grid(args.rows, args.cols, preserve=False)
    if args.cell_size != 5: 
        ca.set_cell_size(args.cell_size)
    if args.rule != "B3/S23": 
        ca.set_rule(args.rule)
        window.rule_combo.setEditText(args.rule)
    if args.max_state != 16: 
        ca.set_max_state(args.max_state)
        window.max_state_spin.setValue(args.max_state)
    if args.fast: 
        ca.backend_manager.set_backend("Numba")
        window.backend_combo.setCurrentText("Numba")
    elif args.backend != "Auto": 
        ca.backend_manager.set_backend(args.backend)
        window.backend_combo.setCurrentText(args.backend)
    ca.set_speed(args.speed)
    window.speed_slider.setValue(args.speed)
    if args.steps_per_frame != 1:
        ca.set_steps_per_frame(args.steps_per_frame)
        window.steps_spin.setValue(args.steps_per_frame)
    if args.no_grid_lines: 
        ca.show_grid_lines = False
        window.gridlines_chk.setChecked(False)
    if args.trail: 
        ca.enable_trail(args.trail_length)
        window.trail_chk.setChecked(True)
        window.trail_length_spin.setValue(args.trail_length)
    if args.no_wrap:
        ca.set_wrap_mode(False)
        window.wrap_chk.setChecked(False)
    if args.symmetry != "none":
        ca.set_symmetry(args.symmetry)
        idx = window.symmetry_combo.findData(args.symmetry)
        if idx >= 0:
            window.symmetry_combo.setCurrentIndex(idx)
    
    window.set_theme(args.theme)
    
    if args.preset:
        if args.preset in PRESETS:
            ca.inject_pattern(args.preset)
            window.preset_combo.setCurrentText(args.preset)
    if args.random_fill: 
        ca.randomize(args.random_fill)

    window.show()
    logging.info("Application started")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()