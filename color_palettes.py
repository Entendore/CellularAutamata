# color_palettes.py
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

    def import_palette(self, path: str) -> bool:
        """Import palette from JSON file."""
        try:
            with open(path, 'r') as f: 
                data = json.load(f)
            palette = Palette.from_json(data)
            self._custom_palettes[palette.name] = palette
            return True
        except Exception: 
            return False


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
        
        # Generate base random grid
        base = rng.rand(noise_rows, noise_cols)
        
        # Bilinear interpolation to full size using numpy
        row_indices = np.linspace(0, noise_rows - 1, rows)
        col_indices = np.linspace(0, noise_cols - 1, cols)
        
        # Calculate interpolation weights
        r0 = np.floor(row_indices).astype(int)
        c0 = np.floor(col_indices).astype(int)
        r1 = np.minimum(r0 + 1, noise_rows - 1)
        c1 = np.minimum(c0 + 1, noise_cols - 1)
        
        dr = (row_indices - r0)[:, np.newaxis]
        dc = (col_indices - c0)[np.newaxis, :]
        
        # Interpolate - fixed 2D indexing
        top = base[np.ix_(r0, c0)] * (1 - dc) + base[np.ix_(r0, c1)] * dc
        bottom = base[np.ix_(r1, c0)] * (1 - dc) + base[np.ix_(r1, c1)] * dc
        noise = top * (1 - dr) + bottom * dr
        
        # Normalize 0-1
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
        self._progress += dt * 6.0  # Transition speed
        if self._progress >= 1.0:
            self._progress = 1.0
            return True  # Complete
        return False

    def get_blended_colors(self, current_grid: np.ndarray, color_lut: np.ndarray) -> np.ndarray:
        if self._old_grid is None or self._progress >= 1.0:
            return color_lut[current_grid]
        
        t = self._progress
        old_colors = color_lut[self._old_grid].astype(np.float32)
        new_colors = color_lut[current_grid].astype(np.float32)
        return (old_colors * (1 - t) + new_colors * t).astype(np.uint8)