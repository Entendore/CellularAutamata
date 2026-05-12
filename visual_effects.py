# visual_effects.py
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