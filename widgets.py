# widgets.py
"""Cellular automata rendering widget."""
import json
import random
import time
import logging
from typing import Optional, Tuple, List
from collections import deque

import numpy as np

from PySide6.QtWidgets import QWidget, QApplication, QToolTip
from PySide6.QtGui import QPainter, QColor, QImage, QPen, QPointF
from PySide6.QtCore import QTimer, Qt, Signal, Point

from presets import PRESETS, rotate_pattern, flip_pattern, parse_rle, pattern_to_rle
from backends import BackendManager
from themes import get_theme_lut, get_grid_color, get_background_color
from color_palettes import PaletteManager
from visual_effects import (
    AgeTracker, HeatmapTracker, VignetteEffect, GlowEffect,
    OutlineRenderer, NeighborCountVisualizer, BirthDeathTracker
)

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