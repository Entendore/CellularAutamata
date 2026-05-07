# widgets.py
"""Cellular Automata rendering widget."""
import json
import random
import time
from typing import Optional, Tuple, List
from collections import deque

import numpy as np

from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtGui import QPainter, QColor, QImage, QPen
from PySide6.QtCore import QTimer, Qt, Signal

from presets import PRESETS, rotate_pattern, parse_rle
from backends import BackendManager
from themes import get_theme_lut, get_grid_color
from color_palettes import PaletteManager
from visual_effects import AgeTracker, HeatmapTracker, VignetteEffect, GlowEffect


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


class CellularAutomataWidget(QWidget):
    """Main cellular automata rendering and simulation widget."""
    
    generation_updated = Signal(int)
    population_updated = Signal(int)
    fps_updated = Signal(float)

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
        self.timer.timeout.connect(self.next_generation)
        
        self._frame_times: deque = deque(maxlen=30)
        self._last_time = time.perf_counter()
        
        # Rule
        self.rule_string = rule
        self.birth_lookup = np.zeros(9, dtype=np.bool_)
        self.survive_lookup = np.zeros(9, dtype=np.bool_)
        self._parse_rule(rule)
        
        # Mouse state
        self._dragging = False
        self._panning = False
        self._pan_start: Optional[QPointF] = None
        self._last_btn: Optional[Qt.MouseButton] = None
        self._last_cell: Optional[Tuple[int, int]] = None
        self._draw_state: Optional[int] = None
        
        # Visual settings
        self.current_theme = "dark"
        self.palette_manager = PaletteManager()
        self.current_palette_name = "Standard"
        self.show_grid_lines = False
        
        # Visual modes
        self.visual_mode = "Standard"  # Standard, Age, Heatmap
        self.trail_enabled = False
        self.trail_grid: Optional[np.ndarray] = None
        self.trail_length = 15
        self.glow_enabled = False
        self.vignette_enabled = False
        
        # Visual effect trackers
        self._age_tracker = AgeTracker(rows, cols)
        self._heatmap_tracker = HeatmapTracker(rows, cols)
        self._vignette = VignetteEffect(0.3)
        self._glow = GlowEffect(2, 0.5)
        
        # Image reference to prevent GC during paint
        self._paint_image: Optional[QImage] = None
        self._paint_rgb: Optional[np.ndarray] = None
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_size()

    def _update_size(self) -> None:
        self.setFixedSize(
            int(self.cols * self.cell_size * self.zoom), 
            int(self.rows * self.cell_size * self.zoom)
        )

    def _parse_rule(self, rule: str) -> None:
        self.rule_string = rule
        self.birth_lookup[:] = False
        self.survive_lookup[:] = False
        if "/" not in rule:
            return
        try:
            b, s = rule.split("/")
            for c in b:
                if c.isdigit() and 0 <= int(c) <= 8:
                    self.birth_lookup[int(c)] = True
            for c in s:
                if c.isdigit() and 0 <= int(c) <= 8:
                    self.survive_lookup[int(c)] = True
        except Exception:
            pass

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
        if mode == "Heatmap":
            self._heatmap_tracker.reset()

    def enable_trail(self, length: int = 15) -> None:
        self.trail_enabled = True
        self.trail_length = length
        self.trail_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
    
    def disable_trail(self) -> None:
        self.trail_enabled = False
        self.trail_grid = None

    def next_generation(self) -> None:
        if self.trail_enabled and self.trail_grid is not None:
            self.trail_grid = np.maximum(self.trail_grid, self.grid)
            self.trail_grid = np.where(self.trail_grid > 0, self.trail_grid - 1, 0)
            self.trail_grid[self.grid > 0] = self.trail_length
            
        self.grid = self.backend_manager.evolve(self.grid, self.birth_lookup, self.survive_lookup)
        self.generation += 1
        self._age_tracker.update(self.grid)
        self._heatmap_tracker.update(self.grid)
        
        now = time.perf_counter()
        self._frame_times.append(now - self._last_time)
        self._last_time = now
        if len(self._frame_times) >= 5:
            avg_time = sum(self._frame_times) / len(self._frame_times)
            if avg_time > 0:
                self.fps_updated.emit(1.0 / avg_time)
        
        self.generation_updated.emit(self.generation)
        self.population_updated.emit(int(np.sum(self.grid > 0)))
        self.update()

    def set_speed(self, ms: int) -> None:
        self.timer.setInterval(max(1, ms))
    
    def start(self) -> None:
        self._last_time = time.perf_counter()
        self.timer.start(50)
    
    def stop(self) -> None:
        self.timer.stop()
        self.fps_updated.emit(0)
    
    def is_running(self) -> bool:
        return self.timer.isActive()
    
    def step(self) -> None:
        if not self.timer.isActive():
            self.next_generation()

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
        self.generation_updated.emit(0)
        self.population_updated.emit(0)
        self.update()

    def randomize(self, density: float = 0.3) -> None:
        self.save_undo()
        self.grid = (np.random.random((self.rows, self.cols)) < density).astype(np.int32)
        self.generation = 0
        self._age_tracker.reset()
        self._heatmap_tracker.reset()
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
        self.trail_grid = np.zeros((r, c), dtype=np.int32) if self.trail_enabled else None
        self._age_tracker.resize(r, c)
        self._heatmap_tracker.resize(r, c)
        self.undo_stack = UndoStack()
        self.generation = 0
        self._update_size()
        self.update()

    def set_cell_size(self, s: int) -> None:
        self.cell_size = s
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

    def inject_pattern(self, name: str, center: bool = True, save: bool = True) -> bool:
        p = PRESETS.get(name)
        if p is None:
            return False
        if not p:
            self.clear_grid(save)
            return True
        if save:
            self.save_undo()
        mr = max(r for r, c in p)
        mc = max(c for r, c in p)
        sr = (self.rows - mr - 1) // 2 if center else 0
        sc = (self.cols - mc - 1) // 2 if center else 0
        for dr, dc in p:
            r, c = sr + dr, sc + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = 1
        self.population_updated.emit(int(np.sum(self.grid > 0)))
        self.update()
        return True

    def save_to_file(self, path: str, code: str = "") -> None:
        data = {
            "v": "2.0", 
            "grid": self.grid.tolist(), 
            "rows": self.rows, 
            "cols": self.cols, 
            "rule": self.rule_string, 
            "gen": self.generation, 
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
            self.initial_grid = self.grid.copy()
            self._age_tracker.resize(self.rows, self.cols)
            self._heatmap_tracker.resize(self.rows, self.cols)
            if self.trail_enabled:
                self.trail_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
            self._update_size()
            self.update()
            return True, data
        except Exception as e:
            return False, {"error": str(e)}

    def export_to_png(self, path: str) -> bool:
        try:
            from PIL import Image
            render_grid = np.maximum(self.grid, self.trail_grid) if self.trail_enabled and self.trail_grid is not None else self.grid
            rgb = self._get_render_rgb(render_grid)
            img = Image.fromarray(rgb, 'RGB')
            img.save(path)
            return True
        except ImportError:
            return False
        except Exception:
            return False

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
        lut = self._get_lut()
        
        if self.visual_mode == "Age":
            max_age = lut.shape[0] - 1
            rgb = lut[np.clip(self._age_tracker.age_grid, 0, max_age)]
            rgb[render_grid == 0] = lut[0]  # Dead stays bg
        elif self.visual_mode == "Heatmap":
            hm_idx = (self._heatmap_tracker.get_heatmap() * 255).astype(np.uint8)
            rgb = lut[hm_idx]
            rgb[render_grid == 0] = lut[0]  # Dead stays bg
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
        
        if self.show_grid_lines and scaled_size >= 6:
            grid_color = get_grid_color(self.current_theme)
            painter.setPen(QPen(QColor(*grid_color), 1))
            for r in range(self.rows + 1):
                painter.drawLine(0, r * scaled_size, w, r * scaled_size)
            for c in range(self.cols + 1):
                painter.drawLine(c * scaled_size, 0, c * scaled_size, h)
        painter.end()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = e.position()
            return
        
        pos = e.position()
        scaled = max(1, int(self.cell_size * self.zoom))
        c, r = int(pos.x() // scaled), int(pos.y() // scaled)
        
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
            
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
        if self._panning and self._pan_start is not None:
            delta = e.position() - self._pan_start
            parent = self.parent()
            if parent and hasattr(parent, 'horizontalScrollBar'):
                parent.horizontalScrollBar().setValue(
                    parent.horizontalScrollBar().value() - int(delta.x())
                )
                parent.verticalScrollBar().setValue(
                    parent.verticalScrollBar().value() - int(delta.y())
                )
            self._pan_start = e.position()
            return
            
        if self._dragging:
            self._handle_draw(e)

    def mouseReleaseEvent(self, e) -> None:
        self._panning = False
        self._pan_start = None
        self._dragging = False
        self._last_cell = None

    def _handle_draw(self, e) -> None:
        pos = e.position()
        scaled = max(1, int(self.cell_size * self.zoom))
        c, r = int(pos.x() // scaled), int(pos.y() // scaled)
        
        if (r, c) == self._last_cell:
            return
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
            
        self._last_cell = (r, c)
        if self._draw_state is not None:
            self.grid[r, c] = self._draw_state
            self.update()

    def wheelEvent(self, e) -> None:
        if e.angleDelta().y() > 0:
            self.set_zoom(self.zoom * 1.15)
        else:
            self.set_zoom(self.zoom / 1.15)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Z and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()