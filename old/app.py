import sys, json, random
import numpy as np

# Attempt to import acceleration libraries
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

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSpinBox, QLabel, QTextEdit, QSlider,
    QComboBox, QGroupBox, QFormLayout, QScrollArea, QMessageBox
)
from PySide6.QtGui import QPainter, QColor, QImage, QPalette
from PySide6.QtCore import QTimer, Qt, Signal

# --- Preset Patterns (relative coordinates: row, col) ---
PRESETS = {
    "Clear": [],
    "Glider": [(0,1), (1,2), (2,0), (2,1), (2,2)],
    "Blinker": [(0,0), (0,1), (0,2)],
    "Toad": [(0,1), (0,2), (0,3), (1,0), (1,1), (1,2)],
    "Beacon": [(0,0), (0,1), (1,0), (2,3), (3,2), (3,3)],
    "Pulsar": [
        (0,2),(0,3),(0,4),(0,8),(0,9),(0,10),
        (2,0),(2,5),(2,7),(2,12),(3,0),(3,5),(3,7),(3,12),
        (4,0),(4,5),(4,7),(4,12),(5,2),(5,3),(5,4),(5,8),(5,9),(5,10),
        (7,2),(7,3),(7,4),(7,8),(7,9),(7,10),(8,0),(8,5),(8,7),(8,12),
        (9,0),(9,5),(9,7),(9,12),(10,0),(10,5),(10,7),(10,12),
        (12,2),(12,3),(12,4),(12,8),(12,9),(12,10)
    ],
    "LWSS": [(0,1),(0,4),(1,0),(2,0),(2,4),(3,0),(3,1),(3,2),(3,3)]
}

# =====================================================================
# ACCELERATION BACKENDS
# =====================================================================

def evolve_python(grid, birth_lookup, survive_lookup):
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            alive = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0: continue
                    if grid[(r + dr) % rows, (c + dc) % cols] > 0: alive += 1
            if grid[r, c] > 0 and alive < len(survive_lookup) and survive_lookup[alive]:
                new_grid[r, c] = grid[r, c]
            elif grid[r, c] == 0 and alive < len(birth_lookup) and birth_lookup[alive]:
                new_grid[r, c] = 1
    return new_grid

if HAS_NUMBA:
    @numba.njit
    def evolve_numba(grid, birth_lookup, survive_lookup):
        rows, cols = grid.shape
        new_grid = np.zeros_like(grid)
        for r in range(rows):
            for c in range(cols):
                alive = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0: continue
                        if grid[(r + dr) % rows, (c + dc) % cols] > 0: alive += 1
                if grid[r, c] > 0 and alive < len(survive_lookup) and survive_lookup[alive]:
                    new_grid[r, c] = grid[r, c]
                elif grid[r, c] == 0 and alive < len(birth_lookup) and birth_lookup[alive]:
                    new_grid[r, c] = 1
        return new_grid

if HAS_CUPY:
    def evolve_cupy(grid_np, birth_lookup, survive_lookup):
        if grid_np.size < 10000: 
            return evolve_numba(grid_np, birth_lookup, survive_lookup) if HAS_NUMBA else evolve_python(grid_np, birth_lookup, survive_lookup)
            
        g = cp.asarray(grid_np)
        g_bool = g > 0
        
        neighbors = (cp.roll(g_bool, 1, 0).astype(cp.int32) +
                     cp.roll(g_bool, -1, 0).astype(cp.int32) +
                     cp.roll(g_bool, 1, 1).astype(cp.int32) +
                     cp.roll(g_bool, -1, 1).astype(cp.int32) +
                     cp.roll(g_bool, (1, 1), (0, 1)).astype(cp.int32) +
                     cp.roll(g_bool, (1, -1), (0, 1)).astype(cp.int32) +
                     cp.roll(g_bool, (-1, 1), (0, 1)).astype(cp.int32) +
                     cp.roll(g_bool, (-1, -1), (0, 1)).astype(cp.int32))
        
        cp_birth = cp.asarray(birth_lookup)
        cp_survive = cp.asarray(survive_lookup)
        
        birth_mask = neighbors < len(cp_birth) 
        birth_mask &= cp_birth[neighbors]
        
        survive_mask = neighbors < len(cp_survive)
        survive_mask &= cp_survive[neighbors]
        
        new_g = cp.zeros_like(g)
        new_g[(g > 0) & survive_mask] = g[(g > 0) & survive_mask]
        new_g[(g == 0) & birth_mask] = 1
        
        return cp.asnumpy(new_g)

# =====================================================================
# CELLULAR AUTOMATA WIDGET
# =====================================================================

class CellularAutomataWidget(QWidget):
    # PySide6 uses Signal instead of pyqtSignal
    generation_updated = Signal(int)

    def __init__(self, rows=100, cols=100, cell_size=6, max_state=4, rule="B3/S23"):
        super().__init__()
        self.rows, self.cols, self.cell_size = rows, cols, cell_size
        self.max_state = max_state
        self.backend = "Auto"
        
        self.grid = np.zeros((rows, cols), dtype=np.int32)
        self.initial_grid = np.zeros((rows, cols), dtype=np.int32)
        self.generation = 0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_generation)
        self.custom_rule_func = None
        self.rule_string = rule
        self.birth_lookup = np.zeros(9, dtype=np.bool_)
        self.survive_lookup = np.zeros(9, dtype=np.bool_)
        self.parse_rule(rule)

        self.dragging = False
        self.last_button = None
        self._render_buffer = None

        self.setFixedSize(cols * cell_size, rows * cell_size)
        self._generate_lut()

    def _generate_lut(self):
        self.lut = np.zeros((self.max_state, 3), dtype=np.uint8)
        self.lut[0] = [240, 240, 240]
        for state in range(1, self.max_state):
            hue = int(((state - 1) / max(1, self.max_state - 1)) * 270)
            color = QColor.fromHsv(hue, 255, 255)
            self.lut[state] = [color.red(), color.green(), color.blue()]

    def paintEvent(self, event):
        rgb_array = self.lut[self.grid] 
        self._render_buffer = rgb_array.copy()
        
        h, w, ch = self._render_buffer.shape
        bytes_per_line = 3 * w
        
        qimg = QImage(self._render_buffer.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        painter = QPainter(self)
        painter.drawImage(0, 0, qimg)
        
        if self.cell_size >= 8:
            painter.setPen(QColor(220, 220, 220))
            for r in range(self.rows + 1):
                painter.drawLine(0, r * self.cell_size, self.cols * self.cell_size, r * self.cell_size)
            for c in range(self.cols + 1):
                painter.drawLine(c * self.cell_size, 0, c * self.cell_size, self.rows * self.cell_size)

    def parse_rule(self, rule):
        self.rule_string = rule
        self.birth_lookup[:] = False
        self.survive_lookup[:] = False
        if "/" in rule:
            try:
                b, s = rule.split("/")
                for x in b:
                    if x.isdigit(): self.birth_lookup[int(x)] = True
                for x in s:
                    if x.isdigit(): self.survive_lookup[int(x)] = True
                self.custom_rule_func = None 
            except Exception:
                pass

    def set_custom_rule(self, code: str, parent_window=None):
        local_env = {"random": random}
        try:
            exec(code, {"__builtins__": {}}, local_env)
            if "rule" in local_env and callable(local_env["rule"]):
                self.custom_rule_func = local_env["rule"]
                return True, "Custom rule applied!"
            return False, "Error: Define a callable named 'rule'."
        except Exception as e:
            return False, f"Syntax Error:\n{str(e)}"

    def next_generation(self):
        if self.custom_rule_func:
            new_grid = np.zeros_like(self.grid)
            for r in range(self.rows):
                for c in range(self.cols):
                    try:
                        new_state = self.custom_rule_func(self.grid, r, c, self.max_state, random)
                        new_grid[r, c] = max(0, min(self.max_state - 1, int(new_state)))
                    except:
                        new_grid[r, c] = self.grid[r, c]
            self.grid = new_grid
        else:
            if self.backend == "CuPy" and HAS_CUPY:
                self.grid = evolve_cupy(self.grid, self.birth_lookup, self.survive_lookup)
            elif (self.backend == "Numba" or self.backend == "Auto") and HAS_NUMBA:
                self.grid = evolve_numba(self.grid, self.birth_lookup, self.survive_lookup)
            else:
                self.grid = evolve_python(self.grid, self.birth_lookup, self.survive_lookup)
                
        self.generation += 1
        self.generation_updated.emit(self.generation)
        self.update()

    def set_speed(self, interval): self.timer.setInterval(interval)
    def start(self): self.timer.start(50)
    def stop(self): self.timer.stop()
    def step(self):
        if not self.timer.isActive(): self.next_generation()

    def clear_grid(self):
        self.grid.fill(0)
        self.generation = 0
        self.generation_updated.emit(0)
        self.update()

    def reset(self):
        self.grid = self.initial_grid.copy()
        self.generation = 0
        self.generation_updated.emit(0)
        self.update()

    def save_state(self): self.initial_grid = self.grid.copy()

    def resize_grid(self, new_rows, new_cols):
        self.rows, self.cols = new_rows, new_cols
        self.grid = np.zeros((new_rows, new_cols), dtype=np.int32)
        self.initial_grid = np.zeros((new_rows, new_cols), dtype=np.int32)
        self.generation = 0
        self.generation_updated.emit(0)
        self.setFixedSize(new_cols * self.cell_size, new_rows * self.cell_size)
        self.update()

    def inject_pattern(self, pattern_name):
        pattern = PRESETS.get(pattern_name, [])
        if not pattern: self.clear_grid(); return
        max_r = max(r for r, c in pattern)
        max_c = max(c for r, c in pattern)
        start_r, start_c = (self.rows - max_r) // 2, (self.cols - max_c) // 2
        for dr, dc in pattern:
            r, c = start_r + dr, start_c + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = 1
        self.update()

    def save_to_file(self, path, rule_code=""):
        data = {
            "grid": self.grid.tolist(), "max_state": self.max_state,
            "rule_string": self.rule_string, "custom_rule_code": rule_code
        }
        with open(path, "w") as f: json.dump(data, f)

    def load_from_file(self, path):
        with open(path, "r") as f: data = json.load(f)
        self.grid = np.array(data.get("grid", self.grid), dtype=np.int32)
        self.max_state = data.get("max_state", self.max_state)
        self.rule_string = data.get("rule_string", "B3/S23")
        self.parse_rule(self.rule_string)
        self._generate_lut()
        self.save_state()
        self.generation = 0
        self.generation_updated.emit(0)
        self.update()
        return data.get("custom_rule_code", "")

    def mousePressEvent(self, event):
        self.dragging, self.last_button = True, event.button()
        self.toggle_cell(event, self.last_button)

    def mouseMoveEvent(self, event):
        if self.dragging: self.toggle_cell(event, self.last_button)

    def mouseReleaseEvent(self, event): self.dragging = False

    def toggle_cell(self, event, button):
        pos = event.position()
        col, row = int(pos.x() // self.cell_size), int(pos.y() // self.cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if button == Qt.MouseButton.LeftButton:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.grid[row, col] = (self.grid[row, col] + 1) % self.max_state
                else:
                    self.grid[row, col] = 1 if self.grid[row, col] == 0 else self.grid[row, col]
            elif button == Qt.MouseButton.RightButton:
                self.grid[row, col] = 0
            self.update()

# =====================================================================
# MAIN WINDOW
# =====================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU/Numba Accelerated Cellular Automata (PySide6)")
        self.resize(1000, 800)

        self.ca_widget = CellularAutomataWidget(rows=150, cols=150, cell_size=5, max_state=4)
        self.ca_widget.generation_updated.connect(self.update_gen_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.ca_widget)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setBackgroundRole(QPalette.ColorRole.Window)

        play_group = QGroupBox("Playback")
        play_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play"); self.play_btn.clicked.connect(self.ca_widget.start)
        self.step_btn = QPushButton("⏭ Step"); self.step_btn.clicked.connect(self.ca_widget.step)
        self.pause_btn = QPushButton("⏸ Pause"); self.pause_btn.clicked.connect(self.ca_widget.stop)
        self.gen_label = QLabel("Gen: 0"); self.gen_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        play_layout.addWidget(self.play_btn); play_layout.addWidget(self.step_btn); play_layout.addWidget(self.pause_btn)
        play_layout.addStretch(); play_layout.addWidget(self.gen_label)
        play_group.setLayout(play_layout)

        settings_group = QGroupBox("Hardware & Settings")
        settings_form = QFormLayout()
        
        self.backend_combo = QComboBox()
        backends = ["Auto"]
        if HAS_NUMBA: backends.append("Numba (CPU)")
        if HAS_CUPY: backends.append("CuPy (GPU)")
        backends.append("Pure Python")
        self.backend_combo.addItems(backends)
        self.backend_combo.currentTextChanged.connect(self.change_backend)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1); self.speed_slider.setMaximum(500); self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.ca_widget.set_speed)
        self.speed_label = QLabel("50 ms")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f"{v} ms"))
        speed_layout = QHBoxLayout(); speed_layout.addWidget(self.speed_slider); speed_layout.addWidget(self.speed_label)

        self.rows_spin = QSpinBox(); self.rows_spin.setRange(5, 2000); self.rows_spin.setValue(150)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(5, 2000); self.cols_spin.setValue(150)
        self.cell_size_spin = QSpinBox(); self.cell_size_spin.setRange(2, 50); self.cell_size_spin.setValue(5)
        self.cell_size_spin.valueChanged.connect(self.update_cell_size)
        
        self.max_state_spin = QSpinBox(); self.max_state_spin.setValue(4); self.max_state_spin.setMinimum(2)
        self.max_state_spin.valueChanged.connect(self.update_max_state)

        self.rule_combo = QComboBox()
        self.rule_combo.addItems(["B3/S23 (Conway's Life)", "B36/S23 (HighLife)", "B3678/S34678 (Day & Night)", "B1357/S1357 (Replicator)"])
        self.rule_combo.setEditable(True); self.rule_combo.setEditText("B3/S23")
        self.rule_combo.editTextChanged.connect(self.apply_standard_rule)

        settings_form.addRow("Compute Backend:", self.backend_combo)
        settings_form.addRow("Speed:", speed_layout)
        settings_form.addRow("Rows:", self.rows_spin); settings_form.addRow("Cols:", self.cols_spin)
        settings_form.addRow("Cell Size:", self.cell_size_spin)
        settings_form.addRow("Max States:", self.max_state_spin)
        settings_form.addRow("Standard Rule:", self.rule_combo)
        settings_group.setLayout(settings_form)

        actions_group = QGroupBox("Grid Actions")
        actions_layout = QHBoxLayout()
        self.resize_btn = QPushButton("Apply Grid Size"); self.resize_btn.clicked.connect(self.apply_grid_size)
        self.preset_combo = QComboBox(); self.preset_combo.addItems(PRESETS.keys())
        self.inject_btn = QPushButton("Inject"); self.inject_btn.clicked.connect(lambda: self.ca_widget.inject_pattern(self.preset_combo.currentText()))
        self.save_state_btn = QPushButton("💾 Save State"); self.save_state_btn.clicked.connect(self.ca_widget.save_state)
        self.reset_btn = QPushButton("↩ Reset"); self.reset_btn.clicked.connect(self.ca_widget.reset)
        self.clear_btn = QPushButton("🗑 Clear"); self.clear_btn.clicked.connect(self.ca_widget.clear_grid)
        actions_layout.addWidget(self.resize_btn); actions_layout.addWidget(self.preset_combo); actions_layout.addWidget(self.inject_btn)
        actions_layout.addStretch(); actions_layout.addWidget(self.save_state_btn); actions_layout.addWidget(self.reset_btn); actions_layout.addWidget(self.clear_btn)
        actions_group.setLayout(actions_layout)

        rule_group = QGroupBox("Custom Python Rule (Disabled in Numba/CuPy modes)")
        rule_layout = QVBoxLayout()
        self.rule_editor = QTextEdit()
        self.rule_editor.setEnabled(False)
        self.rule_editor.setPlaceholderText("Note: Custom rules bypass JIT/GPU acceleration and run in pure Python.")
        self.apply_rule_btn = QPushButton("Apply Custom Rule"); self.apply_rule_btn.setEnabled(False)
        self.apply_rule_btn.clicked.connect(self.apply_rule)
        rule_layout.addWidget(self.rule_editor); rule_layout.addWidget(self.apply_rule_btn)
        rule_group.setLayout(rule_layout)
        self.rule_group_ref = rule_group # Keep reference to change title later

        file_group = QGroupBox("File")
        file_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save to JSON"); self.save_btn.clicked.connect(self.save_config)
        self.load_btn = QPushButton("Load from JSON"); self.load_btn.clicked.connect(self.load_config)
        file_layout.addWidget(self.save_btn); file_layout.addWidget(self.load_btn)
        file_group.setLayout(file_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scroll_area, stretch=1)
        main_layout.addWidget(play_group)
        
        bottom_layout = QHBoxLayout()
        left_panel = QVBoxLayout(); left_panel.addWidget(settings_group); left_panel.addWidget(actions_group); left_panel.addWidget(file_group)
        right_panel = QVBoxLayout(); right_panel.addWidget(rule_group)
        bottom_layout.addLayout(left_panel, stretch=1); bottom_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(bottom_layout)

        container = QWidget(); container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.statusBar().showMessage(f"Initialized. Numba: {HAS_NUMBA} | CuPy: {HAS_CUPY}. Left: Draw | Right: Erase | Shift+Left: Cycle.")

    def update_gen_label(self, gen): self.gen_label.setText(f"Gen: {gen}")

    def change_backend(self, text):
        backend_map = {"Numba (CPU)": "Numba", "CuPy (GPU)": "CuPy", "Pure Python": "Python"}
        self.ca_widget.backend = backend_map.get(text, "Auto")
        
        is_custom_allowed = (text == "Pure Python")
        self.rule_editor.setEnabled(is_custom_allowed)
        self.apply_rule_btn.setEnabled(is_custom_allowed)
        
        if is_custom_allowed:
            self.rule_group_ref.setTitle("Custom Python Rule")
        else:
            self.rule_group_ref.setTitle("Custom Python Rule (Disabled in Numba/CuPy modes)")

    def update_max_state(self, value):
        self.ca_widget.max_state = value
        self.ca_widget._generate_lut()

    def update_cell_size(self, value):
        self.ca_widget.cell_size = value
        self.ca_widget.setFixedSize(self.ca_widget.cols * value, self.ca_widget.rows * value)
        self.ca_widget.update()

    def apply_grid_size(self):
        self.ca_widget.resize_grid(self.rows_spin.value(), self.cols_spin.value())

    def apply_standard_rule(self, text):
        if text: self.ca_widget.parse_rule(text)

    def apply_rule(self):
        code = self.rule_editor.toPlainText()
        success, msg = self.ca_widget.set_custom_rule(code, self)
        if not success: QMessageBox.warning(self, "Rule Error", msg)
        else: QMessageBox.information(self, "Success", msg)

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON Files (*.json)")
        if path: self.ca_widget.save_to_file(path, self.rule_editor.toPlainText())

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if path:
            try:
                custom_code = self.ca_widget.load_from_file(path)
                self.max_state_spin.setValue(self.ca_widget.max_state)
                self.rows_spin.setValue(self.ca_widget.rows)
                self.cols_spin.setValue(self.ca_widget.cols)
                self.rule_combo.setEditText(self.ca_widget.rule_string)
                if custom_code: self.rule_editor.setPlainText(custom_code)
                self.statusBar().showMessage(f"Loaded {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())