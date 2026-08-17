import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QComboBox,
    QSpinBox, QCheckBox, QSlider, QListWidget, QListWidgetItem,
    QTabWidget, QToolBar, QStatusBar, QFileDialog, QColorDialog,
    QSplitter, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QPoint, QRect
from PySide6.QtGui import (
    QPainter, QColor, QPen, QImage, QAction, QFont, QIcon, QBrush
)

# ==========================================
# Cellular Automata Engine (NumPy Vectorized)
# ==========================================
class CAEngine:
    def __init__(self, width=200, height=200):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.num_states = 2
        self.rule_type = "lifelike"  # "lifelike" or "cyclic"
        self.birth = {3}
        self.survival = {2, 3}
        self.cyclic_threshold = 1
        self.neighborhood = "moore"  # "moore" or "vonneumann"
        self.toroidal = True
        self.generation = 0

    def set_size(self, w, h):
        self.width, self.height = w, h
        self.grid = np.zeros((h, w), dtype=np.uint8)
        self.generation = 0

    def clear(self):
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.generation = 0

    def randomize(self, density=0.3):
        if self.rule_type == "cyclic":
            self.grid = np.random.randint(0, self.num_states, (self.height, self.width))
        else:
            self.grid = (np.random.random((self.height, self.width)) < density).astype(np.uint8)
        self.generation = 0

    def _count_neighbors(self, state=None):
        if state is None:
            mask = (self.grid > 0).astype(np.int32)
        else:
            mask = (self.grid == state).astype(np.int32)
        
        total = np.zeros_like(mask)
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        if self.neighborhood == "vonneumann":
            shifts = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            
        for dy, dx in shifts:
            if self.toroidal:
                total += np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            else:
                # Simplified bounded shift (padding with 0)
                padded = np.pad(mask, 1, mode='constant')
                total += padded[1+dy:1+dy+self.height, 1+dx:1+dx+self.width]
        return total

    def step(self):
        if self.rule_type == "lifelike":
            self._step_lifelike()
        elif self.rule_type == "cyclic":
            self._step_cyclic()
        self.generation += 1

    def _step_lifelike(self):
        neighbors = self._count_neighbors()
        new_grid = np.zeros_like(self.grid)
        
        # Birth
        for b in self.birth:
            new_grid[(self.grid == 0) & (neighbors == b)] = 1
            
        # Survival
        for s in self.survival:
            new_grid[(self.grid == 1) & (neighbors == s)] = 1
            
        # Aging (if num_states > 2, living cells age)
        if self.num_states > 2:
            aging_mask = (self.grid > 0) & (self.grid < self.num_states - 1)
            new_grid[aging_mask] = self.grid[aging_mask] + 1
            max_state_mask = self.grid == self.num_states - 1
            new_grid[max_state_mask] = self.grid[max_state_mask] # Stays at max state until it dies
            
        self.grid = new_grid

    def _step_cyclic(self):
        next_state = (self.grid + 1) % self.num_states
        neighbors = self._count_neighbors(state=next_state)
        self.grid = np.where(neighbors >= self.cyclic_threshold, next_state, self.grid)

    def parse_bs(self, text):
        text = text.strip().upper()
        b, s = set(), set()
        parts = text.split('/')
        for p in parts:
            if p.startswith('B'):
                b = {int(c) for c in p[1:] if c.isdigit()}
            elif p.startswith('S'):
                s = {int(c) for c in p[1:] if c.isdigit()}
        return b, s


# ==========================================
# Color Palette Manager
# ==========================================
class PaletteManager:
    def __init__(self):
        self.colors = np.zeros((256, 3), dtype=np.uint8)
        self.set_solid(0, 0, 0)
        self.set_solid(1, 255, 255, 255)
        
    def set_solid(self, state, r, g, b):
        if state < 256:
            self.colors[state] = [r, g, b]
            
    def generate_gradient_rgb(self, c1, c2, n):
        for i in range(n):
            t = i / max(1, n - 1)
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            self.colors[i] = [r, g, b]
            
    def generate_gradient_hsv(self, c1, c2, n):
        from colorsys import rgb_to_hsv, hsv_to_rgb
        h1, s1, v1 = rgb_to_hsv(*[c/255.0 for c in c1])
        h2, s2, v2 = rgb_to_hsv(*[c/255.0 for c in c2])
        
        for i in range(n):
            t = i / max(1, n - 1)
            h = h1 + (h2 - h1) * t
            s = s1 + (s2 - s1) * t
            v = v1 + (v2 - v1) * t
            r, g, b = hsv_to_rgb(h, s, v)
            self.colors[i] = [int(r*255), int(g*255), int(b*255)]

    def get_palette_image(self, num_states):
        return self.colors[:num_states]


# ==========================================
# CA Canvas Widget (High Perf Rendering)
# ==========================================
class CACanvas(QWidget):
    def __init__(self, engine, palette, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.palette = palette
        
        self.zoom = 3.0
        self.offset_x = 0
        self.offset_y = 0
        
        self._dragging = False
        self._last_pos = None
        self._drawing = False
        self.draw_state = 1
        self.brush_size = 1
        self.symmetry = 1  # 1=off, 2=X-axis, 4=quad
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _render_image(self):
        h, w = self.engine.height, self.engine.width
        palette_img = self.palette.get_palette_image(self.engine.num_states)
        
        # Vectorized palette mapping
        rgb_array = palette_img[self.engine.grid]
        rgba_array = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_array[..., :3] = rgb_array
        rgba_array[..., 3] = 255
        
        # QImage from data
        qimg = QImage(rgba_array.data, w, h, w * 4, QImage.Format_RGBA8888)
        # Must keep reference to prevent garbage collection before drawing
        self._render_buffer = rgba_array 
        return qimg

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(30, 30, 30))

        img = self._render_image()
        w, h = self.engine.width, self.engine.height
        cs = self.zoom
        tw, th = int(w * cs), int(h * cs)
        ox = self.offset_x + (self.width() - tw) // 2
        oy = self.offset_y + (self.height() - th) // 2

        p.drawImage(QRect(ox, oy, tw, th), img)

        # Grid lines
        if cs >= 8:
            p.setPen(QPen(QColor(50, 50, 50, 80), 1))
            for x in range(w + 1):
                p.drawLine(int(ox + x * cs), int(oy), int(ox + x * cs), int(oy + th))
            for y in range(h + 1):
                p.drawLine(int(ox), int(oy + y * cs), int(ox + tw), int(oy + y * cs))
        p.end()

    def _screen_to_grid(self, pos):
        w, h = self.engine.width, self.engine.height
        cs = self.zoom
        tw, th = int(w * cs), int(h * cs)
        ox = self.offset_x + (self.width() - tw) // 2
        oy = self.offset_y + (self.height() - th) // 2
        gx = int((pos.x() - ox) / cs)
        gy = int((pos.y() - oy) / cs)
        return gx, gy

    def _paint_cell(self, pos, state):
        gx, gy = self._screen_to_grid(pos)
        w, h = self.engine.width, self.engine.height
        
        coords = [(gx, gy)]
        if self.symmetry >= 2:
            coords.append((w - 1 - gx, gy))
        if self.symmetry >= 4:
            coords.append((gx, h - 1 - gy))
            coords.append((w - 1 - gx, h - 1 - gy))

        for cx, cy in coords:
            for dy in range(-self.brush_size + 1, self.brush_size):
                for dx in range(-self.brush_size + 1, self.brush_size):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        self.engine.grid[ny, nx] = state
        self.update()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.5, min(self.zoom * factor, 50.0))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            self._drawing = True
            self._paint_cell(event.pos(), self.draw_state)
        elif event.button() == Qt.RightButton:
            self._drawing = True
            self._paint_cell(event.pos(), 0)

    def mouseMoveEvent(self, event):
        if self._dragging and self._last_pos:
            delta = event.pos() - self._last_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self._last_pos = event.pos()
            self.update()
        elif self._drawing:
            if event.buttons() & Qt.LeftButton:
                self._paint_cell(event.pos(), self.draw_state)
            elif event.buttons() & Qt.RightButton:
                self._paint_cell(event.pos(), 0)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() in (Qt.LeftButton, Qt.RightButton):
            self._drawing = False

    def export_image(self, path):
        img = self._render_image()
        img.save(path)


# ==========================================
# Main Window & UI Layout
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellular Automata Art Studio")
        self.resize(1200, 800)

        self.engine = CAEngine(150, 150)
        self.palette = PaletteManager()
        
        self.running = false
        self.speed = 100  # ms per step

        self._setup_ui()
        self._setup_timers()
        self._apply_stylesheet()
        self._update_palette_ui()

    def _setup_ui(self):
        # Toolbar
        toolbar = QToolBar("Controls")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_play)
        toolbar.addWidget(self.btn_play)

        btn_step = QPushButton("⏭ Step")
        btn_step.clicked.connect(self.do_step)
        toolbar.addWidget(btn_step)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" Speed: "))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(150)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'speed') or self.timer.setInterval(v))
        toolbar.addWidget(self.speed_slider)

        toolbar.addSeparator()

        btn_export = QPushButton("💾 Export PNG")
        btn_export.clicked.connect(self.export_image)
        toolbar.addWidget(btn_export)

        # Main Layout Splitter
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Canvas
        self.canvas = CACanvas(self.engine, self.palette)
        splitter.addWidget(self.canvas)

        # Right Sidebar Tabs
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(320)
        splitter.addWidget(self.tabs)

        self.tabs.addTab(self._create_sim_tab(), "Simulation")
        self.tabs.addTab(self._create_rules_tab(), "Rules")
        self.tabs.addTab(self._create_palette_tab(), "Palette")

        # Status Bar
        self.status_label = QLabel("Gen: 0 | Pop: 0")
        self.statusBar().addPermanentWidget(self.status_label)

    def _create_sim_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Grid Settings
        g_grp = QGroupBox("Grid Settings")
        g_lay = QVBoxLayout(g_grp)
        
        h_lay = QHBoxLayout()
        h_lay.addWidget(QLabel("W:"))
        self.sp_w = QSpinBox(); self.sp_w.setRange(10, 1000); self.sp_w.setValue(150)
        h_lay.addWidget(self.sp_w)
        h_lay.addWidget(QLabel("H:"))
        self.sp_h = QSpinBox(); self.sp_h.setRange(10, 1000); self.sp_h.setValue(150)
        h_lay.addWidget(self.sp_h)
        g_lay.addLayout(h_lay)

        btn_resize = QPushButton("Resize Grid")
        btn_resize.clicked.connect(self.resize_grid)
        g_lay.addWidget(btn_resize)

        btn_rand = QPushButton("Randomize")
        btn_rand.clicked.connect(lambda: (self.engine.randomize(0.3), self.canvas.update()))
        g_lay.addWidget(btn_rand)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: (self.engine.clear(), self.canvas.update()))
        g_lay.addWidget(btn_clear)

        self.chk_toroidal = QCheckBox("Toroidal (Wrap Edges)")
        self.chk_toroidal.setChecked(True)
        self.chk_toroidal.toggled.connect(lambda v: setattr(self.engine, 'toroidal', v))
        g_lay.addWidget(self.chk_toroidal)

        layout.addWidget(g_grp)

        # Drawing Settings
        d_grp = QGroupBox("Drawing Tool")
        d_lay = QVBoxLayout(d_grp)

        h_lay2 = QHBoxLayout()
        h_lay2.addWidget(QLabel("Brush Size:"))
        self.sp_brush = QSpinBox(); self.sp_brush.setRange(1, 20); self.sp_brush.setValue(1)
        self.sp_brush.valueChanged.connect(lambda v: setattr(self.canvas, 'brush_size', v))
        h_lay2.addWidget(self.sp_brush)
        d_lay.addLayout(h_lay2)

        h_lay3 = QHBoxLayout()
        h_lay3.addWidget(QLabel("Draw State:"))
        self.sp_dstate = QSpinBox(); self.sp_dstate.setRange(1, 255); self.sp_dstate.setValue(1)
        self.sp_dstate.valueChanged.connect(lambda v: setattr(self.canvas, 'draw_state', v))
        h_lay3.addWidget(self.sp_dstate)
        d_lay.addLayout(h_lay3)

        h_lay4 = QHBoxLayout()
        h_lay4.addWidget(QLabel("Symmetry:"))
        self.cb_sym = QComboBox()
        self.cb_sym.addItems(["None", "2-X Axis", "4-Quadrants"])
        self.cb_sym.currentIndexChanged.connect(self._update_symmetry)
        h_lay4.addWidget(self.cb_sym)
        d_lay.addLayout(h_lay4)

        layout.addWidget(d_grp)
        layout.addStretch()
        return widget

    def _create_rules_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        r_grp = QGroupBox("Rule Type")
        r_lay = QVBoxLayout(r_grp)
        
        self.cb_rule_type = QComboBox()
        self.cb_rule_type.addItems(["Lifelike (B/S)", "Cyclic"])
        self.cb_rule_type.currentTextChanged.connect(self._rule_type_changed)
        r_lay.addWidget(self.cb_rule_type)

        n_lay = QHBoxLayout()
        n_lay.addWidget(QLabel("Neighborhood:"))
        self.cb_neigh = QComboBox()
        self.cb_neigh.addItems(["Moore (8)", "Von Neumann (4)"])
        self.cb_neigh.currentIndexChanged.connect(lambda i: setattr(self.engine, 'neighborhood', 'moore' if i==0 else 'vonneumann'))
        n_lay.addWidget(self.cb_neigh)
        r_lay.addLayout(n_lay)

        layout.addWidget(r_grp)

        # Lifelike Group
        self.lifelike_grp = QGroupBox("Lifelike Rules (B/S)")
        l_lay = QVBoxLayout(self.lifelike_grp)

        l_lay.addWidget(QLabel("Presets:"))
        self.cb_life_presets = QComboBox()
        self.cb_life_presets.addItems([
            "B3/S23 (Conway's Life)", 
            "B36/S23 (Highlife)", 
            "B2/S (Seeds)", 
            "B3678/S34678 (Day & Night)",
            "B1/S1 (Gnarl)"
        ])
        self.cb_life_presets.currentTextChanged.connect(self._lifelike_preset_selected)
        l_lay.addWidget(self.cb_life_presets)

        l_lay.addWidget(QLabel("Custom (e.g. B3/S23):"))
        self.le_bs = QLineEdit("B3/S23")
        l_lay.addWidget(self.le_bs)

        btn_apply_bs = QPushButton("Apply Custom B/S")
        btn_apply_bs.clicked.connect(self._apply_bs)
        l_lay.addWidget(btn_apply_bs)

        l_lay.addWidget(QLabel("Num States (2=Std, >2=Aging):"))
        self.sp_lstates = QSpinBox(); self.sp_lstates.setRange(2, 64); self.sp_lstates.setValue(2)
        self.sp_lstates.valueChanged.connect(lambda v: setattr(self.engine, 'num_states', v))
        l_lay.addWidget(self.sp_lstates)

        layout.addWidget(self.lifelike_grp)

        # Cyclic Group
        self.cyclic_grp = QGroupBox("Cyclic Rules")
        c_lay = QVBoxLayout(self.cyclic_grp)

        c_lay.addWidget(QLabel("Number of States:"))
        self.sp_cstates = QSpinBox(); self.sp_cstates.setRange(3, 64); self.sp_cstates.setValue(3)
        self.sp_cstates.valueChanged.connect(lambda v: setattr(self.engine, 'num_states', v))
        c_lay.addWidget(self.sp_cstates)

        c_lay.addWidget(QLabel("Threshold to Advance:"))
        self.sp_cthresh = QSpinBox(); self.sp_cthresh.setRange(1, 8); self.sp_cthresh.setValue(1)
        self.sp_cthresh.valueChanged.connect(lambda v: setattr(self.engine, 'cyclic_threshold', v))
        c_lay.addWidget(self.sp_cthresh)

        btn_apply_cyclic = QPushButton("Apply Cyclic Settings")
        btn_apply_cyclic.clicked.connect(lambda: setattr(self.engine, 'rule_type', 'cyclic'))
        c_lay.addWidget(btn_apply_cyclic)

        self.cyclic_grp.setVisible(False)
        layout.addWidget(self.cyclic_grp)

        layout.addStretch()
        return widget

    def _create_palette_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        p_grp = QGroupBox("Gradient Generator")
        p_lay = QVBoxLayout(p_grp)

        p_lay.addWidget(QLabel("Color 1 (Background/State 0):"))
        self.btn_c1 = QPushButton(); self.btn_c1.setStyleSheet("background-color: #000000;")
        self.btn_c1.clicked.connect(lambda: self._pick_color(self.btn_c1))
        p_lay.addWidget(self.btn_c1)

        p_lay.addWidget(QLabel("Color 2 (Foreground/Last State):"))
        self.btn_c2 = QPushButton(); self.btn_c2.setStyleSheet("background-color: #FFFFFF;")
        self.btn_c2.clicked.connect(lambda: self._pick_color(self.btn_c2))
        p_lay.addWidget(self.btn_c2)

        p_lay.addWidget(QLabel("Interpolation Mode:"))
        self.cb_interp = QComboBox()
        self.cb_interp.addItems(["RGB", "HSV (Rainbow safe)"])
        p_lay.addWidget(self.cb_interp)

        btn_gen = QPushButton("Generate Palette")
        btn_gen.clicked.connect(self._generate_palette)
        p_lay.addWidget(btn_gen)

        layout.addWidget(p_grp)

        pe_grp = QGroupBox("Palette Presets")
        pe_lay = QVBoxLayout(pe_grp)
        
        self.cb_ppreset = QComboBox()
        self.cb_ppreset.addItems(["Monochrome", "Fire", "Ocean", "Neon", "Plasma", "Cyberpunk"])
        pe_lay.addWidget(self.cb_ppreset)
        
        btn_apply_p = QPushButton("Apply Preset")
        btn_apply_p.clicked.connect(self._apply_palette_preset)
        pe_lay.addWidget(btn_apply_p)

        layout.addWidget(pe_grp)

        # Preview
        self.palette_preview = QListWidget()
        self.palette_preview.setFixedHeight(120)
        self.palette_preview.setViewMode(QListWidget.IconMode)
        self.palette_preview.setIconSize(QSize(20, 20))
        layout.addWidget(QLabel("Palette Preview:"))
        layout.addWidget(self.palette_preview)

        layout.addStretch()
        return widget

    def _setup_timers(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.do_step)

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { background-color: #2b2b2b; color: #dddddd; font-size: 13px; }
            QToolBar { background-color: #333333; border: none; spacing: 5px; padding: 5px; }
            QPushButton { background-color: #444444; border: 1px solid #555555; border-radius: 4px; padding: 5px 10px; }
            QPushButton:hover { background-color: #555555; }
            QPushButton:checked { background-color: #dd7733; color: white; }
            QGroupBox { border: 1px solid #555555; border-radius: 4px; margin-top: 10px; padding-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QSpinBox, QLineEdit, QComboBox { background-color: #3c3c3c; border: 1px solid #555555; border-radius: 4px; padding: 3px; }
            QTabWidget::pane { border: 1px solid #555555; }
            QTabBar::tab { background-color: #333333; padding: 8px 20px; border: 1px solid #555555; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #444444; color: #ffaa66; }
            QSlider::groove:horizontal { background: #555555; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #dd7733; width: 14px; margin: -4px 0; border-radius: 7px; }
            QStatusBar { background-color: #333333; color: #aaaaaa; }
        """)

    # --- Actions & Logic ---

    def toggle_play(self, checked):
        if checked:
            self.btn_play.setText("⏸ Pause")
            self.timer.start(self.speed)
        else:
            self.btn_play.setText("▶ Play")
            self.timer.stop()

    def do_step(self):
        self.engine.step()
        pop = int(np.sum(self.engine.grid > 0))
        self.status_label.setText(f"Gen: {self.engine.generation} | Pop: {pop} | Zoom: {self.canvas.zoom:.1f}x")
        self.canvas.update()

    def resize_grid(self):
        self.engine.set_size(self.sp_w.value(), self.sp_h.value())
        self.canvas.update()

    def export_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Image", "ca_art.png", "PNG Files (*.png)")
        if path:
            self.canvas.export_image(path)

    def _update_symmetry(self, idx):
        self.canvas.symmetry = [1, 2, 4][idx]

    def _rule_type_changed(self, text):
        is_cyclic = "Cyclic" in text
        self.cyclic_grp.setVisible(is_cyclic)
        self.lifelike_grp.setVisible(not is_cyclic)
        if is_cyclic:
            self.engine.rule_type = "cyclic"
            self.engine.num_states = self.sp_cstates.value()
        else:
            self.engine.rule_type = "lifelike"
            self.engine.num_states = self.sp_lstates.value()

    def _lifelike_preset_selected(self, text):
        if "B3/S23" in text: self.le_bs.setText("B3/S23")
        elif "B36/S23" in text: self.le_bs.setText("B36/S23")
        elif "B2/S" in text: self.le_bs.setText("B2/S")
        elif "B3678/S34678" in text: self.le_bs.setText("B3678/S34678")
        elif "B1/S1" in text: self.le_bs.setText("B1/S1")
        self._apply_bs()

    def _apply_bs(self):
        b, s = self.engine.parse_bs(self.le_bs.text())
        self.engine.birth = b
        self.engine.survival = s
        self.engine.num_states = self.sp_lstates.value()

    def _pick_color(self, btn):
        current = btn.palette().color(btn.backgroundRole())
        color = QColorDialog.getColor(current, self)
        if color.isValid():
            btn.setStyleSheet(f"background-color: {color.name()};")
            btn.setProperty('color_val', (color.red(), color.green(), color.blue()))

    def _generate_palette(self):
        c1 = self.btn_c1.property('color_val') or (0,0,0)
        c2 = self.btn_c2.property('color_val') or (255,255,255)
        n = self.engine.num_states
        
        if "HSV" in self.cb_interp.currentText():
            self.palette.generate_gradient_hsv(c1, c2, n)
        else:
            self.palette.generate_gradient_rgb(c1, c2, n)
            
        self._update_palette_ui()
        self.canvas.update()

    def _apply_palette_preset(self):
        n = self.engine.num_states
        preset = self.cb_ppreset.currentText()
        
        if preset == "Monochrome":
            self.palette.generate_gradient_rgb((0,0,0), (255,255,255), n)
        elif preset == "Fire":
            self.palette.generate_gradient_rgb((0,0,0), (255,100,0), n//2)
            self.palette.generate_gradient_rgb((255,100,0), (255,255,50), n - n//2)
        elif preset == "Ocean":
            self.palette.generate_gradient_rgb((0,10,30), (0,150,200), n)
        elif preset == "Neon":
            self.palette.generate_gradient_hsv((0,255,100), (255,0,200), n)
        elif preset == "Plasma":
            self.palette.generate_gradient_hsv((200,0,255), (255,255,0), n)
        elif preset == "Cyberpunk":
            self.palette.generate_gradient_rgb((0,0,10), (255,0,150), n//2)
            self.palette.generate_gradient_rgb((255,0,150), (0,255,255), n - n//2)
            
        self._update_palette_ui()
        self.canvas.update()

    def _update_palette_ui(self):
        self.palette_preview.clear()
        colors = self.palette.get_palette_image(self.engine.num_states)
        for i, (r, g, b) in enumerate(colors):
            item = QListWidgetItem(f"{i}")
            item.setBackground(QColor(r, g, b))
            self.palette_preview.addItem(item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())