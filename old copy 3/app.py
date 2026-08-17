import sys
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QSpinBox, QComboBox, QSlider,
    QGroupBox, QScrollArea, QSizePolicy, QCheckBox, QFileDialog, QStatusBar
)
from PySide6.QtCore import Qt, QTimer, Signal, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QShortcut, QKeySequence, QPolygonF
)

# ─── Dark Theme Stylesheet ────────────────────────────────────────────────────

DARK_STYLE = """
    QWidget {
        background-color: #1a1a2e; color: #c8c8d4;
        font-family: "Segoe UI","Helvetica Neue",sans-serif; font-size: 12px;
    }
    QTabWidget::pane { border:1px solid #2e2e48; background:#1a1a2e; }
    QTabBar::tab {
        background:#22223a; color:#7878a0; padding:9px 18px;
        border:1px solid #2e2e48; border-bottom:none;
        border-top-left-radius:6px; border-top-right-radius:6px;
        font-weight:bold; margin-right:2px;
    }
    QTabBar::tab:selected { background:#1a1a2e; color:#00d4a8; border-bottom:2px solid #00d4a8; }
    QTabBar::tab:hover:!selected { background:#2a2a4e; color:#a0a0c0; }
    QPushButton {
        background:#2e2e50; color:#c8c8d4; border:1px solid #3e3e60;
        border-radius:5px; padding:5px 12px; font-weight:bold;
    }
    QPushButton:hover { background:#3e3e68; border-color:#00d4a8; color:#fff; }
    QPushButton:pressed { background:#50508a; }
    QPushButton:disabled { background:#1e1e34; color:#555570; border-color:#2a2a40; }
    QSpinBox, QComboBox {
        background:#22223a; color:#c8c8d4; border:1px solid #3e3e60;
        border-radius:4px; padding:4px 8px; min-width:50px;
    }
    QComboBox::drop-down { border:none; width:20px; }
    QComboBox QAbstractItemView { background:#22223a; color:#c8c8d4; selection-background-color:#3e3e68; }
    QSlider::groove:horizontal { border:1px solid #3e3e60; height:6px; background:#22223a; border-radius:3px; }
    QSlider::handle:horizontal {
        background:#00d4a8; border:none; width:16px; height:16px;
        margin:-5px 0; border-radius:8px;
    }
    QGroupBox {
        color:#9090b0; border:1px solid #2e2e48; border-radius:6px;
        margin-top:14px; padding-top:18px;
    }
    QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; }
    QScrollArea { border:none; }
    QScrollBar:vertical { background:#1a1a2e; width:10px; }
    QScrollBar::handle:vertical { background:#3e3e60; border-radius:5px; min-height:30px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
    QCheckBox { spacing:3px; }
    QCheckBox::indicator { width:15px; height:15px; border-radius:3px; border:1px solid #3e3e60; background:#22223a; }
    QCheckBox::indicator:checked { background:#00d4a8; border-color:#00d4a8; }
    QStatusBar { background:#16162a; color:#7878a0; font-size:11px; border-top:1px solid #2e2e48; }
"""

# ─── Shared Zoom/Pan Canvas Base ──────────────────────────────────────────────

class BaseGridCanvas(QWidget):
    """Base class providing zoom/pan functionality for 2D grids."""
    def __init__(self, rows, cols, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._dragging_pan = False
        self._last_pan_pos = None
        self._cs = 8
        self._xo = 0
        self._yo = 0
        
        self.setMinimumSize(400, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def _recalc(self):
        cw = self.width() / self.cols
        ch = self.height() / self.rows
        base_cs = max(2, min(cw, ch))
        self._cs = base_cs * self.zoom_level
        gw = self.cols * self._cs
        gh = self.rows * self._cs
        self._xo = (self.width() - gw) / 2 + self.pan_x
        self._yo = (self.height() - gh) / 2 + self.pan_y

    def _cell_at(self, pos):
        c = int((pos.x() - self._xo) / self._cs)
        r = int((pos.y() - self._yo) / self._cs)
        return (r, c) if 0 <= r < self.rows and 0 <= c < self.cols else None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        old_zoom = self.zoom_level
        self.zoom_level = max(0.5, min(15.0, self.zoom_level * factor))
        mx = event.position().x()
        my = event.position().y()
        self.pan_x = mx - (mx - self.pan_x) * (self.zoom_level / old_zoom)
        self.pan_y = my - (my - self.pan_y) * (self.zoom_level / old_zoom)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._dragging_pan = True
            self._last_pan_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging_pan:
            pos = event.position()
            dx = pos.x() - self._last_pan_pos.x()
            dy = pos.y() - self._last_pan_pos.y()
            self.pan_x += dx
            self.pan_y += dy
            self._last_pan_pos = pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._dragging_pan = False
            self.setCursor(Qt.ArrowCursor)

    def reset_view(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

# ─── 1D Elementary CA Engine ───────────────────────────────────────────────────

class ElementaryCA:
    def __init__(self, width=301, rule=30):
        self.width = width
        self.rule = rule
        self.generations = []
        self.reset("single")

    def reset(self, mode="single"):
        self.generations.clear()
        if mode == "single":
            row = [0] * self.width; row[self.width // 2] = 1
        else:
            row = [random.randint(0, 1) for _ in range(self.width)]
        self.generations.append(row)

    def step(self):
        cur = self.generations[-1]
        bits = [(self.rule >> i) & 1 for i in range(8)]
        nxt = [0] * self.width
        for i in range(self.width):
            idx = (cur[(i - 1) % self.width] << 2) | (cur[i] << 1) | cur[(i + 1) % self.width]
            nxt[i] = bits[idx]
        self.generations.append(nxt)

    def toggle_cell(self, col):
        if self.generations:
            self.generations[0][col] ^= 1

# ─── Rule Visualizer / Builder ────────────────────────────────────────────────

class RuleVisualizerWidget(QWidget):
    rule_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rule = 30; self.cs = 18; self._out_rects = []
        self.setFixedHeight(100); self.setCursor(Qt.PointingHandCursor)

    def set_rule(self, r): self.rule = r; self.update()

    def mousePressEvent(self, event):
        for i, rect in enumerate(self._out_rects):
            if rect.contains(event.position()):
                self.rule ^= (1 << i)
                self.rule_changed.emit(self.rule)
                self.update(); break

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        bg, on, off = QColor(26, 26, 46), QColor(0, 212, 168), QColor(50, 50, 70)
        p.fillRect(self.rect(), bg)
        cs = self.cs; bw = 3 * cs + 2; sp = bw + 16
        tw = 8 * sp - 16; x0 = max(8, (self.width() - tw) // 2)
        yi = 8; yo = yi + cs + 24; self._out_rects.clear()

        for i in range(8):
            l, c, r_ = (i >> 2) & 1, (i >> 1) & 1, i & 1; out = (self.rule >> i) & 1
            x = x0 + i * sp
            for j, v in enumerate((l, c, r_)):
                p.setBrush(on if v else off); p.setPen(QPen(QColor(80, 80, 100), 1))
                p.drawRoundedRect(QRectF(x + j * (cs + 1), yi, cs, cs), 2, 2)
            mx = x + bw / 2
            p.setPen(QPen(QColor(120, 120, 150), 2))
            p.drawLine(int(mx), yi + cs + 4, int(mx), yi + cs + 14)
            p.drawLine(int(mx), yi + cs + 14, int(mx - 4), yi + cs + 9)
            p.drawLine(int(mx), yi + cs + 14, int(mx + 4), yi + cs + 9)
            ox = x + cs + 1 - cs // 4; out_rect = QRectF(ox, yo, cs, cs)
            self._out_rects.append(out_rect)
            p.setBrush(on if out else off)
            p.setPen(QPen(QColor(255, 220, 60) if out else QColor(80, 80, 100), 2 if out else 1))
            p.drawRoundedRect(out_rect, 2, 2)
            p.setPen(QColor(30, 30, 30) if out else QColor(140, 140, 160))
            p.setFont(QFont("Segoe UI", 8, QFont.Bold)); p.drawText(out_rect, Qt.AlignCenter, str(out))

        p.setPen(QColor(90, 90, 120)); p.setFont(QFont("Segoe UI", 8))
        p.drawText(self.rect().adjusted(0, 0, -8, -4), Qt.AlignRight | Qt.AlignBottom, "click output to toggle")
        p.end()

# ─── 1D CA Canvas ─────────────────────────────────────────────────────────────

class ElementaryCACanvas(QWidget):
    cell_clicked = Signal(int)

    def __init__(self, ca, parent=None):
        super().__init__(parent)
        self.ca = ca; self.color_mode = "teal"
        self.setMinimumSize(400, 200); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor(16, 16, 28))
        ca = self.ca; w, h = self.width(), self.height()
        cs = max(1, w / ca.width); xo = (w - ca.width * cs) / 2
        mr = int(h / cs); total = len(ca.generations); start = max(0, total - mr)

        for ri in range(start, total):
            y = (ri - start) * cs; row = ca.generations[ri]
            for ci, cell in enumerate(row):
                if not cell: continue
                x = xo + ci * cs
                if self.color_mode == "teal":
                    frac = (ri - start) / max(1, mr - 1); b = 0.35 + 0.65 * frac
                    p.fillRect(QRectF(x, y, cs, cs), QColor(0, int(212*b), int(168*b)))
                elif self.color_mode == "rainbow":
                    p.fillRect(QRectF(x, y, cs, cs), QColor.fromHsv((ri * 3) % 360, 210, 220))
                else:
                    p.fillRect(QRectF(x, y, cs, cs), QColor(220, 220, 220))

        if len(ca.generations) <= 1:
            p.setPen(QPen(QColor(0, 212, 168, 120), 1, Qt.DashLine)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(xo, 0, ca.width * cs, cs))
        p.end()

    def mousePressEvent(self, event):
        if len(self.ca.generations) <= 1:
            cs = max(1, self.width() / self.ca.width); xo = (self.width() - self.ca.width * cs) / 2
            col = int((event.position().x() - xo) / cs)
            if 0 <= col < self.ca.width: self.cell_clicked.emit(col)

# ─── 1D Elementary CA Tab ─────────────────────────────────────────────────────

RULE_DESC = {
    30: "Rule 30 — Chaotic, aperiodic. Used in Mathematica's RNG.", 90: "Rule 90 — Sierpiński triangle fractal.",
    110: "Rule 110 — Turing-complete! Complex localized structures.", 184: "Rule 184 — Traffic flow model.",
    150: "Rule 150 — Triple XOR. Nested self-similar patterns.", 60: "Rule 60 — Left⊕center XOR.",
    22: "Rule 22 — Interleaved triangles (Class III).", 45: "Rule 45 — Left-running chaos (Class III).",
    73: "Rule 73 — Localized persistent structures (Class IV).", 182: "Rule 182 — Mix of periodic and chaotic domains.",
}

class ElementaryCATab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ca = ElementaryCA(301, 30); self.running = False; self.speed = 80
        self._build(); self.timer = QTimer(self); self.timer.timeout.connect(self._step)

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(6)
        t = QLabel("Elementary Cellular Automata (1D)")
        t.setFont(QFont("Segoe UI", 15, QFont.Bold)); t.setStyleSheet("color:#00d4a8;"); lay.addWidget(t)
        desc = QLabel("Each cell's next state depends on its 3-cell neighborhood. Click output cells in the rule table to toggle bits!")
        desc.setWordWrap(True); desc.setStyleSheet("color:#a0a0c0;"); lay.addWidget(desc)

        rg = QGroupBox("Rule Table — click output cells to build custom rules")
        rl = QVBoxLayout(rg); self.rule_viz = RuleVisualizerWidget(); self.rule_viz.set_rule(30)
        self.rule_viz.rule_changed.connect(self._on_rule_built); rl.addWidget(self.rule_viz, alignment=Qt.AlignCenter)
        lay.addWidget(rg)

        c1 = QHBoxLayout(); c1.setSpacing(6)
        c1.addWidget(QLabel("Rule:")); self.rule_spin = QSpinBox(); self.rule_spin.setRange(0, 255); self.rule_spin.setValue(30)
        self.rule_spin.valueChanged.connect(self._on_rule); c1.addWidget(self.rule_spin)
        self.famous = QComboBox(); self.famous.addItem("— Famous Rules —")
        for r in (30, 90, 110, 184, 150, 60, 22, 45, 73, 182): self.famous.addItem(f"Rule {r}", r)
        self.famous.currentIndexChanged.connect(self._on_famous); c1.addWidget(self.famous)
        c1.addSpacing(8); c1.addWidget(QLabel("Init:"))
        self.init_combo = QComboBox(); self.init_combo.addItems(["Single Cell", "Random"]); c1.addWidget(self.init_combo)
        c1.addSpacing(8); c1.addWidget(QLabel("Color:"))
        self.color_combo = QComboBox(); self.color_combo.addItems(["Teal Gradient", "Rainbow", "Classic B&W"])
        self.color_combo.currentIndexChanged.connect(lambda i: (setattr(self.canvas, 'color_mode', ("teal", "rainbow", "bw")[i]), self.canvas.update()))
        c1.addWidget(self.color_combo); c1.addSpacing(8); c1.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox(); self.width_spin.setRange(51, 801); self.width_spin.setSingleStep(50); self.width_spin.setValue(301)
        self.width_spin.valueChanged.connect(self._on_width); c1.addWidget(self.width_spin); c1.addStretch(); lay.addLayout(c1)

        c2 = QHBoxLayout(); c2.setSpacing(6)
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setFixedWidth(90); self.play_btn.clicked.connect(self._toggle); c2.addWidget(self.play_btn)
        sb = QPushButton("⏭ Step"); sb.setFixedWidth(80); sb.clicked.connect(self._step); c2.addWidget(sb)
        rb = QPushButton("↺ Reset"); rb.setFixedWidth(80); rb.clicked.connect(self._reset); c2.addWidget(rb)
        c2.addSpacing(4); c2.addWidget(QLabel("Skip ×")); self.stepn_spin = QSpinBox()
        self.stepn_spin.setRange(1, 5000); self.stepn_spin.setValue(100); self.stepn_spin.setFixedWidth(70); c2.addWidget(self.stepn_spin)
        snb = QPushButton("⏩"); snb.setFixedWidth(50); snb.clicked.connect(self._skip); c2.addWidget(snb)
        c2.addStretch()
        c2.addWidget(QLabel("Speed:")); self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 300); self.speed_slider.setValue(220); self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(self._on_speed); c2.addWidget(self.speed_slider); lay.addLayout(c2)

        self.gen_lbl = QLabel("Generation: 0 — Click on the first row to toggle cells")
        self.gen_lbl.setStyleSheet("color:#00d4a8; font-weight:bold;"); lay.addWidget(self.gen_lbl)
        self.canvas = ElementaryCACanvas(self.ca); self.canvas.cell_clicked.connect(self._on_cell_click); lay.addWidget(self.canvas, 1)
        self.rule_desc = QLabel(); self.rule_desc.setWordWrap(True)
        self.rule_desc.setStyleSheet("color:#8888a8; font-size:11px;"); self._update_desc(30); lay.addWidget(self.rule_desc)

    def _on_rule(self, v): self.ca.rule = v; self.rule_viz.set_rule(v); self._update_desc(v); self._reset()
    def _on_rule_built(self, v): self.rule_spin.blockSignals(True); self.rule_spin.setValue(v); self.rule_spin.blockSignals(False); self.ca.rule = v; self._update_desc(v); self._reset()
    def _on_famous(self, i):
        r = self.famous.itemData(i)
        if r is not None: self.rule_spin.setValue(r)
    def _on_width(self, v):
        was = self.running
        if was: self._toggle()
        self.ca = ElementaryCA(v, self.ca.rule); self.canvas.ca = self.ca
        self.ca.reset("single" if self.init_combo.currentIndex() == 0 else "random")
        self.gen_lbl.setText("Generation: 0"); self.canvas.update()
    def _on_speed(self, v): self.speed = 305 - v
    def _on_cell_click(self, col):
        if len(self.ca.generations) <= 1: self.ca.toggle_cell(col); self.canvas.update()
    def _toggle(self):
        self.running = not self.running
        if self.running: self.play_btn.setText("⏸ Pause"); self.timer.start(self.speed)
        else: self.play_btn.setText("▶  Play"); self.timer.stop()
    def _step(self):
        self.ca.step(); self.gen_lbl.setText(f"Generation: {len(self.ca.generations)-1}"); self.canvas.update()
    def _skip(self):
        for _ in range(self.stepn_spin.value()): self.ca.step()
        self.gen_lbl.setText(f"Generation: {len(self.ca.generations)-1}"); self.canvas.update()
    def _reset(self):
        if self.running: self._toggle()
        self.ca.reset("single" if self.init_combo.currentIndex() == 0 else "random")
        self.gen_lbl.setText("Generation: 0"); self.canvas.update()
    def _update_desc(self, r): self.rule_desc.setText(RULE_DESC.get(r, f"Rule {r} — one of 256 elementary CA. Explore!"))
    def toggle_play(self): self._toggle()
    def single_step(self): self._step()
    def do_reset(self): self._reset()

# ─── 1D Comparison Engine & Canvas ────────────────────────────────────────────

class ComparisonCA:
    def __init__(self, width=151):
        self.width = width; self.rules = [0, 4, 30, 110]
        self.class_names = ["Class I (Stable)", "Class II (Periodic)", "Class III (Chaotic)", "Class IV (Complex)"]
        self.generations = [[], [], [], []]; self.reset()

    def reset(self):
        mid = self.width // 2; init_row = [0] * self.width; init_row[mid] = 1
        self.generations = [[init_row[:]] for _ in range(4)]

    def step(self):
        for idx, rule in enumerate(self.rules):
            cur = self.generations[idx][-1]; bits = [(rule >> i) & 1 for i in range(8)]
            nxt = [0] * self.width
            for i in range(self.width):
                j = (cur[(i-1)%self.width]<<2)|(cur[i]<<1)|cur[(i+1)%self.width]; nxt[i] = bits[j]
            self.generations[idx].append(nxt)

class ComparisonCanvas(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent); self.engine = engine
        self.setMinimumSize(500, 300); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor(16, 16, 28))
        eng = self.engine; w = self.width(); h = self.height()
        hw = w // 2; hh = h // 2; pad = 4
        colors = [QColor(100,180,255), QColor(180,100,255), QColor(255,100,100), QColor(0,212,168)]

        for idx in range(4):
            col = idx % 2; row = idx // 2; x0 = col*hw+pad; y0 = row*hh+pad; qw = hw-2*pad; qh = hh-2*pad
            p.fillRect(QRectF(x0, y0, qw, qh), QColor(22, 22, 38))
            p.setPen(QPen(colors[idx], 2)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(x0, y0, qw, qh), 4, 4)
            p.setPen(colors[idx]); p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(int(x0+6), int(y0+14), f"Rule {eng.rules[idx]} — {eng.class_names[idx]}")
            gens = eng.generations[idx]; cs = max(1, (qw-10)/eng.width)
            mr = int((qh-22)/cs); start = max(0, len(gens)-mr)
            for ri in range(start, len(gens)):
                y = y0+20+(ri-start)*cs
                if y+cs > y0+qh: break
                for ci, cell in enumerate(gens[ri]):
                    if cell: p.fillRect(QRectF(x0+5+ci*cs, y, cs, cs), colors[idx])
        p.end()

class ComparisonTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = ComparisonCA(151); self.running = False; self.speed = 100
        self._build(); self.timer = QTimer(self); self.timer.timeout.connect(self._step)

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(6)
        t = QLabel("Wolfram's 4 Classes — Side-by-Side")
        t.setFont(QFont("Segoe UI", 15, QFont.Bold)); t.setStyleSheet("color:#00d4a8;"); lay.addWidget(t)
        desc = QLabel("Observe the four behavioral classes of CA simultaneously. Class I stabilizes, II oscillates, III is chaotic, IV is complex.")
        desc.setWordWrap(True); desc.setStyleSheet("color:#a0a0c0;"); lay.addWidget(desc)

        r1 = QHBoxLayout(); r1.setSpacing(6)
        labels = ["Class I:", "Class II:", "Class III:", "Class IV:"]; defaults = [0, 4, 30, 110]; self.rule_spins = []
        for i in range(4):
            r1.addWidget(QLabel(labels[i])); spin = QSpinBox(); spin.setRange(0, 255); spin.setValue(defaults[i])
            spin.setFixedWidth(60); spin.valueChanged.connect(lambda v, idx=i: self._on_rule(idx, v))
            self.rule_spins.append(spin); r1.addWidget(spin); r1.addSpacing(8)
        r1.addStretch(); lay.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(6)
        for pname, prules in {"Wolfram's Original": [0,4,30,110], "Different Angles": [32,108,45,54], "All Chaotic": [90,105,150,182]}.items():
            btn = QPushButton(pname); btn.clicked.connect(lambda checked, r=prules: self._set_preset(r)); r2.addWidget(btn)
        r2.addStretch(); lay.addLayout(r2)

        r3 = QHBoxLayout(); r3.setSpacing(6)
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setFixedWidth(90); self.play_btn.clicked.connect(self._toggle); r3.addWidget(self.play_btn)
        sb = QPushButton("⏭ Step"); sb.setFixedWidth(80); sb.clicked.connect(self._step); r3.addWidget(sb)
        rb = QPushButton("↺ Reset"); rb.setFixedWidth(80); rb.clicked.connect(self._reset); r3.addWidget(rb)
        r3.addStretch()
        r3.addWidget(QLabel("Speed:")); self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 300); self.speed_slider.setValue(205); self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'speed', 305-v) or (self.timer.setInterval(self.speed) if self.running else None))
        r3.addWidget(self.speed_slider); lay.addLayout(r3)

        self.gen_lbl = QLabel("Generation: 0"); self.gen_lbl.setStyleSheet("color:#00d4a8; font-weight:bold;"); lay.addWidget(self.gen_lbl)
        self.canvas = ComparisonCanvas(self.engine); lay.addWidget(self.canvas, 1)

    def _on_rule(self, idx, val): self.engine.rules[idx] = val; self._reset()
    def _set_preset(self, rules):
        for i, r in enumerate(rules): self.rule_spins[i].setValue(r)
    def _toggle(self):
        self.running = not self.running
        if self.running: self.play_btn.setText("⏸ Pause"); self.timer.start(self.speed)
        else: self.play_btn.setText("▶  Play"); self.timer.stop()
    def _step(self):
        self.engine.step(); self.gen_lbl.setText(f"Generation: {len(self.engine.generations[0])-1}"); self.canvas.update()
    def _reset(self):
        if self.running: self._toggle()
        self.engine.reset(); self.gen_lbl.setText("Generation: 0"); self.canvas.update()
    def toggle_play(self): self._toggle()
    def single_step(self): self._step()
    def do_reset(self): self._reset()

# ─── Population Graph ──────────────────────────────────────────────────────────

class PopulationGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.data = []; self.max_pts = 500; self.setFixedHeight(80)

    def add(self, pop):
        self.data.append(pop)
        if len(self.data) > self.max_pts: self.data.pop(0)
        self.update()

    def clear_data(self): self.data.clear(); self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(20, 20, 36))
        p.setPen(QPen(QColor(40, 40, 60), 1)); p.drawRect(0, 0, self.width()-1, self.height()-1)
        if len(self.data) < 2:
            p.setPen(QColor(80, 80, 110)); p.setFont(QFont("Segoe UI", 9))
            p.drawText(self.rect(), Qt.AlignCenter, "Population over time"); p.end(); return
        mx = max(self.data) or 1; n = len(self.data); w = self.width()-4; h = self.height()-4
        pts = [QRectF(2 + i*w/(n-1), 2+h-(v/mx)*h, 0, 0) for i, v in enumerate(self.data)]
        poly = QPolygonF([QRectF(pts[0].x(), 2+h, 0, 0)] + pts + [QRectF(pts[-1].x(), 2+h, 0, 0)])
        p.setBrush(QColor(0, 212, 168, 25)); p.setPen(Qt.NoPen); p.drawPolygon(poly)
        p.setPen(QPen(QColor(0, 212, 168), 1.5))
        for i in range(len(pts)-1): p.drawLine(pts[i].topLeft(), pts[i+1].topLeft())
        p.setPen(QColor(120, 120, 150)); p.setFont(QFont("Segoe UI", 7))
        p.drawText(4, 10, f"Max: {mx}"); p.end()

# ─── 2D Game of Life Canvas ───────────────────────────────────────────────────

class GameOfLifeCanvas(BaseGridCanvas):
    cell_hovered = Signal(int, int, int, int) # r, c, state, neighbors

    def __init__(self, rows=80, cols=120, parent=None):
        super().__init__(rows, cols, parent)
        self.grid = [[0]*cols for _ in range(rows)]
        self.age_grid = [[0]*cols for _ in range(rows)]
        self.heat_grid = [[0.0]*cols for _ in range(rows)]
        self.display_mode = "standard"; self.rule_type = "bs"
        self._drawing = False; self._draw_val = 1

    def reset(self):
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.age_grid = [[0]*self.cols for _ in range(self.rows)]
        self.heat_grid = [[0.0]*self.cols for _ in range(self.rows)]
        super().reset_view()

    def randomize(self, d=0.22):
        self.grid = [[1 if random.random()<d else 0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.age_grid = [[0]*self.cols for _ in range(self.rows)]
        self.heat_grid = [[0.0]*self.cols for _ in range(self.rows)]; self.update()

    def place_pattern(self, pat, r0=None, c0=None):
        if r0 is None: r0 = self.rows//2 - len(pat)//2
        if c0 is None: c0 = self.cols//2 - (len(pat[0]) if pat else 0)//2
        for r, row in enumerate(pat):
            for c, v in enumerate(row):
                nr, nc = r0+r, c0+c
                if 0<=nr<self.rows and 0<=nc<self.cols: self.grid[nr][nc] = v
        self.update()

    def invert(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.rule_type == "brians_brain": self.grid[r][c] = 0 if self.grid[r][c]==1 else 1
                else: self.grid[r][c] ^= 1
        self.update()

    def step(self, birth=None, survive=None, rule_type="bs"):
        self.rule_type = rule_type; new = [[0]*self.cols for _ in range(self.rows)]
        new_age = [[0]*self.cols for _ in range(self.rows)]; new_heat = [[0.0]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                new_heat[r][c] = self.heat_grid[r][c]*0.90; n = self._nbrs(r, c)
                if rule_type == "brians_brain":
                    s = self.grid[r][c]
                    if s==1: new[r][c]=2
                    elif s==2: new[r][c]=0
                    else: new[r][c]=1 if n==2 else 0
                else:
                    b = birth or {3}; s = survive or {2,3}
                    if self.grid[r][c]: new[r][c] = 1 if n in s else 0
                    else: new[r][c] = 1 if n in b else 0
                if new[r][c]==1:
                    new_age[r][c] = (self.age_grid[r][c]+1) if self.grid[r][c]==1 else 1; new_heat[r][c]=1.0
        self.grid = new; self.age_grid = new_age; self.heat_grid = new_heat; self.update()

    def _nbrs(self, r, c):
        s = 0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                v = self.grid[(r+dr)%self.rows][(c+dc)%self.cols]
                s += 1 if self.rule_type=="brians_brain" and v==1 else v
        return s

    def mousePressEvent(self, event):
        super().mousePressEvent(event) # Handle middle click pan
        if event.button() == Qt.LeftButton:
            cell = self._cell_at(event.position())
            if cell:
                r, c = cell; self._drawing = True
                self._draw_val = 0 if self.grid[r][c] else 1
                self.grid[r][c] = self._draw_val; self.update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event) # Handle pan move
        pos = event.position()
        if self._dragging_pan: return
        if self._drawing:
            cell = self._cell_at(pos)
            if cell: self.grid[cell[0]][cell[1]] = self._draw_val; self.update()
        else:
            cell = self._cell_at(pos)
            if cell: self.cell_hovered.emit(cell[0], cell[1], self.grid[cell[0]][cell[1]], self._nbrs(cell[0], cell[1]))

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton: self._drawing = False

    def paintEvent(self, event):
        self._recalc(); p = QPainter(self); p.fillRect(self.rect(), QColor(16, 16, 28))
        cs, xo, yo, dm = self._cs, self._xo, self._yo, self.display_mode
        grid_pen = QPen(QColor(35, 35, 55), 1) if cs >= 8 else Qt.NoPen

        for r in range(self.rows):
            y = yo + r*cs
            for c in range(self.cols):
                x = xo + c*cs; v = self.grid[r][c]
                if dm == "standard":
                    if v == 1: p.fillRect(QRectF(x,y,cs,cs), QColor(0,212,168))
                    elif v == 2: p.fillRect(QRectF(x,y,cs,cs), QColor(255,140,40))
                    elif cs >= 8: p.setPen(grid_pen); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(x,y,cs,cs))
                elif dm == "age":
                    if v == 1:
                        age = min(self.age_grid[r][c], 60); hue = max(0, 160-age*3)
                        p.fillRect(QRectF(x,y,cs,cs), QColor.fromHsv(hue, 220, 220))
                    elif v == 2: p.fillRect(QRectF(x,y,cs,cs), QColor(255,140,40))
                    elif cs >= 8: p.setPen(grid_pen); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(x,y,cs,cs))
                elif dm == "heat":
                    h_val = self.heat_grid[r][c]
                    if v == 1: p.fillRect(QRectF(x,y,cs,cs), QColor(0,212,168))
                    elif h_val > 0.02:
                        a = int(min(h_val*180, 180)); p.fillRect(QRectF(x,y,cs,cs), QColor(255, 80, 30, a))
                    elif cs >= 8: p.setPen(grid_pen); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(x,y,cs,cs))

        p.setPen(QPen(QColor(50,50,80),1)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(xo, yo, self.cols*cs, self.rows*cs))
        p.end()

# ─── 2D Game of Life Tab ──────────────────────────────────────────────────────

PATTERNS = {
    "Glider": [[0,1,0],[0,0,1],[1,1,1]], "Blinker": [[1,1,1]], "Toad": [[0,1,1,1],[1,1,1,0]],
    "Beacon": [[1,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,1]], "LWSS": [[0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]],
    "R-pentomino": [[0,1,1],[1,1,0],[0,1,0]], "Acorn": [[0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[1,1,0,0,1,1,1]],
    "Pulsar": [
        [0,0,1,1,1,0,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,1,0,0,0,0,1],[0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,1,1,1,0,0],[1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],[1,0,0,0,0,1,0,1,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
    ],
    "Gosper Glider Gun": [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ],
}

RULE_PRESETS = {
    "Conway's Life (B3/S23)": ({3}, {2, 3}), "HighLife (B36/S23)": ({3, 6}, {2, 3}),
    "Seeds (B2/S)": ({2}, set()), "Day & Night (B3678/S34678)": ({3,6,7,8}, {3,4,6,7,8}),
    "Diamoeba (B35678/S5678)": ({3,5,6,7,8}, {5,6,7,8}), "Replicator (B1357/S1357)": ({1,3,5,7}, {1,3,5,7}),
    "Maze (B3/S12345)": ({3}, {1,2,3,4,5}), "Anneal (B4678/S35678)": ({4,6,7,8}, {3,5,6,7,8}),
    "Brian's Brain (3-state)": (None, None),
}

class GameOfLifeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False; self.speed = 80; self.generation = 0
        self.birth = {3}; self.survive = {2,3}; self.rule_type = "bs"
        self._build(); self.timer = QTimer(self); self.timer.timeout.connect(self._step)

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(6)
        t = QLabel("2D Cellular Automata — Game of Life & Beyond")
        t.setFont(QFont("Segoe UI", 15, QFont.Bold)); t.setStyleSheet("color:#00d4a8;"); lay.addWidget(t)
        desc = QLabel("Click/drag to draw. Try different rule presets. Use Custom Rule Editor to design B/S rules!")
        desc.setWordWrap(True); desc.setStyleSheet("color:#a0a0c0;"); lay.addWidget(desc)

        r1 = QHBoxLayout(); r1.setSpacing(6)
        r1.addWidget(QLabel("Rules:")); self.rule_combo = QComboBox()
        for n in RULE_PRESETS: self.rule_combo.addItem(n)
        self.rule_combo.currentTextChanged.connect(self._on_rule); r1.addWidget(self.rule_combo)
        r1.addSpacing(6); r1.addWidget(QLabel("Pattern:"))
        self.pat_combo = QComboBox(); self.pat_combo.addItem("— Place Pattern —")
        for n in PATTERNS: self.pat_combo.addItem(n)
        self.pat_combo.currentTextChanged.connect(self._on_pattern); r1.addWidget(self.pat_combo)
        r1.addSpacing(6); r1.addWidget(QLabel("Display:"))
        self.disp_combo = QComboBox(); self.disp_combo.addItems(["Standard", "Age Gradient", "Heat Trail"])
        self.disp_combo.currentIndexChanged.connect(lambda i: (setattr(self.canvas, 'display_mode', ("standard","age","heat")[i]), self.canvas.update()))
        r1.addWidget(self.disp_combo); r1.addStretch(); lay.addLayout(r1)

        bs_group = QGroupBox("Custom Rule Editor (Birth / Survive)"); bs_lay = QHBoxLayout(bs_group); bs_lay.setSpacing(4)
        bs_lay.addWidget(QLabel("B:")); self.b_cb = []
        for i in range(9):
            cb = QCheckBox(str(i)); self.b_cb.append(cb); cb.toggled.connect(self._on_custom_rule); bs_lay.addWidget(cb)
        bs_lay.addSpacing(12); bs_lay.addWidget(QLabel("S:")); self.s_cb = []
        for i in range(9):
            cb = QCheckBox(str(i)); self.s_cb.append(cb); cb.toggled.connect(self._on_custom_rule); bs_lay.addWidget(cb)
        bs_lay.addStretch(); self.custom_lbl = QLabel("B3/S23")
        self.custom_lbl.setStyleSheet("color:#00d4a8; font-weight:bold; font-size:13px;"); bs_lay.addWidget(self.custom_lbl)
        lay.addWidget(bs_group)

        r2 = QHBoxLayout(); r2.setSpacing(6)
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setFixedWidth(90); self.play_btn.clicked.connect(self._toggle); r2.addWidget(self.play_btn)
        sb = QPushButton("⏭ Step"); sb.setFixedWidth(80); sb.clicked.connect(self._step); r2.addWidget(sb)
        cb = QPushButton("↺ Clear"); cb.setFixedWidth(80); cb.clicked.connect(self._clear); r2.addWidget(cb)
        rb = QPushButton("🎲 Random"); rb.setFixedWidth(90); rb.clicked.connect(self._randomize); r2.addWidget(rb)
        ib = QPushButton("🔄 Invert"); ib.setFixedWidth(80); ib.clicked.connect(self._invert); r2.addWidget(ib)
        r2.addSpacing(4); r2.addWidget(QLabel("Skip ×")); self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 10000); self.skip_spin.setValue(100); self.skip_spin.setFixedWidth(70); r2.addWidget(self.skip_spin)
        skb = QPushButton("⏩"); skb.setFixedWidth(50); skb.clicked.connect(self._skip); r2.addWidget(skb)
        r2.addStretch()
        sav_btn = QPushButton("💾 Save"); sav_btn.setFixedWidth(70); sav_btn.clicked.connect(self._save_grid); r2.addWidget(sav_btn)
        lod_btn = QPushButton("📂 Load"); lod_btn.setFixedWidth(70); lod_btn.clicked.connect(self._load_grid); r2.addWidget(lod_btn)
        r2.addSpacing(6); r2.addWidget(QLabel("Speed:")); self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 300); self.speed_slider.setValue(220); self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'speed', 305-v) or (self.timer.setInterval(self.speed) if self.running else None))
        r2.addWidget(self.speed_slider); lay.addLayout(r2)

        self.hover_lbl = QLabel("Hover over grid for cell info"); self.hover_lbl.setStyleSheet("color:#7878a0; font-size:11px;"); lay.addWidget(self.hover_lbl)
        self.stats = QLabel("Gen: 0 | Pop: 0"); self.stats.setStyleSheet("color:#00d4a8; font-weight:bold;"); lay.addWidget(self.stats)
        self.canvas = GameOfLifeCanvas(80, 120); self.canvas.cell_hovered.connect(self._on_hover); lay.addWidget(self.canvas, 1)
        self.pop_graph = PopulationGraph(); lay.addWidget(self.pop_graph)

    def _on_rule(self, name):
        if name not in RULE_PRESETS: return
        b, s = RULE_PRESETS[name]
        if b is None: self.rule_type = "brians_brain"; self.birth={3}; self.survive={2,3}
        else: self.rule_type = "bs"; self.birth=b; self.survive=s
        for cb in self.b_cb + self.s_cb: cb.setEnabled(self.rule_type=="bs")
        if self.rule_type=="bs": self._sync_checkboxes()
        self._update_custom_lbl()

    def _sync_checkboxes(self):
        for i, cb in enumerate(self.b_cb): cb.blockSignals(True); cb.setChecked(i in self.birth); cb.blockSignals(False)
        for i, cb in enumerate(self.s_cb): cb.blockSignals(True); cb.setChecked(i in self.survive); cb.blockSignals(False)

    def _on_custom_rule(self):
        if self.rule_type != "bs": return
        self.birth = {i for i, cb in enumerate(self.b_cb) if cb.isChecked()}
        self.survive = {i for i, cb in enumerate(self.s_cb) if cb.isChecked()}
        self.rule_combo.blockSignals(True); self.rule_combo.setCurrentIndex(-1); self.rule_combo.blockSignals(False)
        self._update_custom_lbl()

    def _update_custom_lbl(self):
        if self.rule_type == "brians_brain": self.custom_lbl.setText("Brian's Brain (3-state)")
        else: self.custom_lbl.setText(f"B{''.join(map(str,sorted(self.birth)))}/S{''.join(map(str,sorted(self.survive)))}")

    def _on_pattern(self, name):
        if name in PATTERNS:
            self.canvas.reset(); self.generation = 0; self.canvas.place_pattern(PATTERNS[name])
            self.pop_graph.clear_data(); self._update_stats()
            self.pat_combo.blockSignals(True); self.pat_combo.setCurrentIndex(0); self.pat_combo.blockSignals(False)

    def _on_hover(self, r, c, state, neighbors):
        s_name = {0:"Dead", 1:"Alive", 2:"Dying"}.get(state, str(state))
        age = self.canvas.age_grid[r][c] if state == 1 else 0
        txt = f"Cell ({r},{c}) | State: {s_name} | Neighbors: {neighbors}"
        if state == 1: txt += f" | Age: {age}"
        self.hover_lbl.setText(txt)

    def _toggle(self):
        self.running = not self.running
        if self.running: self.play_btn.setText("⏸ Pause"); self.timer.start(self.speed)
        else: self.play_btn.setText("▶  Play"); self.timer.stop()
    def _step(self):
        self.canvas.step(self.birth, self.survive, self.rule_type); self.generation += 1; self._update_stats()
        self.pop_graph.add(sum(1 for r in self.canvas.grid for c in r if c==1))
    def _skip(self):
        for _ in range(self.skip_spin.value()): self.canvas.step(self.birth, self.survive, self.rule_type); self.generation += 1
        self._update_stats(); self.pop_graph.add(sum(1 for r in self.canvas.grid for c in r if c==1))
    def _clear(self):
        if self.running: self._toggle()
        self.canvas.reset(); self.generation = 0; self.pop_graph.clear_data(); self._update_stats()
    def _randomize(self): self.canvas.randomize(0.22); self.generation = 0; self.pop_graph.clear_data(); self._update_stats()
    def _invert(self): self.canvas.invert(); self._update_stats()
    def _update_stats(self):
        pop = sum(1 for r in self.canvas.grid for c in r if c==1)
        dying = sum(1 for r in self.canvas.grid for c in r if c==2) if self.rule_type=="brians_brain" else 0
        txt = f"Gen: {self.generation} | Pop: {pop}"
        if dying: txt += f" | Dying: {dying}"
        self.stats.setText(txt)
    def _save_grid(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Grid", "grid.csv", "CSV (*.csv)")
        if not path: return
        with open(path, 'w') as f:
            f.write(f"{self.canvas.rows},{self.canvas.cols},{self.rule_type}\n")
            for row in self.canvas.grid: f.write(",".join(map(str, row)) + "\n")
    def _load_grid(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Grid", "", "CSV (*.csv)")
        if not path: return
        with open(path, 'r') as f:
            hdr = f.readline().strip().split(','); rows, cols = int(hdr[0]), int(hdr[1])
            if len(hdr)>2: self.rule_type = hdr[2]
            self.canvas.rows = rows; self.canvas.cols = cols; self.canvas.grid = []
            for _ in range(rows): self.canvas.grid.append(list(map(int, f.readline().strip().split(','))))
            self.canvas.age_grid = [[0]*cols for _ in range(rows)]; self.canvas.heat_grid = [[0.0]*cols for _ in range(rows)]
            self.canvas.update(); self.generation = 0; self._update_stats()
    def toggle_play(self): self._toggle()
    def single_step(self): self._step()
    def do_reset(self): self._clear()

# ─── Wireworld Engine & Canvas ─────────────────────────────────────────────────

WIREWORLD_PATTERNS = {
    "Clock": [[3,3,3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3,3,3],[3,3,1,2,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3,3,3]],
    "Diode": [[0,0,0,3,3,3,0,0,0],[0,0,3,3,3,3,3,0,0],[3,3,3,2,1,3,3,3,3],[0,0,3,3,3,3,3,0,0],[0,0,0,3,3,3,0,0,0]],
}

class WireworldCanvas(BaseGridCanvas):
    COLORS = {0: QColor(16,16,28), 1: QColor(0,150,255), 2: QColor(220,50,50), 3: QColor(220,200,50)}
    def __init__(self, rows=60, cols=80, parent=None):
        super().__init__(rows, cols, parent)
        self.grid = [[0]*cols for _ in range(rows)]; self._drawing = False; self._draw_val = 3

    def reset(self): self.grid = [[0]*self.cols for _ in range(self.rows)]; super().reset_view()
    def place_pattern(self, pat, r0=None, c0=None):
        if r0 is None: r0 = self.rows//2 - len(pat)//2
        if c0 is None: c0 = self.cols//2 - (len(pat[0]) if pat else 0)//2
        for r, row in enumerate(pat):
            for c, v in enumerate(row):
                nr, nc = r0+r, c0+c
                if 0<=nr<self.rows and 0<=nc<self.cols: self.grid[nr][nc] = v
        self.update()

    def step(self):
        new = [[0]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                s = self.grid[r][c]
                if s==1: new[r][c]=2
                elif s==2: new[r][c]=3
                elif s==3:
                    heads = sum(1 for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and self.grid[(r+dr)%self.rows][(c+dc)%self.cols]==1)
                    new[r][c] = 1 if heads in (1,2) else 3
        self.grid = new; self.update()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            cell = self._cell_at(event.position())
            if cell:
                r, c = cell; self._drawing = True
                self._draw_val = (self.grid[r][c] + 1) % 4
                if self._draw_val == 0: self._draw_val = 3
                self.grid[r][c] = self._draw_val; self.update()
        elif event.button() == Qt.RightButton:
            cell = self._cell_at(event.position())
            if cell: self.grid[cell[0]][cell[1]] = 0; self.update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._dragging_pan: return
        if self._drawing:
            cell = self._cell_at(event.position())
            if cell: self.grid[cell[0]][cell[1]] = self._draw_val; self.update()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton: self._drawing = False

    def paintEvent(self, event):
        self._recalc(); p = QPainter(self); p.fillRect(self.rect(), QColor(16,16,28))
        cs, xo, yo = self._cs, self._xo, self._yo
        for r in range(self.rows):
            y = yo + r*cs
            for c in range(self.cols):
                v = self.grid[r][c]
                if v == 0 and cs < 8: continue
                x = xo + c*cs; p.fillRect(QRectF(x, y, cs, cs), self.COLORS[v])
                if v == 0 and cs >= 8: p.setPen(QPen(QColor(35,35,55),1)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(x,y,cs,cs))
        p.setPen(QPen(QColor(50,50,80),1)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(xo, yo, self.cols*cs, self.rows*cs)); p.end()

class WireworldTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = WireworldCanvas(40, 60); self.running = False; self.speed = 120; self.generation = 0
        self._build(); self.timer = QTimer(self); self.timer.timeout.connect(self._step)

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(6)
        t = QLabel("Wireworld (Logic Gates)")
        t.setFont(QFont("Segoe UI", 15, QFont.Bold)); t.setStyleSheet("color:#00d4a8;"); lay.addWidget(t)
        desc = QLabel("A 4-state CA simulating electronic logic gates. Copper (yellow), Electron Heads (blue) and Tails (red) flow along them. Left-click draws/cycles, Right-click erases.")
        desc.setWordWrap(True); desc.setStyleSheet("color:#a0a0c0;"); lay.addWidget(desc)

        r1 = QHBoxLayout(); r1.setSpacing(6); r1.addWidget(QLabel("Pattern:"))
        self.pat_combo = QComboBox(); self.pat_combo.addItem("— Place Pattern —")
        for n in WIREWORLD_PATTERNS: self.pat_combo.addItem(n)
        self.pat_combo.currentTextChanged.connect(self._on_pattern); r1.addWidget(self.pat_combo); r1.addStretch(); lay.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(6
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setFixedWidth(90); self.play_btn.clicked.connect(self._toggle); r2.addWidget(self.play_btn)
        sb = QPushButton("⏭ Step"); sb.setFixedWidth(80); sb.clicked.connect(self._step); r2.addWidget(sb)
        rb = QPushButton("↺ Reset"); rb.setFixedWidth(80); rb.clicked.connect(self._reset); r2.addWidget(rb)
        r2.addStretch()
        r2.addWidget(QLabel("Speed:")); self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 300); self.speed_slider.setValue(180); self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'speed', 305-v) or (self.timer.setInterval(self.speed) if self.running else None))
        r2.addWidget(self.speed_slider); lay.addLayout(r2)

        self.stats = QLabel("Gen: 0 | LClick: Draw | RClick: Erase | MidDrag: Pan | Scroll: Zoom")
        self.stats.setStyleSheet("color:#00d4a8; font-weight:bold; font-size:11px;"); lay.addWidget(self.stats)
        lay.addWidget(self.canvas, 1)

    def _on_pattern(self, name):
        if name in WIREWORLD_PATTERNS:
            self.canvas.reset(); self.canvas.place_pattern(WIREWORLD_PATTERNS[name])
            self.generation = 0; self._update_stats()
            self.pat_combo.blockSignals(True); self.pat_combo.setCurrentIndex(0); self.pat_combo.blockSignals(False)
    def _toggle(self):
        self.running = not self.running
        if self.running: self.play_btn.setText("⏸ Pause"); self.timer.start(self.speed)
        else: self.play_btn.setText("▶  Play"); self.timer.stop()
    def _step(self): self.canvas.step(); self.generation += 1; self._update_stats()
    def _reset(self): 
        if self.running: self._toggle()
        self.canvas.reset(); self.generation = 0; self._update_stats()
    def _update_stats(self):
        pop = {1:0, 2:0, 3:0}
        for r in self.canvas.grid:
            for c in r:
                if c in pop: pop[c]+=1
        self.stats.setText(f"Gen: {self.generation} | Heads: {pop[1]} | Tails: {pop[2]} | Copper: {pop[3]}")
    def toggle_play(self): self._toggle()
    def single_step(self): self._step()
    def do_reset(self): self._reset()

# ─── Langton's Ant ─────────────────────────────────────────────────────────────

ANT_PRESETS = {"Classic (RL)":"RL", "Tri-color (RLR)":"RLR", "Square (LLRR)":"LLRR", "Triangle (LRRRRRLLR)":"LRRRRRLLR", "Chaotic (RRLLLRLLLRRR)":"RRLLLRLLLRRR", "Spiral (RLLR)":"RLLR", "Highway (LLR)":"LLR"}

class LangtonsAnt:
    def __init__(self, rows=150, cols=200, rule="RL"):
        self.rows = rows; self.cols = cols; self.rule = rule; self.num_colors = len(rule)
        self.grid = [[0]*cols for _ in range(rows)]
        self.ar = rows//2; self.ac = cols//2; self.ad = 0; self.steps = 0

    def reset(self):
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        self.ar = self.rows//2; self.ac = self.cols//2; self.ad = 0; self.steps = 0

    def step(self):
        s = self.grid[self.ar][self.ac]; turn = self.rule[s % self.num_colors]
        if turn == 'R': self.ad = (self.ad + 1) % 4
        elif turn == 'L': self.ad = (self.ad - 1) % 4
        elif turn == 'U': self.ad = (self.ad + 2) % 4
        self.grid[self.ar][self.ac] = (s + 1) % self.num_colors
        dr, dc = [-1,0,1,0][self.ad], [0,1,0,-1][self.ad]
        self.ar = (self.ar + dr) % self.rows; self.ac = (self.ac + dc) % self.cols; self.steps += 1

class LangtonsAntCanvas(BaseGridCanvas):
    def __init__(self, ant, parent=None):
        super().__init__(ant.rows, ant.cols, parent); self.ant = ant

    def paintEvent(self, event):
        self._recalc(); p = QPainter(self); p.fillRect(self.rect(), QColor(16,16,28))
        a = self.ant; cs, xo, yo = self._cs, self._xo, self._yo; nc = max(a.num_colors, 1)
        for r in range(a.rows):
            y = yo + r*cs
            for c in range(a.cols):
                s = a.grid[r][c]
                if s == 0: continue
                x = xo + c*cs; hue = ((s-1)*360//max(1,nc-1))%360 if nc>1 else 160
                p.fillRect(QRectF(x, y, cs, cs), QColor.fromHsv(hue, 200, 220))
        ax = xo + a.ac*cs + cs/2; ay = yo + a.ar*cs + cs/2; sz = max(cs, 5)
        p.setBrush(QColor(255,255,255)); p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(ax-sz/2, ay-sz/2, sz, sz))
        dx, dy = [0,1,0,-1][a.ad], [-1,0,1,0][a.ad]
        p.setPen(QPen(QColor(255,50,50), 2)); p.drawLine(int(ax), int(ay), int(ax+dx*sz), int(ay+dy*sz))
        p.setPen(QPen(QColor(50,50,80),1)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(xo, yo, a.cols*cs, a.rows*cs)); p.end()

class LangtonsAntTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ant = LangtonsAnt(150, 200, "RL"); self.running = False; self.speed = 30; self.spf = 10
        self._build(); self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(6)
        t = QLabel("Langton's Ant"); t.setFont(QFont("Segoe UI", 15, QFont.Bold)); t.setStyleSheet("color:#00d4a8;"); lay.addWidget(t)
        desc = QLabel("An ant turns based on cell color, advances the color, and moves forward. Classic RL builds a 'highway' after ~10,000 steps!")
        desc.setWordWrap(True); desc.setStyleSheet("color:#a0a0c0;"); lay.addWidget(desc)

        r1 = QHBoxLayout(); r1.setSpacing(6); r1.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for n in ANT_PRESETS: self.preset_combo.addItem(n)
        self.preset_combo.currentTextChanged.connect(self._on_preset); r1.addWidget(self.preset_combo)
        r1.addSpacing(8); r1.addWidget(QLabel("Rule:"))
        self.rule_combo = QComboBox(); self.rule_combo.setEditable(True)
        self.rule_combo.addItems(["RL", "RLR", "LLRR", "RLLR", "LLR", "LRRRRRLLR", "RRLLLRLLLRRR"])
        self.rule_combo.setCurrentText("RL"); r1.addWidget(self.rule_combo); r1.addStretch(); lay.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(6
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setFixedWidth(90); self.play_btn.clicked.connect(self._toggle); r2.addWidget(self.play_btn)
        sb = QPushButton("⏭ Step"); sb.setFixedWidth(80); sb.clicked.connect(self._single_step); r2.addWidget(sb)
        rb = QPushButton("↺ Reset"); rb.setFixedWidth(80); rb.clicked.connect(self._reset); r2.addWidget(rb)
        r2.addSpacing(4); r2.addWidget(QLabel("Skip ×")); self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 100000); self.skip_spin.setValue(1000); self.skip_spin.setFixedWidth(80); r2.addWidget(self.skip_spin)
        skb = QPushButton("⏩"); skb.setFixedWidth(50); skb.clicked.connect(self._skip); r2.addWidget(skb)
        r2.addStretch()
        r2.addWidget(QLabel("Speed:")); self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 200); self.speed_slider.setValue(50); self.speed_slider.setFixedWidth(80)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'speed', max(5, 205-v)) or (self.timer.setInterval(self.speed) if self.running else None))
        r2.addWidget(self.speed_slider); r2.addSpacing(6)
        r2.addWidget(QLabel("Steps/frame:")); self.spf_spin = QSpinBox()
        self.spf_spin.setRange(1, 5000); self.spf_spin.setValue(10); self.spf_spin.setFixedWidth(70)
        self.spf_spin.valueChanged.connect(lambda v: setattr(self, 'spf', v)); r2.addWidget(self.spf_spin); lay.addLayout(r2)

        self.stats_lbl = QLabel("Steps: 0 | Colors: 2 | Rule: RL")
        self.stats_lbl.setStyleSheet("color:#00d4a8; font-weight:bold;"); lay.addWidget(self.stats_lbl)
        self.canvas = LangtonsAntCanvas(self.ant); lay.addWidget(self.canvas, 1)

    def _on_preset(self, name):
        if name in ANT_PRESETS: self.rule_combo.setCurrentText(ANT_PRESETS[name]); self._apply_rule()
    def _apply_rule(self):
        rule = "".join(c for c in self.rule_combo.currentText().upper() if c in "RLUN")
        if not rule: return
        was = self.running
        if was: self._toggle()
        self.ant = LangtonsAnt(150, 200, rule); self.canvas.ant = self.ant; self.canvas.rows = self.ant.rows; self.canvas.cols = self.ant.cols
        self.stats_lbl.setText(f"Steps: 0 | Colors: {self.ant.num_colors} | Rule: {rule}"); self.canvas.update()
    def _toggle(self):
        self.running = not self.running
        if self.running: self.play_btn.setText("⏸ Pause"); self.timer.start(self.speed)
        else: self.play_btn.setText("▶  Play"); self.timer.stop()
    def _tick(self):
        for _ in range(self.spf): self.ant.step()
        self._update_stats(); self.canvas.update()
    def _single_step(self): self.ant.step(); self._update_stats(); self.canvas.update()
    def _skip(self):
        for _ in range(self.skip_spin.value()): self.ant.step()
        self._update_stats(); self.canvas.update()
    def _reset(self): 
        if self.running: self._toggle()
        self._apply_rule()
    def _update_stats(self): self.stats_lbl.setText(f"Steps: {self.ant.steps:,} | Colors: {self.ant.num_colors} | Rule: {self.ant.rule}")
    def toggle_play(self): self._toggle()
    def single_step(self): self._single_step()
    def do_reset(self): self._reset()

# ─── Abelian Sandpile Model ────────────────────────────────────────────────────

class SandpileCanvas(BaseGridCanvas):
    COLORS = {0: QColor(16,16,28), 1: QColor(20,60,180), 2: QColor(0,180,200), 3: QColor(230,200,0)}
    def __init__(self, size=151, parent=None):
        super().__init__(size, size, parent)
        self.grid = [[0]*size for _ in range(size)]; self.total_grains = 0; self.toppled = False

    def reset(self):
        self.grid = [[0]*self.cols for _ in range(self.rows)]; self.total_grains = 0; self.toppled = False; super().reset_view()

    def add_grain(self, r, c, amount=1):
        if 0<=r<self.rows and 0<=c<self.cols: self.grid[r][c] += amount; self.total_grains += amount; self.update()

    def drop_center(self, amount):
        mid = self.rows//2; self.grid[mid][mid] += amount; self.total_grains += amount; self.update()

    def step(self):
        s = self.rows; new = [row[:] for row in self.grid]; self.toppled = False
        for r in range(s):
            for c in range(s):
                if self.grid[r][c] >= 4:
                    self.toppled = True; new[r][c] -= 4
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<s and 0<=nc<s: new[nr][nc] += 1
        self.grid = new; self.update()

    def stabilize(self):
        iters = 0
        while True:
            self.step(); iters += 1
            if not self.toppled or iters > 100000: break
        return iters

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            cell = self._cell_at(event.position())
            if cell: self.add_grain(cell[0], cell[1], 1)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self._dragging_pan: return
        if event.buttons() & Qt.LeftButton:
            cell = self._cell_at(event.position())
            if cell: self.add_grain(cell[0], cell[1], 1)

    def paintEvent(self, event):
        self._recalc(); p = QPainter(self); p.fillRect(self.rect(), QColor(16,16,28))
        cs, xo, yo = self._cs, self._xo, self._yo
        for r in range(self.rows):
            y = yo + r*cs
            for c in range(self.cols):
                v = self.grid[r][c]
                if v == 0: continue
                x = xo + c*cs
                color = self.COLORS.get(v, QColor(255,50,30))
                p.fillRect(QRectF(x, y, cs, cs), color)
        p.setPen(QPen(QColor(50,50,80),1)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(xo, yo, self.cols*cs, self.rows*cs))
        # Legend
        p.setFont(QFont("Segoe UI", 8)); lx = int(xo + 4); ly = int(yo + 12)
        for val, name in [(1, "1 grain"), (2, "2 grains"), (3, "3 grains")]:
            p.setBrush(self.COLORS[val]); p.setPen(Qt.NoPen); p.drawRect(lx, ly-8, 8, 8)
            p.setPen(QColor(180,180,200)); p.drawText(lx+12, ly, name); lx += 75
        p.end()

class SandpileTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = SandpileCanvas(151); self.running = False; self.speed = 30
        self._build(); self.timer = QTimer(self); self.timer.timeout.connect(self._step)

    def _build(self):
        lay = QVBoxLayout(self); lay.setSpacing(6)
        t = QLabel("Abelian Sandpile Model"); t.setFont(QFont("Segoe UI", 15, QFont.Bold)); t.setStyleSheet("color:#00d4a8;"); lay.addWidget(t)
        desc = QLabel("Add grains of sand. When a cell reaches 4, it topples, distributing 1 grain to each neighbor. Creates stunning fractals!")
        desc.setWordWrap(True); desc.setStyleSheet("color:#a0a0c0;"); lay.addWidget(desc)

        r1 = QHBoxLayout(); r1.setSpacing(6); r1.addWidget(QLabel("Center Drop:"))
        self.drop_spin = QSpinBox(); self.drop_spin.setRange(1, 1000000); self.drop_spin.setValue(1000); self.drop_spin.setSingleStep(100); r1.addWidget(self.drop_spin)
        db = QPushButton("💥 Drop Grains"); db.clicked.connect(self._drop_center); r1.addWidget(db)
        stab_btn = QPushButton("⚖️ Stabilize"); stab_btn.clicked.connect(self._stabilize); r1.addWidget(stab_btn)
        r1.addStretch(); lay.addLayout(r1)

        r2 = QHBoxLayout(); r2.setSpacing(6
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setFixedWidth(90); self.play_btn.clicked.connect(self._toggle); r2.addWidget(self.play_btn)
        sb = QPushButton("⏭ Step"); sb.setFixedWidth(80); sb.clicked.connect(self._step); r2.addWidget(sb)
        rb = QPushButton("↺ Reset"); rb.setFixedWidth(80); rb.clicked.connect(self._reset); r2.addWidget(rb)
        r2.addStretch()
        r2.addWidget(QLabel("Speed:")); self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 300); self.speed_slider.setValue(280); self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(lambda v: setattr(self, 'speed', 305-v) or (self.timer.setInterval(self.speed) if self.running else None))
        r2.addWidget(self.speed_slider); lay.addLayout(r2)

        self.stats = QLabel("Grains: 0 | Click to add | MidDrag: Pan | Scroll: Zoom")
        self.stats.setStyleSheet("color:#00d4a8; font-weight:bold; font-size:11px;"); lay.addWidget(self.stats)
        lay.addWidget(self.canvas, 1)

    def _drop_center(self): self.canvas.drop_center(self.drop_spin.value()); self._update_stats()
    def _toggle(self):
        self.running = not self.running
        if self.running: self.play_btn.setText("⏸ Pause"); self.timer.start(self.speed)
        else: self.play_btn.setText("▶  Play"); self.timer.stop()
    def _step(self): self.canvas.step(); self._update_stats()
    def _stabilize(self):
        if self.running: self._toggle()
        iters = self.canvas.stabilize(); self._update_stats()
        self.stats.setText(f"Stabilized in {iters} steps | Grains: {self.canvas.total_grains}")
    def _reset(self):
        if self.running: self._toggle()
        self.canvas.reset(); self._update_stats()
    def _update_stats(self): self.stats.setText(f"Grains: {self.canvas.total_grains} | Click to add | MidDrag: Pan | Scroll: Zoom")
    def toggle_play(self): self._toggle()
    def single_step(self): self._step()
    def do_reset(self): self._reset()

# ─── Learn Tab ─────────────────────────────────────────────────────────────────

class LearnTab(QWidget):
    SECTIONS = [
        ("🔬 What Are Cellular Automata?",
         "Cellular automata (CA) are discrete dynamical systems: a grid of cells, each in a finite set of states, updated synchronously via local rules. Despite their simplicity, CA can produce extraordinarily complex behavior — fractals, chaos, and even universal computation. First studied by Ulam and von Neumann (1940s), popularized by Conway's Game of Life (1970), and systematically classified by Wolfram (1980s)."),
        ("🔑 Key Concepts",
         "• Cell — Basic unit; typically binary (alive/dead) but can have more states.\n• Neighborhood — Cells that influence a cell's next state.\n    ◦ Von Neumann: 4 orthogonal neighbors (N, S, E, W)\n    ◦ Moore: 8 surrounding neighbors (used in Game of Life)\n    ◦ 1D: left and right neighbors\n• Rule — Function mapping neighborhood configurations → next state.\n• Generation — One synchronous update of all cells.\n• Boundary — Edges may wrap (toroidal), reflect, or be fixed.\n• B/S Notation — B{birth counts}/S{survive counts}, e.g. B3/S23."),
        ("📊 Wolfram's Four Classes",
         "Class I — Homogeneous fixed point (e.g. Rule 0). Like reaching equilibrium.\nClass II — Simple periodic structures (e.g. Rule 4). Like a pendulum.\nClass III — Chaotic, aperiodic (e.g. Rule 30). Like turbulence.\nClass IV — Complex localized structures (e.g. Rule 110). The most interesting!\n  Capable of universal computation — the 'edge of chaos'."),
        ("🧬 Conway's Game of Life (1970)",
         "Rules (B3/S23):\n• Birth: dead cell with exactly 3 neighbors → alive.\n• Survival: live cell with 2–3 neighbors → stays alive.\n• Death: all other live cells die.\n\nLife is Turing-complete — it can simulate any computation!\n\nPattern types:\n• Still lifes — Static (Block, Beehive)\n• Oscillators — Periodic (Blinker, Pulsar)\n• Spaceships — Moving (Glider, LWSS)\n• Methuselahs — Small start, long evolution (R-pentomino: 1103 gens)\n• Guns — Periodically emit spaceships (Gosper Glider Gun)"),
        ("⚡ Wireworld",
         "A 4-state CA designed to simulate electronic logic gates:\n• Empty (black) — Background.\n• Electron Head (blue) — The leading edge of an electron.\n• Electron Tail (red) — The trailing edge.\n• Copper (yellow) — Wire conductor.\n\nRules:\n• Empty → Empty\n• Head → Tail\n• Tail → Copper\n• Copper → Head if exactly 1 or 2 neighbors are Heads.\n\nYou can build clocks, diodes, logic gates (AND, OR, XOR), and even entire computers!"),
        ("🏖️ Abelian Sandpile Model",
         "A model of self-organized criticality:\n• Each cell holds a number representing grains of sand.\n• If a cell reaches 4 grains, it 'topples', sending 1 grain to each of its 4 neighbors.\n• Grains falling off the edge are lost.\n\nDespite the simple rule, dropping thousands of grains in the center produces breathtaking fractal patterns. This model is used in physics to study avalanches, earthquakes, and phase transitions."),
        ("🐜 Langton's Ant",
         "A simple 2D Turing machine:\n• At a cell in state k, the ant turns according to rule[k]\n  (R = right 90°, L = left 90°, U = 180°, N = no turn)\n• Changes the cell to state (k+1) mod num_colors\n• Moves forward one cell\n\nClassic ant (RL, 2 colors): chaotic for ~10,000 steps, then builds an infinite 'highway'! Multi-color ants create stunning patterns."),
        ("🚀 Exploration Tips",
         " 1. Rule 30: watch chaos from a single dot.\n 2. Rule 90: see the Sierpiński triangle emerge.\n 3. Click output cells in the Rule Table to design custom rules.\n 4. Place a Glider in Life and watch it travel.\n 5. Drop an R-pentomino — 3 cells → 1103 generations!\n 6. Try Brian's Brain for 3-state neuron-like patterns.\n 7. Switch to Age Gradient or Heat Trail display modes.\n 8. Use the Custom Rule Editor to discover new B/S rules.\n 9. Run Langton's Ant for 10,000+ steps to see the highway.\n10. Use ⚖️ Stabilize in Sandpile to instantly see fractals.\n11. Keyboard: Space = play/pause, → = step, R = reset, S = screenshot."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent); self._build()

    def _build(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget(); lay = QVBoxLayout(content); lay.setSpacing(18); lay.setContentsMargins(32, 24, 32, 24)
        for title, body in self.SECTIONS:
            tl = QLabel(title); tl.setFont(QFont("Segoe UI", 14, QFont.Bold)); tl.setStyleSheet("color:#00d4a8;"); lay.addWidget(tl)
            bl = QLabel(body); bl.setWordWrap(True); bl.setStyleSheet("color:#b0b0c8; line-height:1.6;"); bl.setTextFormat(Qt.PlainText); lay.addWidget(bl)
        lay.addStretch(); scroll.setWidget(content)
        outer = QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.addWidget(scroll)

# ─── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧩 Cellular Automata Explorer")
        self.setMinimumSize(950, 700); self.resize(1280, 900)
        self.setStyleSheet(DARK_STYLE)

        self.tabs = QTabWidget()
        self.tab_1d = ElementaryCATab()
        self.tab_cmp = ComparisonTab()
        self.tab_2d = GameOfLifeTab()
        self.tab_ww = WireworldTab()
        self.tab_ant = LangtonsAntTab()
        self.tab_sand = SandpileTab()
        self.tab_learn = LearnTab()

        self.tabs.addTab(self.tab_1d, "🔬 1D Elementary")
        self.tabs.addTab(self.tab_cmp, "📊 1D Comparison")
        self.tabs.addTab(self.tab_2d, "🧬 2D Game of Life")
        self.tabs.addTab(self.tab_ww, "⚡ Wireworld")
        self.tabs.addTab(self.tab_ant, "🐜 Langton's Ant")
        self.tabs.addTab(self.tab_sand, "🏖️ Sandpile")
        self.tabs.addTab(self.tab_learn, "📚 Learn")
        self.setCentralWidget(self.tabs)

        sb = QStatusBar(); self.setStatusBar(sb)
        sb.showMessage("Space: Play/Pause | →: Step | R: Reset | S: Screenshot | Middle-Drag: Pan | Scroll: Zoom")
        QShortcut(QKeySequence(Qt.Key_Space), self, self._space)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._right)
        QShortcut(QKeySequence(Qt.Key_R), self, self._r_key)
        QShortcut(QKeySequence(Qt.Key_S), self, self._s_key)

    def _current_sim_tab(self):
        i = self.tabs.currentIndex()
        if i == 0: return self.tab_1d
        if i == 1: return self.tab_cmp
        if i == 2: return self.tab_2d
        if i == 3: return self.tab_ww
        if i == 4: return self.tab_ant
        if i == 5: return self.tab_sand
        return None

    def _space(self): t = self._current_sim_tab(); t.toggle_play() if t else None
    def _right(self): t = self._current_sim_tab(); t.single_step() if t else None
    def _r_key(self): t = self._current_sim_tab(); t.do_reset() if t else None
    def _s_key(self):
        t = self._current_sim_tab()
        if t and hasattr(t, 'canvas'):
            path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "ca_screenshot.png", "PNG (*.png)")
            if path: t.canvas.grab().save(path)

# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())