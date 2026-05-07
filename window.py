# window.py
"""Main application window."""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QSpinBox, QLabel, QSlider, QComboBox, QGroupBox, QFormLayout, QScrollArea,
    QCheckBox, QMessageBox, QDoubleSpinBox, QStatusBar, QTabWidget, QSplitter,
    QTextEdit
)
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtCore import Qt

from widgets import CellularAutomataWidget
from backends import BackendManager, HAS_NUMBA, HAS_CUPY
from presets import PRESET_CATEGORIES
from color_palettes import PaletteManager
from rulesets import RuleAnalyzer


class MainWindow(QMainWindow):
    """Main application window containing all controls and the CA widget."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellular Automata Studio v2.0")
        self.resize(1280, 900)
        self.ca_widget = CellularAutomataWidget()
        self.palette_manager = PaletteManager()
        
        self.ca_widget.generation_updated.connect(self._upd_gen)
        self.ca_widget.population_updated.connect(self._upd_pop)
        self.ca_widget.fps_updated.connect(self._upd_fps)
        
        self._build_ui()
        self._build_shortcuts()
        self._update_status()

    def _build_ui(self) -> None:
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.ca_widget)
        self.scroll.setWidgetResizable(False)
        self.scroll.setMinimumSize(400, 400)
        
        tabs = QTabWidget()
        tabs.addTab(self._build_playback_tab(), "Controls")
        tabs.addTab(self._build_visuals_tab(), "Visuals & Palettes")
        tabs.addTab(self._build_rules_tab(), "Rules & Analysis")
        tabs.setMaximumWidth(350)
        tabs.setMinimumWidth(300)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.scroll)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.splitter = splitter
        
        container = QWidget()
        layout = QVBoxLayout(container)
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
        f.addRow("Generation:", self.gen_lbl)
        f.addRow("Population:", self.pop_lbl)
        f.addRow("FPS:", self.fps_lbl)
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
        f2.addRow("Speed (ms):", self.speed_slider)
        
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
        save_st_btn.clicked.connect(
            lambda: setattr(self.ca_widget, 'initial_grid', self.ca_widget.grid.copy())
        )
        reset_btn = QPushButton("↩ Reset")
        reset_btn.clicked.connect(self.ca_widget.undo)
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
        self.visual_mode_combo.addItems(["Standard", "Age (Inferno)", "Heatmap"])
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
        grp2.setLayout(f2)
        lay.addWidget(grp2)
        
        # Effect toggles
        grp3 = QGroupBox("Effects")
        v = QVBoxLayout()
        
        self.trail_chk = QCheckBox("Enable Trails")
        self.trail_chk.toggled.connect(self._on_trail_toggled)
        
        self.glow_chk = QCheckBox("Enable Glow (needs scipy)")
        self.glow_chk.toggled.connect(lambda c: setattr(self.ca_widget, 'glow_enabled', c))
        
        self.vignette_chk = QCheckBox("Enable Vignette")
        self.vignette_chk.toggled.connect(lambda c: setattr(self.ca_widget, 'vignette_enabled', c))
        
        self.gridlines_chk = QCheckBox("Show Grid Lines")
        self.gridlines_chk.setChecked(self.ca_widget.show_grid_lines)
        self.gridlines_chk.toggled.connect(self._on_gridlines_toggled)
        
        v.addWidget(self.trail_chk)
        v.addWidget(self.glow_chk)
        v.addWidget(self.vignette_chk)
        v.addWidget(self.gridlines_chk)
        grp3.setLayout(v)
        lay.addWidget(grp3)
        
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
            "B3/S23 (Life)", 
            "B36/S23 (HighLife)", 
            "B3678/S34678 (Day&Night)", 
            "B1357/S1357 (Replicator)", 
            "B2/S (Seeds)", 
            "B368/S245 (Morley)"
        ]
        self.rule_combo.addItems(rules)
        self.rule_combo.editTextChanged.connect(self._on_rule_changed)
        f.addRow("Rule (B/S):", self.rule_combo)
        
        self.max_state_spin = QSpinBox()
        self.max_state_spin.setRange(2, 256)
        self.max_state_spin.setValue(16)
        self.max_state_spin.valueChanged.connect(lambda v: self.ca_widget.set_max_state(v))
        f.addRow("Max States:", self.max_state_spin)
        grp.setLayout(f)
        lay.addWidget(grp)
        
        # Rule analysis
        grp2 = QGroupBox("Rule Analysis")
        v = QVBoxLayout()
        ana_btn = QPushButton("Analyze Current Rule")
        ana_btn.clicked.connect(self._analyze_rule)
        v.addWidget(ana_btn)
        
        self.ana_text = QTextEdit()
        self.ana_text.setReadOnly(True)
        self.ana_text.setMaximumHeight(200)
        v.addWidget(self.ana_text)
        grp2.setLayout(v)
        lay.addWidget(grp2)
        
        lay.addStretch()
        return w

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.ca_widget.step)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.ca_widget.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.ca_widget.redo)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_file)

    def _upd_gen(self, g: int) -> None:
        self.gen_lbl.setText(f"{g:,}")
    
    def _upd_pop(self, p: int) -> None:
        self.pop_lbl.setText(f"{p:,}")
    
    def _upd_fps(self, f: float) -> None:
        self.fps_lbl.setText(f"{f:.1f}")

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
        else:
            self.ca_widget.set_visual_mode("Standard")
    
    def _on_trail_toggled(self, checked: bool) -> None:
        if checked:
            self.ca_widget.enable_trail()
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
        self.ana_text.setText(
            f"Rule: {res['rule']}\n"
            f"Category: {res['category']}\n\n"
            f"Metrics:\n"
            f"  Expansion: {res['expansion']:.3f}\n"
            f"  Stability: {res['stability']:.3f}\n"
            f"  Chaos: {res['chaos']:.3f}"
        )

    def _save_file(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save", "", "JSON (*.json)")
        if p:
            self.ca_widget.save_to_file(p)

    def _load_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Load", "", "JSON (*.json)")
        if p:
            ok, d = self.ca_widget.load_from_file(p)
            if not ok:
                QMessageBox.critical(self, "Error", d.get("error", "Unknown error"))

    def _export_png(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG (*.png)")
        if p:
            if not self.ca_widget.export_to_png(p):
                QMessageBox.warning(
                    self, "Error", 
                    "Pillow is required for PNG export. Install with: pip install Pillow"
                )

    def set_theme(self, name: str) -> None:
        self.ca_widget.set_theme(name)
        # Auto select palette based on theme
        pal_map = {
            "dark": "Standard", 
            "matrix": "Neon", 
            "ocean": "Ocean", 
            "light": "Pastel"
        }
        pal = pal_map.get(name, "Standard")
        self.ca_widget.set_palette(pal)
        idx = self.palette_combo.findText(pal)
        if idx >= 0:
            self.palette_combo.setCurrentIndex(idx)

    def _update_status(self) -> None:
        b = self.ca_widget.backend_manager.get_effective_backend()
        self.statusBar().showMessage(
            f"Backend: {b} | Left=Draw, Right=Erase, Shift+Left=Cycle, Middle=Pan, Scroll=Zoom"
        )