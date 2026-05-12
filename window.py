# window.py
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

from widgets import CellularAutomataWidget
from backends import BackendManager, HAS_NUMBA, HAS_CUPY
from presets import PRESET_CATEGORIES, PRESETS
from color_palettes import PaletteManager, PaletteGenerator, Color, Palette
from rulesets import RuleAnalyzer, get_suggested_rules
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