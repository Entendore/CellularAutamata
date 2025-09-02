import sys, json
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QSpinBox, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFileDialog
)
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPainter, QColor

# --- Cellular Automata Grid Widget ---
class CellularAutomataWidget(QWidget):
    def __init__(self, rows=20, cols=20, cell_size=20, rule="B3/S23"):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.initial_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_generation)
        self.dragging = False
        self.parse_rule(rule)

    def parse_rule(self, rule):
        try:
            b, s = rule.upper().split("/")
            self.birth = set(int(x) for x in b[1:])
            self.survive = set(int(x) for x in s[1:])
        except:
            self.birth = {3}
            self.survive = {2,3}

    def paintEvent(self, event):
        painter = QPainter(self)
        for r in range(self.rows):
            for c in range(self.cols):
                color = QColor(0,0,0) if self.grid[r][c] else QColor(255,255,255)
                painter.fillRect(c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size, color)
                painter.setPen(QColor(200,200,200))
                painter.drawRect(c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size)

    def mousePressEvent(self, event):
        if self.timer.isActive():  
            return
        self.dragging = True
        self.toggle_cell(event)

    def mouseMoveEvent(self, event):
        if self.dragging and not self.timer.isActive():
            self.toggle_cell(event)

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def toggle_cell(self, event):
        pos = event.position()  # <-- fixed for PyQt6
        x, y = pos.x(), pos.y()
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row][col] = 1 - self.grid[row][col]
            self.update()

    def next_generation(self):
        new_grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                alive_neighbors = sum(
                    self.grid[(r+dr)%self.rows][(c+dc)%self.cols]
                    for dr in (-1,0,1) for dc in (-1,0,1) if not (dr==0 and dc==0)
                )
                if self.grid[r][c] == 1 and alive_neighbors in self.survive:
                    new_grid[r][c] = 1
                elif self.grid[r][c] == 0 and alive_neighbors in self.birth:
                    new_grid[r][c] = 1
        self.grid = new_grid
        self.update()

    def reset_grid(self):
        self.grid = [row.copy() for row in self.initial_grid]
        self.update()
        self.timer.stop()

    def resize_grid(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.initial_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.update()

    def save_state(self, filename):
        data = {
            "rule": {"birth": list(self.birth), "survive": list(self.survive)},
            "rows": self.rows,
            "cols": self.cols,
            "cell_size": self.cell_size,
            "initial_grid": self.grid
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    def load_state(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
            self.rows = data.get("rows",20)
            self.cols = data.get("cols",20)
            self.cell_size = data.get("cell_size",20)
            rule_data = data.get("rule", {"birth":[3], "survive":[2,3]})
            self.birth = set(rule_data.get("birth",[3]))
            self.survive = set(rule_data.get("survive",[2,3]))
            self.grid = data.get("initial_grid", [[0]*self.cols for _ in range(self.rows)])
            self.initial_grid = [row.copy() for row in self.grid]
            self.update()

# --- Main Single-Window App ---
class CellularAutomataApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellular Automata Simulator")
        layout = QVBoxLayout()

        # Rule input
        rule_layout = QHBoxLayout()
        rule_layout.addWidget(QLabel("Rule (B/S):"))
        self.rule_input = QLineEdit("B3/S23")
        rule_layout.addWidget(self.rule_input)
        layout.addLayout(rule_layout)

        # Grid size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(5,100); self.rows_spin.setValue(20)
        size_layout.addWidget(self.rows_spin)
        size_layout.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(5,100); self.cols_spin.setValue(20)
        size_layout.addWidget(self.cols_spin)
        size_layout.addWidget(QLabel("Cell Size:"))
        self.cell_size_spin = QSpinBox(); self.cell_size_spin.setRange(5,50); self.cell_size_spin.setValue(20)
        size_layout.addWidget(self.cell_size_spin)
        layout.addLayout(size_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)
        btn_layout.addWidget(self.play_btn)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause)
        btn_layout.addWidget(self.pause_btn)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset)
        btn_layout.addWidget(self.reset_btn)
        self.save_btn = QPushButton("Save Config + Start State")
        self.save_btn.clicked.connect(self.save_config)
        btn_layout.addWidget(self.save_btn)
        self.load_btn = QPushButton("Load Config + Start State")
        self.load_btn.clicked.connect(self.load_config)
        btn_layout.addWidget(self.load_btn)
        layout.addLayout(btn_layout)

        # CA Widget
        self.ca_widget = CellularAutomataWidget(
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
            cell_size=self.cell_size_spin.value(),
            rule=self.rule_input.text()
        )
        layout.addWidget(self.ca_widget)
        self.setLayout(layout)

        # Update grid when size changes
        self.rows_spin.valueChanged.connect(self.update_grid_size)
        self.cols_spin.valueChanged.connect(self.update_grid_size)
        self.cell_size_spin.valueChanged.connect(self.update_grid_size)

    def update_grid_size(self):
        self.ca_widget.resize_grid(self.rows_spin.value(), self.cols_spin.value())

    def play(self):
        self.ca_widget.parse_rule(self.rule_input.text())
        self.ca_widget.timer.start(200)

    def pause(self):
        self.ca_widget.timer.stop()

    def reset(self):
        self.ca_widget.reset_grid()

    def save_config(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Config + Start State", "", "JSON Files (*.json)", options=options)
        if filename:
            self.ca_widget.save_state(filename)

    def load_config(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Config + Start State", "", "JSON Files (*.json)", options=options)
        if filename:
            self.ca_widget.load_state(filename)
            self.rows_spin.setValue(self.ca_widget.rows)
            self.cols_spin.setValue(self.ca_widget.cols)
            self.cell_size_spin.setValue(self.ca_widget.cell_size)
            self.rule_input.setText(f"B{''.join(map(str,self.ca_widget.birth))}/S{''.join(map(str,self.ca_widget.survive))}")

# --- Main ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CellularAutomataApp()
    window.resize(600, 700)
    window.show()
    sys.exit(app.exec())
