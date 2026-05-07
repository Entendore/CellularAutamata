import sys, json, random
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSpinBox, QLabel, QTextEdit
)
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import QTimer, Qt


class CellularAutomataWidget(QWidget):
    def __init__(self, rows=20, cols=20, cell_size=20, max_state=2, rule="B3/S23"):
        super().__init__()
        self.rows, self.cols, self.cell_size = rows, cols, cell_size
        self.max_state = max_state
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.initial_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_generation)
        self.custom_rule_func = None
        self.parse_rule(rule)

        # Mouse interaction
        self.dragging = False
        self.last_button = None

        self.setFixedSize(cols * cell_size, rows * cell_size)

    # ----------- Drawing -----------
    def paintEvent(self, event):
        painter = QPainter(self)
        for r in range(self.rows):
            for c in range(self.cols):
                state = self.grid[r][c]
                if state == 0:
                    color = QColor(255, 255, 255)
                else:
                    hue = int((state / max(1, self.max_state)) * 255)
                    color = QColor.fromHsv(hue, 255, 255)
                painter.fillRect(c * self.cell_size, r * self.cell_size,
                                 self.cell_size, self.cell_size, color)
                painter.setPen(QColor(200, 200, 200))
                painter.drawRect(c * self.cell_size, r * self.cell_size,
                                 self.cell_size, self.cell_size)

    # ----------- Rules -----------
    def parse_rule(self, rule):
        self.birth, self.survive = [], []
        if "/" in rule:  # B3/S23 style
            try:
                b, s = rule.split("/")
                self.birth = [int(x) for x in b if x.isdigit()]
                self.survive = [int(x) for x in s if x.isdigit()]
            except:
                pass

    def set_custom_rule(self, code: str):
        local_env = {}
        try:
            exec(code, {}, local_env)
            if "rule" in local_env:
                self.custom_rule_func = local_env["rule"]
        except Exception as e:
            print("Error in rule code:", e)
            self.custom_rule_func = None

    # ----------- Evolution -----------
    def next_generation(self):
        new_grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.custom_rule_func:
                    try:
                        new_state = self.custom_rule_func(
                            self.grid, r, c, self.max_state, random
                        )
                        new_grid[r][c] = max(0, min(self.max_state - 1, int(new_state)))
                    except Exception as e:
                        print("Rule error:", e)
                        new_grid[r][c] = self.grid[r][c]
                else:
                    alive_neighbors = sum(
                        1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                        if (dr != 0 or dc != 0)
                        and self.grid[(r + dr) % self.rows][(c + dc) % self.cols] > 0
                    )
                    if self.grid[r][c] > 0 and alive_neighbors in self.survive:
                        new_grid[r][c] = self.grid[r][c]
                    elif self.grid[r][c] == 0 and alive_neighbors in self.birth:
                        new_grid[r][c] = 1
        self.grid = new_grid
        self.update()

    # ----------- Controls -----------
    def start(self): self.timer.start(200)
    def stop(self): self.timer.stop()
    def reset(self):
        self.grid = [row[:] for row in self.initial_grid]
        self.update()

    def save_state(self):
        self.initial_grid = [row[:] for row in self.grid]

    def save_to_file(self, path):
        data = {"grid": self.grid, "max_state": self.max_state}
        with open(path, "w") as f: json.dump(data, f)

    def load_from_file(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.grid = data.get("grid", self.grid)
        self.max_state = data.get("max_state", self.max_state)
        self.save_state()
        self.update()

    # ----------- Interaction -----------
    def mousePressEvent(self, event):
        if self.timer.isActive(): return
        self.dragging, self.last_button = True, event.button()
        self.toggle_cell(event, self.last_button)

    def mouseMoveEvent(self, event):
        if self.dragging and not self.timer.isActive():
            self.toggle_cell(event, self.last_button)

    def mouseReleaseEvent(self, event): self.dragging = False

    def toggle_cell(self, event, button):
        pos = event.position()
        col, row = int(pos.x() // self.cell_size), int(pos.y() // self.cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if button == Qt.MouseButton.LeftButton:
                self.grid[row][col] = (self.grid[row][col] + 1) % self.max_state
            elif button == Qt.MouseButton.RightButton:
                self.grid[row][col] = (self.grid[row][col] - 1) % self.max_state
            self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Cellular Automata")

        self.ca_widget = CellularAutomataWidget(rows=30, cols=30, max_state=4)

        self.play_btn = QPushButton("Play"); self.play_btn.clicked.connect(self.ca_widget.start)
        self.pause_btn = QPushButton("Pause"); self.pause_btn.clicked.connect(self.ca_widget.stop)
        self.reset_btn = QPushButton("Reset"); self.reset_btn.clicked.connect(self.ca_widget.reset)
        self.save_state_btn = QPushButton("Save State"); self.save_state_btn.clicked.connect(self.ca_widget.save_state)

        self.max_state_spin = QSpinBox(); self.max_state_spin.setValue(4); self.max_state_spin.setMinimum(2)
        self.max_state_spin.valueChanged.connect(self.update_max_state)

        self.save_btn = QPushButton("Save Config"); self.save_btn.clicked.connect(self.save_config)
        self.load_btn = QPushButton("Load Config"); self.load_btn.clicked.connect(self.load_config)

        self.rule_editor = QTextEdit()
        self.rule_editor.setPlaceholderText(
            "Define a rule(grid, r, c, max_state, random):\n"
            "Example:\n"
            "def rule(grid, r, c, max_state, random):\n"
            "    current = grid[r][c]\n"
            "    neighbors = sum(1 for dr in (-1,0,1) for dc in (-1,0,1)\n"
            "        if (dr!=0 or dc!=0) and grid[(r+dr)%len(grid)][(c+dc)%len(grid[0])] > 0)\n"
            "    if current == 0 and neighbors == 3 and random.random() < 0.7:\n"
            "        return 1\n"
            "    elif current > 0:\n"
            "        return (current + 1) % max_state\n"
            "    return 0"
        )
        self.apply_rule_btn = QPushButton("Apply Rule")
        self.apply_rule_btn.clicked.connect(self.apply_rule)

        controls = QHBoxLayout()
        controls.addWidget(self.play_btn); controls.addWidget(self.pause_btn)
        controls.addWidget(self.reset_btn); controls.addWidget(self.save_state_btn)
        controls.addWidget(QLabel("Max State:")); controls.addWidget(self.max_state_spin)
        controls.addWidget(self.save_btn); controls.addWidget(self.load_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.ca_widget)
        layout.addLayout(controls)
        layout.addWidget(QLabel("Custom Rule:"))
        layout.addWidget(self.rule_editor)
        layout.addWidget(self.apply_rule_btn)

        container = QWidget(); container.setLayout(layout)
        self.setCentralWidget(container)

    def update_max_state(self, value): self.ca_widget.max_state = value
    def apply_rule(self): self.ca_widget.set_custom_rule(self.rule_editor.toPlainText())

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON Files (*.json)")
        if path: self.ca_widget.save_to_file(path)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON Files (*.json)")
        if path: self.ca_widget.load_from_file(path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(); window.show()
    sys.exit(app.exec())
