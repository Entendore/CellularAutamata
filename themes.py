"""Color themes and Qt styling."""
import numpy as np

THEMES = {
    "light": {"bg": (240, 240, 240), "grid": (220, 220, 220), 
              "states": [(100,149,237),(34,139,34),(220,20,60),(255,165,0),(148,0,211)]},
    "dark": {"bg": (30, 30, 30), "grid": (50, 50, 50),
             "states": [(0,255,127),(0,191,255),(255,0,127),(255,215,0),(138,43,226)]},
    "matrix": {"bg": (0, 10, 0), "grid": (0, 30, 0),
               "states": [(0,255,0),(0,200,0),(50,255,50),(0,150,0),(100,255,100)]},
    "ocean": {"bg": (10, 25, 50), "grid": (20, 40, 70),
              "states": [(0,150,255),(0,200,200),(100,200,255),(0,255,200),(50,100,255)]},
}

def get_theme_lut(theme_name: str, max_state: int) -> np.ndarray:
    theme = THEMES.get(theme_name, THEMES["dark"])
    lut = np.zeros((max_state, 3), dtype=np.uint8)
    lut[0] = theme["bg"]
    for i in range(1, max_state):
        lut[i] = theme["states"][(i - 1) % len(theme["states"])]
    return lut

def get_grid_color(theme_name: str) -> tuple:
    return THEMES.get(theme_name, THEMES["dark"])["grid"]

def apply_theme(theme_name: str) -> None:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if not app: return
    is_dark = theme_name != "light"
    if is_dark:
        app.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #ddd; font-size: 12px; }
            QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px; padding-top: 16px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #333; border: 1px solid #555; border-radius: 4px; padding: 5px 10px; }
            QPushButton:hover { background-color: #444; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2a2a2a; border: 1px solid #555; padding: 3px; }
            QTextEdit { background-color: #222; border: 1px solid #555; font-family: monospace; }
            QSlider::groove:horizontal { border: 1px solid #555; height: 6px; background: #333; border-radius: 3px; }
            QSlider::handle:horizontal { background: #666; width: 14px; margin: -5px 0; border-radius: 7px; }
            QScrollArea { border: none; }
        """)
    else:
        app.setStyleSheet("""
            QWidget { background-color: #f0f0f0; font-size: 12px; }
            QGroupBox { border: 1px solid #bbb; border-radius: 4px; margin-top: 8px; padding-top: 16px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #e0e0e0; border: 1px solid #aaa; border-radius: 4px; padding: 5px 10px; }
            QPushButton:hover { background-color: #d0d0d0; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: white; border: 1px solid #aaa; padding: 3px; }
            QTextEdit { background-color: white; border: 1px solid #aaa; }
        """)