# themes.py
"""Color themes and Qt styling."""
import numpy as np
import logging

logger = logging.getLogger(__name__)

THEMES = {
    "light": {
        "bg": (240, 240, 240), 
        "grid": (220, 220, 220), 
        "states": [(100,149,237), (34,139,34), (220,20,60), (255,165,0), (148,0,211)]
    },
    "dark": {
        "bg": (30, 30, 30), 
        "grid": (50, 50, 50),
        "states": [(0,255,127), (0,191,255), (255,0,127), (255,215,0), (138,43,226)]
    },
    "matrix": {
        "bg": (0, 10, 0), 
        "grid": (0, 30, 0),
        "states": [(0,255,0), (0,200,0), (50,255,50), (0,150,0), (100,255,100)]
    },
    "ocean": {
        "bg": (10, 25, 50), 
        "grid": (20, 40, 70),
        "states": [(0,150,255), (0,200,200), (100,200,255), (0,255,200), (50,100,255)]
    },
    "cyberpunk": {
        "bg": (20, 10, 30), 
        "grid": (40, 20, 50),
        "states": [(255,0,110), (0,255,255), (255,234,0), (131,56,236), (255,100,200)]
    },
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

def get_background_color(theme_name: str) -> tuple:
    return THEMES.get(theme_name, THEMES["dark"])["bg"]

def apply_theme(theme_name: str) -> None:
    """Apply Qt stylesheet based on theme."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if not app:
        return
    
    is_dark = theme_name != "light"
    
    if is_dark:
        if theme_name == "matrix":
            accent = "#00ff00"
            accent_dark = "#003300"
        elif theme_name == "ocean":
            accent = "#00aaff"
            accent_dark = "#001a33"
        elif theme_name == "cyberpunk":
            accent = "#ff006e"
            accent_dark = "#140a1e"
        else:
            accent = "#00ff7f"
            accent_dark = "#1e1e1e"
        
        app.setStyleSheet(f"""
            QWidget {{ 
                background-color: {accent_dark}; 
                color: #ddd; 
                font-size: 12px; 
            }}
            QGroupBox {{ 
                border: 1px solid #444; 
                border-radius: 4px; 
                margin-top: 8px; 
                padding-top: 16px; 
                font-weight: bold; 
            }}
            QGroupBox::title {{ 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
                color: {accent};
            }}
            QPushButton {{ 
                background-color: #333; 
                border: 1px solid #555; 
                border-radius: 4px; 
                padding: 5px 10px; 
                min-height: 20px;
            }}
            QPushButton:hover {{ 
                background-color: #444; 
                border-color: {accent};
            }}
            QPushButton:pressed {{ 
                background-color: {accent_dark}; 
            }}
            QComboBox {{ 
                background-color: #2a2a2a; 
                border: 1px solid #555; 
                padding: 3px; 
                min-height: 20px;
            }}
            QComboBox::drop-down {{ 
                border: none; 
            }}
            QComboBox QAbstractItemView {{
                background-color: #2a2a2a;
                selection-background-color: #444;
                border: 1px solid #555;
            }}
            QSpinBox, QDoubleSpinBox {{ 
                background-color: #2a2a2a; 
                border: 1px solid #555; 
                padding: 3px;
                min-height: 20px;
            }}
            QTextEdit {{ 
                background-color: #222; 
                border: 1px solid #555; 
                font-family: monospace;
                font-size: 11px;
            }}
            QSlider::groove:horizontal {{ 
                border: 1px solid #555; 
                height: 6px; 
                background: #333; 
                border-radius: 3px; 
            }}
            QSlider::handle:horizontal {{ 
                background: {accent}; 
                width: 14px; 
                margin: -5px 0; 
                border-radius: 7px; 
            }}
            QSlider::sub-page:horizontal {{
                background: {accent};
                border-radius: 3px;
            }}
            QScrollArea {{ 
                border: none; 
            }}
            QTabWidget::pane {{
                border: 1px solid #444;
            }}
            QTabBar::tab {{
                background-color: #2a2a2a;
                border: 1px solid #444;
                padding: 6px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: #333;
                border-bottom-color: #333;
                color: {accent};
            }}
            QTabBar::tab:hover {{
                background-color: #3a3a3a;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2a2a2a;
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent};
                border-color: {accent};
            }}
            QStatusBar {{
                background-color: #1a1a1a;
                color: #888;
            }}
            QLabel {{
                background: transparent;
            }}
        """)
    else:
        app.setStyleSheet("""
            QWidget { 
                background-color: #f0f0f0; 
                font-size: 12px; 
            }
            QGroupBox { 
                border: 1px solid #bbb; 
                border-radius: 4px; 
                margin-top: 8px; 
                padding-top: 16px; 
                font-weight: bold; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
            }
            QPushButton { 
                background-color: #e0e0e0; 
                border: 1px solid #aaa; 
                border-radius: 4px; 
                padding: 5px 10px;
                min-height: 20px;
            }
            QPushButton:hover { 
                background-color: #d0d0d0; 
            }
            QPushButton:pressed { 
                background-color: #c0c0c0; 
            }
            QComboBox { 
                background-color: white; 
                border: 1px solid #aaa; 
                padding: 3px;
                min-height: 20px;
            }
            QSpinBox, QDoubleSpinBox { 
                background-color: white; 
                border: 1px solid #aaa; 
                padding: 3px;
                min-height: 20px;
            }
            QTextEdit { 
                background-color: white; 
                border: 1px solid #aaa;
                font-family: monospace;
                font-size: 11px;
            }
            QSlider::groove:horizontal { 
                border: 1px solid #aaa; 
                height: 6px; 
                background: #ddd; 
                border-radius: 3px; 
            }
            QSlider::handle:horizontal { 
                background: #666; 
                width: 14px; 
                margin: -5px 0; 
                border-radius: 7px; 
            }
            QScrollArea { 
                border: none; 
            }
            QTabWidget::pane {
                border: 1px solid #bbb;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #bbb;
                padding: 6px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #f0f0f0;
                border-bottom-color: #f0f0f0;
            }
            QStatusBar {
                background-color: #e0e0e0;
                color: #666;
            }
            QLabel {
                background: transparent;
            }
        """)
    
    logger.debug(f"Applied theme: {theme_name}")