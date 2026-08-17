# app.py
"""
Entry point for Cellular Automata Studio.
Usage: python app.py
"""
import sys
import os
import argparse
import logging

# Absolute imports since all files are in the same directory
from backends import HAS_NUMBA, HAS_CUPY
from window import MainWindow
from themes import apply_theme

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cellular Automata Studio v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --rule B36/S23 --preset Glider
  python app.py --random-fill 0.3 --theme matrix --fast
  python app.py --rows 300 --cols 300 --cell-size 3
        """
    )
    parser.add_argument("--rows", type=int, default=150, help="Grid rows (default: 150)")
    parser.add_argument("--cols", type=int, default=150, help="Grid cols (default: 150)")
    parser.add_argument("--cell-size", type=int, default=5, help="Cell size in pixels (default: 5)")
    parser.add_argument("--rule", type=str, default="B3/S23", help="B/S notation rule")
    parser.add_argument("--max-state", type=int, default=16, help="Max cell state value")
    parser.add_argument("--preset", type=str, default=None, help="Initial pattern to inject")
    parser.add_argument("--random-fill", type=float, default=None, help="Random fill density (0.0-1.0)")
    parser.add_argument("--theme", type=str, choices=["light", "dark", "matrix", "ocean", "cyberpunk"], default="dark")
    parser.add_argument("--backend", type=str, choices=["Auto", "Python", "NumPy", "Numba", "CuPy"], default="Auto")
    parser.add_argument("--fast", action="store_true", help="Shortcut for --backend Numba")
    parser.add_argument("--speed", type=int, default=50, help="Simulation speed in ms (1-500)")
    parser.add_argument("--steps-per-frame", type=int, default=1, help="Generations per frame (1-100)")
    parser.add_argument("--no-grid-lines", action="store_true", help="Hide grid lines")
    parser.add_argument("--trail", action="store_true", help="Enable trail/fade effect")
    parser.add_argument("--trail-length", type=int, default=15, help="Trail fade length (1-50)")
    parser.add_argument("--no-wrap", action="store_true", help="Disable toroidal wrapping")
    parser.add_argument("--symmetry", type=str, choices=["none", "horizontal", "vertical", "both", "rotational"], default="none")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser.parse_args()

def check_dependencies() -> dict:
    status = {'numpy': False, 'pyside6': False, 'numba': HAS_NUMBA, 'cupy': HAS_CUPY, 'scipy': False, 'pillow': False}
    try:
        import numpy; status['numpy'] = True
    except ImportError: pass
    try:
        import PySide6; status['pyside6'] = True
    except ImportError: pass
    try:
        import scipy; status['scipy'] = True
    except ImportError: pass
    try:
        import PIL; status['pillow'] = True
    except ImportError: pass
    return status

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    deps = check_dependencies()
    
    if not deps['numpy']:
        print("ERROR: NumPy is required. Install with: pip install numpy")
        sys.exit(1)
    if not deps['pyside6']:
        print("ERROR: PySide6 is required. Install with: pip install PySide6")
        sys.exit(1)

    logging.info(f"Dependencies: {deps}")

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    apply_theme(args.theme)
    app.setApplicationName("Cellular Automata Studio")
    app.setApplicationVersion("2.1")

    window = MainWindow()
    ca = window.ca_widget

    # Apply command line arguments
    if args.rows != 150 or args.cols != 150: 
        ca.resize_grid(args.rows, args.cols, preserve=False)
    if args.cell_size != 5: 
        ca.set_cell_size(args.cell_size)
    if args.rule != "B3/S23": 
        ca.set_rule(args.rule)
        window.rule_combo.setEditText(args.rule)
    if args.max_state != 16: 
        ca.set_max_state(args.max_state)
        window.max_state_spin.setValue(args.max_state)
    if args.fast: 
        ca.backend_manager.set_backend("Numba")
        window.backend_combo.setCurrentText("Numba")
    elif args.backend != "Auto": 
        ca.backend_manager.set_backend(args.backend)
        window.backend_combo.setCurrentText(args.backend)
    ca.set_speed(args.speed)
    window.speed_slider.setValue(args.speed)
    if args.steps_per_frame != 1:
        ca.set_steps_per_frame(args.steps_per_frame)
        window.steps_spin.setValue(args.steps_per_frame)
    if args.no_grid_lines: 
        ca.show_grid_lines = False
        window.gridlines_chk.setChecked(False)
    if args.trail: 
        ca.enable_trail(args.trail_length)
        window.trail_chk.setChecked(True)
        window.trail_length_spin.setValue(args.trail_length)
    if args.no_wrap:
        ca.set_wrap_mode(False)
        window.wrap_chk.setChecked(False)
    if args.symmetry != "none":
        ca.set_symmetry(args.symmetry)
        idx = window.symmetry_combo.findData(args.symmetry)
        if idx >= 0:
            window.symmetry_combo.setCurrentIndex(idx)
    
    window.set_theme(args.theme)
    
    if args.preset:
        from presets import PRESETS
        if args.preset in PRESETS:
            ca.inject_pattern(args.preset)
            window.preset_combo.setCurrentText(args.preset)
    if args.random_fill: 
        ca.randomize(args.random_fill)

    window.show()
    logging.info("Application started")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()