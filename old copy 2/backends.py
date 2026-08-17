# backends.py
"""Compute backends for cellular automata evolution."""
import numpy as np
import warnings
import logging

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

logger = logging.getLogger(__name__)


class BackendManager:
    AVAILABLE_BACKENDS = ["Auto", "Python", "NumPy"]
    if HAS_NUMBA: AVAILABLE_BACKENDS.append("Numba")
    if HAS_CUPY: AVAILABLE_BACKENDS.append("CuPy")

    def __init__(self, preferred: str = "Auto"):
        self.current_backend = preferred
        self._evolve_func = None
        self._evolve_func_nowrap = None  # For non-wrapping mode
        self._update_backend()

    def _update_backend(self) -> None:
        """Update the active evolve function based on current backend setting."""
        if self.current_backend == "CuPy" and HAS_CUPY:
            self._evolve_func = _evolve_cupy
            self._evolve_func_nowrap = _evolve_cupy_nowrap
        elif self.current_backend == "Numba" and HAS_NUMBA:
            self._evolve_func = _evolve_numba
            self._evolve_func_nowrap = _evolve_numba_nowrap
        elif self.current_backend == "NumPy":
            self._evolve_func = _evolve_numpy
            self._evolve_func_nowrap = _evolve_numpy_nowrap
        elif self.current_backend == "Auto":
            self._evolve_func = _evolve_numba if HAS_NUMBA else _evolve_numpy
            self._evolve_func_nowrap = _evolve_numba_nowrap if HAS_NUMBA else _evolve_numpy_nowrap
        else:
            self._evolve_func = _evolve_python
            self._evolve_func_nowrap = _evolve_python_nowrap
        logger.debug(f"Backend set to {self.get_effective_backend()}")

    def set_backend(self, backend: str) -> bool:
        """Set the computation backend. Returns True if successful."""
        if backend in self.AVAILABLE_BACKENDS:
            self.current_backend = backend
            self._update_backend()
            return True
        warnings.warn(f"Backend '{backend}' not available. Available: {self.AVAILABLE_BACKENDS}")
        return False

    def evolve(self, grid: np.ndarray, birth: np.ndarray, survive: np.ndarray, wrap: bool = True) -> np.ndarray:
        """Evolve the grid by one generation."""
        if wrap:
            return self._evolve_func(grid, birth, survive)
        else:
            return self._evolve_func_nowrap(grid, birth, survive)

    def get_effective_backend(self) -> str:
        """Get the actual backend being used (resolves 'Auto')."""
        if self.current_backend == "Auto":
            return "Numba" if HAS_NUMBA else "NumPy"
        return self.current_backend


def _evolve_python(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Pure Python implementation with wrapping (slow, for reference/testing)."""
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            alive = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if grid[(r + dr) % rows, (c + dc) % cols] > 0:
                        alive += 1
            if grid[r, c] > 0:
                if alive < len(survive) and survive[alive]:
                    new_grid[r, c] = grid[r, c]
            else:
                if alive < len(birth) and birth[alive]:
                    new_grid[r, c] = 1
    return new_grid


def _evolve_python_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Pure Python implementation without wrapping."""
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for r in range(rows):
        for c in range(cols):
            alive = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0:
                        alive += 1
            if grid[r, c] > 0:
                if alive < len(survive) and survive[alive]:
                    new_grid[r, c] = grid[r, c]
            else:
                if alive < len(birth) and birth[alive]:
                    new_grid[r, c] = 1
    return new_grid


def _evolve_numpy(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Vectorized NumPy implementation with wrapping."""
    padded = np.pad(grid, 1, mode='wrap')
    alive = (padded > 0).astype(np.int32)
    neighbors = (alive[:-2, :-2] + alive[:-2, 1:-1] + alive[:-2, 2:] +
                 alive[1:-1, :-2] + alive[1:-1, 2:] +
                 alive[2:, :-2] + alive[2:, 1:-1] + alive[2:, 2:])
    
    is_alive = grid > 0
    
    # Clamp neighbor counts to valid lookup indices
    neighbors_clamped = np.clip(neighbors, 0, len(birth) - 1)
    
    birth_mask = ~is_alive & birth[neighbors_clamped]
    surv_mask = is_alive & survive[neighbors_clamped]
    
    new_grid = np.zeros_like(grid)
    new_grid[birth_mask] = 1
    new_grid[surv_mask] = grid[surv_mask]
    return new_grid


def _evolve_numpy_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
    """Vectorized NumPy implementation without wrapping."""
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    alive = (padded > 0).astype(np.int32)
    neighbors = (alive[:-2, :-2] + alive[:-2, 1:-1] + alive[:-2, 2:] +
                 alive[1:-1, :-2] + alive[1:-1, 2:] +
                 alive[2:, :-2] + alive[2:, 1:-1] + alive[2:, 2:])
    
    is_alive = grid > 0
    neighbors_clamped = np.clip(neighbors, 0, len(birth) - 1)
    
    birth_mask = ~is_alive & birth[neighbors_clamped]
    surv_mask = is_alive & survive[neighbors_clamped]
    
    new_grid = np.zeros_like(grid)
    new_grid[birth_mask] = 1
    new_grid[surv_mask] = grid[surv_mask]
    return new_grid


if HAS_NUMBA:
    @numba.njit(cache=True, parallel=True)
    def _evolve_numba(grid: np.ndarray, birth_lookup: np.ndarray, survive_lookup: np.ndarray) -> np.ndarray:
        """Numba-accelerated implementation with wrapping."""
        rows, cols = grid.shape
        new_grid = np.zeros_like(grid)
        for r in numba.prange(rows):
            for c in range(cols):
                alive = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr = (r + dr) % rows
                        nc = (c + dc) % cols
                        if grid[nr, nc] > 0:
                            alive += 1
                if grid[r, c] > 0:
                    if alive < len(survive_lookup) and survive_lookup[alive]:
                        new_grid[r, c] = grid[r, c]
                else:
                    if alive < len(birth_lookup) and birth_lookup[alive]:
                        new_grid[r, c] = 1
        return new_grid

    @numba.njit(cache=True, parallel=True)
    def _evolve_numba_nowrap(grid: np.ndarray, birth_lookup: np.ndarray, survive_lookup: np.ndarray) -> np.ndarray:
        """Numba-accelerated implementation without wrapping."""
        rows, cols = grid.shape
        new_grid = np.zeros_like(grid)
        for r in numba.prange(rows):
            for c in range(cols):
                alive = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0:
                            alive += 1
                if grid[r, c] > 0:
                    if alive < len(survive_lookup) and survive_lookup[alive]:
                        new_grid[r, c] = grid[r, c]
                else:
                    if alive < len(birth_lookup) and birth_lookup[alive]:
                        new_grid[r, c] = 1
        return new_grid
else:
    def _evolve_numba(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy(grid, birth, survive)
    
    def _evolve_numba_nowrap(grid: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy_nowrap(grid, birth, survive)


if HAS_CUPY:
    def _evolve_cupy(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        """GPU-accelerated implementation using CuPy with wrapping."""
        if grid_np.size < 10000:
            return _evolve_numpy(grid_np, birth, survive)
        
        try:
            g = cp.asarray(grid_np)
            g_bool = g > 0
            
            # Use padding instead of multiple rolls for efficiency
            padded = cp.pad(g_bool, 1, mode='wrap')
            neighbors = (
                padded[:-2, :-2].astype(cp.int32) + padded[:-2, 1:-1].astype(cp.int32) +
                padded[:-2, 2:].astype(cp.int32) + padded[1:-1, :-2].astype(cp.int32) +
                padded[1:-1, 2:].astype(cp.int32) + padded[2:, :-2].astype(cp.int32) +
                padded[2:, 1:-1].astype(cp.int32) + padded[2:, 2:].astype(cp.int32)
            )
            
            cp_b, cp_s = cp.asarray(birth), cp.asarray(survive)
            neighbors_clamped = cp.clip(neighbors, 0, len(cp_b) - 1)
            
            new_g = cp.zeros_like(g)
            alive_mask = g > 0
            new_g[~alive_mask & cp_b[neighbors_clamped]] = 1
            new_g[alive_mask & cp_s[neighbors_clamped]] = g[alive_mask & cp_s[neighbors_clamped]]
            
            return cp.asnumpy(new_g)
        except Exception as e:
            logger.warning(f"CuPy error, falling back to NumPy: {e}")
            return _evolve_numpy(grid_np, birth, survive)

    def _evolve_cupy_nowrap(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        """GPU-accelerated implementation using CuPy without wrapping."""
        if grid_np.size < 10000:
            return _evolve_numpy_nowrap(grid_np, birth, survive)
        
        try:
            g = cp.asarray(grid_np)
            g_bool = g > 0
            
            padded = cp.pad(g_bool, 1, mode='constant', constant_values=False)
            neighbors = (
                padded[:-2, :-2].astype(cp.int32) + padded[:-2, 1:-1].astype(cp.int32) +
                padded[:-2, 2:].astype(cp.int32) + padded[1:-1, :-2].astype(cp.int32) +
                padded[1:-1, 2:].astype(cp.int32) + padded[2:, :-2].astype(cp.int32) +
                padded[2:, 1:-1].astype(cp.int32) + padded[2:, 2:].astype(cp.int32)
            )
            
            cp_b, cp_s = cp.asarray(birth), cp.asarray(survive)
            neighbors_clamped = cp.clip(neighbors, 0, len(cp_b) - 1)
            
            new_g = cp.zeros_like(g)
            alive_mask = g > 0
            new_g[~alive_mask & cp_b[neighbors_clamped]] = 1
            new_g[alive_mask & cp_s[neighbors_clamped]] = g[alive_mask & cp_s[neighbors_clamped]]
            
            return cp.asnumpy(new_g)
        except Exception as e:
            logger.warning(f"CuPy error, falling back to NumPy: {e}")
            return _evolve_numpy_nowrap(grid_np, birth, survive)
else:
    def _evolve_cupy(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy(grid_np, birth, survive)
    
    def _evolve_cupy_nowrap(grid_np: np.ndarray, birth: np.ndarray, survive: np.ndarray) -> np.ndarray:
        return _evolve_numpy_nowrap(grid_np, birth, survive)