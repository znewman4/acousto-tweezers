# src/acousto/analysis/traps_2d.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Trap2D:
    x: float
    y: float
    U: float
    Fx: float
    Fy: float
    K: np.ndarray  # 2x2 stiffness matrix (Hessian of U)
    eigvals: np.ndarray  # (2,)
    eigvecs: np.ndarray  # (2,2)


def _hessian_from_U(U: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hessian components Uxx, Uxy, Uyy using finite differences.
    Uses numpy.gradient twice. Shape preserved.
    """
    dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
    d2Udy2, d2Udydx = np.gradient(dUdy, dy, dx, edge_order=2)
    d2Udydx2, d2Udx2 = np.gradient(dUdx, dy, dx, edge_order=2)
    # d2Udydx and d2Udydx2 should be similar; take the average for symmetry
    Uxy = 0.5 * (d2Udydx + d2Udydx2)
    Uxx = d2Udx2
    Uyy = d2Udy2
    return Uxx, Uxy, Uyy


def find_traps_from_force(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    *,
    max_traps: int = 10,
    force_rel_thresh: float = 0.03,
    border: int = 2,
) -> list[Trap2D]:
    """
    Heuristic trap finder: identify local minima of |F| and return traps with stiffness.

    - Finds candidates where |F| is locally minimal in a 3x3 neighborhood.
    - Filters by |F| <= force_rel_thresh * max(|F|) (relative threshold).
    - Computes Hessian of U and returns eigenpairs for stability info.

    Notes:
      U, Fx, Fy are shape (Ny, Nx) with y-axis first.
    """
    Ny, Nx = U.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    Fmag = np.sqrt(Fx**2 + Fy**2)
    Fmax = float(np.max(Fmag))
    if Fmax == 0.0:
        return []

    thresh = force_rel_thresh * Fmax

    # Hessian for stiffness
    Uxx, Uxy, Uyy = _hessian_from_U(U, dx, dy)

    candidates: list[tuple[float, int, int]] = []

    # avoid edges where derivatives are less reliable
    j0 = border
    j1 = Ny - border
    i0 = border
    i1 = Nx - border

    for j in range(j0, j1):
        for i in range(i0, i1):
            val = Fmag[j, i]
            if val > thresh:
                continue
            # local minimum in 3x3
            nb = Fmag[j-1:j+2, i-1:i+2]
            if val <= np.min(nb):
                candidates.append((float(val), j, i))

    # sort by smallest |F|
    candidates.sort(key=lambda t: t[0])

    traps: list[Trap2D] = []
    used = np.zeros((Ny, Nx), dtype=bool)

    for _, j, i in candidates:
        if len(traps) >= max_traps:
            break
        # simple de-duplication: skip if near an already-chosen trap
        if used[j-2:j+3, i-2:i+3].any():
            continue
        used[j-2:j+3, i-2:i+3] = True

        K = np.array([[Uxx[j, i], Uxy[j, i]],
                      [Uxy[j, i], Uyy[j, i]]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(K)

        traps.append(
            Trap2D(
                x=float(x[i]),
                y=float(y[j]),
                U=float(U[j, i]),
                Fx=float(Fx[j, i]),
                Fy=float(Fy[j, i]),
                K=K,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
        )

    return traps


# =============================================================================
# STAGE B: Trap centre detection per step (field-based)
# =============================================================================

@dataclass
class TrapCenterResult:
    """Result from find_trap_center()."""
    x: float  # trap centre x (m)
    y: float  # trap centre y (m)
    stiffness_eigvals: np.ndarray  # (2,) eigenvalues of Hessian(U)
    is_stable: bool  # True if both eigenvalues are positive (local minimum)
    min_eigenvalue: float  # smallest eigenvalue (trap stiffness, positive = stable)
    U_at_trap: float  # Gor'kov potential at trap centre
    distance_from_particle: float  # distance from particle to trap (m)
    method: str  # 'gradient_descent' or 'grid_search' or 'fallback'
    # Additional diagnostics
    grad_norm: float = np.nan  # |∇U| at trap center
    hess_xx: float = np.nan  # Hessian components for debugging
    hess_xy: float = np.nan
    hess_yy: float = np.nan
    depth: float = np.nan  # trap depth (U_surround - U_at_trap)


def find_trap_center(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    particle_x: float,
    particle_y: float,
    *,
    search_radius: float = 0.3e-3,  # search window ± (m)
    use_gradient_refinement: bool = True,
    max_gradient_steps: int = 20,
    gradient_step_size: float = 0.01e-3,  # m per step
) -> TrapCenterResult:
    """
    Find the stable trap centre nearest to the particle position.
    
    STAGE B: Essential for trap-aware control. This function:
    1. Searches for local minima of U (or zeros of force) near the particle
    2. Chooses the minimum closest to current particle position (not global min)
    3. Computes stability via Hessian eigenvalues
    
    Parameters
    ----------
    x, y : np.ndarray
        1D coordinate arrays (m)
    U : np.ndarray
        2D Gor'kov potential field, shape (Ny, Nx)
    Fx, Fy : np.ndarray
        2D force fields, shape (Ny, Nx)
    particle_x, particle_y : float
        Current particle position (m)
    search_radius : float
        Search window radius around particle (m)
    use_gradient_refinement : bool
        If True, refine trap position using gradient descent on U
    max_gradient_steps : int
        Max iterations for gradient descent refinement
    gradient_step_size : float
        Step size for gradient descent (m)
    
    Returns
    -------
    TrapCenterResult with trap position, stability info, and diagnostics
    """
    Ny, Nx = U.shape
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    
    # Define search window indices
    x_min_idx = max(0, np.searchsorted(x, particle_x - search_radius) - 1)
    x_max_idx = min(Nx, np.searchsorted(x, particle_x + search_radius) + 1)
    y_min_idx = max(0, np.searchsorted(y, particle_y - search_radius) - 1)
    y_max_idx = min(Ny, np.searchsorted(y, particle_y + search_radius) + 1)
    
    # Ensure valid window
    if x_max_idx <= x_min_idx + 2 or y_max_idx <= y_min_idx + 2:
        # Fallback: use particle position
        return _fallback_trap_center(x, y, U, particle_x, particle_y, dx, dy)
    
    # Extract local region
    U_local = U[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
    Fx_local = Fx[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
    Fy_local = Fy[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
    
    # Compute force magnitude in local region
    Fmag_local = np.sqrt(Fx_local**2 + Fy_local**2)
    
    # Find candidates: local minima of |F| in 3x3 neighborhoods
    candidates = []
    border = 1
    local_Ny, local_Nx = U_local.shape
    
    for jl in range(border, local_Ny - border):
        for il in range(border, local_Nx - border):
            val = Fmag_local[jl, il]
            # Check if local minimum in 3x3
            neighborhood = Fmag_local[jl-1:jl+2, il-1:il+2]
            if val <= np.min(neighborhood):
                # Map back to global indices
                j_global = y_min_idx + jl
                i_global = x_min_idx + il
                
                # Distance from particle
                cx = float(x[i_global])
                cy = float(y[j_global])
                dist = np.sqrt((cx - particle_x)**2 + (cy - particle_y)**2)
                
                candidates.append((dist, val, j_global, i_global))
    
    if not candidates:
        # No local minimum found, use grid minimum of U in window
        min_idx = np.unravel_index(np.argmin(U_local), U_local.shape)
        j_global = y_min_idx + min_idx[0]
        i_global = x_min_idx + min_idx[1]
        candidates = [(0.0, 0.0, j_global, i_global)]
    
    # Sort by distance from particle, then by force magnitude
    candidates.sort(key=lambda c: (c[0], c[1]))
    
    # Take the closest candidate
    _, _, j_best, i_best = candidates[0]
    trap_x = float(x[i_best])
    trap_y = float(y[j_best])
    method = "grid_search"
    
    # Optional: gradient descent refinement on U
    if use_gradient_refinement:
        trap_x, trap_y = _refine_trap_gradient_descent(
            x, y, U, trap_x, trap_y,
            max_steps=max_gradient_steps,
            step_size=gradient_step_size,
            dx=dx, dy=dy,
        )
        method = "gradient_descent"
    
    # Compute stiffness (Hessian) at trap centre
    Uxx, Uxy, Uyy = _hessian_from_U(U, dx, dy)
    
    # Compute gradient for diagnostics
    dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
    
    # Find nearest grid indices for interpolation
    i_trap = int(np.clip(np.searchsorted(x, trap_x), 0, Nx - 1))
    j_trap = int(np.clip(np.searchsorted(y, trap_y), 0, Ny - 1))
    
    # Hessian at trap center
    hess_xx = float(Uxx[j_trap, i_trap])
    hess_xy = float(Uxy[j_trap, i_trap])
    hess_yy = float(Uyy[j_trap, i_trap])
    
    K = np.array([
        [hess_xx, hess_xy],
        [hess_xy, hess_yy],
    ], dtype=float)
    eigvals = np.linalg.eigvalsh(K)
    
    # Gradient magnitude at trap center
    grad_norm = float(np.sqrt(dUdx[j_trap, i_trap]**2 + dUdy[j_trap, i_trap]**2))
    
    # STABILITY: For Gor'kov potential, stable trap = local MINIMUM of U
    # At a minimum, Hessian eigenvalues are POSITIVE (bowl shape)
    # Particles naturally move toward lower U (F = -∇U)
    is_stable = bool(np.all(eigvals > 0) and np.all(np.isfinite(eigvals)))
    min_eigenvalue = float(np.min(eigvals))
    U_at_trap = float(U[j_trap, i_trap])
    distance_from_particle = float(np.sqrt(
        (trap_x - particle_x)**2 + (trap_y - particle_y)**2
    ))
    
    # Compute trap depth (difference between surrounding U and U_at_trap)
    # Use 5x5 neighborhood around trap
    j_lo = max(0, j_trap - 2)
    j_hi = min(Ny, j_trap + 3)
    i_lo = max(0, i_trap - 2)
    i_hi = min(Nx, i_trap + 3)
    U_neighborhood = U[j_lo:j_hi, i_lo:i_hi]
    trap_depth = float(np.max(U_neighborhood) - U_at_trap) if U_neighborhood.size > 0 else np.nan
    
    return TrapCenterResult(
        x=trap_x,
        y=trap_y,
        stiffness_eigvals=eigvals,
        is_stable=is_stable,
        min_eigenvalue=min_eigenvalue,
        U_at_trap=U_at_trap,
        distance_from_particle=distance_from_particle,
        method=method,
        grad_norm=grad_norm,
        hess_xx=hess_xx,
        hess_xy=hess_xy,
        hess_yy=hess_yy,
        depth=trap_depth,
    )


def _fallback_trap_center(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    particle_x: float,
    particle_y: float,
    dx: float,
    dy: float,
) -> TrapCenterResult:
    """Fallback when search window is too small."""
    Ny, Nx = U.shape
    Uxx, Uxy, Uyy = _hessian_from_U(U, dx, dy)
    
    i_p = int(np.clip(np.searchsorted(x, particle_x), 0, Nx - 1))
    j_p = int(np.clip(np.searchsorted(y, particle_y), 0, Ny - 1))
    
    # Hessian components
    hess_xx = float(Uxx[j_p, i_p])
    hess_xy = float(Uxy[j_p, i_p])
    hess_yy = float(Uyy[j_p, i_p])
    
    K = np.array([
        [hess_xx, hess_xy],
        [hess_xy, hess_yy],
    ], dtype=float)
    eigvals = np.linalg.eigvalsh(K)
    
    # Gradient for diagnostics
    dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
    grad_norm = float(np.sqrt(dUdx[j_p, i_p]**2 + dUdy[j_p, i_p]**2))
    
    # Compute trap depth
    j_lo = max(0, j_p - 2)
    j_hi = min(Ny, j_p + 3)
    i_lo = max(0, i_p - 2)
    i_hi = min(Nx, i_p + 3)
    U_neighborhood = U[j_lo:j_hi, i_lo:i_hi]
    U_at_trap = float(U[j_p, i_p])
    trap_depth = float(np.max(U_neighborhood) - U_at_trap) if U_neighborhood.size > 0 else np.nan
    
    # STABILITY: positive eigenvalues = minimum = stable trap
    return TrapCenterResult(
        x=particle_x,
        y=particle_y,
        stiffness_eigvals=eigvals,
        is_stable=bool(np.all(eigvals > 0) and np.all(np.isfinite(eigvals))),
        min_eigenvalue=float(np.min(eigvals)),
        U_at_trap=U_at_trap,
        distance_from_particle=0.0,
        method="fallback",
        grad_norm=grad_norm,
        hess_xx=hess_xx,
        hess_xy=hess_xy,
        hess_yy=hess_yy,
        depth=trap_depth,
    )


def _refine_trap_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    trap_x: float,
    trap_y: float,
    max_steps: int,
    step_size: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    """Refine trap position using gradient descent on U."""
    Ny, Nx = U.shape
    x_min, x_max = float(x[0]), float(x[-1])
    y_min, y_max = float(y[0]), float(y[-1])
    
    # Pre-compute gradient of U
    dUdy, dUdx = np.gradient(U, dy, dx, edge_order=2)
    
    for _ in range(max_steps):
        # Bilinear interpolation of gradient at current position
        # Find cell
        i = np.searchsorted(x, trap_x) - 1
        j = np.searchsorted(y, trap_y) - 1
        i = int(np.clip(i, 0, Nx - 2))
        j = int(np.clip(j, 0, Ny - 2))
        
        # Local coordinates
        tx = (trap_x - x[i]) / dx
        ty = (trap_y - y[j]) / dy
        tx = float(np.clip(tx, 0, 1))
        ty = float(np.clip(ty, 0, 1))
        
        # Bilinear interpolation of dU/dx
        dUdx_interp = (
            (1 - tx) * (1 - ty) * dUdx[j, i] +
            tx * (1 - ty) * dUdx[j, i + 1] +
            (1 - tx) * ty * dUdx[j + 1, i] +
            tx * ty * dUdx[j + 1, i + 1]
        )
        
        # Bilinear interpolation of dU/dy
        dUdy_interp = (
            (1 - tx) * (1 - ty) * dUdy[j, i] +
            tx * (1 - ty) * dUdy[j, i + 1] +
            (1 - tx) * ty * dUdy[j + 1, i] +
            tx * ty * dUdy[j + 1, i + 1]
        )
        
        # Gradient descent step (descending U)
        grad_mag = np.sqrt(dUdx_interp**2 + dUdy_interp**2)
        if grad_mag < 1e-20:
            break  # Converged
        
        trap_x -= step_size * (dUdx_interp / grad_mag)
        trap_y -= step_size * (dUdy_interp / grad_mag)
        
        # Clip to domain
        trap_x = float(np.clip(trap_x, x_min, x_max))
        trap_y = float(np.clip(trap_y, y_min, y_max))
    
    return trap_x, trap_y


# =============================================================================
# STAGE C: Trap identity continuity (TrapTracker)
# =============================================================================

@dataclass
class TrackedTrap:
    """A trap being tracked across timesteps."""
    x: float
    y: float
    stiffness_eigvals: np.ndarray
    is_stable: bool
    min_eigenvalue: float
    track_id: int  # unique identifier for this trap track
    frames_tracked: int  # number of consecutive frames this trap has been tracked
    lost: bool  # True if trap was lost (not found this frame)


class TrapTracker:
    """
    STAGE C: Maintains trap identity across timesteps.
    
    Prevents "random trap hopping" by tracking the same trap well over time.
    Uses nearest-neighbour matching with stiffness penalty.
    
    Usage:
        tracker = TrapTracker()
        for each timestep:
            trap_result = find_trap_center(...)
            tracked = tracker.update(trap_result, all_traps_in_domain)
    """
    
    def __init__(
        self,
        max_distance: float = 0.2e-3,  # max distance for matching (m)
        stiffness_weight: float = 0.1,  # weight on stiffness difference in matching
        lost_threshold: int = 5,  # frames before declaring lost
    ):
        self.max_distance = max_distance
        self.stiffness_weight = stiffness_weight
        self.lost_threshold = lost_threshold
        
        self._current_trap: TrackedTrap | None = None
        self._next_track_id: int = 0
        self._frames_lost: int = 0
    
    @property
    def current_trap(self) -> TrackedTrap | None:
        """Get the currently tracked trap, or None if lost."""
        return self._current_trap
    
    @property
    def is_tracking(self) -> bool:
        """True if actively tracking a trap."""
        return self._current_trap is not None and not self._current_trap.lost
    
    def update(
        self,
        trap_center: TrapCenterResult,
        all_traps: list[Trap2D] | None = None,
    ) -> TrackedTrap:
        """
        Update tracker with new trap detection.
        
        Parameters
        ----------
        trap_center : TrapCenterResult
            Result from find_trap_center() for this timestep
        all_traps : list[Trap2D], optional
            All detected traps in domain (for fallback matching)
        
        Returns
        -------
        TrackedTrap with current trap state and tracking info
        """
        if self._current_trap is None:
            # First frame: initialize tracking
            self._current_trap = TrackedTrap(
                x=trap_center.x,
                y=trap_center.y,
                stiffness_eigvals=trap_center.stiffness_eigvals,
                is_stable=trap_center.is_stable,
                min_eigenvalue=trap_center.min_eigenvalue,
                track_id=self._next_track_id,
                frames_tracked=1,
                lost=False,
            )
            self._next_track_id += 1
            self._frames_lost = 0
            return self._current_trap
        
        # Compute distance from previous trap position
        dist = np.sqrt(
            (trap_center.x - self._current_trap.x)**2 +
            (trap_center.y - self._current_trap.y)**2
        )
        
        # Compute matching cost (distance + stiffness difference)
        stiffness_diff = abs(
            trap_center.min_eigenvalue - self._current_trap.min_eigenvalue
        )
        cost = dist + self.stiffness_weight * stiffness_diff
        
        # Check if new detection matches current track
        if dist <= self.max_distance:
            # Good match: update tracked trap
            self._current_trap = TrackedTrap(
                x=trap_center.x,
                y=trap_center.y,
                stiffness_eigvals=trap_center.stiffness_eigvals,
                is_stable=trap_center.is_stable,
                min_eigenvalue=trap_center.min_eigenvalue,
                track_id=self._current_trap.track_id,
                frames_tracked=self._current_trap.frames_tracked + 1,
                lost=False,
            )
            self._frames_lost = 0
        else:
            # No match: increment lost counter
            self._frames_lost += 1
            
            if self._frames_lost >= self.lost_threshold:
                # Trap truly lost: start new track
                self._current_trap = TrackedTrap(
                    x=trap_center.x,
                    y=trap_center.y,
                    stiffness_eigvals=trap_center.stiffness_eigvals,
                    is_stable=trap_center.is_stable,
                    min_eigenvalue=trap_center.min_eigenvalue,
                    track_id=self._next_track_id,
                    frames_tracked=1,
                    lost=False,
                )
                self._next_track_id += 1
                self._frames_lost = 0
            else:
                # Mark as temporarily lost but keep previous position
                self._current_trap = TrackedTrap(
                    x=self._current_trap.x,  # keep old position
                    y=self._current_trap.y,
                    stiffness_eigvals=self._current_trap.stiffness_eigvals,
                    is_stable=self._current_trap.is_stable,
                    min_eigenvalue=self._current_trap.min_eigenvalue,
                    track_id=self._current_trap.track_id,
                    frames_tracked=self._current_trap.frames_tracked,
                    lost=True,
                )
        
        return self._current_trap
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._current_trap = None
        self._frames_lost = 0
    
    def get_status(self) -> dict:
        """Get tracker status for logging."""
        if self._current_trap is None:
            return {"tracking": False, "track_id": -1, "frames_tracked": 0, "lost": True}
        return {
            "tracking": self.is_tracking,
            "track_id": self._current_trap.track_id,
            "frames_tracked": self._current_trap.frames_tracked,
            "lost": self._current_trap.lost,
            "x_mm": self._current_trap.x * 1e3,
            "y_mm": self._current_trap.y * 1e3,
            "min_eigenvalue": self._current_trap.min_eigenvalue,
        }

