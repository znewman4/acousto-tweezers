# src/tweezers/viz/render_3d.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore


def normalize_gorkov_field(U: np.ndarray, verbose: bool = False, frame_id: str = "") -> tuple[np.ndarray, bool]:
    """
    Normalize Gorkov potential U to [0, 1] range for visualization.
    
    Uses PERCENTILE-BASED normalization to handle outliers:
    - Clips extreme values at 1st and 99th percentile
    - Maps the remaining range to [0, 1]
    - This ensures landscape topology is visible even with huge outlier values
    
    Handles edge cases:
    - Completely flat (Umin == Umax)
    - Nearly flat (tiny dynamic range)
    - NaN/Inf values
    
    Returns:
    --------
    Uvis : normalized U in [0, 1] 
    is_flat : True if the landscape was flat/nearly-flat
    
    When is_flat=True, the caller should consider reusing the previous frame
    instead of displaying a constant surface.
    """
    U = np.asarray(U, dtype=float)
    
    # Handle non-finite values
    finite_mask = np.isfinite(U)
    non_finite_count = int((~finite_mask).sum())
    total_count = U.size
    
    if non_finite_count == total_count:
        # All values are non-finite
        print(f"  [normalize {frame_id}] ALL NON-FINITE: returning mid-plane")
        return 0.5 * np.ones_like(U, dtype=float), True
    
    # Work with finite values only for statistics
    U_finite = U[finite_mask]
    
    # Use percentile-based clipping to handle outliers
    # This is KEY: the huge negative values are OUTLIERS that shouldn't dominate the colormap
    p1 = float(np.percentile(U_finite, 1))    # 1st percentile
    p99 = float(np.percentile(U_finite, 99))  # 99th percentile
    p50 = float(np.percentile(U_finite, 50))  # Median
    
    Umin_raw = float(np.nanmin(U))
    Umax_raw = float(np.nanmax(U))
    
    # Use percentile range for normalization
    den = p99 - p1
    
    is_flat = False
    
    if den <= 0 or den < 1e-15 * max(1.0, abs(p99)):
        # Completely flat or nearly flat based on percentile range
        print(f"  [normalize {frame_id}] FLAT: p1={p1:.3e}, p99={p99:.3e}, den={den:.3e}")
        Uvis = 0.5 * np.ones_like(U, dtype=float)
        is_flat = True
    else:
        # PERCENTILE-BASED NORMALIZATION (ignores outliers!)
        # Clip to percentile range then normalize
        U_clipped = np.clip(U, p1, p99)
        Uvis = (U_clipped - p1) / den
        Uvis = np.clip(Uvis, 0, 1)
        
        # Check the actual visible range
        Uvis_min = float(np.nanmin(Uvis))
        Uvis_max = float(np.nanmax(Uvis))
        Uvis_range = Uvis_max - Uvis_min
        
        # Only print if explicitly verbose
        if verbose:
            print(f"  [normalize {frame_id}] PERCENTILE_BASED:")
            print(f"           Raw range: [{Umin_raw:.3e}, {Umax_raw:.3e}]")
            print(f"           Percentile range (1%-99%): [{p1:.3e}, {p99:.3e}]")
            print(f"           Median: {p50:.3e}")
            print(f"           Uvis range: [{Uvis_min:.3f}, {Uvis_max:.3f}]")
        
        # If even after percentile normalization the range is tiny, flag as flat
        if Uvis_range < 0.1:
            if verbose:
                print(f"  [normalize {frame_id}] WARNING: Uvis range too small ({Uvis_range:.3f}), landscape may appear flat")
            is_flat = True
    
    return Uvis, is_flat


def classify_trap(eigvals: np.ndarray) -> str:
    """Classify critical point type from Hessian eigenvalues."""
    if np.all(eigvals > 0):
        return "min"
    if np.all(eigvals < 0):
        return "max"
    return "saddle"


def pick_best_stable_trap(traps) -> Optional[object]:
    """Pick one stable trap to track: lowest-U among minima."""
    mins = [t for t in traps if classify_trap(t.eigvals) == "min"]
    if not mins:
        return None
    return min(mins, key=lambda t: t.U)


def png_is_blankish(png_path: Path) -> tuple[bool, float, float, int]:
    """
    Read back a saved PNG and estimate if it's visually blank.
    Returns: (is_blankish, std, mean, filesize_bytes)
    """
    img = imageio.imread(png_path)
    if img.ndim == 3:
        gray = img[..., :3].mean(axis=2)
    else:
        gray = img
    std = float(gray.std())
    mean = float(gray.mean())
    fsize = int(os.path.getsize(png_path))
    return (std < 1.0), std, mean, fsize


# --- add to: src/tweezers/viz/render_3d.py ---

from dataclasses import dataclass

@dataclass(frozen=True)
class Cylinder2D:
    """A 2D actuator footprint extruded through z for visualisation."""
    x_mm: float
    y_mm: float
    r_mm: float
    z0: float = -0.25
    z1: float = 1.05
    alpha: float = 0.25  # translucent
    edge_alpha: float = 0.55  # slightly stronger outline


def _draw_cylinder_surface(
    ax,
    *,
    cyl: Cylinder2D,
    n_theta: int = 40,
    n_z: int = 2,
) -> None:
    """
    Draw a vertical cylinder as a surface (side wall only).
    Default is a translucent black wall; no top/bottom caps.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z = np.linspace(cyl.z0, cyl.z1, n_z)
    TH, ZZ = np.meshgrid(theta, z)

    XX = cyl.x_mm + cyl.r_mm * np.cos(TH)
    YY = cyl.y_mm + cyl.r_mm * np.sin(TH)

    # Use explicit facecolors so it stays "black translucent" and consistent.
    # Matplotlib needs an RGBA array matching the surface.
    fc = np.zeros((ZZ.shape[0], ZZ.shape[1], 4), dtype=float)
    fc[..., 3] = cyl.alpha  # black with alpha



    ax.plot_surface(
        XX, YY, ZZ,
        rstride=1, cstride=1,
        linewidth=0.6,
        antialiased=True,
        shade=False,
        facecolors=fc,
        edgecolor=(0, 0, 0, cyl.edge_alpha),
    )


def render_gorkov_landscape_frame_3d(
    *,
    out_png: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    U: np.ndarray,
    traps: Iterable[object],
    y_center_mm: Optional[float] = None,
    patch_len_mm: Optional[float] = None,
    track_xy_mm: Optional[list[tuple[float, float]]] = None,
    cylinders: Optional[list[Cylinder2D]] = None,   # NEW
    surface_stride: int = 3,
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    """
    Same as before, but with optional actuator cylinder overlays.
    """
    X, Y = np.meshgrid(x_mm, y_mm)

    # Use centralized normalization
    Uvis, is_flat = normalize_gorkov_field(U, verbose=True)
    
    Umin = float(np.nanmin(U))
    Umax = float(np.nanmax(U))
    den = Umax - Umin

    Xs = X[::surface_stride, ::surface_stride]
    Ys = Y[::surface_stride, ::surface_stride]
    Us = Uvis[::surface_stride, ::surface_stride]

    z0 = -0.25

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")


    ax.plot_surface(Xs, Ys, Us, linewidth=0, antialiased=True, alpha=0.95)
    ax.contour(X, Y, Uvis, levels=18, offset=z0)

    # NEW: cylinders
    if cylinders:
        for cyl in cylinders:
            _draw_cylinder_surface(ax, cyl=cyl)

    # Trap markers
    for t in traps:
        ttype = classify_trap(np.asarray(t.eigvals))
        mx, my = (float(t.x) * 1e3), (float(t.y) * 1e3)
        mz = 0.0 if den == 0.0 else (float(t.U) - Umin) / den

        if ttype == "min":
            ax.scatter(mx, my, mz, s=50, marker="o")
        elif ttype == "saddle":
            ax.scatter(mx, my, mz, s=50, marker="x")
        else:
            ax.scatter(mx, my, mz, s=55, marker="^")

    # Track line on floor
    if track_xy_mm is not None and len(track_xy_mm) >= 2:
        tx = [p[0] for p in track_xy_mm]
        ty = [p[1] for p in track_xy_mm]
        ax.plot(tx, ty, [z0] * len(tx), linewidth=2)

    # Keep old “patch on left boundary” annotation intact if provided
    if (y_center_mm is not None) and (patch_len_mm is not None):
        y0 = float(y_center_mm - 0.5 * patch_len_mm)
        y1 = float(y_center_mm + 0.5 * patch_len_mm)
        ax.plot([float(x_mm.min()), float(x_mm.min())], [y0, y1], [z0, z0], linewidth=4)
        ax.set_title(f"U(x,y) landscape — moving patch (yc={y_center_mm:.3f} mm)")
    else:
        ax.set_title("U(x,y) landscape — 2.5D bottom drive")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("U (normalised)")
    ax.set_zlim(z0, 1.05)
    ax.set_box_aspect((np.ptp(x_mm), np.ptp(y_mm), 0.8))
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
