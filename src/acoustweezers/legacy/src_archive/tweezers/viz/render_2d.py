"""
2D visualization renderers for acoustic trapping.

Provides three complementary 2D views:
  - Force quiver: vector field of radiation forces
  - Contour: Gorkov potential landscape
  - Multiview: both combined (separate files)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def render_force_quiver(
    out_png: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    particle_xy_mm: tuple[float, float],
    traj_xy_mm: list[tuple[float, float]] | None = None,
    pucks: list[tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """
    2D quiver plot showing radiation force field.
    
    Parameters
    ----------
    out_png : Path
        Output file path
    x_mm, y_mm : ndarray
        Coordinate grids (1D arrays, mm)
    Fx, Fy : ndarray
        Force components, shape (Ny, Nx)
    particle_xy_mm : tuple
        Current particle position (x_mm, y_mm)
    traj_xy_mm : list of tuple, optional
        Particle trajectory history
    pucks : list of tuple, optional
        Transducer positions [(x1, y1), (x2, y2), ...]
    figsize : tuple
        Figure size (inches)
    """
    px_mm, py_mm = particle_xy_mm
    
    X, Y = np.meshgrid(x_mm, y_mm)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background: light gray
    ax.set_facecolor("#f0f0f0")
    
    # Sample every Nth point for quiver (N≈6)
    N = max(1, len(x_mm) // 20)  # Roughly 20 arrows per dimension
    
    # Normalize forces for visualization
    Famp = np.sqrt(Fx**2 + Fy**2)
    scale = np.max(Famp) + 1e-12
    Fx_norm = Fx / scale
    Fy_norm = Fy / scale
    
    # Draw force vectors
    ax.quiver(
        X[::N, ::N], Y[::N, ::N],
        Fx_norm[::N, ::N], Fy_norm[::N, ::N],
        Famp[::N, ::N],  # Color by magnitude
        cmap="hot", scale=25, width=0.003, alpha=0.8
    )
    
    # Draw trajectory trail (cyan)
    if traj_xy_mm is not None and len(traj_xy_mm) >= 2:
        tx = np.array([p[0] for p in traj_xy_mm])
        ty = np.array([p[1] for p in traj_xy_mm])
        
        n_pts = len(tx)
        for i in range(n_pts - 1):
            alpha_color = i / max(n_pts - 1, 1)
            color = (1-alpha_color) * np.array([0.5, 0.5, 0.5]) + alpha_color * np.array([0, 1, 1])
            ax.plot(tx[i:i+2], ty[i:i+2], linewidth=2.0, color=color, alpha=0.9)
    
    # Draw particle (large red dot)
    ax.scatter(px_mm, py_mm, s=300, marker="o", color="red", edgecolors="white", 
               linewidth=3, zorder=100, label="particle")
    
    # Draw pucks (small dark circles)
    if pucks is not None:
        for (px, py) in pucks:
            ax.scatter(px, py, s=120, c="black", marker="o", alpha=0.7, zorder=50)
    
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_title("Radiation Force Field (Quiver)", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def render_contour_with_particle(
    out_png: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    U: np.ndarray,
    particle_xy_mm: tuple[float, float],
    traj_xy_mm: list[tuple[float, float]] | None = None,
    pucks: list[tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """
    2D contour plot of Gorkov potential with particle trail.
    
    Parameters
    ----------
    out_png : Path
        Output file path
    x_mm, y_mm : ndarray
        Coordinate grids (1D arrays, mm)
    U : ndarray
        Gorkov potential, shape (Ny, Nx)
    particle_xy_mm : tuple
        Current particle position (x_mm, y_mm)
    traj_xy_mm : list of tuple, optional
        Particle trajectory history
    pucks : list of tuple, optional
        Transducer positions
    figsize : tuple
        Figure size (inches)
    """
    px_mm, py_mm = particle_xy_mm
    
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # Robust U visualization
    U_min = float(np.min(U))
    U_max = float(np.max(U))
    U_mean = float(np.mean(U))
    U_std = float(np.std(U))
    den = U_max - U_min
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if potential has meaningful variation
    # Use std as primary indicator, den as secondary
    if U_std < 1e-22 or (den == 0.0 or not np.isfinite(den)):
        # Potential is essentially flat
        ax.set_facecolor("#ffffcc")  # pale yellow
        ax.text(0.5, 0.5, 
                f"U ≈ {U_mean:.2e} J\nstd = {U_std:.2e} (too small)", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    else:
        # Normalize by centering around mean with std as scale
        # This gives better visualization for small variations
        Uvis = (U - U_mean) / (U_std + 1e-30)  # ±N sigma scale
        
        # Create contours centered around 0 (mean), ranging ±3 sigma
        levels = np.linspace(-3, 3, 25)
        
        contourf = ax.contourf(X, Y, Uvis, levels=levels, cmap="RdBu_r", alpha=0.85)
        ax.contour(X, Y, Uvis, levels=levels[::2], colors="k", linewidths=0.3, alpha=0.15)
        
        # Add colorbar showing sigma levels
        cbar = fig.colorbar(contourf, ax=ax, label=f"U (σ units)\nstd={U_std:.2e} J")
    
    # Always set domain limits so grid is visible even if flat
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(y_mm[0], y_mm[-1])
    
    # Draw trajectory trail (cyan with gradient)
    if traj_xy_mm is not None and len(traj_xy_mm) >= 2:
        tx = np.array([p[0] for p in traj_xy_mm])
        ty = np.array([p[1] for p in traj_xy_mm])
        
        n_pts = len(tx)
        for i in range(n_pts - 1):
            alpha_color = i / max(n_pts - 1, 1)
            color = (1-alpha_color) * np.array([0.5, 0.5, 0.5]) + alpha_color * np.array([0, 1, 1])
            ax.plot(tx[i:i+2], ty[i:i+2], linewidth=2.5, color=color, alpha=0.95)
    
    # Draw particle (large red dot)
    ax.scatter(px_mm, py_mm, s=300, marker="o", color="red", edgecolors="white",
               linewidth=3, zorder=100, label="particle")
    
    # Draw pucks (small dark circles)
    if pucks is not None:
        for (px, py) in pucks:
            ax.scatter(px, py, s=120, c="black", marker="o", alpha=0.7, zorder=50)
    
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_title("Gorkov Potential Landscape", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def render_force_magnitude_landscape(
    out_png: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    particle_xy_mm: tuple[float, float],
    traj_xy_mm: list[tuple[float, float]] | None = None,
    pucks: list[tuple[float, float]] | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> None:
    """
    2D contour plot of force magnitude (acoustic "landscape").
    
    Uses force magnitude instead of potential - forces have larger absolute
    values (~10^-12 N) and render better than small potentials (~10^-18 J).
    
    Parameters
    ----------
    out_png : Path
        Output file path
    x_mm, y_mm : ndarray
        Coordinate grids (1D arrays, mm)
    Fx, Fy : ndarray
        Force components, shape (Ny, Nx)
    particle_xy_mm : tuple
        Current particle position (x_mm, y_mm)
    traj_xy_mm : list of tuple, optional
        Particle trajectory history
    pucks : list of tuple, optional
        Transducer positions
    figsize : tuple
        Figure size (inches)
    """
    px_mm, py_mm = particle_xy_mm
    
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # Compute force magnitude (this is the "landscape")
    F_mag = np.sqrt(Fx**2 + Fy**2)
    F_min = float(np.min(F_mag))
    F_max = float(np.max(F_mag))
    F_std = float(np.std(F_mag))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize force magnitude for better visualization
    if F_max > F_min and np.isfinite(F_max):
        # Use log scale if range spans multiple orders of magnitude
        F_range = F_max - F_min
        if F_range > 0:
            # Normalize to [0, 1]
            F_norm = (F_mag - F_min) / (F_range + 1e-30)
            
            # Use log-like scaling for better visibility
            # Apply power law: F_norm^0.5 stretches lower values
            F_vis = np.power(np.maximum(F_norm, 0), 0.6)
            
            levels = np.linspace(0, 1, 30)
            contourf = ax.contourf(X, Y, F_vis, levels=levels, cmap="viridis", alpha=0.85)
            ax.contour(X, Y, F_vis, levels=levels[::3], colors="k", linewidths=0.3, alpha=0.15)
            
            cbar = fig.colorbar(contourf, ax=ax, label=f"Force Magnitude (landscape)\nmax={F_max:.2e} N")
        else:
            ax.set_facecolor("#e0e0e0")
            ax.text(0.5, 0.5, "F_mag ≈ uniform", ha='center', va='center',
                    transform=ax.transAxes, fontsize=11)
    else:
        ax.set_facecolor("#e0e0e0")
        ax.text(0.5, 0.5, "No force field", ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
    
    # Always set domain limits
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.set_ylim(y_mm[0], y_mm[-1])
    
    # Draw trajectory trail (cyan with gradient)
    if traj_xy_mm is not None and len(traj_xy_mm) >= 2:
        tx = np.array([p[0] for p in traj_xy_mm])
        ty = np.array([p[1] for p in traj_xy_mm])
        
        n_pts = len(tx)
        for i in range(n_pts - 1):
            alpha_color = i / max(n_pts - 1, 1)
            color = (1-alpha_color) * np.array([0.5, 0.5, 0.5]) + alpha_color * np.array([0, 1, 1])
            ax.plot(tx[i:i+2], ty[i:i+2], linewidth=2.5, color=color, alpha=0.95)
    
    # Draw particle (large red dot)
    ax.scatter(px_mm, py_mm, s=300, marker="o", color="red", edgecolors="white",
               linewidth=3, zorder=100, label="particle")
    
    # Draw pucks (small dark circles)
    if pucks is not None:
        for (px, py) in pucks:
            ax.scatter(px, py, s=120, c="black", marker="o", alpha=0.7, zorder=50)
    
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_title("Force Magnitude Landscape", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def render_multiview(
    out_png_prefix: str,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    U: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    particle_xy_mm: tuple[float, float],
    traj_xy_mm: list[tuple[float, float]] | None = None,
    pucks: list[tuple[float, float]] | None = None,
) -> list[Path]:
    """
    Render both quiver and force landscape views.
    
    Parameters
    ----------
    out_png_prefix : str
        Path prefix for output files (e.g., "/path/frame2d_0000")
        Will create: {prefix}_quiver.png, {prefix}_landscape.png
    x_mm, y_mm : ndarray
        Coordinate grids (1D arrays, mm)
    U : ndarray
        Gorkov potential (for diagnostics, not rendered)
    Fx, Fy : ndarray
        Force components
    particle_xy_mm : tuple
        Particle position
    traj_xy_mm : list of tuple, optional
        Trajectory history
    pucks : list of tuple, optional
        Puck positions
    
    Returns
    -------
    list[Path]
        List of created PNG file paths
    """
    prefix = Path(out_png_prefix)
    
    quiver_path = prefix.parent / f"{prefix.name}_quiver.png"
    landscape_path = prefix.parent / f"{prefix.name}_landscape.png"
    
    # Ensure output directory exists
    quiver_path.parent.mkdir(parents=True, exist_ok=True)
    
    render_force_quiver(
        quiver_path,
        x_mm, y_mm, Fx, Fy,
        particle_xy_mm,
        traj_xy_mm=traj_xy_mm,
        pucks=pucks,
    )
    
    # Render force magnitude landscape instead of potential contour
    # (forces have larger values and render better)
    render_force_magnitude_landscape(
        landscape_path,
        x_mm, y_mm, Fx, Fy,
        particle_xy_mm,
        traj_xy_mm=traj_xy_mm,
        pucks=pucks,
    )
    
    return [quiver_path, landscape_path]
