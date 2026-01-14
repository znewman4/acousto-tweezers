#!/usr/bin/env python3
"""
Path following with structured ParticleController.

Demonstrates the new control layer with:
- Jacobian-based control effectiveness estimation
- One-step and MPC control modes
- Trap-aware safety constraints
- Full diagnostics and logging
- Enhanced visualization with control metrics overlay
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from typing import Optional
import shutil

from acousto.force import ParticleProps, bilinear_sample_vec
from acousto.analysis import find_traps_from_force

from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
    # New controller classes
    ControlState, ControlVector, ControlBounds, ControlRateLimits,
    ControllerConfig, ParticleController, SafetyConfig,
    create_visualization_data,
)

from tweezers.viz.render_3d import (
    Cylinder2D, classify_trap, normalize_gorkov_field,
)

# =============================================================================
# Configuration flags
# =============================================================================
PATH_SCALE = 0.2  # Multiply all waypoint coordinates by this around centroid
T_STEPS = 400     # Number of time steps for simulation (increased for smoother GIF)
RENDER_STRIDE = 1 # Render every Nth step (1 = all steps, 2 = every other)

# Part A: Local controllability probe
RUN_CONTROL_PROBE = True  # Run sensitivity probe at initial and mid-run states

# Part B: Guided transducer geometry mode
GUIDED_GEOMETRY = True    # Move transducers to straddle target instead of random
TRANSDUCER_SEPARATION = 1.0e-3  # Target separation between pucks (meters)
TRANSDUCER_Y_FIXED = 0.15e-3    # Fixed y-position for both transducers (meters)


# =============================================================================
# Part A: Local Controllability / Sensitivity Probe
# =============================================================================
def probe_local_sensitivity(
    ev,  # BottomFootprint25DEvaluator
    x: float,  # Particle x position (meters)
    y: float,  # Particle y position (meters)
    u0: Control2Pucks,  # Baseline control
    out_dir: Path,  # Output directory for results
    probe_name: str = "probe",  # Name for output files
) -> dict:
    """
    Compute local sensitivity of force and displacement to control perturbations.
    
    Uses finite differences to estimate Jacobians:
    - dF/du: 2x8 matrix (force sensitivity)
    - dDisp/du: 2x8 matrix (displacement sensitivity per step)
    - dTrap/du: 2x8 matrix (trap centre sensitivity, if computable)
    
    Returns dict with:
    - jacobian_force: (2, 8) array
    - jacobian_disp: (2, 8) array
    - dim_labels: list of control dimension names
    - top_dims_force: top-3 control dimensions by force sensitivity norm
    - top_dims_disp: top-3 control dimensions by displacement sensitivity norm
    - svd_force: singular values of force Jacobian
    - svd_disp: singular values of displacement Jacobian
    """
    import csv
    
    # Control dimension labels and epsilons
    dim_labels = ["xA", "yA", "xB", "yB", "vA", "vB", "phiA", "phiB"]
    
    # Finite difference epsilons (tuned per dimension type)
    eps_pos = 5e-6    # 5 µm for positions
    eps_amp = 1e-5    # Small for amplitudes
    eps_phi = 0.05    # ~3 degrees for phases
    
    epsilons = np.array([
        eps_pos, eps_pos,   # xA, yA
        eps_pos, eps_pos,   # xB, yB
        eps_amp, eps_amp,   # vA, vB
        eps_phi, eps_phi,   # phiA, phiB
    ])
    
    # Baseline evaluation
    xp1_base, yp1_base, _, info_base = ev.step(
        xp=x, yp=y, target_x=x, target_y=y,
        u=u0, u_prev=None, return_fields=False,
    )
    fx_base, fy_base = info_base["fx"], info_base["fy"]
    dx_base = xp1_base - x
    dy_base = yp1_base - y
    
    # Storage for Jacobians
    n_dims = 8
    jacobian_force = np.zeros((2, n_dims))
    jacobian_disp = np.zeros((2, n_dims))
    
    # CSV rows for detailed output
    csv_rows = []
    
    # Convert baseline control to array
    u0_arr = np.array([u0.xA, u0.yA, u0.xB, u0.yB, u0.vA, u0.vB, u0.phiA, u0.phiB])
    
    print(f"\n  [PROBE] Baseline: pos=({x*1e3:.4f}, {y*1e3:.4f}) mm")
    print(f"  [PROBE] Baseline force: fx={fx_base:.3e}, fy={fy_base:.3e} N")
    print(f"  [PROBE] Baseline disp:  dx={dx_base*1e6:.2f}, dy={dy_base*1e6:.2f} µm")
    print(f"  [PROBE] Computing finite differences...")
    
    for i in range(n_dims):
        eps = epsilons[i]
        
        # u+ perturbation
        u_plus_arr = u0_arr.copy()
        u_plus_arr[i] += eps
        u_plus = Control2Pucks(
            xA=u_plus_arr[0], yA=u_plus_arr[1],
            xB=u_plus_arr[2], yB=u_plus_arr[3],
            vA=u_plus_arr[4], vB=u_plus_arr[5],
            phiA=u_plus_arr[6], phiB=u_plus_arr[7],
        )
        
        # u- perturbation
        u_minus_arr = u0_arr.copy()
        u_minus_arr[i] -= eps
        u_minus = Control2Pucks(
            xA=u_minus_arr[0], yA=u_minus_arr[1],
            xB=u_minus_arr[2], yB=u_minus_arr[3],
            vA=u_minus_arr[4], vB=u_minus_arr[5],
            phiA=u_minus_arr[6], phiB=u_minus_arr[7],
        )
        
        # Evaluate at u+
        xp1_plus, yp1_plus, _, info_plus = ev.step(
            xp=x, yp=y, target_x=x, target_y=y,
            u=u_plus, u_prev=None, return_fields=False,
        )
        fx_plus, fy_plus = info_plus["fx"], info_plus["fy"]
        dx_plus = xp1_plus - x
        dy_plus = yp1_plus - y
        
        # Evaluate at u-
        xp1_minus, yp1_minus, _, info_minus = ev.step(
            xp=x, yp=y, target_x=x, target_y=y,
            u=u_minus, u_prev=None, return_fields=False,
        )
        fx_minus, fy_minus = info_minus["fx"], info_minus["fy"]
        dx_minus = xp1_minus - x
        dy_minus = yp1_minus - y
        
        # Central difference Jacobian columns
        dFx_du = (fx_plus - fx_minus) / (2 * eps)
        dFy_du = (fy_plus - fy_minus) / (2 * eps)
        dDx_du = (dx_plus - dx_minus) / (2 * eps)
        dDy_du = (dy_plus - dy_minus) / (2 * eps)
        
        jacobian_force[0, i] = dFx_du
        jacobian_force[1, i] = dFy_du
        jacobian_disp[0, i] = dDx_du
        jacobian_disp[1, i] = dDy_du
        
        # Sensitivity norms for this dimension
        norm_dF = np.sqrt(dFx_du**2 + dFy_du**2)
        norm_dDisp = np.sqrt(dDx_du**2 + dDy_du**2)
        
        csv_rows.append({
            "dim": dim_labels[i],
            "eps": eps,
            "dFx_du": dFx_du,
            "dFy_du": dFy_du,
            "dDx_du": dDx_du,
            "dDy_du": dDy_du,
            "norm_dF": norm_dF,
            "norm_dDisp": norm_dDisp,
        })
    
    # SVD analysis for rank/conditioning
    _, s_force, _ = np.linalg.svd(jacobian_force)
    _, s_disp, _ = np.linalg.svd(jacobian_disp)
    
    # Find top-3 dimensions by sensitivity norm
    force_norms = np.sqrt(jacobian_force[0]**2 + jacobian_force[1]**2)
    disp_norms = np.sqrt(jacobian_disp[0]**2 + jacobian_disp[1]**2)
    
    top_force_idx = np.argsort(force_norms)[::-1][:3]
    top_disp_idx = np.argsort(disp_norms)[::-1][:3]
    
    top_dims_force = [(dim_labels[i], force_norms[i]) for i in top_force_idx]
    top_dims_disp = [(dim_labels[i], disp_norms[i]) for i in top_disp_idx]
    
    # ===== Save outputs =====
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # NPY files
    np.save(out_dir / f"{probe_name}_jacobian_force.npy", jacobian_force)
    np.save(out_dir / f"{probe_name}_jacobian_disp.npy", jacobian_disp)
    
    # CSV file
    csv_path = out_dir / f"{probe_name}_sensitivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # ===== Create visualization plots =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: Force sensitivity norms
    ax1 = axes[0]
    ax1.bar(dim_labels, force_norms, color='steelblue', edgecolor='black')
    ax1.set_xlabel("Control Dimension")
    ax1.set_ylabel("||dF/du|| (N per unit)")
    ax1.set_title(f"Force Sensitivity by Control Dimension\nSVD: σ₁={s_force[0]:.3e}, σ₂={s_force[1]:.3e}")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(force_norms):
        if v > 0:
            ax1.text(i, v * 1.1, f"{v:.1e}", ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Bar chart: Displacement sensitivity norms
    ax2 = axes[1]
    ax2.bar(dim_labels, disp_norms * 1e6, color='coral', edgecolor='black')  # Convert to µm
    ax2.set_xlabel("Control Dimension")
    ax2.set_ylabel("||dΔp/du|| (µm per unit)")
    ax2.set_title(f"Displacement Sensitivity by Control Dimension\nSVD: σ₁={s_disp[0]:.3e}, σ₂={s_disp[1]:.3e}")
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(disp_norms * 1e6):
        if v > 0:
            ax2.text(i, v * 1.1, f"{v:.1f}", ha='center', va='bottom', fontsize=8, rotation=45)
    
    fig.suptitle(f"Local Controllability Probe at ({x*1e3:.3f}, {y*1e3:.3f}) mm", fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / f"{probe_name}_sensitivity.png", dpi=150)
    plt.close(fig)
    
    # ===== Print summary =====
    print(f"\n  [PROBE] Results saved to: {out_dir}")
    print(f"  [PROBE] Force Jacobian SVD: σ₁={s_force[0]:.3e}, σ₂={s_force[1]:.3e}")
    print(f"  [PROBE] Disp Jacobian SVD:  σ₁={s_disp[0]:.3e}, σ₂={s_disp[1]:.3e}")
    print(f"  [PROBE] Top-3 force dims:   {', '.join([f'{d}({n:.2e})' for d,n in top_dims_force])}")
    print(f"  [PROBE] Top-3 disp dims:    {', '.join([f'{d}({n:.2e})' for d,n in top_dims_disp])}")
    
    # Check for controllability issues
    if s_disp[1] < 1e-12:
        print(f"  [PROBE] ⚠️  WARNING: Displacement Jacobian is rank-deficient (σ₂ ≈ 0)")
    if max(disp_norms) < 1e-9:
        print(f"  [PROBE] ⚠️  WARNING: All displacement sensitivities are near-zero!")
    
    return {
        "jacobian_force": jacobian_force,
        "jacobian_disp": jacobian_disp,
        "dim_labels": dim_labels,
        "top_dims_force": top_dims_force,
        "top_dims_disp": top_dims_disp,
        "svd_force": s_force,
        "svd_disp": s_disp,
        "force_norms": force_norms,
        "disp_norms": disp_norms,
    }


# =============================================================================
# Part B: Guided Transducer Geometry
# =============================================================================
def compute_guided_control(
    target_x: float,  # Target waypoint x (meters)
    target_y: float,  # Target waypoint y (meters)
    current_control: "ControlVector",  # Current control (for phases/amplitudes)
    domain_Lx: float,  # Domain width (meters)
    separation: float = TRANSDUCER_SEPARATION,
    y_fixed: float = TRANSDUCER_Y_FIXED,
    margin: float = 0.1e-3,  # Margin from domain edges
) -> tuple[float, float, float, float]:
    """
    Compute transducer positions to straddle the target.
    
    Returns (xA, yA, xB, yB) in meters.
    
    The transducers are positioned symmetrically around the target x,
    with fixed y at the bottom of the domain.
    """
    half_sep = separation / 2.0
    
    # Position transducers to straddle target
    xA_ideal = target_x - half_sep
    xB_ideal = target_x + half_sep
    
    # Clamp to domain with margin
    xA = float(np.clip(xA_ideal, margin, domain_Lx - margin))
    xB = float(np.clip(xB_ideal, margin, domain_Lx - margin))
    
    # Ensure minimum separation even after clamping
    min_sep = 0.2e-3  # 0.2 mm minimum
    if xB - xA < min_sep:
        center = (xA + xB) / 2.0
        xA = center - min_sep / 2.0
        xB = center + min_sep / 2.0
        xA = float(np.clip(xA, margin, domain_Lx - margin))
        xB = float(np.clip(xB, margin, domain_Lx - margin))
    
    return xA, y_fixed, xB, y_fixed


# =============================================================================
# Authority Scaling Sweep
# =============================================================================
def run_authority_sweep(domain, medium, particle, out_dir: Path) -> dict:
    """
    Sweep over alpha_g, transducer amplitude, and dt to find configurations
    with adequate control authority (Δp >= 0.05 mm/step).
    
    Returns dict with best configuration found.
    """
    print("\n" + "=" * 70)
    print("AUTHORITY SCALING SWEEP")
    print("=" * 70)
    
    alpha_g_values = [1e6, 1e7, 1e8]
    v_values = [5e-4, 1e-3, 2e-3]
    dt_values = [5e-3, 2e-2]
    
    test_x, test_y = 1.0e-3, 1.0e-3
    results = []
    
    print(f"Test position: ({test_x*1e3:.1f}, {test_y*1e3:.1f}) mm")
    print("-" * 70)
    print(f"{'alpha_g':>10s}  {'vA=vB':>10s}  {'dt':>10s}  {'|F| (N)':>12s}  {'Δp (mm)':>10s}  {'pmax (Pa)':>12s}")
    print("-" * 70)
    
    for alpha_g in alpha_g_values:
        for v_amp in v_values:
            for dt in dt_values:
                # Create evaluator with these parameters
                cfg = EvaluatorConfig(
                    sigma_x=0.10e-3,
                    bottom_band=0.25e-3,
                    dt=dt,
                    viscosity=1e-3,
                    border_penalty=1e6,
                    smooth_u=0.0,
                    alpha_g=alpha_g,
                    max_step=0.05e-3,  # 50 µm max step to prevent instability
                )
                ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
                
                # Test control: out-of-phase transducers
                u_test = Control2Pucks(
                    xA=0.5e-3, yA=0.15e-3, xB=1.5e-3, yB=0.15e-3,
                    vA=v_amp, vB=v_amp, phiA=0.0, phiB=np.pi,
                )
                
                xp1, yp1, _, info = ev.step(
                    xp=test_x, yp=test_y,
                    target_x=test_x, target_y=test_y,
                    u=u_test, u_prev=None, return_fields=False,
                )
                fx, fy = info["fx"], info["fy"]
                F_mag = np.sqrt(fx**2 + fy**2)
                dp_mm = np.sqrt((xp1-test_x)**2 + (yp1-test_y)**2) * 1e3
                pmax = info["pmax"]
                
                results.append({
                    "alpha_g": alpha_g, "v_amp": v_amp, "dt": dt,
                    "F_mag": F_mag, "dp_mm": dp_mm, "pmax": pmax,
                })
                
                # Mark if adequate
                marker = "✓" if dp_mm >= 0.05 else " "
                print(f"{alpha_g:>10.0e}  {v_amp:>10.0e}  {dt:>10.0e}  {F_mag:>12.3e}  {dp_mm:>10.4f}  {pmax:>12.2e} {marker}")
    
    # Save to CSV
    import csv
    csv_path = out_dir / "authority_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to: {csv_path}")
    
    # Find configurations with adequate authority
    adequate = [r for r in results if r["dp_mm"] >= 0.05]
    print(f"\n{len(adequate)}/{len(results)} configurations have Δp >= 0.05 mm")
    
    if adequate:
        # Return best one (highest Δp)
        best = max(adequate, key=lambda r: r["dp_mm"])
        print(f"Best: alpha_g={best['alpha_g']:.0e}, v={best['v_amp']:.0e}, dt={best['dt']:.0e}")
        print(f"      Δp={best['dp_mm']:.4f} mm, |F|={best['F_mag']:.3e} N")
        return best
    else:
        print("⚠️  No configuration achieves Δp >= 0.05 mm - using max displacement config")
        best = max(results, key=lambda r: r["dp_mm"])
        return best


# =============================================================================
# Trap Centre Finding
# =============================================================================
def find_local_trap_centre(
    x_grid: np.ndarray,  # 1D array (m)
    y_grid: np.ndarray,  # 1D array (m)
    U: np.ndarray,       # 2D Gorkov potential
    particle_x: float,   # particle position (m)
    particle_y: float,   # particle position (m)
    window_mm: float = 0.2,  # search window ± (mm)
) -> tuple[float, float]:
    """
    Find the local minimum of Gorkov potential near the particle.
    
    Returns (trap_x, trap_y) in meters.
    """
    window_m = window_mm * 1e-3
    
    # Find indices within window
    x_mask = (x_grid >= particle_x - window_m) & (x_grid <= particle_x + window_m)
    y_mask = (y_grid >= particle_y - window_m) & (y_grid <= particle_y + window_m)
    
    if not np.any(x_mask) or not np.any(y_mask):
        # Fallback: use particle position
        return particle_x, particle_y
    
    x_idx = np.where(x_mask)[0]
    y_idx = np.where(y_mask)[0]
    
    # Extract local region
    U_local = U[np.ix_(y_idx, x_idx)]
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(U_local), U_local.shape)
    trap_y = y_grid[y_idx[min_idx[0]]]
    trap_x = x_grid[x_idx[min_idx[1]]]
    
    return float(trap_x), float(trap_y)


def render_with_control_overlay(
    *,
    out_png: Path,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    U: np.ndarray,
    traps,
    particle_xy_mm: tuple[float, float],
    target_xy_mm: tuple[float, float],
    track_xy_mm: list[tuple[float, float]] | None = None,
    predicted_xy_mm: list[tuple[float, float]] | None = None,
    cylinders=None,
    control_metrics: dict | None = None,
    force_arrow_mm: tuple[float, float] | None = None,  # A2: Force direction (fx_mm, fy_mm)
    trap_centre_mm: tuple[float, float] | None = None,  # Task 4: Trap centre position
    metrics_history: dict | None = None,  # Part C: Time series data for plotting
    surface_stride: int = 3,
) -> bool:
    """
    Enhanced multi-view rendering with control metrics overlay.
    
    Panels:
      - (Top-left) 3D Gorkov landscape with particle on surface
      - (Top-right) 2D trajectory with predicted motion and target
      - (Bottom-left) Control metrics (stiffness, tracking error)
      - (Bottom-right) Tracking metrics time series (Part C: replaces x/y displacement)
    
    metrics_history: dict with keys like:
      - 'err_mm': list of tracking errors
      - 'trap_to_target_mm': list of trap-to-target distances
      - 'particle_to_trap_mm': list of particle-to-trap distances
      - 'F_mag_N': list of force magnitudes
      - 'cos_to_target': list of directional alignment values
    """
    px_mm, py_mm = particle_xy_mm
    tx_mm, ty_mm = target_xy_mm
    track_len = len(track_xy_mm) if track_xy_mm else 0
    
    U_min = float(np.min(U))
    U_max = float(np.max(U))
    den_orig = U_max - U_min
    
    # Prepare data
    X, Y = np.meshgrid(x_mm, y_mm)
    U_display = U * 1e15  # Scale for visualization
    Uvis, is_flat = normalize_gorkov_field(U_display, verbose=False)
    
    z0 = -0.25
    Xs = X[::surface_stride, ::surface_stride]
    Ys = Y[::surface_stride, ::surface_stride]
    Us = Uvis[::surface_stride, ::surface_stride]
    
    # Create 4-panel figure
    fig = plt.figure(figsize=(16, 12))
    
    # ============ Panel 1: 3D Landscape (top-left) ============
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot_surface(Xs, Ys, Us, linewidth=0, antialiased=True, alpha=0.95, cmap="viridis")
    ax1.contour(X, Y, Uvis, levels=12, offset=z0, alpha=0.4)
    
    # Draw cylinders (transducers)
    if cylinders:
        from tweezers.viz.render_3d import _draw_cylinder_surface
        for cyl in cylinders:
            _draw_cylinder_surface(ax1, cyl=cyl)
    
    # Draw traps
    for t in traps:
        ttype = classify_trap(np.asarray(t.eigvals))
        mx, my = (float(t.x) * 1e3), (float(t.y) * 1e3)
        mz = 0.0 if den_orig == 0.0 else (float(t.U) - U_min) / den_orig
        
        marker_style = {"min": ("o", "green"), "saddle": ("x", "blue"), "max": ("^", "red")}
        m, c = marker_style.get(ttype, ("^", "red"))
        ax1.scatter(mx, my, mz, s=50, marker=m, color=c, alpha=0.7)
    
    # Draw particle on surface
    ix = np.argmin(np.abs(x_mm - px_mm))
    iy = np.argmin(np.abs(y_mm - py_mm))
    pz_on_surface = Uvis[iy, ix] if (0 <= iy < Uvis.shape[0] and 0 <= ix < Uvis.shape[1]) else 0.5
    pz_particle = min(1.15, pz_on_surface + 0.08)
    
    ax1.scatter(px_mm, py_mm, pz_particle, s=1500, marker="o", color="red", alpha=1.0,
                edgecolors="white", linewidth=6, zorder=1000)
    
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_zlabel("U (norm)")
    ax1.set_zlim(z0, 1.05)
    ax1.set_box_aspect((np.ptp(x_mm), np.ptp(y_mm), 0.8))
    ax1.view_init(elev=30, azim=-60)
    ax1.set_title("3D: Gorkov Landscape + Particle", fontsize=10, fontweight="bold")
    
    # ============ Panel 2: 2D Top-Down with Predictions (top-right) ============
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Contour of U in x-y plane
    ax2.contourf(X, Y, Uvis, levels=20, cmap="viridis", alpha=0.7)
    ax2.contour(X, Y, Uvis, levels=12, colors="k", linewidths=0.3, alpha=0.3)
    
    # Draw trajectory history with gradient
    if track_xy_mm and len(track_xy_mm) >= 2:
        tx_hist = np.array([p[0] for p in track_xy_mm])
        ty_hist = np.array([p[1] for p in track_xy_mm])
        
        n_pts = len(tx_hist)
        for i in range(n_pts - 1):
            alpha_color = i / max(n_pts - 1, 1)
            color = (1-alpha_color) * np.array([0.5, 0.5, 0.5]) + alpha_color * np.array([0, 1, 1])
            ax2.plot(tx_hist[i:i+2], ty_hist[i:i+2], linewidth=3.0, color=color, alpha=0.9)
    
    # Draw predicted trajectory (dashed magenta)
    if predicted_xy_mm and len(predicted_xy_mm) >= 1:
        pred_x = [px_mm] + [p[0] for p in predicted_xy_mm]
        pred_y = [py_mm] + [p[1] for p in predicted_xy_mm]
        ax2.plot(pred_x, pred_y, 'mo--', linewidth=2.5, markersize=8, alpha=0.8, label="predicted")
    
    # Task 4: Draw trap centre (cyan X)
    if trap_centre_mm is not None:
        tx_trap, ty_trap = trap_centre_mm
        ax2.scatter(tx_trap, ty_trap, s=150, marker="x", color="cyan", 
                    linewidth=3, label="trap centre", zorder=18)
    
    # Draw target (yellow star)
    ax2.scatter(tx_mm, ty_mm, s=300, marker="*", color="yellow", edgecolors="black", 
                linewidth=2, label="target", zorder=20)
    
    # Draw current particle position
    ax2.scatter(px_mm, py_mm, s=200, marker="o", color="red", edgecolors="white", 
                linewidth=2, label="particle", zorder=15)
    
    # A2: Draw force arrow at particle (if provided)
    if force_arrow_mm is not None:
        fx_mm, fy_mm = force_arrow_mm
        # Scale arrow to be visible (0.3 mm arrow for typical force)
        arrow_scale = 0.3 / max(np.sqrt(fx_mm**2 + fy_mm**2), 1e-9)
        ax2.quiver(px_mm, py_mm, fx_mm * arrow_scale, fy_mm * arrow_scale,
                   angles='xy', scale_units='xy', scale=1,
                   color='lime', width=0.02, headwidth=4, headlength=5,
                   label=f"force", zorder=25)
    
    # Draw transducers
    if cylinders:
        for i, cyl in enumerate(cylinders):
            circle = plt.Circle((cyl.x_mm, cyl.y_mm), cyl.r_mm, fill=False, 
                               edgecolor="yellow", linewidth=2, linestyle="--")
            ax2.add_patch(circle)
            ax2.annotate(f"T{i+1}", (cyl.x_mm, cyl.y_mm), color="yellow", 
                        fontsize=8, ha="center", va="center")
        
        # Part B3: Draw line between transducers to show "straddling"
        if len(cylinders) >= 2:
            ax2.plot([cylinders[0].x_mm, cylinders[1].x_mm],
                    [cylinders[0].y_mm, cylinders[1].y_mm],
                    color='yellow', linewidth=1.5, alpha=0.6, linestyle=':')
    
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_title("2D: Trajectory + Prediction + Target", fontsize=10, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    
    # ============ Panel 3: Control Metrics (bottom-left) ============
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis("off")
    
    if control_metrics:
        metrics_text = "Control Diagnostics\n" + "=" * 30 + "\n\n"
        
        # Force and displacement
        metrics_text += f"|F|: {control_metrics.get('F_mag_N', 0):.3e} N\n"
        metrics_text += f"Δp: {control_metrics.get('dp_mm', 0):.4f} mm\n"
        metrics_text += f"cos_to_target: {control_metrics.get('cos_to_target', 0):.3f}\n\n"
        
        # Trap centre metrics (Task 4)
        metrics_text += f"Trap Centre:\n"
        metrics_text += f"  ({control_metrics.get('trap_x_mm', 0):.3f}, {control_metrics.get('trap_y_mm', 0):.3f}) mm\n"
        metrics_text += f"  p→trap: {control_metrics.get('particle_to_trap_mm', 0):.4f} mm\n"
        metrics_text += f"  trap→tgt: {control_metrics.get('trap_to_target_mm', 0):.4f} mm\n\n"
        
        # Tracking
        metrics_text += f"Tracking Error: {control_metrics.get('tracking_error_mm', 0):.4f} mm\n\n"
        
        # Stiffness
        stiffness = control_metrics.get('stiffness', [0, 0])
        if stiffness is not None and len(stiffness) >= 2:
            metrics_text += f"Trap Stiffness (eigenvalues):\n"
            metrics_text += f"  λ₁ = {stiffness[0]:.3e}\n"
            metrics_text += f"  λ₂ = {stiffness[1]:.3e}\n"
            is_stable = all(s < 0 for s in stiffness) if stiffness else False
            metrics_text += f"  Stable: {'✓' if is_stable else '✗'}\n\n"
        
        # Control
        metrics_text += f"Control Magnitude:\n"
        metrics_text += f"  vA = {control_metrics.get('vA', 0)*1e4:.2f} ×10⁻⁴ m/s\n"
        metrics_text += f"  vB = {control_metrics.get('vB', 0)*1e4:.2f} ×10⁻⁴ m/s\n"
        metrics_text += f"  φA = {control_metrics.get('phiA', 0):.3f} rad\n"
        metrics_text += f"  φB = {control_metrics.get('phiB', 0):.3f} rad\n\n"
        
        # A3: Control deltas
        metrics_text += f"Control Δ (this step):\n"
        metrics_text += f"  ΔxA={control_metrics.get('dxA_um', 0):.1f}μm  ΔxB={control_metrics.get('dxB_um', 0):.1f}μm\n"
        metrics_text += f"  ΔvA={control_metrics.get('dvA_um_s', 0):.1f}  ΔvB={control_metrics.get('dvB_um_s', 0):.1f}\n"
        metrics_text += f"  ΔφA={control_metrics.get('dphiA', 0):.3f}  ΔφB={control_metrics.get('dphiB', 0):.3f}\n\n"
        
        # Safety
        metrics_text += f"Safety:\n"
        metrics_text += f"  Rejected: {control_metrics.get('n_rejected', 0)}/{control_metrics.get('n_candidates', 0)}\n"
        
        ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax3.set_title("Control Metrics", fontsize=10, fontweight="bold")
    
    # ============ Panel 4: Tracking Metrics Time Series (Part C - replaces x/y displacement) ============
    ax4 = fig.add_subplot(2, 2, 4)
    
    if metrics_history and len(metrics_history.get('err_mm', [])) >= 2:
        # Part C: Plot tracking metrics instead of x/y position
        t_steps = np.arange(len(metrics_history['err_mm']))
        
        # Primary axis: Distance metrics (mm)
        err_mm = np.array(metrics_history['err_mm'])
        trap_to_target = np.array(metrics_history.get('trap_to_target_mm', [0] * len(t_steps)))
        particle_to_trap = np.array(metrics_history.get('particle_to_trap_mm', [0] * len(t_steps)))
        
        ax4.plot(t_steps, err_mm, linewidth=2.0, color='red', label='err (p→target)', alpha=0.9)
        ax4.plot(t_steps, trap_to_target, linewidth=2.0, color='blue', linestyle='--', 
                label='trap→target', alpha=0.8)
        ax4.plot(t_steps, particle_to_trap, linewidth=2.0, color='green', linestyle=':', 
                label='p→trap', alpha=0.8)
        
        ax4.set_xlabel("Time Step", fontsize=10)
        ax4.set_ylabel("Distance (mm)", fontsize=10, color='black')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper left", fontsize=8)
        
        # Secondary axis: cos_to_target (alignment indicator)
        ax4b = ax4.twinx()
        cos_to_target = np.array(metrics_history.get('cos_to_target', [0] * len(t_steps)))
        ax4b.plot(t_steps, cos_to_target, linewidth=1.5, color='purple', linestyle='-.',
                 label='cos θ', alpha=0.7)
        ax4b.set_ylabel("cos(θ) alignment", fontsize=9, color='purple')
        ax4b.tick_params(axis='y', labelcolor='purple')
        ax4b.axhline(y=0, color='purple', linestyle=':', alpha=0.3)
        ax4b.set_ylim(-1.1, 1.1)
        
        # Mark current values
        if len(t_steps) > 0:
            ax4.scatter(t_steps[-1], err_mm[-1], s=80, marker='o', color='red', 
                       edgecolors='white', linewidth=2, zorder=10)
        
        ax4.set_title("Tracking Quality (Part C)", fontsize=10, fontweight="bold")
    elif track_xy_mm and len(track_xy_mm) >= 2:
        # Fallback: show position if no metrics history
        tx_hist = np.array([p[0] for p in track_xy_mm])
        ty_hist = np.array([p[1] for p in track_xy_mm])
        t_steps = np.arange(len(tx_hist))
        
        ax4.plot(t_steps, tx_hist, linewidth=2.5, marker="o", markersize=3, 
                label="x position (mm)", color="blue", alpha=0.8)
        ax4.plot(t_steps, ty_hist, linewidth=2.5, marker="s", markersize=3, 
                label="y position (mm)", color="red", alpha=0.8)
        
        ax4.set_xlabel("Time Step", fontsize=10)
        ax4.set_ylabel("Position (mm)", fontsize=10)
        ax4.grid(True, alpha=0.4)
        ax4.legend(loc="upper left", fontsize=9)
        ax4.set_title("Particle Position Over Time", fontsize=10, fontweight="bold")
    
    # Overall title
    tracking_err = control_metrics.get('tracking_error_mm', 0) if control_metrics else 0
    fig.suptitle(
        f"Controlled Path Following | Particle: ({px_mm:.3f}, {py_mm:.3f}) mm | "
        f"Target: ({tx_mm:.3f}, {ty_mm:.3f}) mm | Error: {tracking_err:.4f} mm",
        fontsize=11, fontweight="bold"
    )
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    
    return is_flat


def make_polyline_path(points: list[tuple[float, float]], T: int) -> np.ndarray:
    """Piecewise-linear path sampled at T points."""
    pts = np.array(points, dtype=float)
    seg = pts[1:] - pts[:-1]
    seglen = np.sqrt(np.sum(seg**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1]) if s[-1] > 0 else 1.0

    tvals = np.linspace(0.0, total, T)
    out = np.zeros((T, 2), dtype=float)

    j = 0
    for i, tv in enumerate(tvals):
        while j < len(seglen) - 1 and tv > s[j + 1]:
            j += 1
        if seglen[j] < 1e-12:
            out[i] = pts[j]
        else:
            a = (tv - s[j]) / seglen[j]
            out[i] = pts[j] + a * seg[j]
    return out


def main() -> None:
    REPO = Path(__file__).resolve().parents[1]
    RESULTS = REPO / "results"
    frames_dir = RESULTS / "frames_path_follow_controlled"
    out_dir = RESULTS / "path_follow_controlled"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PATH FOLLOWING WITH STRUCTURED CONTROLLER")
    print("=" * 70)

    # ===== Domain + Physics Setup =====
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=160, Ny=160)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)

    # Run authority sweep first to find good parameters
    best_cfg = run_authority_sweep(domain, medium, particle, out_dir)
    
    # Use best configuration found (or fall back to defaults)
    # Cap alpha_g to prevent constant saturation - we want mostly unsaturated motion
    # with occasional saturation, not always hitting the limiter
    recommended_alpha_g = min(best_cfg.get("alpha_g", 1e6), 1e6)  # Cap at 1e6
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        bottom_band=0.25e-3,
        dt=best_cfg.get("dt", 5e-3),
        viscosity=1e-3,
        border_penalty=1e6,
        smooth_u=0.0,
        alpha_g=recommended_alpha_g,
        max_step=0.05e-3,  # 50 µm max step per iteration to prevent instability
    )
    
    # Use best amplitude for controller
    best_v_amp = best_cfg.get("v_amp", 5e-4)

    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    print(f"\nUsing configuration from sweep:")
    print(f"  alpha_g = {cfg.alpha_g:.0e}")
    print(f"  dt = {cfg.dt:.0e}")
    print(f"  v_amp = {best_v_amp:.0e}")

    # ===== B3: Authority Test - Does the model have control authority? =====
    print("\n" + "=" * 70)
    print("AUTHORITY PROBE TEST")
    print("=" * 70)
    print("Testing whether the physics model produces meaningful forces...")
    
    # Test position: center of domain
    test_x, test_y = 1.0e-3, 1.0e-3
    
    # Define probe controls (intentionally extreme to see if ANYTHING happens)
    probe_controls = [
        ("Baseline: both on, in-phase", Control2Pucks(
            xA=0.5e-3, yA=0.15e-3, xB=1.5e-3, yB=0.15e-3,
            vA=5e-4, vB=5e-4, phiA=0.0, phiB=0.0)),
        ("Out-of-phase (π)", Control2Pucks(
            xA=0.5e-3, yA=0.15e-3, xB=1.5e-3, yB=0.15e-3,
            vA=5e-4, vB=5e-4, phiA=0.0, phiB=np.pi)),
        ("High amplitude", Control2Pucks(
            xA=0.5e-3, yA=0.15e-3, xB=1.5e-3, yB=0.15e-3,
            vA=1e-3, vB=1e-3, phiA=0.0, phiB=np.pi)),
        ("Transducers near edges", Control2Pucks(
            xA=0.1e-3, yA=0.15e-3, xB=1.9e-3, yB=0.15e-3,
            vA=5e-4, vB=5e-4, phiA=0.0, phiB=np.pi)),
        ("Single transducer only", Control2Pucks(
            xA=1.0e-3, yA=0.15e-3, xB=1.0e-3, yB=0.15e-3,
            vA=1e-3, vB=0.0, phiA=0.0, phiB=0.0)),
    ]
    
    authority_results = []
    for name, u_probe in probe_controls:
        xp1, yp1, loss, info = ev.step(
            xp=test_x, yp=test_y,
            target_x=test_x, target_y=test_y,
            u=u_probe, u_prev=None, return_fields=False,
        )
        fx, fy = info["fx"], info["fy"]
        F_mag = np.sqrt(fx**2 + fy**2)
        dx_m = xp1 - test_x
        dy_m = yp1 - test_y
        dp_mm = np.sqrt(dx_m**2 + dy_m**2) * 1e3
        
        result_line = f"  {name:30s} |F|={F_mag:.3e} N  Δp={dp_mm:.4f} mm  pmax={info['pmax']:.2e} Pa"
        print(result_line)
        authority_results.append({
            "name": name, "fx": fx, "fy": fy, "F_mag": F_mag,
            "dx_m": dx_m, "dy_m": dy_m, "dp_mm": dp_mm, "pmax": info["pmax"],
        })
    
    # Save authority probe results
    authority_file = out_dir / "authority_probe.txt"
    with open(authority_file, "w") as f:
        f.write("AUTHORITY PROBE TEST RESULTS\n")
        f.write(f"Test position: ({test_x*1e3:.2f}, {test_y*1e3:.2f}) mm\n")
        f.write(f"alpha_g = {cfg.alpha_g:.0e}\n")
        f.write("=" * 70 + "\n\n")
        for r in authority_results:
            f.write(f"{r['name']:30s}\n")
            f.write(f"  Force: fx={r['fx']:.3e}, fy={r['fy']:.3e}, |F|={r['F_mag']:.3e} N\n")
            f.write(f"  Step:  dx={r['dx_m']*1e6:.2f} μm, dy={r['dy_m']*1e6:.2f} μm, |Δp|={r['dp_mm']:.4f} mm\n")
            f.write(f"  pmax:  {r['pmax']:.3e} Pa\n\n")
        
        # Summary
        max_F = max(r['F_mag'] for r in authority_results)
        max_dp = max(r['dp_mm'] for r in authority_results)
        f.write("=" * 70 + "\n")
        f.write(f"MAX |F| across probes: {max_F:.3e} N\n")
        f.write(f"MAX |Δp| across probes: {max_dp:.4f} mm\n")
        if max_dp < 0.001:
            f.write("⚠️  WARNING: Max displacement < 1 μm - control authority is VERY WEAK\n")
        elif max_dp < 0.01:
            f.write("⚠️  WARNING: Max displacement < 10 μm - control authority is WEAK\n")
        else:
            f.write("✓  Control authority appears reasonable\n")
    
    print(f"\nAuthority probe saved to: {authority_file}")
    print("-" * 70)

    # ===== Controller Setup =====
    bounds = ControlBounds(
        x_min=0.0, x_max=domain.Lx,
        y_min=0.0, y_max=cfg.bottom_band,
        v_min=0.0, v_max=1e-3,
    )
    
    rate_limits = ControlRateLimits(
        dx_max=0.08e-3,    # Smooth transducer motion
        dy_max=0.04e-3,
        dv_max=1e-4,
        dphi_max=0.4,
    )
    
    # Note: tracking cost is ~4e-8 (m² scale), so effort_weight must be tiny
    # to not dominate. Scale tracking to mm² or reduce effort_weight.
    # Using tracking in mm² space: tracking_weight=1e6 to convert m² -> mm² scale
    controller_cfg = ControllerConfig(
        tracking_weight=1e6,      # Scale from m² to mm² so cost is ~0.04 not 4e-8
        effort_weight=0.001,      # Very small - we WANT active control
        stiffness_weight=0.0001,  # Minimal - we care more about tracking
        trap_weight=2e6,          # Scaled similarly
        particle_trap_weight=0.5e6,  # Scaled similarly
        horizon=4,
        n_candidates=80,
        position_noise=0.04e-3,
        amplitude_noise=0.4e-4,
        phase_noise=0.25,
        dt=cfg.dt,
        viscosity=cfg.viscosity,
        particle_radius=particle.a,
    )
    
    safety_cfg = SafetyConfig(
        min_stiffness=-1e-8,
        min_transducer_separation=0.15e-3,
        boundary_margin=0.08e-3,
        reject_saddle_proximity=0.25e-3,
        max_control_magnitude=max(best_v_amp * 1.5, 8e-4),  # Use best amplitude
    )
    
    controller = ParticleController(
        evaluator=ev,
        config=controller_cfg,
        safety_config=safety_cfg,
        bounds=bounds,
        rate_limits=rate_limits,
    )

    print(f"\nController config:")
    print(f"  Horizon: {controller_cfg.horizon}")
    print(f"  Candidates/step: {controller_cfg.n_candidates}")
    print(f"  Safety margin: {safety_cfg.boundary_margin*1e3:.2f} mm")

    # ===== Desired Path (scaled around centroid) =====
    T = T_STEPS
    
    # Original path waypoints (in meters)
    raw_points = [
        (0.3e-3, 0.7e-3),
        (1.7e-3, 0.7e-3),
        (1.7e-3, 1.5e-3),
        (0.3e-3, 1.5e-3),
        (0.3e-3, 0.7e-3),
    ]
    
    # Compute centroid
    pts_arr = np.array(raw_points)
    centroid = pts_arr.mean(axis=0)
    
    # Scale around centroid
    scaled_points = []
    for px, py in raw_points:
        sx = centroid[0] + PATH_SCALE * (px - centroid[0])
        sy = centroid[1] + PATH_SCALE * (py - centroid[1])
        scaled_points.append((sx, sy))
    
    path = make_polyline_path(points=scaled_points, T=T)
    
    # Compute path perimeter
    path_perimeter_mm = 0.0
    for i in range(len(scaled_points) - 1):
        dx = scaled_points[i+1][0] - scaled_points[i][0]
        dy = scaled_points[i+1][1] - scaled_points[i][1]
        path_perimeter_mm += np.sqrt(dx**2 + dy**2) * 1e3
    
    print(f"\nPath configuration:")
    print(f"  PATH_SCALE: {PATH_SCALE}")
    print(f"  T (steps): {T}")
    print(f"  Centroid: ({centroid[0]*1e3:.3f}, {centroid[1]*1e3:.3f}) mm")
    print(f"  Perimeter: {path_perimeter_mm:.4f} mm")
    print(f"  Scaled waypoints (mm):")
    for i, (px, py) in enumerate(scaled_points):
        print(f"    [{i}] ({px*1e3:.3f}, {py*1e3:.3f})")

    # ===== Initial State =====
    state = ControlState(x=float(path[0, 0]), y=float(path[0, 1]))
    
    # Initial control (two transducers at bottom, use best amplitude from sweep)
    initial_control = ControlVector(
        xA=0.5e-3, yA=0.15e-3,
        xB=1.5e-3, yB=0.15e-3,
        vA=best_v_amp, vB=best_v_amp,
        phiA=0.0, phiB=np.pi,
        bounds=bounds,
        rate_limits=rate_limits,
    )
    control = initial_control
    prev_control = initial_control  # A3: Track previous control for deltas

    # ===== Part A: Run Control Probe at Initial State =====
    if RUN_CONTROL_PROBE:
        print("\n" + "=" * 70)
        print("PART A: LOCAL CONTROLLABILITY PROBE (Initial State)")
        print("=" * 70)
        probe_dir = RESULTS / "control_probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        
        # Probe at initial position with initial control
        u0_probe = initial_control.to_control2pucks()
        probe_result_init = probe_local_sensitivity(
            ev=ev,
            x=state.x,
            y=state.y,
            u0=u0_probe,
            out_dir=probe_dir,
            probe_name="init",
        )
        
        # Check if any displacement sensitivity is nonzero
        if max(probe_result_init['disp_norms']) < 1e-12:
            print("\n  ⚠️  CRITICAL: All displacement sensitivities are ~0!")
            print("     This means the control has NO EFFECT on particle motion.")

    # ===== Simulation Loop =====
    render_every = RENDER_STRIDE  # Part D: Use configurable render stride
    cyl_r_mm = (2.0 * cfg.sigma_x) * 1e3
    
    traj_xy_mm: list[tuple[float, float]] = [(state.x * 1e3, state.y * 1e3)]
    frame_paths: list[Path] = []
    prev_frame_path: Optional[Path] = None
    use_mpc = True  # Enable MPC mode
    
    # B2: Diagnostics collection arrays
    diag_steps: list[dict] = []
    
    # Task 6: Directional sanity check - track cos_to_target
    cos_to_target_history: list[float] = []
    
    # Part C: Metrics history for time series plotting
    metrics_history: dict = {
        'err_mm': [],
        'trap_to_target_mm': [],
        'particle_to_trap_mm': [],
        'F_mag_N': [],
        'cos_to_target': [],
    }
    
    print(f"\nStarting simulation (MPC={'enabled' if use_mpc else 'disabled'}, GUIDED_GEOMETRY={GUIDED_GEOMETRY})...")
    print("-" * 140)
    print("Step  px_mm    py_mm    err_mm   |F|_N      Δp_mm    trap→tgt  p→trap   cos_θ   lim  scale   raw_mm  rej/cand")
    print("-" * 140)
    
    # Part A: Mid-run probe flag
    mid_probe_done = False

    for t in range(T - 1):
        # Target for this step
        target = ControlState(x=float(path[t + 1, 0]), y=float(path[t + 1, 1]))
        
        # Future targets for MPC horizon
        horizon_end = min(t + 1 + controller_cfg.horizon, T)
        targets_horizon = [
            ControlState(x=float(path[i, 0]), y=float(path[i, 1]))
            for i in range(t + 1, horizon_end)
        ]
        
        # Part B: Guided geometry mode - set transducer positions to straddle target
        if GUIDED_GEOMETRY:
            # Compute guided transducer positions
            xA_guided, yA_guided, xB_guided, yB_guided = compute_guided_control(
                target_x=target.x,
                target_y=target.y,
                current_control=control,
                domain_Lx=domain.Lx,
            )
            
            # Create guided control (preserving phases/amplitudes from current)
            guided_control = ControlVector(
                xA=xA_guided, yA=yA_guided,
                xB=xB_guided, yB=yB_guided,
                vA=control.vA, vB=control.vB,
                phiA=control.phiA, phiB=control.phiB,
                bounds=bounds,
                rate_limits=rate_limits,
            )
            
            # Use guided control as current (controller can still tweak phases/amplitudes)
            current_for_step = guided_control
        else:
            current_for_step = control
        
        # Controller step
        new_control, new_state, info = controller.step(
            state=state,
            target=target,
            current_control=current_for_step,
            targets_horizon=targets_horizon if use_mpc else None,
            use_mpc=use_mpc,
        )
        
        # Store previous position for displacement computation
        prev_x, prev_y = state.x, state.y
        
        # Update state
        control = new_control
        state = new_state
        traj_xy_mm.append((state.x * 1e3, state.y * 1e3))
        
        # Compute force and field for diagnostics (we need U for trap centre)
        u2p_diag = control.to_control2pucks()
        _, _, _, info_diag, field_diag, U_diag, Fx_diag, Fy_diag = ev.step(
            xp=state.x, yp=state.y, target_x=target.x, target_y=target.y,
            u=u2p_diag, u_prev=None, return_fields=True,
        )
        fx_diag, fy_diag = info_diag["fx"], info_diag["fy"]
        F_mag_diag = np.sqrt(fx_diag**2 + fy_diag**2)
        
        # Task 4: Find local trap centre
        trap_x, trap_y = find_local_trap_centre(
            field_diag.x, field_diag.y, U_diag,
            state.x, state.y, window_mm=0.2
        )
        particle_to_trap_mm = np.sqrt((state.x - trap_x)**2 + (state.y - trap_y)**2) * 1e3
        trap_to_target_mm = np.sqrt((trap_x - target.x)**2 + (trap_y - target.y)**2) * 1e3
        
        # B1: Compact diagnostic line every step
        tracking_err = state.distance_to(target) * 1e3  # mm
        jac = info.get("jacobian")
        stiff_eig = jac.stiffness_eigenvalues if jac else np.array([0, 0])
        
        # Displacement this step
        dx_step = (state.x - prev_x) * 1e3
        dy_step = (state.y - prev_y) * 1e3
        dp_step = np.sqrt(dx_step**2 + dy_step**2)
        
        # Task 6: Directional sanity check - cos(angle between desired and actual motion)
        # Desired direction: target - previous position
        d_desired = np.array([target.x - prev_x, target.y - prev_y])
        # Actual displacement
        d_actual = np.array([state.x - prev_x, state.y - prev_y])
        
        d_desired_norm = np.linalg.norm(d_desired)
        d_actual_norm = np.linalg.norm(d_actual)
        
        if d_desired_norm > 1e-12 and d_actual_norm > 1e-12:
            cos_to_target = float(np.dot(d_desired, d_actual) / (d_desired_norm * d_actual_norm))
        else:
            cos_to_target = 0.0
        cos_to_target_history.append(cos_to_target)
        
        # Print every step (compact line with limiter diagnostics)
        step_limited = 1 if info.get("step_limited", False) else 0
        step_scale = info.get("step_scale", 1.0)
        raw_step_mm = info.get("raw_step_mm", 0.0)
        
        print(f"{t+1:04d}  {state.x*1e3:7.4f}  {state.y*1e3:7.4f}  {tracking_err:7.4f}  {F_mag_diag:.3e}  {dp_step:7.4f}  "
              f"{trap_to_target_mm:7.4f}  {particle_to_trap_mm:7.4f}  {cos_to_target:6.3f}  "
              f"{step_limited:3d}  {step_scale:5.3f}  {raw_step_mm:7.4f}  "
              f"{info.get('n_rejected', 0):3d}/{info.get('n_candidates', 0):3d}")
        
        # Task 6: Warning if rolling mean of cos_to_target is negative
        if len(cos_to_target_history) >= 10:
            rolling_mean = np.mean(cos_to_target_history[-10:])
            if rolling_mean < 0 and t % 10 == 0:
                print(f"  ⚠️  WARNING: rolling cos_to_target = {rolling_mean:.3f} < 0 - controller pushing wrong direction!")
        
        # Part C: Update metrics history for time series plotting
        metrics_history['err_mm'].append(tracking_err)
        metrics_history['trap_to_target_mm'].append(trap_to_target_mm)
        metrics_history['particle_to_trap_mm'].append(particle_to_trap_mm)
        metrics_history['F_mag_N'].append(F_mag_diag)
        metrics_history['cos_to_target'].append(cos_to_target)
        
        # Part A: Mid-run probe (at step T/2)
        if RUN_CONTROL_PROBE and not mid_probe_done and t >= T // 2:
            print(f"\n  [PROBE] Running mid-run probe at step {t}...")
            probe_result_mid = probe_local_sensitivity(
                ev=ev,
                x=state.x,
                y=state.y,
                u0=control.to_control2pucks(),
                out_dir=probe_dir,
                probe_name="midrun",
            )
            mid_probe_done = True
            print("-" * 140)
        
        # B2: Collect diagnostics (with new trap centre fields)
        diag_steps.append({
            "step": t + 1,
            "px_mm": state.x * 1e3,
            "py_mm": state.y * 1e3,
            "target_x_mm": target.x * 1e3,
            "target_y_mm": target.y * 1e3,
            "err_mm": tracking_err,
            "F_mag_N": F_mag_diag,
            "fx_N": fx_diag,
            "fy_N": fy_diag,
            "dp_step_mm": dp_step,
            "xA_mm": control.xA * 1e3,
            "xB_mm": control.xB * 1e3,
            "yA_mm": control.yA * 1e3,
            "yB_mm": control.yB * 1e3,
            "vA": control.vA,
            "vB": control.vB,
            "phiA": control.phiA,
            "phiB": control.phiB,
            "stiff_eig0": float(stiff_eig[0]),
            "stiff_eig1": float(stiff_eig[1]),
            "n_rejected": info.get("n_rejected", 0),
            "n_candidates": info.get("n_candidates", 0),
            # Task 4: Trap centre diagnostics
            "trap_x_mm": trap_x * 1e3,
            "trap_y_mm": trap_y * 1e3,
            "particle_to_trap_mm": particle_to_trap_mm,
            "trap_to_target_mm": trap_to_target_mm,
            # Task 6: Directional sanity check
            "cos_to_target": cos_to_target,
            # Step limiter diagnostics
            "step_limited": step_limited,
            "step_scale": step_scale,
            "raw_step_mm": raw_step_mm,
        })
        
        # Render
        if (t % render_every) == 0:
            # Reuse field data from diagnostics instead of recomputing
            field = field_diag
            U = U_diag
            Fx = Fx_diag
            Fy = Fy_diag
            
            traps = find_traps_from_force(
                field.x, field.y, U, Fx, Fy,
                max_traps=12, force_rel_thresh=0.02, border=3,
            )
            
            cylinders = [
                Cylinder2D(x_mm=control.xA * 1e3, y_mm=control.yA * 1e3, 
                          r_mm=cyl_r_mm, alpha=0.22, edge_alpha=0.60),
                Cylinder2D(x_mm=control.xB * 1e3, y_mm=control.yB * 1e3, 
                          r_mm=cyl_r_mm, alpha=0.22, edge_alpha=0.60),
            ]
            
            # Predicted trajectory for visualization
            # Task 3: ALWAYS provide a predicted trajectory
            predicted_traj = info.get("predicted_trajectory", [])
            if predicted_traj:
                predicted_xy_mm = [(s.x * 1e3, s.y * 1e3) for s in predicted_traj]
            else:
                # Fallback: use predicted_state from one-step mode, or compute it
                pred_state = info.get("predicted_state") or info.get("actual_state")
                if pred_state:
                    predicted_xy_mm = [(pred_state.x * 1e3, pred_state.y * 1e3)]
                else:
                    # Last resort: compute one-step prediction directly
                    u2p = control.to_control2pucks()
                    xp1_pred, yp1_pred, _, _ = ev.step(
                        xp=state.x, yp=state.y, target_x=target.x, target_y=target.y,
                        u=u2p, u_prev=None, return_fields=False,
                    )
                    predicted_xy_mm = [(xp1_pred * 1e3, yp1_pred * 1e3)]
            
            # Control metrics for overlay
            jac = info.get("jacobian")
            
            # A3: Compute control deltas
            dxA = (control.xA - prev_control.xA) * 1e6  # μm
            dxB = (control.xB - prev_control.xB) * 1e6
            dvA = (control.vA - prev_control.vA) * 1e6  # μm/s
            dvB = (control.vB - prev_control.vB) * 1e6
            dphiA = control.phiA - prev_control.phiA
            dphiB = control.phiB - prev_control.phiB
            
            control_metrics = {
                "tracking_error_mm": tracking_err,
                "prediction_error_mm": info.get("prediction_error", 0) * 1e3 if "prediction_error" in info else 0,
                "stiffness": jac.stiffness_eigenvalues.tolist() if jac else None,
                "vA": control.vA,
                "vB": control.vB,
                "phiA": control.phiA,
                "phiB": control.phiB,
                "n_rejected": info.get("n_rejected", 0),
                "n_candidates": info.get("n_candidates", 0),
                # A3: Control deltas
                "dxA_um": dxA, "dxB_um": dxB,
                "dvA_um_s": dvA, "dvB_um_s": dvB,
                "dphiA": dphiA, "dphiB": dphiB,
                # Task 4: Trap centre metrics
                "trap_x_mm": trap_x * 1e3,
                "trap_y_mm": trap_y * 1e3,
                "particle_to_trap_mm": particle_to_trap_mm,
                "trap_to_target_mm": trap_to_target_mm,
                # Task 6: Directional sanity check
                "F_mag_N": F_mag_diag,
                "dp_mm": dp_step,
                "cos_to_target": cos_to_target,
            }
            
            # A2: Compute force at particle position for arrow visualization
            fx_at_p, fy_at_p = bilinear_sample_vec(field.x, field.y, Fx, Fy, state.x, state.y)
            # Convert to mm scale for arrow (forces are in N, we just want direction)
            force_arrow_mm = (fx_at_p * 1e12, fy_at_p * 1e12)  # Scale for visibility
            
            out_png = frames_dir / f"frame_{t:04d}.png"
            
            # Check for flat landscape
            U_display = U * 1e15
            _, is_flat = normalize_gorkov_field(U_display, verbose=False)
            
            if is_flat and prev_frame_path is not None:
                shutil.copy(prev_frame_path, out_png)
            else:
                render_with_control_overlay(
                    out_png=out_png,
                    x_mm=field.x * 1e3,
                    y_mm=field.y * 1e3,
                    U=U,
                    traps=traps,
                    particle_xy_mm=(state.x * 1e3, state.y * 1e3),
                    target_xy_mm=(target.x * 1e3, target.y * 1e3),
                    track_xy_mm=traj_xy_mm,
                    predicted_xy_mm=predicted_xy_mm,
                    cylinders=cylinders,
                    control_metrics=control_metrics,
                    force_arrow_mm=force_arrow_mm,
                    trap_centre_mm=(trap_x * 1e3, trap_y * 1e3),  # Task 4: Pass trap centre
                    metrics_history=metrics_history,  # Part C: Pass metrics history for time series
                )
            
            prev_frame_path = out_png
            frame_paths.append(out_png)
        
        # A3: Update prev_control for next iteration's delta computation
        prev_control = control

    # ===== Save Results =====
    print("-" * 120)
    print("\nSaving results...")
    
    # Save trajectory FIRST (before GIF which can OOM)
    np.save(out_dir / "traj_xy_mm.npy", np.array(traj_xy_mm, dtype=float))
    np.save(out_dir / "desired_xy_mm.npy", path * 1e3)
    print(f"  Trajectory: {out_dir / 'traj_xy_mm.npy'}")
    
    # Save diagnostics CSV and NPY BEFORE GIF (GIF creation can OOM)
    if diag_steps:
        import csv
        csv_path = out_dir / "diagnostics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=diag_steps[0].keys())
            writer.writeheader()
            writer.writerows(diag_steps)
        print(f"  Diagnostics CSV: {csv_path}")
        
        # Convert to structured numpy array
        diag_arr = np.array(
            [tuple(d.values()) for d in diag_steps],
            dtype=[(k, float) for k in diag_steps[0].keys()]
        )
        np.save(out_dir / "diagnostics.npy", diag_arr)
        print(f"  Diagnostics NPY: {out_dir / 'diagnostics.npy'}")
    
    # Build GIF (Part D: memory-efficient approach using PIL)
    gif_path = out_dir / "path_follow_controlled.gif"
    if frame_paths:
        from PIL import Image
        
        # Subsample frames if too many (to avoid OOM)
        stride = max(1, len(frame_paths) // 100)  # Target ~100 frames max
        selected_paths = frame_paths[::stride]
        
        print(f"  Building GIF from {len(selected_paths)} frames (stride={stride})...")
        
        # Load and resize frames for memory efficiency
        pil_frames = []
        for i, fp in enumerate(selected_paths):
            img = Image.open(fp)
            # Resize to 800x600 to reduce memory
            img = img.resize((800, 600), Image.LANCZOS)
            pil_frames.append(img)
        
        # Save GIF
        frame_duration = 100  # 100ms per frame
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True
        )
        print(f"  GIF: {gif_path} ({len(selected_paths)} frames, {frame_duration}ms/frame)")
        
        # Clean up to free memory
        del pil_frames
    
    # Print summary
    summary = controller.logger.get_summary()
    print("\n" + "=" * 70)
    print("CONTROL SUMMARY")
    print("=" * 70)
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4e}")
        else:
            print(f"  {key}: {val}")
    
    # ===== D: Actionable Conclusions =====
    print("\n" + "=" * 70)
    print("ACTIONABLE CONCLUSIONS")
    print("=" * 70)
    
    if diag_steps:
        # Compute key statistics
        max_F = max(d["F_mag_N"] for d in diag_steps)
        avg_F = np.mean([d["F_mag_N"] for d in diag_steps])
        max_dp = max(d["dp_step_mm"] for d in diag_steps)
        avg_dp = np.mean([d["dp_step_mm"] for d in diag_steps])
        final_err = diag_steps[-1]["err_mm"]
        avg_err = np.mean([d["err_mm"] for d in diag_steps])
        
        # Trap centre metrics (Task 4)
        avg_p_to_trap = np.mean([d["particle_to_trap_mm"] for d in diag_steps])
        avg_trap_to_tgt = np.mean([d["trap_to_target_mm"] for d in diag_steps])
        
        # Directional sanity check (Task 6)
        avg_cos = np.mean([d["cos_to_target"] for d in diag_steps])
        
        # Path perimeter (we computed it earlier)
        # Use path_perimeter_mm from the scaled path
        
        # Total actual displacement
        actual_displacement_mm = 0.0
        for i in range(1, len(traj_xy_mm)):
            actual_displacement_mm += np.sqrt(
                (traj_xy_mm[i][0] - traj_xy_mm[i-1][0])**2 +
                (traj_xy_mm[i][1] - traj_xy_mm[i-1][1])**2
            )
        
        # Net displacement (start to end)
        net_displacement_mm = np.sqrt(
            (traj_xy_mm[-1][0] - traj_xy_mm[0][0])**2 +
            (traj_xy_mm[-1][1] - traj_xy_mm[0][1])**2
        )
        
        print(f"\n1. FORCE ANALYSIS:")
        print(f"   Max |F|: {max_F:.3e} N")
        print(f"   Avg |F|: {avg_F:.3e} N")
        if max_F < 1e-11:
            print(f"   ⚠️  CRITICAL: Forces are negligible (<10⁻¹¹ N)")
            print(f"       -> Check: alpha_g scaling, pressure field amplitude")
        elif max_F < 1e-9:
            print(f"   ⚠️  WARNING: Forces are weak (<10⁻⁹ N)")
        else:
            print(f"   ✓  Forces are in reasonable range")
        
        print(f"\n2. DISPLACEMENT ANALYSIS:")
        print(f"   Max Δp/step: {max_dp:.4f} mm ({max_dp*1000:.1f} μm)")
        print(f"   Avg Δp/step: {avg_dp:.4f} mm ({avg_dp*1000:.1f} μm)")
        print(f"   Path perimeter: {path_perimeter_mm:.4f} mm")
        print(f"   Total actual distance: {actual_displacement_mm:.4f} mm")
        print(f"   Net displacement: {net_displacement_mm:.4f} mm")
        print(f"   Coverage: {actual_displacement_mm/path_perimeter_mm*100:.1f}%")
        
        if max_dp < 0.01:  # 10 μm
            print(f"   ⚠️  WARNING: Max step < 10 μm - controller has very limited authority")
            steps_needed = path_perimeter_mm / max_dp if max_dp > 0 else float('inf')
            print(f"       -> Would need ~{steps_needed:.0f} steps to cover path at max rate")
        
        print(f"\n3. TRACKING ERROR:")
        print(f"   Final error: {final_err:.4f} mm ({final_err*1000:.1f} μm)")
        print(f"   Avg error:   {avg_err:.4f} mm ({avg_err*1000:.1f} μm)")
        
        print(f"\n4. TRAP CENTRE ANALYSIS (Task 4):")
        print(f"   Mean particle→trap distance: {avg_p_to_trap:.4f} mm")
        print(f"   Mean trap→target distance: {avg_trap_to_tgt:.4f} mm")
        if avg_p_to_trap > 0.1:
            print(f"   ⚠️  WARNING: Particle not staying near trap centre")
        if avg_trap_to_tgt > 0.2:
            print(f"   ⚠️  WARNING: Trap is not moving toward target")
        
        print(f"\n5. DIRECTIONAL SANITY CHECK (Task 6):")
        print(f"   Mean cos_to_target: {avg_cos:.3f}")
        if avg_cos < 0:
            print(f"   ⚠️  CRITICAL: cos < 0 means controller pushing WRONG direction!")
        elif avg_cos < 0.3:
            print(f"   ⚠️  WARNING: cos < 0.3 means poor alignment with target direction")
        else:
            print(f"   ✓  Positive alignment with target direction")
        
        # Stiffness analysis
        stiff_vals = [d["stiff_eig0"] for d in diag_steps] + [d["stiff_eig1"] for d in diag_steps]
        max_stiff = max(stiff_vals)
        print(f"\n6. STIFFNESS:")
        print(f"   Max eigenvalue: {max_stiff:.3e}")
        if max_stiff > 0:
            print(f"   ⚠️  WARNING: Positive eigenvalue = unstable trap")
        
        print(f"\n7. RECOMMENDATIONS:")
        if max_dp < 0.05:
            print(f"   a) Increase alpha_g (currently {cfg.alpha_g:.0e}) by 10-100x")
            print(f"   b) Increase transducer amplitude v_max")
            print(f"   c) Use larger dt for larger displacement steps")
        if avg_err > 0.1 * path_perimeter_mm:
            print(f"   d) Path is not being followed - authority vs path scale mismatch")
        if avg_cos < 0:
            print(f"   e) CRITICAL: Controller logic needs review - pushing wrong direction")
        if avg_p_to_trap > 0.1:
            print(f"   f) Particle escaping trap - increase trap_weight or stiffness_weight")
    
    # Save conclusions to file (with new metrics)
    conclusions_path = out_dir / "conclusions.txt"
    with open(conclusions_path, "w") as f:
        f.write("ACTIONABLE CONCLUSIONS\n")
        f.write("=" * 70 + "\n\n")
        if diag_steps:
            f.write(f"Configuration:\n")
            f.write(f"  PATH_SCALE: {PATH_SCALE}\n")
            f.write(f"  T_STEPS: {T_STEPS}\n")
            f.write(f"  alpha_g: {cfg.alpha_g:.0e}\n")
            f.write(f"  dt: {cfg.dt:.0e}\n\n")
            
            f.write(f"Force Analysis:\n")
            f.write(f"  Max |F|: {max_F:.3e} N\n")
            f.write(f"  Avg |F|: {avg_F:.3e} N\n\n")
            
            f.write(f"Displacement Analysis:\n")
            f.write(f"  Max Δp/step: {max_dp:.4f} mm ({max_dp*1000:.1f} μm)\n")
            f.write(f"  Avg Δp/step: {avg_dp:.4f} mm ({avg_dp*1000:.1f} μm)\n")
            f.write(f"  Path perimeter: {path_perimeter_mm:.4f} mm\n")
            f.write(f"  Total actual: {actual_displacement_mm:.4f} mm\n")
            f.write(f"  Net displacement: {net_displacement_mm:.4f} mm\n")
            f.write(f"  Coverage: {actual_displacement_mm/path_perimeter_mm*100:.1f}%\n\n")
            
            f.write(f"Tracking Error:\n")
            f.write(f"  Final: {final_err:.4f} mm\n")
            f.write(f"  Avg: {avg_err:.4f} mm\n\n")
            
            f.write(f"Trap Centre Analysis:\n")
            f.write(f"  Mean particle→trap: {avg_p_to_trap:.4f} mm\n")
            f.write(f"  Mean trap→target: {avg_trap_to_tgt:.4f} mm\n\n")
            
            f.write(f"Directional Sanity Check:\n")
            f.write(f"  Mean cos_to_target: {avg_cos:.3f}\n")
            
    print(f"\n  Conclusions saved to: {conclusions_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
