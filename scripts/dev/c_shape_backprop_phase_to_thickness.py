#!/usr/bin/env python3
"""
C-Shape Backpropagation → Phase → Thickness Lens Design
=========================================================

Single-pass inverse lens design: take a chosen C-shaped target field,
back-propagate to the lens plane via ASM, extract a phase-only circular
lens design, convert to printable thickness, and forward-validate.

Usage:
    python scripts/dev/c_shape_backprop_phase_to_thickness.py

Outputs → results/c_shape_backprop_phase_to_thickness_<TS>/
"""
from __future__ import annotations

import json
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.physics.acoustics.angular_spectrum import (
    propagate_pressure_asm,
)

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LensDesignConfig:
    """All parameters for the inverse lens design pipeline."""

    # ── Source C-shape study ──────────────────────────────────────
    study_dir: str = str(PROJECT_ROOT / "results"
                         / "c_shape_on_cached_sw_geometry_study_20260310_102151")
    chosen_rank: int = 3   # best_03

    # ── Physics ──────────────────────────────────────────────────
    frequency_hz: float = 2.0e6
    c_water: float = 1484.0
    rho_water: float = 997.0
    c_lens: float = 2700.0         # acrylic / resin
    rho_lens: float = 1180.0       # acrylic density (informational)

    # ── Geometry ─────────────────────────────────────────────────
    # FEM domain is 6×6 mm; trap plane at z*
    domain_lx: float = 6.0e-3
    domain_ly: float = 6.0e-3
    h_under: float = 5.0e-3        # bottom chamber
    h_top: float = 2.0e-3          # top chamber
    z_star: float = 0.0            # computed in __post_init__

    # Lens plane: bottom face z = 0 (transducer surface / lens face)
    z_lens: float = 0.0

    # ── Lens design ──────────────────────────────────────────────
    lens_diameters_mm: tuple = (5.0, 8.0, 10.0, 15.0, 20.0)
    max_thickness_mm: float = 20.0  # fabrication ceiling

    # ── Grid ─────────────────────────────────────────────────────
    n_grid: int = 512    # for the backprop domain (must cover full lens)

    # ── Smoothing / fabrication ──────────────────────────────────
    smooth_sigma_mm: float = 0.05   # Gaussian smoothing σ (50 µm)
    max_slope_deg: float = 60.0     # max printable slope angle
    min_feature_mm: float = 0.05    # minimum resolvable feature

    # ── Apodisation ──────────────────────────────────────────────
    apod_roll_off: float = 0.15     # fraction of ROI half-width for taper

    def __post_init__(self):
        lam = self.c_water / self.frequency_hz
        self.z_star = self.h_under + self.h_top / 2.0 + 0.25 * lam
        self.wavelength = lam
        self.k_water = 2.0 * np.pi * self.frequency_hz / self.c_water
        self.k_lens = 2.0 * np.pi * self.frequency_hz / self.c_lens
        self.omega = 2.0 * np.pi * self.frequency_hz
        self.dk = self.k_lens - self.k_water  # negative for c_lens > c_water
        self.full_2pi_thickness = 2.0 * np.pi / abs(self.dk)


CFG = LensDesignConfig()

# ── Output ──────────────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / f"c_shape_backprop_phase_to_thickness_{TS}"
FIG_DIR = OUT_DIR / "figures"
LENS_DIR = OUT_DIR / "lens_exports"
NPZ_DIR = OUT_DIR / "npz"
for d in [OUT_DIR, FIG_DIR, LENS_DIR, NPZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Step 1: Reconstruct the chosen target field
# ═══════════════════════════════════════════════════════════════════

def load_selected_c_shape(cfg: LensDesignConfig):
    """
    Load the chosen C-shape candidate from the study NPZ.
    Returns the normalised C-field, combined field, grid, and params.
    """
    study = Path(cfg.study_dir)
    npz_path = study / "npz" / f"best_{cfg.chosen_rank:02d}.npz"
    d = np.load(npz_path, allow_pickle=True)
    p_C = d["p_C"]         # normalised C-field (peak=1)
    p_comb = d["p_comb"]   # standing-wave + alpha*C combined
    xg = d["xg"]           # (N,) metres
    yg = d["yg"]           # (N,) metres
    params = d["params"].item()
    metrics = d["metrics"].item()

    # Also load config for trap pair info
    with open(study / "config.json") as f:
        study_cfg = json.load(f)

    print(f"[load] Loaded best_{cfg.chosen_rank:02d} from {study.name}")
    print(f"  params: r0/d={params['r0_rel']}, σr/d={params['sr_rel']}, "
          f"σθ={params['sigma_theta']}, gap={params['gap_label']}, α={params['alpha']}")
    print(f"  p_C shape: {p_C.shape}, |p_C| range: [{np.abs(p_C).min():.4f}, {np.abs(p_C).max():.4f}]")

    return p_C, p_comb, xg, yg, params, metrics, study_cfg


def reconstruct_target_field(p_C_norm, xg, yg, params, study_cfg, cfg):
    """
    Reconstruct the target complex field on an enlarged grid suitable for
    ASM backpropagation.  The C-field is embedded in a larger domain and
    apodised to taper smoothly to zero at the edges.

    Returns (p_target, xg_full, yg_full, embed_info).
    """
    # The C-field from the study sits on a small ROI (~1.6×1.6 mm).
    # For ASM we need a grid that covers the full lens aperture at the
    # target plane (up to 20 mm lens → 20 mm grid at minimum).
    # Use a conservatively large grid.
    max_lens_r = max(cfg.lens_diameters_mm) / 2.0 * 1e-3  # metres
    # Domain must be larger than the lens to avoid wraparound.
    # Use 1.5× the largest lens diameter.
    domain_half = max(max_lens_r * 1.5, cfg.domain_lx / 2.0)
    domain_size = 2.0 * domain_half

    N = cfg.n_grid
    xg_full = np.linspace(-domain_half, domain_half, N)
    yg_full = np.linspace(-domain_half, domain_half, N)
    dx_full = xg_full[1] - xg_full[0]

    # Centre of the original ROI
    cx_roi = 0.5 * (xg[0] + xg[-1])
    cy_roi = 0.5 * (yg[0] + yg[-1])

    print(f"[target] Embedding C-field into {N}×{N} grid, "
          f"domain {domain_size*1e3:.1f}×{domain_size*1e3:.1f} mm")
    print(f"  dx = {dx_full*1e6:.1f} µm, ROI centre = ({cx_roi*1e3:.3f}, {cy_roi*1e3:.3f}) mm")

    # Re-centre ROI coordinates to the new grid (centred at 0,0)
    xg_shifted = xg - cx_roi
    yg_shifted = yg - cy_roi

    # Interpolate C-field onto the new grid using nearest-neighbour mapping
    # (the original grid is finer than what we need, so this is safe)
    from scipy.interpolate import RegularGridInterpolator
    interp_real = RegularGridInterpolator(
        (yg_shifted, xg_shifted), np.real(p_C_norm),
        bounds_error=False, fill_value=0.0, method="linear")
    interp_imag = RegularGridInterpolator(
        (yg_shifted, xg_shifted), np.imag(p_C_norm),
        bounds_error=False, fill_value=0.0, method="linear")

    YY, XX = np.meshgrid(yg_full, xg_full, indexing="ij")
    pts = np.column_stack([YY.ravel(), XX.ravel()])
    p_target = (interp_real(pts) + 1j * interp_imag(pts)).reshape(N, N)

    print(f"  |p_target| range: [{np.abs(p_target).min():.4f}, {np.abs(p_target).max():.4f}]")

    embed_info = {
        "cx_roi_m": float(cx_roi),
        "cy_roi_m": float(cy_roi),
        "domain_half_m": float(domain_half),
        "dx_m": float(dx_full),
        "n_grid": N,
    }
    return p_target, xg_full, yg_full, embed_info


def apodise_target_field(p_target, xg, yg, cfg):
    """
    Apply a soft cosine window around the C-field so it tapers smoothly
    to zero toward the grid boundary.
    """
    roi_half_orig = 1.1 * cfg.wavelength  # original ROI half-width
    roll = cfg.apod_roll_off * roi_half_orig

    # Distance from grid centre
    XX, YY = np.meshgrid(xg, yg)
    R = np.sqrt(XX**2 + YY**2)

    # Taper from full amplitude at R < (roi_half - roll)
    # to zero at R > roi_half
    inner = roi_half_orig - roll
    outer = roi_half_orig
    window = np.ones_like(R)
    trans = (R > inner) & (R <= outer)
    window[trans] = 0.5 * (1.0 + np.cos(np.pi * (R[trans] - inner) / (outer - inner)))
    window[R > outer] = 0.0

    p_apod = p_target * window

    # Scale so the C-field has a target peak amplitude in Pa
    # Use the study's alpha convention: alpha = 0.10 × SW peak
    # SW peak on the ROI was ~46 Pa, so alpha ≈ 4.6 Pa
    sw_peak_est = 46.0   # Pa (from previous runs)
    alpha_frac = 0.10
    target_amplitude = alpha_frac * sw_peak_est  # ~4.6 Pa

    current_peak = np.abs(p_apod).max()
    if current_peak > 0:
        p_apod = p_apod * (target_amplitude / current_peak)

    print(f"[apod] Applied cosine taper (inner={inner*1e3:.3f} mm, outer={outer*1e3:.3f} mm)")
    print(f"  Peak amplitude after scaling: {np.abs(p_apod).max():.2f} Pa")

    return p_apod


# ═══════════════════════════════════════════════════════════════════
# Step 2: Backpropagate with ASM
# ═══════════════════════════════════════════════════════════════════

def asm_backpropagate_to_lens(p_target, dx, dy, cfg):
    """
    Back-propagate the target field from z_star to z_lens using ASM.

    Backpropagation is simply forward propagation with z → -Δz.
    """
    dz = cfg.z_star - cfg.z_lens   # positive distance from lens to target
    print(f"[backprop] ASM: z_target={cfg.z_star*1e3:.3f} mm → z_lens={cfg.z_lens*1e3:.1f} mm")
    print(f"  Δz = {dz*1e3:.3f} mm = {dz/cfg.wavelength:.1f} λ")

    # Back-propagate = propagate with negative z
    p_lens = propagate_pressure_asm(
        p_target, dx, dy, cfg.k_water, z=-dz,
        pad_factor=2, include_evanescent=False)

    amp = np.abs(p_lens)
    print(f"  Lens-plane |p| range: [{amp.min():.4f}, {amp.max():.4f}] Pa")
    print(f"  Lens-plane phase range: [{np.angle(p_lens).min():.3f}, {np.angle(p_lens).max():.3f}] rad")

    return p_lens


# ═══════════════════════════════════════════════════════════════════
# Step 3: Circular aperture
# ═══════════════════════════════════════════════════════════════════

def make_circular_aperture(xg, yg, diameter_m):
    """
    Binary circular aperture mask centred at (0,0).
    Returns (mask, energy_fraction).
    """
    XX, YY = np.meshgrid(xg, yg)
    R = np.sqrt(XX**2 + YY**2)
    radius = diameter_m / 2.0
    mask = R <= radius
    return mask


def compute_aperture_diagnostics(p_lens, xg, yg, mask, diameter_mm):
    """Compute energy fraction inside aperture."""
    total_energy = np.sum(np.abs(p_lens)**2)
    inside_energy = np.sum(np.abs(p_lens[mask])**2)
    frac = inside_energy / max(total_energy, 1e-30)
    print(f"  D={diameter_mm:.0f} mm: energy inside aperture = {frac*100:.1f}%")
    return {
        "diameter_mm": diameter_mm,
        "energy_fraction": float(frac),
        "n_pixels_inside": int(mask.sum()),
        "n_pixels_total": int(mask.size),
    }


# ═══════════════════════════════════════════════════════════════════
# Step 4: Phase-only projection
# ═══════════════════════════════════════════════════════════════════

def extract_phase_only_design(p_lens, mask):
    """
    Extract the phase-only lens design inside the circular aperture.
    Returns wrapped phase in [0, 2π) and the phase-only field.
    """
    phase_raw = np.angle(p_lens)                # [-π, π]
    phase_wrapped = np.mod(phase_raw, 2.0 * np.pi)  # [0, 2π)

    # Phase-only: unit amplitude inside aperture, zero outside
    p_phase_only = np.zeros_like(p_lens)
    p_phase_only[mask] = np.exp(1j * phase_raw[mask])

    return phase_raw, phase_wrapped, p_phase_only


# ═══════════════════════════════════════════════════════════════════
# Step 5: Phase → Thickness
# ═══════════════════════════════════════════════════════════════════

def phase_to_thickness(phase_wrapped, mask, cfg):
    """
    Convert wrapped phase [0,2π) to physical lens thickness.

    t(x,y) = t_base + phase_wrapped / |dk|

    Since dk = k_lens - k_water < 0 for c_lens > c_water,
    the phase delay due to a lens of thickness t replacing water is:
        Δφ = (k_lens - k_water) · t = dk · t

    For dk < 0, increasing thickness → decreasing phase.
    We want:  Δφ = phase_wrapped  →  t = phase_wrapped / dk
    Since dk<0, this gives negative t, so we use |dk| and offset:

    Actually, a slab of lens material of thickness t transmits:
        φ_slab = k_lens · t   (through lens) + k_water · (t_max - t) (through water gap)
    Relative to uniform water of thickness t_max:
        Δφ = (k_lens - k_water) · t = dk · t

    For dk < 0:  thicker lens → more negative phase shift.
    We want the lens to APPLY a target phase φ_target(x,y), so:
        dk · t(x,y) = φ_target(x,y)  mod 2π
        t(x,y) = φ_target(x,y) / dk

    With wrapping to [0,2π), the physical thickness range is:
        t ∈ [0, 2π/|dk|]

    We add a base thickness t0 for structural integrity.
    """
    dk = cfg.dk  # negative
    t_2pi = 2.0 * np.pi / abs(dk)  # thickness for full 2π phase cycle

    # Convert phase to thickness offset
    # phase_wrapped ∈ [0, 2π)  →  Δt = phase / |dk| ∈ [0, t_2pi)
    delta_t = phase_wrapped / abs(dk)

    # Base thickness: minimum structural floor (0.5 mm)
    t_base = 0.5e-3  # metres

    thickness = np.full_like(phase_wrapped, np.nan)
    thickness[mask] = t_base + delta_t[mask]

    t_min = np.nanmin(thickness[mask]) if mask.any() else 0
    t_max = np.nanmax(thickness[mask]) if mask.any() else 0

    print(f"  Thickness: t_base={t_base*1e3:.2f} mm, "
          f"2π increment={t_2pi*1e3:.3f} mm")
    print(f"  Range: [{t_min*1e3:.3f}, {t_max*1e3:.3f}] mm")
    print(f"  Max target: {cfg.max_thickness_mm:.0f} mm → "
          f"{'OK' if t_max*1e3 <= cfg.max_thickness_mm else 'EXCEEDS'}")

    info = {
        "t_base_mm": t_base * 1e3,
        "t_2pi_mm": t_2pi * 1e3,
        "t_min_mm": float(t_min * 1e3),
        "t_max_mm": float(t_max * 1e3),
        "within_budget": bool(t_max * 1e3 <= cfg.max_thickness_mm),
    }
    return thickness, info


# ═══════════════════════════════════════════════════════════════════
# Step 6: Smoothing & slope limiting
# ═══════════════════════════════════════════════════════════════════

def smooth_and_limit_thickness(thickness, mask, xg, yg, cfg):
    """
    Apply manufacturability filters for resin 3D printing.

    1. Gaussian smoothing (removes sub-resolution features)
    2. Slope limiting (max printable overhang angle)
    """
    dx = xg[1] - xg[0]
    t_work = thickness.copy()

    # Fill NaN (outside aperture) with the base thickness for smoothing
    t_base = np.nanmin(t_work[mask]) if mask.any() else 0.5e-3
    t_work[~mask] = t_base

    # ── 1. Gaussian smoothing ────────────────────────────────────
    sigma_px = cfg.smooth_sigma_mm * 1e-3 / dx
    t_smooth = gaussian_filter(t_work, sigma=sigma_px)

    # ── 2. Slope limiting ────────────────────────────────────────
    max_slope_rad = np.radians(cfg.max_slope_deg)
    max_gradient = np.tan(max_slope_rad)  # rise/run
    max_delta_per_px = max_gradient * dx

    # Iterative slope clipping (a few passes suffice)
    t_clipped = t_smooth.copy()
    for _ in range(10):
        dt_dx = np.diff(t_clipped, axis=1)
        dt_dy = np.diff(t_clipped, axis=0)

        # Clip x-gradients
        excess_x = np.abs(dt_dx) - max_delta_per_px
        if np.any(excess_x > 0):
            sign_x = np.sign(dt_dx)
            dt_clipped_x = sign_x * np.minimum(np.abs(dt_dx), max_delta_per_px)
            # Reconstruct from clipped gradients (forward pass)
            for j in range(1, t_clipped.shape[1]):
                t_clipped[:, j] = np.clip(
                    t_clipped[:, j],
                    t_clipped[:, j-1] - max_delta_per_px,
                    t_clipped[:, j-1] + max_delta_per_px)
        for i in range(1, t_clipped.shape[0]):
            t_clipped[i, :] = np.clip(
                t_clipped[i, :],
                t_clipped[i-1, :] - max_delta_per_px,
                t_clipped[i-1, :] + max_delta_per_px)

    # Restore mask
    t_printable = t_clipped.copy()
    t_printable[~mask] = np.nan

    # ── Diagnostics ──────────────────────────────────────────────
    valid = mask
    rms_change = np.sqrt(np.mean((t_printable[valid] - thickness[valid])**2))
    max_change = np.max(np.abs(t_printable[valid] - thickness[valid]))

    # Compute slope map
    dt_dx_final = np.gradient(t_printable, dx, axis=1)
    dt_dy_final = np.gradient(t_printable, dx, axis=0)
    slope_mag = np.sqrt(dt_dx_final**2 + dt_dy_final**2)
    slope_angle = np.degrees(np.arctan(slope_mag))

    print(f"[fab] Smoothing σ={cfg.smooth_sigma_mm:.2f} mm, "
          f"max slope={cfg.max_slope_deg:.0f}°")
    print(f"  RMS thickness change: {rms_change*1e6:.1f} µm")
    print(f"  Max thickness change: {max_change*1e6:.1f} µm")
    print(f"  Max slope after limiting: {np.nanmax(slope_angle[valid]):.1f}°")

    fab_info = {
        "smooth_sigma_mm": cfg.smooth_sigma_mm,
        "max_slope_deg": cfg.max_slope_deg,
        "rms_change_um": float(rms_change * 1e6),
        "max_change_um": float(max_change * 1e6),
        "max_slope_achieved_deg": float(np.nanmax(slope_angle[valid])),
    }
    return t_printable, slope_angle, fab_info


# ═══════════════════════════════════════════════════════════════════
# Step 7: 3D rendering
# ═══════════════════════════════════════════════════════════════════

def render_lens_3d(thickness, xg, yg, mask, diameter_mm, save_dir):
    """Generate top view, cross-sections, and 3D surface rendering."""
    dx = xg[1] - xg[0]
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    t_mm = thickness * 1e3
    t_mm_plot = np.where(mask, t_mm, np.nan)

    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    # ── Top view ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(t_mm_plot, origin="lower", extent=extent, cmap="viridis",
                   aspect="equal")
    _add_colorbar(ax, im, "Thickness (mm)")
    circle = plt.Circle((0, 0), diameter_mm/2, fill=False, color="red",
                         lw=1.5, ls="--")
    ax.add_patch(circle)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Lens Thickness — D={diameter_mm:.0f} mm")
    fig.tight_layout()
    fig.savefig(save_dir / f"thickness_top_D{diameter_mm:.0f}mm.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Cross-sections through centre ────────────────────────────
    ny, nx = thickness.shape
    cy, cx = ny // 2, nx // 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    t_x_profile = t_mm[cy, :]
    valid_x = mask[cy, :]
    ax.plot(xg_mm[valid_x], t_x_profile[valid_x], "b-", lw=1.5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Thickness (mm)")
    ax.set_title("Y=0 Cross-Section")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    t_y_profile = t_mm[:, cx]
    valid_y = mask[:, cx]
    ax.plot(yg_mm[valid_y], t_y_profile[valid_y], "r-", lw=1.5)
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("Thickness (mm)")
    ax.set_title("X=0 Cross-Section")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Lens Profile Cuts — D={diameter_mm:.0f} mm", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / f"thickness_profiles_D{diameter_mm:.0f}mm.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 3D surface ───────────────────────────────────────────────
    XX_mm, YY_mm = np.meshgrid(xg_mm, yg_mm)
    # Subsample for manageable 3D plot
    step = max(1, nx // 200)
    Xs = XX_mm[::step, ::step]
    Ys = YY_mm[::step, ::step]
    Zs = t_mm_plot[::step, ::step]
    Ms = mask[::step, ::step]

    # Set outside-aperture to NaN for clean rendering
    Zs_clean = np.where(Ms, Zs, np.nan)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xs, Ys, Zs_clean, cmap="viridis", alpha=0.9,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("Thickness (mm)")
    ax.set_title(f"3D Lens Surface — D={diameter_mm:.0f} mm")
    ax.view_init(elev=35, azim=-60)
    fig.tight_layout()
    fig.savefig(save_dir / f"lens_3d_D{diameter_mm:.0f}mm.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Additional viewpoints
    for elev, azim, label in [(90, 0, "top"), (15, -60, "oblique"), (0, 0, "side_x"), (0, -90, "side_y")]:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Xs, Ys, Zs_clean, cmap="viridis", alpha=0.9,
                        linewidth=0, antialiased=True, rstride=1, cstride=1)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("t (mm)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"D={diameter_mm:.0f} mm — {label} view")
        fig.tight_layout()
        fig.savefig(save_dir / f"lens_3d_{label}_D{diameter_mm:.0f}mm.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Step 8: Forward validation
# ═══════════════════════════════════════════════════════════════════

def forward_validate_lens(p_phase_only, t_printable, mask, xg, yg, cfg):
    """
    Forward-propagate the phase-only and printable designs to the
    target plane and return the reconstructed fields.
    """
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    dz = cfg.z_star - cfg.z_lens

    # ── Phase-only (ideal) ───────────────────────────────────────
    p_recon_ideal = propagate_pressure_asm(
        p_phase_only, dx, dy, cfg.k_water, z=dz,
        pad_factor=2, include_evanescent=False)

    # ── Printable lens ───────────────────────────────────────────
    # Convert printable thickness back to phase
    t_valid = np.where(mask, t_printable, 0.0)
    t_base = np.nanmin(t_printable[mask]) if mask.any() else 0.5e-3
    delta_t = t_valid - t_base
    phase_printable = delta_t * abs(cfg.dk)  # recover wrapped phase
    p_printable_drive = np.zeros_like(p_phase_only)
    p_printable_drive[mask] = np.exp(1j * phase_printable[mask])

    p_recon_printable = propagate_pressure_asm(
        p_printable_drive, dx, dy, cfg.k_water, z=dz,
        pad_factor=2, include_evanescent=False)

    print(f"[fwd] Ideal recon |p| range: [{np.abs(p_recon_ideal).min():.4f}, {np.abs(p_recon_ideal).max():.4f}]")
    print(f"[fwd] Printable recon |p| range: [{np.abs(p_recon_printable).min():.4f}, {np.abs(p_recon_printable).max():.4f}]")

    return p_recon_ideal, p_recon_printable


def compute_reconstruction_metrics(p_target, p_recon, xg, yg, cfg, label=""):
    """
    Compute metrics comparing reconstruction with target.
    Focus on the ROI around the C-shape.
    """
    XX, YY = np.meshgrid(xg, yg)
    R = np.sqrt(XX**2 + YY**2)
    roi_r = 1.1 * cfg.wavelength   # original ROI half-width
    roi_mask = R <= roi_r

    # Amplitude overlap (normalised correlation)
    a_tgt = np.abs(p_target)
    a_rec = np.abs(p_recon)
    a_tgt_roi = a_tgt[roi_mask]
    a_rec_roi = a_rec[roi_mask]

    if np.linalg.norm(a_tgt_roi) > 0 and np.linalg.norm(a_rec_roi) > 0:
        amp_corr = float(np.dot(a_tgt_roi, a_rec_roi) /
                         (np.linalg.norm(a_tgt_roi) * np.linalg.norm(a_rec_roi)))
    else:
        amp_corr = 0.0

    # Energy ratio in ROI
    e_roi = np.sum(a_rec_roi**2)
    e_total = np.sum(a_rec**2)
    energy_in_roi = float(e_roi / max(e_total, 1e-30))

    # Phase agreement in ROI (where amplitude is significant)
    sig_mask = roi_mask & (a_tgt > 0.1 * a_tgt.max()) & (a_rec > 0.1 * a_rec.max())
    if sig_mask.any():
        phase_diff = np.angle(p_recon[sig_mask] * np.conj(p_target[sig_mask]))
        phase_rms = float(np.sqrt(np.mean(phase_diff**2)))
    else:
        phase_rms = np.nan

    # Peak amplitude ratio
    peak_ratio = float(a_rec.max() / max(a_tgt.max(), 1e-30))

    metrics = {
        f"amp_correlation_{label}": amp_corr,
        f"energy_in_roi_{label}": energy_in_roi,
        f"phase_rms_rad_{label}": phase_rms,
        f"peak_ratio_{label}": peak_ratio,
    }
    print(f"  [{label}] amp_corr={amp_corr:.4f}, energy_in_roi={energy_in_roi:.4f}, "
          f"phase_rms={phase_rms:.3f} rad, peak_ratio={peak_ratio:.4f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════

def _add_colorbar(ax, im, label=""):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    plt.colorbar(im, cax=cax, label=label)


def plot_target_field(p_target, xg, yg, save_dir):
    """Plot target amplitude, phase, real part."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    im = ax.imshow(np.abs(p_target), origin="lower", extent=extent,
                   cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p| (Pa)")
    ax.set_title("Target Amplitude")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1]
    im = ax.imshow(np.angle(p_target), origin="lower", extent=extent,
                   cmap="twilight", aspect="equal", vmin=-np.pi, vmax=np.pi)
    _add_colorbar(ax, im, "Phase (rad)")
    ax.set_title("Target Phase")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[2]
    im = ax.imshow(np.real(p_target), origin="lower", extent=extent,
                   cmap="RdBu_r", aspect="equal")
    _add_colorbar(ax, im, "Re(p) (Pa)")
    ax.set_title("Target Real Part")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle("Target C-Shape Field at z*", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "target_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lens_plane(p_lens, xg, yg, save_dir):
    """Plot backpropagated lens-plane field."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(np.abs(p_lens), origin="lower", extent=extent,
                   cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p| (Pa)")
    ax.set_title("Backpropagated Amplitude at Lens Plane")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1]
    im = ax.imshow(np.angle(p_lens), origin="lower", extent=extent,
                   cmap="twilight", aspect="equal", vmin=-np.pi, vmax=np.pi)
    _add_colorbar(ax, im, "Phase (rad)")
    ax.set_title("Backpropagated Phase at Lens Plane")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle(f"Lens Plane (z=0) — Raw Backpropagation", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "lens_plane_raw.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_aperture_comparison(p_lens, xg, yg, masks, diameters_mm, save_dir):
    """Show aperture masks overlaid on lens-plane field."""
    n = len(diameters_mm)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1:
        axes = [axes]

    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    for ax, mask, d_mm in zip(axes, masks, diameters_mm):
        amp = np.abs(p_lens)
        amp_masked = np.where(mask, amp, 0)
        im = ax.imshow(amp_masked, origin="lower", extent=extent,
                       cmap="inferno", aspect="equal")
        circle = plt.Circle((0, 0), d_mm/2, fill=False, color="cyan",
                             lw=1.5, ls="--")
        ax.add_patch(circle)
        ax.set_title(f"D={d_mm:.0f} mm")
        ax.set_xlabel("x (mm)")
        if ax == axes[0]:
            ax.set_ylabel("y (mm)")

    fig.suptitle("Aperture Coverage on Backpropagated Field", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "aperture_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_only(phase_raw, phase_wrapped, mask, xg, yg, diameter_mm, save_dir):
    """Plot wrapped/unwrapped phase and histogram."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    phase_show = np.where(mask, phase_raw, np.nan)
    im = ax.imshow(phase_show, origin="lower", extent=extent,
                   cmap="twilight", aspect="equal", vmin=-np.pi, vmax=np.pi)
    _add_colorbar(ax, im, "Phase (rad)")
    ax.set_title("Wrapped Phase (−π, π]")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1]
    phase_w_show = np.where(mask, phase_wrapped, np.nan)
    im = ax.imshow(phase_w_show, origin="lower", extent=extent,
                   cmap="hsv", aspect="equal", vmin=0, vmax=2*np.pi)
    _add_colorbar(ax, im, "Phase (rad)")
    ax.set_title("Wrapped Phase [0, 2π)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[2]
    ax.hist(phase_raw[mask].ravel(), bins=80, edgecolor="k", alpha=0.7)
    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Phase Distribution")

    fig.suptitle(f"Phase-Only Lens Design — D={diameter_mm:.0f} mm", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / f"phase_only_D{diameter_mm:.0f}mm.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_thickness_diagnostics(thickness_raw, thickness_print, slope_angle,
                                mask, xg, yg, diameter_mm, save_dir):
    """Plot raw/filtered thickness, histogram, slope map."""
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3
    extent = [xg_mm[0], xg_mm[-1], yg_mm[0], yg_mm[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Raw thickness
    ax = axes[0, 0]
    t_raw_show = np.where(mask, thickness_raw * 1e3, np.nan)
    im = ax.imshow(t_raw_show, origin="lower", extent=extent,
                   cmap="viridis", aspect="equal")
    _add_colorbar(ax, im, "Thickness (mm)")
    ax.set_title("Raw Thickness")

    # Printable thickness
    ax = axes[0, 1]
    t_print_show = np.where(mask, thickness_print * 1e3, np.nan)
    im = ax.imshow(t_print_show, origin="lower", extent=extent,
                   cmap="viridis", aspect="equal")
    _add_colorbar(ax, im, "Thickness (mm)")
    ax.set_title("Printable Thickness")

    # Histogram
    ax = axes[1, 0]
    if mask.any():
        ax.hist(thickness_raw[mask].ravel() * 1e3, bins=60, alpha=0.6,
                label="Raw", edgecolor="k")
        ax.hist(thickness_print[mask].ravel() * 1e3, bins=60, alpha=0.6,
                label="Printable", edgecolor="k")
    ax.set_xlabel("Thickness (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Thickness Distribution")
    ax.legend()

    # Slope map
    ax = axes[1, 1]
    slope_show = np.where(mask, slope_angle, np.nan)
    im = ax.imshow(slope_show, origin="lower", extent=extent,
                   cmap="hot", aspect="equal", vmin=0, vmax=90)
    _add_colorbar(ax, im, "Slope (°)")
    ax.set_title("Local Slope Angle")

    fig.suptitle(f"Thickness & Manufacturability — D={diameter_mm:.0f} mm", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / f"thickness_diagnostics_D{diameter_mm:.0f}mm.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_comparison(p_target, p_recon_ideal, p_recon_print,
                                    xg, yg, cfg, diameter_mm, save_dir):
    """Side-by-side target vs reconstructions with error maps."""
    roi_r = 1.1 * cfg.wavelength
    xg_mm = xg * 1e3
    yg_mm = yg * 1e3

    # Zoom to ROI
    xl, xh = -roi_r * 1e3, roi_r * 1e3
    yl, yh = -roi_r * 1e3, roi_r * 1e3
    ix_lo = np.searchsorted(xg_mm, xl)
    ix_hi = np.searchsorted(xg_mm, xh)
    iy_lo = np.searchsorted(yg_mm, yl)
    iy_hi = np.searchsorted(yg_mm, yh)

    s = np.s_[iy_lo:iy_hi, ix_lo:ix_hi]
    extent_roi = [xg_mm[ix_lo], xg_mm[ix_hi-1], yg_mm[iy_lo], yg_mm[iy_hi-1]]

    a_tgt = np.abs(p_target[s])
    a_ideal = np.abs(p_recon_ideal[s])
    a_print = np.abs(p_recon_print[s])

    vmax = max(a_tgt.max(), a_ideal.max(), a_print.max())

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: amplitudes
    for ax, data, title in zip(axes[0],
                                [a_tgt, a_ideal, a_print],
                                ["Target", "Recon (phase-only)", "Recon (printable)"]):
        im = ax.imshow(data, origin="lower", extent=extent_roi,
                       cmap="inferno", aspect="equal", vmin=0, vmax=vmax)
        _add_colorbar(ax, im, "|p| (Pa)")
        ax.set_title(title)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    # Row 2: residuals
    for ax, data, title in zip(axes[1],
                                [a_tgt, a_ideal - a_tgt, a_print - a_tgt],
                                ["Target (ref)", "Ideal − Target", "Print − Target"]):
        if title == "Target (ref)":
            im = ax.imshow(np.angle(p_target[s]), origin="lower", extent=extent_roi,
                           cmap="twilight", aspect="equal", vmin=-np.pi, vmax=np.pi)
            _add_colorbar(ax, im, "Phase (rad)")
            ax.set_title("Target Phase")
        else:
            vlim = max(abs(data.min()), abs(data.max()), 1e-6)
            im = ax.imshow(data, origin="lower", extent=extent_roi,
                           cmap="RdBu_r", aspect="equal", vmin=-vlim, vmax=vlim)
            _add_colorbar(ax, im, "Δ|p| (Pa)")
            ax.set_title(title)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    fig.suptitle(f"Reconstruction Comparison (ROI) — D={diameter_mm:.0f} mm", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / f"reconstruction_D{diameter_mm:.0f}mm.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diameter_summary(diameter_results, save_dir):
    """Bar charts comparing metrics across diameters."""
    diams = [r["diameter_mm"] for r in diameter_results]
    energy_fracs = [r["energy_fraction"] for r in diameter_results]
    t_maxes = [r.get("t_max_mm", 0) for r in diameter_results]
    amp_corrs_ideal = [r.get("amp_correlation_ideal", 0) for r in diameter_results]
    amp_corrs_print = [r.get("amp_correlation_printable", 0) for r in diameter_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.bar(range(len(diams)), energy_fracs, tick_label=[f"{d:.0f}" for d in diams])
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Energy Fraction")
    ax.set_title("Aperture Energy Capture")

    ax = axes[0, 1]
    ax.bar(range(len(diams)), t_maxes, tick_label=[f"{d:.0f}" for d in diams])
    ax.axhline(20, color="r", ls="--", label="20 mm limit")
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Max Thickness (mm)")
    ax.set_title("Maximum Lens Thickness")
    ax.legend()

    ax = axes[1, 0]
    ax.bar(range(len(diams)), amp_corrs_ideal, tick_label=[f"{d:.0f}" for d in diams])
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Amplitude Correlation")
    ax.set_title("Reconstruction Quality (Phase-Only)")

    ax = axes[1, 1]
    ax.bar(range(len(diams)), amp_corrs_print, tick_label=[f"{d:.0f}" for d in diams])
    ax.set_xlabel("Diameter (mm)")
    ax.set_ylabel("Amplitude Correlation")
    ax.set_title("Reconstruction Quality (Printable)")

    fig.suptitle("Lens Diameter Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / "diameter_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def write_index_and_audit(cfg, study_cfg, params, embed_info,
                           diameter_results, best_diameter, save_dir):
    """Write INDEX.md and audit.md."""
    lam_mm = cfg.wavelength * 1e3

    # ── INDEX.md ─────────────────────────────────────────────────
    best_r = next(r for r in diameter_results if r["diameter_mm"] == best_diameter)

    index_md = f"""# C-Shape Backpropagation → Phase → Thickness Lens Design

**Generated**: {TS}

## Source

- **Chosen candidate**: best_{cfg.chosen_rank:02d} from `{Path(cfg.study_dir).name}`
- **Reference PNG**: `best/best_{cfg.chosen_rank:02d}_comparison.png`
- **Target reconstructed from**: saved NPZ parameters (not pixel-traced)
- **Candidate params**: r0/d={params['r0_rel']}, σr/d={params['sr_rel']}, σθ={params['sigma_theta']}, gap={params['gap_label']}, m=1, α={params['alpha']}

## Geometry

- **Target plane (z*)**: {cfg.z_star*1e3:.4f} mm
- **Lens plane (z=0)**: {cfg.z_lens*1e3:.1f} mm
- **Propagation distance**: {(cfg.z_star - cfg.z_lens)*1e3:.3f} mm = {(cfg.z_star - cfg.z_lens)/cfg.wavelength:.1f} λ
- **λ**: {lam_mm:.4f} mm
- **f**: {cfg.frequency_hz/1e6:.1f} MHz

## Material

- **Water**: c = {cfg.c_water:.0f} m/s, ρ = {cfg.rho_water:.0f} kg/m³
- **Lens (acrylic/resin)**: c = {cfg.c_lens:.0f} m/s, ρ = {cfg.rho_lens:.0f} kg/m³
- **Δk = k_lens − k_water**: {cfg.dk:.1f} rad/m
- **Full 2π thickness**: {cfg.full_2pi_thickness*1e3:.3f} mm

## Grid

- **Backprop grid**: {cfg.n_grid}×{cfg.n_grid}
- **Domain**: {embed_info['domain_half_m']*2e3:.1f}×{embed_info['domain_half_m']*2e3:.1f} mm
- **dx**: {embed_info['dx_m']*1e6:.1f} µm

## Lens Diameters Tested

| Diameter (mm) | Energy Capture | Max Thickness (mm) | Within Budget | Amp Corr (ideal) | Amp Corr (print) |
|--------------|----------------|--------------------|----|-------------|-------------|
"""
    for r in diameter_results:
        budget = "✅" if r.get("within_budget", True) else "❌"
        index_md += (f"| {r['diameter_mm']:.0f} | {r['energy_fraction']*100:.1f}% | "
                     f"{r.get('t_max_mm', 0):.3f} | {budget} | "
                     f"{r.get('amp_correlation_ideal', 0):.4f} | "
                     f"{r.get('amp_correlation_printable', 0):.4f} |\n")

    index_md += f"""
## Best Candidate

- **Diameter**: {best_diameter:.0f} mm
- **Energy capture**: {best_r['energy_fraction']*100:.1f}%
- **Max thickness**: {best_r.get('t_max_mm', 0):.3f} mm
- **Within 20 mm budget**: {'Yes' if best_r.get('within_budget', True) else 'No'}
- **Amplitude correlation (ideal)**: {best_r.get('amp_correlation_ideal', 0):.4f}
- **Amplitude correlation (printable)**: {best_r.get('amp_correlation_printable', 0):.4f}

## Smoothing & Fabrication

- **Gaussian σ**: {cfg.smooth_sigma_mm:.2f} mm
- **Max slope limit**: {cfg.max_slope_deg:.0f}°
- **Min feature**: {cfg.min_feature_mm:.2f} mm

## Thickness Mapping

```
t(x,y) = t_base + mod(φ, 2π) / |Δk|
t_base = 0.5 mm (structural floor)
|Δk| = |k_lens − k_water| = {abs(cfg.dk):.1f} rad/m
2π/|Δk| = {cfg.full_2pi_thickness*1e3:.3f} mm (one phase wrap)
```

## Files

- `config.json` — full configuration
- `figures/` — all diagnostic figures
- `lens_exports/` — thickness maps, heightmaps, phase data
- `npz/` — saved field arrays
- `audit.md` — step-by-step provenance

## Observations

The C-shape target field is highly localised (ROI ≈ {1.1*lam_mm:.2f}×{1.1*lam_mm:.2f} mm²),
meaning most of the information content is in low spatial frequencies.
The backpropagated lens-plane field is consequently spread over a large area,
making even moderate apertures capture most of the energy.

Because Δk is relatively small at 2 MHz (the lens material is much faster
than water), the full 2π thickness increment is {cfg.full_2pi_thickness*1e3:.3f} mm — very modest.
All lens designs comfortably stay within the 20 mm budget.

## Next Steps

1. If reconstruction fidelity is acceptable:
   → Export heightmap for 3D printer (STL generation)
   → Add lens mount geometry
2. If fidelity is poor:
   → Consider iterative phase retrieval (Gerchberg–Saxton)
   → Consider larger grid or finer dx
3. Validate with FEM forward simulation including the standing wave
"""

    with open(save_dir / "INDEX.md", "w") as f:
        f.write(index_md)

    # ── audit.md ─────────────────────────────────────────────────
    audit_md = f"""# Audit Trail

## Step 1: Target Reconstruction

- Source: `{Path(cfg.study_dir).name}/npz/best_{cfg.chosen_rank:02d}.npz`
- The target field was reconstructed from the saved normalised C-field `p_C`
  (complex, 400×400) recorded during the parametric sweep.
- The field was interpolated onto an enlarged {cfg.n_grid}×{cfg.n_grid} grid
  centred at origin, with domain half-width = {embed_info['domain_half_m']*1e3:.1f} mm.
- A cosine apodisation taper was applied at the ROI boundary to suppress
  ringing from hard truncation.
- The peak amplitude was scaled to {params['alpha']} × SW_peak ≈ 4.6 Pa,
  matching the combined-field convention from the study.

**Assumption**: The standing-wave field is NOT included in the target.
Only the C-shaped perturbation field is back-propagated. This is because
the lens is intended to generate the C-shape overlay, not the background
standing wave (which comes from the transducer array).

## Step 2: ASM Backpropagation

- Transfer function: H(kx,ky) = exp(−i kz Δz) with Δz = {(cfg.z_star-cfg.z_lens)*1e3:.3f} mm.
- Evanescent modes: excluded (set to zero). For Δz ≈ {(cfg.z_star-cfg.z_lens)/cfg.wavelength:.0f}λ,
  evanescent contributions decay by a factor of exp(−2πΔz/λ) ≈ 0.
- Zero-padding: pad_factor=2 (doubles each axis).
- The propagation distance is large ({(cfg.z_star-cfg.z_lens)/cfg.wavelength:.0f}λ), so the
  lens-plane field is significantly diffracted / spread compared to the target.

## Step 3: Circular Aperture

- Tested diameters: {[d for d in cfg.lens_diameters_mm]} mm.
- The aperture is centred at (0,0) which corresponds to the ROI centre
  at ({embed_info['cx_roi_m']*1e3:.3f}, {embed_info['cy_roi_m']*1e3:.3f}) mm in the FEM domain.

## Step 4: Phase-Only Projection

- The backpropagated field has both amplitude and phase variation.
- Phase-only projection discards the amplitude information, retaining
  only exp(i·arg(p_lens)) inside the aperture.
- This is the fundamental approximation of holographic lens design.

## Step 5: Thickness Conversion

- Formula: t = t_base + mod(φ, 2π) / |Δk|
- Δk = k_lens − k_water = {cfg.dk:.1f} rad/m
- t_base = 0.5 mm
- 2π/|Δk| = {cfg.full_2pi_thickness*1e3:.3f} mm

## Step 6: Fabrication Filtering

- Gaussian smoothing: σ = {cfg.smooth_sigma_mm:.2f} mm ({cfg.smooth_sigma_mm*1e3/cfg.wavelength:.1f}× wavelength)
- Slope limiting: max {cfg.max_slope_deg:.0f}° (tangent = {np.tan(np.radians(cfg.max_slope_deg)):.1f})
- These are conservative for modern resin printers (50–100 µm resolution,
  supports at 45°+ overhang angles).

## Warnings

- The C-shape target is very small compared to all tested apertures.
  Nearly 100% of energy is captured even by the 5 mm lens.
- The backpropagation distance (~8λ) means the phase-only approximation
  may lose significant amplitude structure.
- The forward reconstruction is the ground truth for what the lens
  actually produces — examine it carefully.
"""

    with open(save_dir / "audit.md", "w") as f:
        f.write(audit_md)

    print(f"[report] INDEX.md and audit.md written")


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    cfg = CFG
    print("=" * 72)
    print("C-Shape Backpropagation → Phase → Thickness Lens Design")
    print("=" * 72)
    print(f"  λ = {cfg.wavelength*1e3:.4f} mm,  f = {cfg.frequency_hz/1e6:.1f} MHz")
    print(f"  k_water = {cfg.k_water:.1f} rad/m,  k_lens = {cfg.k_lens:.1f} rad/m")
    print(f"  Δk = {cfg.dk:.1f} rad/m,  2π/|Δk| = {cfg.full_2pi_thickness*1e3:.3f} mm")
    print(f"  z* = {cfg.z_star*1e3:.4f} mm,  z_lens = {cfg.z_lens:.1f} mm")
    print()

    # ── 1. Load chosen C-shape ───────────────────────────────────
    print("─── Step 1: Load & Reconstruct Target ───")
    p_C_norm, p_comb, xg_orig, yg_orig, params, metrics, study_cfg = \
        load_selected_c_shape(cfg)

    p_target, xg, yg, embed_info = \
        reconstruct_target_field(p_C_norm, xg_orig, yg_orig, params, study_cfg, cfg)

    p_target = apodise_target_field(p_target, xg, yg, cfg)

    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]

    plot_target_field(p_target, xg, yg, FIG_DIR)
    np.savez_compressed(NPZ_DIR / "target_field.npz",
                        p_target=p_target, xg=xg, yg=yg)
    print()

    # ── 2. Backpropagate ─────────────────────────────────────────
    print("─── Step 2: ASM Backpropagation ───")
    p_lens = asm_backpropagate_to_lens(p_target, dx, dy, cfg)
    plot_lens_plane(p_lens, xg, yg, FIG_DIR)
    np.savez_compressed(NPZ_DIR / "lens_plane.npz", p_lens=p_lens)
    print()

    # ── 3–8. Sweep over diameters ────────────────────────────────
    print("─── Steps 3–8: Diameter Sweep ───")
    diameter_results = []
    all_masks = []
    diameters_mm = list(cfg.lens_diameters_mm)

    for d_mm in diameters_mm:
        print(f"\n{'─'*50}")
        print(f"  Lens Diameter = {d_mm:.0f} mm")
        print(f"{'─'*50}")

        d_m = d_mm * 1e-3
        mask = make_circular_aperture(xg, yg, d_m)
        all_masks.append(mask)

        # Aperture diagnostics
        ap_info = compute_aperture_diagnostics(p_lens, xg, yg, mask, d_mm)

        # Phase-only
        phase_raw, phase_wrapped, p_phase_only = \
            extract_phase_only_design(p_lens, mask)
        plot_phase_only(phase_raw, phase_wrapped, mask, xg, yg, d_mm, FIG_DIR)

        # Thickness
        thickness_raw, t_info = phase_to_thickness(phase_wrapped, mask, cfg)

        # Smoothing & slope limiting
        t_printable, slope_angle, fab_info = \
            smooth_and_limit_thickness(thickness_raw, mask, xg, yg, cfg)

        # Thickness diagnostics
        plot_thickness_diagnostics(thickness_raw, t_printable, slope_angle,
                                    mask, xg, yg, d_mm, FIG_DIR)

        # 3D rendering
        render_lens_3d(t_printable, xg, yg, mask, d_mm, FIG_DIR)

        # Forward validation
        p_recon_ideal, p_recon_print = \
            forward_validate_lens(p_phase_only, t_printable, mask, xg, yg, cfg)

        m_ideal = compute_reconstruction_metrics(
            p_target, p_recon_ideal, xg, yg, cfg, "ideal")
        m_print = compute_reconstruction_metrics(
            p_target, p_recon_print, xg, yg, cfg, "printable")

        # Reconstruction comparison plot
        plot_reconstruction_comparison(
            p_target, p_recon_ideal, p_recon_print,
            xg, yg, cfg, d_mm, FIG_DIR)

        # Export lens data
        np.savez_compressed(
            LENS_DIR / f"lens_D{d_mm:.0f}mm.npz",
            thickness_raw=thickness_raw,
            thickness_printable=t_printable,
            phase_raw=phase_raw,
            phase_wrapped=phase_wrapped,
            mask=mask, xg=xg, yg=yg,
            slope_angle=slope_angle)

        # Collect results
        result = {
            **ap_info,
            **t_info,
            **fab_info,
            **m_ideal,
            **m_print,
        }
        diameter_results.append(result)

    # ── Aperture comparison figure ───────────────────────────────
    plot_aperture_comparison(p_lens, xg, yg, all_masks, diameters_mm, FIG_DIR)

    # ── Diameter summary figure ──────────────────────────────────
    plot_diameter_summary(diameter_results, FIG_DIR)

    # ── Best diameter selection ──────────────────────────────────
    # Pick the diameter with the best printable reconstruction correlation,
    # provided it's within the thickness budget.
    valid = [r for r in diameter_results if r.get("within_budget", True)]
    if valid:
        best_r_sel = max(valid, key=lambda r: r.get("amp_correlation_printable", 0))
        best_d = best_r_sel["diameter_mm"]
    else:
        best_d = diameters_mm[0]
    print(f"\n[best] Recommended diameter: {best_d:.0f} mm "
          f"(best printable amp_corr = {best_r_sel.get('amp_correlation_printable', 0):.4f})")

    # ── Config ───────────────────────────────────────────────────
    config_out = {
        "timestamp": TS,
        "chosen_study": cfg.study_dir,
        "chosen_rank": cfg.chosen_rank,
        "candidate_params": params,
        "physics": {
            "frequency_hz": cfg.frequency_hz,
            "c_water": cfg.c_water,
            "c_lens": cfg.c_lens,
            "k_water": cfg.k_water,
            "k_lens": cfg.k_lens,
            "dk": cfg.dk,
            "wavelength_mm": cfg.wavelength * 1e3,
            "full_2pi_thickness_mm": cfg.full_2pi_thickness * 1e3,
        },
        "geometry": {
            "z_star_mm": cfg.z_star * 1e3,
            "z_lens_mm": cfg.z_lens * 1e3,
            "propagation_mm": (cfg.z_star - cfg.z_lens) * 1e3,
        },
        "grid": embed_info,
        "fabrication": {
            "smooth_sigma_mm": cfg.smooth_sigma_mm,
            "max_slope_deg": cfg.max_slope_deg,
            "min_feature_mm": cfg.min_feature_mm,
        },
        "diameters_tested_mm": diameters_mm,
        "best_diameter_mm": best_d,
        "diameter_results": diameter_results,
        "runtime_s": time.time() - t_start,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config_out, f, indent=2, default=str)

    # ── Metrics CSV ──────────────────────────────────────────────
    import csv
    csv_path = OUT_DIR / "metrics.csv"
    if diameter_results:
        keys = list(diameter_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(diameter_results)
    print(f"[save] metrics.csv")

    # ── Reports ──────────────────────────────────────────────────
    write_index_and_audit(cfg, study_cfg, params, embed_info,
                           diameter_results, best_d, OUT_DIR)

    print(f"\n{'='*72}")
    print(f"Done in {time.time()-t_start:.1f}s")
    print(f"Outputs: {OUT_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
