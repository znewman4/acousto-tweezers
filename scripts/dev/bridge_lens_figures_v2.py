#!/usr/bin/env python3
"""
Bridge lens design — direct backprop figures (v2).

No IASA iteration.  The lens phase is computed from a single backward ASM pass
through the target amplitude field.  This gives the "exact phase profile that
would produce that shape" without any iterative refinement.

Also generates:
  • bigger bridge variants (2× and 4× spatial scale)
  • 3-D oblique pyvista render of the lens STL

Output PNGs (all saved to OUT_DIR):
  bridge_backprop_1x.png  – 4-panel snapshot at native bridge scale
  bridge_backprop_2x.png  – 4-panel snapshot at 2× bridge scale
  bridge_backprop_4x.png  – 4-panel snapshot at 4× bridge scale
  bridge_backprop_1x_stl_render.png – 3-D oblique view of lens (1× scale)
  bridge_backprop_2x_stl_render.png – 3-D oblique view of lens (2× scale)
  bridge_backprop_4x_stl_render.png – 3-D oblique view of lens (4× scale)

Run:
    python scripts/dev/bridge_lens_figures_v2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    make_grid,
    propagate_asm,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_IASA_DIR = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
)

BRIDGE_PRESSURE_NPZ = _IASA_DIR / "bridge_pressure_fields_scaled2x.npz"
BRIDGE_IASA_NPZ     = _IASA_DIR / "bridge_inverse_replica_fields.npz"
OUT_DIR             = PROJECT_ROOT / "results" / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FREQUENCY_HZ        = 2_150_000.0
C_WATER             = 1480.0
C_LENS              = 2636.0
N_GRID              = 400
TRANSDUCER_DIAM_MM  = 20.0
FOCAL_MM            = 13.21309776965029
H_BASE_MM           = 1.0
SOURCE_PRESSURE_PA  = 0.05e6
PARTICLE_RADIUS_MM  = 0.05
PARTICLE_DENSITY    = 1050.0
PARTICLE_C_SOUND    = 2350.0
RHO_WATER           = 998.0
ETA_WATER           = 1.0e-3

# Scale multipliers for the bridge pattern
# 1× and 2× are sub-diffraction (corridor ≈ 1.5λ) → Fresnel focus.
# 4× puts corridor width at ~3λ → well resolvable.
EXTRA_SCALES = [4.0]

DPI        = 190
CMAP_THICK = "viridis"
CMAP_PHASE = "twilight"
CMAP_PRESS = "hot"
CMAP_GORKOV = "RdBu_r"


# ─────────────────────────────────────────────────────────────────────────────
# Build config
# ─────────────────────────────────────────────────────────────────────────────
def _build_cfg() -> ReplicaConfig:
    return ReplicaConfig(
        frequency_hz=FREQUENCY_HZ,
        c_water=C_WATER,
        c_lens=C_LENS,
        transducer_diameter_mm=TRANSDUCER_DIAM_MM,
        focal_distance_mm=FOCAL_MM,
        n_grid=N_GRID,
        h_base_mm=H_BASE_MM,
        n_iter=1,
        source_pressure_pa=SOURCE_PRESSURE_PA,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style(ax, fontsize: int = 8) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize)


def _gorkov(
    p: np.ndarray,
    dx: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gor'kov potential and x/y force for a 30-µm polystyrene bead in water."""
    rho_w = RHO_WATER
    c_w   = C_WATER
    rho_p = PARTICLE_DENSITY
    c_p   = PARTICLE_C_SOUND
    a_m   = PARTICLE_RADIUS_MM * 1e-3

    kappa_w = 1.0 / (rho_w * c_w ** 2)
    kappa_p = 1.0 / (rho_p * c_p ** 2)
    f1 = 1.0 - kappa_p / kappa_w
    f2 = 2.0 * (rho_p - rho_w) / (2.0 * rho_p + rho_w)
    Vp = (4.0 / 3.0) * np.pi * a_m ** 3

    dp_dy, dp_dx = np.gradient(p, dx, dx)
    vx = -(1.0 / (1j * omega * rho_w)) * dp_dx
    vy = -(1.0 / (1j * omega * rho_w)) * dp_dy
    v2 = np.abs(vx) ** 2 + np.abs(vy) ** 2
    p2 = np.abs(p) ** 2

    U   = Vp * (f1 * p2 / (4.0 * rho_w * c_w ** 2) - 3.0 * f2 * rho_w * v2 / 8.0)
    dU_dy, dU_dx = np.gradient(U, dx, dx)
    return U, -dU_dx, -dU_dy


def _lens_thickness_mm(
    lens_field: np.ndarray,
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
) -> np.ndarray:
    phi = np.mod(np.angle(lens_field), 2.0 * np.pi)
    t   = cfg.h_base_m + cfg.h_max_m * (phi / (2.0 * np.pi))
    t[~aperture_mask] = np.nan
    return t * 1e3


def _propagate_scaled(
    lens_field: np.ndarray,
    cfg: ReplicaConfig,
    dx: float,
    aperture_mask: np.ndarray,
) -> np.ndarray:
    p = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)
    scale = SOURCE_PRESSURE_PA / (
        np.sqrt(np.mean(np.abs(lens_field[aperture_mask]) ** 2)) + 1e-12
    )
    return p * scale


def _build_target_at_scale(
    p_bridge: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    iasa_x: np.ndarray,
    iasa_y: np.ndarray,
    base_bridge_scale: float,
    extra_scale: float,
) -> np.ndarray:
    """
    Re-interpolate the bridge pressure field onto the IASA grid using
    base_bridge_scale * extra_scale.  Larger extra_scale → physically bigger
    bridge footprint in the aperture.
    """
    x_center_m = 0.5 * (x_full[0] + x_full[-1])
    y_center_m = 0.5 * (y_full[0] + y_full[-1])
    xc = x_full - x_center_m
    yc = y_full - y_center_m

    amp = np.abs(p_bridge)
    interp = RegularGridInterpolator(
        (yc, xc), amp,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    total_scale = base_bridge_scale * extra_scale
    # IASA sample coords remapped into the bridge's centred frame.
    # Dividing by total_scale zooms the bridge *out* onto the aperture.
    ys, xs = np.meshgrid(iasa_y, iasa_x, indexing="ij")  # (N,N)
    pts = np.column_stack([
        ys.ravel() / total_scale,
        xs.ravel() / total_scale,
    ])
    target_amp = interp(pts).reshape(N_GRID, N_GRID)
    return target_amp.astype(float)


def _direct_backprop_lens(
    target_amp: np.ndarray,
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    dx: float,
) -> np.ndarray:
    """
    Single-step hologram: back-ASM the real amplitude target → extract phase.
    No iteration.  Returns a phase-only lens field (|lens_field|=1 inside aperture).

    Random initial phase avoids a zero-phase target locking the backprop
    into a pure Fresnel lens.
    """
    rng = np.random.default_rng(seed=42)
    target_complex = target_amp * np.exp(
        1j * rng.uniform(0.0, 2.0 * np.pi, target_amp.shape)
    )
    backprop = propagate_asm(target_complex, cfg.k_water, -cfg.focal_distance_m, dx)
    lens = np.exp(1j * np.angle(backprop))
    lens[~aperture_mask] = 0.0
    return lens


# ─────────────────────────────────────────────────────────────────────────────
# Figure generators
# ─────────────────────────────────────────────────────────────────────────────
def fig_backprop_snapshot(
    lens_field: np.ndarray,
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    dx: float,
    iasa_x: np.ndarray,
    iasa_y: np.ndarray,
    extra_scale: float,
    out_path: Path,
) -> np.ndarray:
    """
    4-panel figure (same layout as the IASA snapshots, but labelled 'Direct Backprop'):
      Panel 0 – lens thickness (viridis)
      Panel 1 – wrapped phase (twilight)
      Panel 2 – pressure at focus (hot) zoomed to ±6mm
      Panel 3 – Gor'kov potential (RdBu_r) zoomed to ±6mm

    Returns the complex pressure field (for other uses).
    """
    omega     = 2.0 * np.pi * FREQUENCY_HZ
    ext_mm    = [iasa_x[0]*1e3, iasa_x[-1]*1e3, iasa_y[0]*1e3, iasa_y[-1]*1e3]
    r_ap      = TRANSDUCER_DIAM_MM / 2.0
    theta_c   = np.linspace(0, 2 * np.pi, 300)

    thickness_mm = _lens_thickness_mm(lens_field, cfg, aperture_mask)
    phase        = np.mod(np.angle(lens_field), 2.0 * np.pi)
    phase[~aperture_mask] = np.nan

    p_fwd  = _propagate_scaled(lens_field, cfg, dx, aperture_mask)
    p_kpa  = np.abs(p_fwd) * 1e-3

    U, _, _ = _gorkov(p_fwd, dx, omega)
    U_aJ    = U * 1e18
    uv      = float(np.percentile(np.abs(U_aJ[aperture_mask]), 99.0)) if np.any(aperture_mask) else 1.0
    uv      = max(uv, 1e-12)

    # Pressure zoom: cover the bridge area which is now wider at larger scales
    zoom_mm = min(10.0, r_ap * extra_scale * 1.3)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.suptitle(
        f"Bridge — direct single-step backprop  ({extra_scale:.0f}× bridge scale)",
        fontsize=11, fontweight="bold",
    )

    # Panel 0: thickness
    ax = axes[0]
    im0 = ax.imshow(
        thickness_mm, origin="lower", extent=ext_mm,
        cmap=CMAP_THICK, aspect="equal",
        vmin=float(np.nanmin(thickness_mm)), vmax=float(np.nanmax(thickness_mm)),
    )
    plt.colorbar(im0, ax=ax, label="t [mm]", fraction=0.046, pad=0.02)
    ax.set_title("Lens Thickness")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.plot(r_ap * np.cos(theta_c), r_ap * np.sin(theta_c), "r--", lw=0.7, alpha=0.6)
    _style(ax)

    # Panel 1: phase
    ax = axes[1]
    im1 = ax.imshow(
        phase, origin="lower", extent=ext_mm,
        cmap=CMAP_PHASE, vmin=0, vmax=2.0 * np.pi, aspect="equal",
    )
    plt.colorbar(im1, ax=ax, label="phase [rad]", fraction=0.046, pad=0.02)
    ax.set_title("Hologram Phase Map")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.plot(r_ap * np.cos(theta_c), r_ap * np.sin(theta_c), "w--", lw=0.7, alpha=0.6)
    _style(ax)

    # Panel 2: pressure
    ax = axes[2]
    vp = float(np.percentile(p_kpa[aperture_mask], 99.5)) if np.any(aperture_mask) else p_kpa.max()
    vp = max(vp, 1e-9)
    im2 = ax.imshow(
        p_kpa, origin="lower", extent=ext_mm,
        cmap=CMAP_PRESS, vmin=0, vmax=vp, aspect="equal",
    )
    plt.colorbar(im2, ax=ax, label="|p| [kPa]", fraction=0.046, pad=0.02)
    ax.set_title("Pressure at Focus")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_xlim(-zoom_mm, zoom_mm); ax.set_ylim(-zoom_mm, zoom_mm)
    _style(ax)

    # Panel 3: Gor'kov
    ax = axes[3]
    im3 = ax.imshow(
        U_aJ, origin="lower", extent=ext_mm,
        cmap=CMAP_GORKOV, vmin=-uv, vmax=uv, aspect="equal",
    )
    plt.colorbar(im3, ax=ax, label="U [aJ]", fraction=0.046, pad=0.02)
    ax.set_title("Gor'kov Potential")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_xlim(-zoom_mm, zoom_mm); ax.set_ylim(-zoom_mm, zoom_mm)
    _style(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return p_fwd


def fig_stl_render(
    lens_field: np.ndarray,
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    iasa_x: np.ndarray,
    iasa_y: np.ndarray,
    extra_scale: float,
    out_path: Path,
    stl_save_path: Path | None = None,
) -> None:
    """
    Build a proper 3-D pyvista mesh from the lens thickness map and render an
    oblique perspective view offscreen.  Optionally also saves the STL file
    (at grid_stride=2 for reasonable size).
    """
    import pyvista as pv

    # -- Build thickness array ------------------------------------------------
    phi       = np.mod(np.angle(lens_field), 2.0 * np.pi)
    thickness = cfg.h_base_m + cfg.h_max_m * (phi / (2.0 * np.pi))
    thickness[~aperture_mask] = 0.0          # collapse outside: no floating cap

    # Downsample to keep mesh manageable (stride=2 → 200×200)
    stride = 2
    idx_s  = np.arange(0, N_GRID, stride, dtype=int)
    # ensure last point is included
    if idx_s[-1] != N_GRID - 1:
        idx_s = np.append(idx_s, N_GRID - 1)

    xg_s  = np.outer(np.ones(len(idx_s)), iasa_x[idx_s])    # (M,M) row = const y
    # meshgrid: we want (row=y, col=x)
    xs_1d, ys_1d = iasa_x[idx_s], iasa_y[idx_s]
    xg_s, yg_s   = np.meshgrid(xs_1d, ys_1d)   # (M,M)
    t_s           = thickness[np.ix_(idx_s, idx_s)]

    n = xg_s.shape[0]

    # StructuredGrid expects (n, m, 2) arrays: first layer z=0, second z=t
    x3 = np.stack([xg_s, xg_s], axis=2)  # (M,M,2)
    y3 = np.stack([yg_s, yg_s], axis=2)
    z3 = np.stack([np.zeros_like(t_s), t_s], axis=2)

    grid = pv.StructuredGrid(x3, y3, z3)

    # Extract outer surface, triangulate, clean
    surf = grid.extract_surface().triangulate().clean()

    # Clip to circular aperture to remove squared corners
    r_clip = 0.499 * TRANSDUCER_DIAM_MM * 1e-3  # just inside the aperture edge
    h_clip  = 2.0 * (cfg.h_base_m + cfg.h_max_m)
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.5 * h_clip),
        direction=(0.0, 0.0, 1.0),
        radius=r_clip,
        height=h_clip * 1.1,
        resolution=256,
    ).triangulate().clean()

    try:
        clipped = surf.boolean_intersection(cylinder).triangulate().clean()
    except Exception:
        clipped = surf.clip_surface(cylinder, invert=False).triangulate().clean()

    if clipped.n_cells == 0:
        clipped = surf

    # Optionally save STL
    if stl_save_path is not None:
        try:
            from pymeshfix import MeshFix
            fixer = MeshFix(clipped)
            fixer.repair(joincomp=True, remove_smallest_components=False)
            repaired = fixer.mesh.triangulate().clean()
            if repaired.n_cells == 0:
                repaired = clipped
        except Exception:
            repaired = clipped
        repaired.save(str(stl_save_path))
        print(f"  Saved STL: {stl_save_path.name}  ({repaired.n_cells} cells)")

    # -- Render ---------------------------------------------------------------
    # Filter to top-facing cells so side walls don't pollute the colour map
    surf_all = clipped.extract_surface()
    normals = surf_all.compute_normals(cell_normals=True, point_normals=False)
    top_mask = normals["Normals"][:, 2] > 0.5
    if np.any(top_mask):
        top_layer = surf_all.extract_cells(np.where(top_mask)[0])
    else:
        top_layer = surf_all

    cam_r  = TRANSDUCER_DIAM_MM * 1e-3 * 2.5   # ~50 mm away
    cam_el = 35.0                                # elevation degrees
    cam_az = 45.0                                # azimuth degrees
    cam_x  = cam_r * np.cos(np.radians(cam_az)) * np.cos(np.radians(cam_el))
    cam_y  = cam_r * np.sin(np.radians(cam_az)) * np.cos(np.radians(cam_el))
    cam_z  = cam_r * np.sin(np.radians(cam_el))

    focal_z = 0.5 * float(cfg.h_base_m + 0.5 * cfg.h_max_m)

    try:
        pv.start_xvfb()
    except Exception:
        pass

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1200))
    plotter.set_background("white")

    plotter.add_mesh(
        top_layer,
        scalars=top_layer.points[:, 2] * 1e3,   # colour by thickness in mm
        cmap=CMAP_THICK,
        show_scalar_bar=False,
        smooth_shading=True,
        show_edges=False,
        lighting=True,
    )

    # Scalar bar
    plotter.add_scalar_bar(
        title="Thickness [mm]",
        n_labels=4,
        fmt="%.2f",
        position_x=0.85,
        position_y=0.05,
        width=0.12,
        height=0.6,
        label_font_size=22,
        title_font_size=18,
    )

    # Add outline circle at z=0 for spatial reference
    theta  = np.linspace(0, 2 * np.pi, 256)
    r_vis  = r_clip
    circle_pts = np.column_stack([
        r_vis * np.cos(theta),
        r_vis * np.sin(theta),
        np.zeros(256),
    ])
    lines = pv.Spline(circle_pts, 256).tube(radius=r_vis * 0.005)
    plotter.add_mesh(lines, color="black", opacity=0.6)

    plotter.camera_position = [
        (cam_x, cam_y, cam_z),
        (0.0, 0.0, focal_z),
        (0.0, 0.0, 1.0),
    ]

    title = f"Holographic lens (direct backprop, {extra_scale:.0f}× bridge scale)"
    plotter.add_title(title, font_size=14, color="black")

    img = plotter.screenshot(str(out_path), return_img=True)
    plotter.close()

    print(f"  Saved 3D render: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("BRIDGE LENS FIGURES V2  —  direct backpropagation (no IASA)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    bp       = np.load(BRIDGE_PRESSURE_NPZ)
    x_full   = bp["x_full"].astype(float)
    y_full   = bp["y_full"].astype(float)
    p_bridge = bp["p_bridge_effective_full"].astype(complex)

    ir             = np.load(BRIDGE_IASA_NPZ)
    aperture_mask  = ir["aperture_mask"].astype(bool)

    cfg = _build_cfg()
    iasa_x, iasa_y, _, _, _, _, _, dx_iasa = make_grid(cfg)

    # Base bridge scale (same computation as bridge_lens_figures.py)
    x_center_m   = 0.5 * (x_full[0] + x_full[-1])
    y_center_m   = 0.5 * (y_full[0] + y_full[-1])
    xc           = x_full - x_center_m
    yc           = y_full - y_center_m
    field_half_c = max(float(np.abs(xc).max()), float(np.abs(yc).max()))
    aperture_radius_m   = TRANSDUCER_DIAM_MM * 0.5e-3
    base_bridge_scale   = aperture_radius_m / max(field_half_c, 1e-12)
    print(f"  Base bridge scale: {base_bridge_scale:.4f}x  (fills aperture at 1×)")

    # ── Loop over scale variants ──────────────────────────────────────────
    for extra_scale in EXTRA_SCALES:
        label = f"{extra_scale:.0f}x"
        print(f"\n[{label}] Building target, direct backprop, and figures...")

        # Rebuild target amplitude at this scale
        target_amp = _build_target_at_scale(
            p_bridge, x_full, y_full,
            iasa_x, iasa_y,
            base_bridge_scale, extra_scale,
        )
        print(
            f"  target max={target_amp.max():.2f} Pa, "
            f"non-zero fraction={np.mean(target_amp > 0.01):.4f}"
        )

        # Single-step direct backpropagation
        lens_bp = _direct_backprop_lens(target_amp, cfg, aperture_mask, dx_iasa)

        # 4-panel figure
        p_fwd = fig_backprop_snapshot(
            lens_bp, cfg, aperture_mask, dx_iasa,
            iasa_x, iasa_y, extra_scale,
            OUT_DIR / f"bridge_backprop_{label}.png",
        )

        # 3-D STL render
        stl_path = OUT_DIR / f"bridge_backprop_{label}_lens.stl"
        fig_stl_render(
            lens_bp, cfg, aperture_mask,
            iasa_x, iasa_y,
            extra_scale,
            OUT_DIR / f"bridge_backprop_{label}_stl_render.png",
            stl_save_path=stl_path,
        )

    print("\n" + "=" * 70)
    print("All done!  Files written to:", OUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
