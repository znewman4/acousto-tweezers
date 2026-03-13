#!/usr/bin/env python3
"""
Phase 3.2-3.4 — Local hybrid overlay study at z*.

What this script does:
  1) Reuse cached FEM standing-wave field at z* (no FEM rerun)
  2) Build a calibrated ASM vortex perturbation at z*
  3) Combine fields with alpha / psi sweep
  4) Quantify A->B push vs B / neighbour disturbance trade-off
  5) Save local overlay diagnostics and heatmaps

Outputs are written to:
  results/deliverables/overlay_local/
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.fem_cache_utils import (
    load_fem_cache,
    interpolate_slice,
    gorkov_grid_2d,
    default_particle_params,
    OMEGA,
    RHO0,
    C_WATER,
    LAM,
)
from scripts.lib.asm_utils import (
    make_grid_from_fem,
    make_lens_phase,
    make_cshape_mask,
    propagate_asm,
)
from scripts.lib.overlay_utils import (
    validate_grid_consistency,
    combine_fields,
    scale_field_to_peak,
    bilinear_sample_vector,
    choose_adjacent_trap_pair,
    select_neighbour_traps,
)

OUT = PROJECT_ROOT / "results" / "deliverables" / "overlay_local"
OUT.mkdir(parents=True, exist_ok=True)

CALIBRATION_JSON = OUT / "overlay_calibration_summary.json"
TRAP_JSON = PROJECT_ROOT / "results" / "deliverables" / "trap_map" / "trap_data.json"

# Grids / sweep parameters
N_GRID = 400
ALPHA_VALUES = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
PSI_VALUES = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
PSI_LABELS = ["0", "pi/2", "pi", "3pi/2"]

# Overlay perturbation scaling (sets absolute perturbation level in Pa)
PERTURB_PEAK_RATIO = 0.30   # target |p_asm|_max ~ 30% of SW peak

# Local region around A/B midpoint for detailed maps
LOCAL_HALF_WINDOW = 1.0e-3

# Vortex / C-shape defaults
CHARGE = 1
R_AP = 2.5e-3
CSHAPE_THICKNESS = 0.14e-3
CSHAPE_GAP_WIDTH = 0.40
CSHAPE_BETA = 1.0


# ───────────────────────────────────────────────────────────────────
def _mm(v: float) -> float:
    return float(v * 1e3)


def _load_traps() -> np.ndarray:
    if not TRAP_JSON.exists():
        raise FileNotFoundError(f"Trap data missing: {TRAP_JSON}")
    d = json.loads(TRAP_JSON.read_text())
    traps = d.get("traps", [])
    if len(traps) < 2:
        raise ValueError("trap_data.json has too few traps")

    traps_m = np.array([[t["x_mm"] * 1e-3, t["y_mm"] * 1e-3] for t in traps], dtype=float)
    return traps_m


def _load_calibrated_vortex_params() -> dict:
    """Read selected vortex params from Phase 3.1 summary (fallback safe defaults)."""
    if CALIBRATION_JSON.exists():
        d = json.loads(CALIBRATION_JSON.read_text())
        sel = d.get("selected_vortex", {})
        if sel:
            return {
                "waist_mm": float(sel["waist_mm"]),
                "axicon_angle_deg": float(sel["axicon_angle_deg"]),
                "ring_radius_zstar_mm": float(sel["ring_radius_zstar_mm"]),
            }

    # Fallback if calibration script was not run yet
    return {
        "waist_mm": 1.8,
        "axicon_angle_deg": 16.0,
        "ring_radius_zstar_mm": float("nan"),
    }


def _build_vortex_bg_field_at_zstar(
    XX: np.ndarray,
    YY: np.ndarray,
    dx: float,
    dy: float,
    z_star: float,
    waist_mm: float,
    angle_deg: float,
) -> np.ndarray:
    """Bessel-Gauss vortex (l=1) propagated to z*."""
    waist = waist_mm * 1e-3

    cx = 0.5 * (XX.min() + XX.max())
    cy = 0.5 * (YY.min() + YY.max())
    r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)

    amp = np.exp(-(r ** 2) / (waist ** 2))
    amp[r > R_AP] = 0.0

    phi = make_lens_phase(
        XX,
        YY,
        family="axicon",
        aperture_radius=R_AP,
        axicon_angle_deg=angle_deg,
        charge=CHARGE,
    )
    source = amp * np.exp(-1j * phi)
    return propagate_asm(source, dx, dy, wavelength=LAM, z=z_star)


def _build_cshape_field_at_zstar(
    XX: np.ndarray,
    YY: np.ndarray,
    dx: float,
    dy: float,
    z_star: float,
    gap_angle: float,
) -> np.ndarray:
    """C-shape reference perturbation propagated to z*."""
    source = make_cshape_mask(
        XX,
        YY,
        radius=0.50e-3,
        gap_angle=gap_angle,
        thickness=CSHAPE_THICKNESS,
        charge=CHARGE,
        gap_width=CSHAPE_GAP_WIDTH,
        beta=CSHAPE_BETA,
    )

    cx = 0.5 * (XX.min() + XX.max())
    cy = 0.5 * (YY.min() + YY.max())
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    source = source * (rr <= R_AP)

    return propagate_asm(source, dx, dy, wavelength=LAM, z=z_star)


def _line_profile_delta_f_parallel(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fx_sw: np.ndarray,
    Fy_sw: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    e_ab: np.ndarray,
    n_pts: int = 41,
):
    """Return sampled delta F_parallel profile along A->B."""
    t = np.linspace(0.0, 1.0, n_pts)
    pts = A[None, :] + t[:, None] * (B - A)[None, :]

    fpar = []
    fpar_sw = []
    for pt in pts:
        f = bilinear_sample_vector(Fx, Fy, xg, yg, pt)
        f_sw = bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, pt)
        fpar.append(float(np.dot(f, e_ab)))
        fpar_sw.append(float(np.dot(f_sw, e_ab)))

    fpar = np.array(fpar)
    fpar_sw = np.array(fpar_sw)
    return t, fpar, fpar_sw, (fpar - fpar_sw)


def _evaluate_combo(
    p_sw: np.ndarray,
    p_vortex: np.ndarray,
    alpha: float,
    psi: float,
    xg: np.ndarray,
    yg: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    e_ab: np.ndarray,
    neigh_idx: np.ndarray,
    traps_m: np.ndarray,
    Fx_sw: np.ndarray,
    Fy_sw: np.ndarray,
    gorkov_args: dict,
) -> dict:
    """Compute local manipulation metrics for one alpha/psi point."""
    p_comb = combine_fields(p_sw, p_vortex, alpha=alpha, psi=psi)

    U, Fx, Fy = gorkov_grid_2d(
        p_comb,
        gorkov_args["dx"],
        gorkov_args["dy"],
        gorkov_args["omega"],
        gorkov_args["rho0"],
        gorkov_args["c0"],
        gorkov_args["a"],
        gorkov_args["f1"],
        gorkov_args["f2"],
    )

    F_A = bilinear_sample_vector(Fx, Fy, xg, yg, A)
    F_B = bilinear_sample_vector(Fx, Fy, xg, yg, B)
    F_A_sw = bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, A)
    F_B_sw = bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, B)

    A_push = float(np.dot(F_A - F_A_sw, e_ab))
    B_along_delta = float(np.dot(F_B - F_B_sw, e_ab))
    B_push_away = float(max(0.0, B_along_delta))
    B_disturb = float(np.linalg.norm(F_B - F_B_sw))

    neigh_d = []
    for i in neigh_idx:
        Fi = bilinear_sample_vector(Fx, Fy, xg, yg, traps_m[i])
        Fi_sw = bilinear_sample_vector(Fx_sw, Fy_sw, xg, yg, traps_m[i])
        neigh_d.append(float(np.linalg.norm(Fi - Fi_sw)))
    neigh_d = np.array(neigh_d, dtype=float)
    neigh_rms = float(np.sqrt(np.mean(neigh_d ** 2))) if len(neigh_d) else 0.0
    neigh_max = float(np.max(neigh_d)) if len(neigh_d) else 0.0

    t, fpar, fpar_sw, delta_fpar = _line_profile_delta_f_parallel(
        Fx, Fy, Fx_sw, Fy_sw, xg, yg, A, B, e_ab
    )
    corridor_delta_mean = float(np.mean(delta_fpar))
    corridor_forward_fraction = float(np.mean(delta_fpar > 0.0))

    # High score = strong A->B push with low B/neighbour disruption.
    tradeoff = float(A_push / (1e-30 + B_push_away + neigh_rms))

    return {
        "alpha": alpha,
        "psi": psi,
        "A_push_toward_B": A_push,
        "B_push_away": B_push_away,
        "B_disturb_norm": B_disturb,
        "neighbour_rms_disturb": neigh_rms,
        "neighbour_max_disturb": neigh_max,
        "corridor_delta_mean": corridor_delta_mean,
        "corridor_forward_fraction": corridor_forward_fraction,
        "tradeoff_score": tradeoff,
        "line_t": t,
        "line_delta_fpar": delta_fpar,
        "p_comb": p_comb,
        "U": U,
        "Fx": Fx,
        "Fy": Fy,
    }


def _plot_traps(ax, traps_m, idx_A, idx_B, neigh_idx=None):
    if neigh_idx is None:
        neigh_idx = []

    ax.scatter(traps_m[:, 0] * 1e3, traps_m[:, 1] * 1e3,
               s=15, facecolors="none", edgecolors="w", lw=0.7, alpha=0.7)

    if len(neigh_idx):
        pts = traps_m[neigh_idx]
        ax.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3,
                   s=36, marker="s", facecolors="none", edgecolors="cyan", lw=1.0)

    ax.scatter(traps_m[idx_A, 0] * 1e3, traps_m[idx_A, 1] * 1e3,
               s=80, c="red", marker="o", edgecolors="k", lw=0.8, label="A")
    ax.scatter(traps_m[idx_B, 0] * 1e3, traps_m[idx_B, 1] * 1e3,
               s=80, c="deepskyblue", marker="o", edgecolors="k", lw=0.8, label="B")


def _heatmap_from_metrics(metrics, key, alphas, psis):
    mat = np.full((len(alphas), len(psis)), np.nan, dtype=float)
    for m in metrics:
        ia = alphas.index(m["alpha"])
        ip = psis.index(m["psi"])
        mat[ia, ip] = m[key]
    return mat


# ───────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 64)
    print("Phase 3.2-3.4 — Local overlay study at z*")
    print("=" * 64)

    # 1) FEM standing-wave slice at z*
    cache = load_fem_cache()
    z_star = float(cache["z_star"])
    sl = interpolate_slice(cache, z=z_star, n_grid=N_GRID)
    p_sw = sl["p_grid"]
    xg = sl["xg"]
    yg = sl["yg"]
    dx = float(sl["dx"])
    dy = float(sl["dy"])

    sw_peak = float(np.max(np.abs(p_sw)))
    print(f"z*={_mm(z_star):.3f} mm  SW peak={sw_peak:.3f} Pa")

    # 2) Traps and local A/B scenario
    traps_m = _load_traps()
    domain_center = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])])

    pair = choose_adjacent_trap_pair(traps_m, target_spacing=LAM / 2.0, domain_center=domain_center)
    idx_A = int(pair["idx_A"])
    idx_B = int(pair["idx_B"])
    A = traps_m[idx_A]
    B = traps_m[idx_B]
    e_ab = pair["e_AB"]
    midpoint = pair["midpoint"]
    d_ab = float(pair["d_AB"])

    neigh_idx = select_neighbour_traps(
        traps_m,
        idx_A,
        idx_B,
        midpoint,
        radius=1.2 * LAM,
        min_count=6,
    )

    print(f"Selected pair: A={idx_A}, B={idx_B}")
    print(f"  A=({_mm(A[0]):.3f}, {_mm(A[1]):.3f}) mm")
    print(f"  B=({_mm(B[0]):.3f}, {_mm(B[1]):.3f}) mm")
    print(f"  d_AB={_mm(d_ab):.3f} mm ({d_ab / LAM:.3f} lambda)")
    print(f"  neighbours used in disturbance metric: {len(neigh_idx)}")

    # 3) Build calibrated ASM perturbations on FEM particle-plane grid
    grid_asm = make_grid_from_fem(cache, nx=N_GRID, ny=N_GRID)
    XX, YY = grid_asm["XX"], grid_asm["YY"]

    # Sanity check: ASM and FEM grids must align exactly for field overlay.
    _ = validate_grid_consistency(
        p_sw,
        p_sw,
        xg,
        yg,
        grid_asm["x"],
        grid_asm["y"],
    )

    params = _load_calibrated_vortex_params()
    print(
        "Using calibrated vortex params: waist={:.2f} mm, angle={:.1f} deg".format(
            params["waist_mm"], params["axicon_angle_deg"]
        )
    )

    p_vortex_raw = _build_vortex_bg_field_at_zstar(
        XX,
        YY,
        dx,
        dy,
        z_star=z_star,
        waist_mm=params["waist_mm"],
        angle_deg=params["axicon_angle_deg"],
    )

    # C-shape optional comparison: orient gap along A->B direction.
    gap_angle = float(np.arctan2(e_ab[1], e_ab[0]))
    p_cshape_raw = _build_cshape_field_at_zstar(
        XX,
        YY,
        dx,
        dy,
        z_star=z_star,
        gap_angle=gap_angle,
    )

    p_target_peak = PERTURB_PEAK_RATIO * sw_peak
    p_vortex = scale_field_to_peak(p_vortex_raw, p_target_peak)
    p_cshape = scale_field_to_peak(p_cshape_raw, p_target_peak)

    print(
        f"Perturbation scaling: target peak={p_target_peak:.3f} Pa "
        f"({PERTURB_PEAK_RATIO*100:.0f}% of SW peak)"
    )
    print(
        f"  vortex peak at z*={np.max(np.abs(p_vortex)):.3f} Pa, "
        f"cshape peak at z*={np.max(np.abs(p_cshape)):.3f} Pa"
    )

    # 4) Baseline SW Gor'kov force field
    ppar = default_particle_params()
    U_sw, Fx_sw, Fy_sw = gorkov_grid_2d(
        p_sw,
        dx,
        dy,
        OMEGA,
        RHO0,
        C_WATER,
        ppar["a"],
        ppar["f1"],
        ppar["f2"],
    )

    gorkov_args = {
        "dx": dx,
        "dy": dy,
        "omega": OMEGA,
        "rho0": RHO0,
        "c0": C_WATER,
        "a": ppar["a"],
        "f1": ppar["f1"],
        "f2": ppar["f2"],
    }

    # 5) alpha/psi sweep for calibrated vortex overlay
    print("\nRunning alpha/psi sweep ...")
    metrics = []
    for alpha in ALPHA_VALUES:
        for psi in PSI_VALUES:
            m = _evaluate_combo(
                p_sw,
                p_vortex,
                alpha,
                psi,
                xg,
                yg,
                A,
                B,
                e_ab,
                neigh_idx,
                traps_m,
                Fx_sw,
                Fy_sw,
                gorkov_args,
            )
            metrics.append(m)
            print(
                "  alpha={:.2f} psi={:.2f}pi  A_push={:.3e}  B_away={:.3e}  "
                "N_rms={:.3e}  score={:.3e}".format(
                    alpha,
                    psi / np.pi,
                    m["A_push_toward_B"],
                    m["B_push_away"],
                    m["neighbour_rms_disturb"],
                    m["tradeoff_score"],
                )
            )

    # Pick best trade-off among cases that actually push A toward B.
    good = [m for m in metrics if m["A_push_toward_B"] > 0.0]
    best = max(good, key=lambda m: m["tradeoff_score"]) if good else max(metrics, key=lambda m: m["tradeoff_score"])

    best_alpha = best["alpha"]
    best_psi = best["psi"]
    print(
        "\nBest trade-off point: alpha={:.2f}, psi={:.2f}pi, score={:.3e}".format(
            best_alpha,
            best_psi / np.pi,
            best["tradeoff_score"],
        )
    )

    # Build comparison point for C-shape at same alpha/psi.
    m_cshape = _evaluate_combo(
        p_sw,
        p_cshape,
        best_alpha,
        best_psi,
        xg,
        yg,
        A,
        B,
        e_ab,
        neigh_idx,
        traps_m,
        Fx_sw,
        Fy_sw,
        gorkov_args,
    )

    # 6) Save scalar sweep tables
    csv_rows = []
    for m in metrics:
        csv_rows.append({
            "alpha": m["alpha"],
            "psi": m["psi"],
            "A_push_toward_B": m["A_push_toward_B"],
            "B_push_away": m["B_push_away"],
            "B_disturb_norm": m["B_disturb_norm"],
            "neighbour_rms_disturb": m["neighbour_rms_disturb"],
            "neighbour_max_disturb": m["neighbour_max_disturb"],
            "corridor_delta_mean": m["corridor_delta_mean"],
            "corridor_forward_fraction": m["corridor_forward_fraction"],
            "tradeoff_score": m["tradeoff_score"],
        })

    csv_path = OUT / "overlay_tradeoff_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Saved {csv_path}")

    # 7) Figures
    x_mm = xg * 1e3
    y_mm = yg * 1e3
    ext = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

    # Local ROI mask for midpoint-centered diagnostics
    XG, YG = np.meshgrid(xg, yg)
    roi_mask = (
        (np.abs(XG - midpoint[0]) <= LOCAL_HALF_WINDOW)
        & (np.abs(YG - midpoint[1]) <= LOCAL_HALF_WINDOW)
    )

    # (a) SW reference at z*
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(np.abs(p_sw), origin="lower", extent=ext, cmap="viridis", aspect="equal")
    _plot_traps(axes[0], traps_m, idx_A, idx_B, neigh_idx)
    axes[0].plot([_mm(A[0]), _mm(B[0])], [_mm(A[1]), _mm(B[1])], "w--", lw=1.0)
    axes[0].set_title("Standing-wave reference |p| at z*")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="|p| [Pa]")

    fpar_sw = Fx_sw * e_ab[0] + Fy_sw * e_ab[1]
    fpar_sw_roi = np.where(roi_mask, fpar_sw, np.nan)
    vlim = np.nanmax(np.abs(fpar_sw_roi))
    im = axes[1].imshow(fpar_sw_roi, origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-vlim, vmax=vlim, aspect="equal")
    _plot_traps(axes[1], traps_m, idx_A, idx_B, neigh_idx)
    axes[1].plot([_mm(A[0]), _mm(B[0])], [_mm(A[1]), _mm(B[1])], "k--", lw=1.0)
    axes[1].set_title("SW force projection along A->B (local ROI)")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[1], shrink=0.85, label="F_parallel [N]")

    fig.suptitle("Local scenario setup at z*", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "overlay_local_sw_reference.png", dpi=220)
    plt.close(fig)

    # (b) Vortex overlay at best alpha/psi
    p_best = best["p_comb"]
    Fx_best = best["Fx"]
    Fy_best = best["Fy"]
    delta_mag = np.abs(p_best) - np.abs(p_sw)
    delta_fpar = (Fx_best - Fx_sw) * e_ab[0] + (Fy_best - Fy_sw) * e_ab[1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    im = axes[0, 0].imshow(np.abs(p_vortex), origin="lower", extent=ext,
                           cmap="inferno", aspect="equal")
    _plot_traps(axes[0, 0], traps_m, idx_A, idx_B, neigh_idx)
    axes[0, 0].set_title("Calibrated vortex perturbation |p_asm| at z*")
    axes[0, 0].set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    im = axes[0, 1].imshow(np.abs(p_best), origin="lower", extent=ext,
                           cmap="viridis", aspect="equal")
    _plot_traps(axes[0, 1], traps_m, idx_A, idx_B, neigh_idx)
    axes[0, 1].set_title("Combined |p| (SW + vortex)")
    axes[0, 1].set_xlabel("x [mm]")
    axes[0, 1].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)

    v = np.max(np.abs(delta_mag))
    im = axes[0, 2].imshow(delta_mag, origin="lower", extent=ext,
                           cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
    _plot_traps(axes[0, 2], traps_m, idx_A, idx_B, neigh_idx)
    axes[0, 2].set_title("Delta |p| = |p_comb| - |p_sw|")
    axes[0, 2].set_xlabel("x [mm]")
    axes[0, 2].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8)

    # Local force-corridor panel
    delta_fpar_roi = np.where(roi_mask, delta_fpar, np.nan)
    v2 = np.nanmax(np.abs(delta_fpar_roi))
    im = axes[1, 0].imshow(delta_fpar_roi, origin="lower", extent=ext,
                           cmap="RdBu_r", vmin=-v2, vmax=v2, aspect="equal")
    _plot_traps(axes[1, 0], traps_m, idx_A, idx_B, neigh_idx)
    axes[1, 0].plot([_mm(A[0]), _mm(B[0])], [_mm(A[1]), _mm(B[1])], "k--", lw=1.0)
    axes[1, 0].set_title("Delta force projection along A->B (local ROI)")
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    # AB line profile
    t = best["line_t"]
    axes[1, 1].plot(t, best["line_delta_fpar"], "r-", lw=1.8)
    axes[1, 1].axhline(0.0, color="k", ls="--", lw=0.9)
    axes[1, 1].set_xlabel("Along A->B (0=A, 1=B)")
    axes[1, 1].set_ylabel("Delta F_parallel [N]")
    axes[1, 1].set_title("Corridor profile: induced A->B drive")
    axes[1, 1].grid(True, alpha=0.3)

    # Metric bars for best point
    metric_names = ["A_push", "B_away", "N_rms", "tradeoff"]
    metric_vals = [
        best["A_push_toward_B"],
        best["B_push_away"],
        best["neighbour_rms_disturb"],
        best["tradeoff_score"],
    ]
    axes[1, 2].bar(metric_names, metric_vals,
                   color=["tab:green", "tab:red", "tab:orange", "tab:blue"])
    axes[1, 2].set_title("Best-point local metrics")
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Calibrated vortex overlay at z*  "
        f"(alpha={best_alpha:.2f}, psi={best_psi/np.pi:.1f}pi)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "overlay_local_vortex.png", dpi=220)
    plt.close(fig)

    # (c) Optional second perturbation comparison (vortex vs C-shape)
    p_c_best = m_cshape["p_comb"]
    delta_v = np.abs(best["p_comb"]) - np.abs(p_sw)
    delta_c = np.abs(p_c_best) - np.abs(p_sw)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    im = axes[0, 0].imshow(np.abs(p_vortex), origin="lower", extent=ext,
                           cmap="inferno", aspect="equal")
    _plot_traps(axes[0, 0], traps_m, idx_A, idx_B, neigh_idx)
    axes[0, 0].set_title("Vortex perturbation |p_asm| at z*")
    axes[0, 0].set_xlabel("x [mm]")
    axes[0, 0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    im = axes[0, 1].imshow(np.abs(p_cshape), origin="lower", extent=ext,
                           cmap="inferno", aspect="equal")
    _plot_traps(axes[0, 1], traps_m, idx_A, idx_B, neigh_idx)
    axes[0, 1].set_title("C-shape perturbation |p_asm| at z*")
    axes[0, 1].set_xlabel("x [mm]")
    axes[0, 1].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)

    v3 = max(np.max(np.abs(delta_v)), np.max(np.abs(delta_c)))
    im = axes[1, 0].imshow(delta_v, origin="lower", extent=ext,
                           cmap="RdBu_r", vmin=-v3, vmax=v3, aspect="equal")
    _plot_traps(axes[1, 0], traps_m, idx_A, idx_B, neigh_idx)
    axes[1, 0].set_title("Delta |p| for vortex overlay")
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    im = axes[1, 1].imshow(delta_c, origin="lower", extent=ext,
                           cmap="RdBu_r", vmin=-v3, vmax=v3, aspect="equal")
    _plot_traps(axes[1, 1], traps_m, idx_A, idx_B, neigh_idx)
    axes[1, 1].set_title("Delta |p| for C-shape overlay")
    axes[1, 1].set_xlabel("x [mm]")
    axes[1, 1].set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

    fig.suptitle(
        "Second perturbation check at z* (same alpha/psi as best vortex point)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT / "overlay_local_second_perturbation.png", dpi=220)
    plt.close(fig)

    # (d) Trade-off heatmaps
    mat_A = _heatmap_from_metrics(metrics, "A_push_toward_B", ALPHA_VALUES, PSI_VALUES)
    mat_B = _heatmap_from_metrics(metrics, "B_push_away", ALPHA_VALUES, PSI_VALUES)
    mat_N = _heatmap_from_metrics(metrics, "neighbour_rms_disturb", ALPHA_VALUES, PSI_VALUES)
    mat_S = _heatmap_from_metrics(metrics, "tradeoff_score", ALPHA_VALUES, PSI_VALUES)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    mats = [mat_A, mat_B, mat_N, mat_S]
    cmaps = ["RdYlGn", "RdYlGn_r", "RdYlGn_r", "RdYlGn"]
    titles = [
        "Desired action: A_push_toward_B [N]",
        "Undesired: B_push_away [N]",
        "Undesired: neighbour_rms_disturb [N]",
        "Trade-off score = A_push/(B_away + N_rms)",
    ]

    for ax, mat, cmap, title in zip(axes.ravel(), mats, cmaps, titles):
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(PSI_VALUES)))
        ax.set_xticklabels(PSI_LABELS)
        ax.set_yticks(range(len(ALPHA_VALUES)))
        ax.set_yticklabels([f"{a:.2f}" for a in ALPHA_VALUES])
        ax.set_xlabel("psi")
        ax.set_ylabel("alpha")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(j, i, f"{val:.1e}", ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle("Alpha/psi trade-off maps at z*", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "overlay_tradeoff_heatmaps.png", dpi=220)
    plt.close(fig)

    # 8) Save summary JSON
    summary = {
        "phase": "3.2-3.4",
        "z_star_mm": _mm(z_star),
        "lambda_mm": _mm(LAM),
        "grid": {
            "n_grid": N_GRID,
            "dx_um": dx * 1e6,
            "dy_um": dy * 1e6,
        },
        "pair": {
            "idx_A": idx_A,
            "idx_B": idx_B,
            "A_mm": [_mm(A[0]), _mm(A[1])],
            "B_mm": [_mm(B[0]), _mm(B[1])],
            "midpoint_mm": [_mm(midpoint[0]), _mm(midpoint[1])],
            "d_AB_mm": _mm(d_ab),
            "d_AB_over_lambda": d_ab / LAM,
            "n_neighbours": int(len(neigh_idx)),
            "neighbour_indices": [int(i) for i in neigh_idx],
        },
        "calibrated_vortex": {
            "waist_mm": params["waist_mm"],
            "axicon_angle_deg": params["axicon_angle_deg"],
            "ring_radius_zstar_mm": params["ring_radius_zstar_mm"],
            "scaled_peak_pa": float(np.max(np.abs(p_vortex))),
            "peak_ratio_to_sw": PERTURB_PEAK_RATIO,
        },
        "sweep": {
            "alpha_values": ALPHA_VALUES,
            "psi_values": [float(v) for v in PSI_VALUES],
            "n_points": len(metrics),
        },
        "best_vortex_tradeoff": {
            "alpha": best_alpha,
            "psi": best_psi,
            "psi_over_pi": best_psi / np.pi,
            "A_push_toward_B": best["A_push_toward_B"],
            "B_push_away": best["B_push_away"],
            "B_disturb_norm": best["B_disturb_norm"],
            "neighbour_rms_disturb": best["neighbour_rms_disturb"],
            "neighbour_max_disturb": best["neighbour_max_disturb"],
            "corridor_delta_mean": best["corridor_delta_mean"],
            "corridor_forward_fraction": best["corridor_forward_fraction"],
            "tradeoff_score": best["tradeoff_score"],
        },
        "cshape_same_point": {
            "alpha": best_alpha,
            "psi": best_psi,
            "A_push_toward_B": m_cshape["A_push_toward_B"],
            "B_push_away": m_cshape["B_push_away"],
            "neighbour_rms_disturb": m_cshape["neighbour_rms_disturb"],
            "tradeoff_score": m_cshape["tradeoff_score"],
        },
        "artifacts": {
            "metrics_csv": "results/deliverables/overlay_local/overlay_tradeoff_metrics.csv",
            "fig_sw_reference": "results/deliverables/overlay_local/overlay_local_sw_reference.png",
            "fig_vortex": "results/deliverables/overlay_local/overlay_local_vortex.png",
            "fig_second": "results/deliverables/overlay_local/overlay_local_second_perturbation.png",
            "fig_heatmaps": "results/deliverables/overlay_local/overlay_tradeoff_heatmaps.png",
        },
    }

    summary_path = OUT / "overlay_local_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")

    # Save compact NPZ for follow-on phases
    np.savez_compressed(
        OUT / "overlay_local_fields.npz",
        xg=xg,
        yg=yg,
        p_sw=p_sw,
        p_vortex=p_vortex,
        p_cshape=p_cshape,
        p_best=best["p_comb"],
        Fx_sw=Fx_sw,
        Fy_sw=Fy_sw,
        Fx_best=best["Fx"],
        Fy_best=best["Fy"],
        idx_A=idx_A,
        idx_B=idx_B,
        neigh_idx=neigh_idx,
        traps_m=traps_m,
        midpoint=midpoint,
    )
    print(f"Saved {OUT / 'overlay_local_fields.npz'}")

    print("\nPhase 3 local overlay study complete.")


if __name__ == "__main__":
    main()
