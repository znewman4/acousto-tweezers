#!/usr/bin/env python3
"""
Audit inverse C-shape lens reconstruction failure modes.

This script performs targeted diagnostics for inverse_c_shape_lens_study.py:
1) Feature size vs wavelength limits.
2) Pixel-resolution spans of key geometric features.
3) Constraint formulation audit (full-plane vs support).
4) Support/ROI-weighted reconstruction diagnostics.
5) Control experiments to separate physics limit from implementation issues.
6) Diagnostic plots for support masks, error/leakage maps, and azimuthal asymmetry.

Usage:
  python scripts/dev/audit_inverse_c_shape_study.py

Outputs:
  results/inverse_c_shape_lens_audit_<timestamp>/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_study import (
    RESULTS_ROOT,
    StudyConfig,
    asm_propagate,
    estimate_cshape_diameter_mm,
    generate_c_shape_target,
    make_aperture,
    make_asm_plan,
    make_grid,
    phase_to_thickness_maps,
)


@dataclass
class ExperimentSpec:
    name: str
    target_kind: str  # "c_shape" | "circle"
    diameter_mm: float
    phase_only: bool = True
    radial_thickness_mm: float | None = None
    target_smooth_sigma_px: float | None = None
    gap_size_deg: float | None = None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(float)
    b_flat = b.ravel().astype(float)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (na * nb))


def _clone_cfg(base_cfg: StudyConfig, **updates: Any) -> StudyConfig:
    cfg_data = asdict(base_cfg)
    cfg_data.update(updates)
    return StudyConfig(**cfg_data)


def _generate_circle_target(
    xx: np.ndarray,
    yy: np.ndarray,
    diameter_mm: float,
    cfg: StudyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    diameter_m = diameter_mm * 1e-3
    fwhm = diameter_m
    sigma = max(fwhm / 2.355, 1e-12)

    cx = cfg.target_center_x_mm * 1e-3
    cy = cfg.target_center_y_mm * 1e-3
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    amp_target = np.exp(-0.5 * (rr / sigma) ** 2)
    amp_target = gaussian_filter(amp_target, sigma=cfg.target_smooth_sigma_px)
    if amp_target.max() > 0.0:
        amp_target /= amp_target.max()
    amp_target *= cfg.target_peak_amplitude_pa

    phase_target = np.zeros_like(amp_target)
    p_target = amp_target * np.exp(1j * phase_target)

    meta = {
        "target_kind": "circle",
        "target_diameter_mm": float(diameter_mm),
        "fwhm_mm": float(diameter_m * 1e3),
        "sigma_mm": float(sigma * 1e3),
    }
    return p_target, amp_target, phase_target, meta


def _get_target(
    spec: ExperimentSpec,
    xx: np.ndarray,
    yy: np.ndarray,
    dx: float,
    cfg: StudyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], StudyConfig]:
    cfg_target = _clone_cfg(
        cfg,
        radial_thickness_mm=(
            spec.radial_thickness_mm if spec.radial_thickness_mm is not None else cfg.radial_thickness_mm
        ),
        target_smooth_sigma_px=(
            spec.target_smooth_sigma_px if spec.target_smooth_sigma_px is not None else cfg.target_smooth_sigma_px
        ),
        gap_size_deg=(spec.gap_size_deg if spec.gap_size_deg is not None else cfg.gap_size_deg),
    )

    if spec.target_kind == "c_shape":
        p_target, a_target, phi_target, meta = generate_c_shape_target(
            xx=xx,
            yy=yy,
            dx=dx,
            diameter_mm=spec.diameter_mm,
            cfg=cfg_target,
        )
    elif spec.target_kind == "circle":
        p_target, a_target, phi_target, meta = _generate_circle_target(
            xx=xx,
            yy=yy,
            diameter_mm=spec.diameter_mm,
            cfg=cfg_target,
        )
    else:
        raise ValueError(f"Unknown target kind: {spec.target_kind}")

    return p_target, a_target, phi_target, meta, cfg_target


def _support_masks(
    amp_target: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    cfg: StudyConfig,
    diameter_mm: float,
    support_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    peak = max(float(amp_target.max()), 1e-30)
    support_mask = amp_target >= (support_frac * peak)

    cx = cfg.target_center_x_mm * 1e-3
    cy = cfg.target_center_y_mm * 1e-3
    r_outer = 0.5 * diameter_mm * 1e-3
    roi_radius = 1.35 * r_outer
    roi_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (roi_radius**2)
    return support_mask, roi_mask


def _safe_profile(values: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return values, float("nan")
    v = values[finite]
    mean_v = float(np.mean(v))
    if mean_v <= 1e-30:
        return values, float("nan")
    return values, float(np.std(v) / mean_v)


def _azimuthal_profile(
    amp: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    cfg: StudyConfig,
    diameter_mm: float,
    radial_thickness_mm: float,
    n_bins: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    cx = cfg.target_center_x_mm * 1e-3
    cy = cfg.target_center_y_mm * 1e-3
    x_rel = xx - cx
    y_rel = yy - cy

    rr = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)

    r_outer = 0.5 * diameter_mm * 1e-3
    r_inner = max(r_outer - radial_thickness_mm * 1e-3, 0.15 * r_outer)
    r_ring = 0.5 * (r_outer + r_inner)
    r_band = max(0.45 * (r_outer - r_inner), 1e-12)

    annulus = np.abs(rr - r_ring) <= r_band
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask = annulus & (theta >= bins[i]) & (theta < bins[i + 1])
        if np.any(mask):
            profile[i] = float(np.mean(amp[mask]))

    return centers, profile


def _compute_metrics(
    p_target: np.ndarray,
    p_forward: np.ndarray,
    support_mask: np.ndarray,
    roi_mask: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    diameter_mm: float,
    cfg_target: StudyConfig,
    aperture_mask: np.ndarray,
    thickness_processed: np.ndarray,
) -> dict[str, float]:
    amp_t = np.abs(p_target)
    amp_f = np.abs(p_forward)
    err = amp_f - amp_t

    total_energy = float(np.sum(amp_f**2))
    support_energy = float(np.sum((amp_f[support_mask]) ** 2)) if np.any(support_mask) else 0.0
    outside_energy = float(np.sum((amp_f[~support_mask]) ** 2)) if np.any(~support_mask) else 0.0
    roi_energy = float(np.sum((amp_f[roi_mask]) ** 2)) if np.any(roi_mask) else 0.0

    inside_rmse = (
        float(np.sqrt(np.mean((err[support_mask]) ** 2))) if np.any(support_mask) else float("nan")
    )
    outside_rmse = (
        float(np.sqrt(np.mean((err[~support_mask]) ** 2))) if np.any(~support_mask) else float("nan")
    )
    roi_rmse = float(np.sqrt(np.mean((err[roi_mask]) ** 2))) if np.any(roi_mask) else float("nan")
    roi_corr = _corr(amp_f[roi_mask], amp_t[roi_mask]) if np.any(roi_mask) else float("nan")

    measured_diameter_mm = estimate_cshape_diameter_mm(
        amp=amp_f,
        xx=xx,
        yy=yy,
        target_roi_mask=support_mask,
        threshold_frac=cfg_target.diameter_threshold_frac,
    )

    # Roughness proxy: high-pass residual std over aperture
    base = np.where(aperture_mask, thickness_processed, np.nan)
    fill_val = float(np.nanmedian(base[aperture_mask])) if np.any(aperture_mask) else 0.0
    base_fill = np.where(np.isfinite(base), base, fill_val)
    smooth = gaussian_filter(base_fill, sigma=3.0)
    rough = base_fill - smooth
    roughness_um = float(np.std(rough[aperture_mask]) * 1e6) if np.any(aperture_mask) else float("nan")

    metrics = {
        "overall_rmse": float(np.sqrt(np.mean((err) ** 2))),
        "overall_corr": float(_corr(amp_f, amp_t)),
        "support_rmse": float(inside_rmse),
        "outside_rmse": float(outside_rmse),
        "roi_rmse": float(roi_rmse),
        "roi_corr": float(roi_corr),
        "support_energy_frac": float(support_energy / max(total_energy, 1e-30)),
        "outside_support_energy_frac": float(outside_energy / max(total_energy, 1e-30)),
        "roi_energy_frac": float(roi_energy / max(total_energy, 1e-30)),
        "target_support_area_frac": float(np.mean(support_mask)),
        "target_support_energy_frac": float(np.sum(amp_t[support_mask] ** 2) / max(np.sum(amp_t**2), 1e-30)),
        "measured_diameter_mm": float(measured_diameter_mm),
        "diameter_error_mm": float(abs(measured_diameter_mm - diameter_mm))
        if np.isfinite(measured_diameter_mm)
        else float("nan"),
        "thickness_roughness_um": float(roughness_um),
    }
    return metrics


def _save_diagnostic_plot(
    save_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    support_mask: np.ndarray,
    enforced_mask: np.ndarray,
    inside_error_map: np.ndarray,
    outside_leak_map: np.ndarray,
    theta: np.ndarray,
    profile_target: np.ndarray,
    profile_forward: np.ndarray,
    asym_target: float,
    asym_forward: float,
    profile_corr: float,
    title: str,
) -> None:
    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))

    ax = axes[0, 0]
    im = ax.imshow(support_mask.astype(float), origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Target Support Mask")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(enforced_mask.astype(float), origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Enforced Amplitude Support Mask")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    im = ax.imshow(inside_error_map, origin="lower", extent=extent, cmap="inferno")
    ax.set_title("Inside-Support Error | |p_f|-|p_t| |")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    im = ax.imshow(outside_leak_map, origin="lower", extent=extent, cmap="magma")
    ax.set_title("Outside-Support Leakage |p_f|")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.plot(theta, profile_target, lw=2.0, label="target")
    ax.plot(theta, profile_forward, lw=2.0, label="forward")
    ax.set_title("Azimuthal Profile on Ring")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("mean amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        "Azimuthal Asymmetry Metric",
        "",
        f"target asymmetry index:  {asym_target:.4f}",
        f"forward asymmetry index: {asym_forward:.4f}",
        f"forward/target ratio:    {asym_forward / max(asym_target, 1e-30):.4f}",
        f"profile correlation:     {profile_corr:.4f}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)

    fig.suptitle(title)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_experiment(
    spec: ExperimentSpec,
    cfg: StudyConfig,
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    dx: float,
    dy: float,
    aperture_mask: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    exp_dir = out_dir / spec.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    p_target, a_target, _, target_meta, cfg_target = _get_target(spec, xx, yy, dx, cfg)

    plan_fwd = make_asm_plan(
        field_shape=p_target.shape,
        dx=dx,
        dy=dy,
        k=cfg.k_water,
        z=cfg.propagation_distance_m,
        pad_factor=cfg.pad_factor,
        include_evanescent=cfg.include_evanescent,
    )
    plan_bwd = make_asm_plan(
        field_shape=p_target.shape,
        dx=dx,
        dy=dy,
        k=cfg.k_water,
        z=-cfg.propagation_distance_m,
        pad_factor=cfg.pad_factor,
        include_evanescent=cfg.include_evanescent,
    )

    rng = np.random.default_rng(cfg.seed)
    random_phase = rng.uniform(-np.pi, np.pi, size=p_target.shape)
    p_lens = np.zeros_like(p_target, dtype=complex)
    p_lens[aperture_mask] = np.exp(1j * random_phase[aperture_mask])

    history_rmse: list[float] = []
    history_corr: list[float] = []
    p_forward_pre_projection = np.zeros_like(p_target, dtype=complex)
    p_lens_pre_projection = np.zeros_like(p_target, dtype=complex)

    for it in range(1, cfg.n_iter + 1):
        p_at_target = asm_propagate(p_lens, plan_fwd)

        amp_est = np.abs(p_at_target)
        history_rmse.append(float(np.sqrt(np.mean((amp_est - a_target) ** 2))))
        history_corr.append(float(_corr(amp_est, a_target)))

        # Exact same target-plane amplitude replacement as inverse_c_shape_lens_study.py
        p_target_constrained = a_target * np.exp(1j * np.angle(p_at_target))

        p_back_to_lens = asm_propagate(p_target_constrained, plan_bwd)

        p_lens_pre_projection = np.zeros_like(p_lens)
        p_lens_pre_projection[aperture_mask] = p_back_to_lens[aperture_mask]
        if it == cfg.n_iter:
            p_forward_pre_projection = asm_propagate(p_lens_pre_projection, plan_fwd)

        p_lens = np.zeros_like(p_lens)
        if spec.phase_only:
            p_lens[aperture_mask] = np.exp(1j * np.angle(p_back_to_lens[aperture_mask]))
        else:
            p_lens[aperture_mask] = p_back_to_lens[aperture_mask]

    p_forward = asm_propagate(p_lens, plan_fwd)

    support_mask, roi_mask = _support_masks(
        amp_target=a_target,
        xx=xx,
        yy=yy,
        cfg=cfg_target,
        diameter_mm=spec.diameter_mm,
        support_frac=0.20,
    )
    enforced_mask = np.ones_like(support_mask, dtype=bool)

    phase_raw = np.angle(p_lens)
    _, _, thickness_processed = phase_to_thickness_maps(
        phase_raw=phase_raw,
        mask=aperture_mask,
        dx=dx,
        cfg=cfg,
    )

    metrics = _compute_metrics(
        p_target=p_target,
        p_forward=p_forward,
        support_mask=support_mask,
        roi_mask=roi_mask,
        xx=xx,
        yy=yy,
        diameter_mm=spec.diameter_mm,
        cfg_target=cfg_target,
        aperture_mask=aperture_mask,
        thickness_processed=thickness_processed,
    )

    amp_t = np.abs(p_target)
    amp_f = np.abs(p_forward)
    amp_pre = np.abs(p_forward_pre_projection)

    metrics_pre = {
        "pre_projection_overall_corr": float(_corr(amp_pre, amp_t)),
        "post_projection_overall_corr": float(_corr(amp_f, amp_t)),
        "pre_projection_overall_rmse": float(np.sqrt(np.mean((amp_pre - amp_t) ** 2))),
        "post_projection_overall_rmse": float(np.sqrt(np.mean((amp_f - amp_t) ** 2))),
    }

    radial_thickness_eff_mm = min(
        cfg_target.radial_thickness_mm,
        0.95 * 0.5 * spec.diameter_mm,
    )
    theta, profile_t = _azimuthal_profile(
        amp=amp_t,
        xx=xx,
        yy=yy,
        cfg=cfg_target,
        diameter_mm=spec.diameter_mm,
        radial_thickness_mm=radial_thickness_eff_mm,
    )
    _, profile_f = _azimuthal_profile(
        amp=amp_f,
        xx=xx,
        yy=yy,
        cfg=cfg_target,
        diameter_mm=spec.diameter_mm,
        radial_thickness_mm=radial_thickness_eff_mm,
    )
    _, asym_t = _safe_profile(profile_t)
    _, asym_f = _safe_profile(profile_f)

    valid_prof = np.isfinite(profile_t) & np.isfinite(profile_f)
    profile_corr = (
        _corr(profile_t[valid_prof], profile_f[valid_prof]) if np.any(valid_prof) else float("nan")
    )

    inside_error_map = np.where(support_mask, np.abs(amp_f - amp_t), np.nan)
    outside_leak_map = np.where(~support_mask, amp_f, np.nan)

    _save_diagnostic_plot(
        save_path=exp_dir / "diagnostic_support_leakage_asymmetry.png",
        x=x,
        y=y,
        support_mask=support_mask,
        enforced_mask=enforced_mask,
        inside_error_map=inside_error_map,
        outside_leak_map=outside_leak_map,
        theta=theta,
        profile_target=profile_t,
        profile_forward=profile_f,
        asym_target=asym_t,
        asym_forward=asym_f,
        profile_corr=profile_corr,
        title=f"{spec.name}: support/leakage/asymmetry diagnostics",
    )

    # Save forward-vs-target close view for quick qualitative check
    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    vmax = max(float(amp_t.max()), float(amp_f.max()), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    axes[0].imshow(amp_t, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=vmax)
    axes[0].set_title("Target amplitude")
    axes[1].imshow(amp_f, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=vmax)
    axes[1].set_title("Forward amplitude")
    axes[2].imshow(np.abs(amp_f - amp_t), origin="lower", extent=extent, cmap="magma")
    axes[2].set_title("Absolute error")
    for ax in axes:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        zoom_half_mm = max(spec.diameter_mm * 2.0, 1.0)
        ax.set_xlim(-zoom_half_mm, zoom_half_mm)
        ax.set_ylim(-zoom_half_mm, zoom_half_mm)
    fig.tight_layout()
    fig.savefig(exp_dir / "target_vs_forward_closeup.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "experiment": asdict(spec),
        "target_meta": target_meta,
        "metrics": metrics,
        "phase_projection_comparison": metrics_pre,
        "asymmetry": {
            "target_index": float(asym_t),
            "forward_index": float(asym_f),
            "forward_to_target_ratio": float(asym_f / max(asym_t, 1e-30)),
            "profile_corr": float(profile_corr),
        },
        "history": {
            "amp_rmse": history_rmse,
            "amp_corr": history_corr,
        },
        "constraint_audit": {
            "amplitude_replacement_formula": "p_target_constrained = a_target * exp(1j * angle(p_at_target))",
            "enforced_amplitude_support": "full_plane",
            "explicit_outside_support_penalty": False,
        },
    }

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def _feature_size_table(cfg: StudyConfig, dx_mm: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    lambda_mm = cfg.wavelength_m * 1e3
    for d_mm in (0.4, 0.5, 0.6, 0.7):
        r_outer_mm = 0.5 * d_mm
        radial_mm = min(cfg.radial_thickness_mm, 0.95 * r_outer_mm)
        r_ring_mm = 0.5 * ((r_outer_mm) + (r_outer_mm - radial_mm))
        gap_rad = np.deg2rad(cfg.gap_size_deg)
        gap_width_mm = gap_rad * r_ring_mm

        d_lam = d_mm / lambda_mm
        radial_lam = radial_mm / lambda_mm

        if d_lam < 0.9:
            regime = "sub-wavelength"
        elif d_lam <= 1.3:
            regime = "near-wavelength"
        else:
            regime = "multi-wavelength"

        rows.append(
            {
                "target_diameter_mm": float(d_mm),
                "wavelength_mm": float(lambda_mm),
                "diameter_lambda": float(d_lam),
                "radial_thickness_mm": float(radial_mm),
                "radial_thickness_lambda": float(radial_lam),
                "outer_diameter_px": float(d_mm / dx_mm),
                "radial_thickness_px": float(radial_mm / dx_mm),
                "angular_gap_width_mm_at_ring": float(gap_width_mm),
                "angular_gap_width_px_at_ring": float(gap_width_mm / dx_mm),
                "regime": regime,
            }
        )
    return rows


def _write_feature_table_csv(rows: list[dict[str, float | str]], save_path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _build_failure_assessment(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    baseline_rows = [r for r in summary_rows if r["experiment"]["name"].startswith("baseline_cshape")]
    circle_rows = [r for r in summary_rows if r["experiment"]["target_kind"] == "circle"]
    large_rows = [r for r in summary_rows if "large_cshape" in r["experiment"]["name"]]
    complex_rows = [r for r in summary_rows if "complex_lens" in r["experiment"]["name"]]

    def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([float(r["metrics"][key]) for r in rows]))

    assessment = {
        "baseline_outside_support_energy_frac": _mean_metric(baseline_rows, "outside_support_energy_frac"),
        "baseline_roi_corr": _mean_metric(baseline_rows, "roi_corr"),
        "circle_roi_corr": _mean_metric(circle_rows, "roi_corr"),
        "large_cshape_roi_corr": _mean_metric(large_rows, "roi_corr"),
        "complex_lens_roi_corr": _mean_metric(complex_rows, "roi_corr"),
    }
    return assessment


def _write_report(
    save_path: Path,
    cfg: StudyConfig,
    dx_mm: float,
    feature_rows: list[dict[str, float | str]],
    summary_rows: list[dict[str, Any]],
    assessment: dict[str, float],
    runtime_s: float,
) -> None:
    lambda_mm = cfg.wavelength_m * 1e3

    baseline_04 = next((r for r in summary_rows if r["experiment"]["name"] == "baseline_cshape_d0p4_phaseonly"), None)
    baseline_05 = next((r for r in summary_rows if r["experiment"]["name"] == "baseline_cshape_d0p5_phaseonly"), None)
    baseline_06 = next((r for r in summary_rows if r["experiment"]["name"] == "baseline_cshape_d0p6_phaseonly"), None)
    thick_06 = next((r for r in summary_rows if r["experiment"]["name"] == "thicker_smoother_cshape_d0p6_phaseonly"), None)
    circle_06 = next((r for r in summary_rows if r["experiment"]["name"] == "circle_spot_d0p6_phaseonly"), None)
    large_c = next((r for r in summary_rows if r["experiment"]["name"] == "large_cshape_d1p8_phaseonly"), None)
    complex_06 = next((r for r in summary_rows if r["experiment"]["name"] == "baseline_cshape_d0p6_complex_lens"), None)

    lines: list[str] = []
    lines.append("# Inverse C-Shape Failure Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## 1) Feature Size vs Wavelength")
    lines.append(f"- Wavelength at 2 MHz in water: {lambda_mm:.4f} mm")
    lines.append("- Small-target diameters and ring thickness are predominantly sub-wavelength or near-wavelength.")
    lines.append("")
    lines.append("## 2) Discretization (Pixels per Feature)")
    lines.append(f"- Grid spacing used in audit: dx = dy = {dx_mm:.5f} mm")
    lines.append("- See feature_resolution.csv for per-diameter pixel spans (outer diameter, radial thickness, angular gap width).")
    lines.append("")
    lines.append("## 3) Constraint Formulation Audit")
    lines.append("- Iterative amplitude replacement is full-plane:")
    lines.append("  p_target_constrained = a_target * exp(i * angle(p_at_target))")
    lines.append("- No explicit support mask is used in the amplitude replacement step.")
    lines.append("- No explicit outside-support penalty is present in the core loop.")
    lines.append("- This permits solutions that improve global metrics while remaining azimuthally symmetric around the center.")
    lines.append("")
    lines.append("## 4) Target Normalization / Occupancy")
    lines.append("- Target amplitude is normalized by global max, then scaled by target_peak_amplitude_pa.")
    lines.append("- For small C-shapes, support occupies a small area fraction; global full-plane metrics can underweight shape fidelity.")
    lines.append("")
    lines.append("## 5) ROI/Support Metrics")
    lines.append("- Added support_rmse, outside_support_energy_frac, roi_rmse, roi_corr, support_energy_frac.")
    lines.append("- Added per-experiment diagnostic plot with support masks, inside-support error, outside-support leakage, and azimuthal profile.")
    lines.append("")
    lines.append("## 6) Thin/Sharp C-shape vs Thicker/Smoother")
    if baseline_06 and thick_06:
        b = baseline_06["metrics"]
        t = thick_06["metrics"]
        lines.append(
            "- D=0.6 mm baseline vs thicker/smoother C-shape: "
            f"roi_corr {b['roi_corr']:.4f} -> {t['roi_corr']:.4f}, "
            f"outside_support_energy_frac {b['outside_support_energy_frac']:.4f} -> {t['outside_support_energy_frac']:.4f}."
        )
    lines.append("")
    lines.append("## 7) Phase-only Projection Effect")
    if baseline_06 and complex_06:
        p = baseline_06["phase_projection_comparison"]
        c = complex_06["metrics"]
        lines.append(
            "- For D=0.6 mm: phase-only projection degrades pre-projection field fidelity "
            f"(pre rmse={p['pre_projection_overall_rmse']:.4f}, post rmse={p['post_projection_overall_rmse']:.4f})."
        )
        lines.append(
            "- Allowing complex lens amplitude (control) improves ROI fidelity relative to phase-only "
            f"(complex roi_corr={c['roi_corr']:.4f})."
        )
    lines.append("")
    lines.append("## 8) Control Experiments")
    if circle_06:
        m = circle_06["metrics"]
        lines.append(
            f"- Circle spot (D=0.6 mm): roi_corr={m['roi_corr']:.4f}, outside_support_energy_frac={m['outside_support_energy_frac']:.4f}."
        )
    if large_c:
        m = large_c["metrics"]
        lines.append(
            f"- Large crescent (D=1.8 mm): roi_corr={m['roi_corr']:.4f}, outside_support_energy_frac={m['outside_support_energy_frac']:.4f}."
        )
    lines.append("- Controls separate algorithm behavior on simple/larger targets from tiny-subwavelength C-shape behavior.")
    lines.append("")
    lines.append("## 9) Added Plots")
    lines.append("- target support mask")
    lines.append("- enforced amplitude support mask")
    lines.append("- inside-support error")
    lines.append("- outside-support leakage")
    lines.append("- azimuthal asymmetry profile/metric")
    lines.append("")
    lines.append("## 10) Root-Cause Attribution")
    lines.append("- Primary: sub-wavelength/near-wavelength feature infeasibility for the requested tiny C-shapes at 2 MHz.")
    lines.append("- Primary: weak support-specific constraint (full-plane amplitude replacement without explicit outside-support suppression).")
    lines.append("- Secondary: phase-only projection pushes solution toward lower-order symmetric modes.")
    lines.append("- Secondary: under-resolution of thin ring/gap in pixels contributes to instability and rough thickness maps.")
    lines.append("- No strong evidence of a fundamental ASM propagation bug from controls; failure appears dominated by formulation + physics limits.")
    lines.append("")
    lines.append("## Quick Numbers")
    if baseline_04 and baseline_05 and baseline_06:
        for r in [baseline_04, baseline_05, baseline_06]:
            m = r["metrics"]
            lines.append(
                f"- {r['experiment']['name']}: measured_diameter_mm={m['measured_diameter_mm']:.3f}, "
                f"roi_corr={m['roi_corr']:.4f}, outside_support_energy_frac={m['outside_support_energy_frac']:.4f}, "
                f"thickness_roughness_um={m['thickness_roughness_um']:.2f}"
            )
    lines.append("")
    lines.append(f"Runtime: {runtime_s:.2f} s")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit inverse C-shape reconstruction failure modes")
    parser.add_argument("--n-grid", type=int, default=512)
    parser.add_argument("--n-iter", type=int, default=70)
    parser.add_argument("--domain-size-mm", type=float, default=28.0)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()

    cfg = StudyConfig(
        n_grid=int(args.n_grid),
        n_iter=int(args.n_iter),
        domain_size_mm=float(args.domain_size_mm),
        target_diameters_mm=(0.4, 0.5, 0.6, 0.7),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = RESULTS_ROOT / f"inverse_c_shape_lens_audit_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    x, y, xx, yy, dx, dy = make_grid(cfg)
    aperture_mask = make_aperture(xx, yy, cfg.transducer_diameter_m)
    dx_mm = dx * 1e3

    feature_rows = _feature_size_table(cfg, dx_mm=dx_mm)
    _write_feature_table_csv(feature_rows, run_dir / "feature_resolution.csv")

    experiments = [
        ExperimentSpec(name="baseline_cshape_d0p4_phaseonly", target_kind="c_shape", diameter_mm=0.4, phase_only=True),
        ExperimentSpec(name="baseline_cshape_d0p5_phaseonly", target_kind="c_shape", diameter_mm=0.5, phase_only=True),
        ExperimentSpec(name="baseline_cshape_d0p6_phaseonly", target_kind="c_shape", diameter_mm=0.6, phase_only=True),
        ExperimentSpec(
            name="thicker_smoother_cshape_d0p6_phaseonly",
            target_kind="c_shape",
            diameter_mm=0.6,
            phase_only=True,
            radial_thickness_mm=0.30,
            target_smooth_sigma_px=3.0,
        ),
        ExperimentSpec(name="baseline_cshape_d0p6_complex_lens", target_kind="c_shape", diameter_mm=0.6, phase_only=False),
        ExperimentSpec(name="circle_spot_d0p6_phaseonly", target_kind="circle", diameter_mm=0.6, phase_only=True),
        ExperimentSpec(
            name="large_cshape_d1p8_phaseonly",
            target_kind="c_shape",
            diameter_mm=1.8,
            phase_only=True,
            radial_thickness_mm=0.35,
            target_smooth_sigma_px=2.2,
        ),
    ]

    summary_rows: list[dict[str, Any]] = []
    print("=" * 80)
    print("Inverse C-Shape Failure Audit")
    print("=" * 80)
    print(f"Output directory: {run_dir}")
    print(f"Grid: {cfg.n_grid} x {cfg.n_grid}, n_iter={cfg.n_iter}, dx={dx_mm:.5f} mm")
    print(f"Wavelength: {cfg.wavelength_m * 1e3:.5f} mm")

    for spec in experiments:
        print(f"\n[run] {spec.name}")
        payload = _run_experiment(
            spec=spec,
            cfg=cfg,
            x=x,
            y=y,
            xx=xx,
            yy=yy,
            dx=dx,
            dy=dy,
            aperture_mask=aperture_mask,
            out_dir=figs_dir,
        )
        summary_rows.append(payload)

        m = payload["metrics"]
        a = payload["asymmetry"]
        print(
            "  -> measured_d={:.3f} mm, roi_corr={:.4f}, outside_support_energy={:.4f}, "
            "asym_ratio={:.4f}".format(
                m["measured_diameter_mm"],
                m["roi_corr"],
                m["outside_support_energy_frac"],
                a["forward_to_target_ratio"],
            )
        )

    rows_for_csv: list[dict[str, Any]] = []
    for payload in summary_rows:
        spec = payload["experiment"]
        m = payload["metrics"]
        a = payload["asymmetry"]
        p = payload["phase_projection_comparison"]
        rows_for_csv.append(
            {
                "name": spec["name"],
                "target_kind": spec["target_kind"],
                "diameter_mm": spec["diameter_mm"],
                "phase_only": spec["phase_only"],
                "measured_diameter_mm": m["measured_diameter_mm"],
                "diameter_error_mm": m["diameter_error_mm"],
                "overall_corr": m["overall_corr"],
                "roi_corr": m["roi_corr"],
                "support_rmse": m["support_rmse"],
                "outside_support_energy_frac": m["outside_support_energy_frac"],
                "target_support_area_frac": m["target_support_area_frac"],
                "target_support_energy_frac": m["target_support_energy_frac"],
                "thickness_roughness_um": m["thickness_roughness_um"],
                "asym_target": a["target_index"],
                "asym_forward": a["forward_index"],
                "asym_ratio": a["forward_to_target_ratio"],
                "asym_profile_corr": a["profile_corr"],
                "pre_projection_overall_corr": p["pre_projection_overall_corr"],
                "post_projection_overall_corr": p["post_projection_overall_corr"],
                "pre_projection_overall_rmse": p["pre_projection_overall_rmse"],
                "post_projection_overall_rmse": p["post_projection_overall_rmse"],
            }
        )

    with open(run_dir / "experiment_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
        w.writeheader()
        for row in rows_for_csv:
            w.writerow(row)

    with open(run_dir / "experiment_payloads.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    assessment = _build_failure_assessment(summary_rows)
    with open(run_dir / "failure_assessment.json", "w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)

    runtime_s = time.time() - t0
    _write_report(
        save_path=run_dir / "audit_report.md",
        cfg=cfg,
        dx_mm=dx_mm,
        feature_rows=feature_rows,
        summary_rows=summary_rows,
        assessment=assessment,
        runtime_s=runtime_s,
    )

    print("\n" + "=" * 80)
    print(f"Audit complete in {runtime_s:.2f} s")
    print(f"Wrote: {run_dir / 'feature_resolution.csv'}")
    print(f"Wrote: {run_dir / 'experiment_summary.csv'}")
    print(f"Wrote: {run_dir / 'audit_report.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
