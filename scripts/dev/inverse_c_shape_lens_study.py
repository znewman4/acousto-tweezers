#!/usr/bin/env python3
"""
Iterative inverse design study for a local C-shaped acoustic lens.

This script implements a strict iterative propagation loop (Gerchberg-Saxton
style) for a phase-only passive lens attached to a circular transducer.
The lens is designed from a target complex field, not from direct geometry.

Usage:
    python scripts/dev/inverse_c_shape_lens_study.py

Example quick smoke test:
    python scripts/dev/inverse_c_shape_lens_study.py \
        --target-diameters-mm 0.4 \
        --n-iter 12 \
        --n-grid 384 \
        --domain-size-mm 24
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"


def _fmt_mm(value_m: float) -> str:
    return f"{value_m * 1e3:.4f}"


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(float)
    b_flat = b.ravel().astype(float)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (na * nb))


def _add_colorbar(ax: plt.Axes, im: Any, label: str = "") -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    plt.colorbar(im, cax=cax, label=label)


@dataclass
class StudyConfig:
    # Physical setup
    frequency_hz: float = 2.0e6
    c_water: float = 1484.0
    c_lens: float = 2700.0

    # Hardware constraints
    transducer_diameter_mm: float = 20.0

    # Propagation setup
    propagation_distance_mm: float = 6.0
    include_evanescent: bool = False
    pad_factor: int = 2

    # Numerical domain
    domain_size_mm: float = 28.0
    n_grid: int = 1024

    # Iterative solver
    n_iter: int = 180
    enforce_target_phase: bool = False
    seed: int = 7

    # Anti-vortex controls
    target_phase_mode: str = "free"  # free | flat-support | target-fixed
    init_phase_mode: str = "random"  # random | flat | weak-random
    weak_random_phase_std_rad: float = 0.12
    support_weight: float = 1.0
    gap_weight: float = 1.0
    outside_weight: float = 0.35

    # Target generator
    target_diameters_mm: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7)
    radial_thickness_mm: float = 0.12
    gap_size_deg: float = 70.0
    orientation_deg: float = 0.0
    phase_ramp_rad: float = 0.0
    target_peak_amplitude_pa: float = 1.0
    target_smooth_sigma_px: float = 1.2
    target_center_x_mm: float = 0.0
    target_center_y_mm: float = 0.0
    anti_vortex_offset_x_mm: float = 0.0
    anti_vortex_offset_y_mm: float = 0.0
    target_asymmetry_boost: float = 0.0

    # Phase-thickness conversion
    base_thickness_mm: float = 0.60
    processed_sigma_mm: float = 0.03

    # Validation metrics
    target_roi_threshold_frac: float = 0.10
    diameter_threshold_frac: float = 0.35
    localization_radius_factor: float = 1.6

    # Achievability criteria (for D=0.4 mm statement)
    achievable_diameter_tol_mm: float = 0.15
    achievable_min_localization: float = 0.65
    achievable_max_leakage: float = 0.35
    achievable_min_amp_corr: float = 0.65

    # Exports
    export_stl: bool = False
    stl_max_grid: int = 240

    def __post_init__(self) -> None:
        self.wavelength_m = self.c_water / self.frequency_hz
        self.k_water = 2.0 * np.pi * self.frequency_hz / self.c_water
        self.k_lens = 2.0 * np.pi * self.frequency_hz / self.c_lens
        self.dk = self.k_lens - self.k_water
        if abs(self.dk) <= 1e-30:
            raise ValueError("k_lens and k_water are too close; thickness mapping is ill-posed")

        self.transducer_diameter_m = self.transducer_diameter_mm * 1e-3
        self.propagation_distance_m = self.propagation_distance_mm * 1e-3
        self.domain_size_m = self.domain_size_mm * 1e-3
        self.base_thickness_m = self.base_thickness_mm * 1e-3
        self.full_2pi_thickness_m = 2.0 * np.pi / abs(self.dk)

        if self.n_grid < 128:
            raise ValueError("n_grid must be at least 128")
        if self.pad_factor < 1:
            raise ValueError("pad_factor must be >= 1")
        if self.n_iter < 8:
            raise ValueError("n_iter must be >= 8 to save 5 distinct snapshots")

        if self.target_phase_mode not in {"free", "flat-support", "target-fixed"}:
            raise ValueError("target_phase_mode must be one of: free, flat-support, target-fixed")

        if self.init_phase_mode not in {"random", "flat", "weak-random"}:
            raise ValueError("init_phase_mode must be one of: random, flat, weak-random")

        if self.weak_random_phase_std_rad < 0.0:
            raise ValueError("weak_random_phase_std_rad must be >= 0")

        for name, w in (
            ("support_weight", self.support_weight),
            ("gap_weight", self.gap_weight),
            ("outside_weight", self.outside_weight),
        ):
            if w < 0.0 or w > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if self.target_asymmetry_boost < 0.0:
            raise ValueError("target_asymmetry_boost must be >= 0")

        clean = []
        max_recommended_mm = 0.9 * self.domain_size_mm
        for d in self.target_diameters_mm:
            d_round = round(float(d), 3)
            if d_round <= 0.0:
                raise ValueError(f"target diameter {d} mm must be > 0")
            if d_round >= max_recommended_mm:
                raise ValueError(
                    f"target diameter {d_round:.3f} mm is too large for domain {self.domain_size_mm:.3f} mm"
                )
            clean.append(d_round)
        self.target_diameters_mm = tuple(clean)


@dataclass
class ASMPlan:
    h_transfer: np.ndarray
    ny: int
    nx: int
    ny_pad: int
    nx_pad: int
    y0: int
    x0: int


def make_grid(cfg: StudyConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x = np.linspace(-0.5 * cfg.domain_size_m, 0.5 * cfg.domain_size_m, cfg.n_grid, endpoint=False)
    y = np.linspace(-0.5 * cfg.domain_size_m, 0.5 * cfg.domain_size_m, cfg.n_grid, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return x, y, xx, yy, dx, dy


def make_aperture(xx: np.ndarray, yy: np.ndarray, diameter_m: float) -> np.ndarray:
    rr = np.sqrt(xx**2 + yy**2)
    return rr <= (0.5 * diameter_m)


def make_asm_plan(
    field_shape: tuple[int, int],
    dx: float,
    dy: float,
    k: float,
    z: float,
    pad_factor: int,
    include_evanescent: bool,
) -> ASMPlan:
    ny, nx = field_shape
    ny_pad = ny * pad_factor
    nx_pad = nx * pad_factor
    y0 = (ny_pad - ny) // 2
    x0 = (nx_pad - nx) // 2

    fx = np.fft.fftfreq(nx_pad, d=dx)
    fy = np.fft.fftfreq(ny_pad, d=dy)
    kx, ky = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy)

    kz2 = (k**2) - (kx**2 + ky**2)
    propagating = kz2 >= 0.0

    kz = np.zeros_like(kz2, dtype=complex)
    kz[propagating] = np.sqrt(kz2[propagating])

    if include_evanescent:
        kz[~propagating] = 1j * np.sqrt(-kz2[~propagating])
        h_transfer = np.exp(1j * kz * z)
    else:
        h_transfer = np.zeros_like(kz, dtype=complex)
        h_transfer[propagating] = np.exp(1j * kz[propagating] * z)

    return ASMPlan(
        h_transfer=h_transfer,
        ny=ny,
        nx=nx,
        ny_pad=ny_pad,
        nx_pad=nx_pad,
        y0=y0,
        x0=x0,
    )


def asm_propagate(field: np.ndarray, plan: ASMPlan) -> np.ndarray:
    padded = np.zeros((plan.ny_pad, plan.nx_pad), dtype=complex)
    padded[plan.y0 : plan.y0 + plan.ny, plan.x0 : plan.x0 + plan.nx] = field

    field_pad = np.fft.ifft2(np.fft.fft2(padded) * plan.h_transfer)
    return field_pad[plan.y0 : plan.y0 + plan.ny, plan.x0 : plan.x0 + plan.nx]


def generate_c_shape_target(
    xx: np.ndarray,
    yy: np.ndarray,
    dx: float,
    diameter_mm: float,
    cfg: StudyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    diameter_m = diameter_mm * 1e-3
    r_outer = 0.5 * diameter_m
    radial_thickness_m = min(cfg.radial_thickness_mm * 1e-3, 0.95 * r_outer)
    r_inner = max(r_outer - radial_thickness_m, 0.15 * r_outer)

    cx = (cfg.target_center_x_mm + cfg.anti_vortex_offset_x_mm) * 1e-3
    cy = (cfg.target_center_y_mm + cfg.anti_vortex_offset_y_mm) * 1e-3
    x_rel = xx - cx
    y_rel = yy - cy

    rr = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)

    orientation = np.deg2rad(cfg.orientation_deg)
    theta_local = _wrap_to_pi(theta - orientation)

    r_ring = 0.5 * (r_outer + r_inner)
    sigma_r = max(0.20 * radial_thickness_m, radial_thickness_m / 2.355)
    radial_profile = np.exp(-0.5 * ((rr - r_ring) / max(sigma_r, 1e-12)) ** 2)

    gap_half = 0.5 * np.deg2rad(cfg.gap_size_deg)
    gap_edge = max(np.deg2rad(3.0), 0.18 * gap_half)
    theta_abs = np.abs(theta_local)
    angular_profile = 0.5 * (1.0 + np.tanh((theta_abs - gap_half) / max(gap_edge, 1e-12)))

    local_envelope = np.exp(-(rr / max(1.35 * r_outer, 1e-12)) ** 4)

    amp_target = radial_profile * angular_profile * local_envelope

    if cfg.target_asymmetry_boost > 0.0:
        # Emphasize the back-side arc relative to the opening to break circular degeneracy.
        asym_profile = np.clip(1.0 + cfg.target_asymmetry_boost * np.cos(theta_local - np.pi), 0.05, None)
        amp_target = amp_target * asym_profile

    amp_target = gaussian_filter(amp_target, sigma=cfg.target_smooth_sigma_px)
    if amp_target.max() > 0.0:
        amp_target = amp_target / amp_target.max()
    amp_target = cfg.target_peak_amplitude_pa * amp_target

    phase_target = np.zeros_like(amp_target)
    if abs(cfg.phase_ramp_rad) > 0.0:
        # Ramp along opening direction in local coordinates.
        u = x_rel * np.cos(orientation) + y_rel * np.sin(orientation)
        phase_target = cfg.phase_ramp_rad * np.clip(u / max(r_outer, 1e-12), -1.0, 1.0)

    p_target = amp_target * np.exp(1j * phase_target)

    meta = {
        "target_diameter_mm": float(diameter_mm),
        "r_outer_mm": float(r_outer * 1e3),
        "r_inner_mm": float(r_inner * 1e3),
        "radial_thickness_mm": float((r_outer - r_inner) * 1e3),
        "gap_size_deg": float(cfg.gap_size_deg),
        "orientation_deg": float(cfg.orientation_deg),
        "phase_ramp_rad": float(cfg.phase_ramp_rad),
        "target_peak_amplitude_pa": float(cfg.target_peak_amplitude_pa),
        "target_center_x_mm": float(cx * 1e3),
        "target_center_y_mm": float(cy * 1e3),
        "anti_vortex_offset_x_mm": float(cfg.anti_vortex_offset_x_mm),
        "anti_vortex_offset_y_mm": float(cfg.anti_vortex_offset_y_mm),
        "target_asymmetry_boost": float(cfg.target_asymmetry_boost),
        "dx_um": float(dx * 1e6),
    }
    return p_target, amp_target, phase_target, meta


def build_three_region_masks(
    xx: np.ndarray,
    yy: np.ndarray,
    diameter_mm: float,
    cfg: StudyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diameter_m = diameter_mm * 1e-3
    r_outer = 0.5 * diameter_m
    radial_thickness_m = min(cfg.radial_thickness_mm * 1e-3, 0.95 * r_outer)
    r_inner = max(r_outer - radial_thickness_m, 0.15 * r_outer)

    cx = (cfg.target_center_x_mm + cfg.anti_vortex_offset_x_mm) * 1e-3
    cy = (cfg.target_center_y_mm + cfg.anti_vortex_offset_y_mm) * 1e-3

    x_rel = xx - cx
    y_rel = yy - cy
    rr = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)

    orientation = np.deg2rad(cfg.orientation_deg)
    theta_local = _wrap_to_pi(theta - orientation)

    ring_mask = (rr >= r_inner) & (rr <= r_outer)
    gap_half = 0.5 * np.deg2rad(cfg.gap_size_deg)
    gap_mask = ring_mask & (np.abs(theta_local) <= gap_half)
    support_mask = ring_mask & (~gap_mask)
    outside_mask = ~ring_mask
    return support_mask, gap_mask, outside_mask


def initialize_lens_field(
    aperture_mask: np.ndarray,
    cfg: StudyConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    phase = np.zeros_like(aperture_mask, dtype=float)
    if cfg.init_phase_mode == "random":
        phase[aperture_mask] = rng.uniform(-np.pi, np.pi, size=int(np.sum(aperture_mask)))
    elif cfg.init_phase_mode == "flat":
        phase[aperture_mask] = 0.0
    elif cfg.init_phase_mode == "weak-random":
        phase[aperture_mask] = rng.normal(0.0, cfg.weak_random_phase_std_rad, size=int(np.sum(aperture_mask)))
    else:
        raise ValueError(f"Unsupported init_phase_mode: {cfg.init_phase_mode}")

    p_lens = np.zeros_like(phase, dtype=complex)
    p_lens[aperture_mask] = np.exp(1j * phase[aperture_mask])
    return p_lens


def _build_weight_map(
    support_mask: np.ndarray,
    gap_mask: np.ndarray,
    outside_mask: np.ndarray,
    cfg: StudyConfig,
) -> np.ndarray:
    weights = np.zeros_like(support_mask, dtype=float)
    weights[support_mask] = cfg.support_weight
    weights[gap_mask] = cfg.gap_weight
    weights[outside_mask] = cfg.outside_weight
    return np.clip(weights, 0.0, 1.0)


def enforce_three_region_constraint(
    p_at_target: np.ndarray,
    amp_target: np.ndarray,
    phase_target: np.ndarray,
    support_mask: np.ndarray,
    gap_mask: np.ndarray,
    outside_mask: np.ndarray,
    cfg: StudyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp_est = np.abs(p_at_target)

    # Desired amplitude is explicit three-region: target on support, near-zero on gap and outside.
    amp_desired = np.zeros_like(amp_est)
    amp_desired[support_mask] = amp_target[support_mask]

    weights = _build_weight_map(
        support_mask=support_mask,
        gap_mask=gap_mask,
        outside_mask=outside_mask,
        cfg=cfg,
    )
    amp_constrained = (1.0 - weights) * amp_est + weights * amp_desired

    phase_est = np.angle(p_at_target)
    if cfg.target_phase_mode == "target-fixed":
        phase_constrained = phase_target
    elif cfg.target_phase_mode == "flat-support":
        phase_constrained = phase_est.copy()
        phase_constrained[support_mask] = 0.0
    elif cfg.target_phase_mode == "free":
        phase_constrained = phase_est
    else:
        raise ValueError(f"Unsupported target_phase_mode: {cfg.target_phase_mode}")

    p_target_constrained = amp_constrained * np.exp(1j * phase_constrained)
    return p_target_constrained, amp_desired, weights


def compute_three_region_metrics(
    amp_est: np.ndarray,
    amp_desired: np.ndarray,
    support_mask: np.ndarray,
    gap_mask: np.ndarray,
    outside_mask: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    total_energy = float(np.sum(amp_est**2))
    support_err = (
        float(np.sqrt(np.mean((amp_est[support_mask] - amp_desired[support_mask]) ** 2)))
        if np.any(support_mask)
        else float("nan")
    )
    gap_leakage = float(np.sum((amp_est[gap_mask]) ** 2) / max(total_energy, 1e-30))
    outside_leakage = float(np.sum((amp_est[outside_mask]) ** 2) / max(total_energy, 1e-30))

    weighted_den = float(np.sum(weights))
    weighted_rmse = (
        float(np.sqrt(np.sum(weights * (amp_est - amp_desired) ** 2) / max(weighted_den, 1e-30)))
        if weighted_den > 0.0
        else float("nan")
    )
    return {
        "support_error_rmse": float(support_err),
        "gap_leakage_frac": float(gap_leakage),
        "outside_support_leakage_frac": float(outside_leakage),
        "weighted_error_rmse": float(weighted_rmse),
    }


def compute_phase_winding(
    p_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    diameter_mm: float,
    cfg: StudyConfig,
    n_samples: int = 360,
) -> dict[str, float | int | bool]:
    cx = (cfg.target_center_x_mm + cfg.anti_vortex_offset_x_mm) * 1e-3
    cy = (cfg.target_center_y_mm + cfg.anti_vortex_offset_y_mm) * 1e-3

    dx = abs(float(x[1] - x[0]))
    diameter_m = diameter_mm * 1e-3
    probe_radius = max(0.22 * diameter_m, 4.0 * dx)

    angles = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    xs = cx + probe_radius * np.cos(angles)
    ys = cy + probe_radius * np.sin(angles)

    ix = np.rint((xs - x[0]) / dx).astype(int)
    iy = np.rint((ys - y[0]) / abs(float(y[1] - y[0]))).astype(int)
    ix = np.clip(ix, 0, x.size - 1)
    iy = np.clip(iy, 0, y.size - 1)

    samples = p_field[iy, ix]
    phases = np.angle(samples)
    dphi = np.angle(np.exp(1j * np.diff(np.concatenate([phases, phases[:1]]))))
    winding = float(np.sum(dphi) / (2.0 * np.pi))
    charge = int(np.round(winding))

    amp = np.abs(p_field)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    center_mask = rr <= max(2.0 * dx, 0.08 * diameter_m)
    center_amp = float(np.mean(amp[center_mask])) if np.any(center_mask) else float(np.mean(amp))
    peak_amp = float(np.max(amp))
    singularity = bool(abs(winding) >= 0.5 and center_amp <= 0.2 * max(peak_amp, 1e-30))

    return {
        "phase_winding": float(winding),
        "topological_charge": int(charge),
        "center_amp": float(center_amp),
        "peak_amp": float(peak_amp),
        "vortex_singularity": singularity,
    }


def plot_anti_vortex_diagnostics(history: dict[str, list[float]], save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))

    ax = axes[0, 0]
    ax.plot(history["iteration"], history["weighted_error_rmse"], lw=1.6, color="tab:blue")
    ax.set_title("Weighted Error RMSE")
    ax.set_xlabel("iteration")
    ax.set_ylabel("weighted RMSE")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history["iteration"], history["support_error_rmse"], lw=1.6, color="tab:green")
    ax.set_title("Support Error RMSE")
    ax.set_xlabel("iteration")
    ax.set_ylabel("support RMSE")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(history["iteration"], history["gap_leakage_frac"], lw=1.6, color="tab:red")
    ax.set_title("Gap Leakage")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy fraction")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["iteration"], history["outside_support_leakage_frac"], lw=1.6, color="tab:orange")
    ax.set_title("Outside-Support Leakage")
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy fraction")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history["iteration"], history["phase_winding"], lw=1.6, color="tab:purple")
    ax.axhline(0.0, color="k", ls="--", lw=0.9, alpha=0.6)
    ax.set_title("Phase Winding")
    ax.set_xlabel("iteration")
    ax.set_ylabel("winding number")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(history["iteration"], history["vortex_singularity"], lw=1.6, color="tab:brown")
    ax.set_title("Vortex Singularity Indicator")
    ax.set_xlabel("iteration")
    ax.set_ylabel("0/1")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def compute_snapshot_iterations(n_iter: int) -> list[int]:
    snapshots = [
        1,
        int(round(0.25 * n_iter)),
        int(round(0.50 * n_iter)),
        int(round(0.75 * n_iter)),
        n_iter,
    ]

    for idx in range(1, len(snapshots)):
        if snapshots[idx] <= snapshots[idx - 1]:
            snapshots[idx] = snapshots[idx - 1] + 1

    snapshots[-1] = n_iter

    if len(set(snapshots)) != 5:
        alt = np.linspace(1, n_iter, 5, dtype=int).tolist()
        alt[0] = 1
        alt[-1] = n_iter
        snapshots = alt

    if len(set(snapshots)) != 5:
        raise ValueError("Could not create 5 unique snapshot iterations; increase n_iter")

    return snapshots


def masked_gaussian_smooth(values: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0.0:
        return values.copy()

    val = np.where(mask, values, 0.0)
    wgt = mask.astype(float)

    num = gaussian_filter(val, sigma=sigma_px, mode="nearest")
    den = gaussian_filter(wgt, sigma=sigma_px, mode="nearest")

    out = np.full_like(values, np.nan, dtype=float)
    valid = (den > 1e-12) & mask
    out[valid] = num[valid] / den[valid]
    return out


def phase_to_thickness_maps(
    phase_raw: np.ndarray,
    mask: np.ndarray,
    dx: float,
    cfg: StudyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_wrapped = np.mod(phase_raw, 2.0 * np.pi)

    thickness_raw = np.full_like(phase_raw, np.nan, dtype=float)
    thickness_raw[mask] = (phase_wrapped[mask] / abs(cfg.dk)) + cfg.base_thickness_m

    sigma_px = (cfg.processed_sigma_mm * 1e-3) / max(dx, 1e-12)
    thickness_processed = masked_gaussian_smooth(thickness_raw, mask, sigma_px)

    if np.any(mask):
        fill_val = float(np.nanmedian(thickness_raw[mask]))
        fill_idx = mask & np.isnan(thickness_processed)
        thickness_processed[fill_idx] = fill_val
    thickness_processed = np.where(mask, thickness_processed, np.nan)

    return phase_wrapped, thickness_raw, thickness_processed


def save_map(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    title: str,
    cmap: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: np.ndarray | None = None,
    zoom_half_mm: float | None = None,
) -> None:
    show = data
    if mask is not None:
        show = np.where(mask, data, np.nan)

    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(
        show,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    _add_colorbar(ax, im, cbar_label)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if zoom_half_mm is not None:
        ax.set_xlim(-zoom_half_mm, zoom_half_mm)
        ax.set_ylim(-zoom_half_mm, zoom_half_mm)
    fig.tight_layout()
    fig.savefig(save_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_ideal_pressure_figure(
    p_target: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    title: str,
    zoom_half_mm: float | None = None,
) -> None:
    amp = np.abs(p_target)
    phase = np.angle(p_target)
    real = np.real(p_target)

    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    v_real = max(abs(float(real.min())), abs(float(real.max())), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2))

    ax = axes[0]
    im = ax.imshow(amp, origin="lower", extent=extent, cmap="inferno", aspect="equal")
    _add_colorbar(ax, im, "|p| (Pa)")
    ax.set_title("Ideal Pressure Amplitude")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = axes[1]
    im = ax.imshow(
        phase,
        origin="lower",
        extent=extent,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        aspect="equal",
    )
    _add_colorbar(ax, im, "phase (rad)")
    ax.set_title("Ideal Pressure Phase")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = axes[2]
    im = ax.imshow(
        real,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-v_real,
        vmax=v_real,
        aspect="equal",
    )
    _add_colorbar(ax, im, "Re(p) (Pa)")
    ax.set_title("Ideal Pressure Real Part")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    if zoom_half_mm is not None:
        for _ax in axes:
            _ax.set_xlim(-zoom_half_mm, zoom_half_mm)
            _ax.set_ylim(-zoom_half_mm, zoom_half_mm)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=195, bbox_inches="tight")
    plt.close(fig)


def save_thickness_3d(
    thickness_processed: np.ndarray,
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    """Save a 3-D surface plot of the lens thickness profile."""
    # Downsample to ~200 points per axis for performance
    n = max(x.size, y.size)
    step = max(1, n // 200)

    xs = x[::step] * 1e3
    ys = y[::step] * 1e3
    t_mm = thickness_processed[::step, ::step] * 1e3
    m_ds = mask[::step, ::step]

    # Replace outside-aperture with NaN so they are not drawn
    t_plot = np.where(m_ds, t_mm, np.nan)

    XX, YY = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        XX,
        YY,
        t_plot,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, label="thickness (mm)", pad=0.12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("thickness (mm)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_snapshot_bundle(
    snapshots: dict[int, dict[str, np.ndarray]],
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    case_fig_dir: Path,
    case_npz_dir: Path,
    zoom_half_mm: float | None = None,
) -> None:
    if len(snapshots) != 5:
        raise ValueError(f"Expected exactly 5 snapshots, got {len(snapshots)}")

    ordered_iters = sorted(snapshots.keys())

    for it in ordered_iters:
        snap = snapshots[it]
        save_map(
            data=np.abs(snap["p_target_field"]),
            x=x,
            y=y,
            save_path=case_fig_dir / f"snapshot_iter_{it:04d}_pressure.png",
            title=f"Snapshot Iteration {it}: Pressure Amplitude at Target Plane",
            cmap="inferno",
            cbar_label="|p| (Pa)",
            zoom_half_mm=zoom_half_mm,
        )
        save_map(
            data=np.angle(snap["p_target_field"]),
            x=x,
            y=y,
            save_path=case_fig_dir / f"snapshot_iter_{it:04d}_pressure_phase.png",
            title=f"Snapshot Iteration {it}: Pressure Phase at Target Plane",
            cmap="twilight",
            cbar_label="phase (rad)",
            vmin=-np.pi,
            vmax=np.pi,
            zoom_half_mm=zoom_half_mm,
        )
        save_map(
            data=snap["phase_wrapped"],
            x=x,
            y=y,
            save_path=case_fig_dir / f"snapshot_iter_{it:04d}_phase.png",
            title=f"Snapshot Iteration {it}: Lens Wrapped Phase",
            cmap="hsv",
            cbar_label="phase (rad)",
            vmin=0.0,
            vmax=2.0 * np.pi,
            mask=mask,
        )
        save_map(
            data=snap["thickness_processed"] * 1e3,
            x=x,
            y=y,
            save_path=case_fig_dir / f"snapshot_iter_{it:04d}_thickness_profile.png",
            title=f"Snapshot Iteration {it}: Thickness Profile",
            cmap="viridis",
            cbar_label="thickness (mm)",
            mask=mask,
        )

        np.savez_compressed(
            case_npz_dir / f"snapshot_iter_{it:04d}.npz",
            iteration=it,
            phase_raw=snap["phase_raw"],
            phase_wrapped=snap["phase_wrapped"],
            thickness_raw=snap["thickness_raw"],
            thickness_processed=snap["thickness_processed"],
            p_target_field=snap["p_target_field"],
            p_lens_field=snap["p_lens_field"],
        )

    # Combined 5-snapshot figure
    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    fig, axes = plt.subplots(4, 5, figsize=(21, 15.0))
    pressure_im = None
    pressure_phase_im = None
    lens_phase_im = None
    thick_im = None

    for col, it in enumerate(ordered_iters):
        snap = snapshots[it]

        ax_pa = axes[0, col]
        pressure_im = ax_pa.imshow(
            np.abs(snap["p_target_field"]),
            origin="lower",
            extent=extent,
            cmap="inferno",
            aspect="equal",
        )
        ax_pa.set_title(f"Iter {it}")
        ax_pa.set_xlabel("x (mm)")
        if col == 0:
            ax_pa.set_ylabel("pressure\namp\ny (mm)")

        ax_pp = axes[1, col]
        pressure_phase_im = ax_pp.imshow(
            np.angle(snap["p_target_field"]),
            origin="lower",
            extent=extent,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            aspect="equal",
        )
        ax_pp.set_xlabel("x (mm)")
        if col == 0:
            ax_pp.set_ylabel("pressure\nphase\ny (mm)")

        ax_p = axes[2, col]
        lens_phase_im = ax_p.imshow(
            np.where(mask, snap["phase_wrapped"], np.nan),
            origin="lower",
            extent=extent,
            cmap="hsv",
            vmin=0.0,
            vmax=2.0 * np.pi,
            aspect="equal",
        )
        ax_p.set_xlabel("x (mm)")
        if col == 0:
            ax_p.set_ylabel("lens\nphase\ny (mm)")

        if zoom_half_mm is not None:
            ax_pa.set_xlim(-zoom_half_mm, zoom_half_mm)
            ax_pa.set_ylim(-zoom_half_mm, zoom_half_mm)
            ax_pp.set_xlim(-zoom_half_mm, zoom_half_mm)
            ax_pp.set_ylim(-zoom_half_mm, zoom_half_mm)

        ax_t = axes[3, col]
        thick_im = ax_t.imshow(
            np.where(mask, snap["thickness_processed"] * 1e3, np.nan),
            origin="lower",
            extent=extent,
            cmap="viridis",
            aspect="equal",
        )
        ax_t.set_xlabel("x (mm)")
        if col == 0:
            ax_t.set_ylabel("thickness\ny (mm)")

    fig.subplots_adjust(left=0.05, right=0.90, top=0.93, bottom=0.06, wspace=0.25, hspace=0.24)

    if pressure_im is not None:
        cax_pa = fig.add_axes([0.92, 0.76, 0.012, 0.16])
        fig.colorbar(pressure_im, cax=cax_pa, label="|p| (Pa)")
    if pressure_phase_im is not None:
        cax_pp = fig.add_axes([0.92, 0.55, 0.012, 0.16])
        fig.colorbar(pressure_phase_im, cax=cax_pp, label="phase (rad)")
    if lens_phase_im is not None:
        cax_phase = fig.add_axes([0.92, 0.34, 0.012, 0.16])
        fig.colorbar(lens_phase_im, cax=cax_phase, label="phase (rad)")
    if thick_im is not None:
        cax_thick = fig.add_axes([0.92, 0.13, 0.012, 0.16])
        fig.colorbar(thick_im, cax=cax_thick, label="thickness (mm)")

    fig.suptitle(
        "Iteration Snapshots (exactly 5): Pressure, Pressure Phase, Lens Phase, Thickness",
        fontsize=13,
    )
    fig.savefig(case_fig_dir / "snapshot_combined_5panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history: dict[str, list[float]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))

    ax = axes[0]
    ax.plot(history["iteration"], history["amp_rmse"], color="tab:red", lw=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("amplitude RMSE")
    ax.set_title("Target Amplitude Error")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history["iteration"], history["amp_corr"], color="tab:blue", lw=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("amplitude correlation")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Target Amplitude Correlation")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def estimate_cshape_diameter_mm(
    amp: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    target_roi_mask: np.ndarray,
    threshold_frac: float,
) -> float:
    if amp.max() <= 0.0:
        return float("nan")

    binary = amp >= (threshold_frac * amp.max())
    lbl, n_lbl = label(binary)
    if n_lbl == 0:
        return float("nan")

    best_id = 1
    best_score = -1.0
    for comp_id in range(1, n_lbl + 1):
        comp = lbl == comp_id
        overlap = float(np.sum(comp & target_roi_mask))
        area = float(np.sum(comp))
        score = overlap + 0.1 * area
        if score > best_score:
            best_score = score
            best_id = comp_id

    comp = lbl == best_id
    ys, xs = np.where(comp)
    if ys.size < 4:
        return float("nan")

    weights = amp[ys, xs]
    x_coords = xx[ys, xs]
    y_coords = yy[ys, xs]

    wsum = float(np.sum(weights))
    if wsum <= 1e-30:
        cx = float(np.mean(x_coords))
        cy = float(np.mean(y_coords))
    else:
        cx = float(np.sum(weights * x_coords) / wsum)
        cy = float(np.sum(weights * y_coords) / wsum)

    rr = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    diameter_m = 2.0 * float(np.percentile(rr, 95.0))
    return diameter_m * 1e3


def compute_validation_metrics(
    p_target: np.ndarray,
    p_forward: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    target_diameter_mm: float,
    cfg: StudyConfig,
) -> dict[str, float]:
    amp_target = np.abs(p_target)
    amp_forward = np.abs(p_forward)

    target_roi_mask = amp_target >= (cfg.target_roi_threshold_frac * max(amp_target.max(), 1e-30))

    total_energy = float(np.sum(amp_forward**2))
    roi_energy = float(np.sum((amp_forward[target_roi_mask]) ** 2))
    leakage = 1.0 - (roi_energy / max(total_energy, 1e-30))

    target_radius_m = 0.5 * target_diameter_mm * 1e-3
    local_radius_m = cfg.localization_radius_factor * target_radius_m
    cx = cfg.target_center_x_mm * 1e-3
    cy = cfg.target_center_y_mm * 1e-3
    local_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= local_radius_m**2
    local_energy = float(np.sum((amp_forward[local_mask]) ** 2))
    localization = local_energy / max(total_energy, 1e-30)

    amp_corr = _corr(amp_forward, amp_target)

    measured_diameter_mm = estimate_cshape_diameter_mm(
        amp=amp_forward,
        xx=xx,
        yy=yy,
        target_roi_mask=target_roi_mask,
        threshold_frac=cfg.diameter_threshold_frac,
    )
    diameter_error_mm = float(abs(measured_diameter_mm - target_diameter_mm)) if np.isfinite(measured_diameter_mm) else float("nan")

    metrics = {
        "target_diameter_mm": float(target_diameter_mm),
        "measured_diameter_mm": float(measured_diameter_mm),
        "diameter_error_mm": float(diameter_error_mm),
        "localization": float(localization),
        "leakage_outside_roi": float(leakage),
        "amp_correlation": float(amp_corr),
        "target_peak_pa": float(amp_target.max()),
        "forward_peak_pa": float(amp_forward.max()),
    }
    return metrics


def is_achievable(metrics: dict[str, float], cfg: StudyConfig) -> tuple[bool, dict[str, bool]]:
    checks = {
        "diameter": (
            np.isfinite(metrics["diameter_error_mm"])
            and metrics["diameter_error_mm"] <= cfg.achievable_diameter_tol_mm
        ),
        "localization": metrics["localization"] >= cfg.achievable_min_localization,
        "leakage": metrics["leakage_outside_roi"] <= cfg.achievable_max_leakage,
        "amp_corr": metrics["amp_correlation"] >= cfg.achievable_min_amp_corr,
    }
    return all(checks.values()), checks


def save_validation_figure(
    p_target: np.ndarray,
    p_forward: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    metrics: dict[str, float],
    save_path: Path,
    zoom_half_mm: float | None = None,
) -> None:
    amp_t = np.abs(p_target)
    amp_f = np.abs(p_forward)
    err = amp_f - amp_t

    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    vmax = max(float(amp_t.max()), float(amp_f.max()), 1e-12)
    ev = max(abs(float(err.min())), abs(float(err.max())), 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))

    ax = axes[0, 0]
    im = ax.imshow(amp_t, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=vmax, aspect="equal")
    _add_colorbar(ax, im, "|p_target| (Pa)")
    ax.set_title("Target Amplitude")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = axes[0, 1]
    im = ax.imshow(amp_f, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=vmax, aspect="equal")
    _add_colorbar(ax, im, "|p_forward| (Pa)")
    ax.set_title("Forward Amplitude")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = axes[1, 0]
    im = ax.imshow(err, origin="lower", extent=extent, cmap="RdBu_r", vmin=-ev, vmax=ev, aspect="equal")
    _add_colorbar(ax, im, "|p_forward|-|p_target| (Pa)")
    ax.set_title("Amplitude Error")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "Validation Metrics",
        "",
        f"Target diameter:   {metrics['target_diameter_mm']:.3f} mm",
        f"Measured diameter: {metrics['measured_diameter_mm']:.3f} mm",
        f"Diameter error:    {metrics['diameter_error_mm']:.3f} mm",
        f"Localization:      {metrics['localization']:.4f}",
        f"Leakage outside ROI: {metrics['leakage_outside_roi']:.4f}",
        f"Amplitude correlation: {metrics['amp_correlation']:.4f}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)

    if zoom_half_mm is not None:
        for _ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            _ax.set_xlim(-zoom_half_mm, zoom_half_mm)
            _ax.set_ylim(-zoom_half_mm, zoom_half_mm)

    fig.tight_layout()
    fig.savefig(save_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def export_heightmap(
    thickness_processed: np.ndarray,
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    save_dir: Path,
) -> None:
    height_mm = np.where(mask, thickness_processed * 1e3, 0.0)
    np.save(save_dir / "heightmap_processed_mm.npy", height_mm)

    header = (
        "processed lens heightmap (mm); outside aperture is 0; "
        f"grid={height_mm.shape[0]}x{height_mm.shape[1]}; "
        f"x_mm=[{x[0]*1e3:.4f},{x[-1]*1e3:.4f}]; "
        f"y_mm=[{y[0]*1e3:.4f},{y[-1]*1e3:.4f}]"
    )
    np.savetxt(save_dir / "heightmap_processed_mm.csv", height_mm, delimiter=",", header=header)


def export_ascii_stl_if_requested(
    thickness_processed: np.ndarray,
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cfg: StudyConfig,
    save_path: Path,
) -> bool:
    if not cfg.export_stl:
        return False

    ny, nx = thickness_processed.shape
    step = max(1, int(math.ceil(max(ny, nx) / cfg.stl_max_grid)))
    x_mm = x * 1e3
    y_mm = y * 1e3
    z_mm = np.where(mask, thickness_processed * 1e3, 0.0)

    triangles = []
    for i in range(0, ny - step, step):
        for j in range(0, nx - step, step):
            i2 = min(i + step, ny - 1)
            j2 = min(j + step, nx - 1)
            if not (mask[i, j] and mask[i, j2] and mask[i2, j] and mask[i2, j2]):
                continue

            v00 = (x_mm[j], y_mm[i], z_mm[i, j])
            v01 = (x_mm[j2], y_mm[i], z_mm[i, j2])
            v10 = (x_mm[j], y_mm[i2], z_mm[i2, j])
            v11 = (x_mm[j2], y_mm[i2], z_mm[i2, j2])

            triangles.append((v00, v01, v10))
            triangles.append((v01, v11, v10))

    if not triangles:
        return False

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("solid inverse_c_shape_lens\n")
        for v0, v1, v2 in triangles:
            f.write("  facet normal 0 0 1\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid inverse_c_shape_lens\n")
    return True


def run_iterative_design_case(
    cfg: StudyConfig,
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    dx: float,
    dy: float,
    aperture_mask: np.ndarray,
    diameter_mm: float,
    out_case_dir: Path,
) -> dict[str, Any]:
    case_fig = out_case_dir / "figures"
    case_npz = out_case_dir / "npz"
    case_lens = out_case_dir / "lens_exports"
    for d in (case_fig, case_npz, case_lens):
        d.mkdir(parents=True, exist_ok=True)

    p_target, a_target, phi_target, target_meta = generate_c_shape_target(xx, yy, dx, diameter_mm, cfg)
    support_mask, gap_mask, outside_mask = build_three_region_masks(xx, yy, diameter_mm, cfg)

    # Zoom for pressure-field plots: show ~4x the C-shape diameter, minimum 1 mm half-window
    zoom_half_mm = max(diameter_mm * 2.0, 1.0)

    save_map(
        data=np.abs(p_target),
        x=x,
        y=y,
        save_path=case_fig / "target_amplitude.png",
        title=f"Target C-shape Amplitude (D={diameter_mm:.1f} mm)",
        cmap="inferno",
        cbar_label="|p_target| (Pa)",
        zoom_half_mm=zoom_half_mm,
    )
    save_map(
        data=np.angle(p_target),
        x=x,
        y=y,
        save_path=case_fig / "target_phase.png",
        title=f"Target C-shape Phase (D={diameter_mm:.1f} mm)",
        cmap="twilight",
        cbar_label="phase (rad)",
        vmin=-np.pi,
        vmax=np.pi,
        zoom_half_mm=zoom_half_mm,
    )
    save_ideal_pressure_figure(
        p_target=p_target,
        x=x,
        y=y,
        save_path=case_fig / "ideal_pressure_field.png",
        title=f"Ideal Target Pressure Field (D={diameter_mm:.1f} mm)",
        zoom_half_mm=zoom_half_mm,
    )
    save_map(
        data=support_mask.astype(float),
        x=x,
        y=y,
        save_path=case_fig / "target_support_mask.png",
        title=f"Target C-support Mask (D={diameter_mm:.1f} mm)",
        cmap="gray",
        cbar_label="support mask",
        vmin=0.0,
        vmax=1.0,
        zoom_half_mm=zoom_half_mm,
    )
    save_map(
        data=gap_mask.astype(float),
        x=x,
        y=y,
        save_path=case_fig / "target_gap_mask.png",
        title=f"Target C-gap Mask (D={diameter_mm:.1f} mm)",
        cmap="gray",
        cbar_label="gap mask",
        vmin=0.0,
        vmax=1.0,
        zoom_half_mm=zoom_half_mm,
    )

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
    p_lens = initialize_lens_field(
        aperture_mask=aperture_mask,
        cfg=cfg,
        rng=rng,
    )

    snapshot_iters = compute_snapshot_iterations(cfg.n_iter)
    snapshot_set = set(snapshot_iters)
    snapshots: dict[int, dict[str, np.ndarray]] = {}

    history: dict[str, list[float]] = {
        "iteration": [],
        "amp_rmse": [],
        "amp_corr": [],
        "support_error_rmse": [],
        "gap_leakage_frac": [],
        "outside_support_leakage_frac": [],
        "weighted_error_rmse": [],
        "phase_winding": [],
        "vortex_singularity": [],
    }

    print(f"  [case D={diameter_mm:.1f} mm] iterations={cfg.n_iter}, snapshots={snapshot_iters}")

    for it in range(1, cfg.n_iter + 1):
        # (a) lens -> target propagation
        p_at_target = asm_propagate(p_lens, plan_fwd)

        amp_est = np.abs(p_at_target)
        amp_rmse = float(np.sqrt(np.mean((amp_est - a_target) ** 2)))
        amp_corr = _corr(amp_est, a_target)

        history["iteration"].append(float(it))
        history["amp_rmse"].append(amp_rmse)
        history["amp_corr"].append(amp_corr)

        # (b) enforce target constraint with explicit three-region support/gap/outside rules
        p_target_constrained, amp_desired, weights = enforce_three_region_constraint(
            p_at_target=p_at_target,
            amp_target=a_target,
            phase_target=phi_target,
            support_mask=support_mask,
            gap_mask=gap_mask,
            outside_mask=outside_mask,
            cfg=cfg,
        )

        region_metrics = compute_three_region_metrics(
            amp_est=amp_est,
            amp_desired=amp_desired,
            support_mask=support_mask,
            gap_mask=gap_mask,
            outside_mask=outside_mask,
            weights=weights,
        )
        winding_metrics = compute_phase_winding(
            p_field=p_at_target,
            x=x,
            y=y,
            xx=xx,
            yy=yy,
            diameter_mm=diameter_mm,
            cfg=cfg,
        )

        history["support_error_rmse"].append(region_metrics["support_error_rmse"])
        history["gap_leakage_frac"].append(region_metrics["gap_leakage_frac"])
        history["outside_support_leakage_frac"].append(region_metrics["outside_support_leakage_frac"])
        history["weighted_error_rmse"].append(region_metrics["weighted_error_rmse"])
        history["phase_winding"].append(winding_metrics["phase_winding"])
        history["vortex_singularity"].append(1.0 if winding_metrics["vortex_singularity"] else 0.0)

        # (c) target -> lens propagation
        p_back_to_lens = asm_propagate(p_target_constrained, plan_bwd)

        # (d) enforce lens constraints (circular aperture + phase-only)
        p_lens = np.zeros_like(p_lens)
        p_lens[aperture_mask] = np.exp(1j * np.angle(p_back_to_lens[aperture_mask]))

        if it in snapshot_set:
            phase_raw = np.angle(p_lens)
            phase_wrapped, thickness_raw, thickness_processed = phase_to_thickness_maps(
                phase_raw=phase_raw,
                mask=aperture_mask,
                dx=dx,
                cfg=cfg,
            )
            p_target_snapshot = asm_propagate(p_lens, plan_fwd)
            snapshots[it] = {
                "phase_raw": phase_raw.copy(),
                "phase_wrapped": phase_wrapped.copy(),
                "thickness_raw": thickness_raw.copy(),
                "thickness_processed": thickness_processed.copy(),
                "p_target_field": p_target_snapshot.copy(),
                "p_lens_field": p_lens.copy(),
            }

        stride = max(1, cfg.n_iter // 10)
        if it == 1 or it % stride == 0 or it == cfg.n_iter:
            print(
                f"    iter {it:4d}/{cfg.n_iter}: "
                f"amp_rmse={amp_rmse:.5f}, amp_corr={amp_corr:.4f}, "
                f"support_err={region_metrics['support_error_rmse']:.5f}, "
                f"gap_leak={region_metrics['gap_leakage_frac']:.4f}, "
                f"winding={winding_metrics['phase_winding']:.3f}"
            )

    if len(snapshots) != 5:
        raise RuntimeError(f"Expected exactly 5 snapshots, got {len(snapshots)}")

    save_snapshot_bundle(
        snapshots=snapshots,
        x=x,
        y=y,
        mask=aperture_mask,
        case_fig_dir=case_fig,
        case_npz_dir=case_npz,
        zoom_half_mm=zoom_half_mm,
    )

    # Final lens state
    phase_raw_final = np.angle(p_lens)
    phase_wrapped_final, thickness_raw_final, thickness_processed_final = phase_to_thickness_maps(
        phase_raw=phase_raw_final,
        mask=aperture_mask,
        dx=dx,
        cfg=cfg,
    )

    save_map(
        data=phase_raw_final,
        x=x,
        y=y,
        save_path=case_fig / "final_phase_raw.png",
        title=f"Final Lens Phase Raw (D={diameter_mm:.1f} mm)",
        cmap="twilight",
        cbar_label="phase (rad)",
        vmin=-np.pi,
        vmax=np.pi,
        mask=aperture_mask,
    )
    save_map(
        data=phase_wrapped_final,
        x=x,
        y=y,
        save_path=case_fig / "final_phase_wrapped.png",
        title=f"Final Lens Phase Wrapped (D={diameter_mm:.1f} mm)",
        cmap="hsv",
        cbar_label="phase (rad)",
        vmin=0.0,
        vmax=2.0 * np.pi,
        mask=aperture_mask,
    )
    save_map(
        data=thickness_raw_final * 1e3,
        x=x,
        y=y,
        save_path=case_fig / "final_thickness_raw.png",
        title=f"Final Raw Thickness (D={diameter_mm:.1f} mm)",
        cmap="viridis",
        cbar_label="thickness (mm)",
        mask=aperture_mask,
    )
    save_map(
        data=thickness_processed_final * 1e3,
        x=x,
        y=y,
        save_path=case_fig / "final_thickness_processed.png",
        title=f"Final Processed Thickness (D={diameter_mm:.1f} mm)",
        cmap="viridis",
        cbar_label="thickness (mm)",
        mask=aperture_mask,
    )
    save_thickness_3d(
        thickness_processed=thickness_processed_final,
        mask=aperture_mask,
        x=x,
        y=y,
        save_path=case_fig / "final_thickness_3d.png",
        title=f"Final Lens Thickness 3-D Surface (D={diameter_mm:.1f} mm)",
    )

    plot_convergence(history, case_fig / "convergence.png")
    plot_anti_vortex_diagnostics(history, case_fig / "anti_vortex_diagnostics.png")

    # Forward validation with final phase-only lens field
    p_forward = asm_propagate(p_lens, plan_fwd)
    metrics = compute_validation_metrics(
        p_target=p_target,
        p_forward=p_forward,
        xx=xx,
        yy=yy,
        target_diameter_mm=diameter_mm,
        cfg=cfg,
    )

    amp_forward = np.abs(p_forward)
    amp_desired_final = np.zeros_like(amp_forward)
    amp_desired_final[support_mask] = a_target[support_mask]
    weights_final = _build_weight_map(
        support_mask=support_mask,
        gap_mask=gap_mask,
        outside_mask=outside_mask,
        cfg=cfg,
    )
    region_final = compute_three_region_metrics(
        amp_est=amp_forward,
        amp_desired=amp_desired_final,
        support_mask=support_mask,
        gap_mask=gap_mask,
        outside_mask=outside_mask,
        weights=weights_final,
    )
    winding_final = compute_phase_winding(
        p_field=p_forward,
        x=x,
        y=y,
        xx=xx,
        yy=yy,
        diameter_mm=diameter_mm,
        cfg=cfg,
    )

    metrics["support_error_rmse"] = float(region_final["support_error_rmse"])
    metrics["gap_leakage_frac"] = float(region_final["gap_leakage_frac"])
    metrics["outside_support_leakage_frac"] = float(region_final["outside_support_leakage_frac"])
    metrics["weighted_error_rmse"] = float(region_final["weighted_error_rmse"])
    metrics["phase_winding"] = float(winding_final["phase_winding"])
    metrics["topological_charge"] = int(winding_final["topological_charge"])
    metrics["vortex_singularity"] = bool(winding_final["vortex_singularity"])
    metrics["target_phase_mode"] = str(cfg.target_phase_mode)
    metrics["init_phase_mode"] = str(cfg.init_phase_mode)
    achievable, checks = is_achievable(metrics, cfg)

    metrics["achievable"] = bool(achievable)
    metrics["check_diameter"] = bool(checks["diameter"])
    metrics["check_localization"] = bool(checks["localization"])
    metrics["check_leakage"] = bool(checks["leakage"])
    metrics["check_amp_corr"] = bool(checks["amp_corr"])

    save_validation_figure(
        p_target=p_target,
        p_forward=p_forward,
        x=x,
        y=y,
        metrics=metrics,
        save_path=case_fig / "forward_validation.png",
        zoom_half_mm=zoom_half_mm,
    )

    export_heightmap(
        thickness_processed=thickness_processed_final,
        mask=aperture_mask,
        x=x,
        y=y,
        save_dir=case_lens,
    )
    stl_written = export_ascii_stl_if_requested(
        thickness_processed=thickness_processed_final,
        mask=aperture_mask,
        x=x,
        y=y,
        cfg=cfg,
        save_path=case_lens / "lens_surface.stl",
    )

    np.savez_compressed(
        case_npz / "final_design_data.npz",
        p_target=p_target,
        p_forward=p_forward,
        p_lens=p_lens,
        phase_raw=phase_raw_final,
        phase_wrapped=phase_wrapped_final,
        thickness_raw=thickness_raw_final,
        thickness_processed=thickness_processed_final,
        aperture_mask=aperture_mask,
        target_support_mask=support_mask,
        target_gap_mask=gap_mask,
        target_outside_mask=outside_mask,
        x=x,
        y=y,
        history_iteration=np.asarray(history["iteration"]),
        history_amp_rmse=np.asarray(history["amp_rmse"]),
        history_amp_corr=np.asarray(history["amp_corr"]),
        history_support_error_rmse=np.asarray(history["support_error_rmse"]),
        history_gap_leakage_frac=np.asarray(history["gap_leakage_frac"]),
        history_outside_support_leakage_frac=np.asarray(history["outside_support_leakage_frac"]),
        history_weighted_error_rmse=np.asarray(history["weighted_error_rmse"]),
        history_phase_winding=np.asarray(history["phase_winding"]),
        history_vortex_singularity=np.asarray(history["vortex_singularity"]),
    )

    np.savez_compressed(case_npz / "target_data.npz", p_target=p_target, amp_target=a_target, phase_target=phi_target, x=x, y=y)

    case_info = {
        "target_meta": target_meta,
        "metrics": metrics,
        "anti_vortex_settings": {
            "target_phase_mode": cfg.target_phase_mode,
            "init_phase_mode": cfg.init_phase_mode,
            "support_weight": cfg.support_weight,
            "gap_weight": cfg.gap_weight,
            "outside_weight": cfg.outside_weight,
            "anti_vortex_offset_x_mm": cfg.anti_vortex_offset_x_mm,
            "anti_vortex_offset_y_mm": cfg.anti_vortex_offset_y_mm,
            "target_asymmetry_boost": cfg.target_asymmetry_boost,
        },
        "snapshot_iterations": snapshot_iters,
        "stl_written": bool(stl_written),
        "output_dir": str(out_case_dir),
    }

    with open(out_case_dir / "case_info.json", "w", encoding="utf-8") as f:
        json.dump(case_info, f, indent=2)

    return case_info


def write_metrics_csv(metrics_rows: list[dict[str, Any]], save_path: Path) -> None:
    if not metrics_rows:
        return

    keys: list[str] = []
    seen = set()
    for row in metrics_rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)


def clone_config(cfg: StudyConfig, **updates: Any) -> StudyConfig:
    cfg_data = asdict(cfg)
    cfg_data.update(updates)
    return StudyConfig(**cfg_data)


def run_four_mode_comparison(
    cfg: StudyConfig,
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    dx: float,
    dy: float,
    aperture_mask: np.ndarray,
    diameter_mm: float,
    run_dir: Path,
    run_symmetry_break_test: bool,
    symmetry_break_offset_mm: float,
    symmetry_break_asymmetry_boost: float,
) -> None:
    compare_dir = run_dir / "four_mode_comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    modes = [
        ("free_phase_random_init", "free", "random"),
        ("free_phase_flat_init", "free", "flat"),
        ("flat_phase_random_init", "flat-support", "random"),
        ("flat_phase_flat_init", "flat-support", "flat"),
    ]

    scenarios = [("centered", 0.0, 0.0)]
    if run_symmetry_break_test:
        scenarios.append(("symmetry_break", symmetry_break_offset_mm, symmetry_break_asymmetry_boost))

    summary_rows: list[dict[str, Any]] = []

    for scenario_name, scenario_offset_mm, scenario_asymmetry in scenarios:
        for mode_name, phase_mode, init_mode in modes:
            case_name = f"{scenario_name}_{mode_name}"
            case_dir = compare_dir / case_name
            cfg_mode = clone_config(
                cfg,
                target_phase_mode=phase_mode,
                init_phase_mode=init_mode,
                anti_vortex_offset_x_mm=cfg.anti_vortex_offset_x_mm + scenario_offset_mm,
                anti_vortex_offset_y_mm=cfg.anti_vortex_offset_y_mm,
                target_asymmetry_boost=max(cfg.target_asymmetry_boost, scenario_asymmetry),
            )

            print(f"\n[compare] {case_name}")
            case_info = run_iterative_design_case(
                cfg=cfg_mode,
                x=x,
                y=y,
                xx=xx,
                yy=yy,
                dx=dx,
                dy=dy,
                aperture_mask=aperture_mask,
                diameter_mm=diameter_mm,
                out_case_dir=case_dir,
            )

            m = case_info["metrics"]
            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "mode": mode_name,
                    "target_phase_mode": phase_mode,
                    "init_phase_mode": init_mode,
                    "diameter_mm": diameter_mm,
                    "measured_diameter_mm": float(m.get("measured_diameter_mm", float("nan"))),
                    "amp_correlation": float(m.get("amp_correlation", float("nan"))),
                    "support_error_rmse": float(m.get("support_error_rmse", float("nan"))),
                    "gap_leakage_frac": float(m.get("gap_leakage_frac", float("nan"))),
                    "outside_support_leakage_frac": float(m.get("outside_support_leakage_frac", float("nan"))),
                    "weighted_error_rmse": float(m.get("weighted_error_rmse", float("nan"))),
                    "phase_winding": float(m.get("phase_winding", float("nan"))),
                    "topological_charge": int(m.get("topological_charge", 0)),
                    "vortex_singularity": bool(m.get("vortex_singularity", False)),
                    "achievable": bool(m.get("achievable", False)),
                    "case_dir": str(case_dir),
                }
            )

            print(
                "  -> corr={:.4f}, support_err={:.4f}, gap_leak={:.4f}, "
                "outside_leak={:.4f}, winding={:.3f}, charge={}, vortex={}, achievable={}".format(
                    float(m.get("amp_correlation", float("nan"))),
                    float(m.get("support_error_rmse", float("nan"))),
                    float(m.get("gap_leakage_frac", float("nan"))),
                    float(m.get("outside_support_leakage_frac", float("nan"))),
                    float(m.get("phase_winding", float("nan"))),
                    int(m.get("topological_charge", 0)),
                    bool(m.get("vortex_singularity", False)),
                    bool(m.get("achievable", False)),
                )
            )

    write_metrics_csv(summary_rows, compare_dir / "four_mode_comparison.csv")
    with open(compare_dir / "four_mode_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)


def write_summary_md(
    cfg: StudyConfig,
    run_dir: Path,
    dx: float,
    metrics_rows: list[dict[str, Any]],
    runtime_s: float,
) -> None:
    line_rows = []
    for r in metrics_rows:
        line_rows.append(
            (
                "| {target:.1f} | {meas:.3f} | {err:.3f} | {loc:.4f} | {leak:.4f} | {corr:.4f} | "
                "{s_err:.4f} | {g_leak:.4f} | {o_leak:.4f} | {winding:.3f} | {charge:d} | {vortex} | {ok} |"
            ).format(
                target=float(r["target_diameter_mm"]),
                meas=float(r["measured_diameter_mm"]),
                err=float(r["diameter_error_mm"]),
                loc=float(r["localization"]),
                leak=float(r["leakage_outside_roi"]),
                corr=float(r["amp_correlation"]),
                s_err=float(r.get("support_error_rmse", float("nan"))),
                g_leak=float(r.get("gap_leakage_frac", float("nan"))),
                o_leak=float(r.get("outside_support_leakage_frac", float("nan"))),
                winding=float(r.get("phase_winding", float("nan"))),
                charge=int(r.get("topological_charge", 0)),
                vortex="yes" if bool(r.get("vortex_singularity", False)) else "no",
                ok="yes" if bool(r["achievable"]) else "no",
            )
        )

    row_04 = next((r for r in metrics_rows if abs(float(r["target_diameter_mm"]) - 0.4) < 1e-9), None)
    if row_04 is None:
        verdict_04 = "0.4 mm case not run."
    else:
        verdict_04 = "achievable" if bool(row_04["achievable"]) else "not achieved"
        verdict_04 = (
            f"0.4 mm case: {verdict_04}. "
            f"(diameter check={row_04['check_diameter']}, "
            f"localization check={row_04['check_localization']}, "
            f"leakage check={row_04['check_leakage']}, "
            f"amp-corr check={row_04['check_amp_corr']})"
        )

    md = f"""# Inverse C-Shape Lens Study

Generated: {datetime.now().isoformat(timespec='seconds')}

## Goal

Design a passive phase lens for a 20 mm circular transducer (2 MHz, water)
that generates a local C-shaped pressure feature, with target outer diameters
0.4 to 0.7 mm.

## Implemented Iterative Loop

1. Define target complex field at target plane:
   p_target(x,y) = A_target(x,y) * exp(i * phi_target(x,y)).
2. Initialize lens plane field with random phase inside circular aperture:
   p_lens = exp(i * random_phase).
3. Repeat for N iterations:
   a) propagate lens -> target with angular spectrum,
   b) enforce target constraint (amplitude replaced by A_target; phase kept from propagation unless target-phase mode is enabled),
   c) propagate target -> lens,
   d) enforce lens constraints (circular aperture, phase-only, unit amplitude inside aperture).
4. After convergence, extract lens phase.
5. Convert phase to thickness:
   t = phi / |k_lens - k_water| + base_thickness.
6. Export thickness maps.
7. Forward propagate for validation.

## Numerical Domain

- Domain size: {cfg.domain_size_mm:.2f} mm x {cfg.domain_size_mm:.2f} mm
- Grid size: {cfg.n_grid} x {cfg.n_grid}
- dx = dy: {dx*1e6:.3f} um
- Wavelength (water): {cfg.wavelength_m*1e3:.4f} mm
- Propagation distance: {cfg.propagation_distance_mm:.3f} mm
- Aperture diameter: {cfg.transducer_diameter_mm:.1f} mm
- Pad factor: {cfg.pad_factor}
- Evanescent handling: {'included' if cfg.include_evanescent else 'suppressed'}

## Snapshot Policy

Exactly 5 snapshots were saved per target diameter:
iteration 1, 25%, 50%, 75%, and final.
Each snapshot includes pressure amplitude map, pressure phase map,
lens phase map, thickness-profile map, and NPZ arrays.

## Validation Metrics

| Target D (mm) | Measured D (mm) | Error (mm) | Localization | Leakage outside ROI | Amp Corr | Support Err | Gap Leak | Outside Leak | Winding | Charge | Vortex | 0.4-achievable criteria pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(line_rows)}

## 0.4 mm Assessment

{verdict_04}

## Outputs

- config.json
- metrics.csv
- summary.md
- Per-case figures: ideal pressure field, target maps, convergence, final phase/thickness, forward validation
- Per-case 5 snapshots and combined snapshot figure
- Per-case NPZ data
- Per-case heightmap exports (CSV + NPY)
- STL export: {'enabled' if cfg.export_stl else 'not requested'}

## Runtime

{runtime_s:.2f} s
"""

    with open(run_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative inverse-design workflow for local C-shaped acoustic lens"
    )

    parser.add_argument("--target-diameters-mm", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--n-iter", type=int, default=180)
    parser.add_argument("--enforce-target-phase", action="store_true")
    parser.add_argument(
        "--target-phase-mode",
        type=str,
        choices=["free", "flat-support", "target-fixed"],
        default="free",
    )
    parser.add_argument(
        "--init-phase-mode",
        type=str,
        choices=["random", "flat", "weak-random"],
        default="random",
    )
    parser.add_argument("--weak-random-phase-std-rad", type=float, default=0.12)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--outside-weight", type=float, default=0.35)

    parser.add_argument("--domain-size-mm", type=float, default=28.0)
    parser.add_argument("--n-grid", type=int, default=1024)
    parser.add_argument("--pad-factor", type=int, default=2)
    parser.add_argument("--include-evanescent", action="store_true")

    parser.add_argument("--propagation-distance-mm", type=float, default=6.0)
    parser.add_argument("--transducer-diameter-mm", type=float, default=20.0)

    parser.add_argument("--radial-thickness-mm", type=float, default=0.12)
    parser.add_argument("--gap-size-deg", type=float, default=70.0)
    parser.add_argument("--orientation-deg", type=float, default=0.0)
    parser.add_argument("--phase-ramp-rad", type=float, default=0.0)
    parser.add_argument("--anti-vortex-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--anti-vortex-offset-y-mm", type=float, default=0.0)
    parser.add_argument("--target-asymmetry-boost", type=float, default=0.0)

    parser.add_argument("--base-thickness-mm", type=float, default=0.60)
    parser.add_argument("--processed-sigma-mm", type=float, default=0.03)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export-stl", action="store_true")
    parser.add_argument("--stl-max-grid", type=int, default=240)

    parser.add_argument("--run-four-mode-comparison", action="store_true")
    parser.add_argument("--comparison-diameter-mm", type=float, default=1.5)
    parser.add_argument("--run-symmetry-break-test", action="store_true")
    parser.add_argument("--symmetry-break-offset-mm", type=float, default=0.20)
    parser.add_argument("--symmetry-break-asymmetry-boost", type=float, default=0.55)

    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> StudyConfig:
    phase_mode = "target-fixed" if bool(args.enforce_target_phase) else str(args.target_phase_mode)

    return StudyConfig(
        transducer_diameter_mm=float(args.transducer_diameter_mm),
        propagation_distance_mm=float(args.propagation_distance_mm),
        include_evanescent=bool(args.include_evanescent),
        pad_factor=int(args.pad_factor),
        domain_size_mm=float(args.domain_size_mm),
        n_grid=int(args.n_grid),
        n_iter=int(args.n_iter),
        enforce_target_phase=bool(args.enforce_target_phase),
        target_phase_mode=phase_mode,
        init_phase_mode=str(args.init_phase_mode),
        weak_random_phase_std_rad=float(args.weak_random_phase_std_rad),
        support_weight=float(args.support_weight),
        gap_weight=float(args.gap_weight),
        outside_weight=float(args.outside_weight),
        seed=int(args.seed),
        target_diameters_mm=tuple(float(v) for v in args.target_diameters_mm),
        radial_thickness_mm=float(args.radial_thickness_mm),
        gap_size_deg=float(args.gap_size_deg),
        orientation_deg=float(args.orientation_deg),
        phase_ramp_rad=float(args.phase_ramp_rad),
        anti_vortex_offset_x_mm=float(args.anti_vortex_offset_x_mm),
        anti_vortex_offset_y_mm=float(args.anti_vortex_offset_y_mm),
        target_asymmetry_boost=float(args.target_asymmetry_boost),
        base_thickness_mm=float(args.base_thickness_mm),
        processed_sigma_mm=float(args.processed_sigma_mm),
        export_stl=bool(args.export_stl),
        stl_max_grid=int(args.stl_max_grid),
    )


def main() -> None:
    t0 = time.time()
    args = parse_args()
    cfg = build_config_from_args(args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = RESULTS_ROOT / f"inverse_c_shape_lens_study_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    x, y, xx, yy, dx, dy = make_grid(cfg)
    aperture_mask = make_aperture(xx, yy, cfg.transducer_diameter_m)

    print("=" * 76)
    print("Iterative Inverse C-shape Lens Study")
    print("=" * 76)
    print(f"Output directory: {run_dir}")
    print(f"Domain size: {cfg.domain_size_mm:.3f} mm")
    print(f"Grid size: {cfg.n_grid} x {cfg.n_grid}")
    print(f"dx = dy: {dx*1e6:.3f} um")
    print(f"Wavelength: {cfg.wavelength_m*1e3:.4f} mm")
    print(f"Propagation distance: {cfg.propagation_distance_mm:.4f} mm")
    print(f"Aperture diameter: {cfg.transducer_diameter_mm:.3f} mm")
    print(f"k_water: {cfg.k_water:.3f} rad/m")
    print(f"k_lens: {cfg.k_lens:.3f} rad/m")
    print(f"Delta k = k_lens - k_water: {cfg.dk:.3f} rad/m")
    print(f"2pi/|Delta k| thickness: {cfg.full_2pi_thickness_m*1e3:.4f} mm")
    print(f"Target phase mode: {cfg.target_phase_mode}")
    print(f"Initialization mode: {cfg.init_phase_mode}")
    print(
        "Constraint weights (support/gap/outside): "
        f"{cfg.support_weight:.2f}/{cfg.gap_weight:.2f}/{cfg.outside_weight:.2f}"
    )
    print(
        "Symmetry break (offset_x_mm, offset_y_mm, asym_boost): "
        f"{cfg.anti_vortex_offset_x_mm:.3f}, {cfg.anti_vortex_offset_y_mm:.3f}, {cfg.target_asymmetry_boost:.3f}"
    )
    print(f"Target diameters: {list(cfg.target_diameters_mm)} mm")
    print("=" * 76)

    config_payload: dict[str, Any] = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "derived": {
            "wavelength_mm": cfg.wavelength_m * 1e3,
            "k_water": cfg.k_water,
            "k_lens": cfg.k_lens,
            "delta_k": cfg.dk,
            "full_2pi_thickness_mm": cfg.full_2pi_thickness_m * 1e3,
            "dx_um": dx * 1e6,
            "dy_um": dy * 1e6,
            "domain_size_mm": cfg.domain_size_mm,
            "grid_size": cfg.n_grid,
            "propagation_distance_mm": cfg.propagation_distance_mm,
            "aperture_diameter_mm": cfg.transducer_diameter_mm,
        },
    }

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    metrics_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for d_mm in cfg.target_diameters_mm:
        case_tag = f"D_{str(d_mm).replace('.', 'p')}mm"
        out_case_dir = run_dir / f"case_{case_tag}"

        print(f"\n[run] Designing case target diameter D={d_mm:.1f} mm")
        case_info = run_iterative_design_case(
            cfg=cfg,
            x=x,
            y=y,
            xx=xx,
            yy=yy,
            dx=dx,
            dy=dy,
            aperture_mask=aperture_mask,
            diameter_mm=d_mm,
            out_case_dir=out_case_dir,
        )

        metrics = dict(case_info["metrics"])
        metrics["case_dir"] = str(out_case_dir)
        metrics_rows.append(metrics)
        case_summaries.append(
            {
                "diameter_mm": d_mm,
                "case_dir": str(out_case_dir),
                "snapshot_iterations": case_info["snapshot_iterations"],
                "stl_written": case_info["stl_written"],
            }
        )

        print(
            f"  -> measured D={metrics['measured_diameter_mm']:.3f} mm, "
            f"localization={metrics['localization']:.4f}, "
            f"leakage={metrics['leakage_outside_roi']:.4f}, "
            f"corr={metrics['amp_correlation']:.4f}, "
            f"support_err={metrics.get('support_error_rmse', float('nan')):.4f}, "
            f"gap_leak={metrics.get('gap_leakage_frac', float('nan')):.4f}, "
            f"outside_leak={metrics.get('outside_support_leakage_frac', float('nan')):.4f}, "
            f"winding={metrics.get('phase_winding', float('nan')):.3f}, "
            f"charge={metrics.get('topological_charge', 0)}, "
            f"vortex={metrics.get('vortex_singularity', False)}, "
            f"achievable={metrics['achievable']}"
        )

    if args.run_four_mode_comparison:
        print("\n[compare] Running four-mode anti-vortex comparison")
        run_four_mode_comparison(
            cfg=cfg,
            x=x,
            y=y,
            xx=xx,
            yy=yy,
            dx=dx,
            dy=dy,
            aperture_mask=aperture_mask,
            diameter_mm=float(args.comparison_diameter_mm),
            run_dir=run_dir,
            run_symmetry_break_test=bool(args.run_symmetry_break_test),
            symmetry_break_offset_mm=float(args.symmetry_break_offset_mm),
            symmetry_break_asymmetry_boost=float(args.symmetry_break_asymmetry_boost),
        )

    write_metrics_csv(metrics_rows, run_dir / "metrics.csv")

    runtime_s = time.time() - t0
    write_summary_md(
        cfg=cfg,
        run_dir=run_dir,
        dx=dx,
        metrics_rows=metrics_rows,
        runtime_s=runtime_s,
    )

    final_report = {
        "runtime_s": runtime_s,
        "cases": case_summaries,
    }
    with open(run_dir / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print("\n" + "=" * 76)
    print(f"Done in {runtime_s:.2f} s")
    print(f"Wrote: {run_dir / 'config.json'}")
    print(f"Wrote: {run_dir / 'metrics.csv'}")
    print(f"Wrote: {run_dir / 'summary.md'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
