"""
Utility helpers for the symmetric-vortex limit study.

This module centralizes reusable pieces used by scripts in
scripts/deliverables/vortex_limit_*.py:

- trap loading and stable-minima detection
- symmetric vortex source construction
- potential/force post-processing
- discrete equilibrium and release descent on 2-D grids
- case classification and best-candidate selection
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import minimum_filter

from scripts.lib.asm_utils import LAM, make_lens_phase, make_vortex_field, propagate_asm
from scripts.lib.fem_cache_utils import (
    C_WATER,
    OMEGA,
    RHO0,
    default_particle_params,
    gorkov_grid_2d,
)
from scripts.lib.overlay_utils import (
    choose_adjacent_trap_pair,
    estimate_ring_radius,
    select_neighbour_traps,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def load_traps_from_json(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Trap file not found: {path}")
    data = json.loads(path.read_text())
    traps = data.get("traps", [])
    if len(traps) < 2:
        raise ValueError("Trap file has fewer than 2 traps")
    return np.array([[t["x_mm"] * 1e-3, t["y_mm"] * 1e-3] for t in traps], dtype=float)


def detect_stable_traps(
    U: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    expected_spacing: float = LAM / 2.0,
) -> np.ndarray:
    """Detect Hessian-stable local minima on a regular grid."""
    dx = float(xg[1] - xg[0])
    min_sep_pts = max(3, int(0.4 * expected_spacing / dx))
    nbhd = 2 * min_sep_pts + 1

    local_min = minimum_filter(U, size=nbhd)
    mask = U == local_min

    border = max(5, min_sep_pts)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    U_range = float(U.max() - U.min())
    mask &= U < (U.min() + 0.50 * U_range)

    iy, ix = np.where(mask)

    dUdx = np.gradient(U, dx, axis=1)
    dUdy = np.gradient(U, dx, axis=0)
    Hxx = np.gradient(dUdx, dx, axis=1)
    Hyy = np.gradient(dUdy, dx, axis=0)
    Hxy = np.gradient(dUdx, dx, axis=0)

    traps: List[Tuple[float, float]] = []
    for k in range(len(iy)):
        hxx = Hxx[iy[k], ix[k]]
        hyy = Hyy[iy[k], ix[k]]
        hxy = Hxy[iy[k], ix[k]]
        tr = hxx + hyy
        det = hxx * hyy - hxy * hxy
        disc = max(0.0, tr * tr - 4.0 * det)
        lam_min = 0.5 * (tr - np.sqrt(disc))
        if lam_min > 0.0:
            traps.append((float(xg[ix[k]]), float(yg[iy[k]])))

    if not traps:
        return np.zeros((0, 2), dtype=float)
    return np.array(traps, dtype=float)


def select_local_scenario(
    traps_m: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    lam: float = LAM,
    neighbor_radius: float = 1.2,
    min_neighbors: int = 6,
) -> Dict[str, Any]:
    """Pick adjacent A/B pair near center and nearby neighbor traps."""
    center = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])], dtype=float)
    pair = choose_adjacent_trap_pair(traps_m, target_spacing=lam / 2.0, domain_center=center)

    idx_a = int(pair["idx_A"])
    idx_b = int(pair["idx_B"])
    midpoint = np.asarray(pair["midpoint"], dtype=float)

    neigh = select_neighbour_traps(
        traps_m,
        idx_a,
        idx_b,
        midpoint,
        radius=neighbor_radius * lam,
        min_count=min_neighbors,
    )

    return {
        "idx_A": idx_a,
        "idx_B": idx_b,
        "A_xy": traps_m[idx_a].copy(),
        "B_xy": traps_m[idx_b].copy(),
        "d_AB": float(pair["d_AB"]),
        "e_AB": np.asarray(pair["e_AB"], dtype=float),
        "midpoint": midpoint,
        "neighbor_idx": np.asarray(neigh, dtype=int),
    }


def compute_gorkov_fields(p_grid: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ppar = default_particle_params()
    U, Fx, Fy = gorkov_grid_2d(
        p_grid,
        dx,
        dy,
        OMEGA,
        RHO0,
        C_WATER,
        ppar["a"],
        ppar["f1"],
        ppar["f2"],
    )
    return U, Fx, Fy


def _centered_radius_theta(XX: np.ndarray, YY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cx = 0.5 * (float(XX.min()) + float(XX.max()))
    cy = 0.5 * (float(YY.min()) + float(YY.max()))
    dx = XX - cx
    dy = YY - cy
    return np.sqrt(dx * dx + dy * dy), np.arctan2(dy, dx)


def build_symmetric_vortex_source(
    XX: np.ndarray,
    YY: np.ndarray,
    family: str,
    aperture_radius: float,
    waist: Optional[float],
    cone_angle_deg: Optional[float],
    focal_length: Optional[float],
    charge: int = 1,
) -> np.ndarray:
    """Build a source-plane symmetric vortex field."""
    family = family.lower()
    r, theta = _centered_radius_theta(XX, YY)
    aperture = (r <= aperture_radius).astype(float)

    if family == "lg":
        if waist is None:
            raise ValueError("waist is required for LG family")
        base = make_vortex_field(
            XX,
            YY,
            charge=charge,
            waist=waist,
            aperture_radius=aperture_radius,
        )
        if focal_length is None:
            return base
        phi = make_lens_phase(
            XX,
            YY,
            family="ideal",
            focal_length=focal_length,
            aperture_radius=aperture_radius,
            charge=0,
        )
        return base * np.exp(-1j * phi)

    if family == "bessel_gauss":
        if waist is None:
            raise ValueError("waist is required for bessel_gauss family")
        if cone_angle_deg is None:
            raise ValueError("cone_angle_deg is required for bessel_gauss family")
        amp = np.exp(-(r * r) / (waist * waist)) * aperture
        phi = make_lens_phase(
            XX,
            YY,
            family="axicon",
            aperture_radius=aperture_radius,
            axicon_angle_deg=cone_angle_deg,
            charge=charge,
        )
        return amp * np.exp(-1j * phi)

    if family == "axicon":
        if cone_angle_deg is None:
            raise ValueError("cone_angle_deg is required for axicon family")
        phi = make_lens_phase(
            XX,
            YY,
            family="axicon",
            aperture_radius=aperture_radius,
            axicon_angle_deg=cone_angle_deg,
            charge=charge,
        )
        return aperture * np.exp(-1j * phi)

    raise ValueError(f"Unknown family: {family}")


def evaluate_vortex_metrics(p_field: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> Dict[str, float]:
    mag = np.abs(p_field)
    peak = float(np.max(mag))

    ring = estimate_ring_radius(
        mag,
        xg,
        yg,
        r_min=0.08e-3,
        r_max=1.50e-3,
        n_bins=220,
    )
    ring_r = float(ring["ring_radius_m"])

    XX, YY = np.meshgrid(xg, yg)
    cx = 0.5 * (xg[0] + xg[-1])
    cy = 0.5 * (yg[0] + yg[-1])
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)

    if peak > 0.0:
        central = float(np.mean(mag[rr <= 0.12e-3]) / peak)
    else:
        central = 1.0

    if np.isfinite(ring_r):
        band = np.abs(rr - ring_r) <= 0.12e-3
        core = rr <= 2.00e-3
        e_total = float(np.sum(mag[core] ** 2)) + 1e-30
        e_band = float(np.sum(mag[band] ** 2))
        concentration = e_band / e_total
    else:
        concentration = 0.0

    localization = concentration * max(0.0, 1.0 - central)
    return {
        "peak_abs_p": peak,
        "ring_radius_m": ring_r,
        "central_ratio": central,
        "ring_concentration": concentration,
        "localization": localization,
    }


def select_best_candidate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("No candidate rows provided")

    peak_force_max = max(float(r["peak_force"]) for r in rows)
    force_floor = 0.60 * peak_force_max

    feasible = [
        r
        for r in rows
        if np.isfinite(float(r["ring_radius_m"])) and float(r["peak_force"]) >= force_floor
    ]
    if not feasible:
        feasible = [r for r in rows if np.isfinite(float(r["ring_radius_m"]))]
    if not feasible:
        feasible = list(rows)

    feasible = sorted(
        feasible,
        key=lambda r: (
            float(r["ring_radius_m"]),
            -float(r["peak_force"]),
            -float(r["localization"]),
        ),
    )
    chosen = dict(feasible[0])
    chosen["selection_peak_force_floor"] = force_floor
    return chosen


def bilinear_sample_scalar(field: np.ndarray, xg: np.ndarray, yg: np.ndarray, xy: np.ndarray) -> float:
    ix = int(np.clip(np.searchsorted(xg, xy[0]) - 1, 0, len(xg) - 2))
    iy = int(np.clip(np.searchsorted(yg, xy[1]) - 1, 0, len(yg) - 2))

    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    fx = float(np.clip((xy[0] - xg[ix]) / dx, 0.0, 1.0))
    fy = float(np.clip((xy[1] - yg[iy]) / dy, 0.0, 1.0))

    v = (
        (1.0 - fx) * (1.0 - fy) * field[iy, ix]
        + fx * (1.0 - fy) * field[iy, ix + 1]
        + (1.0 - fx) * fy * field[iy + 1, ix]
        + fx * fy * field[iy + 1, ix + 1]
    )
    return float(np.real(v))


def bilinear_sample_vector(
    Fx: np.ndarray,
    Fy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            bilinear_sample_scalar(Fx, xg, yg, xy),
            bilinear_sample_scalar(Fy, xg, yg, xy),
        ],
        dtype=float,
    )


def xy_to_ij(xg: np.ndarray, yg: np.ndarray, xy: np.ndarray) -> Tuple[int, int]:
    ix = int(np.argmin(np.abs(xg - xy[0])))
    iy = int(np.argmin(np.abs(yg - xy[1])))
    return iy, ix


def ij_to_xy(xg: np.ndarray, yg: np.ndarray, ij: Tuple[int, int]) -> np.ndarray:
    iy, ix = ij
    return np.array([float(xg[ix]), float(yg[iy])], dtype=float)


def descend_potential(
    U: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    start_xy: np.ndarray,
    max_iter: int = 300,
) -> Dict[str, Any]:
    """Discrete steepest descent to a local minimum using 8-neighborhood."""
    ny, nx = U.shape
    iy, ix = xy_to_ij(xg, yg, start_xy)
    path: List[Tuple[int, int]] = [(iy, ix)]

    for _ in range(max_iter):
        y0 = max(0, iy - 1)
        y1 = min(ny - 1, iy + 1)
        x0 = max(0, ix - 1)
        x1 = min(nx - 1, ix + 1)

        neighborhood = U[y0 : y1 + 1, x0 : x1 + 1]
        local_idx = np.unravel_index(int(np.argmin(neighborhood)), neighborhood.shape)
        iy_new = y0 + local_idx[0]
        ix_new = x0 + local_idx[1]

        if iy_new == iy and ix_new == ix:
            break

        iy, ix = iy_new, ix_new
        path.append((iy, ix))

    xy = ij_to_xy(xg, yg, (iy, ix))
    border = min(iy, ix, ny - 1 - iy, nx - 1 - ix)

    return {
        "xy": xy,
        "ij": np.array([iy, ix], dtype=int),
        "path_ij": np.array(path, dtype=int),
        "is_border_minimum": bool(border <= 1),
    }


def classify_release(
    final_xy: np.ndarray,
    traps_m: np.ndarray,
    idx_original: int,
    capture_radius: float,
) -> Dict[str, Any]:
    d = np.linalg.norm(traps_m - final_xy[None, :], axis=1)
    idx = int(np.argmin(d))
    dist = float(d[idx])

    if dist <= capture_radius and idx == idx_original:
        status = "return_original"
    elif dist <= capture_radius:
        status = "jump_other"
    else:
        status = "untrapped"

    return {
        "status": status,
        "nearest_idx": idx,
        "nearest_dist": dist,
    }


def estimate_restoring_scale(
    Fx: np.ndarray,
    Fy: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    trap_xy: np.ndarray,
    probe_radius: float = 0.12e-3,
    n_samples: int = 24,
) -> float:
    vals: List[float] = []
    for k in range(n_samples):
        ang = 2.0 * np.pi * float(k) / float(n_samples)
        pt = trap_xy + probe_radius * np.array([np.cos(ang), np.sin(ang)], dtype=float)
        F = bilinear_sample_vector(Fx, Fy, xg, yg, pt)
        vals.append(float(np.linalg.norm(F)))
    vals = sorted(vals)
    return float(vals[len(vals) // 2]) if vals else 0.0


def project_toward(vec: np.ndarray, direction: np.ndarray) -> float:
    return float(np.dot(vec, direction))


def propagate_candidate(
    source: np.ndarray,
    dx: float,
    dy: float,
    source_distance: float,
) -> np.ndarray:
    return propagate_asm(source, dx, dy, wavelength=LAM, z=source_distance)
