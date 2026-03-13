"""
Utilities for hybrid FEM-ASM field overlays.

This module provides reusable helpers for:
  - safe complex-field combination with alpha/psi controls
  - grid consistency checks between FEM and ASM fields
  - bilinear sampling at trap coordinates
  - radial ring diagnostics for vortex calibration
  - local A/B trap-pair selection and neighbour bookkeeping
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree


def validate_grid_consistency(
    p_sw: np.ndarray,
    p_asm: np.ndarray,
    x_sw: np.ndarray,
    y_sw: np.ndarray,
    x_asm: np.ndarray,
    y_asm: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Dict[str, float]:
    """Validate that FEM and ASM fields live on the same XY grid."""
    if p_sw.shape != p_asm.shape:
        raise ValueError(
            f"Shape mismatch: p_sw={p_sw.shape}, p_asm={p_asm.shape}"
        )
    if x_sw.shape != x_asm.shape or y_sw.shape != y_asm.shape:
        raise ValueError("Coordinate vector shape mismatch between FEM and ASM grids")

    if not np.allclose(x_sw, x_asm, rtol=rtol, atol=atol):
        raise ValueError("x-grid mismatch between FEM and ASM")
    if not np.allclose(y_sw, y_asm, rtol=rtol, atol=atol):
        raise ValueError("y-grid mismatch between FEM and ASM")

    dx = float(x_sw[1] - x_sw[0])
    dy = float(y_sw[1] - y_sw[0])
    return {
        "nx": float(len(x_sw)),
        "ny": float(len(y_sw)),
        "dx": dx,
        "dy": dy,
        "x_centre": float(0.5 * (x_sw[0] + x_sw[-1])),
        "y_centre": float(0.5 * (y_sw[0] + y_sw[-1])),
    }


def combine_fields(
    p_sw: np.ndarray,
    p_asm: np.ndarray,
    alpha: float,
    psi: float,
) -> np.ndarray:
    """Combine standing-wave and perturbation fields.

    p_comb = p_sw + alpha * exp(i*psi) * p_asm
    """
    if p_sw.shape != p_asm.shape:
        raise ValueError("combine_fields requires equal field shapes")
    return p_sw + alpha * np.exp(1j * psi) * p_asm


def scale_field_to_peak(p_field: np.ndarray, target_peak: float) -> np.ndarray:
    """Rescale a field so max(|p|) equals target_peak."""
    p_peak = float(np.max(np.abs(p_field)))
    if p_peak <= 0.0:
        return np.zeros_like(p_field)
    return p_field * (target_peak / p_peak)


def _cell_indices(
    xg: np.ndarray,
    yg: np.ndarray,
    xy: np.ndarray,
) -> Tuple[int, int, float, float]:
    """Return bilinear cell indices and fractional offsets."""
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])

    ix = int(np.clip(np.searchsorted(xg, xy[0]) - 1, 0, len(xg) - 2))
    iy = int(np.clip(np.searchsorted(yg, xy[1]) - 1, 0, len(yg) - 2))

    fx = float((xy[0] - xg[ix]) / dx)
    fy = float((xy[1] - yg[iy]) / dy)
    fx = float(np.clip(fx, 0.0, 1.0))
    fy = float(np.clip(fy, 0.0, 1.0))
    return ix, iy, fx, fy


def bilinear_sample_complex(
    p_field: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    xy: np.ndarray,
) -> complex:
    """Bilinear sample of a complex scalar field at xy."""
    ix, iy, fx, fy = _cell_indices(xg, yg, xy)
    return (
        (1.0 - fx) * (1.0 - fy) * p_field[iy, ix]
        + fx * (1.0 - fy) * p_field[iy, ix + 1]
        + (1.0 - fx) * fy * p_field[iy + 1, ix]
        + fx * fy * p_field[iy + 1, ix + 1]
    )


def bilinear_sample_vector(
    fx_field: np.ndarray,
    fy_field: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    xy: np.ndarray,
) -> np.ndarray:
    """Bilinear sample of a 2-D vector field [Fx, Fy] at xy."""
    fx_val = bilinear_sample_complex(fx_field, xg, yg, xy)
    fy_val = bilinear_sample_complex(fy_field, xg, yg, xy)
    return np.array([float(np.real(fx_val)), float(np.real(fy_val))], dtype=float)


def radial_profile_mean(
    mag: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    center: np.ndarray | None = None,
    n_bins: int = 240,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return azimuthal mean magnitude profile vs radius."""
    if center is None:
        center = np.array([0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])])

    XX, YY = np.meshgrid(xg, yg)
    r = np.sqrt((XX - center[0]) ** 2 + (YY - center[1]) ** 2)

    r_flat = r.ravel()
    m_flat = np.asarray(mag, dtype=float).ravel()

    r_max = float(r_flat.max())
    edges = np.linspace(0.0, r_max, n_bins + 1)
    ids = np.digitize(r_flat, edges) - 1

    good = (ids >= 0) & (ids < n_bins)
    counts = np.bincount(ids[good], minlength=n_bins).astype(float)
    sums = np.bincount(ids[good], weights=m_flat[good], minlength=n_bins)

    profile = np.zeros(n_bins, dtype=float)
    nz = counts > 0
    profile[nz] = sums[nz] / counts[nz]

    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, profile


def estimate_ring_radius(
    mag: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    center: np.ndarray | None = None,
    r_min: float = 0.10e-3,
    r_max: float | None = None,
    n_bins: int = 240,
) -> Dict[str, np.ndarray | float]:
    """Estimate ring radius from azimuthally averaged profile peak."""
    r, prof = radial_profile_mean(mag, xg, yg, center=center, n_bins=n_bins)

    mask = r >= r_min
    if r_max is not None:
        mask &= (r <= r_max)

    if not np.any(mask):
        return {
            "ring_radius_m": float("nan"),
            "ring_peak": float("nan"),
            "r_m": r,
            "profile": prof,
        }

    r_sel = r[mask]
    p_sel = prof[mask]
    i_peak = int(np.argmax(p_sel))
    return {
        "ring_radius_m": float(r_sel[i_peak]),
        "ring_peak": float(p_sel[i_peak]),
        "r_m": r,
        "profile": prof,
    }


def choose_adjacent_trap_pair(
    traps_m: np.ndarray,
    target_spacing: float,
    domain_center: np.ndarray,
) -> Dict[str, np.ndarray | float | int]:
    """Pick a nearest-neighbour A/B trap pair near domain centre."""
    if len(traps_m) < 2:
        raise ValueError("Need at least two traps to define an A/B pair")

    tree = cKDTree(traps_m)
    dists, idxs = tree.query(traps_m, k=2)
    nn_d = dists[:, 1]
    nn_i = idxs[:, 1]

    mids = 0.5 * (traps_m + traps_m[nn_i])
    d_mid = np.linalg.norm(mids - domain_center[None, :], axis=1)

    rel_spacing_err = np.abs(nn_d - target_spacing) / max(target_spacing, 1e-30)
    centre_term = d_mid / max(float(np.max(d_mid)), 1e-30)
    score = rel_spacing_err + 0.15 * centre_term

    idx_a = int(np.argmin(score))
    idx_b = int(nn_i[idx_a])

    ab_vec = traps_m[idx_b] - traps_m[idx_a]
    d_ab = float(np.linalg.norm(ab_vec))
    e_ab = ab_vec / max(d_ab, 1e-30)
    midpoint = 0.5 * (traps_m[idx_a] + traps_m[idx_b])

    return {
        "idx_A": idx_a,
        "idx_B": idx_b,
        "d_AB": d_ab,
        "e_AB": e_ab,
        "midpoint": midpoint,
        "score": float(score[idx_a]),
        "spacing_rel_err": float(rel_spacing_err[idx_a]),
    }


def select_neighbour_traps(
    traps_m: np.ndarray,
    idx_a: int,
    idx_b: int,
    midpoint: np.ndarray,
    radius: float,
    min_count: int = 4,
) -> np.ndarray:
    """Return neighbour trap indices around the A/B midpoint."""
    n = len(traps_m)
    all_idx = np.arange(n, dtype=int)
    mask_ab = (all_idx != idx_a) & (all_idx != idx_b)

    d_mid = np.linalg.norm(traps_m - midpoint[None, :], axis=1)
    near = all_idx[mask_ab & (d_mid <= radius)]

    if len(near) >= min_count:
        return near

    # Fallback: nearest traps to midpoint excluding A/B
    cand = all_idx[mask_ab]
    order = np.argsort(d_mid[cand])
    k = min(len(cand), max(min_count, 1))
    return cand[order[:k]]
