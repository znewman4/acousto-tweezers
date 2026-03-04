#!/usr/bin/env python3
"""
FEM Standing Grid (3×3) + ASM Vortex Overlay Study
====================================================

Overlays a validated ASM vortex hourglass (plastic lens, ℓ=2, R=5 mm,
f=4 mm, cosine taper) onto the FEM standing-wave field within a local
3×3 trap grid region (~2λ wide).

**Pipeline:**

1. Solve the standing-wave-only FEM problem (disk velocity = 0).
2. Build ASM plastic-lens vortex drive on a high-resolution FFT grid.
3. Propagate via ``propagate_pressure_asm`` to every z-plane of interest.
4. Interpolate both fields onto a common local grid centered on the
   central trap.
5. Form  p_total = p_stand + α p_vortex  for  α ∈ {0.05, 0.1, 0.2}.
6. Generate diagnostic figures and REPORT.md.

**Requires:**  micromamba run -n acousto-complex  (complex PETSc scalars)

Usage
-----
    micromamba run -n acousto-complex python scripts/dev/fem_standing_plus_asm_vortex_local_3x3.py

Author: Acousto-Tweezers Project
Date:   March 2026
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from acoustweezers.experiments.farfield_petri_cuboid.config import FarFieldConfig
from acoustweezers.experiments.farfield_petri_cuboid.presets import (
    CORRECTED_PRESET, PETSC_MUMPS,
)
from acoustweezers.experiments.farfield_petri_cuboid.solve_pressure import (
    solve_helmholtz,
)
from acoustweezers.physics.acoustics.angular_spectrum import propagate_pressure_asm
from acoustweezers.physics.acoustics.vortex_lens import (
    PlasticLensConfig,
    create_plastic_lens_drive,
)


# ═══════════════════════════════════════════════════════════════════
# Physical constants from CORRECTED_PRESET
# ═══════════════════════════════════════════════════════════════════
C_WATER   = 1484.0                     # m/s
RHO_WATER = 997.0                      # kg/m³
F_HZ      = 2.0e6                      # Hz
LAM       = C_WATER / F_HZ             # 0.000742 m
K_WATER   = 2.0 * np.pi * F_HZ / C_WATER
OMEGA     = 2.0 * np.pi * F_HZ

# Domain from preset
LX      = float(CORRECTED_PRESET["Lx"])        # 6 mm
LY      = float(CORRECTED_PRESET["Ly"])        # 6 mm
H_UNDER = float(CORRECTED_PRESET["H_under"])   # 3 mm
H_TOP   = float(CORRECTED_PRESET["H_top"])     # 2.0085 mm

# Trap plane = mid-petri + λ/4 (standing-wave antinode)
Z_STAR = H_UNDER + H_TOP / 2.0 + 0.25 * LAM

# Trap spacing for antiphase standing wave on both axes = λ/2
TRAP_SPACING = LAM / 2.0

# Local 3×3 region: 3 peaks span 3×(λ/2), add 0.25λ buffer each side
REGION_HALF = 1.5 * TRAP_SPACING + 0.25 * LAM  # ≈ 1.0λ
REGION_SIZE = 2.0 * REGION_HALF                  # ≈ 2.0λ

# ── Vortex lens parameters (from prompt) ──────────────────────────
LENS_ELL   = 2           # topological charge
LENS_R     = 5.0e-3      # aperture radius 5 mm (prompt spec)
LENS_F     = 4.0e-3      # focal length 4 mm
LENS_C     = 2700.0      # plastic lens speed
LENS_APOD  = "cosine_taper"

# ── Overlay mixing ratios ─────────────────────────────────────────
ALPHAS = [0.05, 0.1, 0.2]

# ── Grid parameters ───────────────────────────────────────────────
NGRID_LOCAL = 600         # local 3×3 region resolution per axis
NGRID_XZ    = 400         # XZ meridional slice resolution per axis
ASM_N       = 512         # ASM source-plane FFT grid
ASM_PAD     = 2           # zero-padding factor

# ── z-planes for XZ meridional ────────────────────────────────────
NZ_MERID = 200            # z-points for XZ slice


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
# Canonical location for reusable standing-wave data
STANDING_DATA_DIR = PROJECT_ROOT / "results" / "fem_standing_wave_cache"


def parse_args():
    p = argparse.ArgumentParser(
        description="FEM standing + ASM vortex 3×3 overlay study")
    p.add_argument("--timestamp",
                   default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--elem-per-lam", type=int, default=6,
                   help="FEM mesh elements per wavelength (default: 6; "
                        "with P2 elements this gives 12 pts/λ)")
    p.add_argument("--save-standing-only", action="store_true",
                   help="Solve FEM standing wave, save it, and exit "
                        "(no vortex overlay). Use this to pre-compute "
                        "reusable standing-wave data.")
    p.add_argument("--load-standing", type=str, default=None,
                   metavar="PATH",
                   help="Load previously saved FEM standing-wave .npz "
                        "instead of re-solving. Accepts a path to the "
                        ".npz file, or 'latest' to auto-detect.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _standing_wave_analytical_2d(xg, yg, k, Lx, Ly):
    """
    Analytical 2-D antiphase standing wave pattern for validation.

    For antiphase BCs on both x- and y-walls the fundamental standing
    wave (at resonance) is p ∝ sin(k x) sin(k y).  Here we use the
    actual wavenumber to check peak spacing.

    NOTE: This is for *validation only*. The FEM solution is the
    authoritative standing-wave field.
    """
    XX, YY = np.meshgrid(xg, yg)
    # Antiphase on both axes → sin pattern
    # The wavelength in the petri is λ, trap spacing = λ/2
    p = np.sin(K_WATER * XX) * np.sin(K_WATER * YY)
    return p


def _measure_trap_spacing_from_field(xg, yg, field_mag):
    """
    Numerically measure trap spacing from |p| by finding peaks in
    the 1-D profile along x through the y-center.

    Returns spacing in metres, or NaN if fewer than 2 peaks found.
    """
    from scipy.signal import find_peaks
    ny2 = field_mag.shape[0] // 2
    profile_x = field_mag[ny2, :]
    # Normalize
    profile_x = profile_x / (profile_x.max() + 1e-30)
    peaks, _ = find_peaks(profile_x, height=0.3, distance=5)
    if len(peaks) < 2:
        return np.nan
    spacings = np.diff(xg[peaks])
    return float(np.mean(spacings))


def _measure_vortex_waist(xg, yg, vortex_mag):
    """
    Measure vortex ring radius at the waist (peak of azimuthal-average
    radial profile) on a single XY slice.

    Returns ring radius [m].
    """
    cx = (xg[0] + xg[-1]) / 2.0
    cy = (yg[0] + yg[-1]) / 2.0
    XX, YY = np.meshgrid(xg, yg)
    RR = np.sqrt((XX - cx)**2 + (YY - cy)**2)

    nbins = 100
    r_max = min(xg[-1] - cx, yg[-1] - cy)
    r_bins = np.linspace(0, r_max, nbins + 1)
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    rad_mean = np.zeros(nbins)

    for ib in range(nbins):
        sel = (RR >= r_bins[ib]) & (RR < r_bins[ib + 1])
        if sel.any():
            rad_mean[ib] = np.mean(vortex_mag[sel])

    if rad_mean.max() > 0:
        peak_idx = np.argmax(rad_mean)
        return float(r_mid[peak_idx])
    return np.nan


def _count_perturbed_traps(xg, yg, p_stand_mag, p_diff_mag,
                           threshold_frac=0.10):
    """
    Count how many standing-wave trap peaks are significantly perturbed.

    A trap is 'significantly perturbed' if the vortex difference |Δp|
    at the peak location exceeds threshold_frac × max(|p_stand|).

    Uses connected-component labelling to handle NearestNDInterpolator
    plateaus (multiple grid pixels share the same DOF value, creating
    flat regions that all satisfy == local_max).

    Returns (n_perturbed, n_total, peak_coords).
    """
    from scipy.ndimage import maximum_filter, label

    dx = xg[1] - xg[0]
    # Neighbourhood ~ 0.8 × trap spacing so each peak is isolated
    nbhood = max(5, int(0.8 * TRAP_SPACING / dx))
    local_max = maximum_filter(p_stand_mag, size=nbhood)
    peaks_mask = ((p_stand_mag == local_max)
                  & (p_stand_mag > 0.3 * p_stand_mag.max()))

    # Label connected peak regions and find their centroids
    labeled, n_total = label(peaks_mask)
    threshold = threshold_frac * p_stand_mag.max()

    n_pert = 0
    peak_coords = []
    for i in range(1, n_total + 1):
        ys, xs = np.where(labeled == i)
        cy_idx = int(np.mean(ys))
        cx_idx = int(np.mean(xs))
        peak_coords.append((float(xg[cx_idx]), float(yg[cy_idx])))
        if p_diff_mag[cy_idx, cx_idx] > threshold:
            n_pert += 1

    return n_pert, n_total, peak_coords


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def _save_standing_wave(coords_fem, pv_fem, cfg, elem_per_lam, solve_time, dofs):
    """
    Save FEM standing-wave solution to a canonical cache directory.

    The file is named with the elements-per-wavelength so different
    resolutions can coexist.  The saved .npz contains everything
    needed to reconstruct the complex pressure field on the FEM DOF
    coordinates — no FEniCSx required to *load* it.

    Returns the path to the saved file.
    """
    STANDING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fname = f"standing_wave_epl{elem_per_lam}.npz"
    path = STANDING_DATA_DIR / fname

    np.savez_compressed(
        path,
        # DOF coordinates (N×3)
        coords=coords_fem,
        # Complex pressure at each DOF
        p_real=np.real(pv_fem),
        p_imag=np.imag(pv_fem),
        # Physics / config (scalar metadata stored as 0-d arrays)
        frequency_hz=np.float64(cfg.frequency_hz),
        c_water=np.float64(cfg.c),
        rho_water=np.float64(cfg.rho),
        Lx=np.float64(cfg.Lx),
        Ly=np.float64(cfg.Ly),
        H_under=np.float64(cfg.H_under),
        H_top=np.float64(cfg.H_top),
        H_total=np.float64(cfg.H_total),
        wavelength=np.float64(cfg.wavelength),
        elements_per_wavelength=np.int64(elem_per_lam),
        mesh_nx=np.int64(cfg.mesh_nx),
        mesh_ny=np.int64(cfg.mesh_ny),
        mesh_nz=np.int64(cfg.mesh_nz),
        solve_time_s=np.float64(solve_time),
        dofs=np.int64(dofs),
        standing_velocity_amplitude=np.float64(
            cfg.standing_velocity_amplitude),
        standing_phase_pattern=np.array(cfg.standing_phase_pattern),
        standing_axis=np.array(cfg.standing_axis),
    )

    # Also write a human-readable sidecar
    meta_path = STANDING_DATA_DIR / f"standing_wave_epl{elem_per_lam}_INFO.txt"
    meta_path.write_text(
        f"FEM Standing-Wave Cache\n"
        f"=======================\n"
        f"Created       : {datetime.now().isoformat()}\n"
        f"elem/λ        : {elem_per_lam}\n"
        f"Mesh          : {cfg.mesh_nx}×{cfg.mesh_ny}×{cfg.mesh_nz}\n"
        f"DOFs          : {dofs}\n"
        f"Solve time    : {solve_time:.1f}s\n"
        f"max|p|        : {np.abs(pv_fem).max():.4f} Pa\n"
        f"Frequency     : {cfg.frequency_hz/1e6:.1f} MHz\n"
        f"λ             : {cfg.wavelength*1e3:.4f} mm\n"
        f"Domain        : {cfg.Lx*1e3:.1f}×{cfg.Ly*1e3:.1f}×{cfg.H_total*1e3:.2f} mm\n"
        f"Standing V    : {cfg.standing_velocity_amplitude*1e6:.1f} µm/s\n"
        f"Pattern       : {cfg.standing_phase_pattern}, axis={cfg.standing_axis}\n"
        f"\n"
        f"To load in Python:\n"
        f"  d = np.load('{fname}')\n"
        f"  coords = d['coords']          # (N,3) float64\n"
        f"  p = d['p_real'] + 1j*d['p_imag']  # complex128\n"
    )
    return path


def _load_standing_wave(path_or_tag):
    """
    Load a previously saved FEM standing-wave .npz.

    Parameters
    ----------
    path_or_tag : str
        Full path to the .npz, or 'latest' to auto-detect the
        highest-resolution file in the cache directory.

    Returns
    -------
    coords : ndarray (N,3)
    p_values : ndarray (N,) complex128
    metadata : dict
    """
    if path_or_tag == "latest":
        candidates = sorted(STANDING_DATA_DIR.glob("standing_wave_epl*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No standing-wave cache found in {STANDING_DATA_DIR}")
        path = candidates[-1]  # highest epl number (alphabetical sort)
    else:
        path = Path(path_or_tag)
        if not path.exists():
            raise FileNotFoundError(f"Standing-wave file not found: {path}")

    d = np.load(path, allow_pickle=True)
    coords = d["coords"]
    p_values = d["p_real"] + 1j * d["p_imag"]
    metadata = {
        k: d[k].item() if d[k].ndim == 0 else str(d[k])
        for k in d.files if k not in ("coords", "p_real", "p_imag")
    }
    return coords, p_values, metadata, path


def main():
    args = parse_args()
    TS = args.timestamp
    ELEM = args.elem_per_lam

    BASE = PROJECT_ROOT / "results" / f"fem_standing_plus_asm_vortex_local_3x3_{TS}"
    FIG_DIR = BASE / "figures"
    DATA_DIR = BASE / "data"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log("=" * 72)
    log("FEM STANDING GRID (3×3) + ASM VORTEX OVERLAY STUDY")
    log("=" * 72)
    log(f"Timestamp        : {TS}")
    log(f"λ                : {LAM*1e3:.4f} mm")
    log(f"k                : {K_WATER:.1f} rad/m")
    log(f"Trap spacing     : {TRAP_SPACING*1e3:.4f} mm  (λ/2)")
    log(f"3×3 region size  : {REGION_SIZE*1e3:.3f} mm  (≈{REGION_SIZE/LAM:.2f}λ)")
    log(f"z* (trap plane)  : {Z_STAR*1e3:.3f} mm")
    log(f"Vortex lens      : ℓ={LENS_ELL}, R={LENS_R*1e3:.1f} mm, "
        f"f={LENS_F*1e3:.1f} mm")
    NF = LENS_R**2 / (LAM * LENS_F)
    log(f"Fresnel N_F      : {NF:.2f}  (R²/λf)")
    log(f"α sweep          : {ALPHAS}")
    log(f"FEM elem/λ       : {ELEM}")
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: FEM standing-wave-only (solve or load from cache)
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 1: FEM standing-wave-only")
    log("─" * 72)

    if args.load_standing:
        # ── Load from cache ───────────────────────────────────────
        log(f"  Loading standing wave from: {args.load_standing}")
        coords_fem, pv_fem, meta_cache, cache_path = _load_standing_wave(
            args.load_standing)
        ELEM_LOADED = int(meta_cache.get("elements_per_wavelength", ELEM))
        log(f"  Loaded {cache_path.name}")
        log(f"  elem/λ (cached)   : {ELEM_LOADED}")
        log(f"  DOFs (cached)     : {meta_cache.get('dofs', '?')}")
        log(f"  Solve time (orig) : {meta_cache.get('solve_time_s', '?')}s")
        log(f"  max|p_stand|      : {np.abs(pv_fem).max():.4f} Pa")
        log(f"  coords shape      : {coords_fem.shape}")
        # Build a FarFieldConfig for use later (e.g. H_total)
        fem_overrides = {
            **CORRECTED_PRESET,
            "disk_velocity_amplitude": 0.0,
            "elements_per_wavelength": ELEM_LOADED,
        }
        cfg_s = FarFieldConfig(**fem_overrides)
        t_fem = 0.0
    else:
        # ── Fresh FEM solve ───────────────────────────────────────
        fem_overrides = {
            **CORRECTED_PRESET,
            "disk_velocity_amplitude": 0.0,   # vortex OFF
            "elements_per_wavelength": ELEM,
        }
        cfg_standing = FarFieldConfig(**fem_overrides)

        log(f"  Domain: {cfg_standing.Lx*1e3:.1f} × {cfg_standing.Ly*1e3:.1f} × "
            f"{cfg_standing.H_total*1e3:.2f} mm")
        log(f"  Mesh  : {cfg_standing.mesh_nx}×{cfg_standing.mesh_ny}×{cfg_standing.mesh_nz}")
        log(f"  Standing V: {cfg_standing.standing_velocity_amplitude*1e6:.1f} µm/s")
        log(f"  Pattern: {cfg_standing.standing_phase_pattern}, "
            f"axis: {cfg_standing.standing_axis}")

        t0 = time.time()
        # Always set MUMPS icntl_14 (memory relaxation) to 500%
        # because DOLFINx's LinearProblem prefix mechanism doesn't
        # propagate mat_mumps_* to the MUMPS library by default.
        # The solve_pressure module now handles icntl injection directly.
        petsc_opts = dict(PETSC_MUMPS)
        petsc_opts["mat_mumps_icntl_14"] = "500"   # 500% memory relaxation
        sol_stand = solve_helmholtz(cfg_standing, verbose=True,
                                    petsc_options=petsc_opts)
        t_fem = time.time() - t0
        log(f"  FEM solve time: {t_fem:.1f}s")
        log(f"  max|p_stand| = {sol_stand.max_pressure:.2f} Pa")
        log(f"  DOFs = {sol_stand.dofs}")

        # Extract arrays and free FEniCSx memory
        coords_fem = sol_stand.coords.copy()
        pv_fem = sol_stand.p_values.copy()
        cfg_s = sol_stand.cfg

        # ── Save standing wave to cache ───────────────────────────
        saved_path = _save_standing_wave(
            coords_fem, pv_fem, cfg_s, ELEM, t_fem, sol_stand.dofs)
        log(f"  ✓ Standing wave saved to: {saved_path}")
        log(f"    (reuse with: --load-standing {saved_path})")
        log(f"    (or:         --load-standing latest)")

        del sol_stand
        gc.collect()

    log("")

    # ── Early exit for --save-standing-only mode ──────────────────
    if args.save_standing_only:
        t_total = time.time() - t_start
        log("=" * 72)
        log("SAVE-STANDING-ONLY MODE — done.")
        log("=" * 72)
        log(f"  Total time: {t_total:.1f}s")
        log(f"  To overlay vortices later, run:")
        log(f"    python {Path(__file__).name} --load-standing latest")
        (BASE / "console_log.txt").write_text("\n".join(report_lines))
        return

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Build ASM vortex volume
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 2: ASM vortex drive + propagation")
    log("─" * 72)

    # ASM source plane covers a region centered on the FEM domain center
    # (so the vortex focus coincides with the 3×3 trap grid center).
    CX, CY = LX / 2, LY / 2

    # Large source grid for the large aperture (R=5mm, domain=6mm)
    # Grid must accommodate the full aperture with margin for FFT wrap
    ASM_DOMAIN = max(LX, 2 * LENS_R + 4 * LAM)  # at least 2R + 4λ margin
    asm_n = ASM_N

    # Center the ASM grid on the FEM domain center (CX, CY)
    x0_asm = CX - ASM_DOMAIN / 2.0
    y0_asm = CY - ASM_DOMAIN / 2.0
    xg_asm = np.linspace(x0_asm, x0_asm + ASM_DOMAIN, asm_n, endpoint=False)
    yg_asm = np.linspace(y0_asm, y0_asm + ASM_DOMAIN, asm_n, endpoint=False)
    dx_asm = float(xg_asm[1] - xg_asm[0])
    dy_asm = float(yg_asm[1] - yg_asm[0])
    XX_asm, YY_asm = np.meshgrid(xg_asm, yg_asm)

    # Lens is centered at the FEM domain center
    cx_asm = CX
    cy_asm = CY

    lens_cfg = PlasticLensConfig(
        topological_charge=LENS_ELL,
        focal_length=LENS_F,
        focus_offset_x=0.0,       # on-axis for symmetric hourglass
        focus_offset_y=0.0,
        c_lens=LENS_C,
        c_water=C_WATER,
        frequency_hz=F_HZ,
        aperture_radius=LENS_R,
        center=None,
        apodization=LENS_APOD,
        apodization_strength=1.0,
    )

    log(f"  ASM grid: {asm_n}×{asm_n}, domain={ASM_DOMAIN*1e3:.2f} mm, "
        f"dx={dx_asm*1e6:.1f} µm")
    log(f"  Lens: ℓ={LENS_ELL}, R={LENS_R*1e3:.1f} mm, f={LENS_F*1e3:.1f} mm")

    D_asm = create_plastic_lens_drive(
        XX_asm.ravel(), YY_asm.ravel(), lens_cfg,
        center_x=cx_asm, center_y=cy_asm, verbose=True,
    ).reshape(XX_asm.shape)

    log(f"  Drive max|D| = {np.abs(D_asm).max():.6f}")

    # Propagate to the trap plane z=z*
    log(f"  Propagating to z* = {Z_STAR*1e3:.3f} mm ...")
    t0 = time.time()
    p_vortex_asm_full = propagate_pressure_asm(
        D_asm, dx_asm, dy_asm, K_WATER, Z_STAR,
        pad_factor=ASM_PAD, include_evanescent=True)
    t_asm_xy = time.time() - t0
    log(f"  ASM XY propagation: {t_asm_xy:.1f}s")
    log(f"  max|p_vortex(z*)| = {np.abs(p_vortex_asm_full).max():.6f}")
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Interpolate both fields to local 3×3 grid
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 3: Interpolate to local 3×3 region at z*")
    log("─" * 72)

    from scipy.interpolate import (
        LinearNDInterpolator, NearestNDInterpolator,
        RegularGridInterpolator,
    )

    # Local grid centered on domain center (= central trap)
    x_lo = CX - REGION_HALF
    x_hi = CX + REGION_HALF
    y_lo = CY - REGION_HALF
    y_hi = CY + REGION_HALF
    xg_loc = np.linspace(x_lo, x_hi, NGRID_LOCAL)
    yg_loc = np.linspace(y_lo, y_hi, NGRID_LOCAL)
    XX_loc, YY_loc = np.meshgrid(xg_loc, yg_loc)

    log(f"  Local region: [{x_lo*1e3:.3f}, {x_hi*1e3:.3f}] × "
        f"[{y_lo*1e3:.3f}, {y_hi*1e3:.3f}] mm")
    log(f"  Grid: {NGRID_LOCAL}×{NGRID_LOCAL}")

    # ── FEM standing → local grid ─────────────────────────────────
    # OPTIMISED: instead of building a 3-D Delaunay on all 762K DOFs
    # (O(n²) – hours!), extract a thin slab around z=Z_STAR and do a
    # fast 2-D Delaunay in the (x, y) plane.
    h_elem = LAM / max(ELEM if not args.load_standing else ELEM_LOADED, 4)
    z_tol = 3.0 * h_elem          # ~3 element layers → plenty of points
    z_mask = np.abs(coords_fem[:, 2] - Z_STAR) < z_tol
    coords_slab = coords_fem[z_mask, :2]   # (x, y) only
    pv_slab = pv_fem[z_mask]
    log(f"  Interpolating FEM → local grid (2-D LinearNDInterpolator) ...")
    log(f"    z-slab: |z - z*| < {z_tol*1e3:.3f} mm → {coords_slab.shape[0]} / {len(pv_fem)} DOFs")
    lin_re = LinearNDInterpolator(coords_slab, np.real(pv_slab))
    lin_im = LinearNDInterpolator(coords_slab, np.imag(pv_slab))
    nn_re  = NearestNDInterpolator(coords_slab, np.real(pv_slab))
    nn_im  = NearestNDInterpolator(coords_slab, np.imag(pv_slab))
    pts_2d_fem = np.column_stack([XX_loc.ravel(), YY_loc.ravel()])
    re_vals = lin_re(pts_2d_fem)
    im_vals = lin_im(pts_2d_fem)
    nan_mask = np.isnan(re_vals)
    if nan_mask.any():
        re_vals[nan_mask] = nn_re(pts_2d_fem[nan_mask])
        im_vals[nan_mask] = nn_im(pts_2d_fem[nan_mask])
        log(f"    (filled {nan_mask.sum()} boundary NaN with nearest)")
    p_stand_local = (re_vals + 1j * im_vals).reshape(XX_loc.shape)
    del lin_re, lin_im, nn_re, nn_im, coords_slab, pv_slab

    log(f"  |p_stand| on local grid: max={np.abs(p_stand_local).max():.4f} Pa")

    # ── ASM vortex → local grid ───────────────────────────────────
    # RegularGridInterpolator from ASM grid
    ire_v = RegularGridInterpolator(
        (yg_asm, xg_asm), np.real(p_vortex_asm_full),
        method="linear", bounds_error=False, fill_value=0.0)
    iim_v = RegularGridInterpolator(
        (yg_asm, xg_asm), np.imag(p_vortex_asm_full),
        method="linear", bounds_error=False, fill_value=0.0)
    pts_2d = np.column_stack([YY_loc.ravel(), XX_loc.ravel()])
    p_vortex_local = (ire_v(pts_2d) + 1j * iim_v(pts_2d)).reshape(XX_loc.shape)

    log(f"  |p_vortex| on local grid: max={np.abs(p_vortex_local).max():.6f}")
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Measure vortex waist + trap spacing (VALIDATION)
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 4: Validation measurements")
    log("─" * 72)

    # 4a. Trap spacing
    trap_sp_meas = _measure_trap_spacing_from_field(
        xg_loc, yg_loc, np.abs(p_stand_local))
    log(f"  Measured trap spacing  : {trap_sp_meas*1e3:.4f} mm")
    log(f"  Expected (λ/2)        : {TRAP_SPACING*1e3:.4f} mm")
    if not np.isnan(trap_sp_meas):
        sp_err = abs(trap_sp_meas - TRAP_SPACING) / TRAP_SPACING * 100
        log(f"  Spacing error         : {sp_err:.1f}%")
    else:
        log(f"  WARNING: Could not measure trap spacing (too few peaks)")

    # 4b. Vortex waist radius
    # Use the full ASM domain for accurate waist measurement
    xg_full = np.linspace(CX - 3*LAM, CX + 3*LAM, 400)
    yg_full = np.linspace(CY - 3*LAM, CY + 3*LAM, 400)
    XX_f, YY_f = np.meshgrid(xg_full, yg_full)
    pts_f = np.column_stack([YY_f.ravel(), XX_f.ravel()])
    pv_full = (ire_v(pts_f) + 1j * iim_v(pts_f)).reshape(XX_f.shape)
    waist_radius = _measure_vortex_waist(xg_full, yg_full, np.abs(pv_full))
    waist_diameter = 2.0 * waist_radius

    log(f"  Vortex waist radius   : {waist_radius*1e3:.4f} mm")
    log(f"  Vortex waist diameter : {waist_diameter*1e3:.4f} mm")
    log(f"  Waist diameter / λ    : {waist_diameter/LAM:.3f}")

    if waist_diameter > LAM:
        log(f"  ⚠ WARNING: Waist diameter ({waist_diameter/LAM:.2f}λ) > 1.0λ — "
            f"vortex may perturb too many traps!")
    else:
        log(f"  ✓ Waist diameter < 1.0λ — localised perturbation expected")
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Superposition and perturbation analysis
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 5: Superposition  p_total = p_stand + α p_vortex")
    log("─" * 72)

    # Normalize vortex so max|p_vortex| corresponds to unity
    # The mixing ratio α then directly sets the relative amplitude
    stand_peak = np.abs(p_stand_local).max()
    vortex_peak = np.abs(p_vortex_local).max()
    # Scale vortex to unit peak relative to standing
    p_vortex_norm = p_vortex_local / (vortex_peak + 1e-30) * stand_peak

    perturbation_results = {}

    for alpha in ALPHAS:
        p_total = p_stand_local + alpha * p_vortex_norm
        p_diff = p_total - p_stand_local  # = α * p_vortex_norm

        n_pert, n_total, peak_locs = _count_perturbed_traps(
            xg_loc, yg_loc, np.abs(p_stand_local),
            np.abs(p_total) - np.abs(p_stand_local),
            threshold_frac=0.05)

        ratio_at_center = (alpha * np.abs(p_vortex_norm).max()) / stand_peak * 100

        log(f"  α = {alpha}:")
        log(f"    Vortex peak / standing peak = {ratio_at_center:.1f}%")
        log(f"    Traps significantly perturbed: {n_pert} / {n_total}")

        perturbation_results[alpha] = {
            "p_total": p_total,
            "p_diff": np.abs(p_diff),
            "n_perturbed": n_pert,
            "n_total": n_total,
            "peak_locs": peak_locs,
            "ratio_pct": ratio_at_center,
        }
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 6: XZ meridional slice (vortex hourglass + standing envelope)
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 6: XZ meridional slice through vortex center")
    log("─" * 72)

    # FEM standing XZ slice at y = Ly/2
    # OPTIMISED: thin y-slab → 2-D Delaunay in (x, z) instead of 3-D.
    y_tol = 3.0 * h_elem
    y_mask = np.abs(coords_fem[:, 1] - CY) < y_tol
    coords_yslab = coords_fem[y_mask][:, [0, 2]]   # (x, z)
    pv_yslab = pv_fem[y_mask]
    log(f"  Interpolating FEM → XZ slice (2-D LinearNDInterpolator) ...")
    log(f"    y-slab: |y - CY| < {y_tol*1e3:.3f} mm → {coords_yslab.shape[0]} / {len(pv_fem)} DOFs")
    lin_re_xz = LinearNDInterpolator(coords_yslab, np.real(pv_yslab))
    lin_im_xz = LinearNDInterpolator(coords_yslab, np.imag(pv_yslab))
    nn_re_xz  = NearestNDInterpolator(coords_yslab, np.real(pv_yslab))
    nn_im_xz  = NearestNDInterpolator(coords_yslab, np.imag(pv_yslab))

    # z range: from just above bottom to top of domain
    zg_xz = np.linspace(H_UNDER * 0.5, cfg_s.H_total, NZ_MERID)
    xg_xz = np.linspace(x_lo, x_hi, NGRID_XZ)
    XX_xz, ZZ_xz = np.meshgrid(xg_xz, zg_xz)
    pts_xz_2d = np.column_stack([XX_xz.ravel(), ZZ_xz.ravel()])
    re_xz = lin_re_xz(pts_xz_2d)
    im_xz = lin_im_xz(pts_xz_2d)
    nan_xz = np.isnan(re_xz)
    if nan_xz.any():
        re_xz[nan_xz] = nn_re_xz(pts_xz_2d[nan_xz])
        im_xz[nan_xz] = nn_im_xz(pts_xz_2d[nan_xz])
        log(f"    (filled {nan_xz.sum()} boundary NaN with nearest)")
    p_stand_xz = (re_xz + 1j * im_xz).reshape(XX_xz.shape)
    del lin_re_xz, lin_im_xz, nn_re_xz, nn_im_xz, coords_yslab, pv_yslab

    # ASM vortex XZ slice — propagate to each z-plane
    log(f"  Propagating vortex to {NZ_MERID} z-planes for XZ slice...")
    p_vortex_xz = np.zeros((NZ_MERID, NGRID_XZ), dtype=complex)

    # Precompute the 1-D x-interpolation points (at y = domain center on ASM grid)
    t0 = time.time()
    for iz, zz in enumerate(zg_xz):
        p_z = propagate_pressure_asm(D_asm, dx_asm, dy_asm, K_WATER, zz,
                                     pad_factor=ASM_PAD, include_evanescent=True)
        # Extract line at y = cy_asm
        iy_cen = np.argmin(np.abs(yg_asm - cy_asm))
        # Interpolate from ASM x-grid to local x-grid
        from numpy import interp as np_interp
        p_line = p_z[iy_cen, :]
        p_vortex_xz[iz, :] = np.interp(xg_xz, xg_asm, np.real(p_line)) + \
                              1j * np.interp(xg_xz, xg_asm, np.imag(p_line))

        if (iz + 1) % 50 == 0 or iz == 0 or iz == NZ_MERID - 1:
            log(f"    z[{iz:3d}] = {zz*1e3:6.3f} mm  "
                f"max|p_v| = {np.abs(p_vortex_xz[iz]).max():.6f}")
    t_xz = time.time() - t0
    log(f"  XZ vortex propagation: {t_xz:.1f}s")

    # Normalize XZ vortex same way
    p_vortex_xz_norm = p_vortex_xz / (vortex_peak + 1e-30) * stand_peak
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Generate figures
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 7: Generating figures")
    log("─" * 72)

    # Common extent for local plots (in mm)
    ext_xy = [x_lo * 1e3, x_hi * 1e3, y_lo * 1e3, y_hi * 1e3]

    # ── Figure 1: Standing-wave only XY ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(p_stand_local), extent=ext_xy, origin="lower",
                   cmap="inferno")
    ax.set_title(f"|p_stand| at z* = {Z_STAR*1e3:.3f} mm\n"
                 f"3×3 region ({REGION_SIZE*1e3:.2f} mm ≈ {REGION_SIZE/LAM:.1f}λ)",
                 fontsize=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="|p| [Pa]", shrink=0.85)
    # Annotate trap spacing
    ax.annotate(f"λ/2 = {TRAP_SPACING*1e3:.3f} mm",
                xy=(0.02, 0.96), xycoords="axes fraction",
                fontsize=9, color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_standing_only_xy.png", dpi=200)
    plt.close(fig)
    log("  Saved 01_standing_only_xy.png")

    # ── Figure 2: Vortex-only XY at z* ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(p_vortex_local), extent=ext_xy, origin="lower",
                   cmap="magma")
    ax.set_title(f"|p_vortex| at z* (ASM)\n"
                 f"ℓ={LENS_ELL}, R={LENS_R*1e3:.0f} mm, f={LENS_F*1e3:.0f} mm",
                 fontsize=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="|p| [arb]", shrink=0.85)
    # Annotate waist
    circle = plt.Circle((CX * 1e3, CY * 1e3), waist_radius * 1e3,
                         fill=False, color="cyan", linewidth=1.5, linestyle="--")
    ax.add_patch(circle)
    ax.annotate(f"waist r = {waist_radius*1e3:.3f} mm\n"
                f"= {waist_radius/LAM:.2f}λ",
                xy=(0.02, 0.96), xycoords="axes fraction",
                fontsize=9, color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_vortex_only_xy.png", dpi=200)
    plt.close(fig)
    log("  Saved 02_vortex_only_xy.png")

    # ── Figure 3: Combined + difference for each α ────────────────
    for alpha in ALPHAS:
        pr = perturbation_results[alpha]
        p_total = pr["p_total"]

        vmax_shared = max(np.abs(p_stand_local).max(),
                          np.abs(p_total).max())

        fig, axes = plt.subplots(1, 4, figsize=(26, 5.5))
        fig.suptitle(f"α = {alpha}  (vortex peak = {pr['ratio_pct']:.1f}% of "
                     f"standing peak)  —  {pr['n_perturbed']}/{pr['n_total']} "
                     f"traps perturbed", fontsize=13, fontweight="bold")

        # Standing only
        ax = axes[0]
        im = ax.imshow(np.abs(p_stand_local), extent=ext_xy, origin="lower",
                       cmap="inferno", vmin=0, vmax=vmax_shared)
        ax.set_title("|p_stand|")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75)

        # Vortex only (scaled) — own colorscale so it's visible
        vortex_scaled = np.abs(alpha * p_vortex_norm)
        ax = axes[1]
        im = ax.imshow(vortex_scaled, extent=ext_xy,
                       origin="lower", cmap="magma",
                       vmin=0, vmax=vortex_scaled.max())
        ax.set_title(f"|α·p_vortex| (α={alpha})")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75)

        # Combined
        ax = axes[2]
        im = ax.imshow(np.abs(p_total), extent=ext_xy, origin="lower",
                       cmap="inferno", vmin=0, vmax=vmax_shared)
        ax.set_title("|p_total|")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75)

        # Difference |p_total| - |p_stand|
        ax = axes[3]
        diff = np.abs(p_total) - np.abs(p_stand_local)
        vlim = max(abs(diff.min()), abs(diff.max()))
        im = ax.imshow(diff, extent=ext_xy, origin="lower",
                       cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title("|p_total| − |p_stand|")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75, label="ΔPa")

        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fname = f"03_overlay_alpha_{alpha:.2f}.png"
        fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved {fname}")

    # ── Figure 4: XZ meridional slice ─────────────────────────────
    # Use α=0.1 as the representative overlay
    alpha_xz = 0.1
    p_total_xz = p_stand_xz + alpha_xz * p_vortex_xz_norm
    ext_xz = [x_lo * 1e3, x_hi * 1e3, zg_xz[0] * 1e3, zg_xz[-1] * 1e3]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"XZ meridional through vortex center (y = {CY*1e3:.1f} mm, "
                 f"α = {alpha_xz})", fontsize=13, fontweight="bold")

    vmax_xz = max(np.abs(p_stand_xz).max(), np.abs(p_total_xz).max())

    # Standing
    ax = axes[0]
    im = ax.imshow(np.abs(p_stand_xz), extent=ext_xz, origin="lower",
                   aspect="auto", cmap="inferno", vmin=0, vmax=vmax_xz)
    ax.axhline(Z_STAR * 1e3, color="cyan", ls="--", lw=1, label=f"z* = {Z_STAR*1e3:.2f} mm")
    ax.set_title("|p_stand| (FEM)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    ax.legend(fontsize=8, loc="upper left")
    plt.colorbar(im, ax=ax, shrink=0.75)

    # Vortex — own colorscale so the hourglass structure is visible
    vortex_xz_scaled = np.abs(alpha_xz * p_vortex_xz_norm)
    ax = axes[1]
    im = ax.imshow(vortex_xz_scaled, extent=ext_xz,
                   origin="lower", aspect="auto", cmap="magma",
                   vmin=0, vmax=vortex_xz_scaled.max())
    ax.axhline(Z_STAR * 1e3, color="cyan", ls="--", lw=1)
    ax.set_title(f"|α·p_vortex| (ASM, α={alpha_xz})")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    plt.colorbar(im, ax=ax, shrink=0.75)

    # Combined
    ax = axes[2]
    im = ax.imshow(np.abs(p_total_xz), extent=ext_xz, origin="lower",
                   aspect="auto", cmap="inferno", vmin=0, vmax=vmax_xz)
    ax.axhline(Z_STAR * 1e3, color="cyan", ls="--", lw=1)
    # Mark standing-wave nodes (λ/2 apart in x)
    for i_node in range(-4, 5):
        x_node = CX + i_node * TRAP_SPACING
        if x_lo <= x_node <= x_hi:
            ax.axvline(x_node * 1e3, color="white", ls=":", lw=0.5, alpha=0.5)
    ax.set_title("|p_total|")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    plt.colorbar(im, ax=ax, shrink=0.75)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "04_xz_meridional.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("  Saved 04_xz_meridional.png")

    # ── Figure 5: Phase plots (standing and combined) ─────────────
    alpha_ph = 0.1
    p_total_ph = p_stand_local + alpha_ph * p_vortex_norm

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    fig.suptitle(f"Phase at z* — standing vs combined (α = {alpha_ph})",
                 fontsize=13, fontweight="bold")

    for ax, pc, title in zip(axes,
                              [p_stand_local, p_vortex_local, p_total_ph],
                              ["arg(p_stand)", "arg(p_vortex)", "arg(p_total)"]):
        im = ax.imshow(np.angle(pc), extent=ext_xy, origin="lower",
                       cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(title)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75, label="Phase [rad]")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG_DIR / "05_phase_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("  Saved 05_phase_comparison.png")
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 8: Save data
    # ─────────────────────────────────────────────────────────────
    log("─" * 72)
    log("STEP 8: Saving data")
    log("─" * 72)

    np.savez_compressed(DATA_DIR / "local_fields.npz",
                        xg_loc=xg_loc, yg_loc=yg_loc,
                        p_stand_local_re=np.real(p_stand_local),
                        p_stand_local_im=np.imag(p_stand_local),
                        p_vortex_local_re=np.real(p_vortex_local),
                        p_vortex_local_im=np.imag(p_vortex_local))

    np.savez_compressed(DATA_DIR / "xz_fields.npz",
                        xg_xz=xg_xz, zg_xz=zg_xz,
                        p_stand_xz_re=np.real(p_stand_xz),
                        p_stand_xz_im=np.imag(p_stand_xz),
                        p_vortex_xz_re=np.real(p_vortex_xz),
                        p_vortex_xz_im=np.imag(p_vortex_xz))
    log("  Saved local_fields.npz and xz_fields.npz")

    # Metadata
    metadata = {
        "timestamp": TS,
        "physics": {
            "c_water": C_WATER,
            "rho_water": RHO_WATER,
            "frequency_hz": F_HZ,
            "wavelength_m": LAM,
            "k_water": K_WATER,
            "trap_spacing_m": TRAP_SPACING,
            "z_star_m": Z_STAR,
        },
        "vortex_lens": {
            "topological_charge": LENS_ELL,
            "aperture_radius_m": LENS_R,
            "focal_length_m": LENS_F,
            "c_lens": LENS_C,
            "apodization": LENS_APOD,
            "fresnel_number": NF,
        },
        "grid": {
            "local_region_half_m": REGION_HALF,
            "local_region_size_m": REGION_SIZE,
            "ngrid_local": NGRID_LOCAL,
            "asm_n": asm_n,
            "asm_domain_m": ASM_DOMAIN,
            "dx_asm_m": dx_asm,
        },
        "fem": {
            "elements_per_wavelength": ELEM,
            "domain_Lx_m": LX,
            "domain_Ly_m": LY,
            "source": str(args.load_standing) if args.load_standing else "fresh_solve",
            "solve_time_s": t_fem,
        },
        "validation": {
            "measured_trap_spacing_m": float(trap_sp_meas) if not np.isnan(trap_sp_meas) else None,
            "expected_trap_spacing_m": TRAP_SPACING,
            "waist_radius_m": float(waist_radius),
            "waist_diameter_m": float(waist_diameter),
            "waist_diameter_over_lambda": float(waist_diameter / LAM),
        },
        "perturbation": {},
    }
    for alpha in ALPHAS:
        pr = perturbation_results[alpha]
        metadata["perturbation"][str(alpha)] = {
            "n_perturbed": pr["n_perturbed"],
            "n_total": pr["n_total"],
            "vortex_ratio_pct": pr["ratio_pct"],
        }

    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log("  Saved metadata.json")
    log("")

    # ─────────────────────────────────────────────────────────────
    # STEP 9: Write REPORT.md
    # ─────────────────────────────────────────────────────────────
    t_total = time.time() - t_start

    # Determine conclusion
    best_alpha = 0.1
    pr_best = perturbation_results[best_alpha]
    n_pert = pr_best["n_perturbed"]

    if n_pert <= 1:
        conclusion = "1 trap"
        suitability = "Excellent — single-trap selection achievable."
    elif n_pert <= 2:
        conclusion = "2 traps"
        suitability = "Good — perturbation is highly localised, nearest-neighbour selection feasible."
    elif n_pert <= 4:
        conclusion = "2–4 traps"
        suitability = "Acceptable — perturbation affects a small cluster; suitable for controlled selection with minor spillover."
    else:
        conclusion = f"{n_pert} traps"
        suitability = "Marginal — vortex perturbs a significant fraction of the 3×3 grid. Consider reducing aperture or α."

    report = f"""# FEM Standing Grid (3×3) + ASM Vortex Overlay Study

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run directory:** `{BASE.relative_to(PROJECT_ROOT)}`
**Total runtime:** {t_total:.1f}s

---

## Objective

Study how a focused ASM acoustic vortex (plastic lens) perturbs a
local 3×3 standing-wave trap grid.  The standing-wave field is from
a validated FEM solve; the vortex field is from the validated ASM
propagator.  The two are overlaid in post-processing.

---

## Physical Parameters

| Parameter | Value |
|-----------|-------|
| Frequency | {F_HZ/1e6:.1f} MHz |
| λ | {LAM*1e3:.4f} mm |
| k | {K_WATER:.1f} rad/m |
| c_water | {C_WATER} m/s |
| ρ_water | {RHO_WATER} kg/m³ |
| Trap spacing (λ/2) | {TRAP_SPACING*1e3:.4f} mm |
| z* (trap plane) | {Z_STAR*1e3:.3f} mm |
| 3×3 region size | {REGION_SIZE*1e3:.3f} mm ≈ {REGION_SIZE/LAM:.2f}λ |

### Vortex Lens

| Parameter | Value |
|-----------|-------|
| ℓ (topological charge) | {LENS_ELL} |
| R (aperture radius) | {LENS_R*1e3:.1f} mm |
| f (focal length) | {LENS_F*1e3:.1f} mm |
| Fresnel number N_F | {NF:.2f} |
| c_lens | {LENS_C} m/s |
| Apodization | {LENS_APOD} |

---

## Validation Checks

| Metric | Measured | Expected | Status |
|--------|----------|----------|--------|
| Trap spacing | {trap_sp_meas*1e3:.4f} mm | {TRAP_SPACING*1e3:.4f} mm | {"PASS" if not np.isnan(trap_sp_meas) and abs(trap_sp_meas - TRAP_SPACING)/TRAP_SPACING < 0.15 else "CHECK"} |
| Waist diameter | {waist_diameter*1e3:.4f} mm | < {LAM*1e3:.4f} mm (1λ) | {"PASS" if waist_diameter < LAM else "WARN: > 1λ"} |
| Waist diameter / λ | {waist_diameter/LAM:.3f} | < 1.0 | {"PASS" if waist_diameter < LAM else "WARN"} |

---

## Perturbation Results

| α | Vortex/Standing (%) | Traps perturbed | Total traps |
|---|---------------------|-----------------|-------------|
"""
    for alpha in ALPHAS:
        pr = perturbation_results[alpha]
        report += f"| {alpha} | {pr['ratio_pct']:.1f}% | {pr['n_perturbed']} | {pr['n_total']} |\n"

    report += f"""
---

## Conclusion

At α = {best_alpha} (vortex peak = {pr_best['ratio_pct']:.1f}% of standing peak):

**The vortex significantly perturbs {conclusion}.**

{suitability}

---

## Limitations

1. **No coupled FEM re-solve.** Standing and vortex fields are superposed
   linearly in post-processing.  There is no cavity–lens interaction.
2. **Linear acoustics assumption.** Superposition is only valid in the
   linear regime; no nonlinear radiation forces are computed.
3. **ASM is free-space.** The vortex propagation does not include
   reflections from the petri dish walls or water–air interface.
4. **FEM interpolation uses LinearNDInterpolator** — the P2 FEM field is
   sampled at DOF coordinates and linearly interpolated (Delaunay).
   For this qualitative study this is acceptable.

---

## Deliverables

- `figures/01_standing_only_xy.png` — Standing wave |p| at z*
- `figures/02_vortex_only_xy.png` — Vortex |p| at z* with waist annotation
- `figures/03_overlay_alpha_*.png` — Combined + difference for each α
- `figures/04_xz_meridional.png` — XZ slice showing hourglass + standing envelope
- `figures/05_phase_comparison.png` — Phase maps (standing, vortex, combined)
- `data/local_fields.npz` — Complex fields on the local 3×3 grid
- `data/xz_fields.npz` — Complex fields on the XZ meridional plane
- `data/metadata.json` — All parameters and computed metrics
- `REPORT.md` — This file
"""

    report_path = BASE / "REPORT.md"
    report_path.write_text(report)

    # Also save the log
    (BASE / "console_log.txt").write_text("\n".join(report_lines))

    log("=" * 72)
    log("STUDY COMPLETE")
    log("=" * 72)
    log(f"  Total time         : {t_total:.1f}s")
    log(f"  Waist diameter / λ : {waist_diameter/LAM:.3f}")
    log(f"  Traps perturbed    : {conclusion} (at α={best_alpha})")
    log(f"  Output             : {BASE.relative_to(PROJECT_ROOT)}")
    log("")


if __name__ == "__main__":
    main()
