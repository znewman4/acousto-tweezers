#!/usr/bin/env python3
"""
Study A — MPC vortex-merge robustness: adjacent-start geometry.

Correct physical setup:
  - Particle A (code pos_A): already inside the vortex ring.  The vortex
    starts at the trap ADJACENT to B (1 trap spacing = λ/2 ≈ 371 µm away)
    — a different adjacent trap is sampled per trial to test all 8 approach
    directions (N/NE/E/SE/S/SW/W/NW).
  - Particle B (code pos_B): sits in a randomly selected interior trap.
    Noise is modelled as additive complex noise on the **total pressure field**
    p_tot = β·p_sw + α·e^{iψ}·p_v (model-mismatch paradigm): phase_sweep()
    optimises against the clean ideal field; the chosen ψ is then applied to
    a noisy realisation of p_tot whose Gor'kov forces actually drive the
    particle dynamics.  One noise field per control step (dynamic) or per
    trial (static).
  - The vortex path moves step-by-step from the adjacent start trap toward
    B's trap centre over N_VORTEX_STEPS=20 steps (~19 µm/step), opening the
    ring with MPC phase control to merge A and B.

Physics:
  - MPC phase-sequence controller: at each of N_VORTEX_STEPS=20 vortex
    positions, phase_sweep() evaluates N_PSI=24 candidate ψ values and picks
    the one minimising  w_barrier·ΔU − w_pull·F_in_B + w_lateral·F_perp − w_retain·F_A_in.
  - Overdamped particle dynamics (Gor'kov force, 0.1 ms time step, 150 steps
    per vortex position).
  - ALL raw data (full trajectories + force metrics) are stored; thresholds
    are NOT baked into the saved data so figures can be regenerated with
    different CAPTURE_RADIUS values without re-running the simulation.

Varied parameters (100 trials):
  - B trap index: random interior trap (traps with ≥ 8 immediate neighbours)
  - Vortex start: random immediate neighbour of B → tests all 8 approach dirs
  - Noise amplitude: uniform random in [0, 20%] of peak total pressure per trial
  - Noise mode: static (one field per trial) or dynamic (one field per control step)

File outputs:
    results/dev/study_a_robustness/<timestamp>/
        trial_000.npz … trial_099.npz  (per-trial raw data, written live)
        trial_summary.csv               (one row per trial, flushed live)
        study_a_results.npz             (consolidated scalars, written at end)

    Deliverables/study_a_figures/
        figA1_outcome_scatter.{pdf,png}
        figA2_trajectories.{pdf,png}
        figA3_statistics.{pdf,png}
        figA4_direction_and_noise.{pdf,png}

Usage:
    python scripts/dev/study_a_robustness_figures.py
    python scripts/dev/study_a_robustness_figures.py --n_trials 3
    python scripts/dev/study_a_robustness_figures.py --skip_simulation \\
        --out_dir results/dev/study_a_robustness/<timestamp>
    python scripts/dev/study_a_robustness_figures.py --no_figures
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# ── Project path ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.lib.fem_cache_utils import (
    C_WATER, F_HZ, OMEGA, RHO0,
)
from scripts.lib.particle_dynamics_utils import (
    LAM, TRAP_SP, CAPTURE_RADIUS, SCALE, DT_DEFAULT,
    gorkov_normalised,
)
from acoustweezers.experiments.vortex_entry.fields.vortex_source import load_data
from acoustweezers.experiments.vortex_entry.fields.field_superposition import total_pressure
from acoustweezers.experiments.vortex_entry.control.vortex_entry import (
    build_vortex_path,
    phase_sweep,
)
from acoustweezers.experiments.vortex_entry.particles.dynamics import (
    update_particles,
)
from acoustweezers.experiments.vortex_entry.utils.interpolation import (
    eval_at, make_interp,
)

# ── Canonical field NPZ (same as vortex_entry_test.py) ────────────────────────
FIELD_NPZ = (
    PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
    / "transport" / "transport_case_for_gif.npz"
)

# ── MPC baseline parameters (match vortex_entry_test.py defaults) ──────────
ALPHA          = 2.0
BETA_FIXED     = 1.0
N_PSI          = 24
N_VORTEX_STEPS = 20
N_DYN_STEPS    = 150
W_BARRIER      = 1.0
W_PULL         = 1.0
W_LATERAL      = 0.5
W_RETAIN       = 1.0
MAX_STEP       = 2e-6      # 2 µm per dynamics step
APERTURE_M     = 3.5e-3
PROP_DIST_M    = 3.0e-3
FOCUS_M        = 3.0e-3

DT_MS          = DT_DEFAULT * 1e3   # 0.1 ms per step
NBOUR_MAX_DIST   = 1.6 * TRAP_SP   # distance threshold for context-tracking neighbours
NOISE_CORR_FAC   = 0.5            # correlation length = NOISE_CORR_FAC × TRAP_SP
OBS_NOISE_MAX    = 0.15 * TRAP_SP   # max controller observation noise σ (metres)

RNG_SEED       = 42
HEARTBEAT_FREQ = 10   # print heartbeat every N trials

DIR_ORDER = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
N_SIGMA_STRATA = 20   # number of strata across [0, 1] noise scale

# ── Figure style ──────────────────────────────────────────────────────────────
FIG_WIDTH_MM  = 190.0
DPI           = 300
FONT_SANS     = "DejaVu Sans"

COL_CAPTURED  = "#2ecc71"   # green
COL_TIMEOUT   = "#e67e22"   # orange
COL_VORTEX    = "#e74c3c"   # red (A home star)
COL_TRAP_LAT  = "#aaaaaa"   # light grey lattice markers

plt.rcParams.update({
    "font.family":      FONT_SANS,
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.labelweight": "bold",
    "axes.titlesize":   11,
    "figure.dpi":       DPI,
    "savefig.dpi":      DPI,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
})

FIG_OUT_DIR = Path("/home/znewman4/projects/acousto-tweezers/Deliverables/study_a_figures")

# Capture threshold used IN FIGURES (post-hoc, independent of simulation).
# 250 µm = 0.67 × TRAP_SP — chosen because the distribution of min(d_AB)
# shows a natural gap from 250–371 µm: trials either clearly enter the
# Gor'kov well (<250 µm) or fail to engage at all (>371 µm).
FIG_CAPTURE_RADIUS = 0.67 * TRAP_SP   # ≈ 250 µm


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def _inches(mm: float) -> float:
    return mm / 25.4


def _update_single(
    pos: np.ndarray,
    iFx,
    iFy,
    max_step: float,
    xg: np.ndarray,
    yg: np.ndarray,
) -> np.ndarray:
    """
    Advance a single particle by one overdamped dynamics step.
    Mirrors the exact physics in update_particles() without requiring a dummy
    second particle.
    """
    Fx = float(eval_at(iFx, pos[None, :])[0])
    Fy = float(eval_at(iFy, pos[None, :])[0])
    new = pos + np.clip(
        np.array([SCALE * DT_DEFAULT * Fx, SCALE * DT_DEFAULT * Fy]),
        -max_step, max_step,
    )
    new[0] = np.clip(new[0], xg[2], xg[-3])
    new[1] = np.clip(new[1], yg[2], yg[-3])
    return new


def _save_fig(fig: plt.Figure, stem: str) -> None:
    """Save figure as both PDF and PNG to FIG_OUT_DIR."""
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG_OUT_DIR / f"{stem}.{ext}"
        fig.savefig(str(p), dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def _add_scale_bar(
    ax: plt.Axes,
    bar_mm: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    """Draw a horizontal scale bar in the lower-left corner of ax (mm units)."""
    x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    y_text = ylim[0] + 0.09 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + bar_mm], [y0, y0], "k-", lw=2,
            solid_capstyle="butt", zorder=10)
    ax.text(x0 + bar_mm / 2, y_text, f"{bar_mm:g} mm",
            ha="center", va="bottom", fontsize=9, zorder=10)


def _make_pressure_noise(
    shape: Tuple[int, int],
    noise_frac: float,
    p_ref_peak: float,
    corr_length_px: float,
    rng: np.random.Generator,
    noise_white_frac: float = 0.0,
) -> np.ndarray:
    """
    Generate composite complex Gaussian noise with RMS amplitude equal to
    ``noise_frac * p_ref_peak``.

    The noise is a weighted mix of two independently normalised components:
      * Smooth component (weight 1 - noise_white_frac): white noise smoothed
        with a Gaussian kernel of width ``corr_length_px`` pixels.  Models
        transducer phase/amplitude jitter and low-spatial-frequency field error.
      * White component  (weight noise_white_frac): unsmoothed pixel-scale
        noise.  Models secondary radiation forces, streaming, sub-wavelength
        reflections and other effects that act at scales below the trap spacing.

    Each component is normalised to unit complex RMS before mixing so that
    ``noise_white_frac`` controls the *energy* split, not the amplitude split.
    The composite is then jointly scaled to ``noise_frac * p_ref_peak`` RMS.
    """
    re_w = rng.standard_normal(shape).astype(float)
    im_w = rng.standard_normal(shape).astype(float)

    # ── Smooth component ──────────────────────────────────────────────────
    if corr_length_px > 0 and noise_white_frac < 1.0:
        re_s = gaussian_filter(re_w, sigma=corr_length_px)
        im_s = gaussian_filter(im_w, sigma=corr_length_px)
        rms_s = float(np.sqrt(np.mean(re_s ** 2 + im_s ** 2)))
        if rms_s > 1e-30:
            re_s /= rms_s
            im_s /= rms_s
    else:
        re_s = np.zeros(shape, dtype=float)
        im_s = np.zeros(shape, dtype=float)

    # ── White component ───────────────────────────────────────────────────
    if noise_white_frac > 0.0:
        # Independent draw so smooth and white components are uncorrelated
        re_u = rng.standard_normal(shape).astype(float)
        im_u = rng.standard_normal(shape).astype(float)
        rms_u = float(np.sqrt(np.mean(re_u ** 2 + im_u ** 2)))
        if rms_u > 1e-30:
            re_u /= rms_u
            im_u /= rms_u
    else:
        re_u = np.zeros(shape, dtype=float)
        im_u = np.zeros(shape, dtype=float)

    # ── Mix, normalise, scale ─────────────────────────────────────────────
    w_s = (1.0 - noise_white_frac)
    w_u = noise_white_frac
    re = w_s * re_s + w_u * re_u
    im = w_s * im_s + w_u * im_u
    rms = float(np.sqrt(np.mean(re ** 2 + im ** 2)))
    if rms < 1e-30:
        return np.zeros(shape, dtype=complex)
    scale = noise_frac * p_ref_peak / rms
    return (re + 1j * im) * scale


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% confidence interval; returns (lo_pct, hi_pct)."""
    if n == 0:
        return 0.0, 0.0
    centre = (k + z ** 2 / 2.0) / (n + z ** 2)
    half = z * np.sqrt(k * (n - k) / n + z ** 2 / 4.0) / (n + z ** 2)
    return max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0


def _angle_to_compass(ang_deg: float) -> str:
    """Map atan2 angle (degrees, 0=East CCW) to one of 8 compass labels."""
    ang_deg = ang_deg % 360.0
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    return dirs[int((ang_deg + 22.5) // 45) % 8]


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(args) -> Path:
    """
    Run N_TRIALS MPC vortex-merge trials with randomised B starting positions.
    Saves per-trial NPZ + summary CSV incrementally. Returns out_dir Path.
    """
    print("=" * 72)
    print("Study A — MPC Vortex-Merge Robustness (N = {})".format(args.n_trials))
    print("=" * 72)

    # ── Load field data ────────────────────────────────────────────────────────
    print("\n[Setup] Loading field data …", flush=True)
    t_load = time.time()
    (p_sw, xg, yg, dx, dy, traps_m,
     vortex_gen, r_barrier) = load_data(
        APERTURE_M, PROP_DIST_M, FOCUS_M, field_npz=FIELD_NPZ)
    print(f"  Loaded in {time.time() - t_load:.1f}s  |  "
          f"{len(traps_m)} traps  |  r_barrier = {r_barrier * 1e6:.1f} µm",
          flush=True)

    # ── All traps are eligible as B; vortex start is sampled from nearest 8 ──
    # Every trap has 8 nearest traps by rank; no grid-regularity assumption.
    eligible_pool = list(range(len(traps_m)))
    print(f"  Eligible pool : {len(eligible_pool)} traps (all traps on lattice)",
          flush=True)

    # ── Noise mode ─────────────────────────────────────────────────────────
    noise_mode: str = getattr(args, "noise_mode", "dynamic")
    sw_noise_pct: float = float(getattr(args, "sw_noise_pct", 10.0))
    vortex_noise_pct: float = float(getattr(args, "vortex_noise_pct", 10.0))
    sw_noise_white_frac: float = float(getattr(args, "sw_noise_white_frac", 0.0))
    vortex_noise_white_frac: float = float(getattr(args, "vortex_noise_white_frac", 0.0))
    print(f"  Noise mode    : {noise_mode}", flush=True)
    print(f"  SW noise      : {sw_noise_pct:.1f}% of peak |p_sw|  "
          f"(white frac {sw_noise_white_frac*100:.0f}%)", flush=True)
    print(f"  Vortex noise  : {vortex_noise_pct:.1f}% of peak |p_v|  "
          f"(white frac {vortex_noise_white_frac*100:.0f}%)", flush=True)

    obs_noise_frac: float = float(getattr(args, "obs_noise_frac", 1.0))
    print(f"  Obs noise     : {obs_noise_frac:.2f} × {OBS_NOISE_MAX*1e6:.0f} µm = "
          f"{obs_noise_frac * OBS_NOISE_MAX * 1e6:.0f} µm max σ", flush=True)

    # ── Pre-compute correlation length in pixels (needs dx from load_data) ──
    corr_length_px = float((NOISE_CORR_FAC * TRAP_SP) / dx)

    # ── Random-ψ baseline flag ─────────────────────────────────────────────
    random_psi: bool = getattr(args, "random_psi", False)
    if random_psi:
        print("  Mode          : RANDOM-ψ BASELINE (MPC result discarded)",
              flush=True)

    n_trials = args.n_trials

    # ── Direction schedule ─────────────────────────────────────────────────
    fixed_direction: Optional[str] = getattr(args, "direction", None)

    if fixed_direction:
        # --direction overrides: all trials use this single direction
        dir_schedule = np.array([fixed_direction] * n_trials)
        print(f"  Direction lock: {fixed_direction} (vortex always approaches from this side)",
              flush=True)
    else:
        # Balanced schedule: exactly n_trials/8 per compass direction
        assert n_trials % 8 == 0, (
            f"--n_trials must be divisible by 8 for balanced direction "
            f"sampling; got {n_trials}")
        dir_schedule = np.repeat(DIR_ORDER, n_trials // 8)
        rng_sched = np.random.default_rng(RNG_SEED)
        rng_sched.shuffle(dir_schedule)
        print(f"  Direction     : balanced schedule ({n_trials // 8} trials "
              f"per direction)", flush=True)

    # ── Noise-scale schedule ──────────────────────────────────────────────────
    # noise_scale in [0, 1]: fraction of the configured max noise pcts.
    # Effective noise per trial:  sw = noise_scale * sw_noise_pct/100 * peak_sw
    #                             vtx = noise_scale * vortex_noise_pct/100 * peak_vtx
    noise_grid_scale: Optional[np.ndarray] = None
    noise_schedule_scale: Optional[np.ndarray] = None
    if getattr(args, "sigma_grid", None):
        noise_grid_scale = np.array([float(v) for v in args.sigma_grid.split(",")]) / 100.0
        # Build a balanced + shuffled noise schedule so that noise level is
        # independent of the direction schedule (avoids confounding).
        n_levels = len(noise_grid_scale)
        per_level = n_trials // n_levels
        assert n_trials % n_levels == 0, (
            f"--n_trials ({n_trials}) must be divisible by the number of "
            f"sigma_grid levels ({n_levels})")
        noise_schedule_scale = np.repeat(noise_grid_scale, per_level)
        rng_noise_sched = np.random.default_rng(RNG_SEED + 2)
        rng_noise_sched.shuffle(noise_schedule_scale)
        print(f"  Noise grid    : {args.sigma_grid} %  ({n_levels} levels, "
              f"{per_level} trials/level, shuffled)",
              flush=True)
    else:
        # Stratified sampling: N_SIGMA_STRATA equal strata across [0, 1]
        assert n_trials % N_SIGMA_STRATA == 0, (
            f"--n_trials must be divisible by {N_SIGMA_STRATA} for stratified "
            f"noise sampling; got {n_trials}")
        strata_edges = np.linspace(0.0, 1.0, N_SIGMA_STRATA + 1)
        rng_sigma = np.random.default_rng(RNG_SEED + 1)
        noise_schedule_scale = np.empty(n_trials)
        per_stratum = n_trials // N_SIGMA_STRATA
        for si in range(N_SIGMA_STRATA):
            lo, hi = strata_edges[si], strata_edges[si + 1]
            noise_schedule_scale[si * per_stratum:(si + 1) * per_stratum] = \
                rng_sigma.uniform(lo, hi, size=per_stratum)
        rng_sigma.shuffle(noise_schedule_scale)
        print(f"  Noise         : stratified ({N_SIGMA_STRATA} strata × "
              f"{per_stratum}/stratum, [0, 100%])",
              flush=True)

    # ── Output directory ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "dev" / "study_a_robustness" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir    : {out_dir}", flush=True)

    # ── Pre-compute constants ──────────────────────────────────────────────────
    psi_values = np.linspace(0, 2 * np.pi, N_PSI, endpoint=False)
    rng = np.random.default_rng(RNG_SEED)
    rng_trap = np.random.default_rng(RNG_SEED + 100)  # dedicated trap RNG

    # ── Open CSV for live appending ────────────────────────────────────────────
    csv_path = out_dir / "trial_summary.csv"
    csv_fh = open(csv_path, "w", newline="", buffering=1)
    writer = csv.writer(csv_fh)
    writer.writerow([
        "trial", "b_trap_idx",
        "b_trap_x_mm", "b_trap_y_mm",
        "vortex_start_x_mm", "vortex_start_y_mm",
        "approach_dir", "noise_scale_pct",
        "sw_noise_eff_pct", "vortex_noise_eff_pct",
        "outcome", "sequence_length",
        "capture_dyn_step", "capture_time_ms",
    ])
    csv_fh.flush()

    # ── In-memory accumulators (for consolidated save at end) ──────────────────
    acc_outcomes:    List[str]        = []
    acc_b_traps:     List[np.ndarray] = []
    acc_dirs:        List[str]        = []
    acc_noise_scales: List[float]     = []
    acc_seq_lens:    List[int]        = []
    acc_cap_steps:   List[int]        = []

    t_study_start = time.time()
    recent_batch:  List[str]   = []

    # ══════════════════════════════════════════════════════════════════════════
    # Trial loop
    # ══════════════════════════════════════════════════════════════════════════
    for trial_i in range(n_trials):
        t_trial_start = time.time()

        # ── Sample B's trap and vortex start (adjacent to B) ─────────────────
        b_idx  = int(rng_trap.choice(eligible_pool))
        B_trap = traps_m[b_idx].copy()      # B's lattice trap centre

        # Pick a random immediate neighbour of B as the vortex start position
        d_to_B    = np.linalg.norm(traps_m - B_trap, axis=1)
        nn_dists  = d_to_B.copy()
        nn_dists[b_idx] = np.inf         # exclude B itself
        nn8_idxs  = np.argsort(nn_dists)[:8]   # 8 nearest traps by rank

        # Direction selection: pick the nearest neighbour whose approach
        # angle best matches the scheduled compass direction.
        target_dir = dir_schedule[trial_i]
        dir_labels = []
        for idx in nn8_idxs:
            dx_ = B_trap[0] - traps_m[idx, 0]
            dy_ = B_trap[1] - traps_m[idx, 1]
            dir_labels.append(_angle_to_compass(
                float(np.degrees(np.arctan2(-dy_, -dx_)))))
        exact_matches = [nn8_idxs[k] for k, d in enumerate(dir_labels)
                         if d == target_dir]
        if exact_matches:
            vtx_start_idx = int(rng_trap.choice(exact_matches))
        else:
            # No exact match — pick the closest-angle neighbour
            dir_idx = DIR_ORDER.index(target_dir)
            def _dir_dist(label: str) -> int:
                j = DIR_ORDER.index(label)
                return min(abs(j - dir_idx), 8 - abs(j - dir_idx))
            best_k = int(np.argmin([_dir_dist(d) for d in dir_labels]))
            vtx_start_idx = int(nn8_idxs[best_k])

        A_start   = traps_m[vtx_start_idx].copy()   # vortex start = particle A home

        # Approach direction: use the scheduled target direction to guarantee
        # exact balance; the neighbour selection already matched this as
        # closely as the local lattice allows.
        approach_dir = target_dir

        # ── Noise scale: shuffled schedule (sigma_grid or stratified) > uniform random ─
        if noise_schedule_scale is not None:
            noise_scale = float(noise_schedule_scale[trial_i])
        else:
            noise_scale = float(rng.uniform(0.0, 1.0))
        sw_noise_eff = noise_scale * sw_noise_pct / 100.0    # fraction of peak |p_sw|
        vortex_noise_eff = noise_scale * vortex_noise_pct / 100.0  # fraction of peak |p_v|
        B_start = B_trap.copy()

        # ── 8 nearest-neighbour traps for context tracking ────────────────────
        d_nbr = d_to_B.copy()
        d_nbr[b_idx] = np.inf   # exclude B itself
        nbr_indices = np.argsort(d_nbr)[:8]
        nbr_home    = traps_m[nbr_indices].copy()   # (8, 2) metres

        # ── Vortex path: A_start → B_trap (targets trap centre, not noisy pos) ─
        sep_m       = float(np.linalg.norm(B_trap - A_start))
        vortex_path = build_vortex_path(A_start, B_trap, N_VORTEX_STEPS)

        # ── Initialise particle positions ─────────────────────────────────────
        pos_A   = A_start.copy()   # particle inside vortex ring
        pos_B   = B_start.copy()   # particle B at true trap centre
        nbr_pos = nbr_home.copy()  # (8, 2)

        # ── Per-vortex-step arrays ─────────────────────────────────────────────
        # Allocated at full size; only [:seq_len] stored in NPZ on capture.
        vc_arr    = np.empty((N_VORTEX_STEPS, 2))
        psi_arr   = np.empty(N_VORTEX_STEPS)
        dU_arr    = np.empty(N_VORTEX_STEPS)
        Fin_arr   = np.empty(N_VORTEX_STEPS)
        Fain_arr  = np.empty(N_VORTEX_STEPS)
        score_arr = np.empty(N_VORTEX_STEPS)

        # ── Per-dynamics-step storage (filled during the loops) ───────────────
        traj_A_list:   List[np.ndarray] = []   # pos_A (diss A) each dyn step
        traj_B_list:   List[np.ndarray] = []   # pos_B (diss B) each dyn step
        traj_nbr_list: List[np.ndarray] = []   # nbr_pos (8,2) each dyn step
        d_ab_list:     List[float]      = []   # ||pos_A – pos_B|| each dyn step

        # ── Outcome state ─────────────────────────────────────────────────────
        captured         = False
        capture_dyn_step = -1             # total dynamics step index of capture
        seq_len          = N_VORTEX_STEPS  # overwritten if captured

        print(f"\n[Trial {trial_i + 1:03d}/{n_trials}]  "
              f"b_idx={b_idx:3d}  dir={approach_dir}  "
              f"sep={sep_m * 1e3:.3f}mm  scale={noise_scale*100:.0f}%  "
              f"sw={sw_noise_eff*100:.1f}%  vtx={vortex_noise_eff*100:.1f}%  mode={noise_mode}",
              flush=True)

        # ── Compute per-component peak amplitudes ─────────────────────────────
        peak_sw = float(np.max(np.abs(p_sw)))
        p_v_sample = vortex_gen.get_field(vortex_path[N_VORTEX_STEPS // 2])
        peak_vortex = float(np.max(np.abs(p_v_sample)))

        # ── Static mode: one noise field pair for the entire trial ────────────
        n_sw_trial: Optional[np.ndarray] = None
        n_v_trial: Optional[np.ndarray] = None
        if noise_mode == "static":
            n_sw_trial = _make_pressure_noise(
                p_sw.shape, sw_noise_eff, peak_sw, corr_length_px, rng,
                noise_white_frac=sw_noise_white_frac)
            n_v_trial = _make_pressure_noise(
                p_sw.shape, vortex_noise_eff, peak_vortex, corr_length_px, rng,
                noise_white_frac=vortex_noise_white_frac)

        # ── MPC loop ────────────────────────────────────────────────────────
        for v_step in range(N_VORTEX_STEPS):
            vortex_center = vortex_path[v_step].copy()
            vc_arr[v_step] = vortex_center

            t_ps = time.time()
            # Noise realisation for this control step: one field pair per step
            # (dynamic) or reuse the per-trial frozen fields (static).
            if noise_mode == "static":
                n_sw_step = n_sw_trial
                n_v_step = n_v_trial
            else:  # dynamic
                n_sw_step = _make_pressure_noise(
                    p_sw.shape, sw_noise_eff, peak_sw, corr_length_px, rng,
                    noise_white_frac=sw_noise_white_frac)
                n_v_step = _make_pressure_noise(
                    p_sw.shape, vortex_noise_eff, peak_vortex, corr_length_px, rng,
                    noise_white_frac=vortex_noise_white_frac)

            # Controller sees the CLEAN field but a NOISY estimate of B's
            # position — models measurement / imaging noise.  The observation
            # noise σ scales with noise_scale so that at 0% noise the
            # controller has a perfect view and at max noise it is maximally
            # degraded.   This guarantees a monotonically harmful noise effect
            # because optimising for the wrong target is strictly sub-optimal.
            obs_sigma = noise_scale * obs_noise_frac * OBS_NOISE_MAX
            pos_B_obs = pos_B + rng.normal(0.0, max(obs_sigma, 1e-30), size=2)
            best = phase_sweep(
                p_sw, vortex_gen, xg, yg, dx, dy,
                psi_values, ALPHA, BETA_FIXED,
                vortex_center, pos_A, pos_B_obs, r_barrier,
                w_barrier=W_BARRIER, w_pull=W_PULL,
                w_lateral=W_LATERAL, w_retain=W_RETAIN,
                p_noise=None,
            )
            psi_arr[v_step]   = float(best["psi"])
            dU_arr[v_step]    = float(best["DeltaU"])
            Fin_arr[v_step]   = float(best["F_in_B"])
            Fain_arr[v_step]  = float(best["F_A_in"])
            score_arr[v_step] = float(best["score"])

            # Physics runs on the NOISY field — perturb SW and vortex separately
            # before superposition so each component's noise is scaled to its own
            # peak amplitude.  gorkov_normalised matches the units that
            # phase_sweep / update_particles expect.
            p_v_phys = vortex_gen.get_field(vortex_center)
            p_tot_phys = (BETA_FIXED * (p_sw + n_sw_step)
                          + ALPHA * np.exp(1j * psi_arr[v_step]) * (p_v_phys + n_v_step))
            _, Fx_phys, Fy_phys = gorkov_normalised(p_tot_phys, dx, dy)
            iFx = make_interp(Fx_phys, xg, yg)
            iFy = make_interp(Fy_phys, xg, yg)

            # ── Random-ψ baseline: discard MPC result, use random phase ──
            if random_psi:
                psi_rand = float(rng.uniform(0.0, 2 * np.pi))
                psi_arr[v_step] = psi_rand
                p_v_rand = vortex_gen.get_field(vortex_center)
                p_total_rand = (BETA_FIXED * (p_sw + n_sw_step)
                                + ALPHA * np.exp(1j * psi_rand) * (p_v_rand + n_v_step))
                _, Fx_rand, Fy_rand = gorkov_normalised(p_total_rand, dx, dy)
                iFx = make_interp(Fx_rand, xg, yg)
                iFy = make_interp(Fy_rand, xg, yg)

            dt_ps = time.time() - t_ps

            print(
                f"  v{v_step + 1:02d}/{N_VORTEX_STEPS}  "
                f"psi={psi_arr[v_step]:.2f}  "
                f"dU={dU_arr[v_step]:.2e}  "
                f"F_in={Fin_arr[v_step]:.2e}  "
                f"F_Ain={Fain_arr[v_step]:.2e}  "
                f"({dt_ps * 1e3:.0f}ms)",
                flush=True,
            )

            # ── Dynamics ──────────────────────────────────────────────────────
            for dyn_i in range(N_DYN_STEPS):
                pos_A, pos_B = update_particles(
                    pos_A, pos_B, iFx, iFy, MAX_STEP, xg, yg)
                for j in range(8):
                    nbr_pos[j] = _update_single(
                        nbr_pos[j], iFx, iFy, MAX_STEP, xg, yg)

                traj_A_list.append(pos_A.copy())
                traj_B_list.append(pos_B.copy())
                traj_nbr_list.append(nbr_pos.copy())

                d_ab = float(np.linalg.norm(pos_A - pos_B))
                d_ab_list.append(d_ab)

                # Check for capture at this exact step
                if not captured and d_ab < CAPTURE_RADIUS:
                    captured         = True
                    total_dyn_idx    = v_step * N_DYN_STEPS + dyn_i
                    capture_dyn_step = total_dyn_idx
                    seq_len          = v_step + 1
                    print(
                        f"  *** CAPTURED  v_step={v_step + 1}  "
                        f"dyn_i={dyn_i}  d_AB={d_ab * 1e6:.1f} µm ***",
                        flush=True,
                    )

            if captured:
                # Complete this vortex step's dynamics are already done;
                # stop the MPC loop so we don't simulate further steps.
                break

        # ── Derive outcome ─────────────────────────────────────────────────────
        outcome  = "CAPTURED" if captured else "TIMEOUT"
        cap_t_ms = (capture_dyn_step + 1) * DT_MS if captured else None

        acc_outcomes.append(outcome)
        acc_b_traps.append(B_trap.copy())
        acc_dirs.append(approach_dir)
        acc_noise_scales.append(noise_scale)
        acc_seq_lens.append(seq_len)
        acc_cap_steps.append(capture_dyn_step)
        recent_batch.append(outcome)

        dt_trial = time.time() - t_trial_start
        if cap_t_ms is not None:
            print(
                f"  → {outcome}  seq={seq_len}  "
                f"t_cap={cap_t_ms:.1f} ms  "
                f"(trial took {dt_trial:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"  → {outcome}  seq={seq_len}  "
                f"(trial took {dt_trial:.1f}s)",
                flush=True,
            )

        # ── Save per-trial NPZ ─────────────────────────────────────────────────
        # Convert lists to arrays
        T = len(d_ab_list)
        traj_A_arr   = np.array(traj_A_list)    # (T, 2)  metres
        traj_B_arr   = np.array(traj_B_list)    # (T, 2)  metres
        traj_nbr_arr = np.array(traj_nbr_list)  # (T, 8, 2) metres
        d_ab_arr     = np.array(d_ab_list)       # (T,)   metres

        # Per-vortex-step arrays: only save completed steps
        n_vsteps_done = seq_len  # vortex steps actually executed
        np.savez(
            str(out_dir / f"trial_{trial_i:03d}.npz"),
            # ── Trial identity ──────────────────────────────────────────
            trial_i=np.int32(trial_i),
            b_diss_idx=np.int32(b_idx),
            # ── Starting positions (metres) ─────────────────────────────
            b_diss_start_xy=B_start,         # (2,)
            a_diss_start_xy=A_start,          # (2,)
            separation_m=np.float64(sep_m),
            # ── Neighbour information ───────────────────────────────────
            neighbour_indices=nbr_indices.astype(np.int32),   # (8,)
            neighbour_home_xy=nbr_home,                       # (8, 2)
            # ── Per-vortex-step arrays (only completed steps saved) ─────
            vortex_centers=vc_arr[:n_vsteps_done],   # (S, 2)
            psi_best=psi_arr[:n_vsteps_done],         # (S,)  rad
            delta_U=dU_arr[:n_vsteps_done],           # (S,)
            F_in_B=Fin_arr[:n_vsteps_done],           # (S,)
            F_A_in=Fain_arr[:n_vsteps_done],          # (S,)
            score=score_arr[:n_vsteps_done],          # (S,)
            # ── Per-dynamics-step arrays (T rows, T ≤ 3000) ────────────
            traj_A_diss=traj_A_arr,      # (T, 2)  dissertation A (inside vortex)
            traj_B_diss=traj_B_arr,      # (T, 2)  dissertation B (SW trap particle)
            traj_neighbours=traj_nbr_arr,# (T, 8, 2)
            d_ab_series=d_ab_arr,        # (T,)    ||A–B|| at each step
            # ── Outcome ─────────────────────────────────────────────────
            outcome=np.bytes_(outcome),
            sequence_length=np.int32(seq_len),
            capture_dyn_step=np.int32(capture_dyn_step),
            # ── Frozen reference constants (so figures are threshold-free) ─
            CAPTURE_RADIUS_USED=np.float64(CAPTURE_RADIUS),
            TRAP_SP_USED=np.float64(TRAP_SP),
            DT_S=np.float64(DT_DEFAULT),
            ALPHA_USED=np.float64(ALPHA),
            N_VORTEX_STEPS_USED=np.int32(N_VORTEX_STEPS),
            N_DYN_STEPS_USED=np.int32(N_DYN_STEPS),
        )

        # ── Append to CSV ──────────────────────────────────────────────────────
        writer.writerow([
            trial_i,
            b_idx,
            f"{B_trap[0] * 1e3:.4f}",
            f"{B_trap[1] * 1e3:.4f}",
            f"{A_start[0] * 1e3:.4f}",
            f"{A_start[1] * 1e3:.4f}",
            approach_dir,
            f"{noise_scale * 100:.4f}",
            f"{sw_noise_eff * 100:.4f}",
            f"{vortex_noise_eff * 100:.4f}",
            outcome,
            seq_len,
            capture_dyn_step,
            f"{cap_t_ms:.3f}" if cap_t_ms is not None else "",
        ])
        csv_fh.flush()

        # ── Heartbeat every HEARTBEAT_FREQ trials ──────────────────────────────
        if (trial_i + 1) % HEARTBEAT_FREQ == 0:
            elapsed  = time.time() - t_study_start
            done     = trial_i + 1
            n_cap_batch = sum(1 for o in recent_batch if o == "CAPTURED")
            n_cap_total = sum(1 for o in acc_outcomes if o == "CAPTURED")
            eta_s    = (elapsed / done) * (n_trials - done)
            print()
            print("─" * 72)
            print(
                f"[Heartbeat] Trial {done:3d}/{n_trials}  |  "
                f"Batch {HEARTBEAT_FREQ}: {n_cap_batch}/{len(recent_batch)} cap  |  "
                f"Overall: {n_cap_total}/{done} ({100.0 * n_cap_total / done:.0f}%)  |  "
                f"Elapsed: {elapsed:.0f}s  |  ETA: ~{eta_s / 60:.0f}min"
            )
            print("─" * 72)
            recent_batch.clear()

    csv_fh.close()

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_study_start
    n_cap = sum(1 for o in acc_outcomes if o == "CAPTURED")
    print()
    print("=" * 72)
    print(
        f"[Simulation complete]  "
        f"{n_cap}/{n_trials} CAPTURED ({100.0 * n_cap / n_trials:.0f}%)  |  "
        f"Total: {total_elapsed / 60:.1f} min"
    )

    # ── Consolidated NPZ (scalar arrays only — trajectories live in per-trial files) ─
    np.savez(
        str(out_dir / "study_a_results.npz"),
        outcomes=np.array(acc_outcomes),                           # (N,) str
        b_traps=np.array(acc_b_traps),                             # (N, 2) trap centres (m)
        approach_dirs=np.array(acc_dirs),                          # (N,) compass labels
        noise_scales=np.array(acc_noise_scales, dtype=float),      # (N,) noise scale [0-1]
        sequence_lengths=np.array(acc_seq_lens, dtype=int),        # (N,)
        capture_dyn_steps=np.array(acc_cap_steps, dtype=int),      # (N,)
        traps_m=traps_m,                                           # (M, 2)
        TRAP_SP=np.float64(TRAP_SP),
        CAPTURE_RADIUS=np.float64(CAPTURE_RADIUS),
        DT_S=np.float64(DT_DEFAULT),
        # study metadata — empty string encodes "None"
        fixed_direction=np.str_(fixed_direction or ""),
        noise_grid_pct=(noise_grid_scale * 100.0 if noise_grid_scale is not None
                        else np.array([], dtype=float)),
        noise_mode=np.str_(noise_mode),
        sw_noise_pct=np.float64(sw_noise_pct),
        vortex_noise_pct=np.float64(vortex_noise_pct),
        sw_noise_white_frac=np.float64(sw_noise_white_frac),
        vortex_noise_white_frac=np.float64(vortex_noise_white_frac),
        obs_noise_frac=np.float64(obs_noise_frac),
        random_psi=np.bool_(random_psi),
    )
    print(f"  Consolidated : {out_dir / 'study_a_results.npz'}")
    print(f"  Summary CSV  : {csv_path}")
    print("=" * 72)

    return out_dir


# ══════════════════════════════════════════════════════════════════════════════
# Figure data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_results(out_dir: Path) -> dict:
    """
    Load consolidated scalars + all per-trial raw arrays.
    Outcomes are RE-DERIVED from d_ab_series using FIG_CAPTURE_RADIUS,
    which may differ from the threshold used during simulation.
    """
    cons = np.load(str(out_dir / "study_a_results.npz"), allow_pickle=True)

    b_traps      = np.array(cons["b_traps"])          # (N, 2) trap centres, metres
    approach_dirs= np.array(cons["approach_dirs"])    # (N,) compass strings
    noise_scales = np.array(cons["noise_scales"])     # (N,) noise scale [0-1]
    traps_m      = np.array(cons["traps_m"])          # (M, 2)
    TRAP_SP_     = float(cons["TRAP_SP"])
    CR_sim       = float(cons["CAPTURE_RADIUS"])
    DT_S_        = float(cons["DT_S"])
    n_trials     = len(b_traps)

    # Load per-trial B trajectories and derive outcomes at FIG_CAPTURE_RADIUS
    t0       = np.load(str(out_dir / "trial_000.npz"), allow_pickle=True)
    N_VSTEPS = int(t0["N_VORTEX_STEPS_USED"])
    N_DSTEPS = int(t0["N_DYN_STEPS_USED"])

    fig_cr        = FIG_CAPTURE_RADIUS
    trajs_B:      List[np.ndarray] = []
    fig_outcomes  = np.empty(n_trials, dtype=object)
    fig_cap_steps = np.full(n_trials, -1, dtype=int)
    fig_seq_lens  = np.full(n_trials, N_VSTEPS, dtype=int)

    for i in range(n_trials):
        td       = np.load(str(out_dir / f"trial_{i:03d}.npz"), allow_pickle=True)
        d_series = td["d_ab_series"]
        trajs_B.append(td["traj_B_diss"])   # (T, 2) metres

        hits = np.where(d_series < fig_cr)[0]
        if len(hits) > 0:
            fig_outcomes[i]  = "CAPTURED"
            fig_cap_steps[i] = int(hits[0])
            fig_seq_lens[i]  = int(hits[0] // N_DSTEPS) + 1
        else:
            fig_outcomes[i]  = "TIMEOUT"

    n_cap_sim = int(np.sum(np.array([str(o) for o in cons["outcomes"]]) == "CAPTURED"))
    n_cap_fig = int(np.sum(fig_outcomes == "CAPTURED"))
    print(f"  Sim threshold  : {CR_sim * 1e6:.0f} µm  →  {n_cap_sim}/{n_trials} CAPTURED")
    print(f"  Figure threshold: {fig_cr * 1e6:.0f} µm  →  {n_cap_fig}/{n_trials} CAPTURED")

    return dict(
        outcomes=fig_outcomes,
        b_traps=b_traps,
        approach_dirs=approach_dirs,
        noise_scales=noise_scales,
        seq_lens=fig_seq_lens,
        cap_steps=fig_cap_steps,
        traps_m=traps_m,
        TRAP_SP=TRAP_SP_,
        CR=fig_cr,
        DT_S=DT_S_,
        trajs_B=trajs_B,
        n_trials=n_trials,
        # study metadata (may be absent in data from before this feature)
        fixed_direction=str(cons["fixed_direction"]) if "fixed_direction" in cons else "",
        noise_grid_pct=np.array(cons["noise_grid_pct"]) if "noise_grid_pct" in cons else np.array([]),
        noise_mode=str(cons["noise_mode"]) if "noise_mode" in cons else "dynamic",
        sw_noise_pct=float(cons["sw_noise_pct"]) if "sw_noise_pct" in cons else 10.0,
        vortex_noise_pct=float(cons["vortex_noise_pct"]) if "vortex_noise_pct" in cons else 10.0,
        sw_noise_white_frac=float(cons["sw_noise_white_frac"]) if "sw_noise_white_frac" in cons else 0.0,
        vortex_noise_white_frac=float(cons["vortex_noise_white_frac"]) if "vortex_noise_white_frac" in cons else 0.0,
        obs_noise_frac=float(cons["obs_noise_frac"]) if "obs_noise_frac" in cons else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Figure A1 — Outcome scatter map
# ══════════════════════════════════════════════════════════════════════════════

def fig_A1_outcome_scatter(d: dict) -> None:
    """
    figA1_outcome_scatter: scatter of B trap positions coloured by outcome,
    on SW trap lattice background, with inset pie chart. No fixed vortex
    home is shown — the vortex starts adjacent to B for every trial.
    """
    from matplotlib.lines import Line2D

    outcomes   = d["outcomes"]
    b_traps_mm = d["b_traps"] * 1e3         # mm — B trap centres
    traps_mm   = d["traps_m"] * 1e3         # mm
    cap_steps  = d["cap_steps"]
    n_trials   = d["n_trials"]

    max_total_steps = N_VORTEX_STEPS * N_DYN_STEPS
    S_MIN, S_MAX = 25, 220   # scatter marker size range (pt²)

    fig_w = _inches(FIG_WIDTH_MM)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.90))

    # ── SW trap lattice (grey +) ───────────────────────────────────────────────
    ax.scatter(
        traps_mm[:, 0], traps_mm[:, 1],
        marker="+", c=COL_TRAP_LAT, s=16, linewidths=0.6,
        zorder=1, alpha=0.5,
    )

    # ── Scatter circles (B trap positions, coloured by outcome) ───────────────
    for i, (outcome, xy, cs) in enumerate(zip(outcomes, b_traps_mm, cap_steps)):
        colour = COL_CAPTURED if outcome == "CAPTURED" else COL_TIMEOUT
        frac   = (cs + 1) / max_total_steps if (outcome == "CAPTURED" and cs >= 0) else 1.0
        size   = S_MIN + frac * (S_MAX - S_MIN)
        ax.scatter(
            xy[0], xy[1],
            s=size, c=colour,
            edgecolors="white", linewidths=0.4,
            zorder=4, alpha=0.85,
        )

    # ── Axis cosmetics ─────────────────────────────────────────────────────────
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Figure A1 — Outcome scatter by B trap position", pad=6)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _add_scale_bar(ax, 1.0, xlim, ylim)

    # ── Legend ──────────────────────────────────────────────────────────────────
    legend_elems = [
        mpatches.Patch(facecolor=COL_CAPTURED, label="CAPTURED"),
        mpatches.Patch(facecolor=COL_TIMEOUT,  label="TIMEOUT"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markersize=5,
               label="small = fast capture"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markersize=11,
               label="large = slow / timeout"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              fontsize=9, framealpha=0.85)

    # ── Inset pie chart (top-right) ────────────────────────────────────────────
    n_cap = int(np.sum(outcomes == "CAPTURED"))
    n_to  = n_trials - n_cap
    axins = ax.inset_axes([0.77, 0.77, 0.21, 0.21])
    axins.pie(
        [n_cap, n_to],
        colors=[COL_CAPTURED, COL_TIMEOUT],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    axins.set_title(f"N={n_trials}", fontsize=8, pad=2)

    fig.tight_layout()
    _save_fig(fig, "figA1_outcome_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# Figure A2 — Trajectory overlay
# ══════════════════════════════════════════════════════════════════════════════

def fig_A2_trajectories(d: dict) -> None:
    """
    figA2_trajectories: two-panel overlay of B trajectories coloured by
    positional noise σ. Panel (a) CAPTURED, panel (b) TIMEOUT.
    """
    outcomes  = d["outcomes"]
    traps_mm  = d["traps_m"] * 1e3    # mm
    noise_pct = d["noise_scales"] * 100   # noise scale percent
    trajs_B   = d["trajs_B"]             # list of (T,2) arrays in metres
    n_trials  = d["n_trials"]

    # Colormap bounds for noise level
    sig_min = float(noise_pct.min())
    sig_max = float(noise_pct.max())
    if sig_max <= sig_min:
        sig_max = sig_min + 1.0

    # ── Compute shared axis limits from all B trajectories ────────────────────
    all_x_mm = np.concatenate([t[:, 0] * 1e3 for t in trajs_B])
    all_y_mm = np.concatenate([t[:, 1] * 1e3 for t in trajs_B])
    pad = 0.35
    xlim = (float(all_x_mm.min()) - pad, float(all_x_mm.max()) + pad)
    ylim = (float(all_y_mm.min()) - pad, float(all_y_mm.max()) + pad)

    viridis = plt.cm.viridis
    sm = plt.cm.ScalarMappable(
        cmap=viridis,
        norm=plt.Normalize(vmin=sig_min, vmax=sig_max),
    )
    sm.set_array([])

    fig_w = _inches(FIG_WIDTH_MM)
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_w * 0.55), sharey=True)

    for ax_idx, (ax, panel_label, panel_outcome) in enumerate(zip(
        axes,
        ["(a)", "(b)"],
        ["CAPTURED", "TIMEOUT"],
    )):
        # Faint grey trap lattice
        ax.scatter(
            traps_mm[:, 0], traps_mm[:, 1],
            marker="+", c=COL_TRAP_LAT, s=10, linewidths=0.5,
            zorder=1, alpha=0.25,
        )

        n_panel = 0
        for i in range(n_trials):
            if outcomes[i] != panel_outcome:
                continue
            traj_mm = trajs_B[i] * 1e3   # (T, 2) mm
            colour  = viridis((noise_pct[i] - sig_min) / (sig_max - sig_min))
            ax.plot(
                traj_mm[:, 0], traj_mm[:, 1],
                color=colour, linewidth=1.0, alpha=0.45, zorder=3,
            )
            # mark starting point
            ax.scatter(traj_mm[0, 0], traj_mm[0, 1],
                       s=12, color=colour, alpha=0.7, zorder=4, linewidths=0)
            n_panel += 1

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [mm]")
        if ax_idx == 0:
            ax.set_ylabel("y [mm]")
        ax.set_title(f"{panel_label} {panel_outcome.title()} (N={n_panel})")
        _add_scale_bar(ax, 1.0, xlim, ylim)

    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.72, pad=0.02)
    cbar.set_label("Noise scale [%]")

    sw_pct  = d.get("sw_noise_pct", 0)
    vtx_pct = d.get("vortex_noise_pct", 0)
    fig.suptitle(f"Figure A2 — B trajectory overlay  "
                 f"(SW {sw_pct:.0f}%, vortex {vtx_pct:.0f}%)", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    _save_fig(fig, "figA2_trajectories")


# ══════════════════════════════════════════════════════════════════════════════
# Figure A3 — Statistics summary
# ══════════════════════════════════════════════════════════════════════════════

def fig_A3_statistics(d: dict) -> None:
    """
    figA3_statistics: histograms of capture time and MPC sequence length
    for CAPTURED trials, with mean ± std annotation.
    """
    outcomes  = d["outcomes"]
    cap_steps = d["cap_steps"]
    seq_lens  = d["seq_lens"]
    DT_MS     = d["DT_S"] * 1e3

    cap_mask = outcomes == "CAPTURED"
    cap_t_ms = (cap_steps[cap_mask].astype(float) + 1.0) * DT_MS
    cap_seq  = seq_lens[cap_mask].astype(float)
    n_cap    = int(cap_mask.sum())

    fig_w = _inches(FIG_WIDTH_MM)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.45))

    panel_configs = [
        ("a", cap_t_ms,  "Capture time [ms]",                    15),
        ("b", cap_seq,   "Sequence length (ψ transitions)", None),
    ]

    for ax, (char, data, xlabel, n_bins) in zip(axes, panel_configs):
        if n_cap == 0:
            ax.text(0.5, 0.5, "No CAPTURED trials",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"({char})")
            continue

        if n_bins is None:
            # Integer bins for sequence length
            lo_bin = int(data.min())
            hi_bin = int(data.max())
            n_bins = max(hi_bin - lo_bin + 1, 2)
            bin_edges = np.arange(lo_bin - 0.5, hi_bin + 1.5, 1.0)
        else:
            bin_edges = n_bins   # let matplotlib choose

        mu    = float(np.mean(data))
        sigma = float(np.std(data))

        ax.hist(
            data, bins=bin_edges,
            color="#3498db", edgecolor="black", linewidth=0.5,
        )
        ax.axvline(mu, color="black", linestyle="--", linewidth=1.2, zorder=5)

        ylim_hist = ax.get_ylim()
        x_text = mu + 0.03 * (float(data.max()) - float(data.min()))
        ax.text(
            x_text, 0.93 * ylim_hist[1],
            f"µ = {mu:.1f} ± {sigma:.1f}",
            fontsize=9, va="top",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"({char})  N = {n_cap} CAPTURED")

    fig.suptitle("Figure A3 — Statistics summary", y=1.02)
    fig.tight_layout()
    # Ensure x-labels are not clipped
    fig.subplots_adjust(bottom=0.18)
    _save_fig(fig, "figA3_statistics")


# ══════════════════════════════════════════════════════════════════════════════
# Figure A4 — Robustness by approach direction and positional noise
# ══════════════════════════════════════════════════════════════════════════════

def _fit_logistic_bootstrap(
    y: np.ndarray,
    x: np.ndarray,
    n_boot: int = 2000,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Fit logistic regression P(y=1) = sigmoid(a + b*x) via MLE, then
    bootstrap 2000 resamples for a pointwise 95% CI.

    Returns (x_plot, p_fit, ci_lo, ci_hi, p_value_slope) where:
      x_plot  : 200-point grid spanning x range
      p_fit   : fitted probability on x_plot
      ci_lo   : 2.5th percentile bootstrap curve
      ci_hi   : 97.5th percentile bootstrap curve
      p_value : two-sided Wald p-value for the slope coefficient
    """
    from scipy.optimize import minimize
    from scipy.special import expit  # sigmoid

    def neg_log_lik(params: np.ndarray, y_: np.ndarray, x_: np.ndarray) -> float:
        a, b = params
        p = expit(a + b * x_)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -float(np.sum(y_ * np.log(p) + (1 - y_) * np.log(1 - p)))

    def hessian_diag(params: np.ndarray, x_: np.ndarray) -> np.ndarray:
        """Diagonal of the observed Fisher information (for Wald SE)."""
        from scipy.special import expit
        a, b = params
        mu = expit(a + b * x_)
        w  = mu * (1 - mu)
        H00 = np.sum(w)
        H11 = np.sum(w * x_ ** 2)
        H01 = np.sum(w * x_)
        det = H00 * H11 - H01 ** 2
        if abs(det) < 1e-30:
            return np.array([np.inf, np.inf])
        return np.array([H11 / det, H00 / det])   # diag of inv(H)

    x0 = np.array([0.0, 0.0])
    res = minimize(neg_log_lik, x0, args=(y, x), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000})
    a_hat, b_hat = res.x

    # Wald p-value for slope
    var_diag = hessian_diag(res.x, x)
    se_b = float(np.sqrt(max(var_diag[1], 0.0)))
    z_stat = b_hat / se_b if se_b > 0 else 0.0
    from scipy.stats import norm as _norm
    p_value = float(2 * _norm.sf(abs(z_stat)))

    x_plot = np.linspace(x.min(), x.max(), 200)
    p_fit  = expit(a_hat + b_hat * x_plot)

    # Bootstrap CI
    rng_b  = np.random.default_rng(rng_seed)
    boot_curves = np.empty((n_boot, 200))
    n = len(y)
    for k in range(n_boot):
        idx = rng_b.integers(0, n, size=n)
        res_b = minimize(neg_log_lik, res.x, args=(y[idx], x[idx]),
                         method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
        a_b, b_b = res_b.x
        boot_curves[k] = expit(a_b + b_b * x_plot)

    ci_lo = np.percentile(boot_curves, 2.5,  axis=0)
    ci_hi = np.percentile(boot_curves, 97.5, axis=0)
    return x_plot, p_fit, ci_lo, ci_hi, p_value


def fig_A4_direction_and_noise(d: dict) -> None:
    """
    figA4_direction_and_noise: 2-panel figure.
    Panel (a): success rate vs 8 approach directions — bar chart, Wilson CI.
    Panel (b): binary outcome scatter vs noise σ with logistic regression fit
               and 2000-resample bootstrap 95% CI band.  The logistic fit
               correctly controls for the confound that approach direction
               varies independently of noise across trials.
    """
    from scipy.special import expit

    outcomes      = d["outcomes"]
    approach_dirs = np.array([str(x) for x in d["approach_dirs"]])
    noise_pct     = d["noise_scales"] * 100   # noise scale percent
    y_bin         = (outcomes == "CAPTURED").astype(float)

    fig_w = _inches(FIG_WIDTH_MM)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.52))

    # ── Panel (a): success rate vs approach direction ─────────────────────────
    ax_a = axes[0]
    dir_rates, dir_elos, dir_ehis, dir_ns = [], [], [], []
    for label in DIR_ORDER:
        mask = approach_dirs == label
        n    = int(mask.sum())
        k    = int(np.sum(outcomes[mask] == "CAPTURED")) if n > 0 else 0
        rate = 100.0 * k / n if n > 0 else 0.0
        ci_lo, ci_hi = _wilson_ci(k, n)
        dir_rates.append(rate)
        dir_elos.append(rate - ci_lo)
        dir_ehis.append(ci_hi - rate)
        dir_ns.append(n)

    x = np.arange(len(DIR_ORDER))
    ax_a.bar(
        x, dir_rates,
        yerr=np.vstack([dir_elos, dir_ehis]),
        color="#3498db", edgecolor="black", linewidth=0.5,
        capsize=3, error_kw={"elinewidth": 0.8},
        align="center", zorder=2,
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(DIR_ORDER, fontsize=9)
    ax_a.set_ylim(0, 115)
    ax_a.set_xlabel("Approach direction")
    ax_a.set_ylabel("Success rate [%]")
    ax_a.set_title("(a) Success rate vs approach direction")
    for xi, n_i in zip(x, dir_ns):
        ax_a.text(xi, 3, f"n={n_i}", ha="center", va="bottom",
                  fontsize=7, rotation=90, color="white")

    # ── Panel (b): σ-binned success rate + logistic overlay ───────────────────
    print("    Fitting logistic regression + bootstrap CI (n=2000) …", flush=True)
    ax_b = axes[1]

    noise_max_pct = float(noise_pct.max()) if len(noise_pct) > 0 else 100.0
    noise_grid    = d.get("noise_grid_pct", np.array([]))

    if len(noise_grid) > 0:
        # sigma_grid was used: bin by exact grid levels
        centres  = np.array(noise_grid, dtype=float)
        n_bins   = len(centres)
        half_gap = np.min(np.diff(centres)) / 2.0 if n_bins > 1 else 5.0
        widths   = np.full(n_bins, half_gap * 1.70)
        rates, elos, ehis, ns = [], [], [], []
        for nv in centres:
            mask = np.abs(noise_pct - nv) < 0.5
            n    = int(mask.sum())
            k    = int(np.sum(outcomes[mask] == "CAPTURED")) if n > 0 else 0
            rate = 100.0 * k / n if n > 0 else 0.0
            ci_lo_w, ci_hi_w = _wilson_ci(k, n)
            rates.append(rate)
            elos.append(rate - ci_lo_w)
            ehis.append(ci_hi_w - rate)
            ns.append(n)
    else:
        # Continuous noise: use 8 uniform bins
        n_bins   = 8
        edges    = np.linspace(0.0, noise_max_pct, n_bins + 1)
        centres  = 0.5 * (edges[:-1] + edges[1:])
        widths   = np.diff(edges) * 0.85
        rates, elos, ehis, ns = [], [], [], []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if i == n_bins - 1:
                mask = (noise_pct >= lo) & (noise_pct <= hi)
            else:
                mask = (noise_pct >= lo) & (noise_pct <  hi)
            n    = int(mask.sum())
            k    = int(np.sum(outcomes[mask] == "CAPTURED")) if n > 0 else 0
            rate = 100.0 * k / n if n > 0 else 0.0
            ci_lo_w, ci_hi_w = _wilson_ci(k, n)
            rates.append(rate)
            elos.append(rate - ci_lo_w)
            ehis.append(ci_hi_w - rate)
            ns.append(n)

    # 1) Binned success-rate bars + Wilson CI
    ax_b.bar(
        centres, rates, width=widths,
        yerr=np.vstack([elos, ehis]),
        color="#3498db", edgecolor="black", linewidth=0.5,
        capsize=3, error_kw={"elinewidth": 0.8},
        align="center", zorder=2,
    )
    for xi, n_i in zip(centres, ns):
        ax_b.text(xi, 3, f"n={n_i}", ha="center", va="bottom",
                  fontsize=7, rotation=90, color="white")

    # 2) Logistic fit + 95% bootstrap CI band, rescaled to the % axis
    x_plot, p_fit, ci_lo, ci_hi, p_val = _fit_logistic_bootstrap(y_bin, noise_pct)
    ax_b.plot(x_plot, 100.0 * p_fit, color="#2c3e50", linewidth=1.6, zorder=5,
              label="Logistic fit")
    ax_b.fill_between(x_plot, 100.0 * ci_lo, 100.0 * ci_hi,
                      color="#2c3e50", alpha=0.15, zorder=4,
                      label="95% bootstrap CI")

    # 3) Raw-data rug — one tick per trial along the baseline
    rug_y_cap = -3.0    # CAPTURED rug row (green)
    rug_y_to  = -7.0    # TIMEOUT rug row  (orange)
    n_cap = int((y_bin == 1).sum())
    n_to  = int((y_bin == 0).sum())
    ax_b.scatter(
        noise_pct[y_bin == 1], np.full(n_cap, rug_y_cap),
        marker="|", s=18, color=COL_CAPTURED, alpha=0.75,
        linewidths=0.8, zorder=3,
    )
    ax_b.scatter(
        noise_pct[y_bin == 0], np.full(n_to, rug_y_to),
        marker="|", s=18, color=COL_TIMEOUT, alpha=0.75,
        linewidths=0.8, zorder=3,
    )

    # 4) p-value annotation
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax_b.text(0.97, 0.92, p_str, transform=ax_b.transAxes,
              ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8))

    bar_pad = float(max(widths)) / 2.0 + 2.0
    ax_b.set_xlim(-bar_pad, noise_max_pct + bar_pad)
    ax_b.set_ylim(-10, 115)                    # leave room for rug below 0
    ax_b.set_yticks([0, 25, 50, 75, 100])      # hide negative rug band
    ax_b.axhline(0, color="black", linewidth=0.5, zorder=1)
    sw_pct  = d.get("sw_noise_pct", 0)
    vtx_pct = d.get("vortex_noise_pct", 0)
    ax_b.set_xlabel("Noise scale [%]")
    ax_b.set_ylabel("Success rate [%]")
    ax_b.set_title("(b) Success rate vs noise scale")
    ax_b.legend(fontsize=7, loc="upper left", framealpha=0.85)

    fig.suptitle(f"Figure A4 — Robustness by direction and noise  "
                 f"(SW {sw_pct:.0f}%, vortex {vtx_pct:.0f}%)", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figA4_direction_and_noise")


# ══════════════════════════════════════════════════════════════════════════════
# Figure entry point
# ══════════════════════════════════════════════════════════════════════════════

def fig_B2_stratified_noise(d: dict) -> None:
    """
    figB2_stratified_noise: direction-stratified noise robustness figure.

    Produced only when --direction and --sigma_grid were both specified.
    Shows:
      Panel (a): success rate at each σ level with Wilson 95% CI (bar chart).
                 Each bar is labelled with n_trials.
      Panel (b): individual outcomes (jittered binary strip) + logistic
                 regression fit + 2000-sample bootstrap 95% CI.
                 p-value annotated so statistical significance is clear.

    This cleanly isolates the noise effect by holding approach direction fixed.
    """
    from scipy.special import expit

    outcomes      = d["outcomes"]
    noise_pct     = d["noise_scales"] * 100
    noise_levels  = d["noise_grid_pct"]          # the grid values used (in %)
    fixed_dir     = d.get("fixed_direction", "")
    y_bin         = (outcomes == "CAPTURED").astype(float)

    fig_w = _inches(FIG_WIDTH_MM)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.52))

    dir_label = f" (direction = {fixed_dir})" if fixed_dir else ""
    fig.suptitle(
        f"Figure B2 — Noise robustness{dir_label}", y=1.02)

    # ── Panel (a): success rate per σ level ───────────────────────────────────
    ax_a = axes[0]
    rates, elos, ehis, ns = [], [], [], []
    for nv in noise_levels:
        # Match by rounded value to handle floating-point noise
        mask = np.abs(noise_pct - nv) < 0.05   # within 0.05% of the grid point
        n    = int(mask.sum())
        k    = int(np.sum(outcomes[mask] == "CAPTURED")) if n > 0 else 0
        rate = 100.0 * k / n if n > 0 else 0.0
        ci_lo, ci_hi = _wilson_ci(k, n)
        rates.append(rate)
        elos.append(rate - ci_lo)
        ehis.append(ci_hi - rate)
        ns.append(n)

    x = np.arange(len(noise_levels))
    ax_a.bar(
        x, rates,
        yerr=np.vstack([elos, ehis]),
        color="#27ae60", edgecolor="black", linewidth=0.5,
        capsize=3, error_kw={"elinewidth": 0.8},
        align="center", zorder=2,
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{nv:.0f}" for nv in noise_levels], fontsize=9)
    ax_a.set_ylim(0, 115)
    ax_a.set_xlabel("Noise scale [%]")
    ax_a.set_ylabel("Success rate [%]")
    ax_a.set_title(f"(a) Success rate vs noise scale{dir_label}")
    for xi, n_i in zip(x, ns):
        ax_a.text(xi, 3, f"n={n_i}", ha="center", va="bottom",
                  fontsize=8, rotation=90 if len(noise_levels) > 5 else 0,
                  color="white")

    # ── Panel (b): logistic regression scatter ────────────────────────────────
    print("    Fitting logistic regression + bootstrap CI (n=2000) …", flush=True)
    ax_b = axes[1]

    x_plot, p_fit, ci_lo, ci_hi, p_val = _fit_logistic_bootstrap(y_bin, noise_pct)

    rng_j = np.random.default_rng(42)
    jitter = rng_j.uniform(-0.03, 0.03, size=len(noise_pct))
    ax_b.scatter(
        noise_pct[y_bin == 1], jitter[y_bin == 1] + 1.0,
        s=14, color=COL_CAPTURED, alpha=0.6, linewidths=0, zorder=3,
        label="CAPTURED",
    )
    ax_b.scatter(
        noise_pct[y_bin == 0], jitter[y_bin == 0] + 0.0,
        s=14, color=COL_TIMEOUT, alpha=0.6, linewidths=0, zorder=3,
        label="TIMEOUT",
    )
    ax_b.plot(x_plot, p_fit, color="#1a5276", linewidth=1.8, zorder=5,
              label="Logistic fit")
    ax_b.fill_between(x_plot, ci_lo, ci_hi,
                      color="#1a5276", alpha=0.18, zorder=4,
                      label="95% bootstrap CI")

    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    sig_str = "  ✱ significant" if p_val < 0.05 else "  (not significant)"
    ax_b.text(0.97, 0.50, p_str + sig_str, transform=ax_b.transAxes,
              ha="right", va="center", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8))

    ax_b.set_xlim(-0.5, max(noise_levels) + 1)
    ax_b.set_ylim(-0.12, 1.12)
    ax_b.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_yticklabels(["0", "25", "50", "75", "100"])
    ax_b.set_xlabel("Noise scale [%]")
    ax_b.set_ylabel("P(CAPTURED) [%]")
    ax_b.set_title(f"(b) Logistic regression{dir_label}")
    ax_b.legend(fontsize=8, loc="center left", framealpha=0.85)

    fig.tight_layout()
    stem = f"figB2_stratified_noise{'_' + fixed_dir if fixed_dir else ''}"
    _save_fig(fig, stem)


# ══════════════════════════════════════════════════════════════════════════════
# Figure A5 — MPC vs random-ψ comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig_A5_comparison(d_MPC: dict, d_baseline: dict) -> None:
    """
    figA5: MPC controller vs random-ψ baseline.
    Panel (a): grouped bar chart by approach direction (Wilson 95% CI).
    Panel (b): overall success rate with two-proportion z-test p-value.
    """
    from scipy.stats import norm as _norm

    fig_w = _inches(FIG_WIDTH_MM)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.52),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── Panel (a): grouped bars by direction ──────────────────────────────────
    ax_a = axes[0]
    bar_w = 0.35
    x = np.arange(len(DIR_ORDER))

    for di, (d, label, colour, offset) in enumerate([
        (d_MPC,   "MPC",   "#3498db", -bar_w / 2),
        (d_baseline, "Random ψ", "#95a5a6",  bar_w / 2),
    ]):
        dirs_arr = np.array([str(x) for x in d["approach_dirs"]])
        outcomes = d["outcomes"]
        rates, elos, ehis = [], [], []
        for comp in DIR_ORDER:
            mask = dirs_arr == comp
            n = int(mask.sum())
            k = int(np.sum(outcomes[mask] == "CAPTURED")) if n > 0 else 0
            rate = 100.0 * k / n if n > 0 else 0.0
            ci_lo, ci_hi = _wilson_ci(k, n)
            rates.append(rate)
            elos.append(rate - ci_lo)
            ehis.append(ci_hi - rate)

        ax_a.bar(
            x + offset, rates,
            width=bar_w,
            yerr=np.vstack([elos, ehis]),
            color=colour, edgecolor="black", linewidth=0.5,
            capsize=2, error_kw={"elinewidth": 0.7},
            label=label, zorder=2,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(DIR_ORDER, fontsize=9)
    ax_a.set_ylim(0, 115)
    ax_a.set_xlabel("Approach direction")
    ax_a.set_ylabel("Success rate [%]")
    ax_a.set_title("(a) Success rate by direction")
    ax_a.legend(fontsize=9, loc="upper right", framealpha=0.85)

    # ── Panel (b): overall comparison with z-test ─────────────────────────────
    ax_b = axes[1]
    datasets = [
        (d_MPC,   "MPC",   "#3498db"),
        (d_baseline, "Random ψ", "#95a5a6"),
    ]
    x_bar = np.arange(len(datasets))
    rates_overall, elos_overall, ehis_overall = [], [], []
    n_caps, n_tots = [], []

    for d, label, colour in datasets:
        outcomes = d["outcomes"]
        n = d["n_trials"]
        k = int(np.sum(outcomes == "CAPTURED"))
        rate = 100.0 * k / n
        ci_lo, ci_hi = _wilson_ci(k, n)
        rates_overall.append(rate)
        elos_overall.append(rate - ci_lo)
        ehis_overall.append(ci_hi - rate)
        n_caps.append(k)
        n_tots.append(n)

    colours = [c for _, _, c in datasets]
    labels  = [l for _, l, _ in datasets]
    ax_b.bar(
        x_bar, rates_overall,
        yerr=np.vstack([elos_overall, ehis_overall]),
        color=colours, edgecolor="black", linewidth=0.5,
        capsize=3, error_kw={"elinewidth": 0.8},
        zorder=2,
    )
    ax_b.set_xticks(x_bar)
    ax_b.set_xticklabels(labels, fontsize=9)
    ax_b.set_ylim(0, 115)
    ax_b.set_ylabel("Success rate [%]")
    ax_b.set_title("(b) Overall")

    # Two-proportion z-test
    k1, n1 = n_caps[0], n_tots[0]
    k2, n2 = n_caps[1], n_tots[1]
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 else 1e-10
    z_stat = (k1 / n1 - k2 / n2) / se
    p_val = float(2 * _norm.sf(abs(z_stat)))
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax_b.text(0.5, 108, p_str, ha="center", va="bottom", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.8))

    fig.suptitle("Figure A5 — MPC vs random-ψ comparison", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figA5_MPC_vs_random_psi")


def generate_figures(out_dir: Path, baseline_dir: Optional[Path] = None) -> None:
    print("\n" + "=" * 72)
    print("Generating figures …")
    print(f"  Loading from : {out_dir}")
    d = _load_results(out_dir)

    n_cap = int(np.sum(d["outcomes"] == "CAPTURED"))
    n_tot = d["n_trials"]
    noise_pct = d["noise_scales"] * 100
    print(f"  {n_cap}/{n_tot} CAPTURED ({100.0 * n_cap / n_tot:.0f}%)  "
          f"— noise scale range {noise_pct.min():.1f} … {noise_pct.max():.1f}%")
    print(f"  Output       : {FIG_OUT_DIR}")

    print("\n  [figA1] Outcome scatter …")
    fig_A1_outcome_scatter(d)

    print("\n  [figA2] Trajectory overlay …")
    fig_A2_trajectories(d)

    print("\n  [figA3] Statistics summary …")
    fig_A3_statistics(d)

    print("\n  [figA4] Direction and noise robustness …")
    fig_A4_direction_and_noise(d)

    # figB2: stratified noise study (only when direction + noise_grid were fixed)
    if len(d.get("noise_grid_pct", [])) > 0:
        print("\n  [figB2] Stratified noise robustness …")
        fig_B2_stratified_noise(d)

    # figA5: MPC vs random-ψ comparison (only if baseline_dir provided)
    if baseline_dir is not None:
        print("\n  [figA5] MPC vs random-ψ comparison …")
        print(f"    Baseline dir: {baseline_dir}")
        d_base = _load_results(baseline_dir)
        fig_A5_comparison(d, d_base)

    print("\nAll figures saved to:", FIG_OUT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Study A: MPC vortex-merge robustness across "
                    "randomised initial conditions.")
    p.add_argument(
        "--n_trials", type=int, default=100,
        help="Number of randomised trials (default: 100)")
    p.add_argument(
        "--skip_simulation", action="store_true",
        help="Skip simulation; load results from --out_dir")
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Existing output directory for --skip_simulation")
    p.add_argument(
        "--no_figures", action="store_true",
        help="Simulation only: do not generate figures after runs finish")
    p.add_argument(
        "--direction", type=str, default=None,
        choices=["E", "NE", "N", "NW", "W", "SW", "S", "SE"],
        help="Fix vortex approach direction (compass label). "
             "When set, each trial's vortex start is chosen as the nearest "
             "neighbour of B that best matches this direction.")
    p.add_argument(
        "--sigma_grid", type=str, default=None,
        help="Comma-separated noise-scale values as percentages (0-100). "
             "Trials cycle through these values. Example: '0,20,40,60,80,100'. "
             "At scale=100%%, SW noise = sw_noise_pct%% of peak |p_sw|, "
             "vortex noise = vortex_noise_pct%% of peak |p_v|.")
    p.add_argument(
        "--sw_noise_pct", type=float, default=10.0,
        help="Max SW-field noise as %% of peak |p_sw| (at noise_scale=1). Default: 10.")
    p.add_argument(
        "--vortex_noise_pct", type=float, default=10.0,
        help="Max vortex-field noise as %% of peak |p_v| (at noise_scale=1). Default: 10.")
    p.add_argument(
        "--sw_noise_white_frac", type=float, default=0.0,
        help="Fraction [0-1] of SW noise energy that is spatially white. Default: 0.")
    p.add_argument(
        "--vortex_noise_white_frac", type=float, default=0.0,
        help="Fraction [0-1] of vortex noise energy that is spatially white. Default: 0.")
    p.add_argument(
        "--noise_mode", type=str, default="dynamic",
        choices=["static", "dynamic"],
        help="Noise regime: 'static' = one noise field per trial (frozen, models "
             "fabrication/alignment error); 'dynamic' = one noise field per "
             "control step (refreshed, models instability/drift). Default: dynamic.")
    p.add_argument(
        "--obs_noise_frac", type=float, default=1.0,
        help="Multiplier on OBS_NOISE_MAX for controller observation noise. "
             "At noise_scale=1, σ_obs = obs_noise_frac × OBS_NOISE_MAX. "
             "Set to 0 to disable observation noise. Default: 1.0.")
    p.add_argument(
        "--random_psi", action="store_true",
        help="Baseline mode: evaluate phase_sweep() as normal but discard the "
             "MPC result and substitute a uniform random ψ draw.")
    p.add_argument(
        "--baseline_dir", type=str, default=None,
        help="Path to a random-ψ result directory for figA5 comparison. "
             "Omit to skip figA5.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.skip_simulation:
        if not args.out_dir:
            raise ValueError("--skip_simulation requires --out_dir")
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = run_simulation(args)

    # Resolve baseline_dir for figA5
    baseline_dir: Optional[Path] = None
    if getattr(args, "baseline_dir", None):
        baseline_dir = Path(args.baseline_dir)
        if not baseline_dir.is_absolute():
            baseline_dir = PROJECT_ROOT / baseline_dir

    if not args.no_figures:
        generate_figures(out_dir, baseline_dir=baseline_dir)


if __name__ == "__main__":
    main()
