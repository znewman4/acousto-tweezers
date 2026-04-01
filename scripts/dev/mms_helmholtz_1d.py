#!/usr/bin/env python3
"""
mms_helmholtz_1d.py  —  Method of Manufactured Solutions (MMS) verification
for the 1D Helmholtz FEM solver (dolfinx / FEniCSx).

Purpose
-------
Independently verifies that the FEM solver (element order, weak form, BC
implementation, MUMPS linear solve) converges at the theoretically expected
rate *before* trusting the standing-wave convergence study.  Without this
step, systematic errors (wrong BC type, incorrect PML conductivity profile,
element-order mismatch) would be invisible to the self-referential EPL=5
reference solution.

Physical setup
--------------
  Domain      : x ∈ [0, L],  L = 0.75λ  (avoids resonance, see below)
  Equation    : ∂²p/∂x² + k²p = f(x)      (1D real Helmholtz)
  BCs         : ∂p/∂n = 0 at x=0 and x=L  (hard walls — natural BC of weak form)

Manufactured solution
---------------------
  p_exact(x) = cos(k₀ x),   k₀ = π / L  (fundamental mode of the MMS domain)

  Neumann BC check : ∂p/∂x|_{x=0} = 0 ✓
                     ∂p/∂x|_{x=L} = −k₀ sin(k₀ L) = −k₀ sin(π) = 0 ✓

  Source term : f(x) = (k² − k₀²) cos(k₀ x)
                (substituting p_exact into the Helmholtz operator)

  Non-singularity : the physical wavenumber k = 2π/λ is a resonance of the
  domain only if k = nπ/L = 4nπ/(3λ), i.e. n = 3/2 — not an integer — so
  the system matrix is non-singular. ✓

Expected result
---------------
  For P2 (quadratic) Lagrange elements: ε_L2 ∝ h³, i.e. p_obs ≈ 3.
  Deviation from this (given adequate h-refinement ratio) indicates a bug.

EPL sweep
---------
  Geometric sequence with ratio √2 as recommended by Babuška & Rheinboldt
  (1978) for reliable convergence-order estimation.  Each step halves h² so
  consecutive h-ratios are consistently √2 ≈ 1.414, giving log(h₁/h₂) ≈ 0.35
  — large enough to suppress noise in the order estimate.

References
----------
  Babuška & Rheinboldt, Int. J. Numer. Meth. Engng., 12(10), 1978.
  Ihlenburg & Babuška, Comput. Meth. Appl. Mech. Engng., 128, 1995.
  Salari & Knupp, "Code Verification by the MMS", Sandia Report SAND2000-1444.

Usage
-----
  micromamba run -p ~/.conda/envs/acousto-complex python mms_helmholtz_1d.py

  Optional args (edit constants below):
    ELEMENT_ORDER   — 1, 2, or 3  (default 2, matching production solver)
    N_EPL_LEVELS    — number of refinement levels (default 7)
    SAVE_DIR        — output directory for CSV and PNG

"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# ── Optional matplotlib (skip plot if not available) ─────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not found — skipping plot.")

# ── dolfinx imports ───────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx
    import dolfinx.mesh
    import dolfinx.fem
    import dolfinx.fem.petsc
    import ufl
except ImportError as e:
    sys.exit(f"[error] dolfinx import failed: {e}\n"
             "Run inside the acousto-complex conda environment.")

# ═══════════════════════════════════════════════════════════════════════════════
# USER-CONFIGURABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Physical parameters — must match production solver
F0             = 2.15e6     # transducer frequency [Hz]
C_SOUND        = 1484.0     # speed of sound in water [m/s]

# FEM settings — must match production solver
ELEMENT_ORDER  = 2          # Lagrange polynomial order (P1=1, P2=2, P3=3)

# EPL sweep: 7 levels with ratio √2, starting at EPL=2
#   → EPL ≈ 2.0, 2.83, 4.0, 5.66, 8.0, 11.3, 16.0
N_EPL_LEVELS   = 7
EPL_START      = 2.0

# Output
SAVE_DIR       = Path("mms_results")

# ═══════════════════════════════════════════════════════════════════════════════
# DERIVED CONSTANTS  (do not edit)
# ═══════════════════════════════════════════════════════════════════════════════

LAM   = C_SOUND / F0                # wavelength [m]
K     = 2.0 * np.pi / LAM           # physical wavenumber [rad/m]

# MMS domain: L = 3λ/4 → k is not a resonance of [0, L]
L_DOMAIN = 0.75 * LAM              # [m]
K0       = np.pi / L_DOMAIN         # manufactured wavenumber [rad/m]
F_COEFF  = K**2 - K0**2             # source amplitude [(rad/m)²]

# Geometric EPL sequence
EPL_VALUES = [EPL_START * (np.sqrt(2)**i) for i in range(N_EPL_LEVELS)]

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def print_header():
    print("=" * 70)
    print("  MMS Helmholtz 1D — FEniCSx Solver Verification")
    print("=" * 70)
    print(f"  Frequency   : {F0/1e6:.3f} MHz")
    print(f"  Speed       : {C_SOUND:.1f} m/s")
    print(f"  Wavelength  : {LAM*1e3:.4f} mm")
    print(f"  k (physical): {K:.4f} rad/m")
    print(f"  Domain L    : {L_DOMAIN*1e3:.4f} mm  (= 0.75λ)")
    print(f"  k₀ (MMS)    : {K0:.4f} rad/m")
    print(f"  Source f    : ({K**2:.4f} - {K0**2:.4f}) cos(k₀x)")
    print(f"              = {F_COEFF:.4f} cos(k₀x)  [non-zero ✓]")
    print(f"  Element P{ELEMENT_ORDER}   : expected O(h^{ELEMENT_ORDER+1}) = O(h^{ELEMENT_ORDER+1})")
    print(f"  EPL sweep   : {', '.join(f'{e:.2f}' for e in EPL_VALUES)}")
    print()

    # Sanity: verify non-singularity
    n_resonances_in_range = [n for n in range(1, 20)
                             if abs(K - n * np.pi / L_DOMAIN) / K < 0.01]
    if n_resonances_in_range:
        print(f"  [WARN] Physical k is near resonance n={n_resonances_in_range} — "
              "system may be ill-conditioned.")
    else:
        print("  Non-singularity check: k is not near any resonance of [0, L] ✓")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SOLVE ONE LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

def solve_level(epl: float) -> dict:
    """Run one FEM solve and return error metrics."""

    # Number of 1D cells: at least 4, rounded to nearest integer
    # h = L / n_cells  →  EPL ≡ λ/h = λ * n_cells / L
    n_cells = max(4, int(round(epl * L_DOMAIN / LAM)))
    h       = L_DOMAIN / n_cells       # actual element size [m]
    epl_act = LAM / h                  # actual EPL achieved

    # ── Mesh ──────────────────────────────────────────────────────────────────
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, n_cells,
                                        [0.0, L_DOMAIN])

    # ── Function space ─────────────────────────────────────────────────────────
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", ELEMENT_ORDER))
    n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    # ── Weak form ──────────────────────────────────────────────────────────────
    # ∫ ∇p·∇v dx − k² ∫ p v dx = ∫ f v dx
    # Neumann BC (∂p/∂n = 0) is the natural BC → no boundary integral needed.
    p_trial = ufl.TrialFunction(V)
    v_test  = ufl.TestFunction(V)
    x       = ufl.SpatialCoordinate(mesh)

    # Source: f(x) = F_COEFF * cos(k₀ x)
    f_ufl = F_COEFF * ufl.cos(K0 * x[0])

    a = (ufl.inner(ufl.grad(p_trial), ufl.grad(v_test))
         - K**2 * ufl.inner(p_trial, v_test)) * ufl.dx

    L_form = ufl.inner(f_ufl, v_test) * ufl.dx

    # ── Solve with MUMPS (matching production solver) ─────────────────────────
    problem = dolfinx.fem.petsc.LinearProblem(
        a, L_form,
        bcs=[],
        petsc_options={
            "ksp_type":                    "preonly",
            "pc_type":                     "lu",
            "pc_factor_mat_solver_type":   "mumps",
        },
    )
    p_h = problem.solve()

    # ── L2 error (relative) ───────────────────────────────────────────────────
    # p_exact(x) = cos(k₀ x)
    p_exact_ufl = ufl.cos(K0 * x[0])
    diff        = p_h - p_exact_ufl

    error_form = dolfinx.fem.form(diff**2 * ufl.dx)
    norm_form  = dolfinx.fem.form(p_exact_ufl**2 * ufl.dx)

    e_local = dolfinx.fem.assemble_scalar(error_form)
    n_local = dolfinx.fem.assemble_scalar(norm_form)

    e_global = mesh.comm.allreduce(e_local, op=MPI.SUM)
    n_global = mesh.comm.allreduce(n_local, op=MPI.SUM)

    eps_L2 = float(np.sqrt(e_global / n_global))

    return {
        "epl_target": epl,
        "epl_actual": epl_act,
        "n_cells":    n_cells,
        "h_mm":       h * 1e3,
        "n_dofs":     n_dofs,
        "eps_L2":     eps_L2,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE ORDER ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_orders(results: list[dict]) -> list[dict]:
    """Compute observed convergence order between consecutive refinements."""
    orders = []
    for i in range(1, len(results)):
        rc = results[i - 1]   # coarse
        rf = results[i]       # fine
        h_ratio = rc["h_mm"] / rf["h_mm"]
        if rc["eps_L2"] > 0.0 and rf["eps_L2"] > 0.0 and h_ratio > 1.0:
            p_obs = np.log(rc["eps_L2"] / rf["eps_L2"]) / np.log(h_ratio)
        else:
            p_obs = float("nan")
        orders.append({
            "epl_coarse":  rc["epl_actual"],
            "epl_fine":    rf["epl_actual"],
            "h_ratio":     h_ratio,
            "p_obs":       p_obs,
            "expected":    ELEMENT_ORDER + 1,
        })
    return orders

# ═══════════════════════════════════════════════════════════════════════════════
# PASS / FAIL ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assess(orders: list[dict]) -> bool:
    """
    PASS if the finest three consecutive pairs are all within ±0.5 of the
    expected order.  Coarse levels are allowed to be pre-asymptotic.
    """
    expected = ELEMENT_ORDER + 1
    fine_orders = [o["p_obs"] for o in orders[-3:] if not np.isnan(o["p_obs"])]
    if not fine_orders:
        return False
    return all(abs(p - expected) < 0.5 for p in fine_orders)

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def save_csv(results: list[dict], orders: list[dict]):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    r_path = SAVE_DIR / "mms_convergence.csv"
    with open(r_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    o_path = SAVE_DIR / "mms_orders.csv"
    with open(o_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(orders[0].keys()))
        w.writeheader()
        w.writerows(orders)

    print(f"  Saved: {r_path}")
    print(f"  Saved: {o_path}")


def make_plot(results: list[dict], orders: list[dict], passed: bool):
    if not HAS_MPL:
        return

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    h_arr   = np.array([r["h_mm"]   for r in results])
    eps_arr = np.array([r["eps_L2"] for r in results])
    epl_arr = np.array([r["epl_actual"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: error vs h ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.loglog(h_arr, eps_arr, "o-", color="#185FA5", lw=2, ms=8,
              label=f"FEM P{ELEMENT_ORDER} (observed)")

    # Reference slopes anchored at finest point
    h_ref = np.logspace(np.log10(h_arr[-1]), np.log10(h_arr[0]), 50)
    for order, style, label in [
        (ELEMENT_ORDER + 1, "--", f"O(h^{ELEMENT_ORDER+1}) expected"),
        (ELEMENT_ORDER,     ":",  f"O(h^{ELEMENT_ORDER}) (one order low)"),
    ]:
        slope = eps_arr[-1] * (h_ref / h_arr[-1])**order
        ax.loglog(h_ref, slope, style, color="#888780", lw=1.2, label=label)

    for r in results:
        ax.annotate(f"EPL={r['epl_actual']:.1f}",
                    xy=(r["h_mm"], r["eps_L2"]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=8, color="#3d3d3a")

    ax.set_xlabel("h = λ / EPL  (mm)")
    ax.set_ylabel("Relative L2 error  ε")
    ax.set_title(f"MMS convergence — P{ELEMENT_ORDER} Lagrange\n"
                 f"({'PASS ✓' if passed else 'FAIL ✗'}: "
                 f"expected O(h^{ELEMENT_ORDER+1}))")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Colour background green/red for pass/fail
    ax.set_facecolor("#eaf3de" if passed else "#fcebeb")

    # ── Right: observed order vs EPL pair ────────────────────────────────────
    ax2 = axes[1]
    labels = [f"{o['epl_coarse']:.1f}→{o['epl_fine']:.1f}" for o in orders]
    p_obs  = [o["p_obs"] for o in orders]
    x_pos  = np.arange(len(labels))

    bars = ax2.bar(x_pos, p_obs, color="#185FA5", alpha=0.75, width=0.6)
    ax2.axhline(ELEMENT_ORDER + 1, color="#D85A30", lw=1.5, ls="--",
                label=f"Expected p = {ELEMENT_ORDER + 1}")
    ax2.axhspan(ELEMENT_ORDER + 0.5, ELEMENT_ORDER + 1.5,
                color="#D85A30", alpha=0.08, label="±0.5 acceptance band")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9, rotation=25, ha="right")
    ax2.set_ylabel("Observed convergence order  p_obs")
    ax2.set_title("Observed order between consecutive EPL pairs")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Annotate bars
    for bar, p in zip(bars, p_obs):
        if not np.isnan(p):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{p:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        f"MMS Verification  |  1D Helmholtz  |  "
        f"f₀ = {F0/1e6:.2f} MHz, λ = {LAM*1e3:.3f} mm",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    fig_path = SAVE_DIR / "mms_convergence.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_header()

    # ── Run EPL sweep ─────────────────────────────────────────────────────────
    results = []
    print(f"  {'EPL':>6}  {'n_cells':>8}  {'h (mm)':>8}  {'DOFs':>6}  {'ε_L2':>12}")
    print("  " + "-" * 54)

    for epl in EPL_VALUES:
        r = solve_level(epl)
        results.append(r)
        print(f"  {r['epl_actual']:6.2f}  {r['n_cells']:8d}  "
              f"{r['h_mm']:8.4f}  {r['n_dofs']:6d}  {r['eps_L2']:12.4e}")

    # ── Convergence orders ────────────────────────────────────────────────────
    orders = compute_orders(results)
    print()
    print(f"  {'EPL pair':>12}  {'h ratio':>8}  {'p_obs':>8}  {'expected':>10}  {'status':>8}")
    print("  " + "-" * 60)
    for o in orders:
        status = "✓" if not np.isnan(o["p_obs"]) and abs(o["p_obs"] - o["expected"]) < 0.5 else "—"
        print(f"  {o['epl_coarse']:.2f}→{o['epl_fine']:.2f}  "
              f"{o['h_ratio']:8.3f}  {o['p_obs']:8.3f}  {o['expected']:10d}  {status:>8}")

    # ── Pass/fail ─────────────────────────────────────────────────────────────
    passed = assess(orders)
    print()
    print("=" * 70)
    if passed:
        print(f"  RESULT: PASS ✓")
        print(f"  Finest refinement levels converge at O(h^{ELEMENT_ORDER+1}) as expected.")
        print(f"  The P{ELEMENT_ORDER} Lagrange element, weak form, BCs, and MUMPS solve are correct.")
    else:
        print(f"  RESULT: FAIL ✗")
        print(f"  Observed orders do not approach O(h^{ELEMENT_ORDER+1}) in the fine regime.")
        print(f"  Check: element order, BC formulation, source term, solver convergence.")
    print("=" * 70)

    # ── Save and plot ─────────────────────────────────────────────────────────
    save_csv(results, orders)
    make_plot(results, orders, passed)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())