
#!/usr/bin/env python3
"""
plot_l2_vs_epl.py
-----------------
Standalone script to generate a clean mesh-convergence plot of
relative L2 error in the ROI versus EPL.

Expected input:
    results/mesh_convergence_study/convergence_analysis.csv
or
    mesh_convergence_study/convergence_analysis.csv

Output:
    fig_l2_vs_epl.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REFERENCE_EPL = 5.0

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
})

COLOR_MAIN = "#185FA5"
COLOR_REF  = "#777777"


def _candidate_csv_paths(base_dir: Path) -> list[Path]:
    """Return likely convergence CSV paths ordered from most to least specific."""
    direct_candidates = [
        base_dir / "convergence_analysis.csv",
        base_dir / "convergence_summary.csv",
        base_dir / "analysis_v2" / "convergence_analysis_v2.csv",
    ]

    analysis_candidates = []
    for pattern in ("analysis_*", "analysis_v*"):
        analysis_candidates.extend(
            sorted(base_dir.glob(f"{pattern}/convergence_analysis*.csv"), reverse=True)
        )

    # Keep order stable while removing duplicates.
    seen = set()
    ordered = []
    for path in [*direct_candidates, *analysis_candidates]:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered

# -----------------------------------------------------------------------------
# Locate CSV
# -----------------------------------------------------------------------------

here = Path.cwd()
project_root = Path(__file__).resolve().parents[2]

study_dirs = [
    here / "results" / "mesh_convergence_study",
    here / "mesh_convergence_study",
    project_root / "results" / "mesh_convergence_study",
    project_root / "mesh_convergence_study",
    Path(__file__).resolve().parent / "results" / "mesh_convergence_study",
    Path(__file__).resolve().parent / "mesh_convergence_study",
]

candidates = []
seen = set()
for study_dir in study_dirs:
    for candidate in _candidate_csv_paths(study_dir):
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

csv_path = next((p for p in candidates if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError(
        "Could not find convergence_analysis.csv.\n"
        "Looked in:\n" + "\n".join(str(p) for p in candidates)
    )

print(f"[info] Using CSV: {csv_path}")

# -----------------------------------------------------------------------------
# Load and normalise columns
# -----------------------------------------------------------------------------

df = pd.read_csv(csv_path)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

rename_map = {
    "requested_epl": "epl",
    "eps_l2_roi": "eps_L2_roi",
    "physical_size_mm": "physical_size_mm",
    "pml_n_wavelengths_xy": "pml_n_wavelengths_xy",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

required = ["epl", "eps_L2_roi"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")

for col in ["epl", "eps_L2_roi", "physical_size_mm", "pml_n_wavelengths_xy"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Optional filtering if those columns exist
if "physical_size_mm" in df.columns:
    df = df[df["physical_size_mm"].round(1) == 3.0]
if "pml_n_wavelengths_xy" in df.columns:
    df = df[df["pml_n_wavelengths_xy"].round(1) == 1.0]

df = df[df["epl"] < REFERENCE_EPL]
df = df[df["eps_L2_roi"].notna() & np.isfinite(df["eps_L2_roi"])]
df = df.sort_values("epl")

if df.empty:
    raise ValueError("No valid rows remained after filtering.")

epl = df["epl"].values
eps = df["eps_L2_roi"].values

print("[info] EPL values used:", epl)
print("[info] L2 errors used:", eps)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.2, 4.8))

ax.semilogy(
    epl,
    eps,
    "o-",
    color=COLOR_MAIN,
    ms=7,
    lw=1.8,
    label=r"Computed $\varepsilon_{\mathrm{ROI}}$"
)

# Expected asymptotic trend: h^2 and h = lambda/EPL  => error ~ EPL^-2
epl_ref = np.linspace(min(epl) * 0.9, max(epl) * 1.05, 200)
ref_curve = eps[-1] * (epl[-1] / epl_ref) ** 2

ax.semilogy(
    epl_ref,
    ref_curve,
    "--",
    color=COLOR_REF,
    lw=1.2,
    label=r"Expected asymptotic trend, $\mathcal{O}(\mathrm{EPL}^{-2})$"
)

offsets = {
    2.0: (6, 6),
    3.0: (6, -14),
    3.5: (6, 6),
    4.0: (-24, 6),
    4.5: (6, -14),
}
for x, y in zip(epl, eps):
    dx, dy = offsets.get(float(x), (5, 5))
    ax.annotate(
        f"{x:.1f}",
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel("EPL (elements per wavelength)")
ax.set_ylabel(r"Relative $L_2$ error in ROI, $\varepsilon_{\mathrm{ROI}}$")
ax.set_title("Mesh convergence — ROI pressure-field error vs EPL")
ax.legend(loc="best", framealpha=0.95)
ax.grid(True, which="both", alpha=0.3)

out_path = csv_path.parent / "fig_l2_vs_epl.png"
fig.savefig(out_path)
plt.close(fig)

print(f"[done] Saved: {out_path}")