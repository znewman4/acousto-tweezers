"""
Canonical physics presets for the far-field petri cuboid model.

Every run script should import from here so there's ONE source of truth
for the corrected two-region model.
"""
from __future__ import annotations

# ── Corrected two-region model (Feb 2026) ─────────────────────────
#
# H_petri = 2 mm (fixed)
# Standing-wave BC *only* on petri slab side walls
# Top BC: physical water–air Robin (Z_air = ρ_air·c_air = 411.6 Pa·s/m)
# MUMPS direct solver
#
CORRECTED_PRESET: dict = dict(
    # geometry
    Lx=6e-3,
    Ly=6e-3,
    H_under=3e-3,          # water-bath depth (= H_bath)
    H_top=2.0085e-3,       # petri slab thickness — m=14 quarter-wave resonance (optimal)
    # frequency
    frequency_hz=2.0e6,
    # disk source
    disk_radius=1.0e-3,
    disk_velocity_amplitude=1e-6,       # 1 µm/s
    vortex_topological_charge=1,
    # standing wave
    standing_velocity_amplitude=10e-6,  # 10 µm/s
    standing_phase_pattern="antiphase",
    standing_axis="both",
    # PML
    pml_n_wavelengths_xy=1.0,
    pml_n_wavelengths_z=1.0,
    pml_degree=2,
    pml_sigma_max_factor=5.0,
    pml_enabled=True,
    # lens
    lens_drive="plastic",
    lens_l=1,
    lens_focal_length=2e-3,            # f=2mm gives tightest ring (0.90mm ≈ 1.2λ) at trap plane
    lens_focus_offset_x=0.2e-3,
    lens_focus_offset_y=0.0,
    lens_c_lens=2700.0,
    lens_apodization="cosine_taper",
    lens_apodization_strength=1.0,
)

PETSC_MUMPS: dict = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": "80",
    "mat_mumps_icntl_23": "0",
}
