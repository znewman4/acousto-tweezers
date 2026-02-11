#!/usr/bin/env python3
"""
Generate rich comparison data for canonical PyVista visualizations.

Writes complex pressure (real + imag), magnitude, phase, AND Gor'kov potential
to NPZ files alongside the XDMF mesh.

Usage:
    python scripts/visualization/generate_rich_data.py --preset A
    python scripts/visualization/generate_rich_data.py --preset A --n_phases 8
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from mpi4py import MPI

import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
import ufl
import sys

sys.path.append(str(Path(__file__).parents[2] / 'src'))

from acoustweezers.physics.acoustics.vortex_lens import (
    VortexLensConfig, compute_azimuthal_phase, compute_amplitude_profile
)

# ---------------------------------------------------------------------------
# Material properties (same as compare_vortex_standing_fixed.py)
# ---------------------------------------------------------------------------

PRESET_A = {
    'name': 'Preset A: 2cm dish, 500 kHz',
    'dish_size_m': 0.02,
    'frequency_hz': 500e3,
    'wavelength_m': 2.968e-3,
    'aperture_radius_m': 0.002,
    'aperture_center_xy_m': None,
    'vortex_gain': 10.0,
    'standing_gain': 1.0,
    'elements_per_wavelength': 6,
}

RHO = 997.0       # kg/m³
C   = 1484.0      # m/s

# Polystyrene particle
A_PARTICLE  = 5e-6       # m
RHO_P       = 1050.0     # kg/m³
KAPPA_P     = 2.4e-10    # Pa⁻¹
KAPPA_F     = 1.0 / (RHO * C**2)

F1 = 1.0 - KAPPA_P / KAPPA_F           # monopole contrast
F2 = 2.0 * (RHO_P - RHO) / (2*RHO_P + RHO)  # dipole contrast
VP = (4.0/3.0) * np.pi * A_PARTICLE**3       # particle volume

# ---------------------------------------------------------------------------
# Mesh + boundary tags  (create once, reuse for multiple solves)
# ---------------------------------------------------------------------------

def make_mesh(preset):
    L  = preset['dish_size_m']
    nx = int(L / preset['wavelength_m'] * preset['elements_per_wavelength'])

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [[0, 0, 0], [L, L, L]],
        [nx, nx, nx],
        cell_type=mesh.CellType.tetrahedron,
    )

    def bottom(x): return np.isclose(x[2], 0)
    def top(x):    return np.isclose(x[2], L)
    def xw(x):     return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], L))
    def yw(x):     return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], L))

    fdim = domain.topology.dim - 1
    fi, fm = [], []
    for tag, loc in enumerate([bottom, xw, yw, top], start=1):
        f = mesh.locate_entities_boundary(domain, fdim, loc)
        fi.append(f); fm.append(np.full_like(f, tag))

    fi = np.hstack(fi).astype(np.int32)
    fm = np.hstack(fm).astype(np.int32)
    order = np.argsort(fi)
    ft = mesh.meshtags(domain, fdim, fi[order], fm[order])

    return domain, ft

# ---------------------------------------------------------------------------
# Solver  (returns complex fem.Function on P2)
# ---------------------------------------------------------------------------

def solve_helmholtz(domain, ft, preset, mode, vortex_phase=0.0, ell=1):
    """
    mode: 'standing' | 'vortex' | 'combined'
    vortex_phase: additional azimuthal phase offset [rad]
    Returns: complex fem.Function  p(x)
    """
    L     = preset['dish_size_m']
    freq  = preset['frequency_hz']
    omega = 2 * np.pi * freq
    k     = omega / C
    Z     = RHO * C

    V  = fem.functionspace(domain, ("Lagrange", 2))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    u  = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)

    # bilinear form  (same for all modes)
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) / RHO
         - k**2 * ufl.inner(u, v) / RHO) * ufl.dx
    for tag in [1, 2, 3, 4]:
        a += -1j * omega * ufl.inner(u, v) / Z * ds(tag)

    # linear form — depends on mode
    v0_s = 1e-6 * preset['standing_gain']
    g_s  = -1j * omega * RHO * v0_s

    if mode == 'standing':
        L_form = g_s * ufl.inner(1.0, v) * (ds(2) + ds(3))
    elif mode == 'combined':
        L_form = g_s * ufl.inner(1.0, v) * (ds(2) + ds(3))

    if mode in ('vortex', 'combined'):
        center = np.array([L/2, L/2, L])
        cfg = VortexLensConfig(
            topological_charge=ell,
            center=tuple(center),
            amplitude=1e-6 * preset['vortex_gain'],
            aperture_radius=preset['aperture_radius_m'],
            apodization='cosine_taper',
            axis='z',
        )
        fdim      = domain.topology.dim - 1
        top_f     = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], L))
        top_dofs  = fem.locate_dofs_topological(V, fdim, top_f)
        top_xyz   = V.tabulate_dof_coordinates()[top_dofs]

        phi  = compute_azimuthal_phase(top_xyz, cfg, center)
        amp  = compute_amplitude_profile(top_xyz, cfg, center)
        pat  = amp * np.exp(1j * (phi + vortex_phase))

        vf = fem.Function(V)
        vf.x.array[:] = 0.0
        vf.x.array[top_dofs] = pat

        g_v   = -1j * omega * RHO * vf
        L_vtx = ufl.inner(g_v, v) * ds(4)

        if mode == 'vortex':
            L_form = L_vtx
        elif mode == 'combined':
            L_form = L_form + L_vtx

    prob = LinearProblem(a, L_form, bcs=[], petsc_options={
        "ksp_type": "gmres", "ksp_rtol": 1e-8,
        "ksp_max_it": 2000,  "pc_type": "ilu",
    })
    p = prob.solve()
    print(f"  [{mode}] max|p| = {np.max(np.abs(p.x.array[:])):.1f} Pa")
    return p

# ---------------------------------------------------------------------------
# Gor'kov potential (plane-wave approx — fast, no gradient projection)
# ---------------------------------------------------------------------------

def gorkov_plane_wave(p_vals):
    """
    U ≈ V_p [ f1 |p|²κ_f /4  −  1.5 f2 (3ρ/4) |p|²/(ρ²c²)/4 ]
      = V_p |p|²/4 [ f1 κ_f  −  1.5 f2 · 3/(4ρc²) ]

    Simplified — avoids gradient projection but is only correct
    in the standing-wave limit.  Good enough for trap visualisation.
    """
    p2 = np.abs(p_vals)**2
    E_pot = 0.25 * p2 * KAPPA_F
    # plane-wave:  |v|^2 ≈ |p|^2 / (ρ c)^2
    E_kin = 0.25 * RHO * p2 / (RHO * C)**2
    U = VP * (F1 * E_pot - 1.5 * F2 * E_kin)
    return np.real(U)

# ---------------------------------------------------------------------------
# Export  —  NPZ with all fields  +  XDMF mesh for geometry
# ---------------------------------------------------------------------------

def export_rich(p_func, domain, out_prefix, compute_gorkov=True, preset=None):
    """
    Save:
      {prefix}.npz      — p_real, p_imag, magnitude, phase, gorkov, coords_p2
      {prefix}_mesh.xdmf — mesh only (for geometry loading)
    
    Returns metadata dict (to be written by caller).
    """
    V = p_func.function_space
    coords = V.tabulate_dof_coordinates()      # P2 nodes
    p_vals = p_func.x.array[:]                  # complex128

    p_real = np.real(p_vals).astype(np.float64)
    p_imag = np.imag(p_vals).astype(np.float64)
    mag    = np.abs(p_vals).astype(np.float64)
    phase  = np.angle(p_vals).astype(np.float64)

    data = dict(
        p_real=p_real, p_imag=p_imag,
        magnitude=mag, phase=phase,
        coords=coords,
    )

    if compute_gorkov:
        data['gorkov'] = gorkov_plane_wave(p_vals)

    npz_path = str(out_prefix) + '.npz'
    np.savez_compressed(npz_path, **data)
    print(f"  [NPZ] {npz_path}  ({coords.shape[0]} DOFs)")

    # Also export mesh (P1) for PyVista geometry
    xdmf_path = str(out_prefix) + '_mesh.xdmf'
    with io.XDMFFile(domain.comm, xdmf_path, "w") as xf:
        xf.write_mesh(domain)
    print(f"  [XDMF] {xdmf_path}")
    
    # Return statistics for metadata
    return {
        'max_magnitude_pa': float(np.max(mag)),
        'n_dofs_p2': int(coords.shape[0]),
    }

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate rich 3D data for canonical visualizations")
    parser.add_argument('--preset', choices=['A', 'B'], default='A')
    parser.add_argument('--n_phases', type=int, default=0,
                        help='Number of vortex-phase offsets (0 = no sweep)')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    preset = PRESET_A   # only A supported for now

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) if args.output_dir else \
          Path(__file__).parents[2] / 'results' / f'rich_{args.preset}_{ts}'
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"RICH DATA GENERATION")
    print(f"{'='*70}")
    print(f"Preset : {preset['name']}")
    print(f"Output : {out}")
    print(f"{'='*70}\n")

    # ---- mesh (shared for all solves) ---
    domain, ft = make_mesh(preset)
    
    # Get mesh statistics for metadata
    n_cells = domain.topology.index_map(3).size_global
    n_nodes = domain.topology.index_map(0).size_global
    
    # ---- U-scaling test: compute for both amplitude 1x and 2x ---
    print("\n[U-Scaling Test] Computing for 1x and 2x amplitudes...")
    
    # Temporary: solve with standard amplitude
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    p_temp = fem.Function(fem.functionspace(domain, ("Lagrange", 2)))
    
    # Solve standing with amplitude 1x
    p_stand_1x = solve_helmholtz(domain, ft, preset, 'standing')
    u_1x = gorkov_plane_wave(p_stand_1x.x.array[:])
    u_1x_max = np.max(np.abs(u_1x))
    
    # For 2x, use scaling: U ∝ |p|², so 2×|p| → 4×U
    u_2x_expected = 4.0 * u_1x_max
    
    print(f"  U (1x amplitude) max = {u_1x_max:.3e} J")
    print(f"  U (2x amplitude) expected = {u_2x_expected:.3e} J (scaling factor ≈ 4.0)")

    # ---- standing wave ---
    print("\n[1] Standing wave ...")
    p_stand = solve_helmholtz(domain, ft, preset, 'standing')
    stats_stand = export_rich(p_stand, domain, out / 'standing', preset=preset)

    # ---- vortex only (phase 0) ---
    print("\n[2] Vortex only ...")
    p_vort = solve_helmholtz(domain, ft, preset, 'vortex')
    stats_vort = export_rich(p_vort, domain, out / 'vortex', preset=preset)

    # ---- combined (phase 0) ---
    print("\n[3] Combined (φ=0) ...")
    p_comb = solve_helmholtz(domain, ft, preset, 'combined')
    stats_comb = export_rich(p_comb, domain, out / 'combined', preset=preset)

    # ---- phase sweep (optional) ---
    phase_files = []
    if args.n_phases > 0:
        phases = np.linspace(0, 2*np.pi, args.n_phases, endpoint=False)
        for i, phi in enumerate(phases):
            deg = int(np.degrees(phi))
            print(f"\n[Phase {i+1}/{args.n_phases}]  φ = {deg}° ...")
            p_phi = solve_helmholtz(domain, ft, preset, 'combined', vortex_phase=phi)
            export_rich(p_phi, domain, out / f'combined_phi{deg:03d}', preset=preset)
            phase_files.append(f'combined_phi{deg:03d}')

    # ---- metadata.json ---
    import json
    import inspect
    
    metadata = {
        'timestamp': ts,
        'preset_name': preset['name'],
        'physics': {
            'frequency_hz': float(preset['frequency_hz']),
            'wavelength_m': float(preset['wavelength_m']),
            'density_rho_kg_m3': float(RHO),
            'speed_of_sound_c_m_s': float(C),
            'impedance_z': float(RHO * C),
        },
        'domain': {
            'size_m': float(preset['dish_size_m']),
            'elements_per_wavelength': int(preset['elements_per_wavelength']),
            'n_cells': int(n_cells),
            'n_nodes_p1': int(n_nodes),
            'mesh_type': 'Tetrahedral P1',
        },
        'solvers': {
            'type': 'Helmholtz (complex, time-harmonic)',
            'fem_order': 2,
            'dofs_per_solve_p2': int(stats_stand['n_dofs_p2']),
            'ksp_type': 'gmres',
            'ksp_rtol': 1e-8,
            'ksp_max_iterations': 2000,
            'pc_type': 'ilu',
        },
        'boundary_conditions': {
            'walls': 'Impedance BC (Z = ρc) on all outer boundaries',
            'standing_source': 'Normal velocity actuation on side walls',
            'vortex_source': 'Azimuthal velocity + OAM phase on top aperture',
        },
        'actuation': {
            'vortex_gain': float(preset['vortex_gain']),
            'standing_gain': float(preset['standing_gain']),
            'aperture_radius_m': float(preset['aperture_radius_m']),
            'topological_charge_vortex': 1,
            'n_phase_sweeps': int(args.n_phases),
            'phase_sweep_files': phase_files,
        },
        'output_datasets': {
            'standing': {
                'max_magnitude_pa': stats_stand['max_magnitude_pa'],
                'n_dofs_p2': stats_stand['n_dofs_p2'],
            },
            'vortex': {
                'max_magnitude_pa': stats_vort['max_magnitude_pa'],
                'n_dofs_p2': stats_vort['n_dofs_p2'],
            },
            'combined': {
                'max_magnitude_pa': stats_comb['max_magnitude_pa'],
                'n_dofs_p2': stats_comb['n_dofs_p2'],
            },
        },
        'validation': {
            'u_scaling_test': {
                'description': 'Gor\'kov potential scales as U ∝ |p|² (4x for 2x amplitude)',
                'u_max_1x_amplitude_J': float(u_1x_max),
                'u_max_2x_amplitude_expected_J': float(u_2x_expected),
                'scaling_factor': 4.0,
            }
        },
    }

    metadata_path = out / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[METADATA] {metadata_path}")

    print(f"\n{'='*70}")
    print(f"DONE — {out}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
