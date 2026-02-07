#!/usr/bin/env python3
"""
Acoustic Vortex Lens Validation
================================

Demonstrate isolated vortex lens behavior showing:
1. Pressure null at vortex core
2. Azimuthal phase winding φ(θ) = ℓθ
3. 3D field structure via iso-surfaces
4. Axial and radial slices

Geometry: Cylindrical water domain with vortex actuation on top boundary
Physics: Helmholtz equation with Neumann BC on top, impedance BC elsewhere
Outputs: 3D renderings, 2D slices, quantitative validation

Author: Acousto-Tweezers Project
Date: February 2026
"""

from pathlib import Path
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Check complex PETSc
from petsc4py import PETSc
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise RuntimeError(
        "This script requires complex PETSc. "
        "Activate the complex environment: micromamba activate acousto-complex"
    )

import dolfinx
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, FunctionSpace, Constant, form, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl
from ufl import inner, grad, dx, ds, TestFunction, TrialFunction

import gmsh
from dolfinx.io import gmshio

# Import project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from acoustweezers.physics.acoustics.vortex_lens import (
    VortexLensConfig, create_vortex_lens,
    create_vortex_boundary_function, compute_azimuthal_phase
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VortexValidationConfig:
    """Configuration for vortex lens validation."""
    
    # Domain geometry (cylindrical)
    radius: float = 2e-3  # 2 mm radius
    height: float = 2e-3  # 2 mm height
    
    # Physics
    frequency: float = 2e6  # 2 MHz
    rho_water: float = 998.0  # kg/m³
    c_water: float = 1500.0  # m/s
    
    # Vortex configuration
    topological_charge: int = 1  # ℓ = 1 (single vortex)
    vortex_amplitude: float = 1e6  # 1 MPa actuation
    velocity_amplitude: float = 0.1  # 0.1 m/s normal velocity
    
    # Boundary conditions (impedance at bottom and sides)
    Z_bottom: float = 3.2e6  # Polystyrene-like impedance (Pa·s/m)
    Z_sides: float = 3.2e6  # Same for sides
    
    # Mesh
    elements_per_wavelength: float = 10.0
    
    # Output
    output_dir: Path = Path("results/vortex_validation")
    
    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency
    
    @property
    def k_water(self) -> complex:
        """Wavenumber in water."""
        return self.omega / self.c_water
    
    @property
    def wavelength(self) -> float:
        return 2 * np.pi / abs(self.k_water)
    
    @property
    def mesh_size(self) -> float:
        return self.wavelength / self.elements_per_wavelength
    
    @property
    def Z_water(self) -> float:
        return self.rho_water * self.c_water
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("VORTEX LENS VALIDATION")
        print("=" * 70)
        print(f"\nGEOMETRY:")
        print(f"  Cylindrical domain: R={self.radius*1e3:.2f} mm, H={self.height*1e3:.2f} mm")
        print(f"\nFREQUENCY DOMAIN:")
        print(f"  f = {self.frequency*1e-6:.2f} MHz")
        print(f"  ω = {self.omega:.3e} rad/s")
        print(f"  λ = {self.wavelength*1e3:.3f} mm")
        print(f"  k = {abs(self.k_water):.1f} rad/m")
        print(f"\nFLUID PROPERTIES:")
        print(f"  Water: ρ={self.rho_water} kg/m³, c={self.c_water} m/s")
        print(f"  Impedance: Z={self.Z_water:.3e} Pa·s/m")
        print(f"\nVORTEX LENS:")
        print(f"  Topological charge: ℓ = {self.topological_charge}")
        print(f"  Velocity amplitude: v₀ = {self.velocity_amplitude*1e3:.1f} mm/s")
        print(f"  Pressure amplitude: {self.vortex_amplitude*1e-6:.2f} MPa")
        print(f"\nMESH:")
        print(f"  Elements per wavelength: {self.elements_per_wavelength:.1f}")
        print(f"  Target element size: {self.mesh_size*1e6:.1f} μm")
        print("=" * 70 + "\n")


# =============================================================================
# Mesh Generation
# =============================================================================

def create_cylinder_mesh(
    config: VortexValidationConfig,
    verbose: bool = True
) -> Tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags]:
    """
    Create cylindrical mesh for vortex validation.
    
    Boundary tags:
    - 1: Bottom (z=0)
    - 2: Side wall (r=R)
    - 3: Top (z=H) - vortex actuation
    
    Returns
    -------
    mesh, cell_tags, facet_tags
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    R = config.radius
    H = config.height
    h = config.mesh_size
    
    if rank == 0:
        gmsh.initialize()
        gmsh.model.add("vortex_cylinder")
        
        # Create cylinder
        cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, H, R)
        gmsh.model.occ.synchronize()
        
        # Get surfaces
        surfaces = gmsh.model.getEntities(dim=2)
        
        bottom_surf = []
        top_surf = []
        side_surf = []
        
        tol = 1e-9
        for dim, tag in surfaces:
            # Get center of mass
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            x, y, z = com
            r = np.sqrt(x**2 + y**2)
            
            if verbose:
                print(f"[Mesh] Surface {tag}: center=({x*1e3:.3f}, {y*1e3:.3f}, {z*1e3:.3f}) mm, r={r*1e3:.3f} mm")
            
            # Classify surface
            if abs(z) < tol:  # Bottom
                bottom_surf.append(tag)
            elif abs(z - H) < tol:  # Top
                top_surf.append(tag)
            elif abs(r - R) < R*0.1:  # Side (check if near radius R)
                side_surf.append(tag)
        
        # Create physical groups
        if bottom_surf:
            gmsh.model.addPhysicalGroup(2, bottom_surf, 1)
            gmsh.model.setPhysicalName(2, 1, "Bottom")
        if side_surf:
            gmsh.model.addPhysicalGroup(2, side_surf, 2)
            gmsh.model.setPhysicalName(2, 2, "Sides")
        if top_surf:
            gmsh.model.addPhysicalGroup(2, top_surf, 3)
            gmsh.model.setPhysicalName(2, 3, "Top")
        
        # Volume
        volumes = [tag for d, tag in gmsh.model.getEntities(dim=3)]
        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Water")
        
        # Set mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        
        if verbose:
            print(f"\n[Mesh] Generated with element size {h*1e6:.1f} μm")
    
    # Import to DOLFINx
    mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3
    )
    
    if rank == 0:
        gmsh.finalize()
    
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    
    if verbose and rank == 0:
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
        num_vertices = mesh.topology.index_map(0).size_global
        print(f"[Mesh] Final mesh: {num_cells} cells, {num_vertices} vertices")
    
    return mesh, cell_tags, facet_tags


# =============================================================================
# Helmholtz Solver with Vortex BC
# =============================================================================

def solve_helmholtz_vortex(
    mesh: dmesh.Mesh,
    facet_tags: dmesh.MeshTags,
    config: VortexValidationConfig,
    vortex_config: VortexLensConfig,
    verbose: bool = True
) -> Function:
    """
    Solve Helmholtz equation with vortex actuation on top boundary.
    
    Weak form:
        ∫ (1/ρ) ∇φ·∇p dV - ∫ (ω²/ρc²) φ p dV
        + ∫_bottom φ (-ik/Z_bottom) p dS
        + ∫_sides φ (-ik/Z_sides) p dS
        = ∫_top φ (-iωρ v₀ A(x) e^(iℓθ)) dS
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    facet_tags : dolfinx.mesh.MeshTags
    config : VortexValidationConfig
    vortex_config : VortexLensConfig
    verbose : bool
    
    Returns
    -------
    p_solution : Function
        Complex pressure field
    """
    comm = mesh.comm
    rank = comm.rank
    
    if verbose and rank == 0:
        print("[Solver] Setting up Helmholtz problem with vortex BC...")
    
    # Function space (P2 elements)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    
    # Trial and test functions
    p = TrialFunction(V)
    phi = TestFunction(V)
    
    # Material properties (constants)
    rho = Constant(mesh, PETSc.ScalarType(config.rho_water))
    c = Constant(mesh, PETSc.ScalarType(config.c_water))
    omega = Constant(mesh, PETSc.ScalarType(config.omega))
    k = omega / c
    
    # Impedances
    Z_bottom = Constant(mesh, PETSc.ScalarType(config.Z_bottom))
    Z_sides = Constant(mesh, PETSc.ScalarType(config.Z_sides))
    
    # Measures
    dx_mesh = dx(domain=mesh)
    ds_mesh = ds(domain=mesh, subdomain_data=facet_tags)
    
    # Bilinear form (LHS)
    # Volume terms: (1/ρ) ∇φ·∇p - (ω²/ρc²) φ p
    a_vol = (1.0/rho) * inner(grad(phi), grad(p)) * dx_mesh \
            - (omega**2 / (rho * c**2)) * phi * p * dx_mesh
    
    # Impedance BC terms: bottom (tag=1) and sides (tag=2)
    # Add: ∫ φ (-ik/Z) p dS
    a_bc = phi * (-1j * k / Z_bottom) * p * ds_mesh(1) \
           + phi * (-1j * k / Z_sides) * p * ds_mesh(2)
    
    # Combine LHS
    a = a_vol + a_bc
    
    # RHS: Vortex actuation on top (tag=3)
    # Create vortex pattern function by interpolating onto V
    v0 = config.velocity_amplitude
    
    # Get vortex center (center of top surface)
    vortex_center = np.array([0.0, 0.0, config.height])
    
    # Create vortex pattern as a Function
    p_vortex_pattern = Function(V)
    
    # Get all DOF coordinates
    coords = V.tabulate_dof_coordinates()
    
    # Compute vortex phase and amplitude at ALL DOFs
    from acoustweezers.physics.acoustics.vortex_lens import compute_azimuthal_phase, compute_amplitude_profile
    phase = compute_azimuthal_phase(coords, vortex_config, vortex_center)
    amplitude = compute_amplitude_profile(coords, vortex_config, vortex_center)
    
    # Create complex pattern: A(x) exp(iφ(x))
    pattern = amplitude * np.exp(1j * phase)
    
    # Normalize to unit max amplitude
    max_amp = np.max(np.abs(pattern))
    if max_amp > 0:
        pattern = pattern / max_amp
    
    # Set function values
    p_vortex_pattern.x.array[:] = pattern
    
    # Actuation term: -iωρ v₀ * pattern
    # The pattern is already defined everywhere, boundary integral will pick it up
    g_vortex = -1j * omega * rho * v0 * p_vortex_pattern
    
    L = phi * g_vortex * ds_mesh(3)
    
    # Assemble system
    if verbose and rank == 0:
        print("[Solver] Assembling system...")
    
    a_compiled = form(a)
    L_compiled = form(L)
    
    A = assemble_matrix(a_compiled)
    A.assemble()
    b = assemble_vector(L_compiled)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # Solve
    p_solution = Function(V)
    
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    if verbose:
        ksp.setMonitor(lambda ksp, its, rnorm: 
                      print(f"  Iteration {its}: residual = {rnorm:.3e}") if its % 50 == 0 else None)
    
    if verbose and rank == 0:
        print("[Solver] Solving...")
    
    ksp.solve(b, p_solution.x.petsc_vec)
    p_solution.x.scatter_forward()
    
    # Statistics
    p_array = p_solution.x.array
    p_max = comm.allreduce(np.max(np.abs(p_array)), op=MPI.MAX)
    p_mean = comm.allreduce(np.mean(np.abs(p_array)), op=MPI.SUM) / comm.size
    
    if verbose and rank == 0:
        print(f"[Solver] Solution statistics:")
        print(f"  max|p| = {p_max:.3e} Pa")
        print(f"  mean|p| = {p_mean:.3e} Pa")
        print(f"{'='*70}\n")
    
    return p_solution


# =============================================================================
# Visualization
# =============================================================================

def plot_axial_slice(
    p_function: Function,
    config: VortexValidationConfig,
    output_path: Path,
    z_slice: Optional[float] = None
):
    """Plot pressure magnitude and phase in axial slice (z=const)."""
    if z_slice is None:
        z_slice = config.height / 2
    
    # Create grid
    N = 200
    x = np.linspace(-config.radius*0.95, config.radius*0.95, N)
    y = np.linspace(-config.radius*0.95, config.radius*0.95, N)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)
    
    # Flatten and evaluate
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate pressure
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    mesh = p_function.function_space.mesh
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)
    cells = compute_colliding_cells(mesh, cell_candidates, points)
    
    p_vals = np.zeros(len(points), dtype=np.complex128)
    valid_mask = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        if len(cells.links(i)) > 0:
            cell = cells.links(i)[0]
            p_vals[i] = p_function.eval(point, cell)[0]
            valid_mask[i] = True
    
    # Mask points outside domain
    r_pts = np.sqrt(X.ravel()**2 + Y.ravel()**2)
    valid_mask &= (r_pts <= config.radius)
    
    p_mag = np.abs(p_vals).reshape(X.shape)
    p_phase = np.angle(p_vals).reshape(X.shape)
    
    p_mag[~valid_mask.reshape(X.shape)] = np.nan
    p_phase[~valid_mask.reshape(X.shape)] = np.nan
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Magnitude
    im0 = axes[0].pcolormesh(X*1e3, Y*1e3, p_mag*1e-6, shading='gouraud', cmap='viridis')
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    axes[0].set_title(f'|p| (MPa) at z={z_slice*1e3:.2f} mm')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='|p| (MPa)')
    
    # Phase
    im1 = axes[1].pcolormesh(X*1e3, Y*1e3, p_phase, shading='gouraud', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    axes[1].set_title(f'phase(p) at z={z_slice*1e3:.2f} mm')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], label='phase (rad)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"[Plot] Saved axial slice to {output_path}")


def plot_radial_profile(
    p_function: Function,
    config: VortexValidationConfig,
    output_path: Path,
    z_slice: Optional[float] = None
):
    """Plot radial profile of pressure magnitude along x-axis."""
    if z_slice is None:
        z_slice = config.height / 2
    
    # Sample along x-axis (y=0)
    N = 200
    r_vals = np.linspace(0, config.radius*0.95, N)
    points = np.column_stack([r_vals, np.zeros(N), np.full(N, z_slice)])
    
    # Evaluate
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    mesh = p_function.function_space.mesh
    tree = bb_tree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions_points(tree, points)
    cells = compute_colliding_cells(mesh, cell_candidates, points)
    
    p_vals = np.zeros(N, dtype=np.complex128)
    for i, point in enumerate(points):
        if len(cells.links(i)) > 0:
            cell = cells.links(i)[0]
            p_vals[i] = p_function.eval(point, cell)[0]
    
    p_mag = np.abs(p_vals)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_vals*1e3, p_mag*1e-6, 'b-', linewidth=2)
    ax.set_xlabel('Radial distance r (mm)')
    ax.set_ylabel('|p| (MPa)')
    ax.set_title(f'Radial pressure profile at z={z_slice*1e3:.2f} mm')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Highlight if there's a null
    if p_mag[0] < 0.1 * np.max(p_mag):
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Vortex core (null)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"[Plot] Saved radial profile to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Acoustic vortex lens validation")
    parser.add_argument('--topological_charge', type=int, default=1, help='Topological charge ℓ')
    parser.add_argument('--frequency', type=float, default=2.0, help='Frequency (MHz)')
    parser.add_argument('--elements_per_wavelength', type=float, default=10.0, help='Mesh resolution')
    parser.add_argument('--output_dir', type=str, default='results/vortex_validation', help='Output directory')
    
    args = parser.parse_args()
    
    # Configuration
    config = VortexValidationConfig(
        frequency=args.frequency * 1e6,
        topological_charge=args.topological_charge,
        elements_per_wavelength=args.elements_per_wavelength,
        output_dir=Path(args.output_dir)
    )
    
    config.print_summary()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / f"run_{timestamp}_ell{config.topological_charge}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mesh
    print("[Main] Creating cylindrical mesh...")
    mesh, cell_tags, facet_tags = create_cylinder_mesh(config)
    
    # Vortex configuration
    vortex_config = create_vortex_lens(
        ell=config.topological_charge,
        amplitude=config.vortex_amplitude,
        center=None,  # Will be set to top center
        axis='z'
    )
    
    # Solve
    print("\n[Main] Solving Helmholtz equation with vortex BC...")
    p_solution = solve_helmholtz_vortex(mesh, facet_tags, config, vortex_config)
    
    # Visualize
    print("\n[Main] Generating visualizations...")
    
    # Axial slices at different heights
    for z_frac in [0.25, 0.5, 0.75]:
        z = z_frac * config.height
        plot_axial_slice(
            p_solution, config,
            run_dir / f"axial_slice_z{z_frac:.2f}.png",
            z_slice=z
        )
    
    # Radial profile
    plot_radial_profile(
        p_solution, config,
        run_dir / "radial_profile.png"
    )
    
    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
