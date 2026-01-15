#!/usr/bin/env python3
"""
Trap Stability Verification Script

Clarifies the sign conventions and verifies stability calculations:

Physics relationships:
- F = -∇U  (force is negative gradient of potential)
- Stable trap = local minimum of U
- At local minimum of U: Hessian(U) is positive definite
- Since F = -∇U, Jacobian dF/dx = -Hessian(U)
- So stable trap => dF/dx is NEGATIVE definite (eigenvalues < 0)

This script:
1. Finds a trap center
2. Computes |F| at trap (should be ~0)
3. Computes Hessian of U (should be positive definite for stable)
4. Computes Jacobian of F (should be negative definite for stable)
5. Verifies the relationship: eig(dF/dx) = -eig(Hessian(U))

Usage:
    python scripts/verify_trap_stability.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.analysis import find_trap_center
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control2Pucks, BottomFootprint25DEvaluator,
)


@dataclass
class TrapStabilityReport:
    """Comprehensive trap stability analysis."""
    # Location
    trap_x: float
    trap_y: float
    
    # Force at trap (should be ~0)
    fx_at_trap: float
    fy_at_trap: float
    force_magnitude: float
    
    # Hessian of U at trap
    hessian_U: np.ndarray  # 2x2
    hessian_eigvals: np.ndarray  # 2
    hessian_eigvecs: np.ndarray  # 2x2
    is_U_minimum: bool  # True if both eigenvalues > 0
    
    # Jacobian of F at trap
    jacobian_F: np.ndarray  # 2x2 = [[dFx/dx, dFx/dy], [dFy/dx, dFy/dy]]
    jacobian_eigvals: np.ndarray  # 2
    jacobian_eigvecs: np.ndarray  # 2x2
    is_F_stable: bool  # True if both eigenvalues < 0
    
    # Verification
    relationship_verified: bool  # True if eig(dF/dx) ≈ -eig(Hessian(U))
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "TRAP STABILITY REPORT",
            "=" * 60,
            f"Location: ({self.trap_x*1e3:.4f}, {self.trap_y*1e3:.4f}) mm",
            "",
            "--- Force at Trap ---",
            f"  Fx = {self.fx_at_trap:.3e} N",
            f"  Fy = {self.fy_at_trap:.3e} N",
            f"  |F| = {self.force_magnitude:.3e} N",
            f"  (Should be ~0 at trap center)",
            "",
            "--- Hessian of U (∂²U/∂x∂y) ---",
            f"  Eigenvalues: {self.hessian_eigvals[0]:.3e}, {self.hessian_eigvals[1]:.3e}",
            f"  Is U local minimum? {'YES ✓' if self.is_U_minimum else 'NO ✗'}",
            f"  (Stable trap requires POSITIVE eigenvalues)",
            "",
            "--- Jacobian of F (∂F/∂x) ---",
            f"  Eigenvalues: {self.jacobian_eigvals[0]:.3e}, {self.jacobian_eigvals[1]:.3e}",
            f"  Is F stable? {'YES ✓' if self.is_F_stable else 'NO ✗'}",
            f"  (Stable trap requires NEGATIVE eigenvalues)",
            "",
            "--- Verification: dF/dx = -Hessian(U) ---",
            f"  Relationship holds? {'YES ✓' if self.relationship_verified else 'NO ✗'}",
            "",
            "--- Conclusion ---",
        ]
        
        if self.is_U_minimum and self.is_F_stable:
            lines.append("  STABLE TRAP (particle will be attracted)")
        elif not self.is_U_minimum and not self.is_F_stable:
            lines.append("  UNSTABLE (saddle point or maximum)")
        else:
            lines.append("  INCONSISTENT (check computations)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_hessian_U(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    trap_x: float,
    trap_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Hessian of U at trap location using finite differences.
    
    Returns (Hessian matrix, eigenvalues, eigenvectors)
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    
    # Find indices
    ix = np.argmin(np.abs(x - trap_x))
    iy = np.argmin(np.abs(y - trap_y))
    
    # Ensure we have neighbors
    ix = np.clip(ix, 1, len(x) - 2)
    iy = np.clip(iy, 1, len(y) - 2)
    
    # Second derivatives using central differences
    # ∂²U/∂x² ≈ (U[i+1,j] - 2*U[i,j] + U[i-1,j]) / dx²
    Uxx = (U[iy, ix+1] - 2*U[iy, ix] + U[iy, ix-1]) / (dx * dx)
    Uyy = (U[iy+1, ix] - 2*U[iy, ix] + U[iy-1, ix]) / (dy * dy)
    
    # Mixed derivative: ∂²U/∂x∂y
    Uxy = (U[iy+1, ix+1] - U[iy+1, ix-1] - U[iy-1, ix+1] + U[iy-1, ix-1]) / (4 * dx * dy)
    
    H = np.array([
        [Uxx, Uxy],
        [Uxy, Uyy]
    ], dtype=float)
    
    eigvals, eigvecs = np.linalg.eigh(H)
    
    return H, eigvals, eigvecs


def compute_jacobian_F(
    x: np.ndarray,
    y: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    trap_x: float,
    trap_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Jacobian of F at trap location using finite differences.
    
    J = [[∂Fx/∂x, ∂Fx/∂y],
         [∂Fy/∂x, ∂Fy/∂y]]
    
    Returns (Jacobian matrix, eigenvalues, eigenvectors)
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    
    # Find indices
    ix = np.argmin(np.abs(x - trap_x))
    iy = np.argmin(np.abs(y - trap_y))
    
    # Ensure we have neighbors
    ix = np.clip(ix, 1, len(x) - 2)
    iy = np.clip(iy, 1, len(y) - 2)
    
    # First derivatives using central differences
    dFx_dx = (Fx[iy, ix+1] - Fx[iy, ix-1]) / (2 * dx)
    dFx_dy = (Fx[iy+1, ix] - Fx[iy-1, ix]) / (2 * dy)
    dFy_dx = (Fy[iy, ix+1] - Fy[iy, ix-1]) / (2 * dx)
    dFy_dy = (Fy[iy+1, ix] - Fy[iy-1, ix]) / (2 * dy)
    
    J = np.array([
        [dFx_dx, dFx_dy],
        [dFy_dx, dFy_dy]
    ], dtype=float)
    
    eigvals, eigvecs = np.linalg.eig(J)  # Use eig, not eigh (J may not be symmetric)
    eigvals = np.real(eigvals)  # Take real part
    
    return J, eigvals, eigvecs


def analyze_trap_stability(
    x: np.ndarray,
    y: np.ndarray,
    U: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    trap_x: float,
    trap_y: float,
    alpha_g: float = 1.0,
) -> TrapStabilityReport:
    """
    Complete trap stability analysis at given location.
    
    Note: alpha_g is the force scaling factor. Fx, Fy should already be scaled.
    """
    # Sample force at trap
    fx_at_trap, fy_at_trap = bilinear_sample_vec(x, y, Fx, Fy, trap_x, trap_y)
    force_magnitude = np.sqrt(fx_at_trap**2 + fy_at_trap**2)
    
    # Compute Hessian of U
    H, h_eigvals, h_eigvecs = compute_hessian_U(x, y, U, trap_x, trap_y)
    is_U_minimum = np.all(h_eigvals > 0)
    
    # Compute Jacobian of F
    J, j_eigvals, j_eigvecs = compute_jacobian_F(x, y, Fx, Fy, trap_x, trap_y)
    is_F_stable = np.all(j_eigvals < 0)
    
    # Verify relationship: dF/dx should equal -Hessian(U) * alpha_g
    # Since F = -alpha_g * ∇U, dF/dx = -alpha_g * Hessian(U)
    # So J ≈ -alpha_g * H
    expected_J = -alpha_g * H
    relationship_error = np.linalg.norm(J - expected_J) / (np.linalg.norm(expected_J) + 1e-20)
    relationship_verified = relationship_error < 0.1  # Within 10%
    
    return TrapStabilityReport(
        trap_x=trap_x,
        trap_y=trap_y,
        fx_at_trap=fx_at_trap,
        fy_at_trap=fy_at_trap,
        force_magnitude=force_magnitude,
        hessian_U=H,
        hessian_eigvals=h_eigvals,
        hessian_eigvecs=h_eigvecs,
        is_U_minimum=is_U_minimum,
        jacobian_F=J,
        jacobian_eigvals=j_eigvals,
        jacobian_eigvecs=j_eigvecs,
        is_F_stable=is_F_stable,
        relationship_verified=relationship_verified,
    )


def main():
    print("=" * 60)
    print("TRAP STABILITY VERIFICATION")
    print("=" * 60)
    
    # Setup
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=120, Ny=120)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    alpha_g = 1e3
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=alpha_g,
        max_step=0.05e-3,
        use_2d_forcing=True,
    )
    ev = BottomFootprint25DEvaluator(domain, medium, particle, cfg)
    
    # Create a control configuration
    v_amp = 0.05
    u = Control2Pucks(
        xA=0.5e-3, yA=0.05e-3,
        xB=1.5e-3, yB=0.05e-3,
        vA=v_amp, vB=v_amp,
        phiA=0.0, phiB=np.pi,
    )
    
    print(f"\nControl: transducers at ({u.xA*1e3:.2f}, {u.yA*1e3:.2f}) and "
          f"({u.xB*1e3:.2f}, {u.yB*1e3:.2f}) mm")
    print(f"Phase difference: {u.phiB - u.phiA:.2f} rad")
    
    # Compute field
    vb_x = ev.control_to_forcing_band_vb(u)
    field = ev.op.solve_for_bottom_vb(vb_x)
    U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
    
    # Apply scaling to forces
    Fx_scaled = Fx * alpha_g
    Fy_scaled = Fy * alpha_g
    
    # Find trap center
    search_x = (u.xA + u.xB) / 2
    search_y = domain.Ly / 2
    
    trap_result = find_trap_center(
        field.x, field.y, U, Fx, Fy,
        particle_x=search_x, particle_y=search_y,
        search_radius=0.5e-3,
    )
    
    print(f"\nTrap found at ({trap_result.x*1e3:.4f}, {trap_result.y*1e3:.4f}) mm")
    print(f"find_trap_center reports: is_stable = {trap_result.is_stable}")
    print(f"find_trap_center eigenvalues: {trap_result.stiffness_eigvals}")
    
    # Full stability analysis
    report = analyze_trap_stability(
        field.x, field.y, U, Fx_scaled, Fy_scaled,
        trap_result.x, trap_result.y,
        alpha_g=alpha_g,
    )
    
    print("\n" + str(report))
    
    # Check for inconsistencies
    print("\n--- Cross-Check with find_trap_center ---")
    our_stability = report.is_F_stable
    their_stability = trap_result.is_stable
    
    if our_stability == their_stability:
        print("✓ Stability classification matches find_trap_center()")
    else:
        print("✗ MISMATCH: Our analysis says", "STABLE" if our_stability else "UNSTABLE")
        print("            find_trap_center says", "STABLE" if their_stability else "UNSTABLE")
        print("\n  Possible causes:")
        print("  - Different eigenvalue sign conventions")
        print("  - find_trap_center uses Hessian(U), we use Jacobian(F)")
        print("  - Numerical precision issues")
    
    # Visualize
    output_dir = Path(__file__).parents[1] / "results" / "stability_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x_mm = field.x * 1e3
    y_mm = field.y * 1e3
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # Potential
    ax = axes[0]
    ax.contourf(X, Y, U * 1e15, levels=30, cmap='viridis')
    ax.scatter(trap_result.x * 1e3, trap_result.y * 1e3, 
               s=200, marker='x', c='red', linewidths=3, label='trap')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Gor\'kov Potential U')
    ax.set_aspect('equal')
    ax.legend()
    
    # Force magnitude
    ax = axes[1]
    F_mag = np.sqrt(Fx_scaled**2 + Fy_scaled**2)
    ax.contourf(X, Y, F_mag, levels=30, cmap='hot')
    ax.scatter(trap_result.x * 1e3, trap_result.y * 1e3,
               s=200, marker='x', c='cyan', linewidths=3, label='trap')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Force Magnitude |F|')
    ax.set_aspect('equal')
    ax.legend()
    
    # Force vector field (quiver)
    ax = axes[2]
    skip = 8
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              Fx_scaled[::skip, ::skip], Fy_scaled[::skip, ::skip],
              alpha=0.7, scale=1e-6)
    ax.scatter(trap_result.x * 1e3, trap_result.y * 1e3,
               s=200, marker='x', c='red', linewidths=3, label='trap')
    
    # Draw eigenvectors
    scale = 0.2  # mm
    tx, ty = trap_result.x * 1e3, trap_result.y * 1e3
    for i, (eigval, color) in enumerate(zip(report.jacobian_eigvals, ['blue', 'green'])):
        vec = report.jacobian_eigvecs[:, i]
        label = f'λ={eigval:.2e}'
        ax.arrow(tx, ty, scale*vec[0], scale*vec[1], 
                 head_width=0.05, color=color, label=label)
    
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Force Vectors + Jacobian Eigenvectors')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_xlim(tx - 0.5, tx + 0.5)
    ax.set_ylim(ty - 0.5, ty + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "stability_analysis.png", dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'stability_analysis.png'}")


if __name__ == "__main__":
    main()
