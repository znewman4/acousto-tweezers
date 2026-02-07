"""
FieldSolver interface for dimension-agnostic evaluator architecture.

This allows the evaluator to work with any field solver (1D, 2D, 2.5D, 3D, etc.)
without code modification, following professional multiphysics design patterns.
"""
from __future__ import annotations

from typing import Protocol, Tuple, Union
import numpy as np


class Field(Protocol):
    """Abstract field interface.
    
    Any field type (Field1D, Field2D, Field3D) should implement this protocol.
    """

    @property
    def p(self) -> np.ndarray:
        """Pressure array (shape varies by dimension)."""
        ...

    @property
    def omega(self) -> float:
        """Angular frequency (rad/s)."""
        ...

    @property
    def c0(self) -> float:
        """Sound speed (m/s)."""
        ...

    @property
    def rho0(self) -> float:
        """Density (kg/m³)."""
        ...

    def gradient_p(self) -> Tuple[np.ndarray, ...]:
        """Compute ∇p in field coordinates.
        
        Returns
        -------
        Tuple of gradient arrays in order (∂p/∂x, ∂p/∂y) for 2D,
        (∂p/∂x, ∂p/∂y, ∂p/∂z) for 3D, etc.
        """
        ...

    def sample_at(self, pos: np.ndarray) -> complex:
        """Interpolate pressure at position.
        
        Parameters
        ----------
        pos : ndarray of shape (ndim,)
            Position coordinates (x, y, z, etc.)
        
        Returns
        -------
        complex
            Interpolated pressure.
        """
        ...


class FieldSolver(Protocol):
    """Abstract field solver interface.
    
    Implementations: 2.5D Helmholtz, 3D Helmholtz, etc.
    """

    def solve(self, control) -> Field:
        """Solve for acoustic field given control input.
        
        Parameters
        ----------
        control : dict-like
            Control parameters (varies by solver).
        
        Returns
        -------
        Field
            Acoustic pressure field satisfying chosen BC and domain.
        """
        ...

    @property
    def dimensionality(self) -> int:
        """Spatial dimension (2, 3, etc.)."""
        ...


# ============================================================================
# Extended Field implementations with gradient/sampling capabilities
# ============================================================================


def extend_field_2d(field_2d):
    """Wrap 2D Field2D to add gradient_p and sample_at methods."""
    from acousto.solvers.fd_helmholtz_2d import Field2D

    class ExtendedField2D:
        def __init__(self, base_field: Field2D):
            self._field = base_field

        @property
        def p(self):
            return self._field.p

        @property
        def omega(self):
            return self._field.omega

        @property
        def c0(self):
            return self._field.c0

        @property
        def rho0(self):
            return self._field.rho0

        def gradient_p(self) -> tuple[np.ndarray, np.ndarray]:
            """Return ∂p/∂x, ∂p/∂y."""
            dx = self._field.x[1] - self._field.x[0]
            dy = self._field.y[1] - self._field.y[0]
            dpdy, dpdx = np.gradient(self._field.p, dy, dx, edge_order=2)
            return (dpdx, dpdy)

        def sample_at(self, pos: np.ndarray) -> complex:
            """Interpolate p(x, y) at position."""
            from acousto.force.interp_2d import bilinear_sample

            x_pos, y_pos = pos[0], pos[1]
            return bilinear_sample(
                self._field.x, self._field.y, self._field.p, x_pos, y_pos
            )

    return ExtendedField2D(field_2d)


def extend_field_3d(field_3d):
    """Wrap 3D Field3D to add gradient_p and sample_at methods."""
    from acousto.solvers.helmholtz_3d_simple import Field3D

    class ExtendedField3D:
        def __init__(self, base_field: Field3D):
            self._field = base_field

        @property
        def p(self):
            return self._field.p

        @property
        def omega(self):
            return self._field.omega

        @property
        def c0(self):
            return self._field.c0

        @property
        def rho0(self):
            return self._field.rho0

        def gradient_p(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Return ∂p/∂x, ∂p/∂y, ∂p/∂z."""
            dx = self._field.x[1] - self._field.x[0]
            dy = self._field.y[1] - self._field.y[0]
            dz = self._field.z[1] - self._field.z[0]
            dpz, dpy, dpx = np.gradient(self._field.p, dz, dy, dx, edge_order=2)
            return (dpx, dpy, dpz)

        def sample_at(self, pos: np.ndarray) -> complex:
            """Interpolate p(x, y, z) at position using trilinear interpolation."""
            x_pos, y_pos, z_pos = pos[0], pos[1], pos[2]

            # Find nearest grid points
            ix = np.searchsorted(self._field.x, x_pos)
            iy = np.searchsorted(self._field.y, y_pos)
            iz = np.searchsorted(self._field.z, z_pos)

            # Clamp indices
            ix = np.clip(ix, 0, len(self._field.x) - 1)
            iy = np.clip(iy, 0, len(self._field.y) - 1)
            iz = np.clip(iz, 0, len(self._field.z) - 1)

            # Simple nearest-neighbor (could upgrade to trilinear)
            return self._field.p[iz, iy, ix]

    return ExtendedField3D(field_3d)


# ============================================================================
# Solver wrapper interface
# ============================================================================


class Helmholtz25DSolver:
    """Wrapper making 2.5D operator compatible with FieldSolver interface."""

    def __init__(self, operator, control_func, particle_props):
        """
        Parameters
        ----------
        operator : Helmholtz25DOperator
            The low-level solver.
        control_func : callable
            Maps control input -> vb_x (1D array).
        particle_props : ParticleProps
            Particle properties (for dimensionality check).
        """
        self.op = operator
        self.control_func = control_func
        self.particle_props = particle_props

    @property
    def dimensionality(self) -> int:
        return 2  # 2D (with 2.5D physics)

    def solve(self, control_input) -> ExtendedField2D:
        """Solve with control and return wrapped field."""
        vb_x = self.control_func(control_input)
        field_2d = self.op.solve_for_bottom_vb(vb_x)
        return extend_field_2d(field_2d)


class Helmholtz3DSolver:
    """Wrapper making 3D solver compatible with FieldSolver interface."""

    def __init__(self, solve_func, control_func):
        """
        Parameters
        ----------
        solve_func : callable
            Main solve function.
        control_func : callable
            Maps control input -> vb_xy (2D array).
        """
        self.solve_func = solve_func
        self.control_func = control_func

    @property
    def dimensionality(self) -> int:
        return 3

    def solve(self, control_input) -> ExtendedField3D:
        """Solve with control and return wrapped field."""
        vb_xy = self.control_func(control_input)
        field_3d = self.solve_func(vb=vb_xy)
        return extend_field_3d(field_3d)
