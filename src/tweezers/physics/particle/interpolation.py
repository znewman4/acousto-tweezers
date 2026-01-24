"""
Interpolation utilities for 3D fields.

Provides trilinear and higher-order interpolation for
pressure, velocity, and force fields on structured grids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class Grid3D:
    """
    3D structured grid definition.
    
    Parameters
    ----------
    x : np.ndarray
        x coordinates (1D array), length nx.
    y : np.ndarray
        y coordinates (1D array), length ny.
    z : np.ndarray
        z coordinates (1D array), length nz.
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (nx, ny, nz)."""
        return (len(self.x), len(self.y), len(self.z))
    
    @property
    def dx(self) -> float:
        """Grid spacing in x."""
        return self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
    
    @property
    def dy(self) -> float:
        """Grid spacing in y."""
        return self.y[1] - self.y[0] if len(self.y) > 1 else 1.0
    
    @property
    def dz(self) -> float:
        """Grid spacing in z."""
        return self.z[1] - self.z[0] if len(self.z) > 1 else 1.0
    
    @property
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Grid bounds ((xmin, xmax), (ymin, ymax), (zmin, zmax))."""
        return (
            (self.x[0], self.x[-1]),
            (self.y[0], self.y[-1]),
            (self.z[0], self.z[-1]),
        )
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is within grid bounds."""
        x, y, z = point
        return (
            self.x[0] <= x <= self.x[-1] and
            self.y[0] <= y <= self.y[-1] and
            self.z[0] <= z <= self.z[-1]
        )
    
    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create full meshgrid (shape: nx, ny, nz)."""
        return np.meshgrid(self.x, self.y, self.z, indexing='ij')


class TrilinearInterpolator:
    """
    Trilinear interpolation for 3D scalar fields.
    
    Parameters
    ----------
    grid : Grid3D
        Grid definition.
    field : np.ndarray
        Field values, shape (nx, ny, nz).
    fill_value : float or complex
        Value to use outside grid bounds.
    """
    
    def __init__(
        self,
        grid: Grid3D,
        field: np.ndarray,
        fill_value: float | complex = 0.0,
    ):
        self.grid = grid
        self.field = field
        self.fill_value = fill_value
        
        # Precompute inverse spacings
        self._inv_dx = 1.0 / self.grid.dx
        self._inv_dy = 1.0 / self.grid.dy
        self._inv_dz = 1.0 / self.grid.dz
    
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Interpolate field at given points.
        
        Parameters
        ----------
        points : np.ndarray
            Points to interpolate at, shape (n, 3) or (3,).
        
        Returns
        -------
        values : np.ndarray
            Interpolated values, shape (n,) or scalar.
        """
        points = np.atleast_2d(points)
        n_points = points.shape[0]
        values = np.zeros(n_points, dtype=self.field.dtype)
        
        nx, ny, nz = self.grid.shape
        x0, y0, z0 = self.grid.x[0], self.grid.y[0], self.grid.z[0]
        
        for i in range(n_points):
            x, y, z = points[i]
            
            # Normalized coordinates
            xi = (x - x0) * self._inv_dx
            yi = (y - y0) * self._inv_dy
            zi = (z - z0) * self._inv_dz
            
            # Check bounds
            if xi < 0 or xi > nx - 1 or yi < 0 or yi > ny - 1 or zi < 0 or zi > nz - 1:
                values[i] = self.fill_value
                continue
            
            # Integer indices
            i0 = min(int(xi), nx - 2)
            j0 = min(int(yi), ny - 2)
            k0 = min(int(zi), nz - 2)
            
            # Fractional parts
            xd = xi - i0
            yd = yi - j0
            zd = zi - k0
            
            # Trilinear interpolation
            c000 = self.field[i0, j0, k0]
            c001 = self.field[i0, j0, k0 + 1]
            c010 = self.field[i0, j0 + 1, k0]
            c011 = self.field[i0, j0 + 1, k0 + 1]
            c100 = self.field[i0 + 1, j0, k0]
            c101 = self.field[i0 + 1, j0, k0 + 1]
            c110 = self.field[i0 + 1, j0 + 1, k0]
            c111 = self.field[i0 + 1, j0 + 1, k0 + 1]
            
            # Interpolate along x
            c00 = c000 * (1 - xd) + c100 * xd
            c01 = c001 * (1 - xd) + c101 * xd
            c10 = c010 * (1 - xd) + c110 * xd
            c11 = c011 * (1 - xd) + c111 * xd
            
            # Interpolate along y
            c0 = c00 * (1 - yd) + c10 * yd
            c1 = c01 * (1 - yd) + c11 * yd
            
            # Interpolate along z
            values[i] = c0 * (1 - zd) + c1 * zd
        
        return values[0] if points.shape[0] == 1 else values


class VectorFieldInterpolator:
    """
    Trilinear interpolation for 3D vector fields.
    
    Parameters
    ----------
    grid : Grid3D
        Grid definition.
    vx, vy, vz : np.ndarray
        Vector field components, each shape (nx, ny, nz).
    fill_value : float or complex
        Value to use outside grid bounds.
    """
    
    def __init__(
        self,
        grid: Grid3D,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        fill_value: float | complex = 0.0,
    ):
        self.grid = grid
        self._interp_x = TrilinearInterpolator(grid, vx, fill_value)
        self._interp_y = TrilinearInterpolator(grid, vy, fill_value)
        self._interp_z = TrilinearInterpolator(grid, vz, fill_value)
    
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Interpolate vector field at given points.
        
        Parameters
        ----------
        points : np.ndarray
            Points to interpolate at, shape (n, 3) or (3,).
        
        Returns
        -------
        vectors : np.ndarray
            Interpolated vectors, shape (n, 3) or (3,).
        """
        points = np.atleast_2d(points)
        vx = self._interp_x(points)
        vy = self._interp_y(points)
        vz = self._interp_z(points)
        
        result = np.stack([vx, vy, vz], axis=-1)
        return result[0] if points.shape[0] == 1 else result


class GradientInterpolator:
    """
    Interpolate gradient of a scalar field.
    
    Computes gradient using central differences on the grid,
    then interpolates gradient components.
    
    Parameters
    ----------
    grid : Grid3D
        Grid definition.
    field : np.ndarray
        Scalar field values, shape (nx, ny, nz).
    """
    
    def __init__(self, grid: Grid3D, field: np.ndarray):
        self.grid = grid
        self.field = field
        
        # Compute gradient on grid
        grad_x, grad_y, grad_z = np.gradient(
            field, grid.dx, grid.dy, grid.dz
        )
        
        # Create vector interpolator for gradient
        self._gradient_interp = VectorFieldInterpolator(
            grid, grad_x, grad_y, grad_z, fill_value=0.0
        )
    
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Interpolate gradient at given points.
        
        Parameters
        ----------
        points : np.ndarray
            Points to interpolate at, shape (n, 3) or (3,).
        
        Returns
        -------
        gradients : np.ndarray
            Interpolated gradients, shape (n, 3) or (3,).
        """
        return self._gradient_interp(points)


class SplineInterpolator3D:
    """
    Tricubic spline interpolation for smooth fields.
    
    Uses scipy's RegularGridInterpolator with cubic method
    for smoother interpolation than trilinear.
    
    Parameters
    ----------
    grid : Grid3D
        Grid definition.
    field : np.ndarray
        Field values, shape (nx, ny, nz).
    method : str
        Interpolation method ('linear', 'cubic', 'quintic').
    fill_value : float or complex or None
        Value outside bounds. None for extrapolation.
    """
    
    def __init__(
        self,
        grid: Grid3D,
        field: np.ndarray,
        method: str = 'cubic',
        fill_value: float | complex = 0.0,
    ):
        from scipy.interpolate import RegularGridInterpolator
        
        self.grid = grid
        self._interp = RegularGridInterpolator(
            (grid.x, grid.y, grid.z),
            field,
            method=method,
            bounds_error=False,
            fill_value=fill_value,
        )
    
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Interpolate field at given points.
        
        Parameters
        ----------
        points : np.ndarray
            Points to interpolate at, shape (n, 3) or (3,).
        
        Returns
        -------
        values : np.ndarray
            Interpolated values.
        """
        points = np.atleast_2d(points)
        values = self._interp(points)
        return values[0] if len(values) == 1 else values


def compute_hessian(grid: Grid3D, field: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Compute Hessian matrix of scalar field.
    
    Returns 6 independent components of symmetric Hessian:
    H_xx, H_yy, H_zz, H_xy, H_xz, H_yz
    
    Parameters
    ----------
    grid : Grid3D
        Grid definition.
    field : np.ndarray
        Scalar field, shape (nx, ny, nz).
    
    Returns
    -------
    Hessian components : tuple of 6 arrays
        H_xx, H_yy, H_zz, H_xy, H_xz, H_yz
    """
    # First derivatives
    grad_x = np.gradient(field, grid.dx, axis=0)
    grad_y = np.gradient(field, grid.dy, axis=1)
    grad_z = np.gradient(field, grid.dz, axis=2)
    
    # Second derivatives
    H_xx = np.gradient(grad_x, grid.dx, axis=0)
    H_yy = np.gradient(grad_y, grid.dy, axis=1)
    H_zz = np.gradient(grad_z, grid.dz, axis=2)
    H_xy = np.gradient(grad_x, grid.dy, axis=1)
    H_xz = np.gradient(grad_x, grid.dz, axis=2)
    H_yz = np.gradient(grad_y, grid.dz, axis=2)
    
    return H_xx, H_yy, H_zz, H_xy, H_xz, H_yz
