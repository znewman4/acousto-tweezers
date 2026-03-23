"""
Vortex perturbation generator for compare_vortex_vs_cshape.py.

Wraps a precomputed symmetric-vortex pressure field and provides:

    gen = VortexPerturbation(p_vortex_centered, xg, yg)
    p_perturb = gen.get_field(center_xy)   # raw field, no alpha or psi

The returned field is sampled by coordinate remapping from a larger
reference grid:

    p_shift(x, y) = p_ref(x - dx, y - dy)

with (dx, dy) = center_xy - source_center.

This avoids hard seams caused by zero-padding when translating within a
small ROI grid.

Source data
-----------
p_vortex_centered comes from the vortex transport NPZ:
    results/deliverables/vortex_stage_transport/transport/transport_case_for_gif.npz
    key: 'p_vortex_centered'  (shape 400×400, complex, on 6 mm grid)

The class can evaluate onto any output grid (e.g., the ROI grid) while
keeping translation support from the larger source grid.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import RectBivariateSpline


class VortexPerturbation:
    """
    Symmetric-vortex perturbation generator.

    Parameters
    ----------
    p_vortex_centered : complex ndarray (ny, nx)
        Vortex field with its azimuthal-phase centre at (xg centre, yg centre).
    xg, yg : 1-D float arrays
        Coordinate grids corresponding to p_vortex_centered.
    out_xg, out_yg : 1-D float arrays or None
        Target grid for get_field output. If None, output uses source grid.
    """

    def __init__(
        self,
        p_vortex_centered: np.ndarray,
        xg: np.ndarray,
        yg: np.ndarray,
        out_xg: Optional[np.ndarray] = None,
        out_yg: Optional[np.ndarray] = None,
    ) -> None:
        self._p_v = np.asarray(p_vortex_centered, dtype=complex)
        self._xg = np.asarray(xg, dtype=float)
        self._yg = np.asarray(yg, dtype=float)
        self._out_xg = np.asarray(out_xg, dtype=float) if out_xg is not None else self._xg
        self._out_yg = np.asarray(out_yg, dtype=float) if out_yg is not None else self._yg

        # Source centre = centre of the coordinate grid
        self._source_centre = np.array(
            [0.5 * (float(xg[0]) + float(xg[-1])),
             0.5 * (float(yg[0]) + float(yg[-1]))],
            dtype=float,
        )

        # Pre-build splines once; evaluated repeatedly in get_field.
        self._spline_re = RectBivariateSpline(self._yg, self._xg, np.real(self._p_v), kx=3, ky=3)
        self._spline_im = RectBivariateSpline(self._yg, self._xg, np.imag(self._p_v), kx=3, ky=3)

    def get_field(self, center_xy: np.ndarray) -> np.ndarray:
        """
        Return the vortex field shifted so its centre is at center_xy.

        Parameters
        ----------
        center_xy : array-like (2,)   [x, y] in metres

        Returns
        -------
        p_shifted : complex ndarray (len(out_yg), len(out_xg))
        """
        centre = np.asarray(center_xy, dtype=float)
        dxy = centre - self._source_centre

        # Translate by remapping coordinates into the source field frame.
        xq = self._out_xg - float(dxy[0])
        yq = self._out_yg - float(dxy[1])

        # Keep interpolation in-domain to avoid extrapolation artefacts.
        xq = np.clip(xq, float(self._xg[0]), float(self._xg[-1]))
        yq = np.clip(yq, float(self._yg[0]), float(self._yg[-1]))

        re = self._spline_re(yq, xq)
        im = self._spline_im(yq, xq)
        return (re + 1j * im).astype(complex)
