"""
C-shape lens perturbation generator for compare_vortex_vs_cshape.py.

Wraps the backpropagated 15 mm lens pressure field and provides:

    gen = CShapePerturbation(p_lens_roi, xg, yg)
    p_perturb = gen.get_field(center_xy)   # raw field, no alpha or psi

The field is spatially windowed with a Gaussian if window_sigma is set,
otherwise the full lens field is returned unchanged.

The returned field is a static pattern (the C-shape lens does not
translate with the window centre in the best-known configuration:
full_lens, translation=static).  The comparison script always calls
with center_xy = midpoint.

Source data
-----------
p_lens_roi comes from the overlay study NPZ:
    results/c_shape_lens_15mm_overlay_study_20260310_170620/npz/roi_fields.npz
    key: 'p_lens_roi'  (shape 400×400, complex, on ROI grid 2.18–3.82 mm)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CShapePerturbation:
    """
    C-shape lens perturbation generator.

    Parameters
    ----------
    p_lens_roi : complex ndarray (ny, nx)
        Propagated lens field in the trap ROI.
    xg, yg : 1-D float arrays
        Coordinate grids for p_lens_roi.
    window_sigma : float or None
        If set, apply Gaussian spatial window W(x,y) = exp(-r²/2σ²)
        centred at center_xy when get_field is called.
        None (default) → no windowing (full lens, best known config).
    """

    def __init__(
        self,
        p_lens_roi: np.ndarray,
        xg: np.ndarray,
        yg: np.ndarray,
        window_sigma: Optional[float] = None,
    ) -> None:
        self._p_lens = np.asarray(p_lens_roi, dtype=complex)
        self._xg     = np.asarray(xg, dtype=float)
        self._yg     = np.asarray(yg, dtype=float)
        self._sigma  = window_sigma

        # Pre-build meshgrid for optional windowing
        if window_sigma is not None:
            XX, YY = np.meshgrid(xg, yg)
            self._XX = XX
            self._YY = YY

    def get_field(self, center_xy: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the lens field, optionally windowed around center_xy.

        Parameters
        ----------
        center_xy : array-like (2,) or None
            Window centre [x, y] in metres.  Ignored when window_sigma is None.

        Returns
        -------
        p_eff : complex ndarray (ny, nx)
        """
        if self._sigma is None:
            return self._p_lens.copy()

        if center_xy is None:
            # Fall back to grid centre
            center_xy = np.array(
                [0.5 * (self._xg[0] + self._xg[-1]),
                 0.5 * (self._yg[0] + self._yg[-1])],
                dtype=float,
            )
        cx, cy = float(center_xy[0]), float(center_xy[1])
        r2 = (self._XX - cx)**2 + (self._YY - cy)**2
        W = np.exp(-r2 / (2.0 * self._sigma**2))
        return W * self._p_lens
