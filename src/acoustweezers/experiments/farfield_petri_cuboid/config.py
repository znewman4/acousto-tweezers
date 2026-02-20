"""
Configuration for Far-Field Petri Cuboid Experiment.

Domain layout (side-view cross-section, x–z plane at y = Ly/2):

      z = H_total ─────── top face (water–air impedance Robin) ──────
      │                       petri slab region                      │
      z = H_under ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
      │                    under-bath  (far field)                   │
      │  PML(side) │         physical center          │  PML(side)   │
      │            │                                  │              │
      │  PML(side) │                                  │  PML(side)   │
      z = 0 ────── PML(bot) ──  bottom_disk  ── PML(bot) ───────────
           x=0    x_pml_in               Lx-x_pml_in           x=Lx

Standing-wave transducers are placed on the side walls in the petri
slab (z ∈ [H_under, H_total]).

Author: Acousto-Tweezers Project
Date: 2026-02-16
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FarFieldConfig:
    """
    Configuration for far-field petri-cuboid simulation with PML.

    The total mesh box is  Lx × Ly × H_total.
      Lx = Ly = dish lateral extent  (includes PML thickness on each side)
      H_total = H_under + H_top
    Physical (non-PML) region sits inset by t_pml_xy from each side
    and by t_pml_z from the bottom (outside the disk column).
    """

    # ── geometry ──────────────────────────────────────────────────────
    Lx: float = 10e-3           # lateral size [m]
    Ly: float = 10e-3
    H_under: float = 5e-3       # under-bath depth (below petri) [m]
    H_top: float = 1e-3         # petri slab thickness [m]

    # ── frequency ─────────────────────────────────────────────────────
    frequency_hz: float = 2.0e6   # 2 MHz

    # ── material (water 20 °C) ────────────────────────────────────────
    rho: float = 997.0
    c: float = 1484.0

    # ── bottom disk source ────────────────────────────────────────────
    disk_radius: float = 2.0e-3           # lens aperture radius [m]
    disk_velocity_amplitude: float = 10e-6  # V_n amplitude [m/s]
    vortex_topological_charge: int = 1
    vortex_apodization: str = "cosine_taper"

    # lens model — set lens_drive to "plastic", "ideal", or "axicon"
    lens_drive: str = "plastic"   # "ideal", "plastic", or "axicon"
    lens_l: int = 1               # topological charge (overrides vortex_topological_charge when plastic)
    lens_focal_length: float = 10e-3      # focusing focal distance [m]
    lens_focus_offset_x: float = 0.2e-3   # off-axis bias for translation [m]
    lens_focus_offset_y: float = 0.0      # [m]
    lens_c_lens: float = 2700.0           # speed of sound in plastic [m/s]
    lens_apodization: str = "cosine_taper"  # "cosine_taper", "tukey", "uniform"
    lens_apodization_strength: float = 1.0  # taper parameter

    # axicon lens — set lens_drive="axicon" to enable
    lens_axicon_angle_deg: float = 15.0   # axicon half-angle [degrees]

    # ── standing-wave transducers (petri slab side-walls) ─────────────
    standing_velocity_amplitude: float = 1e-6
    standing_phase_pattern: str = "antiphase"
    standing_axis: str = "both"

    # ── top boundary ──────────────────────────────────────────────────
    top_bc_type: str = "impedance"       # "impedance" or "dirichlet"
    top_impedance_Zrel: float = 0.001    # Z_top / Z_water

    # ── PML ───────────────────────────────────────────────────────────
    pml_n_wavelengths_xy: float = 1.5    # PML thickness in wavelengths (each side)
    pml_n_wavelengths_z: float = 1.5     # PML thickness below disk
    pml_degree: int = 2                  # polynomial ramp order
    pml_sigma_max_factor: float = 5.0    # σ_max = factor * ω  (dimensionless)
    pml_enabled: bool = True             # False → rigid walls (comparison)

    # ── mesh ──────────────────────────────────────────────────────────
    elements_per_wavelength: int = 6
    min_elements_z: int = 10

    # ── solver ────────────────────────────────────────────────────────
    verbose_solver: bool = True   # print KSP stats, timing breakdown

    # ── derived ───────────────────────────────────────────────────────
    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency_hz

    @property
    def k(self) -> float:
        return self.omega / self.c

    @property
    def wavelength(self) -> float:
        return self.c / self.frequency_hz

    @property
    def Z_water(self) -> float:
        return self.rho * self.c

    @property
    def Z_top(self) -> float:
        return self.top_impedance_Zrel * self.Z_water

    @property
    def H_total(self) -> float:
        return self.H_under + self.H_top

    @property
    def t_pml_xy(self) -> float:
        """PML thickness on each lateral side [m]."""
        return self.pml_n_wavelengths_xy * self.wavelength

    @property
    def t_pml_z(self) -> float:
        """PML thickness below the domain [m]."""
        return self.pml_n_wavelengths_z * self.wavelength

    @property
    def physical_x_range(self):
        """(x_min, x_max) of physical (non-PML) region."""
        return (self.t_pml_xy, self.Lx - self.t_pml_xy)

    @property
    def physical_y_range(self):
        return (self.t_pml_xy, self.Ly - self.t_pml_xy)

    @property
    def physical_z_min(self) -> float:
        """z above which the bottom-PML ends (for the non-disk region)."""
        return self.t_pml_z

    @property
    def disk_center_x(self) -> float:
        return self.Lx / 2

    @property
    def disk_center_y(self) -> float:
        return self.Ly / 2

    @property
    def mesh_nx(self) -> int:
        return max(20, int(self.Lx / self.wavelength * self.elements_per_wavelength))

    @property
    def mesh_ny(self) -> int:
        return max(20, int(self.Ly / self.wavelength * self.elements_per_wavelength))

    @property
    def mesh_nz(self) -> int:
        nz = int(self.H_total / self.wavelength * self.elements_per_wavelength)
        return max(self.min_elements_z, nz)

    @property
    def sigma_max(self) -> float:
        """Maximum PML absorption coefficient [1/s]."""
        return self.pml_sigma_max_factor * self.omega

    def to_dict(self) -> dict:
        d = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            if val is None:
                d[key] = None
            elif isinstance(val, np.ndarray):
                d[key] = val.tolist()
            elif isinstance(val, tuple):
                d[key] = list(val)
            else:
                d[key] = val
        # Add derived
        d["omega"] = self.omega
        d["k"] = self.k
        d["wavelength"] = self.wavelength
        d["H_total"] = self.H_total
        d["t_pml_xy"] = self.t_pml_xy
        d["t_pml_z"] = self.t_pml_z
        d["sigma_max"] = self.sigma_max
        d["mesh_nx"] = self.mesh_nx
        d["mesh_ny"] = self.mesh_ny
        d["mesh_nz"] = self.mesh_nz
        return d

    def describe(self) -> str:
        if self.lens_drive == "axicon":
            lens_info = (
                f"  Lens drive:  axicon  l={self.lens_l}  "
                f"alpha={self.lens_axicon_angle_deg:.1f} deg  "
                f"apod={self.lens_apodization}\n"
            )
        elif self.lens_drive == "plastic":
            lens_info = (
                f"  Lens drive:  plastic  l={self.lens_l}  "
                f"f={self.lens_focal_length*1e3:.1f} mm  "
                f"offset=({self.lens_focus_offset_x*1e3:.2f}, {self.lens_focus_offset_y*1e3:.2f}) mm\n"
                f"  Lens plastic: c_lens={self.lens_c_lens:.0f} m/s  "
                f"apod={self.lens_apodization}\n"
            )
        else:
            lens_info = f"  Lens drive:  {self.lens_drive}\n"
        return (
            f"Far-Field Petri Cuboid Config\n"
            f"{'='*40}\n"
            f"Box:  {self.Lx*1e3:.1f} × {self.Ly*1e3:.1f} × {self.H_total*1e3:.1f} mm\n"
            f"  Under-bath:  {self.H_under*1e3:.1f} mm\n"
            f"  Petri slab:  {self.H_top*1e3:.1f} mm\n"
            f"Freq:        {self.frequency_hz/1e6:.2f} MHz\n"
            f"Wavelength:  {self.wavelength*1e3:.3f} mm\n"
            f"Disk radius: {self.disk_radius*1e3:.2f} mm\n"
            + lens_info +
            f"PML xy:      {self.t_pml_xy*1e3:.3f} mm ({self.pml_n_wavelengths_xy:.1f}λ)\n"
            f"PML z:       {self.t_pml_z*1e3:.3f} mm ({self.pml_n_wavelengths_z:.1f}λ)\n"
            f"σ_max:       {self.sigma_max:.2e}  (factor {self.pml_sigma_max_factor})\n"
            f"Mesh:        {self.mesh_nx}×{self.mesh_ny}×{self.mesh_nz}\n"
            f"Top BC:      {self.top_bc_type} (Z_rel={self.top_impedance_Zrel})\n"
        )


# =====================================================================
# Factory presets
# =====================================================================

def fast_mode_config(**overrides) -> FarFieldConfig:
    """
    Return a FarFieldConfig with coarser mesh for quick qualitative runs.

    Uses 4 elem/λ (vs default 6) and a smaller domain.
    WARNING: results are qualitative only — do not use for quantitative analysis.
    """
    import warnings
    warnings.warn("fast_mode_config: 4 elem/λ is qualitative only", stacklevel=2)
    defaults = dict(
        Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
        frequency_hz=2.0e6, disk_radius=1.0e-3,
        elements_per_wavelength=4,
        pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
    )
    defaults.update(overrides)
    return FarFieldConfig(**defaults)


def demo_config(**overrides) -> FarFieldConfig:
    """
    Standard 6×6×4 mm demo config used by farfield_vortex_plus_standing.py.
    """
    defaults = dict(
        Lx=6e-3, Ly=6e-3, H_under=3e-3, H_top=1e-3,
        frequency_hz=2.0e6, disk_radius=1.0e-3,
        disk_velocity_amplitude=10e-6, vortex_topological_charge=1,
        standing_velocity_amplitude=1e-6, standing_phase_pattern="antiphase",
        standing_axis="both", top_bc_type="impedance", top_impedance_Zrel=0.001,
        pml_n_wavelengths_xy=1.0, pml_n_wavelengths_z=1.0,
        pml_degree=2, pml_sigma_max_factor=5.0, pml_enabled=True,
        elements_per_wavelength=5,
        lens_drive="plastic", lens_l=1, lens_focal_length=10e-3,
        lens_focus_offset_x=0.2e-3, lens_focus_offset_y=0.0,
        lens_c_lens=2700.0, lens_apodization="cosine_taper",
    )
    defaults.update(overrides)
    return FarFieldConfig(**defaults)
