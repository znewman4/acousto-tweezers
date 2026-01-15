#!/usr/bin/env python3
"""
Reachability Atlas for 3-Puck Acoustic Tweezers Control.

PHASE 6 IMPLEMENTATION:
Maps the control space to trap positions for 3-transducer configurations.

Two modes:
1. CONTROL SPACE SCAN: Maps control configurations to trap positions
2. MACRO ACTION ATLAS: Maps macro_action → Δ(trap_candidate) with controlled simplification

The macro action atlas:
- Fixes amplitudes (vA, vB, vC)
- Fixes two puck positions
- Varies only ONE parameter at a time (phase or position)
- Records trap displacement per macro action

Usage:
    # Control space scan (original)
    python scripts/reachability_atlas_3puck.py
    python scripts/reachability_atlas_3puck.py --trajectory circle
    
    # Macro action atlas (Phase 6)
    python scripts/reachability_atlas_3puck.py --macro-atlas
    python scripts/reachability_atlas_3puck.py --macro-atlas --sweep-phases
    python scripts/reachability_atlas_3puck.py --macro-atlas --all-sweeps
    python scripts/reachability_atlas_3puck.py --visualize-macros
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import json

from acousto.force import ParticleProps, gorkov_potential_and_force_2d, bilinear_sample_vec
from acousto.analysis import find_trap_center
from tweezers.control import (
    DishDomain, MediumProps, EvaluatorConfig,
    Control3Pucks, Evaluator3Pucks,
)

# Import macro actions
try:
    from scripts.macro_actions_3puck import (
        MacroActionType3Puck, MacroAction3Puck,
        apply_macro_action_3puck,
    )
    MACRO_ACTIONS_AVAILABLE = True
except ImportError:
    MACRO_ACTIONS_AVAILABLE = False


# ============================================================
# PHASE 6: MACRO ACTION ATLAS
# ============================================================

@dataclass
class MacroAtlasConfig:
    """Configuration for macro action atlas generation."""
    # Domain
    Lx: float = 2e-3
    Ly: float = 2e-3
    Nx: int = 80
    Ny: int = 80
    
    # Fixed amplitudes
    fixed_vA: float = 0.08
    fixed_vB: float = 0.08
    fixed_vC: float = 0.08
    
    # Base puck positions
    base_xA: float = 0.4e-3
    base_xB: float = 1.6e-3
    base_xC: float = 1.0e-3
    base_yA: float = 0.05e-3
    base_yB: float = 0.05e-3
    base_yC: float = 0.2e-3
    
    # Sweep ranges
    phase_min: float = -np.pi
    phase_max: float = np.pi
    phase_steps: int = 16
    
    position_min: float = 0.1e-3
    position_max: float = 1.9e-3
    position_steps: int = 12
    
    # Macro action settings
    macro_magnitude: float = 0.05e-3
    macro_phase_step: float = 0.15
    
    # Physics
    alpha_g: float = 2000.0
    
    # Multi-probe grid configuration for surf force evaluation
    # Grid of probe points across domain interior (avoids boundary artifacts)
    probe_grid_n: int = 3  # 3x3 grid = 9 probe points
    probe_margin: float = 0.2e-3  # margin from domain edges (m)
    
    def get_probe_grid(self) -> list[tuple[float, float]]:
        """
        Generate probe points for surf force evaluation.
        
        Returns list of (x, y) probe positions in meters.
        For probe_grid_n=3 and margin=0.2mm, returns 9 points:
            x ∈ {0.2, 1.0, 1.8} mm
            y ∈ {0.2, 1.0, 1.8} mm
        """
        x_min = self.probe_margin
        x_max = self.Lx - self.probe_margin
        y_min = self.probe_margin
        y_max = self.Ly - self.probe_margin
        
        xs = np.linspace(x_min, x_max, self.probe_grid_n)
        ys = np.linspace(y_min, y_max, self.probe_grid_n)
        
        probes = []
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                probes.append((x, y))
        return probes


@dataclass
class MacroAtlasEntry:
    """Single entry in the macro action reachability atlas."""
    action_type: str
    action_id: int
    varied_param: str
    varied_value: float
    
    # Trap candidate (from evaluator - always filled if trap found)
    init_trap_x: float
    init_trap_y: float
    init_stable: bool
    init_stiff_min: float
    init_depth: float
    
    final_trap_x: float
    final_trap_y: float
    final_stable: bool
    final_stiff_min: float
    final_depth: float
    
    delta_trap_x: float
    delta_trap_y: float
    delta_stiff: float
    
    # Proxy trap diagnostics (global argmin U - always available)
    init_proxy_trap_x: float = np.nan
    init_proxy_trap_y: float = np.nan
    init_proxy_trap_U: float = np.nan
    final_proxy_trap_x: float = np.nan
    final_proxy_trap_y: float = np.nan
    final_proxy_trap_U: float = np.nan
    
    # Particle point used for force measurement
    init_particle_x: float = np.nan
    init_particle_y: float = np.nan
    final_particle_x: float = np.nan
    final_particle_y: float = np.nan
    particle_point_source: str = "unknown"  # trap_candidate|proxy_trap|center
    
    # Legacy: Force at particle (at trap - deprecated, force ≈ 0 at equilibrium!)
    init_Fx_p: float = np.nan
    init_Fy_p: float = np.nan
    init_Fmag_p: float = np.nan
    final_Fx_p: float = np.nan
    final_Fy_p: float = np.nan
    final_Fmag_p: float = np.nan
    
    # SURF FORCE at FIXED particle position (USE THIS for surf mode!)
    # This is independent of trap location - measures actual force field
    surf_particle_x: float = np.nan
    surf_particle_y: float = np.nan
    init_Fp_x: float = np.nan
    init_Fp_y: float = np.nan
    init_Fp_mag: float = np.nan
    final_Fp_x: float = np.nan
    final_Fp_y: float = np.nan
    final_Fp_mag: float = np.nan
    
    # Surf directionality (computed from Fp_*, not Fx_p!)
    desired_dir_x: float = 0.0
    desired_dir_y: float = 0.0
    init_Fp_hat_dot_d: float = np.nan
    init_Fp_dot_d: float = np.nan
    final_Fp_hat_dot_d: float = np.nan
    final_Fp_dot_d: float = np.nan
    
    # Legacy surf directionality (deprecated - computed at equilibrium)
    init_Fhat_dot_d: float = np.nan
    init_F_dot_d: float = np.nan
    final_Fhat_dot_d: float = np.nan
    final_F_dot_d: float = np.nan
    
    # Delta particle (NaN if not simulated)
    delta_particle_x: float = np.nan
    delta_particle_y: float = np.nan
    
    # Probe point index (for multi-probe surf atlas)
    probe_id: int = 0


class MacroActionAtlas:
    """
    Phase 6: Geometric exploration of macro action controllability.
    
    Maps: macro_action → Δ(trap_candidate_x, trap_candidate_y)
    
    Controlled simplification:
    - Fix amplitudes
    - Fix two puck positions
    - Vary only ONE parameter at a time
    """
    
    MACRO_ACTIONS = [
        MacroActionType3Puck.TRANSLATE_TRAP_X_POS,
        MacroActionType3Puck.TRANSLATE_TRAP_X_NEG,
        MacroActionType3Puck.TRANSLATE_TRAP_Y_POS,
        MacroActionType3Puck.TRANSLATE_TRAP_Y_NEG,
        MacroActionType3Puck.ROTATE_INTERFERENCE_CW,
        MacroActionType3Puck.ROTATE_INTERFERENCE_CCW,
        MacroActionType3Puck.WIDEN,
        MacroActionType3Puck.NARROW,
        MacroActionType3Puck.PHASE_SHIFT_B_POS,
        MacroActionType3Puck.PHASE_SHIFT_B_NEG,
        MacroActionType3Puck.PHASE_SHIFT_C_POS,
        MacroActionType3Puck.PHASE_SHIFT_C_NEG,
    ] if MACRO_ACTIONS_AVAILABLE else []
    
    def __init__(self, cfg: MacroAtlasConfig):
        self.cfg = cfg
        self.entries: list[MacroAtlasEntry] = []
        
        # Build evaluator
        self.domain = DishDomain(Lx=cfg.Lx, Ly=cfg.Ly, Nx=cfg.Nx, Ny=cfg.Ny)
        self.medium = MediumProps(f=1e6, c0=1500.0, rho0=1000.0)
        self.particle = ParticleProps(a=50e-6, rho_p=1050.0, c_p=1600.0)
        
        eval_cfg = EvaluatorConfig(
            dt=1e-4,
            alpha_g=cfg.alpha_g,
            sigma_x=0.2e-3,
            bottom_band=0.25e-3,
            viscosity=1e-3,
        )
        
        self.ev = Evaluator3Pucks(
            domain=self.domain,
            medium=self.medium,
            particle=self.particle,
            cfg=eval_cfg,
        )
    
    def _get_base_control(self, **overrides) -> Control3Pucks:
        """Get base control with optional overrides."""
        cfg = self.cfg
        params = {
            'xA': cfg.base_xA, 'yA': cfg.base_yA, 'vA': cfg.fixed_vA, 'phiA': 0.0,
            'xB': cfg.base_xB, 'yB': cfg.base_yB, 'vB': cfg.fixed_vB, 'phiB': 0.0,
            'xC': cfg.base_xC, 'yC': cfg.base_yC, 'vC': cfg.fixed_vC, 'phiC': 0.0,
        }
        params.update(overrides)
        return Control3Pucks(**params)
    
    def _measure_trap_extended(self, u: Control3Pucks, fixed_particle_xy: tuple[float, float] | None = None) -> dict:
        """
        Measure trap state for a control configuration with extended metrics.
        
        Returns trap position, stability, proxy trap, and force fields for surf analysis.
        
        Parameters
        ----------
        u : Control3Pucks
            Control configuration to evaluate.
        fixed_particle_xy : tuple[float, float] | None
            If provided, compute surf force at this fixed position (for surf mode).
            If None, defaults to domain center.
        
        Returns dict with:
        - Trap metrics: trap_x, trap_y, stable, stiff_min, depth
        - Proxy trap: proxy_trap_x, proxy_trap_y, proxy_trap_U
        - Surf force at FIXED particle position: Fp_x, Fp_y, Fp_mag (independent of trap!)
        - Legacy: particle_x, particle_y, particle_source, Fx_p, Fy_p, Fmag_p (at trap - deprecated)
        - field_data: contains field, Fx, Fy for multi-probe sampling
        """
        u = self.ev.clip_control(u)
        center_x, center_y = self.cfg.Lx / 2, self.cfg.Ly / 2
        
        # Fixed particle position for surf force evaluation
        # This is INDEPENDENT of trap location - crucial for surf mode!
        if fixed_particle_xy is not None:
            surf_particle_x, surf_particle_y = fixed_particle_xy
        else:
            surf_particle_x, surf_particle_y = center_x, center_y
        
        # Call with return_fields=True to get Fx, Fy
        result = self.ev.step(
            xp=center_x, yp=center_y,
            target_x=center_x, target_y=center_y,
            u=u, return_metrics=True, return_fields=True,
        )
        xp1, yp1, loss, info, field, U, Fx, Fy = result
        
        m = info.get("metrics", {})
        
        # Core trap metrics
        trap_x = m.get("trap_candidate_x", np.nan)
        trap_y = m.get("trap_candidate_y", np.nan)
        stable = m.get("trap_stable", False)
        stiff_min = m.get("stiff_min", np.nan)
        depth = m.get("trap_depth", np.nan)
        
        # Proxy trap (global argmin U)
        proxy_trap_x = m.get("proxy_trap_x", np.nan)
        proxy_trap_y = m.get("proxy_trap_y", np.nan)
        proxy_trap_U = m.get("proxy_trap_U", np.nan)
        
        # ====== SURF FORCE at FIXED particle position ======
        # This is the key metric for surf mode - force at a position
        # that is NOT the trap equilibrium!
        Fp_x, Fp_y = bilinear_sample_vec(field.x, field.y, Fx, Fy, surf_particle_x, surf_particle_y)
        Fp_mag = np.sqrt(Fp_x**2 + Fp_y**2)
        
        # ====== Legacy: particle point at trap (deprecated for surf) ======
        # Kept for backward compatibility but NOT useful for surf planning
        # (force at trap ≈ 0 by definition!)
        if np.isfinite(trap_x) and np.isfinite(trap_y) and stable:
            particle_x, particle_y = trap_x, trap_y
            particle_source = "trap_candidate"
        elif np.isfinite(proxy_trap_x) and np.isfinite(proxy_trap_y):
            particle_x, particle_y = proxy_trap_x, proxy_trap_y
            particle_source = "proxy_trap"
        else:
            particle_x, particle_y = center_x, center_y
            particle_source = "center"
        
        # Legacy force at trap point (deprecated)
        if np.isfinite(particle_x) and np.isfinite(particle_y):
            Fx_p, Fy_p = bilinear_sample_vec(field.x, field.y, Fx, Fy, particle_x, particle_y)
            Fmag_p = np.sqrt(Fx_p**2 + Fy_p**2)
        else:
            Fx_p, Fy_p, Fmag_p = np.nan, np.nan, np.nan
        
        return {
            "trap_x": trap_x,
            "trap_y": trap_y,
            "stable": stable,
            "stiff_min": stiff_min,
            "depth": depth,
            "proxy_trap_x": proxy_trap_x,
            "proxy_trap_y": proxy_trap_y,
            "proxy_trap_U": proxy_trap_U,
            # Surf force at FIXED position (use this for surf mode!)
            "surf_particle_x": surf_particle_x,
            "surf_particle_y": surf_particle_y,
            "Fp_x": Fp_x,
            "Fp_y": Fp_y,
            "Fp_mag": Fp_mag,
            # Legacy: particle at trap (deprecated for surf)
            "particle_x": particle_x,
            "particle_y": particle_y,
            "particle_source": particle_source,
            "Fx_p": Fx_p,
            "Fy_p": Fy_p,
            "Fmag_p": Fmag_p,
            # Field data for multi-probe sampling (new!)
            "field": field,
            "Fx": Fx,
            "Fy": Fy,
        }
    
    def _sample_force_at_probe(self, field, Fx, Fy, probe_x: float, probe_y: float) -> tuple[float, float, float]:
        """Sample force at a probe point using bilinear interpolation.
        
        Returns (Fp_x, Fp_y, Fp_mag) at the probe location.
        """
        Fp_x, Fp_y = bilinear_sample_vec(field.x, field.y, Fx, Fy, probe_x, probe_y)
        Fp_mag = np.sqrt(Fp_x**2 + Fp_y**2)
        return Fp_x, Fp_y, Fp_mag
    
    # Map action types to desired directions for surf metrics
    ACTION_DESIRED_DIRS = {
        "TRANSLATE_TRAP_X_POS": (1.0, 0.0),
        "TRANSLATE_TRAP_X_NEG": (-1.0, 0.0),
        "TRANSLATE_TRAP_Y_POS": (0.0, 1.0),
        "TRANSLATE_TRAP_Y_NEG": (0.0, -1.0),
    }
    
    def _compute_surf_metrics(self, Fx: float, Fy: float, Fmag: float,
                               dx: float, dy: float) -> tuple[float, float]:
        """Compute surf directionality metrics.
        
        Returns (Fhat_dot_d, F_dot_d) where d = (dx, dy) is desired direction.
        """
        eps = 1e-15
        F_dot_d = Fx * dx + Fy * dy
        Fhat_dot_d = F_dot_d / (Fmag + eps) if np.isfinite(Fmag) else np.nan
        return Fhat_dot_d, F_dot_d
    
    def _record_effect(
        self,
        action_type: MacroActionType3Puck,
        action_id: int,
        varied_param: str,
        varied_value: float,
        u_before: Control3Pucks,
        u_after: Control3Pucks,
    ):
        """Record macro action effect with multi-probe surf metrics.
        
        Creates one entry per probe point for spatial surf force sampling.
        Trap-related metrics are shared across all probes (same control configuration).
        """
        # Measure trap state and get force fields (one evaluator call per control config)
        before = self._measure_trap_extended(u_before)
        after = self._measure_trap_extended(u_after)
        
        # Get desired direction for this action type
        action_name = action_type.name
        dx, dy = self.ACTION_DESIRED_DIRS.get(action_name, (0.0, 0.0))
        
        # Legacy surf metrics at trap (deprecated - force ≈ 0 at equilibrium)
        init_Fhat_dot_d, init_F_dot_d = self._compute_surf_metrics(
            before["Fx_p"], before["Fy_p"], before["Fmag_p"], dx, dy
        )
        final_Fhat_dot_d, final_F_dot_d = self._compute_surf_metrics(
            after["Fx_p"], after["Fy_p"], after["Fmag_p"], dx, dy
        )
        
        # Get probe grid for surf force sampling
        probe_grid = self.cfg.get_probe_grid()
        
        # Create one entry per probe point
        for probe_id, (probe_x, probe_y) in enumerate(probe_grid):
            # Sample force at this probe point from the field
            init_Fp_x, init_Fp_y, init_Fp_mag = self._sample_force_at_probe(
                before["field"], before["Fx"], before["Fy"], probe_x, probe_y
            )
            final_Fp_x, final_Fp_y, final_Fp_mag = self._sample_force_at_probe(
                after["field"], after["Fx"], after["Fy"], probe_x, probe_y
            )
            
            # Compute surf directionality at this probe point
            init_Fp_hat_dot_d, init_Fp_dot_d = self._compute_surf_metrics(
                init_Fp_x, init_Fp_y, init_Fp_mag, dx, dy
            )
            final_Fp_hat_dot_d, final_Fp_dot_d = self._compute_surf_metrics(
                final_Fp_x, final_Fp_y, final_Fp_mag, dx, dy
            )
            
            self.entries.append(MacroAtlasEntry(
                action_type=action_name,
                action_id=action_id,
                varied_param=varied_param,
                varied_value=varied_value,
                probe_id=probe_id,
                # Trap metrics (shared across all probes)
                init_trap_x=before["trap_x"],
                init_trap_y=before["trap_y"],
                init_stable=before["stable"],
                init_stiff_min=before["stiff_min"],
                init_depth=before["depth"],
                final_trap_x=after["trap_x"],
                final_trap_y=after["trap_y"],
                final_stable=after["stable"],
                final_stiff_min=after["stiff_min"],
                final_depth=after["depth"],
                delta_trap_x=after["trap_x"] - before["trap_x"],
                delta_trap_y=after["trap_y"] - before["trap_y"],
                delta_stiff=after["stiff_min"] - before["stiff_min"],
                # Proxy trap diagnostics (shared)
                init_proxy_trap_x=before["proxy_trap_x"],
                init_proxy_trap_y=before["proxy_trap_y"],
                init_proxy_trap_U=before["proxy_trap_U"],
                final_proxy_trap_x=after["proxy_trap_x"],
                final_proxy_trap_y=after["proxy_trap_y"],
                final_proxy_trap_U=after["proxy_trap_U"],
                # Legacy particle point (at trap - deprecated)
                init_particle_x=before["particle_x"],
                init_particle_y=before["particle_y"],
                final_particle_x=after["particle_x"],
                final_particle_y=after["particle_y"],
                particle_point_source=before["particle_source"],
                # Legacy force at particle (deprecated - force ≈ 0 at equilibrium)
                init_Fx_p=before["Fx_p"],
                init_Fy_p=before["Fy_p"],
                init_Fmag_p=before["Fmag_p"],
                final_Fx_p=after["Fx_p"],
                final_Fy_p=after["Fy_p"],
                final_Fmag_p=after["Fmag_p"],
                # SURF force at THIS probe point (USE THIS!)
                surf_particle_x=probe_x,
                surf_particle_y=probe_y,
                init_Fp_x=init_Fp_x,
                init_Fp_y=init_Fp_y,
                init_Fp_mag=init_Fp_mag,
                final_Fp_x=final_Fp_x,
                final_Fp_y=final_Fp_y,
                final_Fp_mag=final_Fp_mag,
                # SURF directionality at this probe
                desired_dir_x=dx,
                desired_dir_y=dy,
                init_Fp_hat_dot_d=init_Fp_hat_dot_d,
                init_Fp_dot_d=init_Fp_dot_d,
                final_Fp_hat_dot_d=final_Fp_hat_dot_d,
                final_Fp_dot_d=final_Fp_dot_d,
                # Legacy surf directionality (deprecated)
                init_Fhat_dot_d=init_Fhat_dot_d,
                init_F_dot_d=init_F_dot_d,
                final_Fhat_dot_d=final_Fhat_dot_d,
                final_F_dot_d=final_F_dot_d,
                # Delta particle (not simulated yet)
                delta_particle_x=np.nan,
                delta_particle_y=np.nan,
            ))
    
    def sweep_phase(self, param: str = "phiB"):
        """Sweep one phase parameter."""
        cfg = self.cfg
        values = np.linspace(cfg.phase_min, cfg.phase_max, cfg.phase_steps)
        
        print(f"\n  Sweeping {param} ({len(values)} values)...")
        
        for val in tqdm(values, desc=f"  {param}"):
            u_base = self._get_base_control(**{param: val})
            
            for action_id, action_type in enumerate(self.MACRO_ACTIONS):
                action = MacroAction3Puck(
                    action_type=action_type,
                    magnitude=cfg.macro_magnitude,
                    phase_step=cfg.macro_phase_step,
                )
                u_after = apply_macro_action_3puck(u_base, action)
                
                self._record_effect(
                    action_type, action_id, param, val, u_base, u_after
                )
    
    def sweep_position(self, param: str = "xA"):
        """Sweep one position parameter."""
        cfg = self.cfg
        values = np.linspace(cfg.position_min, cfg.position_max, cfg.position_steps)
        
        print(f"\n  Sweeping {param} ({len(values)} values)...")
        
        for val in tqdm(values, desc=f"  {param}"):
            u_base = self._get_base_control(**{param: val})
            
            for action_id, action_type in enumerate(self.MACRO_ACTIONS):
                action = MacroAction3Puck(
                    action_type=action_type,
                    magnitude=cfg.macro_magnitude,
                    phase_step=cfg.macro_phase_step,
                )
                u_after = apply_macro_action_3puck(u_base, action)
                
                self._record_effect(
                    action_type, action_id, param, val, u_base, u_after
                )
    
    def run_phase_sweeps(self):
        """Sweep phase parameters only."""
        self.sweep_phase("phiB")
        self.sweep_phase("phiC")
    
    def run_position_sweeps(self):
        """Sweep position parameters only."""
        for param in ["xA", "xB", "xC", "yC"]:
            self.sweep_position(param)
    
    def run_all_sweeps(self):
        """Run all parameter sweeps."""
        self.run_phase_sweeps()
        self.run_position_sweeps()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert entries to DataFrame with all surf metrics."""
        return pd.DataFrame([
            {
                # Core identification
                "action_type": e.action_type,
                "action_id": e.action_id,
                "varied_param": e.varied_param,
                "varied_value": e.varied_value,
                "probe_id": e.probe_id,
                # Trap candidate positions
                "init_trap_x": e.init_trap_x,
                "init_trap_y": e.init_trap_y,
                "init_stable": e.init_stable,
                "init_stiff_min": e.init_stiff_min,
                "init_depth": e.init_depth,
                "final_trap_x": e.final_trap_x,
                "final_trap_y": e.final_trap_y,
                "final_stable": e.final_stable,
                "final_stiff_min": e.final_stiff_min,
                "final_depth": e.final_depth,
                "delta_trap_x": e.delta_trap_x,
                "delta_trap_y": e.delta_trap_y,
                "delta_stiff": e.delta_stiff,
                # Proxy trap diagnostics
                "init_proxy_trap_x": e.init_proxy_trap_x,
                "init_proxy_trap_y": e.init_proxy_trap_y,
                "init_proxy_trap_U": e.init_proxy_trap_U,
                "final_proxy_trap_x": e.final_proxy_trap_x,
                "final_proxy_trap_y": e.final_proxy_trap_y,
                "final_proxy_trap_U": e.final_proxy_trap_U,
                # Legacy: Particle point at trap (deprecated)
                "init_particle_x": e.init_particle_x,
                "init_particle_y": e.init_particle_y,
                "final_particle_x": e.final_particle_x,
                "final_particle_y": e.final_particle_y,
                "particle_point_source": e.particle_point_source,
                # Legacy: Force at particle at trap (deprecated)
                "init_Fx_p": e.init_Fx_p,
                "init_Fy_p": e.init_Fy_p,
                "init_Fmag_p": e.init_Fmag_p,
                "final_Fx_p": e.final_Fx_p,
                "final_Fy_p": e.final_Fy_p,
                "final_Fmag_p": e.final_Fmag_p,
                # SURF force at FIXED position (USE THIS!)
                "surf_particle_x": e.surf_particle_x,
                "surf_particle_y": e.surf_particle_y,
                "init_Fp_x": e.init_Fp_x,
                "init_Fp_y": e.init_Fp_y,
                "init_Fp_mag": e.init_Fp_mag,
                "final_Fp_x": e.final_Fp_x,
                "final_Fp_y": e.final_Fp_y,
                "final_Fp_mag": e.final_Fp_mag,
                # SURF directionality from Fp_*
                "desired_dir_x": e.desired_dir_x,
                "desired_dir_y": e.desired_dir_y,
                "init_Fp_hat_dot_d": e.init_Fp_hat_dot_d,
                "init_Fp_dot_d": e.init_Fp_dot_d,
                "final_Fp_hat_dot_d": e.final_Fp_hat_dot_d,
                "final_Fp_dot_d": e.final_Fp_dot_d,
                # Legacy: surf directionality (deprecated)
                "init_Fhat_dot_d": e.init_Fhat_dot_d,
                "init_F_dot_d": e.init_F_dot_d,
                "final_Fhat_dot_d": e.final_Fhat_dot_d,
                "final_F_dot_d": e.final_F_dot_d,
                # Delta particle (placeholder)
                "delta_particle_x": e.delta_particle_x,
                "delta_particle_y": e.delta_particle_y,
            }
            for e in self.entries
        ])
    
    def save(self, output_dir: Path):
        """Save atlas to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.to_dataframe()
        df.to_csv(output_dir / "macro_action_atlas.csv", index=False)
        
        # Save as NPZ for fast loading (key arrays only)
        np.savez(
            output_dir / "macro_action_atlas.npz",
            action_types=df["action_type"].values,
            action_ids=df["action_id"].values,
            varied_params=df["varied_param"].values,
            varied_values=df["varied_value"].values,
            probe_ids=df["probe_id"].values,
            surf_particle_x=df["surf_particle_x"].values,
            surf_particle_y=df["surf_particle_y"].values,
            delta_trap_x=df["delta_trap_x"].values,
            delta_trap_y=df["delta_trap_y"].values,
            delta_stiff=df["delta_stiff"].values,
            init_stable=df["init_stable"].values,
            final_stable=df["final_stable"].values,
            # SURF metrics at probe positions (USE THIS!)
            init_Fp_x=df["init_Fp_x"].values,
            init_Fp_y=df["init_Fp_y"].values,
            init_Fp_hat_dot_d=df["init_Fp_hat_dot_d"].values,
            final_Fp_hat_dot_d=df["final_Fp_hat_dot_d"].values,
            init_Fp_mag=df["init_Fp_mag"].values,
            final_Fp_mag=df["final_Fp_mag"].values,
            # Legacy (deprecated)
            init_Fhat_dot_d=df["init_Fhat_dot_d"].values,
            final_Fhat_dot_d=df["final_Fhat_dot_d"].values,
        )
        
        print(f"\nSaved macro action atlas:")
        print(f"  CSV: {output_dir / 'macro_action_atlas.csv'}")
        print(f"  NPZ: {output_dir / 'macro_action_atlas.npz'}")
        
        # ======== VALIDATION OUTPUT ========
        self._print_validation_summary(df)
    
    def _print_validation_summary(self, df: pd.DataFrame):
        """Print built-in validation output for multi-probe surf atlas."""
        print("\n" + "="*60)
        print("MULTI-PROBE SURF ATLAS VALIDATION")
        print("="*60)
        
        # Basic counts
        n_rows = len(df)
        n_unique_probes = df[['surf_particle_x', 'surf_particle_y']].drop_duplicates().shape[0]
        print(f"\nTotal rows: {n_rows}")
        print(f"Unique probe points: {n_unique_probes}")
        
        # Probe positions
        probe_x_mm = df['surf_particle_x'].unique() * 1e3
        probe_y_mm = df['surf_particle_y'].unique() * 1e3
        print(f"Probe X positions (mm): {np.sort(probe_x_mm)}")
        print(f"Probe Y positions (mm): {np.sort(probe_y_mm)}")
        
        # Force magnitude stats
        fp_mag = df['init_Fp_mag']
        fp_mag_valid = fp_mag[np.isfinite(fp_mag)]
        if len(fp_mag_valid) > 0:
            print(f"\ninit_Fp_mag: min={fp_mag_valid.min():.2e}, mean={fp_mag_valid.mean():.2e}, max={fp_mag_valid.max():.2e}")
        
        # Alignment stats
        fp_hat_dot_d = df['init_Fp_hat_dot_d']
        fp_hat_valid = fp_hat_dot_d[np.isfinite(fp_hat_dot_d)]
        if len(fp_hat_valid) > 0:
            print(f"init_Fp_hat_dot_d: min={fp_hat_valid.min():.3f}, mean={fp_hat_valid.mean():.3f}, max={fp_hat_valid.max():.3f}")
        
        # Per action_type summary
        print("\n--- Per-Action Surf Alignment Summary ---")
        print(f"{'Action Type':<30} {'mean(Fp_hat_dot_d)':>18} {'% probes > 0.5':>15}")
        print("-" * 65)
        
        for action_type in sorted(df['action_type'].unique()):
            df_act = df[df['action_type'] == action_type]
            fhat = df_act['init_Fp_hat_dot_d']
            fhat_valid = fhat[np.isfinite(fhat)]
            
            if len(fhat_valid) > 0:
                mean_align = fhat_valid.mean()
                pct_high = 100 * (fhat_valid > 0.5).sum() / len(fhat_valid)
                print(f"{action_type:<30} {mean_align:>+18.3f} {pct_high:>14.1f}%")
            else:
                print(f"{action_type:<30} {'N/A':>18} {'N/A':>15}")
        
        print("="*60)


def visualize_macro_atlas(df: pd.DataFrame, output_dir: Path):
    """Create visualizations for macro action atlas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid entries
    df_valid = df[df["delta_trap_x"].notna() & df["delta_trap_y"].notna()].copy()
    print(f"Valid entries: {len(df_valid)}/{len(df)} ({100*len(df_valid)/max(1,len(df)):.1f}%)")
    
    if len(df_valid) == 0:
        print("No valid entries to visualize!")
        return
    
    action_types = df_valid["action_type"].unique()
    
    # ===== Vector field by action type =====
    n_actions = min(12, len(action_types))
    rows = (n_actions + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if n_actions == 1 else axes
    
    for idx, action_type in enumerate(action_types[:n_actions]):
        ax = axes[idx]
        df_act = df_valid[df_valid["action_type"] == action_type]
        
        stable = df_act["final_stable"].values
        
        # Stable (green)
        df_s = df_act[stable]
        if len(df_s) > 0:
            ax.quiver(
                df_s["init_trap_x"] * 1e3, df_s["init_trap_y"] * 1e3,
                df_s["delta_trap_x"] * 1e3, df_s["delta_trap_y"] * 1e3,
                angles='xy', scale_units='xy', scale=1,
                color='green', alpha=0.6, width=0.004,
            )
        
        # Unstable (red)
        df_u = df_act[~stable]
        if len(df_u) > 0:
            ax.quiver(
                df_u["init_trap_x"] * 1e3, df_u["init_trap_y"] * 1e3,
                df_u["delta_trap_x"] * 1e3, df_u["delta_trap_y"] * 1e3,
                angles='xy', scale_units='xy', scale=1,
                color='red', alpha=0.4, width=0.003,
            )
        
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.set_title(action_type.replace("_", "\n"), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(n_actions, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Trap Displacement Vectors by Macro Action\n(Green=Stable, Red=Unstable)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "macro_vector_field.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'macro_vector_field.png'}")
    
    # ===== Mean displacement bar chart =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Displacement magnitude
    df_valid["delta_mag"] = np.sqrt(
        df_valid["delta_trap_x"]**2 + df_valid["delta_trap_y"]**2
    ) * 1e6  # µm
    
    ax = axes[0]
    mean_disp = df_valid.groupby("action_type")["delta_mag"].mean().sort_values(ascending=True)
    colors = ['green' if v > 5 else 'orange' if v > 1 else 'red' for v in mean_disp.values]
    mean_disp.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Mean displacement (µm)")
    ax.set_title("Mean Trap Displacement by Macro Action")
    ax.grid(True, alpha=0.3, axis='x')
    
    # Stability rate
    ax = axes[1]
    stab_rate = df_valid.groupby("action_type")["final_stable"].mean().sort_values(ascending=True)
    colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in stab_rate.values]
    stab_rate.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Stability Rate")
    ax.set_xlim(0, 1)
    ax.set_title("Final Trap Stability Rate by Macro Action")
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / "macro_effectiveness.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'macro_effectiveness.png'}")
    
    # ===== Directional bias scatter =====
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    translate_actions = [
        ("TRANSLATE_TRAP_X_POS", (1, 0)),
        ("TRANSLATE_TRAP_X_NEG", (-1, 0)),
        ("TRANSLATE_TRAP_Y_POS", (0, 1)),
        ("TRANSLATE_TRAP_Y_NEG", (0, -1)),
        ("WIDEN", (0, 0)),
        ("NARROW", (0, 0)),
    ]
    
    for idx, (action_type, expected) in enumerate(translate_actions):
        ax = axes[idx // 3, idx % 3]
        df_act = df_valid[df_valid["action_type"] == action_type]
        
        if len(df_act) == 0:
            ax.set_title(f"{action_type}\n(no data)")
            continue
        
        sc = ax.scatter(
            df_act["delta_trap_x"] * 1e6,
            df_act["delta_trap_y"] * 1e6,
            c=df_act["final_stable"].astype(float),
            cmap='RdYlGn', alpha=0.6, s=20,
        )
        
        if expected != (0, 0):
            ax.arrow(0, 0, expected[0] * 20, expected[1] * 20,
                    head_width=3, head_length=2, fc='blue', ec='blue', lw=2)
        
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel("Δx (µm)")
        ax.set_ylabel("Δy (µm)")
        ax.set_title(action_type.replace("_", " "))
        ax.set_aspect('equal')
    
    plt.suptitle("Directional Bias (Blue arrow = expected direction)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "macro_directional_bias.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'macro_directional_bias.png'}")
    
    # ===== Summary statistics =====
    print("\n=== MACRO ACTION ATLAS SUMMARY ===")
    print(f"Total entries: {len(df)}")
    print(f"Valid entries: {len(df_valid)}")
    print(f"\nMean displacement by action (µm):")
    for action_type in action_types:
        df_act = df_valid[df_valid["action_type"] == action_type]
        mean_dx = df_act["delta_trap_x"].mean() * 1e6
        mean_dy = df_act["delta_trap_y"].mean() * 1e6
        stab = df_act["final_stable"].mean()
        print(f"  {action_type:30s}: Δx={mean_dx:+6.1f}, Δy={mean_dy:+6.1f}, stable={stab:.1%}")


# ============================================================
# ORIGINAL CONTROL SPACE SCAN (kept for compatibility)
# ============================================================

@dataclass
class TrapPoint3Puck:
    """A reachable trap position with 3-puck control metadata."""
    trap_x: float  # m
    trap_y: float  # m
    stiffness_min: float  # min eigenvalue
    stiffness_max: float
    is_stable: bool
    
    # Control parameters (3 pucks)
    xA: float
    yA: float
    vA: float
    phiA: float
    xB: float
    yB: float
    vB: float
    phiB: float
    xC: float
    yC: float
    vC: float
    phiC: float
    
    def control_centroid_x(self) -> float:
        """X centroid of transducer positions."""
        return (self.xA + self.xB + self.xC) / 3
    
    def control_centroid_y(self) -> float:
        """Y centroid of transducer positions."""
        return (self.yA + self.yB + self.yC) / 3


def generate_structured_controls(
    domain: DishDomain,
    cfg: EvaluatorConfig,
    *,
    n_positions: int = 8,
    n_phases: int = 6,
    n_spreads: int = 4,
    v_amp: float = 0.08,
) -> list[Control3Pucks]:
    """
    Generate structured control configurations for reachability scanning.
    
    Instead of random sampling, uses:
    - Systematic transducer position grid
    - Meaningful phase combinations
    - Different transducer spreads
    """
    Lx = domain.Lx
    Ly = domain.Ly
    y_band = cfg.bottom_band
    
    controls: list[Control3Pucks] = []
    
    margin = 0.1e-3
    
    # Position ranges
    x_range = np.linspace(margin, Lx - margin, n_positions)
    y_bottom = np.linspace(margin, y_band, 3)
    y_C_range = np.linspace(0.05 * Ly, 0.25 * Ly, 3)  # C can be higher
    
    # Phase combinations (constructive/destructive interference)
    phase_combos = [
        (0, np.pi, np.pi/2),      # Symmetric interference
        (0, np.pi, 0),             # A and C in phase
        (0, np.pi, np.pi),         # B and C in phase
        (0, 0, np.pi),             # A and B in phase
        (0, np.pi/2, np.pi),       # Mixed 1
        (np.pi/2, np.pi, 0),       # Mixed 2
    ]
    
    # Spread configurations
    spreads = [
        (0.25, 0.75, 0.50),  # Wide: A left, B right, C center
        (0.35, 0.65, 0.50),  # Medium
        (0.30, 0.70, 0.35),  # Asymmetric C left
        (0.30, 0.70, 0.65),  # Asymmetric C right
    ]
    
    for spread in spreads:
        for (phiA, phiB, phiC) in phase_combos:
            for y_C in y_C_range:
                for y_AB in y_bottom:
                    # Create control configuration
                    u = Control3Pucks(
                        xA=spread[0] * Lx,
                        yA=y_AB,
                        vA=v_amp,
                        phiA=phiA,
                        xB=spread[1] * Lx,
                        yB=y_AB,
                        vB=v_amp,
                        phiB=phiB,
                        xC=spread[2] * Lx,
                        yC=y_C,
                        vC=v_amp,
                        phiC=phiC,
                    )
                    controls.append(u)
    
    # Also add some parametric variations
    for center_frac in np.linspace(0.25, 0.75, 5):
        for spread_frac in [0.2, 0.3, 0.4]:
            for y_C in y_C_range:
                center = center_frac * Lx
                for (phiA, phiB, phiC) in phase_combos[:3]:  # Just first 3
                    u = Control3Pucks(
                        xA=max(margin, center - spread_frac * Lx),
                        yA=0.03 * Ly,
                        vA=v_amp,
                        phiA=phiA,
                        xB=min(Lx - margin, center + spread_frac * Lx),
                        yB=0.03 * Ly,
                        vB=v_amp,
                        phiB=phiB,
                        xC=center,
                        yC=y_C,
                        vC=v_amp,
                        phiC=phiC,
                    )
                    controls.append(u)
    
    return controls


def scan_reachability_3puck(
    ev: Evaluator3Pucks,
    particle: ParticleProps,
    controls: list[Control3Pucks],
    *,
    verbose: bool = True,
) -> list[TrapPoint3Puck]:
    """
    Scan control configurations and find trap positions.
    
    For each control:
    1. Solve Helmholtz
    2. Compute Gor'kov potential and force
    3. Find trap center
    4. Record stability metrics
    """
    domain = ev.domain
    Lx = domain.Lx
    Ly = domain.Ly
    margin = 0.1e-3
    
    results: list[TrapPoint3Puck] = []
    
    iterator = tqdm(controls, desc="Scanning") if verbose else controls
    
    for u in iterator:
        try:
            # Compute field
            vb = ev.control_to_forcing_band_vb(u)
            field = ev.op.solve_for_bottom_vb(vb)
            U, Fx, Fy = gorkov_potential_and_force_2d(field, particle)
            
            # Find trap (search near centroid of transducers)
            search_x = (u.xA + u.xB + u.xC) / 3
            search_y = Ly / 2
            
            trap_result = find_trap_center(
                field.x, field.y, U, Fx, Fy,
                particle_x=search_x, particle_y=search_y,
                search_radius=0.6e-3,
            )
            
            # Skip boundary traps
            if trap_result.x < margin or trap_result.x > Lx - margin:
                continue
            if trap_result.y < margin or trap_result.y > Ly - margin:
                continue
            
            eigvals = trap_result.stiffness_eigvals
            stiff_min = float(np.min(eigvals))
            stiff_max = float(np.max(eigvals))
            
            results.append(TrapPoint3Puck(
                trap_x=trap_result.x,
                trap_y=trap_result.y,
                stiffness_min=stiff_min,
                stiffness_max=stiff_max,
                is_stable=trap_result.is_stable,
                xA=u.xA, yA=u.yA, vA=u.vA, phiA=u.phiA,
                xB=u.xB, yB=u.yB, vB=u.vB, phiB=u.phiB,
                xC=u.xC, yC=u.yC, vC=u.vC, phiC=u.phiC,
            ))
            
        except Exception:
            continue
    
    return results


def generate_trajectory(
    trajectory_type: str,
    Lx: float,
    Ly: float,
    n_points: int = 100,
    scale: float = 0.6,
) -> np.ndarray:
    """Generate target trajectory for reachability analysis."""
    if trajectory_type == "circle":
        radius = scale * min(Lx, Ly) / 2
        center_x = 0.5 * Lx
        center_y = 0.55 * Ly
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        return np.column_stack([x, y])
    
    elif trajectory_type == "sweep_x":
        x = np.linspace(0.2 * Lx, 0.8 * Lx, n_points)
        y = np.full(n_points, 0.55 * Ly)
        return np.column_stack([x, y])
    
    elif trajectory_type == "sweep_y":
        x = np.full(n_points, 0.5 * Lx)
        y = np.linspace(0.25 * Ly, 0.75 * Ly, n_points)
        return np.column_stack([x, y])
    
    elif trajectory_type == "figure8":
        t = np.linspace(0, 2 * np.pi, n_points)
        radius = scale * min(Lx, Ly) / 3
        x = 0.5 * Lx + radius * np.sin(t)
        y = 0.5 * Ly + radius * np.sin(2 * t) / 2
        return np.column_stack([x, y])
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")


def compute_reachability_fraction(
    trap_points: list[TrapPoint3Puck],
    trajectory: np.ndarray,
    tolerance: float = 0.1e-3,
    require_stable: bool = False,
) -> tuple[float, np.ndarray, list[int]]:
    """
    Compute reachability of trajectory.
    
    Returns:
    - fraction: fraction of points reachable
    - reachable_mask: boolean mask
    - nearest_trap_indices: index of nearest trap for each point
    """
    if require_stable:
        trap_positions = np.array([
            [tp.trap_x, tp.trap_y] for tp in trap_points if tp.is_stable
        ])
        trap_indices = [i for i, tp in enumerate(trap_points) if tp.is_stable]
    else:
        trap_positions = np.array([
            [tp.trap_x, tp.trap_y] for tp in trap_points
        ])
        trap_indices = list(range(len(trap_points)))
    
    if len(trap_positions) == 0:
        return 0.0, np.zeros(len(trajectory), dtype=bool), []
    
    reachable_mask = np.zeros(len(trajectory), dtype=bool)
    nearest_indices: list[int] = []
    
    for i, (tx, ty) in enumerate(trajectory):
        distances = np.sqrt((trap_positions[:, 0] - tx)**2 + 
                           (trap_positions[:, 1] - ty)**2)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        if min_dist < tolerance:
            reachable_mask[i] = True
        nearest_indices.append(trap_indices[min_idx])
    
    fraction = np.mean(reachable_mask)
    return fraction, reachable_mask, nearest_indices


def plot_reachability_atlas_3puck(
    trap_points: list[TrapPoint3Puck],
    trajectory: np.ndarray | None,
    output_path: Path,
    domain: DishDomain,
):
    """Create comprehensive reachability visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    Lx = domain.Lx * 1e3
    Ly = domain.Ly * 1e3
    
    # === Panel 1: All traps colored by stiffness ===
    ax = axes[0, 0]
    
    trap_x = np.array([tp.trap_x for tp in trap_points]) * 1e3
    trap_y = np.array([tp.trap_y for tp in trap_points]) * 1e3
    stiffness = np.array([tp.stiffness_min for tp in trap_points])
    is_stable = np.array([tp.is_stable for tp in trap_points])
    
    # Color by log stiffness
    stiff_mag = np.abs(stiffness) + 1e-20
    colors = np.log10(stiff_mag)
    
    sc = ax.scatter(trap_x, trap_y, c=colors, s=15, alpha=0.5, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='log₁₀|stiffness|')
    
    # Mark stable traps
    stable_mask = is_stable
    ax.scatter(trap_x[stable_mask], trap_y[stable_mask], s=30, marker='x', 
               c='red', alpha=0.7, label=f'stable ({np.sum(stable_mask)})')
    
    if trajectory is not None:
        ax.plot(trajectory[:, 0] * 1e3, trajectory[:, 1] * 1e3, 'w-', lw=2, label='trajectory')
        ax.plot(trajectory[:, 0] * 1e3, trajectory[:, 1] * 1e3, 'k--', lw=1)
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Reachability Atlas: {len(trap_points)} trap positions')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # === Panel 2: Reachability along trajectory ===
    ax = axes[0, 1]
    
    if trajectory is not None:
        frac_any, mask_any, _ = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3, require_stable=False
        )
        frac_stable, mask_stable, _ = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3, require_stable=True
        )
        
        traj_x = trajectory[:, 0] * 1e3
        traj_y = trajectory[:, 1] * 1e3
        
        # Color by reachability
        for i in range(len(trajectory) - 1):
            if mask_stable[i]:
                color = 'green'
            elif mask_any[i]:
                color = 'orange'
            else:
                color = 'red'
            ax.plot(traj_x[i:i+2], traj_y[i:i+2], color=color, lw=3)
        
        ax.set_title(f'Trajectory Coverage\n'
                    f'Any: {frac_any*100:.1f}% | Stable: {frac_stable*100:.1f}%')
    else:
        ax.text(0.5, 0.5, 'No trajectory', transform=ax.transAxes, ha='center')
        ax.set_title('Trajectory Coverage')
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # === Panel 3: Control centroid vs trap position ===
    ax = axes[1, 0]
    
    ctrl_cx = np.array([tp.control_centroid_x() for tp in trap_points]) * 1e3
    ctrl_cy = np.array([tp.control_centroid_y() for tp in trap_points]) * 1e3
    
    ax.scatter(ctrl_cx, trap_x, c=trap_y, s=10, alpha=0.3, cmap='viridis')
    ax.plot([0, Lx], [0, Lx], 'k--', alpha=0.5, label='ideal (trap=center)')
    ax.set_xlabel('Control centroid x (mm)')
    ax.set_ylabel('Trap x (mm)')
    ax.set_title('Trap X vs Control Centroid X')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Panel 4: Phase influence on trap Y ===
    ax = axes[1, 1]
    
    phase_diff_AB = np.array([tp.phiB - tp.phiA for tp in trap_points])
    phase_diff_AC = np.array([tp.phiC - tp.phiA for tp in trap_points])
    
    sc = ax.scatter(phase_diff_AB, trap_y, c=phase_diff_AC, s=10, alpha=0.3, cmap='hsv')
    plt.colorbar(sc, ax=ax, label='φC - φA (rad)')
    ax.set_xlabel('φB - φA (rad)')
    ax.set_ylabel('Trap y (mm)')
    ax.set_title('Trap Y vs Phase Differences')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "reachability_atlas_3puck.png", dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'reachability_atlas_3puck.png'}")


def save_atlas_data(
    trap_points: list[TrapPoint3Puck],
    output_path: Path,
):
    """Save reachability data for use by surrogate model."""
    # NumPy arrays for trap positions
    trap_xy = np.array([[tp.trap_x, tp.trap_y] for tp in trap_points])
    np.save(output_path / "trap_positions.npy", trap_xy)
    
    # Control configurations
    controls = np.array([
        [tp.xA, tp.yA, tp.vA, tp.phiA,
         tp.xB, tp.yB, tp.vB, tp.phiB,
         tp.xC, tp.yC, tp.vC, tp.phiC]
        for tp in trap_points
    ])
    np.save(output_path / "control_configs.npy", controls)
    
    # Stiffness data
    stiffness = np.array([[tp.stiffness_min, tp.stiffness_max, tp.is_stable] 
                          for tp in trap_points])
    np.save(output_path / "stiffness_data.npy", stiffness)
    
    print(f"Saved atlas data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="3-Puck Reachability Atlas")
    parser.add_argument("--output", type=str, default="results/reachability_3puck")
    parser.add_argument("--trajectory", type=str, default="circle",
                        choices=["circle", "sweep_x", "sweep_y", "figure8", "none"])
    parser.add_argument("--fine", action="store_true", help="Fine-grained scan")
    
    # Phase 6: Macro action atlas options
    parser.add_argument("--macro-atlas", action="store_true",
                       help="Generate macro action reachability atlas (Phase 6)")
    parser.add_argument("--sweep-phases", action="store_true",
                       help="Sweep phase parameters only (with --macro-atlas)")
    parser.add_argument("--sweep-positions", action="store_true",
                       help="Sweep position parameters only (with --macro-atlas)")
    parser.add_argument("--all-sweeps", action="store_true",
                       help="Run all sweeps (with --macro-atlas)")
    parser.add_argument("--visualize-macros", type=str, default=None,
                       help="Visualize existing macro atlas from directory")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode with fewer samples")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ===== Phase 6: Macro Action Atlas =====
    if args.visualize_macros:
        atlas_file = Path(args.visualize_macros) / "macro_action_atlas.csv"
        if not atlas_file.exists():
            print(f"Atlas not found: {atlas_file}")
            return
        df = pd.read_csv(atlas_file)
        visualize_macro_atlas(df, Path(args.visualize_macros))
        return
    
    if args.macro_atlas:
        if not MACRO_ACTIONS_AVAILABLE:
            print("ERROR: macro_actions_3puck module not available")
            return
        
        print("=" * 70)
        print("PHASE 6: MACRO ACTION REACHABILITY ATLAS")
        print("=" * 70)
        
        cfg = MacroAtlasConfig()
        if args.quick:
            cfg.phase_steps = 8
            cfg.position_steps = 6
        
        print(f"\nConfiguration:")
        print(f"  Domain: {cfg.Lx*1e3:.1f} x {cfg.Ly*1e3:.1f} mm")
        print(f"  Grid: {cfg.Nx} x {cfg.Ny}")
        print(f"  Phase steps: {cfg.phase_steps}")
        print(f"  Position steps: {cfg.position_steps}")
        print(f"  Macro actions: {len(MacroActionAtlas.MACRO_ACTIONS)}")
        print(f"  Output: {output_path}")
        
        atlas = MacroActionAtlas(cfg)
        
        if args.sweep_phases:
            atlas.run_phase_sweeps()
        elif args.sweep_positions:
            atlas.run_position_sweeps()
        elif args.all_sweeps:
            atlas.run_all_sweeps()
        else:
            # Default: phase sweeps only (faster)
            atlas.run_phase_sweeps()
        
        atlas.save(output_path)
        
        df = atlas.to_dataframe()
        visualize_macro_atlas(df, output_path)
        
        print("\n" + "=" * 70)
        print("MACRO ACTION ATLAS COMPLETE")
        print("=" * 70)
        return
    
    # ===== Original Control Space Scan =====
    print("=" * 70)
    print("3-PUCK REACHABILITY ATLAS SCANNER")
    print("=" * 70)
    
    # Setup
    domain = DishDomain(Lx=2e-3, Ly=2e-3, Nx=100, Ny=100)
    medium = MediumProps(f=2e6, c0=1500.0, rho0=1000.0, loss_eta=1e-3, kz=0.0, coupling_alpha=1.0)
    particle = ParticleProps(a=5e-6, rho_p=1050.0, c_p=2350.0)
    
    cfg = EvaluatorConfig(
        sigma_x=0.10e-3,
        sigma_y=0.15e-3,
        bottom_band=0.25e-3,
        dt=5e-3,
        viscosity=1e-3,
        alpha_g=2e3,
        max_step=0.08e-3,
        use_2d_forcing=True,
    )
    
    ev = Evaluator3Pucks(domain, medium, particle, cfg)
    
    # Generate structured controls
    print("\nGenerating structured control configurations...")
    controls = generate_structured_controls(
        domain, cfg,
        n_positions=10 if args.fine else 6,
        n_phases=8 if args.fine else 5,
        n_spreads=6 if args.fine else 4,
    )
    print(f"  Generated {len(controls)} control configurations")
    
    # Scan reachability
    print("\nScanning reachability...")
    trap_points = scan_reachability_3puck(ev, particle, controls)
    
    print(f"\nFound {len(trap_points)} trap positions")
    n_stable = sum(1 for tp in trap_points if tp.is_stable)
    print(f"  Stable: {n_stable} ({100*n_stable/max(1,len(trap_points)):.1f}%)")
    
    # Generate trajectory
    trajectory = None
    if args.trajectory != "none":
        trajectory = generate_trajectory(args.trajectory, domain.Lx, domain.Ly, scale=0.6)
        
        frac_any, _, _ = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3, require_stable=False
        )
        frac_stable, _, _ = compute_reachability_fraction(
            trap_points, trajectory, tolerance=0.1e-3, require_stable=True
        )
        
        print(f"\nTrajectory: {args.trajectory}")
        print(f"  Reachability (any trap):    {100*frac_any:.1f}%")
        print(f"  Reachability (stable only): {100*frac_stable:.1f}%")
    
    # Analyze trap distribution
    trap_x = np.array([tp.trap_x for tp in trap_points])
    trap_y = np.array([tp.trap_y for tp in trap_points])
    
    print(f"\nTrap position statistics:")
    print(f"  X range: [{trap_x.min()*1e3:.3f}, {trap_x.max()*1e3:.3f}] mm")
    print(f"  Y range: [{trap_y.min()*1e3:.3f}, {trap_y.max()*1e3:.3f}] mm")
    print(f"  X std: {trap_x.std()*1e3:.3f} mm")
    print(f"  Y std: {trap_y.std()*1e3:.3f} mm")
    
    # Save results
    save_atlas_data(trap_points, output_path)
    
    # Save summary
    summary = {
        "n_controls_scanned": len(controls),
        "n_traps_found": len(trap_points),
        "n_stable": n_stable,
        "trajectory_type": args.trajectory,
        "trap_x_range_mm": [float(trap_x.min()*1e3), float(trap_x.max()*1e3)],
        "trap_y_range_mm": [float(trap_y.min()*1e3), float(trap_y.max()*1e3)],
    }
    if trajectory is not None:
        summary["reachability_any"] = float(frac_any)
        summary["reachability_stable"] = float(frac_stable)
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_path / 'summary.json'}")
    
    # Create plots
    plot_reachability_atlas_3puck(trap_points, trajectory, output_path, domain)
    
    print("\n" + "=" * 70)
    print("REACHABILITY ATLAS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
