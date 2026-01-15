"""
Flight Recorder: Per-step diagnostic logging for acoustic tweezers control.

Records comprehensive metrics for every simulation step to enable post-hoc
debugging of control failures, flat frames, and trap instabilities.

Usage:
    recorder = FlightRecorder(out_dir=Path("results/run_001"), enabled=True)
    
    for step in range(n_steps):
        # ... simulation step ...
        recorder.record_step(
            step_idx=step,
            control=ctrl,
            particle_xy=(px, py),
            target_xy=(tx, ty),
            trap_xy=(trap_x, trap_y),
            info=controller_info,
            fields={"p": p, "U": U, "Fx": Fx, "Fy": Fy} if save_fields else None
        )
    
    recorder.finalize()
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, Union
import numpy as np


@dataclass
class StepMetrics:
    """Standard scalar metrics recorded every step."""
    
    step_idx: int
    timestamp: str = ""
    
    # Particle state
    particle_x: float = np.nan
    particle_y: float = np.nan
    target_x: float = np.nan
    target_y: float = np.nan
    tracking_error: float = np.nan
    
    # Trap metrics
    trap_x: float = np.nan
    trap_y: float = np.nan
    trap_found: bool = False
    trap_stable: bool = False
    stiff_eig_1: float = np.nan
    stiff_eig_2: float = np.nan
    stiff_min: float = np.nan
    trap_depth: float = np.nan
    
    # Trap candidate (always set when minimum found, regardless of stability)
    trap_candidate_x: float = np.nan
    trap_candidate_y: float = np.nan
    trap_candidate_depth: float = np.nan
    
    # Proxy trap (global argmin of U - always available for planning)
    proxy_trap_x: float = np.nan
    proxy_trap_y: float = np.nan
    proxy_trap_U: float = np.nan
    
    # Hessian diagnostics at trap
    U_hess_xx: float = np.nan
    U_hess_xy: float = np.nan
    U_hess_yy: float = np.nan
    trap_grad_norm: float = np.nan
    
    # Pressure field metrics
    p_min: float = np.nan
    p_max: float = np.nan
    p_mean: float = np.nan
    p_std: float = np.nan
    p_nan_frac: float = 0.0
    p_inf_frac: float = 0.0
    
    # Potential field metrics
    U_min: float = np.nan
    U_max: float = np.nan
    U_ptp: float = np.nan  # peak-to-peak
    U_std: float = np.nan
    U_nan_frac: float = 0.0
    U_inf_frac: float = 0.0
    
    # Force metrics
    Fx_max: float = np.nan
    Fy_max: float = np.nan
    Fmag_max: float = np.nan
    Fx_mean: float = np.nan
    Fy_mean: float = np.nan
    
    # Rendering flags
    render_flat_flag: bool = False
    render_nan_flag: bool = False
    U_range_tiny: bool = False
    
    # Control state (3 pucks)
    ctrl_xA: float = np.nan
    ctrl_yA: float = np.nan
    ctrl_vA: float = np.nan
    ctrl_phiA: float = np.nan
    ctrl_xB: float = np.nan
    ctrl_yB: float = np.nan
    ctrl_vB: float = np.nan
    ctrl_phiB: float = np.nan
    ctrl_xC: float = np.nan
    ctrl_yC: float = np.nan
    ctrl_vC: float = np.nan
    ctrl_phiC: float = np.nan
    
    # Control deltas (unweighted)
    delta_u_norm: float = np.nan
    delta_delta_u_norm: float = np.nan  # jitter
    
    # Control deltas (weighted for GP training)
    # L_scale=1e-3, V_scale=0.08, Phi_scale=π
    delta_u_norm_w: float = np.nan
    delta_delta_u_norm_w: float = np.nan
    
    # Bounds/rate limit flags
    bounds_clipped_x: bool = False
    bounds_clipped_y: bool = False
    bounds_clipped_v: bool = False
    rate_limited: bool = False
    max_step_clipped: bool = False
    
    # MPC/candidate stats
    candidate_cost_min: float = np.nan
    candidate_cost_max: float = np.nan
    candidate_cost_mean: float = np.nan
    candidate_cost_std: float = np.nan
    chosen_candidate_idx: int = -1
    n_candidates: int = 0
    
    # Solver stats
    solver_time_ms: float = np.nan
    solver_residual: float = np.nan
    
    # Controller mode
    control_mode: str = ""
    
    # Forcing stats
    forcing_vb_min: float = np.nan
    forcing_vb_max: float = np.nan
    forcing_vb_mean: float = np.nan
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/CSV."""
        return asdict(self)


class FlightRecorder:
    """
    Per-step diagnostic recorder for acoustic tweezers simulation.
    
    Writes:
    - steps.csv: One row per step with scalar metrics
    - step_XXXX.json: Full info dict for each step
    - step_XXXX.npz: Field arrays (optional, for debugging flat frames)
    - summary.json: Run summary with aggregate stats
    """
    
    def __init__(
        self,
        out_dir: Path,
        enabled: bool = True,
        stride: int = 1,
        save_fields_stride: int = 0,  # 0 = never, >0 = every N steps
        save_fields_on_flat: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            out_dir: Directory to write diagnostic files.
            enabled: If False, all recording is skipped.
            stride: Record every N steps to CSV/JSON.
            save_fields_stride: Save .npz every N steps (0=never).
            save_fields_on_flat: Always save .npz when flat frame detected.
            verbose: Print progress messages.
        """
        self.out_dir = Path(out_dir)
        self.enabled = enabled
        self.stride = stride
        self.save_fields_stride = save_fields_stride
        self.save_fields_on_flat = save_fields_on_flat
        self.verbose = verbose
        
        self._csv_file = None
        self._csv_writer = None
        self._step_count = 0
        self._flat_count = 0
        self._nan_count = 0
        self._metrics_history: list[StepMetrics] = []
        self._prev_ctrl_arr: Optional[np.ndarray] = None
        self._prev_delta_u: Optional[np.ndarray] = None
        self._run_start_time: Optional[datetime] = None
        
        if self.enabled:
            self._setup()
    
    def _setup(self):
        """Create output directory and initialize files."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self._run_start_time = datetime.now()
        
        # Create CSV file with header
        csv_path = self.out_dir / "steps.csv"
        self._csv_file = open(csv_path, "w", newline="")
        
        # Get field names from StepMetrics
        sample = StepMetrics(step_idx=0)
        fieldnames = list(sample.to_dict().keys())
        
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()
        
        if self.verbose:
            print(f"[FlightRecorder] Recording to: {self.out_dir}")
    
    def _extract_control_array(self, control) -> np.ndarray:
        """Extract control as numpy array from Control3Pucks or similar."""
        if hasattr(control, "xA"):
            # Control3Pucks
            return np.array([
                control.xA, control.yA, control.vA, control.phiA,
                control.xB, control.yB, control.vB, control.phiB,
                control.xC, control.yC, control.vC, control.phiC,
            ])
        elif isinstance(control, np.ndarray):
            return control
        elif hasattr(control, "__iter__"):
            return np.array(list(control))
        else:
            return np.array([np.nan] * 12)
    
    def _compute_field_metrics(
        self,
        fields: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, float]:
        """Compute scalar metrics from field arrays."""
        result = {}
        
        if fields is None:
            return result
        
        # Pressure field
        if "p" in fields:
            p = fields["p"]
            p_abs = np.abs(p) if np.iscomplexobj(p) else p
            finite_mask = np.isfinite(p_abs)
            
            result["p_nan_frac"] = float(np.sum(np.isnan(p_abs)) / p_abs.size)
            result["p_inf_frac"] = float(np.sum(np.isinf(p_abs)) / p_abs.size)
            
            if np.any(finite_mask):
                p_finite = p_abs[finite_mask]
                result["p_min"] = float(np.min(p_finite))
                result["p_max"] = float(np.max(p_finite))
                result["p_mean"] = float(np.mean(p_finite))
                result["p_std"] = float(np.std(p_finite))
        
        # Potential field
        if "U" in fields:
            U = fields["U"]
            finite_mask = np.isfinite(U)
            
            result["U_nan_frac"] = float(np.sum(np.isnan(U)) / U.size)
            result["U_inf_frac"] = float(np.sum(np.isinf(U)) / U.size)
            
            if np.any(finite_mask):
                U_finite = U[finite_mask]
                result["U_min"] = float(np.min(U_finite))
                result["U_max"] = float(np.max(U_finite))
                result["U_ptp"] = float(np.ptp(U_finite))
                result["U_std"] = float(np.std(U_finite))
                
                # Check for flat field
                eps = 1e-18 * max(1.0, np.abs(result["U_max"]))
                result["U_range_tiny"] = result["U_ptp"] < eps
        
        # Force fields
        if "Fx" in fields and "Fy" in fields:
            Fx = fields["Fx"]
            Fy = fields["Fy"]
            
            Fx_finite = Fx[np.isfinite(Fx)]
            Fy_finite = Fy[np.isfinite(Fy)]
            
            if len(Fx_finite) > 0:
                result["Fx_max"] = float(np.max(np.abs(Fx_finite)))
                result["Fx_mean"] = float(np.mean(Fx_finite))
            if len(Fy_finite) > 0:
                result["Fy_max"] = float(np.max(np.abs(Fy_finite)))
                result["Fy_mean"] = float(np.mean(Fy_finite))
            
            Fmag = np.sqrt(Fx**2 + Fy**2)
            Fmag_finite = Fmag[np.isfinite(Fmag)]
            if len(Fmag_finite) > 0:
                result["Fmag_max"] = float(np.max(Fmag_finite))
        
        # Forcing boundary values
        if "vb" in fields:
            vb = fields["vb"]
            vb_abs = np.abs(vb) if np.iscomplexobj(vb) else vb
            finite_mask = np.isfinite(vb_abs)
            if np.any(finite_mask):
                vb_finite = vb_abs[finite_mask]
                result["forcing_vb_min"] = float(np.min(vb_finite))
                result["forcing_vb_max"] = float(np.max(vb_finite))
                result["forcing_vb_mean"] = float(np.mean(vb_finite))
        
        return result
    
    def record_step(
        self,
        step_idx: int,
        control,
        particle_xy: tuple[float, float],
        target_xy: tuple[float, float],
        trap_xy: tuple[float, float],
        info: Dict[str, Any],
        fields: Optional[Dict[str, np.ndarray]] = None,
        force_save_fields: bool = False,
    ):
        """
        Record metrics for a single simulation step.
        
        Args:
            step_idx: Current step index.
            control: Control object (Control3Pucks or array).
            particle_xy: Current particle position (x, y).
            target_xy: Target position (x, y).
            trap_xy: Trap center position (x, y), can be (nan, nan).
            info: Controller info dict with mode, costs, metrics, etc.
            fields: Optional dict with field arrays (p, U, Fx, Fy, vb).
            force_save_fields: Force saving .npz even if stride not met.
        
        Note: The primary source of field metrics is info["metrics"] from the
        evaluator. Fields arrays are only used as fallback/cross-check.
        """
        if not self.enabled:
            return
        
        self._step_count += 1
        
        # Skip if not on stride (but still track for delta computation)
        skip_record = (step_idx % self.stride != 0)
        
        # Extract control array
        ctrl_arr = self._extract_control_array(control)
        
        # Scaling for weighted norms (for GP training consistency)
        L_SCALE = 1e-3   # position scale (mm)
        V_SCALE = 0.08   # velocity scale (typical v0)
        PHI_SCALE = np.pi  # phase scale
        
        # Weight vector: [x, y, v, phi] for each of 3 pucks
        weights = np.array([
            L_SCALE, L_SCALE, V_SCALE, PHI_SCALE,  # A
            L_SCALE, L_SCALE, V_SCALE, PHI_SCALE,  # B
            L_SCALE, L_SCALE, V_SCALE, PHI_SCALE,  # C
        ])
        
        # Compute control deltas (both unweighted and weighted)
        delta_u_norm = np.nan
        delta_delta_u_norm = np.nan
        delta_u_norm_w = np.nan
        delta_delta_u_norm_w = np.nan
        
        if self._prev_ctrl_arr is not None:
            delta_u = ctrl_arr - self._prev_ctrl_arr
            delta_u_norm = float(np.linalg.norm(delta_u))
            
            # Weighted norm: divide by scale factors
            delta_u_scaled = delta_u / weights
            delta_u_norm_w = float(np.linalg.norm(delta_u_scaled))
            
            if self._prev_delta_u is not None:
                delta_delta_u = delta_u - self._prev_delta_u
                delta_delta_u_norm = float(np.linalg.norm(delta_delta_u))
                
                # Weighted jitter norm
                delta_delta_u_scaled = delta_delta_u / weights
                delta_delta_u_norm_w = float(np.linalg.norm(delta_delta_u_scaled))
            
            self._prev_delta_u = delta_u.copy()
        
        self._prev_ctrl_arr = ctrl_arr.copy()
        
        if skip_record:
            return
        
        # Build metrics object with basic values
        metrics = StepMetrics(
            step_idx=step_idx,
            timestamp=datetime.now().isoformat(),
            particle_x=float(particle_xy[0]),
            particle_y=float(particle_xy[1]),
            target_x=float(target_xy[0]),
            target_y=float(target_xy[1]),
            tracking_error=float(np.sqrt(
                (particle_xy[0] - target_xy[0])**2 + 
                (particle_xy[1] - target_xy[1])**2
            )),
            trap_x=float(trap_xy[0]) if np.isfinite(trap_xy[0]) else np.nan,
            trap_y=float(trap_xy[1]) if np.isfinite(trap_xy[1]) else np.nan,
            delta_u_norm=delta_u_norm,
            delta_delta_u_norm=delta_delta_u_norm,
            delta_u_norm_w=delta_u_norm_w,
            delta_delta_u_norm_w=delta_delta_u_norm_w,
        )
        
        # Control values
        if len(ctrl_arr) >= 12:
            metrics.ctrl_xA = float(ctrl_arr[0])
            metrics.ctrl_yA = float(ctrl_arr[1])
            metrics.ctrl_vA = float(ctrl_arr[2])
            metrics.ctrl_phiA = float(ctrl_arr[3])
            metrics.ctrl_xB = float(ctrl_arr[4])
            metrics.ctrl_yB = float(ctrl_arr[5])
            metrics.ctrl_vB = float(ctrl_arr[6])
            metrics.ctrl_phiB = float(ctrl_arr[7])
            metrics.ctrl_xC = float(ctrl_arr[8])
            metrics.ctrl_yC = float(ctrl_arr[9])
            metrics.ctrl_vC = float(ctrl_arr[10])
            metrics.ctrl_phiC = float(ctrl_arr[11])
        
        # Extract info dict values
        metrics.trap_found = info.get("trap_found", False)
        metrics.trap_stable = info.get("trap_stable", False)
        metrics.control_mode = info.get("mode", "")
        
        if "stiffness" in info and info["stiffness"] is not None:
            stiff = np.array(info["stiffness"])
            if len(stiff) >= 2:
                metrics.stiff_eig_1 = float(stiff[0])
                metrics.stiff_eig_2 = float(stiff[1])
                metrics.stiff_min = float(np.min(stiff))
        
        metrics.trap_depth = info.get("trap_depth", np.nan)
        
        # MPC candidate stats
        if "candidate_costs" in info:
            costs = np.array(info["candidate_costs"])
            finite_costs = costs[np.isfinite(costs)]
            if len(finite_costs) > 0:
                metrics.candidate_cost_min = float(np.min(finite_costs))
                metrics.candidate_cost_max = float(np.max(finite_costs))
                metrics.candidate_cost_mean = float(np.mean(finite_costs))
                metrics.candidate_cost_std = float(np.std(finite_costs))
            metrics.n_candidates = len(costs)
        
        metrics.chosen_candidate_idx = info.get("chosen_idx", -1)
        
        # Bounds/rate limit flags
        metrics.bounds_clipped_x = info.get("bounds_clipped_x", False)
        metrics.bounds_clipped_y = info.get("bounds_clipped_y", False)
        metrics.bounds_clipped_v = info.get("bounds_clipped_v", False)
        metrics.rate_limited = info.get("rate_limited", False)
        metrics.max_step_clipped = info.get("max_step_clipped", False)
        
        # Rendering flags
        metrics.render_flat_flag = info.get("render_flat_flag", False)
        metrics.render_nan_flag = info.get("render_nan_flag", False)
        
        # ===== FIELD METRICS: Primary source is info["metrics"] from evaluator =====
        # This ensures we have scalar metrics EVERY step, not just when fields are provided
        evaluator_metrics = info.get("metrics", {})
        
        # Solver stats (from evaluator metrics first)
        if "solver_time_ms" in evaluator_metrics:
            metrics.solver_time_ms = float(evaluator_metrics["solver_time_ms"])
        else:
            metrics.solver_time_ms = info.get("solver_time_ms", np.nan)
        
        if "solver_residual" in evaluator_metrics:
            val = evaluator_metrics["solver_residual"]
            metrics.solver_residual = float(val) if val is not None else np.nan
        else:
            metrics.solver_residual = info.get("solver_residual", np.nan)
        
        # Forcing stats
        for key in ["forcing_vb_min", "forcing_vb_max", "forcing_vb_mean"]:
            if key in evaluator_metrics:
                setattr(metrics, key, float(evaluator_metrics[key]))
        
        # Pressure field stats
        for key in ["p_min", "p_max", "p_mean", "p_std", "p_nan_frac", "p_inf_frac"]:
            if key in evaluator_metrics:
                setattr(metrics, key, float(evaluator_metrics[key]))
        
        # Gor'kov potential stats
        for key in ["U_min", "U_max", "U_ptp", "U_std", "U_nan_frac", "U_inf_frac"]:
            if key in evaluator_metrics:
                setattr(metrics, key, float(evaluator_metrics[key]))
        
        # Force stats
        for key in ["Fx_max", "Fy_max", "Fmag_max", "Fx_mean", "Fy_mean"]:
            if key in evaluator_metrics:
                setattr(metrics, key, float(evaluator_metrics[key]))
        
        # Trap metrics from evaluator (these override the trap_xy passed in)
        if "trap_found" in evaluator_metrics:
            metrics.trap_found = bool(evaluator_metrics["trap_found"])
        if "trap_stable" in evaluator_metrics:
            metrics.trap_stable = bool(evaluator_metrics["trap_stable"])
        if "trap_x" in evaluator_metrics and np.isfinite(evaluator_metrics["trap_x"]):
            metrics.trap_x = float(evaluator_metrics["trap_x"])
        if "trap_y" in evaluator_metrics and np.isfinite(evaluator_metrics["trap_y"]):
            metrics.trap_y = float(evaluator_metrics["trap_y"])
        
        # Stiffness from evaluator metrics
        for key in ["stiff_eig_1", "stiff_eig_2", "stiff_min", "trap_depth"]:
            if key in evaluator_metrics:
                val = evaluator_metrics[key]
                if val is not None and np.isfinite(val):
                    setattr(metrics, key, float(val))
        
        # Trap candidate fields (always set when minimum found)
        for key in ["trap_candidate_x", "trap_candidate_y", "trap_candidate_depth"]:
            if key in evaluator_metrics:
                val = evaluator_metrics[key]
                if val is not None and np.isfinite(val):
                    setattr(metrics, key, float(val))
        
        # Proxy trap fields (global argmin of U - always available)
        for key in ["proxy_trap_x", "proxy_trap_y", "proxy_trap_U"]:
            if key in evaluator_metrics:
                val = evaluator_metrics[key]
                if val is not None and np.isfinite(val):
                    setattr(metrics, key, float(val))
        
        # Hessian diagnostics at trap
        for key in ["U_hess_xx", "U_hess_xy", "U_hess_yy", "trap_grad_norm"]:
            if key in evaluator_metrics:
                val = evaluator_metrics[key]
                if val is not None and np.isfinite(val):
                    setattr(metrics, key, float(val))
        
        # ===== FALLBACK: Compute from field arrays if provided =====
        # This provides cross-check and handles cases where evaluator didn't return metrics
        field_metrics = self._compute_field_metrics(fields)
        for key, value in field_metrics.items():
            # Only override if current value is NaN
            if hasattr(metrics, key):
                current = getattr(metrics, key)
                if (isinstance(current, float) and np.isnan(current)):
                    setattr(metrics, key, value)
        
        # Check for flat flag from field metrics or U_ptp
        if metrics.U_ptp is not None and np.isfinite(metrics.U_ptp):
            # Very small U range indicates flat/weak potential
            if metrics.U_ptp < 1e-18:
                metrics.U_range_tiny = True
                metrics.render_flat_flag = True
        
        if field_metrics.get("U_range_tiny", False):
            metrics.U_range_tiny = True
            metrics.render_flat_flag = True
        
        if metrics.render_flat_flag:
            self._flat_count += 1
        
        if metrics.U_nan_frac > 0 or metrics.p_nan_frac > 0:
            metrics.render_nan_flag = True
            self._nan_count += 1
        
        # Store and write
        self._metrics_history.append(metrics)
        
        # Write to CSV
        self._csv_writer.writerow(metrics.to_dict())
        self._csv_file.flush()
        
        # Write JSON
        json_path = self.out_dir / f"step_{step_idx:05d}.json"
        with open(json_path, "w") as f:
            json.dump({
                "metrics": metrics.to_dict(),
                "info": self._sanitize_for_json(info),
            }, f, indent=2, default=str)
        
        # Write NPZ if requested
        should_save_npz = force_save_fields
        if self.save_fields_stride > 0 and step_idx % self.save_fields_stride == 0:
            should_save_npz = True
        if self.save_fields_on_flat and metrics.render_flat_flag:
            should_save_npz = True
        
        if should_save_npz and fields is not None:
            npz_path = self.out_dir / f"step_{step_idx:05d}.npz"
            np.savez_compressed(
                npz_path,
                step_idx=step_idx,
                control=ctrl_arr,
                particle_xy=np.array(particle_xy),
                target_xy=np.array(target_xy),
                trap_xy=np.array(trap_xy),
                **{k: v for k, v in fields.items() if isinstance(v, np.ndarray)}
            )
            if self.verbose:
                print(f"[FlightRecorder] Saved fields: {npz_path.name}")
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Convert numpy types and other non-JSON types."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif hasattr(obj, "__dict__"):
            return self._sanitize_for_json(obj.__dict__)
        else:
            return obj
    
    def finalize(self):
        """Close files and write summary."""
        if not self.enabled:
            return
        
        if self._csv_file:
            self._csv_file.close()
        
        # Compute summary statistics
        if self._metrics_history:
            errors = [m.tracking_error for m in self._metrics_history if np.isfinite(m.tracking_error)]
            stiffs = [m.stiff_min for m in self._metrics_history if np.isfinite(m.stiff_min)]
            U_ptps = [m.U_ptp for m in self._metrics_history if np.isfinite(m.U_ptp)]
            
            summary = {
                "run_start": self._run_start_time.isoformat() if self._run_start_time else None,
                "run_end": datetime.now().isoformat(),
                "total_steps": self._step_count,
                "recorded_steps": len(self._metrics_history),
                "flat_frame_count": self._flat_count,
                "nan_frame_count": self._nan_count,
                "flat_frame_fraction": self._flat_count / max(1, len(self._metrics_history)),
                
                "tracking_error": {
                    "mean": float(np.mean(errors)) if errors else None,
                    "std": float(np.std(errors)) if errors else None,
                    "min": float(np.min(errors)) if errors else None,
                    "max": float(np.max(errors)) if errors else None,
                },
                
                "stiffness_min": {
                    "mean": float(np.mean(stiffs)) if stiffs else None,
                    "min": float(np.min(stiffs)) if stiffs else None,
                    "max": float(np.max(stiffs)) if stiffs else None,
                },
                
                "U_ptp": {
                    "mean": float(np.mean(U_ptps)) if U_ptps else None,
                    "min": float(np.min(U_ptps)) if U_ptps else None,
                    "max": float(np.max(U_ptps)) if U_ptps else None,
                },
                
                "control_modes": {},
            }
            
            # Count control modes
            modes = [m.control_mode for m in self._metrics_history if m.control_mode]
            for mode in set(modes):
                summary["control_modes"][mode] = modes.count(mode)
            
            summary_path = self.out_dir / "run_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            if self.verbose:
                print(f"\n[FlightRecorder] Run Summary:")
                print(f"  Total steps: {self._step_count}")
                print(f"  Recorded: {len(self._metrics_history)}")
                print(f"  Flat frames: {self._flat_count} ({self._flat_count/max(1,len(self._metrics_history))*100:.1f}%)")
                print(f"  NaN frames: {self._nan_count}")
                if errors:
                    print(f"  Mean tracking error: {np.mean(errors):.4f}")
                print(f"  Saved to: {self.out_dir}")
    
    def get_metrics_array(self, key: str) -> np.ndarray:
        """Get array of a specific metric across all recorded steps."""
        return np.array([getattr(m, key, np.nan) for m in self._metrics_history])
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False
