"""
Reachability Row Builder: Convert flight recorder steps to GP training rows.

This module provides utilities to transform per-step diagnostic data
into training data format for Bayesian/GP surrogate models.

The feature space is designed for learning:
- How control changes (Δu) affect trap position changes (Δtrap)
- What macro actions produce what effects
- Stiffness and stability indicators

Usage:
    from tweezers.diagnostics.reachability_rows import ReachabilityRowBuilder
    
    builder = ReachabilityRowBuilder()
    rows = builder.build_from_csv("results/run_xxx/steps.csv")
    X, Y, valid_mask = builder.to_arrays(rows)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


# Macro action name to one-hot index mapping
MACRO_ACTION_NAMES = [
    "HOLD",
    "TRANSLATE_TRAP_X_POS",
    "TRANSLATE_TRAP_X_NEG",
    "TRANSLATE_TRAP_Y_POS",
    "TRANSLATE_TRAP_Y_NEG",
    "MOVE_A_RIGHT",
    "MOVE_A_LEFT",
    "MOVE_B_RIGHT",
    "MOVE_B_LEFT",
    "MOVE_C_UP",
    "MOVE_C_DOWN",
    "PHASE_SHIFT_A_POS",
    "PHASE_SHIFT_A_NEG",
    "PHASE_SHIFT_B_POS",
    "PHASE_SHIFT_B_NEG",
    "PHASE_SHIFT_C_POS",
    "PHASE_SHIFT_C_NEG",
    "INTENSITY_A_UP",
    "INTENSITY_A_DOWN",
    "INTENSITY_B_UP",
    "INTENSITY_B_DOWN",
    "INTENSITY_C_UP",
    "INTENSITY_C_DOWN",
]

N_MACRO_ACTIONS = len(MACRO_ACTION_NAMES)


@dataclass
class ReachabilityRow:
    """
    Single training row for reachability/surrogate learning.
    
    Features (X):
        - control_u: Current control state (12 dims for 3 pucks)
        - macro_action_onehot: One-hot encoding of macro action (N_MACRO_ACTIONS dims)
        - macro_magnitude: Scalar magnitude of the action
        - delta_u: Control change vector (12 dims)
        
    Targets (Y):
        - delta_trap_x: Change in trap x position
        - delta_trap_y: Change in trap y position
        - stiff_min: Minimum stiffness eigenvalue (proxy for trap quality)
        - trap_found: Whether trap was found (0 or 1)
        - trap_stable: Whether trap was stable (0 or 1)
    
    Metadata:
        - step_idx: Original step index
        - valid: Whether this row has valid Y values
    """
    
    step_idx: int
    
    # Features (X)
    control_u: np.ndarray  # (12,)
    macro_action_name: str = ""
    macro_action_onehot: np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO_ACTIONS))
    macro_magnitude: float = 0.0
    delta_u: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # Targets (Y)
    delta_trap_x: float = np.nan
    delta_trap_y: float = np.nan
    stiff_min: float = np.nan
    trap_found: bool = False
    trap_stable: bool = False
    
    # Metadata
    valid: bool = False
    
    # Additional context (not used in training, for analysis)
    particle_x: float = np.nan
    particle_y: float = np.nan
    target_x: float = np.nan
    target_y: float = np.nan
    tracking_error: float = np.nan
    
    def get_X(self) -> np.ndarray:
        """Get feature vector."""
        return np.concatenate([
            self.control_u,  # 12
            self.macro_action_onehot,  # N_MACRO_ACTIONS
            [self.macro_magnitude],  # 1
            self.delta_u,  # 12
        ])
    
    def get_Y(self) -> np.ndarray:
        """Get target vector."""
        return np.array([
            self.delta_trap_x,
            self.delta_trap_y,
            self.stiff_min,
            float(self.trap_found),
            float(self.trap_stable),
        ])
    
    @staticmethod
    def X_dim() -> int:
        """Feature dimension."""
        return 12 + N_MACRO_ACTIONS + 1 + 12
    
    @staticmethod
    def Y_dim() -> int:
        """Target dimension."""
        return 5


class ReachabilityRowBuilder:
    """
    Build training rows from flight recorder CSV data.
    """
    
    def __init__(
        self,
        macro_magnitude_default: float = 0.05e-3,
    ):
        self.macro_magnitude_default = macro_magnitude_default
        self._action_to_idx = {name: i for i, name in enumerate(MACRO_ACTION_NAMES)}
    
    def _action_to_onehot(self, action_name: str) -> np.ndarray:
        """Convert action name to one-hot vector."""
        onehot = np.zeros(N_MACRO_ACTIONS)
        idx = self._action_to_idx.get(action_name.upper(), 0)
        onehot[idx] = 1.0
        return onehot
    
    def build_from_csv(
        self,
        csv_path: Path,
        require_trap_found: bool = False,
    ) -> List[ReachabilityRow]:
        """
        Build training rows from a steps.csv file.
        
        Args:
            csv_path: Path to steps.csv
            require_trap_found: Only include rows where trap was found
        
        Returns:
            List of ReachabilityRow objects
        """
        import csv
        
        csv_path = Path(csv_path)
        rows: List[ReachabilityRow] = []
        
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            prev_row_data = None
            
            for row_data in reader:
                step_idx = int(row_data.get("step_idx", -1))
                
                # Extract control state
                control_u = np.array([
                    float(row_data.get("ctrl_xA", 0)),
                    float(row_data.get("ctrl_yA", 0)),
                    float(row_data.get("ctrl_vA", 0)),
                    float(row_data.get("ctrl_phiA", 0)),
                    float(row_data.get("ctrl_xB", 0)),
                    float(row_data.get("ctrl_yB", 0)),
                    float(row_data.get("ctrl_vB", 0)),
                    float(row_data.get("ctrl_phiB", 0)),
                    float(row_data.get("ctrl_xC", 0)),
                    float(row_data.get("ctrl_yC", 0)),
                    float(row_data.get("ctrl_vC", 0)),
                    float(row_data.get("ctrl_phiC", 0)),
                ])
                
                # Extract trap info
                trap_x = self._safe_float(row_data.get("trap_x", "nan"))
                trap_y = self._safe_float(row_data.get("trap_y", "nan"))
                trap_found = row_data.get("trap_found", "False").lower() == "true"
                trap_stable = row_data.get("trap_stable", "False").lower() == "true"
                stiff_min = self._safe_float(row_data.get("stiff_min", "nan"))
                
                # Extract control mode / macro action
                control_mode = row_data.get("control_mode", "")
                
                # Build row
                row = ReachabilityRow(
                    step_idx=step_idx,
                    control_u=control_u,
                    trap_found=trap_found,
                    trap_stable=trap_stable,
                    stiff_min=stiff_min,
                    particle_x=self._safe_float(row_data.get("particle_x", "nan")),
                    particle_y=self._safe_float(row_data.get("particle_y", "nan")),
                    target_x=self._safe_float(row_data.get("target_x", "nan")),
                    target_y=self._safe_float(row_data.get("target_y", "nan")),
                    tracking_error=self._safe_float(row_data.get("tracking_error", "nan")),
                )
                
                # Compute delta_u from previous row
                if prev_row_data is not None:
                    prev_control = np.array([
                        float(prev_row_data.get("ctrl_xA", 0)),
                        float(prev_row_data.get("ctrl_yA", 0)),
                        float(prev_row_data.get("ctrl_vA", 0)),
                        float(prev_row_data.get("ctrl_phiA", 0)),
                        float(prev_row_data.get("ctrl_xB", 0)),
                        float(prev_row_data.get("ctrl_yB", 0)),
                        float(prev_row_data.get("ctrl_vB", 0)),
                        float(prev_row_data.get("ctrl_phiB", 0)),
                        float(prev_row_data.get("ctrl_xC", 0)),
                        float(prev_row_data.get("ctrl_yC", 0)),
                        float(prev_row_data.get("ctrl_vC", 0)),
                        float(prev_row_data.get("ctrl_phiC", 0)),
                    ])
                    row.delta_u = control_u - prev_control
                    
                    # Compute delta_trap (Y values)
                    prev_trap_x = self._safe_float(prev_row_data.get("trap_x", "nan"))
                    prev_trap_y = self._safe_float(prev_row_data.get("trap_y", "nan"))
                    
                    if np.isfinite(prev_trap_x) and np.isfinite(trap_x):
                        row.delta_trap_x = trap_x - prev_trap_x
                    if np.isfinite(prev_trap_y) and np.isfinite(trap_y):
                        row.delta_trap_y = trap_y - prev_trap_y
                    
                    # Mark as valid if we have delta_trap
                    row.valid = np.isfinite(row.delta_trap_x) and np.isfinite(row.delta_trap_y)
                    
                    # Set macro action based on control mode
                    if control_mode == "macro":
                        # Infer action from delta_u
                        row.macro_action_name = self._infer_macro_action(row.delta_u)
                        row.macro_action_onehot = self._action_to_onehot(row.macro_action_name)
                        row.macro_magnitude = self.macro_magnitude_default
                    else:
                        row.macro_action_name = "HOLD"
                        row.macro_action_onehot = self._action_to_onehot("HOLD")
                        row.macro_magnitude = np.linalg.norm(row.delta_u[:8])  # Position/velocity changes
                
                # Apply filter
                if require_trap_found and not trap_found:
                    prev_row_data = row_data
                    continue
                
                rows.append(row)
                prev_row_data = row_data
        
        return rows
    
    def _safe_float(self, val: str) -> float:
        """Convert string to float, handling NaN."""
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan
    
    def _infer_macro_action(self, delta_u: np.ndarray) -> str:
        """Infer likely macro action from control delta."""
        # delta_u layout: [xA, yA, vA, phiA, xB, yB, vB, phiB, xC, yC, vC, phiC]
        
        dx_A = delta_u[0]
        dy_A = delta_u[1]
        dx_B = delta_u[4]
        dy_B = delta_u[5]
        dx_C = delta_u[8]
        dy_C = delta_u[9]
        
        dphi_A = delta_u[3]
        dphi_B = delta_u[7]
        dphi_C = delta_u[11]
        
        # Find dominant change
        changes = {
            "MOVE_A_RIGHT": dx_A,
            "MOVE_A_LEFT": -dx_A,
            "MOVE_B_RIGHT": dx_B,
            "MOVE_B_LEFT": -dx_B,
            "MOVE_C_UP": dy_C,
            "MOVE_C_DOWN": -dy_C,
            "PHASE_SHIFT_B_POS": dphi_B,
            "PHASE_SHIFT_B_NEG": -dphi_B,
            "TRANSLATE_TRAP_X_POS": (dx_A + dx_B) / 2,
            "TRANSLATE_TRAP_X_NEG": -(dx_A + dx_B) / 2,
            "TRANSLATE_TRAP_Y_POS": (dy_A + dy_B + dy_C) / 3,
            "TRANSLATE_TRAP_Y_NEG": -(dy_A + dy_B + dy_C) / 3,
        }
        
        best_action = "HOLD"
        best_score = 0.0
        
        for action, score in changes.items():
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def to_arrays(
        self,
        rows: List[ReachabilityRow],
        only_valid: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert rows to numpy arrays for training.
        
        Args:
            rows: List of ReachabilityRow objects
            only_valid: Only include rows with valid=True
        
        Returns:
            X: Feature array (N, X_dim)
            Y: Target array (N, Y_dim)
            valid_mask: Boolean mask indicating which rows were used
        """
        if only_valid:
            valid_rows = [r for r in rows if r.valid]
        else:
            valid_rows = rows
        
        if len(valid_rows) == 0:
            return np.zeros((0, ReachabilityRow.X_dim())), \
                   np.zeros((0, ReachabilityRow.Y_dim())), \
                   np.array([], dtype=bool)
        
        X = np.array([r.get_X() for r in valid_rows])
        Y = np.array([r.get_Y() for r in valid_rows])
        valid_mask = np.array([r.valid for r in rows])
        
        return X, Y, valid_mask
    
    def save_training_data(
        self,
        rows: List[ReachabilityRow],
        out_dir: Path,
        only_valid: bool = True,
    ) -> Dict[str, Path]:
        """
        Save training data to NPZ files.
        
        Args:
            rows: List of ReachabilityRow objects
            out_dir: Output directory
            only_valid: Only include valid rows
        
        Returns:
            Dict with paths to saved files
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        X, Y, valid_mask = self.to_arrays(rows, only_valid=only_valid)
        
        np.savez_compressed(
            out_dir / "training_data.npz",
            X=X,
            Y=Y,
            valid_mask=valid_mask,
            feature_names=[
                *[f"ctrl_{name}" for name in ["xA", "yA", "vA", "phiA", "xB", "yB", "vB", "phiB", "xC", "yC", "vC", "phiC"]],
                *[f"action_{name}" for name in MACRO_ACTION_NAMES],
                "macro_magnitude",
                *[f"delta_{name}" for name in ["xA", "yA", "vA", "phiA", "xB", "yB", "vB", "phiB", "xC", "yC", "vC", "phiC"]],
            ],
            target_names=["delta_trap_x", "delta_trap_y", "stiff_min", "trap_found", "trap_stable"],
        )
        
        print(f"Saved training data: {len(X)} valid rows out of {len(rows)} total")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        
        return {
            "training_data": out_dir / "training_data.npz",
        }
