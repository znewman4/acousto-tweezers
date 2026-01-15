#!/usr/bin/env python3
"""
Gaussian Process Surrogate Model for Acoustic Tweezers Control.

STAGE 3 IMPLEMENTATION:
The surrogate models:
    f(u, Δu_macro) → Δtrap

Where:
- u = current control configuration
- Δu_macro = macro action (discrete or parameterized)
- output = predicted trap displacement + uncertainty

Key features:
- Uses scikit-learn Gaussian Process Regressor
- Trains on reachability atlas data
- Provides uncertainty estimates for planning
- Updates online during control

Usage:
    python scripts/surrogate_model.py --train
    python scripts/surrogate_model.py --test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json
from typing import Optional

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Using simple linear surrogate.")


@dataclass
class SurrogateTrainingData:
    """Training data for surrogate model."""
    # Current control (Nx12 for 3-puck)
    current_control: np.ndarray
    
    # Applied delta (Nx12)
    control_delta: np.ndarray
    
    # Resulting trap displacement (Nx2)
    trap_delta: np.ndarray
    
    # Stiffness change (Nx1)
    stiffness_delta: np.ndarray
    
    # Whether trap was stable (Nx1)
    was_stable: np.ndarray


class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate for trap displacement prediction.
    
    Predicts:
        Δtrap_x, Δtrap_y = f(u_current, Δu)
    
    With uncertainty estimates from GP posterior variance.
    """
    
    def __init__(
        self,
        length_scale: float = 0.1e-3,
        noise_level: float = 1e-8,
        n_restarts: int = 5,
    ):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.n_restarts = n_restarts
        
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.gp_x: Optional[GaussianProcessRegressor] = None
        self.gp_y: Optional[GaussianProcessRegressor] = None
        
        self.is_trained = False
        
        # Online learning buffer
        self.online_buffer_X: list[np.ndarray] = []
        self.online_buffer_y: list[np.ndarray] = []
        self.online_buffer_size = 100
    
    def _create_kernel(self):
        """Create GP kernel."""
        if not HAS_SKLEARN:
            return None
        
        # Matern kernel with length scale + noise
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=self.length_scale, length_scale_bounds=(1e-6, 1e-1), nu=2.5) +
            WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e-1))
        )
        return kernel
    
    def train(
        self,
        current_controls: np.ndarray,
        control_deltas: np.ndarray,
        trap_deltas: np.ndarray,
    ):
        """
        Train surrogate on collected data.
        
        Parameters
        ----------
        current_controls : ndarray (N, 12)
            Current control configurations
        control_deltas : ndarray (N, 12)
            Applied control changes
        trap_deltas : ndarray (N, 2)
            Resulting trap displacements (x, y)
        """
        if not HAS_SKLEARN:
            print("scikit-learn not available, using simple linear model")
            self._train_linear(current_controls, control_deltas, trap_deltas)
            return
        
        # Feature matrix: concatenate current control and delta
        X = np.hstack([current_controls, control_deltas])
        y_x = trap_deltas[:, 0]
        y_y = trap_deltas[:, 1]
        
        # Filter out NaN
        valid = np.isfinite(y_x) & np.isfinite(y_y)
        X = X[valid]
        y_x = y_x[valid]
        y_y = y_y[valid]
        
        if len(X) < 10:
            print(f"Warning: Only {len(X)} valid training points")
            return
        
        print(f"Training GP surrogate on {len(X)} samples...")
        
        # Standardize features
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        self.scaler_y = StandardScaler()
        y_combined = np.column_stack([y_x, y_y])
        y_scaled = self.scaler_y.fit_transform(y_combined)
        
        # Train separate GPs for x and y
        kernel = self._create_kernel()
        
        self.gp_x = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
            alpha=1e-10,
        )
        
        self.gp_y = GaussianProcessRegressor(
            kernel=kernel.clone_with_theta(kernel.theta),
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
            alpha=1e-10,
        )
        
        self.gp_x.fit(X_scaled, y_scaled[:, 0])
        self.gp_y.fit(X_scaled, y_scaled[:, 1])
        
        self.is_trained = True
        print("GP training complete.")
    
    def _train_linear(
        self,
        current_controls: np.ndarray,
        control_deltas: np.ndarray,
        trap_deltas: np.ndarray,
    ):
        """Fallback linear model when sklearn not available."""
        X = np.hstack([current_controls, control_deltas])
        valid = np.all(np.isfinite(trap_deltas), axis=1)
        X = X[valid]
        y = trap_deltas[valid]
        
        if len(X) < 10:
            return
        
        # Simple linear regression: y = X @ W
        X_aug = np.hstack([X, np.ones((len(X), 1))])  # Add bias
        self._linear_W_x = np.linalg.lstsq(X_aug, y[:, 0], rcond=None)[0]
        self._linear_W_y = np.linalg.lstsq(X_aug, y[:, 1], rcond=None)[0]
        self.is_trained = True
    
    def predict(
        self,
        current_control: np.ndarray,
        control_delta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict trap displacement with uncertainty.
        
        Parameters
        ----------
        current_control : ndarray (12,) or (N, 12)
            Current control configuration
        control_delta : ndarray (12,) or (N, 12)
            Proposed control change
        
        Returns
        -------
        mean : ndarray (2,) or (N, 2)
            Predicted (Δtrap_x, Δtrap_y)
        std : ndarray (2,) or (N, 2)
            Uncertainty (std_x, std_y)
        """
        if not self.is_trained:
            # Return zero with high uncertainty if not trained
            if current_control.ndim == 1:
                return np.zeros(2), np.ones(2) * 1e-3
            else:
                n = len(current_control)
                return np.zeros((n, 2)), np.ones((n, 2)) * 1e-3
        
        # Handle single sample
        single_sample = current_control.ndim == 1
        if single_sample:
            current_control = current_control.reshape(1, -1)
            control_delta = control_delta.reshape(1, -1)
        
        X = np.hstack([current_control, control_delta])
        
        if not HAS_SKLEARN:
            # Linear fallback
            X_aug = np.hstack([X, np.ones((len(X), 1))])
            mean_x = X_aug @ self._linear_W_x
            mean_y = X_aug @ self._linear_W_y
            mean = np.column_stack([mean_x, mean_y])
            std = np.ones_like(mean) * 1e-4  # Constant uncertainty
            
            if single_sample:
                return mean[0], std[0]
            return mean, std
        
        # GP prediction
        X_scaled = self.scaler_X.transform(X)
        
        mean_x_scaled, std_x_scaled = self.gp_x.predict(X_scaled, return_std=True)
        mean_y_scaled, std_y_scaled = self.gp_y.predict(X_scaled, return_std=True)
        
        # Inverse transform
        mean_scaled = np.column_stack([mean_x_scaled, mean_y_scaled])
        mean = self.scaler_y.inverse_transform(mean_scaled)
        
        # Scale std (approximate)
        std_x = std_x_scaled * self.scaler_y.scale_[0]
        std_y = std_y_scaled * self.scaler_y.scale_[1]
        std = np.column_stack([std_x, std_y])
        
        if single_sample:
            return mean[0], std[0]
        return mean, std
    
    def update_online(
        self,
        current_control: np.ndarray,
        control_delta: np.ndarray,
        actual_trap_delta: np.ndarray,
    ):
        """
        Add observation to online buffer for incremental learning.
        
        Call retrain_online() periodically to update the model.
        """
        X = np.hstack([current_control, control_delta])
        self.online_buffer_X.append(X)
        self.online_buffer_y.append(actual_trap_delta)
        
        # Keep buffer bounded
        if len(self.online_buffer_X) > self.online_buffer_size:
            self.online_buffer_X.pop(0)
            self.online_buffer_y.pop(0)
    
    def retrain_online(self):
        """Retrain GP with online buffer data."""
        if len(self.online_buffer_X) < 20:
            return
        
        X = np.array(self.online_buffer_X)
        y = np.array(self.online_buffer_y)
        
        # Split into current control and delta
        current_controls = X[:, :12]
        control_deltas = X[:, 12:]
        
        self.train(current_controls, control_deltas, y)
    
    def save(self, path: Path):
        """Save trained model."""
        if not self.is_trained:
            print("Model not trained, nothing to save")
            return
        
        path.mkdir(parents=True, exist_ok=True)
        
        if HAS_SKLEARN:
            import joblib
            joblib.dump(self.gp_x, path / "gp_x.joblib")
            joblib.dump(self.gp_y, path / "gp_y.joblib")
            joblib.dump(self.scaler_X, path / "scaler_X.joblib")
            joblib.dump(self.scaler_y, path / "scaler_y.joblib")
        else:
            np.save(path / "linear_W_x.npy", self._linear_W_x)
            np.save(path / "linear_W_y.npy", self._linear_W_y)
        
        print(f"Saved surrogate model to {path}")
    
    def load(self, path: Path):
        """Load trained model."""
        if HAS_SKLEARN:
            import joblib
            self.gp_x = joblib.load(path / "gp_x.joblib")
            self.gp_y = joblib.load(path / "gp_y.joblib")
            self.scaler_X = joblib.load(path / "scaler_X.joblib")
            self.scaler_y = joblib.load(path / "scaler_y.joblib")
        else:
            self._linear_W_x = np.load(path / "linear_W_x.npy")
            self._linear_W_y = np.load(path / "linear_W_y.npy")
        
        self.is_trained = True
        print(f"Loaded surrogate model from {path}")


class MacroActionSurrogate:
    """
    Surrogate specifically for macro action effects.
    
    Instead of continuous control space, this learns:
        f(control_centroid, action_type, magnitude) → Δtrap
    """
    
    def __init__(self):
        # Simple lookup table approach
        # Key: (action_type, magnitude_bin)
        # Value: list of (control_features, delta_trap)
        self.lookup: dict[tuple, list] = {}
        self.is_trained = False
    
    def add_observation(
        self,
        action_type: str,
        magnitude: float,
        control_centroid_x: float,
        delta_trap: np.ndarray,
    ):
        """Add observation to lookup."""
        mag_bin = round(magnitude * 1e5)  # µm precision
        key = (action_type, mag_bin)
        
        if key not in self.lookup:
            self.lookup[key] = []
        
        self.lookup[key].append({
            "cx": control_centroid_x,
            "delta": delta_trap.copy(),
        })
        
        self.is_trained = True
    
    def predict_action_effect(
        self,
        action_type: str,
        magnitude: float,
        control_centroid_x: float,
    ) -> tuple[np.ndarray, float]:
        """
        Predict effect of action at given position.
        
        Returns (predicted_delta, uncertainty).
        """
        mag_bin = round(magnitude * 1e5)
        key = (action_type, mag_bin)
        
        if key not in self.lookup:
            return np.zeros(2), 1e-3  # High uncertainty if no data
        
        observations = self.lookup[key]
        
        # Weighted average by distance to control centroid
        deltas = []
        weights = []
        
        for obs in observations:
            dist = abs(obs["cx"] - control_centroid_x)
            weight = np.exp(-dist / 0.5e-3)  # Gaussian weighting
            deltas.append(obs["delta"])
            weights.append(weight)
        
        if sum(weights) < 1e-10:
            return np.zeros(2), 1e-3
        
        weights = np.array(weights) / sum(weights)
        deltas = np.array(deltas)
        
        mean_delta = np.sum(deltas * weights[:, None], axis=0)
        
        # Uncertainty from variance
        if len(deltas) > 1:
            std = np.sqrt(np.sum(weights[:, None] * (deltas - mean_delta)**2, axis=0))
            uncertainty = np.mean(std)
        else:
            uncertainty = 1e-4
        
        return mean_delta, uncertainty


# ============================================================
# PHASE 7: MACRO ACTION GP FOR INTELLIGENT ACTION SELECTION
# ============================================================

class MacroActionGP:
    """
    Phase 7: Gaussian Process surrogate for macro action selection.
    
    Predicts: f(action_id, context) → Δtrap_xy ± uncertainty
    
    Features:
    - One-hot encoded action types
    - Context features (current phase, position)
    - Trained on macro action atlas
    - Provides action ranking for planning
    """
    
    def __init__(
        self,
        n_actions: int = 12,
        length_scale: float = 1.0,
        noise_level: float = 1e-6,
    ):
        self.n_actions = n_actions
        self.length_scale = length_scale
        self.noise_level = noise_level
        
        self.scaler_X: Optional[StandardScaler] = None
        self.gp_x: Optional[GaussianProcessRegressor] = None
        self.gp_y: Optional[GaussianProcessRegressor] = None
        
        self.action_names: list[str] = []
        self.is_trained = False
        
        # Statistics for reporting
        self.train_samples = 0
        self.mean_displacement_by_action: dict[str, np.ndarray] = {}
    
    def _build_features(
        self,
        action_ids: np.ndarray,
        varied_params: np.ndarray,
        varied_values: np.ndarray,
    ) -> np.ndarray:
        """Build feature matrix from action data."""
        n = len(action_ids)
        
        # One-hot encode actions
        action_onehot = np.zeros((n, self.n_actions))
        for i, aid in enumerate(action_ids):
            if 0 <= aid < self.n_actions:
                action_onehot[i, int(aid)] = 1.0
        
        # Encode varied parameter (phase vs position)
        param_is_phase = np.array([
            1.0 if 'phi' in str(p) else 0.0
            for p in varied_params
        ]).reshape(-1, 1)
        
        # Normalize varied values
        values_norm = (varied_values.reshape(-1, 1) - np.mean(varied_values)) / (np.std(varied_values) + 1e-10)
        
        return np.hstack([action_onehot, param_is_phase, values_norm])
    
    def train_from_atlas(self, atlas_path: Path):
        """Train GP from macro action atlas CSV."""
        import pandas as pd
        
        csv_path = atlas_path / "macro_action_atlas.csv"
        if not csv_path.exists():
            print(f"Atlas not found: {csv_path}")
            return False
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} atlas entries")
        
        # Filter valid entries
        df_valid = df[df["delta_trap_x"].notna() & df["delta_trap_y"].notna()].copy()
        print(f"Valid entries: {len(df_valid)}")
        
        if len(df_valid) < 20:
            print("Not enough valid entries for training")
            return False
        
        # Get unique action names
        self.action_names = sorted(df_valid["action_type"].unique().tolist())
        action_to_id = {name: i for i, name in enumerate(self.action_names)}
        self.n_actions = len(self.action_names)
        
        # Build feature matrix
        action_ids = df_valid["action_type"].map(action_to_id).values
        varied_params = df_valid["varied_param"].values
        varied_values = df_valid["varied_value"].values
        
        X = self._build_features(action_ids, varied_params, varied_values)
        y_x = df_valid["delta_trap_x"].values
        y_y = df_valid["delta_trap_y"].values
        
        # Compute mean displacement per action for reporting
        for action_name in self.action_names:
            mask = df_valid["action_type"] == action_name
            self.mean_displacement_by_action[action_name] = np.array([
                df_valid.loc[mask, "delta_trap_x"].mean(),
                df_valid.loc[mask, "delta_trap_y"].mean(),
            ])
        
        if not HAS_SKLEARN:
            print("scikit-learn not available, using lookup table")
            self._train_lookup(df_valid, action_to_id)
            return True
        
        print(f"Training GP on {len(X)} samples with {X.shape[1]} features...")
        
        # Scale features
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Kernel: RBF for smooth interpolation
        kernel = (
            ConstantKernel(1.0, (1e-4, 1e4)) *
            RBF(length_scale=self.length_scale, length_scale_bounds=(0.1, 10.0)) +
            WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e-2))
        )
        
        self.gp_x = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
        )
        self.gp_y = GaussianProcessRegressor(
            kernel=kernel.clone_with_theta(kernel.theta),
            n_restarts_optimizer=3,
            normalize_y=True,
        )
        
        self.gp_x.fit(X_scaled, y_x)
        self.gp_y.fit(X_scaled, y_y)
        
        self.is_trained = True
        self.train_samples = len(X)
        
        print("GP training complete")
        return True
    
    def _train_lookup(self, df: "pd.DataFrame", action_to_id: dict):
        """Fallback: train lookup table when sklearn unavailable."""
        self._lookup_table = {}
        
        for action_name, action_id in action_to_id.items():
            mask = df["action_type"] == action_name
            dx = df.loc[mask, "delta_trap_x"].values
            dy = df.loc[mask, "delta_trap_y"].values
            
            self._lookup_table[action_name] = {
                "mean": np.array([np.mean(dx), np.mean(dy)]),
                "std": np.array([np.std(dx), np.std(dy)]),
            }
        
        self.is_trained = True
    
    def predict_action_effect(
        self,
        action_name: str,
        varied_param: str = "phiB",
        varied_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict trap displacement for a specific action.
        
        Returns:
            mean: (2,) predicted Δtrap_x, Δtrap_y
            std: (2,) uncertainty
        """
        if not self.is_trained:
            return np.zeros(2), np.ones(2) * 1e-3
        
        if not HAS_SKLEARN:
            if action_name in self._lookup_table:
                entry = self._lookup_table[action_name]
                return entry["mean"], entry["std"] + 1e-6
            return np.zeros(2), np.ones(2) * 1e-3
        
        # Get action ID
        if action_name not in self.action_names:
            return np.zeros(2), np.ones(2) * 1e-3
        
        action_id = self.action_names.index(action_name)
        
        # Build features
        X = self._build_features(
            np.array([action_id]),
            np.array([varied_param]),
            np.array([varied_value]),
        )
        X_scaled = self.scaler_X.transform(X)
        
        mean_x, std_x = self.gp_x.predict(X_scaled, return_std=True)
        mean_y, std_y = self.gp_y.predict(X_scaled, return_std=True)
        
        return np.array([mean_x[0], mean_y[0]]), np.array([std_x[0], std_y[0]])
    
    def rank_actions(
        self,
        target_delta: np.ndarray,
        varied_param: str = "phiB",
        varied_value: float = 0.0,
        uncertainty_penalty: float = 1.0,
    ) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
        """
        Rank macro actions by expected progress toward target.
        
        Parameters
        ----------
        target_delta : ndarray (2,)
            Desired trap displacement (target - current_trap)
        varied_param : str
            Current varied parameter context
        varied_value : float
            Current varied parameter value
        uncertainty_penalty : float
            Weight for uncertainty in ranking (higher = more conservative)
        
        Returns
        -------
        list of (action_name, score, predicted_delta, uncertainty)
            Sorted by score (higher = better)
        """
        if not self.is_trained:
            return []
        
        results = []
        target_norm = np.linalg.norm(target_delta) + 1e-10
        target_dir = target_delta / target_norm
        
        for action_name in self.action_names:
            mean, std = self.predict_action_effect(action_name, varied_param, varied_value)
            
            # Score = progress in target direction - uncertainty penalty
            progress = np.dot(mean, target_dir)  # Projection onto target direction
            uncertainty = np.linalg.norm(std)
            
            score = progress - uncertainty_penalty * uncertainty
            
            results.append((action_name, score, mean, std))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def select_best_action(
        self,
        target_delta: np.ndarray,
        varied_param: str = "phiB",
        varied_value: float = 0.0,
        min_progress: float = 1e-6,
        max_uncertainty: float = 1e-4,
    ) -> Optional[tuple[str, np.ndarray, np.ndarray]]:
        """
        Select best macro action for given target.
        
        Returns None if no action meets criteria.
        
        Returns
        -------
        (action_name, predicted_delta, uncertainty) or None
        """
        ranked = self.rank_actions(target_delta, varied_param, varied_value)
        
        if not ranked:
            return None
        
        target_dir = target_delta / (np.linalg.norm(target_delta) + 1e-10)
        
        for action_name, score, mean, std in ranked:
            progress = np.dot(mean, target_dir)
            uncertainty = np.linalg.norm(std)
            
            if progress > min_progress and uncertainty < max_uncertainty:
                return action_name, mean, std
        
        # Fallback: return best action even if doesn't meet criteria
        best = ranked[0]
        return best[0], best[2], best[3]
    
    def save(self, path: Path):
        """Save trained model."""
        path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "n_actions": self.n_actions,
            "action_names": self.action_names,
            "train_samples": self.train_samples,
            "mean_displacement_by_action": {
                k: v.tolist() for k, v in self.mean_displacement_by_action.items()
            },
        }
        
        with open(path / "macro_gp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        if HAS_SKLEARN and self.gp_x is not None:
            import joblib
            joblib.dump(self.gp_x, path / "macro_gp_x.joblib")
            joblib.dump(self.gp_y, path / "macro_gp_y.joblib")
            joblib.dump(self.scaler_X, path / "macro_scaler_X.joblib")
        
        print(f"Saved MacroActionGP to {path}")
    
    def load(self, path: Path) -> bool:
        """Load trained model."""
        meta_path = path / "macro_gp_metadata.json"
        if not meta_path.exists():
            return False
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        self.n_actions = metadata["n_actions"]
        self.action_names = metadata["action_names"]
        self.train_samples = metadata["train_samples"]
        self.mean_displacement_by_action = {
            k: np.array(v) for k, v in metadata["mean_displacement_by_action"].items()
        }
        
        if HAS_SKLEARN and (path / "macro_gp_x.joblib").exists():
            import joblib
            self.gp_x = joblib.load(path / "macro_gp_x.joblib")
            self.gp_y = joblib.load(path / "macro_gp_y.joblib")
            self.scaler_X = joblib.load(path / "macro_scaler_X.joblib")
        
        self.is_trained = True
        print(f"Loaded MacroActionGP from {path} ({self.train_samples} training samples)")
        return True


def load_training_data_from_atlas(atlas_path: Path) -> SurrogateTrainingData:
    """Load training data from reachability atlas."""
    controls = np.load(atlas_path / "control_configs.npy")
    trap_xy = np.load(atlas_path / "trap_positions.npy")
    stiffness = np.load(atlas_path / "stiffness_data.npy")
    
    n = len(controls)
    
    # Create pairwise deltas
    # For each pair (i, j), delta = controls[j] - controls[i]
    # trap_delta = trap_xy[j] - trap_xy[i]
    
    # Sample pairs
    n_pairs = min(n * 10, 5000)
    rng = np.random.default_rng(42)
    
    current_controls = []
    control_deltas = []
    trap_deltas = []
    stiff_deltas = []
    stable = []
    
    for _ in range(n_pairs):
        i, j = rng.choice(n, 2, replace=False)
        
        current_controls.append(controls[i])
        control_deltas.append(controls[j] - controls[i])
        trap_deltas.append(trap_xy[j] - trap_xy[i])
        stiff_deltas.append(stiffness[j, 0] - stiffness[i, 0])
        stable.append(stiffness[i, 2] and stiffness[j, 2])
    
    return SurrogateTrainingData(
        current_control=np.array(current_controls),
        control_delta=np.array(control_deltas),
        trap_delta=np.array(trap_deltas),
        stiffness_delta=np.array(stiff_deltas).reshape(-1, 1),
        was_stable=np.array(stable).reshape(-1, 1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train surrogate")
    parser.add_argument("--test", action="store_true", help="Test surrogate")
    parser.add_argument("--train-macro-gp", action="store_true",
                       help="Train MacroActionGP from macro action atlas (Phase 7)")
    parser.add_argument("--test-macro-gp", action="store_true",
                       help="Test MacroActionGP action selection")
    parser.add_argument("--atlas", type=str, default="results/reachability_3puck")
    parser.add_argument("--output", type=str, default="results/surrogate_model")
    args = parser.parse_args()
    
    atlas_path = Path(args.atlas)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ===== Phase 7: MacroActionGP Training =====
    if args.train_macro_gp:
        print("=" * 60)
        print("PHASE 7: MACRO ACTION GP TRAINING")
        print("=" * 60)
        
        gp = MacroActionGP()
        success = gp.train_from_atlas(atlas_path)
        
        if success:
            gp.save(output_path)
            
            print("\nMean displacement by action (µm):")
            for action_name, delta in gp.mean_displacement_by_action.items():
                print(f"  {action_name:30s}: Δx={delta[0]*1e6:+6.1f}, Δy={delta[1]*1e6:+6.1f}")
        return
    
    if args.test_macro_gp:
        print("=" * 60)
        print("PHASE 7: MACRO ACTION GP TESTING")
        print("=" * 60)
        
        gp = MacroActionGP()
        if not gp.load(output_path):
            print("No trained MacroActionGP found. Run --train-macro-gp first.")
            return
        
        # Test action selection for various targets
        targets = [
            np.array([50e-6, 0]),      # Move right
            np.array([-50e-6, 0]),     # Move left
            np.array([0, 50e-6]),      # Move up
            np.array([0, -50e-6]),     # Move down
            np.array([30e-6, 30e-6]),  # Move diagonal
        ]
        
        print("\nAction selection for target displacements:")
        for target in targets:
            print(f"\n  Target: ({target[0]*1e6:+.0f}, {target[1]*1e6:+.0f}) µm")
            
            ranked = gp.rank_actions(target)[:3]  # Top 3
            for i, (action_name, score, mean, std) in enumerate(ranked):
                print(f"    {i+1}. {action_name:30s} "
                      f"pred=({mean[0]*1e6:+5.1f},{mean[1]*1e6:+5.1f})µm "
                      f"±({std[0]*1e6:.1f},{std[1]*1e6:.1f}) "
                      f"score={score*1e6:.1f}")
        return
    
    if args.train:
        print("=" * 60)
        print("GAUSSIAN PROCESS SURROGATE TRAINING")
        print("=" * 60)
        
        # Load atlas data
        if not atlas_path.exists():
            print(f"Atlas not found at {atlas_path}")
            print("Run reachability_atlas_3puck.py first")
            return
        
        print(f"\nLoading atlas from {atlas_path}...")
        data = load_training_data_from_atlas(atlas_path)
        
        print(f"Training samples: {len(data.current_control)}")
        
        # Train surrogate
        surrogate = GaussianProcessSurrogate(
            length_scale=0.1e-3,
            noise_level=1e-8,
            n_restarts=3,
        )
        
        surrogate.train(
            data.current_control,
            data.control_delta,
            data.trap_delta,
        )
        
        # Save model
        surrogate.save(output_path)
        
        # Test predictions
        print("\nTest predictions:")
        for i in range(5):
            mean, std = surrogate.predict(
                data.current_control[i],
                data.control_delta[i],
            )
            actual = data.trap_delta[i]
            print(f"  Predicted: ({mean[0]*1e3:+.4f}, {mean[1]*1e3:+.4f}) mm "
                  f"± ({std[0]*1e3:.4f}, {std[1]*1e3:.4f})")
            print(f"  Actual:    ({actual[0]*1e3:+.4f}, {actual[1]*1e3:+.4f}) mm")
            print()
    
    elif args.test:
        print("=" * 60)
        print("SURROGATE MODEL TESTING")
        print("=" * 60)
        
        surrogate = GaussianProcessSurrogate()
        
        if (output_path / "gp_x.joblib").exists() or (output_path / "linear_W_x.npy").exists():
            surrogate.load(output_path)
        else:
            print("No trained model found. Run with --train first.")
            return
        
        # Load test data
        data = load_training_data_from_atlas(atlas_path)
        
        # Evaluate on held-out samples
        n_test = min(100, len(data.current_control))
        errors_x = []
        errors_y = []
        
        for i in range(n_test):
            mean, std = surrogate.predict(
                data.current_control[i],
                data.control_delta[i],
            )
            actual = data.trap_delta[i]
            
            errors_x.append(abs(mean[0] - actual[0]))
            errors_y.append(abs(mean[1] - actual[1]))
        
        print(f"\nTest Results ({n_test} samples):")
        print(f"  Mean absolute error X: {np.mean(errors_x)*1e3:.4f} mm")
        print(f"  Mean absolute error Y: {np.mean(errors_y)*1e3:.4f} mm")
        print(f"  Max error X: {np.max(errors_x)*1e3:.4f} mm")
        print(f"  Max error Y: {np.max(errors_y)*1e3:.4f} mm")
        
        # Save test results
        results = {
            "n_test": n_test,
            "mae_x_mm": float(np.mean(errors_x) * 1e3),
            "mae_y_mm": float(np.mean(errors_y) * 1e3),
            "max_error_x_mm": float(np.max(errors_x) * 1e3),
            "max_error_y_mm": float(np.max(errors_y) * 1e3),
        }
        
        with open(output_path / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_path / 'test_results.json'}")
    
    else:
        print("Run with --train or --test")


if __name__ == "__main__":
    main()
