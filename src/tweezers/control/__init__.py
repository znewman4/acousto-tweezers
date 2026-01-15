from .evaluator import (
    DishDomain,
    MediumProps,
    EvaluatorConfig,
    Control2Pucks,
    BottomFootprint25DEvaluator,
)

from .controller import (
    ControlState,
    ControlBounds,
    ControlRateLimits,
    ControlVector,
    JacobianInfo,
    JacobianEstimator,
    SafetyConfig,
    SafetyChecker,
    ControlLog,
    ControlLogger,
    ControllerConfig,
    ParticleController,
    VisualizationData,
    create_visualization_data,
)

# 3-Puck extension
from .pucks_3 import (
    Control3Pucks,
    ControlBounds3Pucks,
    ControlRateLimits3Pucks,
    ControlVector3Pucks,
    control_to_forcing_band_vb_3pucks,
    default_3puck_config,
    default_3puck_spread,
)

from .evaluator_3pucks import (
    EvaluatorConfig3Pucks,
    Evaluator3Pucks,
)

# Smooth MPC controller
from .smooth_controller import (
    SmoothMPCConfig,
    ControlHistory,
    CEMCandidateGenerator,
    SmoothMPCController,
    compute_jitter_cost,
    compute_reference_cost,
    plot_control_smoothness,
)

__all__ = [
    # Evaluator
    "DishDomain",
    "MediumProps",
    "EvaluatorConfig",
    "Control2Pucks",
    "BottomFootprint25DEvaluator",
    # Controller
    "ControlState",
    "ControlBounds",
    "ControlRateLimits",
    "ControlVector",
    "JacobianInfo",
    "JacobianEstimator",
    "SafetyConfig",
    "SafetyChecker",
    "ControlLog",
    "ControlLogger",
    "ControllerConfig",
    "ParticleController",
    "VisualizationData",
    "create_visualization_data",
    # 3-Puck extension
    "Control3Pucks",
    "ControlBounds3Pucks",
    "ControlRateLimits3Pucks",
    "ControlVector3Pucks",
    "control_to_forcing_band_vb_3pucks",
    "default_3puck_config",
    "default_3puck_spread",
    "EvaluatorConfig3Pucks",
    "Evaluator3Pucks",
    # Smooth MPC
    "SmoothMPCConfig",
    "ControlHistory",
    "CEMCandidateGenerator",
    "SmoothMPCController",
    "compute_jitter_cost",
    "compute_reference_cost",
    "plot_control_smoothness",
]
