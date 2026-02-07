# src/tweezers/control/bc_variants.py
# Defines BC variants for boundary condition sensitivity analysis

BC_VARIANTS = {
    "V0_baseline": {
        "left_type": "neumann",
        "right_type": "neumann",
        "top_type": "neumann",
        "bottom_type": "neumann",
        "loss_eta": 1e-3,
        "description": "Reflective cavity, rigid walls everywhere"
    },
    "V1_dirichlet_top": {
        "left_type": "neumann",
        "right_type": "neumann",
        "top_type": "dirichlet",
        "bottom_type": "neumann",
        "loss_eta": 1e-3,
        "description": "Open dish proxy (pressure-release top)"
    },
    "V2_lossy": {
        "left_type": "neumann",
        "right_type": "neumann",
        "top_type": "neumann",
        "bottom_type": "neumann",
        "loss_eta": 1e-2,
        "description": "Higher bulk damping, less resonant"
    },
    # Robin BCs for lossiness experiment:
    "V3_robin_ik": {
        "left_type": "robin",
        "right_type": "robin",
        "top_type": "robin",
        "bottom_type": "dirichlet",
        "robin_alpha_scale": 1.0,
        "description": "Robin BC (alpha = ik) on sides/top, Dirichlet bottom"
    },
    "V4_robin_ik_10x_weaker": {
        "left_type": "robin",
        "right_type": "robin",
        "top_type": "robin",
        "bottom_type": "dirichlet",
        "robin_alpha_scale": 0.1,
        "description": "Robin BC (alpha = ik/10) on sides/top, Dirichlet bottom"
    },
    "V5_robin_ik_100x_weaker": {
        "left_type": "robin",
        "right_type": "robin",
        "top_type": "robin",
        "bottom_type": "dirichlet",
        "robin_alpha_scale": 0.01,
        "description": "Robin BC (alpha = ik/100) on sides/top, Dirichlet bottom"
    },
    # Optional V3: Dirichlet top + higher loss
    # "V3_dirichlet_top_lossy": {
    #     "left_type": "neumann",
    #     "right_type": "neumann",
    #     "top_type": "dirichlet",
    #     "bottom_type": "neumann",
    #     "loss_eta": 1e-2,
    #     "description": "Open dish + stronger loss"
    # },
}
