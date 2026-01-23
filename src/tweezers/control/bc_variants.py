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
