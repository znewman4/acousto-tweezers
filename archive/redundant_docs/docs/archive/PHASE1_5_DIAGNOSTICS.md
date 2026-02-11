# Phase 1.5 Diagnostics and Impedance Verification

**Date:** February 6, 2026  
**Status:** Implemented and tested

## Overview

This document describes the diagnostic capabilities added to Phase 1 to ensure correctness and guide mesh resolution choice before proceeding to time evolution.

## Impedance Boundary Conditions

### Exact Weak-Form Implementation

The impedance boundary conditions are implemented in the weak form of the Helmholtz equation as follows:

#### **Bottom Boundary (z = 0)**

**Physical condition:** Robin/impedance BC
```
∂p/∂n = -ik (1/Z_b) p
```

**Weak form contribution:**
```
a_bottom = ∫_Γ_bottom (-ik/Z_b) p φ̄ dS
```

**Impedance value (REAL, no imaginary part):**
- Material: Polystyrene substrate
- Z_b = ρ_polystyrene × c_polystyrene
- Z_b = 1050 kg/m³ × 2350 m/s = **2.4675 × 10⁶ Pa·s/m** = 2.468 MPa·s/m

**Reflection coefficient:**
- R_bottom = |(**Z_b - Z_w)/(Z_b + Z_w)|** = 0.246 (partial reflection)

#### **Top Boundary (z = 2 mm)**

**Physical condition:** Water-air interface
```
∂p/∂n = -ik (1/Z_a) p  
```

**Weak form contribution:**
```
a_top = ∫_Γ_top (-ik/Z_a) p φ̄ dS
```

**Impedance value (REAL):**
- Material: Air
- Z_a = ρ_air × c_air
- Z_a = 1.2 kg/m³ × 343 m/s = **411.6 Pa·s/m** = 0.4116 kPa·s/m

**Reflection coefficient:**
- R_top = |(Z_a - Z_w)/(Z_a + Z_w)| = 0.999 (near pressure-release)

#### **Side Walls (x=0, x=Lx, y=0, y=Ly)**

**Physical condition:** Velocity actuation (Neumann BC)
```
∂p/∂n = -iωρ v₀ exp(iφᵢ)
```

**Weak form contribution (RHS):**
```
L_wall = ∫_Γ_wall (-iωρ v₀ exp(iφᵢ)) φ̄ dS
```

where:
- ω = 2π × 2 MHz = 1.257 × 10⁷ rad/s
- ρ = 997 kg/m³ (water density)
- v₀ = 1 mm/s (velocity amplitude)
- φᵢ ∈ {0, π, π/2, ...} (phase per wall)

### Configurable Impedance Parameters

Impedance values can be set via `SquareDishConfig`:

```python
config = SquareDishConfig()
config.rho_polystyrene = 1050.0  # kg/m³
config.c_polystyrene = 2350.0    # m/s
config.rho_air = 1.2              # kg/m³
config.c_air = 343.0              # m/s
```

Bottom BC mode can be switched:
```python
from square_dish_phase_control import ImpedanceBCMode

config.bottom_bc_mode = ImpedanceBCMode.IMPEDANCE  # Standard (default)
config.bottom_bc_mode = ImpedanceBCMode.RIGID      # Sound-hard wall (∂p/∂n = 0)
```

All values are saved in `results/square_dish_phase1/run_*/config.json`.

## Field Diagnostics

For each phase configuration solve, the following scalars are computed and saved:

### Pressure Field
- `max_p`: Maximum pressure magnitude, max(|p|) [Pa]
- `mean_p`: Mean pressure magnitude, mean(|p|) [Pa]  
- `l2_p`: L2 norm of pressure, √(∫|p|² dV) [Pa·m^(3/2)]

### Gor'kov Potential
Computed on mid-height plane (z = H/2):

- **Minima detection:** Local minima found using `scipy.ndimage.minimum_filter`
- **N deepest stored:** N=10 by default
- **Data saved:** JSON file per phase case with:
  - (x, y) positions of each minimum [m]
  - Gor'kov potential value U [J]
  - Trap depth (U_max - U_min) [J]
  - Trap depth in units of thermal energy kT

**Output:** `results/.../minima_{phase_name}.json`

## Convergence Study

### Running Convergence Study

```bash
# Single resolution (default)
python scripts/square_dish_phase_control.py

# Convergence study with 3 mesh levels
python scripts/square_dish_phase_control.py --convergence
```

### Mesh Resolution Levels

By default, three mesh levels are tested:

| Level  | Elements/wavelength | Element size | Description |
|--------|---------------------|--------------|-------------|
| Coarse | 9.6  (0.8×12)       | ~78 μm       | Fast solve  |
| Medium | 12.0 (1.0×12)       | ~62 μm       | Default     |
| Fine   | 15.0 (1.25×12)      | ~50 μm       | High accuracy|

Wavelength λ = c/f = 1497 m/s / 2 MHz = 0.749 mm at 2 MHz.

### Convergence Metrics

For each pair of consecutive mesh levels:

1. **Minima position displacement:**
   - Match K=5 deepest minima using nearest-neighbor
   - Compute Euclidean distance between matched pairs
   - Report: mean displacement, max displacement [μm]

2. **Pass criterion:**
   - Minima positions should stabilize as mesh refines
   - Typical target: mean displacement < 10 μm between medium→fine

**Output:** `results/.../convergence_metrics.json`

### Interpreting Results

**Good convergence:**
- Mean displacement (medium→fine) < 5-10 μm
- Field scalars (max|p|, mean|p|) vary < 1%
- Trap patterns visually similar

**Poor convergence:**
- Large minima displacements (>50 μm)
- Field values change significantly
- → Need finer mesh

## Particle Configuration

### Size Choice

**Default:** 40 μm radius (previously 50 μm)

**Rationale:** Gor'kov potential approximation requires ka << 1, where:
- k = 8394 m⁻¹ (wavenumber in water at 2 MHz)
- a = 40 μm = 4×10⁻⁵ m
- **ka = 0.336**

This is more conservative than 50 μm (ka = 0.42), improving Gor'kov validity while still representing typical microparticles (30-50 μm polystyrene beads).

### Initial Positions

**5 particles in deterministic cross/quincunx pattern:**

```
         4 (back)
            |
1 (left) - 0 (center) - 2 (right)
            |
         3 (front)
```

All at mid-height z = H/2 = 1 mm.

**Offset from center:** ±L/4 = ±0.5 mm

**Reproducibility:** No randomization; same positions every run for interpretability.

## Impedance Verification: Rigid vs. Impedance Bottom

### Test Case

Compare two bottom BC modes for **"All_In_Phase"** configuration:

1. **Case 1:** Bottom = impedance (Z_b = 2.468 MPa·s/m)
2. **Case 2:** Bottom = rigid (∂p/∂n = 0)

Top impedance unchanged (Z_a = 411.6 Pa·s/m).

### Expected Qualitative Differences

**Impedance BC (Case 1):**
- Partial reflection at bottom (R = 0.246)
- Some energy transmitted into substrate
- Pressure antinodes less pronounced near bottom

**Rigid BC (Case 2):**
- Total reflection at bottom (R = 1.0)
- Pressure node exactly at z=0
- Stronger standing wave pattern
- Slightly higher max|p| values

### Diagnostic Comparison

**Scalars to compare:**
- max|p|, mean|p|, L2(p)
- Number of Gor'kov minima
- Trap depth

**Visual comparison:**
- |p| mid-plane slice: Rigid should show sharper modal structure
- U mid-plane slice: Rigid may have more/deeper traps

**Output:** Separate run directories with `_impedance` and `_rigid` suffixes.

## Output Files

For each run `results/square_dish_phase1/run_YYYYMMDD_HHMMSS/`:

```
config.json                       # Full configuration
diagnostics.json                  # Field scalars for all phases
minima_all_in_phase.json          # Gor'kov minima (phase 1)
minima_lr_opposite.json           # Gor'kov minima (phase 2)
minima_fb_opposite.json           # Gor'kov minima (phase 3)
minima_quadrature.json            # Gor'kov minima (phase 4)
all_in_phase.png                  # |p|, U, particles overlay
lr_opposite.png
fb_opposite.png
quadrature.png
```

**Convergence mode:** Additional subdirectories per mesh level + `convergence_metrics.json`.

## Next Steps

Once mesh resolution is chosen based on convergence:

1. **Phase 2:** Time evolution with smooth phase ramping
2. **Phase 3:** Trajectory optimization for multi-particle manipulation
3. **Phase 4:** Advanced physics (elastic solids, streaming)

## Code Locations

- Main script: `scripts/square_dish_phase_control.py`
- Diagnostics: `scripts/diagnostics_utils.py`
- Documentation: `docs/PHASE1_5_DIAGNOSTICS.md`

---

**Author:** Acousto-Tweezers Project  
**Last Updated:** February 6, 2026
