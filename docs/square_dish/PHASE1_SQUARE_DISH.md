# Phase 1: Square Dish with Side-Wall Actuation

**Date:** February 6, 2026  
**Status:** ✅ Implemented  
**Script:** `scripts/square_dish_phase_control.py`

---

## Overview

Phase 1 transforms the acousto-tweezers simulator to study phase-driven standing wave control with a clean, physically interpretable setup:

- **Square 3D fluid domain** (2mm × 2mm × 2mm water-filled cavity)
- **Four side-wall transducers** with phase-only control
- **Realistic boundary conditions** (impedance BCs, not rigid walls)
- **Closed cavity** (no PML, standing waves dominant)
- **Five particles** (30-50 μm radius) in Gor'kov landscape

This configuration enables understanding how **phase differences between identical transducers** reshape the standing wave pattern and influence particle trapping.

---

## Key Design Decisions

### 1. Geometry: Square Cavity

**Choice:** 2mm × 2mm × 2mm water box

**Rationale:**
- Maintains similar scale to previous Petri dish (comparable to inner water region)
- Square geometry simplifies mode structure vs cylinder
- Small enough for ~12 elements/wavelength mesh (30k DOFs manageable)
- Four distinct walls allow independent phase control

**Trade-offs:**
- Lost cylindrical symmetry (more analysis needed)
- ✅ Gained simpler boundary identification
- ✅ Clearer interpretation of phase effects

### 2. Actuation: Full-Span Side Walls

**Choice:** All four vertical walls (x=0, x=Lx, y=0, y=Ly) as transducers

**Boundary Condition:**
```
∂p/∂n = -iωρ v₀ exp(iφᵢ)
```

**Parameters:**
- Same frequency: f = 2 MHz (λ = 0.749 mm in water)
- Same amplitude: v₀ = 1 mm/s
- **Phase-only differences:** φᵢ ∈ [0, 2π] per wall

**Rationale:**
- Maximizes control authority (four independent phases)
- Full-span avoids complicated patch geometry
- Phase-only control isolates modal interference effects
- Physically realizable (piezo arrays bonded to walls)

**NOT modelled yet:**
- ❌ Transducer size/shape (idealized as uniform velocity)
- ❌ Mounting compliance
- ❌ Frequency response

### 3. Bottom Boundary: Realistic Impedance

**Choice:** Impedance BC based on polystyrene substrate

```
∂p/∂n = -ik (1/Z_bottom) p
Z_bottom = ρ_polystyrene × c_polystyrene = 2.47 MPa·s/m
```

**Reflection coefficient:** R = 0.246 (24.6% reflection, 75.4% transmission/loss)

**Rationale:**
- Typical Petri dish material (polystyrene, ~1mm thick)
- More realistic than sound-hard (R=1) or pressure-release (R=-1)
- Allows partial energy loss without full solid mechanics
- Impedance formula captures first-order reflection/transmission

**Approximations:**
- ✅ Uses bulk acoustic properties (longitudinal wave speed)
- ⚠️ Ignores plate bending modes, thickness resonances
- ⚠️ Assumes local reaction (no lateral wave propagation in plate)

**When to upgrade:**
- If standing wave patterns don't match experiments → add elastic plate
- If bottom resonances suspected → couple Helmholtz to plate modes

### 4. Top Boundary: Water-Air Interface

**Choice:** Impedance BC for water-air interface

```
∂p/∂n = -ik (1/Z_air) p
Z_air = ρ_air × c_air = 411.6 Pa·s/m
```

**Reflection coefficient:** R = 0.999 (99.9% reflection, nearly pressure-release)

**Rationale:**
- Huge impedance mismatch (Z_water / Z_air ≈ 3630)
- Acts almost like pressure-release (p ≈ 0) but physically motivated
- Avoids Dirichlet BC artifacts

**Approximations:**
- ✅ Correct for plane waves at normal incidence
- ⚠️ Neglects surface tension, meniscus effects
- ⚠️ Neglects viscous/thermal boundary layers at interface

### 5. No PML (Closed Cavity)

**Choice:** Hard boundaries, standing waves dominant

**Rationale:**
- **Goal:** Understand resonant cavity modes
- **Physics:** Real chambers have reflective walls
- PML would artificially damp natural modes
- Easier to interpret standing wave nodes/antinodes

**When to add PML:**
- If studying infinite medium propagation
- If radiation losses dominate physics
- Not for this phase (we **want** cavity modes)

---

## Physics Model

### Governing Equation

Frequency-domain Helmholtz in water:
```
∇²p + k²p = 0
```

where k = ω/c (1 + iη/2) includes small loss (η = 10⁻³).

### Weak Form

Find p ∈ V such that for all φ ∈ V:

```
∫_Ω (1/ρ) ∇φ·∇p dV - ∫_Ω (k²/ρ) φ p dV
+ ∫_walls (-iωρ v₀ e^{iφᵢ}) φ dS      [actuation, 4 walls]
+ ∫_bottom (-ik/Z_b) φ p dS            [impedance BC]
+ ∫_top (-ik/Z_a) φ p dS               [impedance BC]
= 0
```

**Complex PETSc required:** Uses `conj(φ)` in bilinear form for proper complex-mode assembly.

### Solver

- **Type:** GMRES + ILU preconditioner
- **DOFs:** ~80,000 (P2 elements, 12 elements/wavelength)
- **Tolerance:** rtol = 10⁻¹⁰, atol = 10⁻¹²
- **Convergence:** Typically <100 iterations

---

## Gor'kov Potential

Computed from pressure field p(r):

```
U(r) = (4π/3)a³ [f₁·⟨p²⟩/(2ρc²) - f₂·(3ρ/4)·⟨|v|²⟩]
```

**Contrast factors:**
```
f₁ = 1 - κ_p/κ_f    (monopole, compressibility contrast)
f₂ = 2(ρ_p - ρ_f)/(2ρ_p + ρ_f)    (dipole, density contrast)
```

**Particle properties:**
- Radius: a = 40 μm
- Density: ρ_p = 1050 kg/m³ (polystyrene beads)
- Compressibility: κ_p = 2.4×10⁻¹⁰ Pa⁻¹

**Typical trap depths:**
- Single-frequency standing wave: 10⁻¹⁶–10⁻¹⁵ J
- Expressed in thermal energy: 10²–10³ kT at 300K

**Particle dynamics (optional relaxation):**
```
dx/dt = -∇U / γ
```
where γ = 6πηa is Stokes drag (η = 0.89 mPa·s for water at 25°C).

---

## Phase Configurations (Diagnostic Set)

Four test cases to validate physics:

### 1. All In Phase
```
φ = [0, 0, 0, 0]
```
**Expected:** Uniform excitation, complex 3D mode pattern

### 2. Left-Right Opposite
```
φ = [0, π, 0, π]
```
**Expected:** x-direction standing wave, pressure nodes at x = Lx/2

### 3. Front-Back Opposite
```
φ = [0, 0, π, π]
```
**Expected:** y-direction standing wave, pressure nodes at y = Ly/2

### 4. Quadrature
```
φ = [0, π/2, π, 3π/2]
```
**Expected:** Rotating/diagonal mode structure

---

## Outputs

For each phase configuration:

1. **Pressure magnitude |p|(x,y)** at mid-height (z = 1mm)
   - Confirms standing wave structure
   - Verifies boundary conditions

2. **Gor'kov potential U(x,y)** at mid-height
   - Shows trapping landscape
   - Identifies minima (trap locations)

3. **Particle overlay**
   - Five particles (initial quincunx pattern)
   - Optionally relaxed to nearby minima
   - Red markers on combined plot

**File format:** Static PNG images (no GIFs yet)

**Output directory:** `results/square_dish_phase1/run_YYYYMMDD_HHMMSS/`

---

## Validation Checklist

Use generated images to verify:

### Boundary Conditions
- [ ] Side walls: Non-zero pressure amplitude (actuated)
- [ ] Bottom: Partial reflection (not |p|=0, not rigid)
- [ ] Top: Nearly pressure-release (small |p|)
- [ ] Corners: Smooth transitions (no singular behavior)

### Standing Wave Structure
- [ ] All In Phase: Central antinode
- [ ] LR Opposite: Node line at x = Lx/2
- [ ] FB Opposite: Node line at y = Ly/2
- [ ] Quadrature: Diagonal or rotating pattern
- [ ] Wavelength: λ ≈ 0.75 mm visible in fringes

### Gor'kov Landscape
- [ ] Minima near pressure nodes (expected for f₁ > 0)
- [ ] Maxima near pressure antinodes
- [ ] Smooth gradient (no mesh artifacts)
- [ ] Reasonable trap depth (10²–10³ kT)

### Mesh Resolution
- [ ] No checkerboard patterns in |p|
- [ ] Smooth pressure contours
- [ ] Stable Gor'kov minima locations
- [ ] Convergence: Run with 15 elements/wavelength, check if minima shift

### Physical Realism
- [ ] max|p| ~ 10⁴–10⁶ Pa (reasonable for 1 mm/s actuation)
- [ ] Trap depth ~ 10⁻¹⁶–10⁻¹⁵ J (order of magnitude check)
- [ ] Particles relax toward minima (if relaxation enabled)

---

## Known Limitations (Acceptable for Phase 1)

### Approximations
1. **Impedance BCs are local** (no lateral propagation in walls/plate)
2. **No thermoviscous boundary layers** (μm-scale losses neglected)
3. **No streaming** (second-order flow ignored)
4. **Overdamped dynamics only** (no particle inertia)
5. **One-way coupling** (particles don't scatter waves)

### Numerical
1. **No h-adaptivity** (uniform mesh throughout)
2. **No p-adaptivity** (fixed P2 elements)
3. **Point evaluation for Gor'kov** (not L² projected)

### Control
1. **No optimization** (manual phase selection only)
2. **No time evolution** (static snapshots)
3. **No multi-particle interactions** (independent particles)

---

## Next Steps (Future Phases)

### Phase 2: Time Evolution
- Smooth phase ramping
- Particle trajectory integration
- GIF animations
- Trap translation demonstrations

### Phase 3: Optimization
- Gradient-based phase optimization
- Target position tracking
- Multi-particle choreography

### Phase 4: Advanced Physics
- Add elastic bottom plate (full coupling)
- Include acoustic streaming
- Multi-frequency excitation
- Particle-particle interactions (secondary Bjerknes)

---

## Usage

### Run Script
```bash
micromamba activate acousto-complex
python scripts/square_dish_phase_control.py
```

### Expected Runtime
- Mesh generation: ~10 seconds
- Per solve: ~5-15 seconds (depends on system)
- Total (4 phase configs): ~1-2 minutes

### Output
```
results/square_dish_phase1/run_YYYYMMDD_HHMMSS/
├── config.json                    # Configuration snapshot
├── all_in_phase.png              # Diagnostic plots
├── lr_opposite.png
├── fb_opposite.png
└── quadrature.png
```

---

## Technical Notes

### Why Complex PETSc?

The Helmholtz equation in frequency domain is inherently complex-valued. Using real-valued solvers would:
- Miss phase information (only magnitude)
- Require splitting into real/imaginary parts (doubles DOFs)
- Lose mathematical elegance of variational form

Complex PETSc allows direct representation: p ∈ ℂⁿ.

### Why P2 Elements?

- **Accuracy:** Standing waves need smooth representation
- **Gor'kov:** Requires computing ∇p accurately
- **Cost:** Acceptable (P2 ≈ 8× DOFs vs P1, but worth it for this problem)

### Why 12 Elements/Wavelength?

**Rule of thumb:** 10-12 elements/wavelength for <1% error in L² norm

At 2 MHz in water:
- λ = 0.749 mm
- Target h = 62 μm
- 2mm domain → ~32 elements per edge → ~30k tets

This is manageable for direct solvers on modern workstations.

---

## References

- Settnes & Bruus (2012): "Forces acting on a small particle in an acoustical field in a viscous fluid"
- Gor'kov (1962): "On the forces acting on a small particle in an acoustical field"
- Bruus (2012): "Acoustofluidics 7: The acoustic radiation force on small particles"

---

**Author:** Acousto-Tweezers Project  
**Implementation:** Claude Sonnet 4.5  
**Date:** February 6, 2026
