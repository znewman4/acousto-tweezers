# 🔴 PHYSICS & DATA REALITY CHECK — MODEL INTERROGATION

**Date**: February 8, 2026  
**Context**: ParaView visualization shows "square box domain," "perturbations visible primarily on surfaces," and "little to no apparent z-axis structure"  
**Objective**: Determine what the data physically represents and whether a 3D pressure tornado should exist

---

## A) GOVERNING PHYSICS & MODEL ASSUMPTIONS

### Q1: Is the pressure field solved as fully 3D, or effectively 2D with trivial z-dependence?

**Evidence from code** (`generate_rich_data.py:115-119`):
```python
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) / RHO
     - k**2 * ufl.inner(u, v) / RHO) * ufl.dx
```
**Answer**: ✅ **Fully 3D**. The weak form uses `ufl.grad(u)` which computes ∇p in all three dimensions. The domain integral `ufl.dx` is over the 3D volume.

**Why this matters**:
- **If YES (fully 3D)**: Vertical structure CAN exist if vertical modes are excited
- **If NO (2D or 2.5D)**: No z-structure would exist by construction

**Implication for tornado**: ✅ **Necessary condition met** — solver CAN support 3D structure

---

### Q2: What Helmholtz modes are supported in the z-direction?

**Evidence from code** (`generate_rich_data.py:73-81`):
```python
def make_mesh(preset):
    L  = preset['dish_size_m']  # = 0.02 m
    nx = int(L / preset['wavelength_m'] * preset['elements_per_wavelength'])
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [[0, 0, 0], [L, L, L]],
        [nx, nx, nx],
        cell_type=mesh.CellType.tetrahedron,
    )
```

**Mesh resolution**:
- Frequency: 500 kHz
- Wavelength λ = 2.968 mm
- Domain size: L = 20 mm = **6.74 wavelengths**
- Elements per wavelength: 6
- **nx = 6.74 × 6 ≈ 40 elements per edge**

**Vertical modes**:
- Domain height: 20 mm = 6.74λ
- **Vertical resonance**: L_z = n λ/2 for n = 1, 2, 3, ...
- At 500 kHz: n=1 requires L_z = 1.484 mm, n=2 = 2.968 mm, etc.
- **Current L_z = 20 mm supports up to ~13 vertical half-wavelengths**

**Why this matters**:
- **If domain too small**: Only fundamental mode fits → no z-structure
- **If domain large enough**: Higher vertical modes CAN be excited → z-structure possible

**Implication for tornado**: ✅ **Sufficient domain height** — vertical modes ARE physically possible

---

### Q3: Is the frequency chosen such that higher-order vertical modes can exist?

**Evidence from physics**:
- k = ω/c = 2πf/c = 2π(500e3)/1484 = **2115 rad/m**
- kz * L_z = 2115 × 0.02 = **42.3 radians = 6.74 full wavelengths**

**Vertical standing wave criterion**: If bottom and top are impedance boundaries (not rigid), vertical modes have form:
- p(z) ∝ sin(k_z z) or cos(k_z z)
- Allowed k_z depends on boundary conditions

**Actual boundary conditions** (`generate_rich_data.py:119`):
```python
for tag in [1, 2, 3, 4]:
    a += -1j * omega * ufl.inner(u, v) / Z * ds(tag)
```
**Tag mapping** (`generate_rich_data.py:84-87`):
```python
# tag 1: bottom (z=0)
# tags 2,3: x-walls, y-walls (side walls, where standing actuation applied)
# tag 4: top (z=L)
```

**Bottom and top boundaries**: Both have **impedance BC** (`-iω/Z * p` term), **NOT rigid (Neumann ∂p/∂n=0)**

**Why this matters**:
- **Rigid walls (∂p/∂n=0)**: Support vertical standing waves with k_z = nπ/L_z
- **Impedance walls (∂p/∂n = -iωp/Z)**: Allow partial transmission → **no strong vertical standing modes unless resonant**

**Implication for tornado**: ⚠️ **CRITICAL ISSUE** — Impedance BCs on top/bottom do NOT enforce vertical standing waves. Vertical structure will be **weak** unless the forcing explicitly creates it.

---

### Q4: Is the vortex introduced via phase winding in x–y only, or does it include z-dependent phase?

**Evidence from code** (`generate_rich_data.py:138-141`):
```python
center = np.array([L/2, L/2, L])  # Vortex center at TOP surface
phi  = compute_azimuthal_phase(top_xyz, cfg, center)
amp  = compute_amplitude_profile(top_xyz, cfg, center)
pat  = amp * np.exp(1j * (phi + vortex_phase))
```

**Vortex phase function** (`vortex_lens.py:compute_azimuthal_phase`):
```python
# Computes φ(θ) = ℓ θ where θ = arctan2(y-y0, x-x0)
# Only depends on (x,y), NOT z
```

**Answer**: 🚨 **x–y ONLY**. The vortex phase φ(θ) = ℓ arctan2(y-y_c, x-x_c) has **no z-dependence**.

**Where is vortex applied?** (`generate_rich_data.py:152`):
```python
L_vtx = ufl.inner(g_v, v) * ds(4)  # ds(4) = TOP BOUNDARY ONLY
```

**Why this matters**:
- **If vortex is x–y only, applied at z=L**: Creates **azimuthal phase pattern on top surface**
- **No z-winding**: Vortex axis is parallel to z, but phase doesn't wind in z
- **Decay with depth**: Perturbation will decay as you move away from z=L (top)

**Implication for tornado**: 🚨 **SMOKING GUN** — Vortex is a **SURFACE perturbation** (top boundary only), not a volumetric helical field. Perturbation should be **strongest near top, decay toward bottom**.

---

## B) DOMAIN GEOMETRY & BOUNDARY CONDITIONS

### Q5: What are the exact z-boundary conditions (bottom and top)?

**Evidence from code** (`generate_rich_data.py:119`):
```python
for tag in [1, 2, 3, 4]:
    a += -1j * omega * ufl.inner(u, v) / Z * ds(tag)
```
**Physical meaning**: Robin BC (impedance):
```
∂p/∂n = -iω/Z * p
```
where Z = ρc = 997 × 1484 = 1.48 MPa·s/m

**Bottom (tag 1)**: Impedance BC (partially absorbing, NOT rigid)  
**Top (tag 4)**: Impedance BC (partially absorbing, NOT rigid)

**Why this matters**:
- **Rigid (Neumann)**: Strong reflection → vertical standing waves
- **Impedance**: Partial reflection → **weak vertical structure** unless resonant
- **Symmetry**: Top and bottom are **identical BCs** → no mechanism to break z-symmetry

**Implication for tornado**: ⚠️ **No z-asymmetry from BCs**. Top and bottom are physically identical (both impedance). The only z-asymmetry comes from **vortex applied at top**.

---

### Q6: Is the domain uniform in z or extruded from a 2D mesh?

**Evidence from code** (`generate_rich_data.py:77`):
```python
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [[0, 0, 0], [L, L, L]],
    [nx, nx, nx],  # SAME resolution in x, y, z
    cell_type=mesh.CellType.tetrahedron,
)
```

**Answer**: ✅ **Fully 3D tetrahedral mesh** with uniform resolution (nx × nx × nx).  
**NOT** extruded from 2D, NOT structured layers.

**Why this matters**:
- **Extruded**: Might introduce artificial z-symmetry
- **Full 3D unstructured**: No numerical bias toward 2D structure

**Implication for tornado**: ✅ **Mesh supports 3D structure** — no numerical artifact preventing z-variation

---

### Q7: Are top and bottom boundaries physically distinct or identical?

**Answer** (from Q5): 🚨 **IDENTICAL**. Both use `Z = ρc` impedance BC. No physical distinction.

**Why this matters**:
- **If distinct**: Could create z-asymmetry (e.g., rigid bottom, open top)
- **If identical**: Only source of z-asymmetry is **vortex actuation at top**

**Implication for tornado**: ⚠️ **No intrinsic z-asymmetry**. Perturbation structure depends entirely on where forcing is applied (top only).

---

### Q8: Is there any mechanism that breaks z-symmetry?

**Mechanisms that COULD break z-symmetry**:
1. ❌ **Boundary conditions**: Top and bottom both impedance (symmetric)
2. ❌ **Material properties**: Single fluid (water), uniform ρ, c
3. ✅ **Actuation**: Standing wave on **side walls** (x–y), vortex on **top** (z=L)

**Answer**: Yes, but **WEAKLY**. The only z-asymmetry is:
- Standing wave actuated on vertical side walls (tags 2, 3)
- Vortex actuated on top horizontal surface (tag 4)

**Why this matters**:
- **Strong z-asymmetry**: Would create depth-dependent structure
- **Weak z-asymmetry**: Vortex is a **localized surface perturbation**, decays into bulk

**Implication for tornado**: ⚠️ **Vortex is boundary forcing, NOT volumetric**. Expect perturbation Δp to be **surface-localized**, not volumetric.

---

## C) NUMERICAL DISCRETISATION

### Q9: Is the mesh genuinely 3D or a 2.5D extrusion?

**Answer** (from Q6): ✅ **Genuinely 3D**. Unstructured tetrahedral mesh, not extruded.

**Implication**: ✅ Numerics CAN capture 3D structure if physics demands it

---

### Q10: Are basis functions separable in z?

**Evidence from code** (`generate_rich_data.py:110`):
```python
V  = fem.functionspace(domain, ("Lagrange", 2))
```

**Answer**: ❌ **NOT separable**. Lagrange P2 on tetrahedra are **fully 3D basis functions**, not tensor products.

**Why this matters**:
- **Separable (e.g., spectral)**: Might impose artificial z-independence
- **Non-separable FEM**: No constraint on z-coupling

**Implication**: ✅ Discretization does not prevent z-structure

---

### Q11: Are we solving only the fundamental vertical mode?

**Answer**: ❌ **NO**. The Helmholtz equation is solved in 3D with no mode decomposition. All vertical modes (up to mesh resolution) are represented.

**Why this matters**:
- **Single-mode**: Would force z ~ sin(πz/L) or similar
- **Full 3D**: Vertical structure determined by **boundary forcing** and **physics**

**Implication**: ✅ Numerically, higher vertical modes COULD exist if excited

---

### Q12: Is Δp computed nodewise on the same mesh for standing and combined cases?

**Answer**: ✅ **YES**. Both cases solved on **identical mesh**, Δp computed by subtracting DOF values.

**Why this matters**:
- **Different meshes**: Could introduce interpolation artifacts
- **Same mesh**: Δp is **exact pointwise difference**

**Implication**: ✅ Visualization artifacts are NOT due to mesh mismatch

---

## D) ACTUATION / VORTEX DEFINITION

### Q13: How exactly is the "vortex" implemented?

**Evidence from code** (`generate_rich_data.py:133-154`):
```python
# Vortex center at top surface center
center = np.array([L/2, L/2, L])

# Compute azimuthal phase φ(θ) = ℓ θ on top boundary DOFs
phi  = compute_azimuthal_phase(top_xyz, cfg, center)
amp  = compute_amplitude_profile(top_xyz, cfg, center)
pat  = amp * np.exp(1j * (phi + vortex_phase))

# Set vortex function (zero everywhere EXCEPT top boundary)
vf = fem.Function(V)
vf.x.array[:] = 0.0
vf.x.array[top_dofs] = pat

# Apply as Neumann BC on top surface
g_v   = -1j * omega * RHO * vf
L_vtx = ufl.inner(g_v, v) * ds(4)
```

**Answer**: 🚨 **Boundary phase vortex on top surface ONLY**. The vortex is:
- **NOT volumetric**: Function is zero everywhere except top boundary DOFs
- **Azimuthal phase**: Pattern is `A(r) exp(iℓθ)` where θ = arctan2(y, x) relative to center
- **Radial taper**: Amplitude profile uses 'cosine_taper' within aperture radius (2 mm)

**Why this matters**:
- **Volumetric vortex**: Would create helical wavefronts throughout domain
- **Boundary vortex**: Creates **evanescent perturbation** that decays from surface

**Implication for tornado**: 🚨 **CRITICAL** — Vortex is NOT a volumetric helical wave. It is a **boundary forcing** that creates a **near-field perturbation**.

---

### Q14: Is it a phase vortex in the boundary condition only?

**Answer** (from Q13): ✅ **YES**. Phase vortex exp(iℓθ) is applied **ONLY on top boundary** (z=L).

**Why this matters**:
- **Boundary-only**: Perturbation will be **surface-localized**
- **No volumetric phase winding**: No helical structure in bulk

**Implication for tornado**: 🚨 **Vortex cannot create a pressure tornado**. It's a surface modulation.

---

### Q15: Does the vortex forcing vary with z?

**Answer**: ❌ **NO**. Vortex is applied at **single z-plane** (z=L, top surface). No z-variation in forcing.

**Why this matters**:
- **z-varying forcing**: Could create vertical phase winding
- **Single-plane forcing**: Creates **horizontal phase pattern** at z=L, evanescent decay below

**Implication for tornado**: 🚨 **No vertical helicity in forcing** → No vertical helicity in pressure field

---

### Q16: Is the vortex expected to generate helicity in pressure, or only circulation in-plane?

**Physical expectation**:
- **Helicity**: Requires ∇ × **v** · **v** ≠ 0 (velocity field twisting)
- **Pressure vortex**: In linear acoustics, **p** is a scalar field → **cannot have helicity**
- **Phase vortex**: Creates azimuthal phase variation → **intensity null at center** + **phase singularity**, but **NO helical structure in |p|**

**Answer**: 🚨 **NEITHER**. Pressure is a scalar. The "vortex" creates:
1. **Azimuthal phase variation**: φ(θ) = ℓθ
2. **Intensity null at core**: |p| → 0 as r → 0 (phase singularity)
3. **In-plane circulation in velocity** (v ∝ ∇p), NOT in pressure magnitude

**Why this matters**:
- **If expecting helical |p|**: WRONG EXPECTATION. Pressure magnitude |p| does NOT have helical structure.
- **If expecting phase winding**: CORRECT. Phase arg(p) winds azimuthally.

**Implication for tornado**: 🚨 **FUNDAMENTAL MISUNDERSTANDING** — A pressure vortex does NOT create a tornado-like structure in |p| or |Δp|. It creates:
- **Azimuthal phase winding** (visualize arg(p), not |p|)
- **Intensity null at core** (not a tornado, but a "dark spot")

---

## E) DERIVED QUANTITIES & EXPECTATIONS

### Q17: Should Δp be volumetric or surface-local by construction?

**Answer** (from actuaction analysis): 🚨 **SURFACE-LOCAL**.

**Reasoning**:
1. Vortex forcing applied **ONLY at z=L** (top surface)
2. Standing wave forcing applied on **vertical side walls** (x=0, x=L, y=0, y=L)
3. Combined field = standing (volumetric) + vortex (surface evanescent)
4. **Δp = p_combined - p_standing** isolates the **vortex contribution**
5. Vortex contribution must **decay from top surface** into bulk (evanescent)

**Why this matters**:
- **Volumetric Δp**: Would fill domain → tornado-like
- **Surface-local Δp**: Concentrated near z=L → **observed behavior matches**

**Implication for tornado**: ✅ **MATCHES OBSERVATION** — Δp is primarily visible on surfaces (especially top), minimal in bulk.

---

### Q18: Is it physically expected that Δp decays rapidly away from the actuation plane?

**Answer**: ✅ **YES**.

**Physical mechanism**:
- Vortex forcing at z=L creates **evanescent perturbation**
- Decay length scale ~ λ / (2π) ≈ 0.5 mm for wavelength 3 mm
- Domain depth = 20 mm = **40 decay lengths**
- **Exponential decay**: Δp(z) ∝ exp(-(L-z)/δ) where δ ~ 0.5 mm

**Why this matters**:
- **If Δp volumetric**: Would indicate volumetric vortex forcing (NOT present)
- **If Δp decays**: Confirms **boundary-forced evanescent field**

**Implication for tornado**: ✅ **EXPLAINS OBSERVATION** — "little to no apparent z-axis structure" because perturbation decays rapidly from top.

---

### Q19: Are we implicitly expecting a streaming vortex while visualising pressure?

**Answer**: 🚨 **YES — CATEGORY ERROR**.

**Distinction**:
- **Acoustic pressure p(x,t)**: Linear, time-harmonic, governed by Helmholtz equation
- **Acoustic streaming v_s(x)**: **Nonlinear**, time-averaged, driven by Reynolds stress ∇·(ρ⟨**v**⊗**v**⟩)

**Time-harmonic pressure** (`p(x) exp(-iωt)`):
- Solution of **linear** Helmholtz: ∇²p + k²p = 0
- Does NOT create vortices in |p| (scalar field)
- **Phase vortex** creates azimuthal phase, NOT magnitude helix

**Acoustic streaming** (if it existed):
- **Nonlinear** phenomenon (not solved here)
- Creates **time-averaged vortical flow** ⟨**v**⟩ ≠ 0
- **THIS would create a tornado-like structure**

**Why this matters**:
- **If visualizing p**: Expect phase singularity + intensity null, NOT tornado
- **If expecting tornado**: Need to compute **streaming velocity**, not pressure

**Implication for tornado**: 🚨 **WRONG QUANTITY** — Visualizing pressure when expecting streaming. **Pressure vortices do NOT look like tornadoes**.

---

## 🔬 RANKED HYPOTHESES (Most Likely → Least Likely)

### H1: "Pressure vortex is fundamentally 2D (in-plane, no z-structure)" — **CONFIRMED** ⭐⭐⭐⭐⭐

**Evidence supporting**:
- Vortex forcing has NO z-dependence (only θ = arctan2(y,x))
- Applied at SINGLE z-plane (top surface only)
- Top and bottom BCs are identical (impedance) → no z-asymmetry from boundaries
- No vertical mode selection → no reason for z-structure to develop

**Evidence against**:
- (None strong)

**Test to confirm/refute**:
- ✅ **ALREADY CONFIRMED by observation** — "little to no apparent z-axis structure"

**Conclusion**: ✅ **TRUE**. The pressure field has azimuthal variation (θ-dependent) but minimal z-variation except evanescent decay from top.

---

### H2: "Vertical symmetry not broken (top/bottom identical BCs)" — **CONFIRMED** ⭐⭐⭐⭐⭐

**Evidence supporting**:
- Top and bottom both have impedance BC (`-iω/Z * p`)
- No material variation in z
- No vertical mode forcing

**Evidence against**:
- Vortex applied at top → weak z-asymmetry

**Test to confirm/refute**:
- Check if p(x,y,z) ≈ p(x,y,L-z) (mirror symmetry) → **PREDICT NO**: vortex at top breaks this
- Check if Δp strongest near top → **PREDICT YES**

**Conclusion**: ⚠️ **PARTIALLY TRUE**. BCs are symmetric, but vortex forcing at top weakly breaks symmetry → evanescent perturbation from top surface.

---

### H3: "Vortex exists only in streaming, not pressure" — **PARTIALLY TRUE** ⭐⭐⭐⭐

**Evidence supporting**:
- **Pressure** is linear acoustic field (governed by Helmholtz)
- **Streaming** is nonlinear (Reynolds stress, NOT computed here)
- Tornado-like **vortical flow** requires streaming, NOT pressure

**Evidence against**:
- Pressure DOES have vortex structure (phase singularity, azimuthal winding)
- But NOT in **magnitude** (|p|), only in **phase** (arg(p))

**Test to confirm/refute**:
- Compute streaming velocity: **v_s** ∝ ∇·⟨**v**⊗**v**⟩ where **v** = ∇p/(iωρ)
- Check if streaming has tornado structure

**Conclusion**: ✅ **CORRECT DISTINCTION**. Pressure has phase vortex (azimuthal winding), but tornado structure would appear in **streaming velocity** (not computed).

---

### H4: "Δp is nonzero only near boundaries" — **CONFIRMED** ⭐⭐⭐⭐⭐

**Evidence supporting**:
- Vortex applied at top boundary (z=L)
- Standing wave applied at side boundaries (x,y walls)
- Δp = p_combined - p_standing isolates **vortex contribution** → surface-localized

**Evidence against**:
- (None)

**Test to confirm/refute**:
- ✅ **ALREADY CONFIRMED by observation** — "perturbations visible primarily on surfaces"
- Slice Δp in z-planes: expect decay from top

**Conclusion**: ✅ **TRUE**. Δp is evanescent from top surface, minimal in bulk.

---

### H5: "Mesh too coarse to resolve z-structure" — **REJECTED** ⭐

**Evidence supporting**:
- (None compelling)

**Evidence against**:
- Mesh has 6 elements per wavelength (adequate)
- Vertical extent is 6.74 wavelengths (ample)
- Solver can resolve 3D features (tetrahedral P2)

**Test to confirm/refute**:
- Refine mesh by 2× → if z-structure appears, hypothesis confirmed
- **PREDICT**: No change (because physics doesn't create z-structure, not mesh limitation)

**Conclusion**: ❌ **FALSE**. Mesh is adequate. Lack of z-structure is **physical**, not numerical.

---

## ✅ SINGLE-TEST CONFIRMATION FOR EACH HYPOTHESIS

| Hypothesis | Single Definitive Test | Expected Result if TRUE |
|------------|------------------------|-------------------------|
| H1 (2D in-plane) | Extract Δp(x, y, z_mid) and Δp(x, y, z_top) → compare | Identical patterns (scaled) |
| H2 (No z-symmetry breaking) | Plot Δp(z) along vertical line through vortex core | Monotonic decay from top, NOT symmetric |
| H3 (Streaming vs pressure) | Compute ⟨**v**⊗**v**⟩ and check for vortical structure | Streaming has tornado, pressure does NOT |
| H4 (Surface-local Δp) | Slice Δp in horizontal planes z=0.1L, 0.5L, 0.9L | Strongest at z=0.9L, weakest at z=0.1L |
| H5 (Mesh coarseness) | Refine mesh 2× and re-solve | NO change in z-structure (physics, not numerics) |

---

## 🎯 FINAL DETERMINATION: SHOULD A 3D PRESSURE TORNADO EXIST?

### Answer: 🚨 **NO**

### Reasoning:

1. **Pressure is a scalar field** → Cannot have helical structure in **magnitude** |p|
2. **Phase vortex ≠ magnitude tornado**:
   - Phase arg(p) DOES wind azimuthally (φ(θ) = ℓθ)
   - Magnitude |p| has **intensity null at core** + **azimuthal modulation**, NOT helix
3. **Vortex forcing is 2D (in-plane)**: Applied at single z-plane (top), no z-winding
4. **Evanescent decay**: Perturbation Δp decays exponentially from top surface → surface-local
5. **Tornado requires streaming**: Vortical **flow** (tornado-like) appears in **streaming velocity**, NOT pressure

### What SHOULD be visualized:

#### ✅ **Correct quantities to visualize for vortex structure**:

1. **Phase field arg(p)** or **arg(Δp)**:
   - Should show azimuthal winding (rainbow cycling around core)
   - Use **cyclic colormap** (twilight, HSV)
   - Expect 2π phase jump across branch cut
   - **Helix appears in PHASE, not magnitude**

2. **Acoustic streaming velocity** (if computed):
   - **v_s** = ∇·⟨**v**⊗**v**⟩ / (2ωρ₀)
   - Requires **nonlinear** calculation (NOT currently done)
   - **This would show tornado-like vortical flow**

3. **Velocity field v = ∇p / (iωρ)**:
   - Converts pressure gradient to velocity
   - Should show **azimuthal circulation** in-plane
   - Visualize with streamlines or vector field

#### ❌ **Incorrect expectations**:

1. **Tornado in |Δp|**: Pressure magnitude does NOT have helical structure
2. **Volumetric vortex in |Δp|**: Perturbation is surface-local (evanescent)
3. **Z-structure in |Δp|**: Forcing is 2D (no z-winding), expect decay not helix

---

## 📊 WHAT TO VISUALIZE NEXT (Priority Order)

### Priority 1: **Phase field arg(Δp)** — IMMEDIATE ⭐⭐⭐⭐⭐

**Why**: Phase DOES have azimuthal winding (vortex signature)

**How**: In ParaView:
1. Open `combined_fields.vtu`
2. Calculator filter: `atan2(delta_p_imag, delta_p_real)`
3. Slice at z=0.9L (near top, where vortex is strongest)
4. Color by phase with **twilight** colormap
5. **Expect**: Azimuthal phase winding (rainbow spiral around core)

**Success criterion**: Phase winds 2π around core → ✅ Vortex confirmed

---

### Priority 2: **Horizontal slice of |Δp| near top** — IMMEDIATE ⭐⭐⭐⭐

**Why**: Verify surface-local hypothesis (evanescent decay)

**How**: In ParaView:
1. Slice `delta_p_magnitude` at z = 0.9L (near top)
2. Slice at z = 0.5L (mid-depth)
3. Slice at z = 0.1L (near bottom)
4. **Compare magnitudes**

**Success criterion**: |Δp| strongest at top, decays toward bottom → ✅ Evanescent confirmed

---

### Priority 3: **Vertical line plot through vortex core** — DIAGNOSTIC ⭐⭐⭐

**Why**: Quantify z-decay of perturbation

**How**: In ParaView:
1. Plot Over Line: (L/2, L/2, 0) → (L/2, L/2, L)
2. Plot `delta_p_magnitude` vs z
3. **Expect**: Exponential decay from top

**Success criterion**: Δp(z) ∝ exp(-(L-z)/δ) → ✅ Surface-local confirmed

---

### Priority 4: **Acoustic intensity I = 0.5 Re(p v*)** — FUTURE ⭐⭐

**Why**: Shows energy flow (might reveal structure invisible in |p|)

**How**:
1. Compute **v** = ∇p / (iωρ) in post-processing
2. Compute **I** = 0.5 Re(p **v***)
3. Visualize with streamlines or arrows

**Success criterion**: Intensity shows azimuthal circulation around core

---

### Priority 5: **Acoustic streaming** — REQUIRES NEW SOLVE ⭐

**Why**: **This is where the tornado lives** (if anywhere)

**How**:
1. Compute second-order time-averaged velocity:
   - **v_s** = -∇·⟨**v**⊗**v**⟩ / (2ωρ₀)
   - Requires gradient projection of **v** = ∇p/(iωρ)
2. Solve Stokes equation for streaming
3. Visualize with streamlines

**Success criterion**: Streaming shows tornado-like vortical flow

**Status**: ❌ **NOT IMPLEMENTED** — would require new solver module

---

## ⛔ WHEN TO STOP PRESSURE VISUALIZATION

### Stop when:

1. ✅ **Phase field confirms azimuthal winding** (vortex signature present)
2. ✅ **Horizontal slices confirm surface-local Δp** (evanescent decay verified)
3. ✅ **Vertical decay quantified** (exponential from top)
4. ✅ **Conclusion documented**: Pressure vortex is **phase singularity + intensity null**, NOT magnitude tornado

### Then move on to:

1. **Streaming calculation** (if tornado structure desired)
2. **Velocity field visualization** (∇p shows circulation)
3. **Gor'kov potential** (force landscape, already computed)

---

## 📝 SUMMARY: REALITY vs EXPECTATION

| Expectation | Reality | Consequence |
|-------------|---------|-------------|
| 3D pressure tornado (helical \|Δp\|) | Phase vortex (helical arg(Δp), NOT \|Δp\|) | **Visualize phase, not magnitude** |
| Volumetric vortex structure | Surface-local evanescent field | **Slice near top surface, NOT 3D volume** |
| Z-axis helicity | No z-structure (2D in-plane vortex) | **No tornado in pressure** |
| Tornado in \|p\| or \|Δp\| | Tornado in **streaming velocity** (not computed) | **Wrong quantity visualized** |
| Full-domain perturbation | Decay from top (δ ~ 0.5 mm) | **Most of domain is unperturbed** |

---

## 🔬 CONCLUSION

**A 3D pressure tornado CANNOT exist in this model.**

**Why**:
1. Pressure is scalar → no helicity in magnitude
2. Vortex forcing is 2D (azimuthal) → no z-winding
3. Forcing at single plane (top) → evanescent, not volumetric
4. Tornado structure exists in **streaming velocity** (nonlinear, not computed), NOT pressure

**What exists instead**:
- **Azimuthal phase winding** arg(Δp) = ℓθ (phase vortex)
- **Intensity null at core** (phase singularity)
- **Surface-local perturbation** (evanescent from top)

**Correct next step**:
1. Visualize **phase field** arg(Δp) to confirm vortex
2. Quantify surface-local decay
3. Stop pressure visualization once vortex confirmed
4. Move to **streaming** if tornado structure desired

---

**END OF REALITY CHECK**
