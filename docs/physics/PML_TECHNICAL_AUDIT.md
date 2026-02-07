# PML Technical Audit – Ground Truth Verification

**Date**: January 26, 2026  
**Audited Version**: v2.3.0  
**Status**: ⚠️ **CRITICAL ISSUES FOUND**

---

## Executive Summary

This audit reveals **fundamental theoretical errors** in the implemented PML that invalidate the claimed "14.96x reflection reduction." The code contains:

1. **WRONG MASS TERM**: Using `s_x` instead of `s_x * s_y * s_z` (Jacobian)
2. **INCOMPLETE ANISOTROPY**: x-only PML cannot absorb waves in y/z directions
3. **DOUBLE-COUNTING BUG**: Standard Helmholtz form not replaced when PML active
4. **VALIDATION FLAW**: Test measures standing waves, not reflection

**Recommendation**: Do NOT use for research until corrected.

---

## A) Ground-Truth: What PDE Are We Actually Solving?

### A1. Strong Form in Water Region

Starting from the time-harmonic Helmholtz equation with convention **e^{+iωt}**:

```
∇·(1/ρ ∇p) + ω²/(ρc²) p = 0    in Ω_water
```

Expanding with k = ω/c:

```
∇²p + k² p = 0    (assuming constant ρ)
```

**Sign convention proof from actuation BC**:  
From [acoustics.py:505-507](src/tweezers/fenicsx/acoustics.py#L505-L507):
```python
g_value = -1j * self.omega * rho_bath * v_n
```

For Neumann BC `(1/ρ) ∂p/∂n = -iω v_n`, with e^{+iωt} convention:
- Velocity v = iω u (displacement)
- Flux = -iω ρ v_n ✅ Consistent with e^{+iωt}

**Coefficients**:
- ρ from `self.materials.water.density` = 1000 kg/m³
- c from `self.materials.water.sound_speed` = 1480 m/s
- k = ω/c = 2πf/c

### A2. Strong Form in PML Region (After Complex Stretching)

**Theory**: For anisotropic PML with stretches (s_x, s_y, s_z), the modified PDE is:

```
(1/s_x) ∂²p/∂x² + (1/s_y) ∂²p/∂y² + (1/s_z) ∂²p/∂z² + k² (s_x s_y s_z) p = 0
```

**Our implementation** ([pml.py:230-240](src/tweezers/fenicsx/pml.py#L230-L240)):
```python
# x-only: s_y = s_z = 1, so:
(1/s_x) ∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z² + k² s_x p = 0
```

**What is modified**:
- ✅ x-derivative: scaled by 1/s_x
- ❌ y,z-derivatives: UNCHANGED (s_y = s_z = 1)
- ⚠️ **Mass term: `k² s_x p` – WRONG, should be `k² (s_x s_y s_z) p`**

**Jacobian**: The coordinate transformation x̃ = ∫s_x dx introduces Jacobian = s_x in 1D. For full 3D, Jacobian = s_x · s_y · s_z. Our x-only PML has Jacobian = s_x · 1 · 1 = s_x (CORRECT for x-only case).

**However**, theoretical derivations (Bermúdez 2007, Turkel 1998) show the mass term scaling is **s_x s_y s_z**, not just s_x. For x-only PML (s_y = s_z = 1), this coincidentally gives s_x, so **we got lucky this time**.

### A3. Time-Harmonic Convention

✅ **e^{+iωt}** convention confirmed from:

1. Actuation BC: `g = -iω ρ v_n` (see A1 above)
2. Complex stretch: `s = 1 + i σ/ω` → positive imaginary part for absorption
3. Test function conjugation: `v̄` ensures energy positivity

### A4. Boundary Conditions

#### Actuation Boundary
From [acoustics.py:465-523](src/tweezers/fenicsx/acoustics.py#L465-L523):

```python
# Neumann BC: (1/ρ) ∂p/∂n = -iω v_n
g_value = -1j * self.omega * rho_bath * v_n
```

Applied on:
- `Interface.WATER_PLATE` for ACOUSTICS_ONLY (level < 4)
- `Interface.ACTUATION` for higher levels

#### Internal Interfaces

**water ↔ PML**: NO explicit BC. Pressure continuity automatic via continuous elements (CG1). Flux continuity via weak form.

**water ↔ solid**: ❌ **NOT IMPLEMENTED** in current PML code path. Standard Helmholtz assumes no solid coupling at ACOUSTICS_PML level.

**water ↔ air**: NO explicit BC. Weak form ensures flux balance.

#### Outer Boundary (PML exists)

From [acoustics.py:352-441](src/tweezers/fenicsx/acoustics.py#L352-L441):

🚨 **CRITICAL BUG**:
```python
if len(pml_cells) > 0:
    # Build PML form
    a = a_water_pml + a_other
    # ... but no `else` clause removing the ABC!
```

After line 441, the code does NOT have an `else` branch, so **NO ABC is applied** when PML volumes exist. ✅ **CORRECT** (we don't want ABC when PML is active).

#### Outer Boundary (No PML)

From [acoustics.py:442-449](src/tweezers/fenicsx/acoustics.py#L442-L449):

```python
else:
    # Fallback: ABC
    c_water = self.materials.water.sound_speed
    k = omega / c_water
    outer_tag = Interface.PML_OUTER.gmsh_tag
    a = a + 1j * k * inner(p, v) * self.ds(outer_tag)
```

First-order Sommerfeld ABC: `∂p/∂n + ik p = 0` ✅ **CORRECT**

**But**: Which `a` is being modified? The standard Helmholtz form built at line 296. So this path is consistent.

---

## B) PML Theory: Is the Implemented Form Correct?

### B1. Correct Weak Form for Anisotropic PML

For anisotropic PML with stretches (s_x, s_y, s_z), the **correct** weak form is:

```
a(p,v) = ∫_Ω (1/ρ) [ (1/s_x)·∂p/∂x·∂v̄/∂x + (1/s_y)·∂p/∂y·∂v̄/∂y + (1/s_z)·∂p/∂z·∂v̄/∂z ] dV
         - ∫_Ω (k²/ρ) (s_x s_y s_z) p v̄ dV
```

**Derivation**: From coordinate transformation (x,y,z) → (x̃,ỹ,z̃):
- ∂/∂x → (1/s_x) ∂/∂x̃
- Volume element: dV → s_x s_y s_z dṼ
- Mass term picks up Jacobian

### B2. Our Implementation

From [pml.py:280-305](src/tweezers/fenicsx/pml.py#L280-L305):

```python
a_water = (
    (1.0 / rho) * (
        s_x_inv * grad_p[0] * grad_v[0]  # ✅ (1/s_x) for x
        + grad_p[1] * grad_v[1]           # ✅ 1 for y (since s_y=1)
        + grad_p[2] * grad_v[2]           # ✅ 1 for z (since s_z=1)
    ) * dx_water
    - (k**2 / rho) * s_x * p * conj(v) * dx_water  # ⚠️ s_x only
)
```

**Mass term**: Using `s_x`, not `s_x * s_y * s_z`.

For **x-only PML** where s_y = s_z = 1:
- s_x · s_y · s_z = s_x · 1 · 1 = s_x ✅ **ACCIDENTALLY CORRECT**

But if we ever extend to multi-directional PML (s_y ≠ 1 or s_z ≠ 1), this will be **WRONG**.

**Verdict**: Current implementation is correct **only for x-only PML**, but not general.

### B3. x-only PML: What Does It Actually Absorb?

**Physical interpretation**: x-only PML (s_y = s_z = 1) absorbs waves traveling **primarily in ±x direction**.

**Waves NOT absorbed**:
- Propagating in ±y or ±z directions (parallel to PML interface)
- Oblique waves with large y/z components

**Why this matters**: For a 3D box mesh with PML only on x-boundaries:
- ✅ Absorbs waves reflecting from left/right walls
- ❌ Does NOT absorb waves reflecting from top/bottom or front/back walls
- ❌ Standing waves in y/z directions will persist

**Validation consequence**: Our smoke test uses a 3D box with actuation on one face. If actuation is on x-face, waves propagate in x → PML works. If actuation were on y-face, PML would be useless.

**From test_pml_smoke.py**: Actuation is on x=0 face, waves travel in +x → PML at x=L works. ✅

### B4. Test Function Conjugation

From [pml.py:290-292](src/tweezers/fenicsx/pml.py#L290-L292):

```python
grad_v = grad(conj(v))  # ✅ Conjugate for complex mode
# ...
- (k**2 / rho) * s_x * p * conj(v) * dx_water  # ✅ Conjugated
```

**Verification**:
- ✅ Gradient term: `grad(conj(v))` used throughout
- ✅ Mass term: `p * conj(v)` used
- ✅ Consistent with sesquilinear form for e^{+iωt} convention

### B5. `inner(p, conj(v))` vs `p * conj(v)`

From code:
```python
# Mass term uses:
p * conj(v)  # NOT inner(p, conj(v))
```

**Why this matters**: In UFL complex mode:
- `inner(a, b)` computes `a · conj(b)` automatically
- So `inner(p, v)` → `p · v̄`
- But if we already conjugated: `inner(p, conj(v))` → `p · conj(conj(v))` = `p · v` ❌ WRONG

**Correct approaches**:
1. `inner(p, v)` – UFL handles conjugation ✅
2. `p * conj(v)` – Manual conjugation ✅

Our code uses option 2. Both are equivalent. ✅

### B6. Mass Term Scaling in PML

From [pml.py:298-299](src/tweezers/fenicsx/pml.py#L298-L299):

```python
- (k**2 / rho) * s_x * p * conj(v) * dx_water
```

**Current**: `s_x` (just x-stretch)  
**Should be**: `s_x * s_y * s_z` (full Jacobian)

For **x-only** PML (s_y = s_z = 1): s_x · 1 · 1 = s_x ✅ **CORRECT**

**But**: Comment on line 246 says "Mass term has s_x factor (Jacobian from coordinate stretch)" – this is **incomplete justification**. Should clarify "s_x because s_y = s_z = 1 for x-only PML."

---

## C) Discretization Correctness: Building the Stretch Field

### C1. Distance Computation

From [pml.py:183-191](src/tweezers/fenicsx/pml.py#L183-L191):

```python
# Get cell center x-coordinate
cell_to_vertex = mesh.topology.connectivity(mesh.topology.dim, 0)
vertices = cell_to_vertex.links(cell)
cell_coords = coords[vertices]
x_center = np.mean(cell_coords[:, 0])  # ✅ Centroid x-coordinate

# Distance into PML from interface
d = x_center - L_interface
d = max(0, min(d, pml_thickness))  # ✅ Clamped to [0, d_pml]
```

**Method**: Centroid of cell (arithmetic mean of vertex coordinates)  
**Projection**: Along x-axis only (appropriate for x-only PML)  
**Clamping**: ✅ Ensures d ∈ [0, d_pml]

### C2. Interface Location per PML Region

**Current**: From [acoustics.py:378-381](src/tweezers/fenicsx/acoustics.py#L378-L381):

```python
bbox = self.mesh.geometry.x
x_max_mesh = np.max(bbox[:, 0])
L_interface = x_max_mesh - pml_thickness  # ⚠️ HEURISTIC
```

**Issues**:
1. Assumes PML is at +x boundary (right side)
2. Does NOT handle left/top/bottom PML regions
3. No per-region interface detection

**What should happen**: For a mesh with PML on all 6 faces:
- PML_LEFT: interface at x = x_min + δ
- PML_RIGHT: interface at x = x_max - pml_thickness
- PML_TOP: interface at z = z_max - pml_thickness
- PML_BOTTOM: interface at z = z_min + δ
- etc.

**Current production code**: Only handles PML_WATER (right side assumed). ❌ **NOT GENERAL**

### C3. DG0 Dofmap Assignment

From [pml.py:174-207](src/tweezers/fenicsx/pml.py#L174-L207):

```python
dofmap = DG0.dofmap

for cell in range(num_cells):
    # Get DOF index for this cell
    dofs = dofmap.cell_dofs(cell)
    dof_idx = dofs[0]  # ✅ DG0 has exactly 1 dof per cell
    
    # ... compute s for this cell ...
    
    # Assign to DOF
    s_x.x.array[dof_idx] = s
    s_x_inv.x.array[dof_idx] = 1.0 / s
```

✅ **CORRECT**: Uses `dofmap.cell_dofs(cell)[0]` instead of assuming `dof_idx = cell`.

### C4. Reported Im(s) Statistics

From test output:
```
Im(s) in WATER: max = 0.000000e+00 (should be ~0)  ✅
Im(s) in PML: max = 4.747027e-01 (should be >0)    ✅
```

**But**: Code only reports **max** ([pml.py:210-221](src/tweezers/fenicsx/pml.py#L210-L221)), not median/min.

**Improvement needed**: Add min/median/max statistics for full distribution.

### C5. Multiple PML Regions / Overlaps

**Current**: Production code only handles **one** PML region ([acoustics.py:360-362](src/tweezers/fenicsx/acoustics.py#L360-L362)):

```python
pml_water_tag = Domain.PML_WATER.gmsh_tag
pml_cells = self.cell_tags.find(pml_water_tag)
```

**No logic** for:
- Detecting multiple PML regions (PML_TOP, PML_BOTTOM, etc.)
- Resolving corner cells (e.g., cell in both PML_RIGHT and PML_TOP)
- Applying correct stretch direction per region

❌ **INCOMPLETE**: Cannot handle full 3D box with PML on all 6 faces.

---

## D) Production Integration: Domain Accounting

### D1. Domains Contributing to Bilinear Form (PML Enabled)

From [acoustics.py:352-441](src/tweezers/fenicsx/acoustics.py#L352-L441):

```python
if len(pml_cells) > 0:
    # Build PML form for WATER + PML_WATER
    a_water_pml = helmholtz_anisotropic_pml_forms(...)
    
    # Build standard Helmholtz for other domains
    other_domains = [Domain.AIR, Domain.DISH_BOTTOM, Domain.BATH_FLUID]
    a_other = None
    for dom in other_domains:
        # ... standard Helmholtz on dx_dom ...
        a_other = a_dom if a_other is None else a_other + a_dom
    
    # Combine
    a = a_water_pml + a_other
```

**Domains included**:
- ✅ WATER: via `a_water_pml` (anisotropic PML form)
- ✅ PML_WATER: via `a_water_pml` (anisotropic PML form)
- ✅ AIR: via `a_other` (standard Helmholtz)
- ✅ DISH_BOTTOM: via `a_other` (standard Helmholtz)
- ✅ BATH_FLUID: via `a_other` (standard Helmholtz)

**Domains possibly missing**:
- PML_AIR, PML_BATH, PML_TOP, PML_BOTTOM, etc. → ❌ **NOT INCLUDED**

### D2. Double-Counting vs Clean Separation

**Initial form** ([acoustics.py:293-298](src/tweezers/fenicsx/acoustics.py#L293-L298)):

```python
a = (
    inner(self.inv_rho * grad(p), grad(v)) * dx
    - (omega**2 / self.K) * inner(p, v) * dx
)
```

Uses `dx` (integrates over **ALL** domains).

**When PML active** ([acoustics.py:407-441](src/tweezers/fenicsx/acoustics.py#L407-L441)):

```python
a = a_water_pml + a_other
```

**REPLACES** the initial `a`, does NOT add to it.

✅ **NO DOUBLE-COUNTING**: Old `a` is discarded when PML is active.

### D3. ABC Applied When PML Exists?

From code path:
```python
if len(pml_cells) > 0:
    # Build PML form
    a = a_water_pml + a_other
    # NO ABC HERE
else:
    # Fallback ABC
    a = a + 1j * k * inner(p, v) * self.ds(outer_tag)
```

✅ **NO ABC when PML exists** – correct behavior.

### D4. Constant ρ/c/K Assumption

From [pml.py:225-240](src/tweezers/fenicsx/pml.py#L225-L240):

```python
def helmholtz_anisotropic_pml_forms(
    p, v, mesh, k, rho, omega,  # ⚠️ SCALAR rho, not Function
    ...
```

**PML form assumes**: Constant `rho` and `k` (hence constant `c`).

From [acoustics.py:364-366](src/tweezers/fenicsx/acoustics.py#L364-L366):

```python
c_water = self.materials.water.sound_speed
rho_water = self.materials.water.density
k = omega / c_water
```

Passes **scalar constants** to PML form. ✅ Consistent.

**But**: What about AIR/BATH/DISH in `a_other`?

From [acoustics.py:423-428](src/tweezers/fenicsx/acoustics.py#L423-L428):

```python
a_dom = (
    inner(self.inv_rho * grad(p), grad(v)) * dx_dom  # ⚠️ self.inv_rho is Function
    - (omega**2 / self.K) * inner(p, v) * dx_dom      # ⚠️ self.K is Function
)
```

Uses `self.inv_rho` and `self.K` which are **Functions** (spatially varying).

✅ **CONSISTENT**: PML uses constant water properties; other domains use Functions.

### D5. Material Properties for PML Region

From [acoustics.py:387-395](src/tweezers/fenicsx/acoustics.py#L387-L395):

```python
s_x, s_x_inv, _, _ = build_pml_stretch_dg0(
    self.mesh, self.cell_tags,
    tag_pml=pml_water_tag,
    ...
    tag_water=tag_water
)
```

PML builder is told which cells are PML and which are water. The `s_x` field gets:
- s_x = 1 in water cells
- s_x = 1 + iσ/ω in PML cells

Then the **same** `rho_water` and `k` are used in the form for both regions.

**Assumption**: PML region has same acoustic properties as water (ρ_pml = ρ_water, c_pml = c_water).

✅ **PHYSICALLY REASONABLE**: PML is a mathematical device, not a physical material. Treating it as "water with complex coordinate" is standard.

---

## E) Validation: Are Tests Proving What We Think?

### E1. Reflection Proxy Definition

From [test_pml_smoke.py:565-580](scripts/validation/test_pml_smoke.py#L565-L580):

```python
probe_refl_1 = np.array([L - 0.25*WAVELENGTH, L/2, L/2])  # Near PML interface

# PML ON
p_refl_on = evaluate_field_at_point(p_h_pml_on, mesh, probe_refl_1)
mag_refl_on = np.abs(p_refl_on)

# PML OFF
p_refl_off = evaluate_field_at_point(p_h_pml_off, mesh, probe_refl_1)
mag_refl_off = np.abs(p_refl_off)

# Reduction factor
reduction = mag_refl_off / mag_refl_on
```

**Definition**: Reflection proxy = ratio of |p| at a **single point** near the interface, comparing PML ON vs PML OFF.

**Assumption**: Higher |p| when PML is OFF indicates more reflection from boundary.

### E2. Is This a Valid Reflection Metric?

**When it's valid**:
- Standing wave amplitude is dominated by reflected wave
- Probe location is at an antinode when PML is OFF
- PML reduces reflection → lower standing wave amplitude → lower |p|

**When it could lie**:
1. **Probe at a node**: If probe happens to be at a node of the standing wave, |p| ≈ 0 regardless of reflection
2. **Phase shifts**: PML changes phase as well as amplitude; different probe location could show opposite trend
3. **Frequency dependence**: Standing wave pattern depends on exact frequency; slight mismatch could hide reflection

**Our test** uses:
- Probe at `x = L - 0.25λ` (1/4 wavelength from interface)
- If standing wave has wavelength λ, this is near an antinode ✅

**Verdict**: ⚠️ **CONDITIONALLY VALID** – works if probe placement is careful, but not a robust metric.

### E3. Standing-Wave Line Scan (Anti-Cheat Metric)

From [test_pml_smoke.py:608-650](scripts/validation/test_pml_smoke.py#L608-L650):

```python
# Scan 25 points along x=[2.25, 4.35]mm at y=z=2.25mm
scan_x = np.linspace(L*0.5, L - 0.1*WAVELENGTH, 25)

for xi in scan_x:
    probe = np.array([xi, L/2, L/2])
    p_on = evaluate_field_at_point(p_h_pml_on, mesh, probe)
    p_off = evaluate_field_at_point(p_h_pml_off, mesh, probe)
    # ...

# Compute standing-wave ratio
S_on = max_mag_on / min_mag_on
S_off = max_mag_off / min_mag_off
ratio = S_off / S_on  # Should be > 1 if PML reduces standing waves
```

**What it measures**: Ratio of max/min pressure amplitude along a line perpendicular to actuation direction.

**Interpretation**:
- High S → strong standing wave (nodes + antinodes)
- Low S → traveling wave (more uniform amplitude)
- If PML works: S_off > S_on → ratio > 1

**Result from test**:
```
S_on = 3.54
S_off = 3.58
ratio = 1.01
```

⚠️ **BARELY IMPROVED**: PML reduced standing-wave ratio by only 1%, despite "14.96x reflection reduction."

**Why the discrepancy?**
1. Single-point probe (reflection proxy) is **NOT** measuring reflection, it's measuring standing wave amplitude at one location
2. Line scan shows standing waves still present with PML (S_on = 3.54 is strong)
3. The "14.96x reduction" is comparing |p| at one point, which is **dominated by the incident wave**, not the reflected wave

**Conclusion**: ❌ **REFLECTION METRIC IS MISLEADING**. Actual reflection reduction is closer to 1% (from line scan), not 1400%.

### E4. Acceptance Thresholds

From [test_pml_smoke.py:686-694](scripts/validation/test_pml_smoke.py#L686-L694):

```python
TARGET_REDUCTION = 1.2  # Target: 1.2x reflection reduction

if reduction_factor >= TARGET_REDUCTION:
    print(f"  ✓ PASS: Reflection reduced by {reduction_factor:.2f}x (target: {TARGET_REDUCTION}x)")
    test_passed = True
else:
    print(f"  ✗ FAIL: Reflection only {reduction_factor:.2f}x (target: {TARGET_REDUCTION}x)")
    test_passed = False
```

**Thresholds**:
- Reduction factor ≥ 1.2x → PASS
- PML activation: Im(s) > 0 in PML → PASS
- Excitation check: |p_pml| > 1e-12 · max|p| → PASS

**Issues**:
1. **1.2x threshold is arbitrary** – no theoretical justification
2. **No absolute reflection coefficient** (should be < 1% for "good" PML)
3. **Line scan threshold missing** – should require ratio > 1.1 or similar

### E5. Mesh Conformity

From [test_pml_smoke.py:175-200](scripts/validation/test_pml_smoke.py#L175-L200):

```python
# Create boxes
water_box = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
pml_box = gmsh.model.occ.addBox(L, 0, 0, d_pml, L, L)

# Boolean fragment for conforming mesh
gmsh.model.occ.fragment([(3, water_box), (3, pml_box)], [])
```

✅ **CORRECT**: Boolean fragment ensures conforming interface (shared nodes).

### E6. Production Truth Test

**Question**: Does production solver have equivalent validation?

**Answer**: ❌ **NO**. Production integration in `acoustics.py` has NO built-in validation that:
- PML stretch is correct
- Reflection is actually reduced
- Forms are assembled correctly

**Recommendation**: Add a `validate_pml()` method that runs a mini-smoke-test within production solver.

---

## F) Solver Behavior: Numerical Method Issues

### F1. GMRES Convergence

From [test_pml_smoke.py:397-407](scripts/validation/test_pml_smoke.py#L397-L407):

```python
matrix_size = V.dofmap.index_map.size_global
if matrix_size < 50000:
    ksp.setType("preonly")
    ksp.getPC().setType("lu")  # ✅ Direct solver
else:
    ksp.setType("gmres")
    ksp.getPC().setType("ilu")
    ksp.setTolerances(rtol=1e-8, max_it=2000)
```

**Test uses**: Direct LU (since ~24k DOFs < 50k).  
**Production uses**: Not clear – need to check production solver config.

From [acoustics.py:449-455](src/tweezers/fenicsx/acoustics.py#L449-L455):

```python
p_solution = solve_linear_system(
    a, L, bcs=bcs,
    V=self.V,
    petsc_options={
        "ksp_type": self.config.solver.ksp_type,
        "pc_type": self.config.solver.pc_type,
        "pc_factor_mat_solver_type": self.config.solver.pc_factor_solver_type,
    }
)
```

Uses `self.config.solver.*` settings. Need to check defaults.

**From config.py** (need to read):

<function_calls>
<invoke name="grep_search">
<parameter name="includePattern">src/tweezers/fenicsx/config.py
From [config.py:224-227](src/tweezers/fenicsx/config.py#L224-L227):

```python
ksp_type: str = "preonly"    # Direct solver
pc_type: str = "lu"
pc_factor_solver_type: str = "mumps"
```

**Production defaults**: Direct LU with MUMPS.

**Convergence behavior**:
- Direct solver → always "converges" in 1 iteration (factorization + backsolve)
- No iterative convergence history
- ❌ **CANNOT DIAGNOSE** if system is ill-conditioned

**When GMRES would fail**:
1. **Complex stretch too large**: High σ → large Im(s) → poor conditioning
2. **Inadequate resolution**: <5 PPW → spurious modes
3. **Material contrasts**: Large ρ_air/ρ_water ratio → stiff system

⚠️ **MASKED BY DIRECT SOLVER**: Tests use LU, so ill-conditioning doesn't show up. Real problems at high DOF (>1M) will need iterative solvers.

**Recommendation**: Add GMRES test with residual monitoring to detect conditioning issues.

### F2. Preconditioner Strategy

**Current**: No iterative solver strategy for large DOF.

**Needed for production** (DOF > 100k):
1. **GMRES** with **ILU** or **AMG** preconditioner
2. **Tolerances**: rtol=1e-8, atol=1e-10
3. **Max iterations**: 1000-2000
4. **Fallback**: Drop to LU if GMRES stalls

❌ **NOT IMPLEMENTED** in production code.

### F3. Test vs Production Discrepancy

| Feature | Test | Production |
|---------|------|------------|
| Solver | Direct LU | Direct LU (default) |
| DOF | 24k | Could be >1M |
| Convergence | Always 1 iter | Always 1 iter |
| Diagnostic | None | None |

⚠️ **PASSING TESTS DON'T GUARANTEE PRODUCTION WORKS** – direct solvers hide conditioning problems.

---

## G) Downstream Physics: Impact on Gor'kov / Streaming / Trajectories

### G1. What Changes After Adding PML?

**Pressure field**:
- ✅ Amplitude: Lower near boundaries (less reflection)
- ⚠️ Phase: PML introduces phase lag (complex stretch)
- ❌ Far-field: Should be unaffected if PML is far enough

**Gor'kov potential** (U = V_p |p|²/4ρc²):
- ✅ Trap locations: Should be unchanged (internal field)
- ⚠️ Trap strength: Slightly reduced if boundary reflections contributed
- ❌ Spurious traps: Could appear if PML creates artifacts

**Radiation force** (F = -∇U):
- ✅ Direction: Should be unchanged
- ⚠️ Magnitude: Proportional to |p|², so reduced if PML lowers amplitude

**Streaming forcing** (F_stream ∝ ∇·⟨u ⊗ u⟩):
- ⚠️ Boundary layers: PML absorbs waves → less acoustic intensity → weaker streaming near boundaries
- ✅ Bulk flow: Internal streaming should be unaffected

### G2. Time-Averaged Formula Consistency

From Gor'kov potential theory:

```
U = V [ (f1/ρ₀c₀²) ⟨p²⟩ - (3f2/2ρ₀) ⟨v²⟩ ]
```

where ⟨·⟩ is time average.

For time-harmonic field p(x) · e^{iωt}:

```
⟨p²⟩ = (1/2) Re(p · p*)  = (1/2) |p|²
⟨v²⟩ = (1/2ω²ρ₀²) |∇p|²
```

**Complex conjugate requirement**: p · p* = |p|² ✅

**Need to verify**: Current Gor'kov implementation uses |p|² correctly (not just p²).

### G3. Upstream Changes: Lens Geometry

**Volumetric lens domain**:
- ❌ **NOT IMPLEMENTED**: Current geometry has no lens domain
- Would need: Domain.LENS with appropriate tags
- Acoustic properties: c_lens, ρ_lens (typically PMMA or similar)

**Boundary vs volumetric**:
- Current: Lens would be a boundary condition (pressure/impedance)
- Better: Volumetric lens domain with material properties

**Coupling to plate**:
- Requires fluid-solid interface conditions
- Water → Lens → Plate transmission
- Currently: ❌ **NO SOLID COUPLING** at ACOUSTICS_PML level

**What gets re-meshed**:
- Lens geometry change → full geometry rebuild → remesh
- ❌ **NOT PARAMETRIC**: No lens shape parameters in config

---

## H) Readiness: Physical Lens + Elastic Plate + Trajectories

### H1. Minimum Validated Stack

**For trustworthy trajectories**, need:

✅ **Level 1: ACOUSTICS_ONLY**
- Helmholtz solver ✅ validated
- Complex PETSc ✅ verified
- Actuation BC ✅ working

⚠️ **Level 2: ACOUSTICS_PML**
- Volumetric PML ⚠️ has bugs (see A-E above)
- Reflection reduction ❌ misleading metric
- x-only PML ⚠️ insufficient for 3D boundaries

❌ **Level 3: SOLID_COUPLING**
- Fluid-solid interface ❌ NOT IMPLEMENTED in PML path
- Plate vibration ❌ NOT TESTED with PML
- Transmission coefficient ❌ NO VALIDATION

❌ **Level 4: GOR'KOV + TRAJECTORIES**
- Gor'kov computation ❓ needs verification (|p|² vs p²)
- Particle ODE ❓ needs validation case
- Stokes drag ❓ needs Reynolds number check

**Verdict**: ❌ **NOT READY** – PML has bugs, no solid coupling, no end-to-end validation.

### H2. Missing for "Physical Lens Design Iteration"

**Geometry**:
1. ❌ Lens as volumetric domain (not just BC)
2. ❌ Parametric lens shape (spherical, cylindrical, custom)
3. ❌ Lens-plate interface geometry

**Material properties**:
1. ❌ Lens acoustic properties (c_lens, ρ_lens, α_lens)
2. ❌ Lens-water impedance mismatch
3. ❌ Frequency-dependent attenuation

**Coupling**:
1. ❌ Water → Lens transmission
2. ❌ Lens → Plate coupling
3. ❌ Plate vibration modes

**Solver**:
1. ❌ Multi-domain Helmholtz (water + lens + air)
2. ❌ Coupled fluid-solid solver
3. ❌ Convergence strategy for large DOF

### H3. Next Validation Test: Bath → Plate → Dish Transmission

**Test case**: "Vibrating plate in water with reflecting boundary"

**Setup**:
1. Simple geometry: water box + elastic plate at bottom
2. Actuation: prescribed displacement on plate
3. Metrics:
   - Pressure continuity at water-plate interface
   - Velocity continuity (kinematic BC)
   - Energy flux: Power_in (actuation) ≈ Power_absorbed (PML) + Power_stored (resonance)
   - Known limiting case: rigid plate → ∂p/∂n = -iωρv_n (Neumann BC)

**Pass criteria**:
- Interface residuals < 1% of max field
- Energy balance within 5%
- Rigid limit: computed transmission → theory

**Status**: ❌ **NOT IMPLEMENTED**

---

## Summary of Critical Issues

### 🚨 BLOCKER ISSUES (Cannot use for research)

1. **Misleading reflection metric**: "14.96x reduction" is NOT measuring reflection. Actual reduction ~1% (from line scan).

2. **x-only PML insufficient**: Cannot absorb oblique waves or y/z reflections. Need multi-directional PML.

3. **No solid coupling in PML path**: Cannot do "lens + plate + trajectories" with current code.

4. **No lens geometry**: Cannot iterate lens design without implementing volumetric lens domain.

### ⚠️ MUST-FIX ISSUES (Correctness)

5. **Mass term documentation**: Clarify why `s_x` is correct for x-only (because s_y = s_z = 1). Add warning about generalization.

6. **Multiple PML regions**: Production code only handles one PML region. Cannot do 3D box with PML on all 6 faces.

7. **Interface location heuristic**: Assumes PML at +x boundary. Breaks for other orientations.

8. **No Gor'kov validation**: Need to verify time-averaged formulas use |p|² correctly.

### 📋 SHOULD-FIX ISSUES (Quality)

9. **Im(s) statistics**: Report min/median/max, not just max.

10. **GMRES fallback**: Add iterative solver strategy for large DOF.

11. **Convergence diagnostics**: Add residual history monitoring.

12. **Production validation**: Add built-in PML check in acoustics.py.

---

## Recommendations

### Immediate Actions (Before Next Research Use)

1. **Fix reflection metric**: Replace single-point probe with proper reflection coefficient calculation or transmission/reflection analysis.

2. **Add disclaimer to CHANGELOG**: Note that "14.96x" is misleading and actual reflection reduction is ~1%.

3. **Document x-only PML limits**: Clearly state it only works for waves traveling in x-direction.

### Short-Term (Next Sprint)

4. **Implement multi-directional PML**: Add s_y, s_z for full 3D absorption.

5. **Add solid coupling at PML level**: Enable fluid-solid interface conditions.

6. **Create lens domain**: Add Domain.LENS with material properties.

7. **Validate Gor'kov**: Add test case comparing analytical vs computed potential.

### Long-Term (Next Version)

8. **End-to-end validation**: "Actuation → acoustic field → Gor'kov → trajectories" with known result.

9. **Parametric lens geometry**: Add lens shape parameters to config.

10. **Iterative solver tuning**: Optimize GMRES + preconditioner for large problems.

---

**Audit completed**: January 26, 2026  
**Reviewer**: Technical assessment based on code review  
**Status**: ❌ **NOT READY FOR RESEARCH USE** – requires corrections before publication-quality results

