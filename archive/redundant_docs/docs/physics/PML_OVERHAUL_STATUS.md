# PML System Overhaul - Status Report

**Date**: January 2026  
**Issue**: Audit revealed "14.96x reflection reduction" was misleading (actual ~1%)  
**Goal**: Deliver trustworthy 3D tensor PML with proper validation

---

## Executive Summary

✅ **COMPLETED**:
- Task A: New validation with proper 2-wave reflection coefficient
- Task B: Full 6-face tensor PML implementation with corner handling

🔄 **IN PROGRESS**:
- Task C: Production integration (needs environment fix)
- Task D: 6-face validation test

⚠️ **BLOCKER**: Current environment has PETSc built for real scalars, not complex.  
Must switch to complex-enabled PETSc environment before running validation.

---

## Task A: Proper Reflection Metric ✅

**File**: `scripts/validation/test_pml_reflection_fit.py` (540 lines)

### Implementation

Replaced single-point amplitude ratio with proper 2-wave fit:

```python
# Design matrix for least-squares fit
# p(x) = A*exp(-ikx) + B*exp(+ikx)
X = np.column_stack([np.exp(-1j * k * x_probe), 
                      np.exp(1j * k * x_probe)])

# Solve for amplitudes A, B
XH_X = X.conj().T @ X
XH_p = X.conj().T @ p_probe
AB = np.linalg.solve(XH_X, XH_p)

# Reflection coefficient
R = |B| / |A|
```

### Validation Criteria

- **PML active**: `Im(s_x) > 1e-6`
- **Reflection with PML**: `R_on < 0.10` (< 10% reflection)
- **Reflection without PML**: `R_off > 0.20` (baseline exists)
- **Reduction factor**: `R_off / R_on > 2.0` (PML must reduce by 2x minimum)

### Features

- 100-point probe line in water near PML interface
- Proper least-squares fitting (not single-point heuristic)
- MPI-safe field evaluation
- JSON diagnostics output with all parameters
- Production PML code (single source of truth)

### Status

**Complete** but untested due to environment issue (PETSc not complex).

---

## Task B: 6-Face Tensor PML ✅

**File**: `src/tweezers/fenicsx/pml.py` (lines 60-327)

### New Functions

#### 1. `build_pml_stretch_tensor_dg0(...)`

Builds full 3D tensor PML for rectangular domain:

```python
# Per-axis distances (6 faces)
d_x = max(0, x-(x_max-t)) + max(0, (x_min+t)-x)  # Left + Right
d_y = max(0, y-(y_max-t)) + max(0, (y_min+t)-y)  # Front + Back  
d_z = max(0, z-(z_max-t)) + max(0, (z_min+t)-z)  # Bottom + Top

# Per-axis complex stretches (additive in corners)
s_x = 1 + 1j*σ(d_x)/ω
s_y = 1 + 1j*σ(d_y)/ω
s_z = 1 + 1j*σ(d_z)/ω
```

**Returns**: `(s_x, s_y, s_z, s_x_inv, s_y_inv, s_z_inv, diagnostics)`

**Key Features**:
- Handles all 6 faces of a box domain
- Corner cells get additive sigma (multiple stretches active)
- Returns inverses for gradient term efficiency
- PML diagnostics (Im(s) ranges, cell counts)
- Supports multiple PML tags (water, air, etc.)

#### 2. `helmholtz_tensor_pml_forms(...)`

Weak form with full tensor PML:

```python
# Anisotropic gradient term
a = (1/ρ) * [(1/s_x)*p_x*v̄_x + (1/s_y)*p_y*v̄_y + (1/s_z)*p_z*v̄_z]

# Mass term with FULL Jacobian
a -= (k²/ρ) * (s_x * s_y * s_z) * p * v̄
```

**Key Details**:
- All 3 gradient components have independent stretches
- Mass term uses full Jacobian `(s_x * s_y * s_z)`, not just `s_x`
- Reduces to standard Helmholtz when `s = 1`
- Works for complex wavenumber (time convention: e^{+iωt})

### Legacy Functions Marked

- `build_pml_stretch_dg0()` → "LEGACY - FOR DIRECTIONAL TESTING ONLY"
- `helmholtz_anisotropic_pml_forms()` → "LEGACY - FOR DIRECTIONAL TESTING ONLY"

Docstrings updated to recommend tensor versions for production.

### Status

**Complete**. Ready for integration testing (blocked by environment).

---

## Task C: Production Integration ⏳

**File**: `src/tweezers/fenicsx/acoustics.py` (lines 352-441 need update)

### Required Changes

1. **Switch to tensor PML builder**:
   ```python
   # OLD (x-only)
   s_x, s_x_inv, im_s_water, im_s_pml = build_pml_stretch_dg0(...)
   
   # NEW (tensor)
   bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
   s_x, s_y, s_z, s_x_inv, s_y_inv, s_z_inv, diag = build_pml_stretch_tensor_dg0(
       mesh, cell_tags, pml_tags, bbox, pml_thickness, omega, sigma_max
   )
   ```

2. **Detect all PML tags**:
   ```python
   # OLD: Only PML_WATER
   pml_tags = [Domain.PML_WATER.value]
   
   # NEW: All PML regions
   pml_tags = [tag for tag in Domain if "PML" in tag.name]
   ```

3. **Use tensor weak form**:
   ```python
   # OLD
   a_form, _ = helmholtz_anisotropic_pml_forms(...)
   
   # NEW
   a_form, _ = helmholtz_tensor_pml_forms(...)
   ```

4. **Add diagnostics method**:
   ```python
   def _pml_diagnostics(self, s_x, s_y, s_z, diagnostics):
       """Log PML statistics for verification."""
       for axis in ['x', 'y', 'z']:
           print(f"  Im(s_{axis}) water: {diag[f'im_s_{axis}_water']}")
           print(f"  Im(s_{axis}) PML:   {diag[f'im_s_{axis}_pml']}")
   ```

### Status

**Not started** (blocked by environment - need complex PETSc to test).

---

## Task D: 6-Face Validation ⏳

**File**: `scripts/validation/test_pml_6face_box.py` (to be created)

### Design

- 3D box with PML on all 6 sides
- Excite from one face (e.g., x=0)
- Use 2-wave fit along x-axis for R_on
- Optional: line scans in y/z directions

### Status

**Not started** (waiting for Task C completion).

---

## Critical Findings from Audit

### 1. Misleading Reflection Metric (BLOCKER)

**Problem**: v2.3.0 claimed "14.96x reflection reduction" from single-point amplitude.

**Reality**: Standing-wave line scan revealed only 1.01x (1% actual improvement).

**Root Cause**: Single point measures standing wave amplitude, not reflection coefficient.

**Fix**: Task A implements proper 2-wave fitting → R = |B|/|A|.

### 2. x-only PML Insufficient (BLOCKER)

**Problem**: Only absorbs waves traveling in ±x direction.

**Impact**: Cannot handle oblique waves or y/z propagation.

**Fix**: Task B implements full tensor (s_x, s_y, s_z) for 3D box.

### 3. Mass Term Documentation (MUST-FIX)

**Problem**: Current docs say "s_x factor" without clarifying it's the Jacobian.

**Clarification**:
- x-only PML: Jacobian = s_x × 1 × 1 = s_x ✓
- Tensor PML: Jacobian = s_x × s_y × s_z ✓

**Fix**: Task B docstrings clarify this.

### 4. Multiple PML Regions (MUST-FIX)

**Problem**: Production code only handles PML_WATER at +x boundary.

**Impact**: Cannot support 6-face PML.

**Fix**: Task C will detect all PML tags and use tensor builder.

---

## Environment Requirements

### PETSc Complex Scalars (REQUIRED)

All validation and production code requires PETSc built with complex scalar support:

```bash
# Check current environment
python -c "from petsc4py import PETSc; import numpy as np; \
    print(f'PETSc.ScalarType: {PETSc.ScalarType}'); \
    print(f'Is complex: {np.issubdtype(PETSc.ScalarType, np.complexfloating)}')"
```

**Expected Output**:
```
PETSc.ScalarType: <class 'numpy.complex128'>
Is complex: True
```

**Current Status**: Environment has `numpy.float64` (real scalars).

### How to Fix

Need to either:
1. Switch to existing complex-enabled environment (e.g., `acousto-complex`)
2. Rebuild PETSc with `--with-scalar-type=complex`
3. Install petsc4py complex variant: `pip install petsc petsc4py --complex`

### Validation Scripts Check

All validation scripts now include guard:

```python
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("ERROR: PETSc must be built with complex scalar support!")
    sys.exit(1)
```

---

## Testing Plan

### Phase 1: Environment Setup

- [ ] Confirm complex PETSc environment exists
- [ ] Activate complex environment
- [ ] Verify: `python -c "from petsc4py import PETSc; assert 'complex' in str(PETSc.ScalarType).lower()"`

### Phase 2: Validation (Task A)

- [ ] Run `test_pml_reflection_fit.py` with PML on
- [ ] Verify: R_on < 0.10 (good absorption)
- [ ] Run with PML off (baseline)
- [ ] Verify: R_off > 0.20 (reflection exists)
- [ ] Verify: R_off/R_on > 2.0 (PML improves by 2x)
- [ ] Check diagnostics.json for full record

### Phase 3: Integration (Task C)

- [ ] Update `acoustics.py` to use tensor PML
- [ ] Test with existing demos (`run_fem_demo.py`)
- [ ] Verify no regression in particle trap
- [ ] Check PML diagnostics in solver logs

### Phase 4: 6-Face Validation (Task D)

- [ ] Create `test_pml_6face_box.py`
- [ ] Run with all 6 sides PML
- [ ] Verify R_on < 0.10 in all directions
- [ ] Compare vs x-only baseline

---

## Documentation Updates (Pending)

### CHANGELOG.md

```markdown
## [v2.4.0] - 2026-01-XX

### Fixed
- **CRITICAL**: Deprecated misleading "14.96x" PML reflection metric (actual ~1%)
- Replaced single-point amplitude ratio with proper 2-wave reflection coefficient

### Added
- Full 6-face tensor PML with (s_x, s_y, s_z) coordinate stretches
- Corner handling with additive sigma profiles
- Proper reflection validation via least-squares fitting: R = |B|/|A|
- PML diagnostics (Im(s) ranges, cell counts)

### Changed
- Marked x-only PML functions as "LEGACY - FOR DIRECTIONAL TESTING ONLY"
- Production code now uses `build_pml_stretch_tensor_dg0` and `helmholtz_tensor_pml_forms`
- Mass term clarified: Jacobian = s_x * s_y * s_z
```

### README.md

```markdown
## PML System

**v2.4.0+**: Uses full 3D tensor PML with proper validation.

**IMPORTANT**: v2.3.0 claimed "14.96x reflection reduction" but actual measurement was only 1.01x (1% improvement). v2.4.0 implements proper 2-wave fitting to measure true reflection coefficient.

### Validation
- Reflection coefficient: R = |B|/|A| from 2-wave fit
- Acceptance: R < 0.10 (< 10% reflection)
- See: `scripts/validation/test_pml_reflection_fit.py`
```

---

## Implementation Summary

| Task | Status | File | Lines | Notes |
|------|--------|------|-------|-------|
| A: Reflection Metric | ✅ Complete | `test_pml_reflection_fit.py` | 540 | Blocked by env |
| B: Tensor PML | ✅ Complete | `pml.py` (lines 60-327) | ~270 | Ready for integration |
| C: Integration | ⏳ Pending | `acoustics.py` (lines 352-441) | ~90 | Needs complex PETSc |
| D: 6-Face Validation | ⏳ Pending | `test_pml_6face_box.py` | ~500 | After Task C |
| Docs: CHANGELOG | ⏳ Pending | `CHANGELOG.md` | ~15 | After testing |
| Docs: README | ⏳ Pending | `README.md` | ~10 | After testing |

**Total new code**: ~1400 lines  
**Time to completion**: ~2-4 hours (after environment fix)

---

## Next Immediate Steps

1. **Fix environment** (BLOCKER):
   ```bash
   # Check for complex-enabled environment
   mamba env list | grep complex
   
   # OR create one
   mamba create -n fenicsx-complex python=3.11
   mamba activate fenicsx-complex
   mamba install fenics-dolfinx petsc=*=*complex*
   ```

2. **Run Task A validation**:
   ```bash
   python scripts/validation/test_pml_reflection_fit.py
   ```

3. **Verify proper reflection**:
   - Check `results/pml_reflection_validation/diagnostics.json`
   - R_on should be < 0.10
   - R_off should be > 0.20

4. **Implement Task C** (production integration)

5. **Create Task D** (6-face validation)

6. **Update docs** (CHANGELOG, README)

---

## Key Insights

### Why Single-Point Was Wrong

```
Standing wave: p(x) = A*exp(-ikx) + B*exp(+ikx)
Amplitude: |p(x)| = |A + B*exp(2ikx)|

At certain x: |p| can be small even when |B|/|A| is large!
→ Single point measures standing-wave NODE, not reflection.
```

### Why Tensor is Required

```
For oblique wave: k_vec = (k_x, k_y, k_z)
- x-only PML: only attenuates k_x component
- Tensor PML: attenuates all components via (s_x, s_y, s_z)

Example: 45° wave has k_x = k_y = k/√2
→ x-only PML: insufficient attenuation
→ Tensor PML: both s_x and s_y active
```

### Corner Handling

```
Corner cell: distance from two boundaries
→ d_x > 0 AND d_y > 0

WRONG: σ_total = max(σ_x, σ_y)  [not additive]
RIGHT: s_x = 1 + 1j*σ_x/ω, s_y = 1 + 1j*σ_y/ω  [independent]

Jacobian in corner: s_x * s_y > max(s_x, s_y)  ✓
```

---

**Prepared by**: Acousto-Tweezers Development Team  
**Review Status**: Ready for environment setup and testing  
**Confidence**: High (theory verified, implementation complete, waiting on execution)
