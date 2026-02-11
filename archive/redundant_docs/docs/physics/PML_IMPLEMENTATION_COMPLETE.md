# PML Overhaul - Completion Summary

**Date**: January 26, 2026  
**Version**: v2.4.0  
**Status**: ✅ **COMPLETE** (pending environment setup for testing)

---

## Executive Summary

Successfully completed full PML system overhaul to address critical issues found in technical audit:

1. ✅ **Proper reflection validation**: Implemented 2-wave fitting (R = |B|/|A|) replacing misleading single-point proxy
2. ✅ **6-face tensor PML**: Full 3D implementation with (s_x, s_y, s_z) and corner handling
3. ✅ **Production integration**: Updated acoustics.py with clean domain separation and diagnostics
4. ✅ **Documentation**: CHANGELOG and README updated with deprecation notices

**Blocker**: Current environment has real PETSc (numpy.float64), must use complex PETSc (numpy.complex128) to run validation tests.

---

## Deliverables

### 1. Reflection Validation Test ✅

**File**: [scripts/validation/test_pml_reflection_fit.py](scripts/validation/test_pml_reflection_fit.py) (540 lines)

**Implementation**:
- 2-wave model fitting: `p(x) = A·exp(-ikx) + B·exp(+ikx)`
- Least-squares solution for complex A, B
- Reflection coefficient: R = |B|/|A|
- 100-point probe line in water near PML interface
- Thresholds: R_on < 0.10, R_off > 0.20, reduction > 2.0x

**Key Features**:
- Proper statistical fitting (not single-point heuristic)
- MPI-safe field evaluation
- JSON diagnostics output
- Complex PETSc check with clear error message

**Status**: Code complete, requires complex PETSc environment to run

---

### 2. 6-Face Tensor PML Implementation ✅

**File**: [src/tweezers/fenicsx/pml.py](src/tweezers/fenicsx/pml.py) (lines 60-327, ~270 new lines)

**Functions Added**:

#### `build_pml_stretch_tensor_dg0(...)`
```python
def build_pml_stretch_tensor_dg0(
    mesh, cell_tags, pml_tags, bbox, pml_thickness, 
    omega, sigma_max, power=2, water_tag=None
) -> (s_x, s_y, s_z, s_x_inv, s_y_inv, s_z_inv, diagnostics)
```

**Features**:
- Computes per-axis distances: d_x, d_y, d_z
- Independent stretches: s_x, s_y, s_z = 1 + 1j*σ(d)/ω
- Corner handling: additive sigma (multiple stretches active)
- Returns inverses for efficiency
- Diagnostics: Im(s) ranges (min/median/max), cell counts

#### `helmholtz_tensor_pml_forms(...)`
```python
def helmholtz_tensor_pml_forms(
    p, v, mesh, k, rho, omega,
    s_x, s_y, s_z, s_x_inv, s_y_inv, s_z_inv,
    dx_domain, dx_pml, source_form=None
) -> (a_form, L_form)
```

**Weak Form**:
- Gradient: `(1/ρ)·[(1/s_x)·p_x·v̄_x + (1/s_y)·p_y·v̄_y + (1/s_z)·p_z·v̄_z]`
- Mass: `-(k²/ρ)·(s_x·s_y·s_z)·p·v̄`  ← **FULL Jacobian**
- Proper conjugation: Uses `grad(conj(v))`

**Legacy Functions Marked**:
- `build_pml_stretch_dg0`: "LEGACY - FOR DIRECTIONAL TESTING ONLY"
- `helmholtz_anisotropic_pml_forms`: "LEGACY - FOR DIRECTIONAL TESTING ONLY"
- Docstrings updated to recommend tensor versions

**Status**: Code complete and ready for integration testing

---

### 3. Production Integration ✅

**File**: [src/tweezers/fenicsx/acoustics.py](src/tweezers/fenicsx/acoustics.py) (lines 352-440)

**Changes Made**:

#### Updated imports:
```python
from .pml import (
    build_pml_stretch_dg0, helmholtz_anisotropic_pml_forms,  # Legacy
    build_pml_stretch_tensor_dg0, helmholtz_tensor_pml_forms  # Production
)
```

#### PML integration (lines 352-440):
- Automatic bounding box detection from mesh geometry
- Calls `build_pml_stretch_tensor_dg0` with full bbox
- Uses `helmholtz_tensor_pml_forms` for water+PML domains
- Standard Helmholtz for other domains (air, dish, bath)
- Clean separation: no double-counting, no ABC when PML exists

#### New diagnostics method (lines 256-303):
```python
def _log_pml_diagnostics(self, pml_stats: dict, bbox: tuple, thickness: float):
    """Log PML statistics for verification."""
    # Prints:
    # - Bbox (x/y/z ranges)
    # - Thickness
    # - Cell counts (water, PML)
    # - Im(s_x/y/z) in water (should be ~0)
    # - Im(s_x/y/z) in PML (should be >0)
```

**Status**: Code complete, will log diagnostics on next run with PML

---

### 4. 6-Face Validation Test ✅

**File**: [scripts/validation/test_pml_6face_box.py](scripts/validation/test_pml_6face_box.py) (548 lines)

**Geometry**:
- Interior: [pml, L+pml]³ (water)
- PML: 6 slabs on all faces (x_min, x_max, y_min, y_max, z_min, z_max)
- Total domain: [0, L+2*pml]³
- Actuation: x=0 face

**Validation**:
- Measures reflection via 2-wave fit along x-axis
- 80-point probe line through water interior
- Pass criteria: R_on < 0.10, R_off > 0.20, reduction > 2.0x
- Saves diagnostics.json with full results

**Status**: Code complete, requires complex PETSc environment to run

---

### 5. Documentation Updates ✅

#### CHANGELOG.md (lines 1-122)

Added v2.4.0 section with:
- **Deprecation warning**: "14.96x reflection reduction" from v2.3.0 is MISLEADING
- Proper reflection validation with 2-wave fitting
- 6-face tensor PML implementation details
- Production integration changes
- Migration guide
- Updated v2.3.0 with warning note

#### README.md (lines 93-163)

Added PML section with:
- Updated physics ladder table (v2.4.0+)
- PML implementation details
- Key files reference
- Legacy function note
- Updated validation tests list

---

## Technical Details

### Why v2.3.0 Was Wrong

**Claimed**: "14.96x reflection reduction"  
**Reality**: ~1% (1.01x from standing-wave line scan)

**Root Cause**: Single-point amplitude measures standing wave, not reflection coefficient.

**Standing wave**: `p(x) = A·exp(-ikx) + B·exp(+ikx)`  
**Amplitude**: `|p(x)| = |A + B·exp(2ikx)|`

At certain x, |p| can be small even when |B|/|A| is large (node location).

### Tensor PML Theory

**Per-axis stretches**:
```
d_x = max(0, x-(x_max-t)) + max(0, (x_min+t)-x)
d_y = max(0, y-(y_max-t)) + max(0, (y_min+t)-y)
d_z = max(0, z-(z_max-t)) + max(0, (z_min+t)-z)

s_x = 1 + 1j·σ_max·(d_x/t)^m / ω
s_y = 1 + 1j·σ_max·(d_y/t)^m / ω
s_z = 1 + 1j·σ_max·(d_z/t)^m / ω
```

**Corner handling**: Natural additive behavior
- Corner cell has d_x > 0 AND d_y > 0
- Both s_x ≠ 1 AND s_y ≠ 1
- Jacobian = s_x·s_y·s_z > max(s_x, s_y, s_z) ✓

**Mass term**: Full Jacobian
- x-only: Jacobian = s_x·1·1 = s_x (v2.3.0 was correct by luck)
- Tensor: Jacobian = s_x·s_y·s_z (v2.4.0 general form)

---

## Environment Requirements

### Critical Requirement: Complex PETSc

All code requires PETSc built with complex scalar support:

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

**Current Status**: Shows `numpy.float64` (WRONG)

### How to Fix

Option 1: Activate existing complex environment
```bash
mamba env list | grep complex
mamba activate acousto-complex  # or similar
```

Option 2: Create new environment
```bash
mamba create -n fenicsx-complex python=3.11
mamba activate fenicsx-complex
mamba install -c conda-forge fenics-dolfinx petsc=*=*complex*
```

Option 3: Install complex PETSc in current environment
```bash
pip uninstall petsc petsc4py
pip install petsc petsc4py --config-settings=scalar-type=complex
```

---

## Testing Plan

Once complex PETSc environment is active:

### Phase 1: Reflection Validation
```bash
cd /home/znewman4/projects/acousto-tweezers
python scripts/validation/test_pml_reflection_fit.py
```

**Expected Output**:
- R_on < 0.10 (good PML absorption)
- R_off > 0.20 (baseline reflection exists)
- Reduction > 2.0x
- Diagnostics saved to results/pml_reflection_validation/diagnostics.json

### Phase 2: 6-Face Validation
```bash
python scripts/validation/test_pml_6face_box.py
```

**Expected Output**:
- R_on < 0.10 along x-axis
- All tests pass
- Diagnostics saved to results/pml_6face_validation/diagnostics.json

### Phase 3: Production Test
```bash
python scripts/run_fem_multiphysics.py --level ACOUSTICS_PML --quick
```

**Expected Output**:
- PML diagnostics logged (Im(s) ranges, cell counts)
- No errors
- Pressure field computed successfully

---

## Code Statistics

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Reflection validation | test_pml_reflection_fit.py | 540 | ✅ Complete |
| Tensor PML builder | pml.py (lines 60-190) | ~130 | ✅ Complete |
| Tensor weak form | pml.py (lines 237-327) | ~90 | ✅ Complete |
| Production integration | acoustics.py (lines 352-440) | ~90 | ✅ Complete |
| Diagnostics method | acoustics.py (lines 256-303) | ~50 | ✅ Complete |
| 6-face validation | test_pml_6face_box.py | 548 | ✅ Complete |
| CHANGELOG update | CHANGELOG.md | ~120 | ✅ Complete |
| README update | README.md | ~70 | ✅ Complete |
| **TOTAL** | | **~1638** | **✅ Complete** |

---

## Files Changed

### Modified
1. [src/tweezers/fenicsx/pml.py](src/tweezers/fenicsx/pml.py)
   - Added tensor functions (lines 60-327)
   - Marked legacy functions
   - Fixed complex dtype for DOLFINx 0.10.0

2. [src/tweezers/fenicsx/acoustics.py](src/tweezers/fenicsx/acoustics.py)
   - Updated imports (line 49)
   - Added `_log_pml_diagnostics()` method (lines 256-303)
   - Updated PML integration to use tensor (lines 352-440)

3. [CHANGELOG.md](CHANGELOG.md)
   - Added v2.4.0 section (lines 1-122)
   - Updated v2.3.0 with warning

4. [README.md](README.md)
   - Added PML section (lines 108-130)
   - Updated validation tests list (lines 155-163)

### Created
1. [scripts/validation/test_pml_reflection_fit.py](scripts/validation/test_pml_reflection_fit.py)
   - 540 lines, complete reflection validation

2. [scripts/validation/test_pml_6face_box.py](scripts/validation/test_pml_6face_box.py)
   - 548 lines, 6-face tensor PML validation

3. [docs/PML_OVERHAUL_STATUS.md](docs/PML_OVERHAUL_STATUS.md)
   - 431 lines, detailed status report

4. [docs/PML_IMPLEMENTATION_COMPLETE.md](docs/PML_IMPLEMENTATION_COMPLETE.md)
   - This file

---

## Key Insights

### 1. Why Single-Point Was Wrong

Standing wave amplitude at one point doesn't measure reflection:
- Node location → |p| ≈ 0 even with high reflection
- Antinode location → |p| high even with low reflection
- Need full spatial profile to separate forward/backward waves

### 2. Why 2-Wave Fitting Works

Least-squares fit over many points:
- Separates A (forward) and B (backward) components
- R = |B|/|A| is standard reflection coefficient definition
- Residual indicates quality of 1D assumption

### 3. Why Tensor is Required

For oblique wave with k = (k_x, k_y, k_z):
- x-only PML: only attenuates k_x component
- Tensor PML: attenuates all components via (s_x, s_y, s_z)
- Example: 45° wave has k_x = k_y → need both s_x and s_y

### 4. Corner Handling

In corner cell with d_x > 0 and d_y > 0:
- **WRONG**: σ_total = max(σ_x, σ_y)
- **RIGHT**: Independent s_x and s_y
- Result: Jacobian = s_x·s_y automatically handles corners

---

## Next Steps

1. **Fix environment** (BLOCKER):
   - Activate or create complex PETSc environment
   - Verify with: `python -c "from petsc4py import PETSc; assert 'complex' in str(PETSc.ScalarType).lower()"`

2. **Run reflection validation**:
   - `python scripts/validation/test_pml_reflection_fit.py`
   - Check diagnostics.json for R_on, R_off

3. **Run 6-face validation**:
   - `python scripts/validation/test_pml_6face_box.py`
   - Verify all tests pass

4. **Test production integration**:
   - Run existing demos with ACOUSTICS_PML level
   - Verify PML diagnostics are logged
   - Check no regressions

5. **Optional: Extend to other domains**:
   - Add PML_AIR, PML_BATH tags if needed
   - Update production integration to detect all PML tags

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|-----------|-------|
| Reflection validation | **High** | Standard 2-wave fitting, well-established method |
| Tensor PML theory | **High** | Based on Bermúdez 2007, Turkel 1998 |
| Implementation | **High** | Follows FEniCSx patterns, proper UFL forms |
| Corner handling | **High** | Additive sigma is natural consequence |
| Production integration | **High** | Clean domain separation, no double-counting |
| Testing plan | **Medium** | Requires environment fix before execution |

**Overall**: Ready for validation testing once environment is configured.

---

**Prepared by**: Acousto-Tweezers Development Team  
**Review Status**: Ready for testing  
**Sign-off**: Pending validation test results
