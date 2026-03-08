# Standing-Wave FEM Data Pipeline Audit

**Date:** 2026-03-08  
**Purpose:** Full investigation of what data is available from the standing-wave FEM solution, whether proper FEM interpolation is possible, and what is causing gradient artefacts in the trap pipeline.

---

## 1. Standing-Wave Cache Contents

**File:** `results/fem_standing_wave_cache/standing_wave_epl6.npz` (15.4 MB)  
**Generated:** 2026-03-04T10:43:36 (solve time: 151.5 s)

### Arrays stored

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `coords` | (762129, 3) | float64 | DOF coordinates (x, y, z) in metres |
| `p_real` | (762129,) | float64 | Re(p) at each DOF |
| `p_imag` | (762129,) | float64 | Im(p) at each DOF |
| `dofs` | scalar | int64 | 762129 |
| `frequency_hz` | scalar | float64 | 2 000 000 |
| `wavelength` | scalar | float64 | 0.000742 m |
| `c_water` | scalar | float64 | 1484 m/s |
| `rho_water` | scalar | float64 | 997 kg/m³ |
| `Lx` | scalar | float64 | 0.006 m |
| `Ly` | scalar | float64 | 0.006 m |
| `H_total` | scalar | float64 | 0.0050085 m |
| `H_top` | scalar | float64 | 0.0020085 m |
| `H_under` | scalar | float64 | 0.003 m |
| `mesh_nx` | scalar | int64 | 48 |
| `mesh_ny` | scalar | int64 | 48 |
| `mesh_nz` | scalar | int64 | 40 |
| `elements_per_wavelength` | scalar | int64 | 6 |
| `standing_velocity_amplitude` | scalar | float64 | 1e-5 m/s |
| `standing_axis` | scalar | str | "both" |
| `standing_phase_pattern` | scalar | str | "antiphase" |
| `solve_time_s` | scalar | float64 | 151.479 s |

### What is NOT in the cache

- **No cell connectivity** (cell-to-vertex mapping)
- **No element topology** (tetrahedron indices)
- **No DOF-to-cell mapping** (dofmap)
- **No element order metadata** (only inferrable from DOF count + mesh dims)
- **No function space definition** (element type, polynomial degree)
- **No quadrature data**
- **No facet/boundary tags**
- **No PML sigma fields**

The cache is a **scatter of (coordinate, value) pairs** with physical metadata.
It does not encode the FEM mesh structure.

---

## 2. Can We Perform True FEM Interpolation From the Cache?

**No. The cache is insufficient for true FEM interpolation.**

### Why FEM interpolation requires the mesh

To evaluate a P2 FEM field at an arbitrary point **x**, you must:

1. **Locate the containing element** — find which tetrahedron contains **x** (point-location / bounding-box tree search).
2. **Map to reference coordinates** — compute the barycentric/reference coordinates (ξ, η, ζ) within that element.
3. **Evaluate basis functions** — compute the 10 P2 shape functions φ₁…φ₁₀ at those reference coordinates.
4. **Combine DOF values** — p(**x**) = Σᵢ pᵢ φᵢ(ξ, η, ζ), where pᵢ are the DOF coefficients for that element.

Steps 1–3 all require the **mesh connectivity** (which vertices form each tetrahedron), the **DOF mapping** (which DOFs belong to each element), and the **element type** (P2 Lagrange on tetrahedra). None of this is stored in the `.npz` cache.

### What we actually have

The cache stores the output of `dolfinx FunctionSpace.tabulate_dof_coordinates()` (the geometric location of each DOF) paired with the DOF values from `Function.x.array`. These are the **coefficients** of the P2 expansion, not point samples. Between DOF locations, the true field is a **quadratic polynomial** determined by the element shape functions — the DOF values alone, without knowing which DOFs share an element, cannot reconstruct that polynomial.

### Structure of the DOF grid

The DOFs form a **structured 3D grid** with uniform spacing:
- **x-spacing:** 62.50 µm (= Lx / (2·nx) = 6.0 mm / 96)
- **y-spacing:** 62.50 µm
- **z-spacing:** 62.61 µm (= H_total / (2·nz) = 5.0085 mm / 80)

Grid dimensions: **97 × 97 × 81 = 762 129** DOFs — this matches exactly.

The DOFs arise from **P2 Lagrange elements on a structured tetrahedral mesh** (`mesh.create_box` with `CellType.tetrahedron`, nx=48, ny=48, nz=40, each hex split into 6 tets). P2 on this grid places DOFs at all vertices and edge midpoints, which on a structured grid coincide with a half-cell-spacing Cartesian grid.

**Complication:** `tabulate_dof_coordinates()` introduces floating-point jitter in z, producing **174 unique z-values** instead of 81. Pairs of z-values differ by ~1e-10 m ≈ 0.0001 µm. At some z-planes, DOFs are split across 2–3 numerically distinct z-values with different subsets of DOFs. This means a KDTree query at a single z-plane may pull DOFs from multiple numerically-separate z-levels, causing averaging artefacts.

---

## 3. Original FEM Solve Location

### Generator script

**`scripts/dev/fem_standing_plus_asm_vortex_local_3x3.py`** (line 380):

```python
sol_stand = solve_helmholtz(cfg_standing, verbose=True, petsc_options=PETSC_MUMPS)
coords_fem = sol_stand.coords.copy()      # tabulate_dof_coordinates()
pv_fem = sol_stand.p_values.copy()         # Function.x.array.copy()
del sol_stand    # <── mesh + function space DESTROYED here
gc.collect()
```

The `PressureSolution` object (which holds `p_function`, `domain`, `V`, `facet_tags`, `cell_tags`) is **explicitly deleted** after extracting only coords and DOF values. The mesh, function space, and solver context are not preserved.

### FEM solver

**`src/acoustweezers/experiments/farfield_petri_cuboid/solve_pressure.py`** → `solve_helmholtz()`:
- Creates mesh via `dolfinx.mesh.create_box()` (P2 Lagrange on tetrahedra)
- Solves PML-Helmholtz with MUMPS direct solver
- Returns `PressureSolution` which attaches: `domain`, `V`, `facet_tags`, `cell_tags`
- These are available for post-processing **if the solution is not deleted**

### XDMF export capability

**`src/acoustweezers/io/export_fields.py`** → `export_pressure_fields()`:
- Can export mesh + fields to XDMF+HDF5 (ParaView-compatible)
- Writes: `mesh.xdmf`/`.h5`, `p_real.xdmf`/`.h5`, `p_imag.xdmf`/`.h5`, `p_mag.xdmf`/`.h5`, `p_phase.xdmf`/`.h5`
- This code exists but **was not invoked** when generating the cache

### Other FEM output files

The `results/fem_standing_wave_cache/` directory contains **only** the `.npz` and `_INFO.txt` files. No XDMF, HDF5, VTU, or VTK files exist for the standing-wave solution. XDMF/VTU files exist elsewhere in `archive/results/` from earlier experiments (deposition, streaming) but **not for this standing-wave configuration**.

---

## 4. Current IDW Interpolation Implementation

### Code location

**`scripts/experiments/trap_localisation_debug_standing.py`**, lines 107–125 (identical code in `bridge_master_study.py:119` and `trap_localisation_validation_study.py:115`).

### `sample_idw()` function

```python
def sample_idw(tree, p, pts, k=16, power=2.0, eps=1e-12):
    if k == 1:
        d, i = tree.query(pts, k=1)
        return p[i]
    d, i = tree.query(pts, k=k)
    w = 1.0 / (d**power + eps)
    w /= w.sum(axis=1, keepdims=True)
    return (p[i] * w).sum(axis=1)
```

| Parameter | Value |
|-----------|-------|
| **Method** | Inverse Distance Weighting (Shepard's method) |
| **k (neighbours)** | 16 |
| **Power exponent** | 2.0 |
| **Regularisation** | ε = 1e-12 (prevents division by zero) |
| **Normalisation** | Yes — weights sum to 1 |
| **Source data** | DOF coordinates (762 129 points, full 3D domain) |
| **Neighbour search** | `scipy.spatial.cKDTree`, querying all DOFs (no z-slice) |

### Pipeline

1. Load cache → `coords` (N, 3), `p = p_real + 1j * p_imag` (N,)
2. Build `cKDTree(coords)` from all 762 129 DOFs
3. For each evaluation grid: create `np.linspace` over ROI, `np.meshgrid`, flatten to (M, 3)
4. Query k=16 nearest DOFs in 3D space
5. IDW-interpolate complex pressure
6. Compute |p|², ∇p via `np.gradient`, Gor'kov potential, then `-np.gradient(U)` for force

### Key problem: 3D neighbour search for 2D evaluation

The KDTree contains **all** DOFs across the full 3D domain (762 129 points). When evaluating on a 2D z-plane, the k=16 nearest neighbours come from the closest few z-levels (typically 1–2 planes above and below). For an evaluation point at z = Z_STAR:
- The nearest DOF z-plane is at z = 4.1946 mm (offset 4.9 µm from Z_STAR = 4.1898 mm)
- Each DOF z-plane contains ~9409 DOFs in the (x,y) plane
- With k=16, the neighbours span ~4×4 in (x,y) and potentially 2 z-levels
- The z-offset means the interpolation mixes data from two z-planes, adding an out-of-plane averaging artefact

---

## 5. Spatial Resolution Analysis

### FEM mesh scales

| Quantity | Value |
|----------|-------|
| Wavelength λ | 742 µm |
| Elements per wavelength | 6 |
| Tet element size | 125 µm |
| P2 DOF spacing | 62.5 µm |
| Mesh dimensions | 48 × 48 × 40 hex cells → 552 960 tetrahedra |
| Total DOFs | 762 129 |

### ROI grid scales

ROI half-width = 1.1λ = 816.2 µm → ROI width = 1632.4 µm

| ngrid | Grid spacing (µm) | Grid spacing / DOF spacing | Grid pts per element |
|-------|-------------------|---------------------------|---------------------|
| 100 | 16.49 | 0.264 | 7.6 |
| 200 | 8.20 | 0.131 | 15.2 |
| 400 | 4.09 | 0.065 | 30.6 |
| 800 | 2.04 | 0.033 | 61.2 |

### Key finding

**The ROI grid is 4–30× finer than the FEM DOF spacing.**

At ngrid=400 (the default), the ROI grid spacing is 4.1 µm vs the DOF spacing of 62.5 µm. This means:
- Each inter-DOF region is sampled at ~15 grid points
- The IDW interpolant must fill in structure between DOFs that it has no basis to reconstruct
- The true FEM field is a piecewise-quadratic polynomial between DOFs; IDW produces a smooth but **incorrect** interpolant that does not match the P2 polynomial

### DOFs in the ROI z-plane

At the target z-plane, approximately **27 × 27 = 729 DOFs** lie within the ROI in (x,y). These 729 scattered points are used to fill a 400 × 400 = 160 000 evaluation grid.

---

## 6. Is IDW the Source of Gradient Artefacts?

**Yes. IDW is almost certainly the primary source of the observed artefacts.**

### Why IDW fails for gradient computation

1. **Non-polynomial reconstruction.** The true FEM field between DOFs is a P2 (quadratic) polynomial determined by the 10 DOF coefficients of the containing tetrahedron. IDW with power=2 produces a rational interpolant (weighted sum of 1/d² terms) that does **not** reproduce the P2 polynomial. It is C⁰ at best, with kinks at DOF locations.

2. **Gradient amplification near DOFs.** The IDW weight function w = 1/(d² + ε) has gradient ∂w/∂x ∝ -2x/d⁴. Near a DOF (d → 0), the weight gradient diverges as 1/d³. When the evaluation grid passes near a DOF, the interpolated field develops a sharp peak or inflection, and `np.gradient` of this produces a spike.

3. **Resolution dependence.** At coarser grids (ngrid=100, dx=16.5 µm), grid points are far enough from DOFs that the IDW smoothing dominates. At finer grids (ngrid=800, dx=2.0 µm), grid points approach DOF locations and the 1/d³ gradient singularity becomes resolved. This explains why trap counts **change with resolution** (7→15→17 at 200→400→800): higher resolution resolves more IDW artefacts that create or destroy local minima in the Gor'kov potential.

4. **Z-gradient contamination.** The z-gradient stencil uses DZ_GRAD = λ/15 ≈ 49.5 µm, which is 0.79× the DOF z-spacing (62.6 µm). This means the central-difference z-gradient samples the IDW interpolant at z-offsets that don't align with any DOF z-plane. The IDW for z ± DZ_GRAD pulls in different subsets of z-plane DOFs, producing inconsistent gradients.

5. **Floating-point z-jitter.** The 174 unique z-values (vs 81 expected) from DOF coordinate jitter mean that a KDTree query at a single z mixes DOFs from numerically-separated z-planes. At material/PML boundaries, some z-planes have split DOF populations (e.g., 2305 + 7104 instead of 9409), which biases the IDW toward whichever sub-population is closer.

### Consistency with observed symptoms

- **Unstable trap count across resolution (7/15/17 at 200/400/800):** ✓ IDW artefacts scale with resolution
- **Bizarre z-plane sensitivity (0 traps at +0.05λ, 13–15 at adjacent offsets):** ✓ IDW reconstruction at z-offsets away from DOF planes is unreliable
- **Force spikes at specific locations:** ✓ IDW gradient singularities near DOFs

---

## 7. Proposed Solutions (Not Yet Implemented)

### Option A: Re-solve and use native FEM evaluation

**Approach:** Run the FEM solve, keep the `PressureSolution` (with mesh + function space) alive, and use DOLFINx's built-in point evaluation (`dolfinx.fem.Function.eval()`) which performs proper FEM interpolation (point location → reference coordinates → basis function evaluation).

**Feasibility:** Straightforward. The solver already works and takes ~2.5 minutes. The `PressureSolution` already has `.domain` and `.V` attached. `Function.eval()` accepts arrays of points.

**Effort:** Moderate. Requires restructuring the pipeline so the FEM solve (or at minimum the loaded `Function` on its mesh) is available when the trap study runs, rather than being deleted and cached as raw arrays.

**Pros:** Exact FEM interpolation, smooth gradients, resolution-independent.  
**Cons:** Requires either re-solving (2.5 min) or saving/reloading the full FEM Function (Option B).

### Option B: Save mesh + Function to XDMF/checkpoint for reload

**Approach:** Modify the cache-generation script to also export the mesh and FEM Function in a format that can be reloaded by DOLFINx. Options:
- **XDMF + HDF5** via `export_pressure_fields()` (which already exists but saves real-valued fields only). Would need to save complex Function or real/imag pair with mesh.
- **DOLFINx checkpoint** (`dolfinx.io.XDMFFile` write/read of `Function`).

After reloading, construct a `Function` on the same mesh and function space, populate its DOF vector, and use `Function.eval()` for interpolation.

**Feasibility:** Achievable. `export_pressure_fields()` already writes XDMF with mesh. The challenge is that DOLFINx XDMF read requires the mesh to be read first, then the function space recreated, then the DOF values loaded. The mesh geometry can be saved/loaded via XDMF; the function space is determined by (`"Lagrange"`, 2) which we know. P2 on the saved mesh will produce the same DOF layout.

**Effort:** Moderate. Need to:
1. Modify cache script to also call `export_pressure_fields()` (or a custom checkpoint function).
2. Write a loader that reads mesh from XDMF, creates P2 function space, loads DOF values, returns a live `Function`.
3. Use `Function.eval()` in the trap pipeline.

**Pros:** Avoids re-solving; exact FEM interpolation.  
**Cons:** More complex I/O; XDMF reading in DOLFINx can have quirks.

### Option C: Exploit the structured DOF grid for proper interpolation

**Approach:** Since the DOFs form a perfectly regular Cartesian grid (97 × 97 × 81, spacing 62.5 µm), we can:
1. **Reshape** the DOF values into a 3D array.
2. Use **trilinear or tricubic interpolation** (`scipy.interpolate.RegularGridInterpolator`) which respects the grid structure.
3. For higher accuracy, use **cubic spline interpolation** on the regular grid, which produces C² smooth fields with analytical gradients.

**Feasibility:** Very high. The DOFs are already on a regular grid — we just need to sort them into the right order and reshape. `RegularGridInterpolator` with `method='cubic'` gives smooth C¹ interpolation. For analytical gradients, we could use `scipy.interpolate.RectBivariateSpline` (2D slices) or compute spline derivatives.

**Caveats:**
- The DOF values are P2 Lagrange **coefficients**, not direct point samples. On a structured tet mesh, the P2 nodal basis is interpolatory at DOF locations, so DOF values ARE function values at those points. Cubic spline interpolation between them is not identical to P2 FEM interpolation, but it is **C² smooth** and reproduces quadratics exactly (which P2 FEM fields are, element-wise). This is vastly superior to IDW.
- Must handle the z-jitter: sort DOFs to (ix, iy, iz) indices, average or select unique representatives at each grid point.

**Effort:** Low-to-moderate. Mostly grid-sorting and replacing `sample_idw()` with `RegularGridInterpolator.__call__()`.

**Pros:** Fast; no FEM infrastructure needed; C² smooth; analytical gradients available; no re-solve.  
**Cons:** Not exactly P2 FEM interpolation (cubic spline ≠ P2 basis), but the error is negligible compared to IDW artefacts. Does not handle non-structured meshes if the solver changes.

---

## Summary

| Question | Answer |
|----------|--------|
| What does the cache contain? | DOF coordinates + DOF values + physical metadata. **No mesh, no connectivity, no function space.** |
| Can we do proper FEM interpolation from the cache? | **No.** Missing cell connectivity and DOF mapping prevent basis function evaluation. |
| Where is the original FEM solution? | **Deleted** after extracting coords + values. No XDMF/VTU/HDF5 checkpoint exists. |
| What does IDW do? | k=16 nearest-neighbour inverse-distance weighting (power=2) on raw 3D DOF scatter. |
| Is IDW the source of artefacts? | **Yes.** Non-polynomial reconstruction + 1/d³ gradient singularities near DOFs + z-plane misalignment. |
| ROI grid vs DOF resolution? | ROI grid is **4–30× finer** than DOF spacing. IDW must fabricate sub-DOF detail it cannot know. |

### Recommended path forward

**Option C (structured-grid interpolation) is the pragmatic first step.** It requires no FEM infrastructure, no re-solve, and replaces IDW with a mathematically well-founded C² interpolant that produces smooth, resolution-independent gradients. The regular DOF grid makes this straightforward.

**Option B (XDMF checkpoint)** should be implemented as a longer-term improvement for correctness, but is not blocking.

**Option A (re-solve)** is the gold standard but adds 2.5 min to every pipeline run and requires FEM dependencies to be available.
