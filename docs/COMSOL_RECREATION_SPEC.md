# COMSOL Recreation Specification

## FEniCSx Shallow Square Dish Model — Complete Replication Checklist

**Version:** 1.0  
**Date:** 2026-02-12  
**Reference code:** `src/acoustweezers/experiments/shallow_square_dish/solve_pressure.py`  
**Reference config:** `src/acoustweezers/experiments/shallow_square_dish/config.py`

---

## 1. Geometry

### Domain

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Lateral length (x) | $L_x$ | 10 × 10⁻³ (Batch 1) or 50 × 10⁻³ (default) | m |
| Lateral length (y) | $L_y$ | $= L_x$ (square) | m |
| Depth | $H$ | 1 × 10⁻³ (Batch 1) or 5 × 10⁻³ (default) | m |

The domain is a rectangular box $[0, L_x] \times [0, L_y] \times [0, H]$.

**Coordinate system:** Cartesian, right-handed.
- $x$: first lateral direction, walls at $x = 0$ and $x = L_x$.
- $y$: second lateral direction, walls at $y = 0$ and $y = L_y$.
- $z$: vertical (depth), $z = 0$ is the bottom, $z = H$ is the top (free surface).

### Bottom Transducer Patch

A circular disc transducer is located on the bottom face ($z = 0$):

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Disc radius | $R_\mathrm{disc}$ | 3 × 10⁻³ (Batch 1) or 4 × 10⁻³ (default) | m |
| Centre x-coordinate | $c_x$ | $L_x / 2$ | m |
| Centre y-coordinate | $c_y$ | $L_y / 2$ | m |

The bottom face is partitioned into:
- **disc**: $\{(x, y, 0) : (x - c_x)^2 + (y - c_y)^2 \le R_\mathrm{disc}^2\}$
- **rigid floor**: the complement on $z = 0$.

### Boundary Naming Convention

| Name | Surface | Tag (FEniCSx) |
|------|---------|---------------|
| `bottom_disc` | $z = 0$, inside disc | 1 |
| `bottom_rigid` | $z = 0$, outside disc | 7 |
| `top` | $z = H$ | 2 |
| `x0` | $x = 0$ | 3 |
| `xL` | $x = L_x$ | 4 |
| `y0` | $y = 0$ | 5 |
| `yL` | $y = L_y$ | 6 |

In COMSOL: create a box geometry, add a Work Plane at $z = 0$, draw a circle of radius $R_\mathrm{disc}$ centred at $(c_x, c_y)$, and use **Partition Boundaries** to split the bottom face.

---

## 2. Material Properties

Single-domain, single-phase: water at 20 °C.

| Property | Symbol | Value | Unit |
|----------|--------|-------|------|
| Density | $\rho$ | 997.0 | kg/m³ |
| Speed of sound | $c$ | 1484.0 | m/s |
| Dynamic viscosity | $\mu$ | 1.002 × 10⁻³ | Pa·s |
| Bulk modulus | $K = \rho c^2$ | 2.195 × 10⁹ | Pa |
| Acoustic impedance | $Z_w = \rho c$ | 1.4795 × 10⁶ | Pa·s/m |
| Compressibility | $\kappa_f = 1/K$ | 4.556 × 10⁻¹⁰ | Pa⁻¹ |

**Attenuation:** None. The Helmholtz equation is solved losslessly ($k$ is real). The only dissipation mechanism is radiation through impedance boundaries. Viscosity enters only for particle dynamics (Stokes drag), not the acoustic PDE.

---

## 3. Governing Physics (Helmholtz)

### PDE

The model solves the homogeneous Helmholtz equation for the complex pressure phasor $p(\mathbf{x})$:

$$\nabla^2 p + k^2 p = 0$$

where $k = \omega / c$ is the wavenumber.

### Harmonic Convention

Time dependence assumed: $\hat{p}(\mathbf{x}, t) = \mathrm{Re}\bigl[p(\mathbf{x}) \, e^{-i\omega t}\bigr]$.

- $\omega = 2\pi f$ with $f = 500 \times 10^3$ Hz (nominal).
- At $f = 500$ kHz: $\omega = 3.14159 \times 10^6$ rad/s, $k = 2116.976$ rad/m, $\lambda = 2.968$ mm.

**COMSOL note:** COMSOL Pressure Acoustics uses the same $e^{+i\omega t}$ OR $e^{-i\omega t}$ convention depending on version. Verify which convention is active. Our convention is $e^{-i\omega t}$, which means:
- Euler equation: $\nabla p = i\omega\rho\,\mathbf{v}$ (positive $i$).
- Normal derivative from velocity: $\partial p / \partial n = i\omega\rho\,v_n$.
- If COMSOL uses $e^{+i\omega t}$, all $i$ factors below must be negated.

### Weak Form

The FEniCSx implementation uses integration by parts. Multiplying the Helmholtz equation by the conjugate test function $\bar{v}$ and integrating:

$$\int_\Omega \nabla p \cdot \nabla \bar{v} \, dx - k^2 \int_\Omega p \, \bar{v} \, dx = \oint_{\partial\Omega} \frac{\partial p}{\partial n} \bar{v} \, ds$$

The boundary integral on the right-hand side contains both source terms (Neumann) and absorbing terms (Robin, moved to LHS).

**UFL form (bilinear):**
```
a = ∫(∇u · ∇v̄ - k²u v̄) dx  +  Robin boundary terms
```

**Conjugated test function:** UFL's `inner(u, v)` for complex scalars computes $u \bar{v}$. The test function is conjugated automatically.

### Losses

None in the PDE. The wavenumber $k$ is purely real. There is no $k \to k + i\alpha$ damping.

---

## 4. Boundary Conditions

### 4a. Side Walls ($x = 0$, $x = L_x$, $y = 0$, $y = L_y$) — Standing-Wave Neumann Source

Each wall is a flat-face transducer. The BC is a prescribed normal velocity (Neumann type):

$$\frac{\partial p}{\partial n} = -i\omega\rho \, v_n$$

where $v_n$ is the normal velocity **into** the domain (i.e., opposite to the outward normal $\hat{n}$).

**Antiphase pairing** (`standing_phase_pattern = "antiphase"`):

| Wall | Outward normal | Source amplitude $g$ added to RHS |
|------|---------------|-----------------------------------|
| $x = 0$ | $-\hat{x}$ | $g_s = -i\omega\rho\,V_s$ |
| $x = L$ | $+\hat{x}$ | $-g_s = +i\omega\rho\,V_s$ |
| $y = 0$ | $-\hat{y}$ | $g_s = -i\omega\rho\,V_s$ |
| $y = L$ | $+\hat{y}$ | $-g_s = +i\omega\rho\,V_s$ |

Here $V_s$ = `standing_velocity_amplitude`.

**Physical interpretation:** Both walls of each pair drive with the same physical amplitude into the domain but with a relative phase shift of $\pi$. This produces a standing wave with antinodes at the centre and (depending on frequency) at the walls.

**Sign convention detail:** In DOLFINx, the weak-form boundary integral from IBP is $\oint (\partial p / \partial n)\,\bar{v}\,ds$ where $n$ is the **outward** normal. For a transducer at $x = 0$ pushing inward (into $+x$), the physical velocity into the domain is $V_s$ in the $+x$ direction, but the outward normal points in $-x$. The Euler relation gives $\partial p / \partial n = i\omega\rho \, (v \cdot \hat{n})$. Since the physical velocity is in $+x$ and $\hat{n}$ at $x=0$ is in $-x$: $v \cdot \hat{n} = -V_s$. So $\partial p / \partial n = -i\omega\rho V_s$ at $x=0$.

At $x = L$: $\hat{n} = +\hat{x}$, antiphase velocity direction is $-x$ (into domain), so $v \cdot \hat{n} = -V_s$ but with $\pi$ flip: the code applies $+g_s$ at $x=0$ and $-g_s$ at $x=L$.

**COMSOL implementation:**
- Use **Normal Velocity** BC on each wall.
- $x=0$: $v_n = V_s$ (into domain).  
- $x=L$: $v_n = V_s \cdot e^{i\pi} = -V_s$ (antiphase).  
- Same for $y$ walls when `standing_axis = "both"`.

**No impedance/Robin term** on side walls. They are purely reflective (rigid) when inactive, and pure Neumann source when active. No absorption at the walls.

**Mode logic:**
- `standing` or `combined`: all 4 walls active with above pattern.
- `vortex`: all 4 walls inactive (rigid, $\partial p / \partial n = 0$).

### 4b. Bottom Disc ($z = 0$, $r \le R_\mathrm{disc}$) — Robin Impedance + Vortex Source

This boundary has **two** contributions:

**Robin impedance (absorbing):**

$$\frac{\partial p}{\partial n}\bigg|_\mathrm{Robin} = \frac{i\omega\rho}{Z_w} \, p$$

with $Z_w = \rho c = 1.4795 \times 10^6$ Pa·s/m.

In the weak form, this moves to the LHS bilinear form:
$$a \mathrel{+}= \alpha_\mathrm{disc} \int_{\Gamma_\mathrm{disc}} u \, \bar{v} \, ds, \quad \alpha_\mathrm{disc} = -\frac{i\omega\rho}{Z_w} = -ik$$

(negative sign from moving RHS to LHS).

**Vortex pattern source (Neumann):**

$$\frac{\partial p}{\partial n}\bigg|_\mathrm{source} = -i\omega\rho \, v_\mathrm{vortex}(x, y)$$

The vortex velocity pattern is:

$$v_\mathrm{vortex}(x, y) = V_0 \, A(r) \, e^{i(\ell\theta + \varphi_0)}$$

where:
- $V_0$ = `vortex_velocity_amplitude` = 10 × 10⁻⁶ m/s.
- $\ell$ = `vortex_topological_charge` = 1 (integer azimuthal mode number).
- $\theta = \mathrm{atan2}(y - c_y, \, x - c_x)$ is the azimuthal angle around the disc centre.
- $\varphi_0$ = `vortex_phase_offset` = 0 rad.
- $r = \sqrt{(x - c_x)^2 + (y - c_y)^2}$.

**Radial amplitude envelope** $A(r)$ for `vortex_apodization = "cosine_taper"`:

$$A(r) = \frac{1}{2}\left(1 + \cos\frac{\pi r}{R_\mathrm{disc}}\right), \quad r \le R_\mathrm{disc}$$

$A(r) = 0$ for $r > R_\mathrm{disc}$ (but these points are on `bottom_rigid`, not the disc).

**Full RHS contribution from vortex source (added to linear form):**

$$L_\mathrm{vtx}(\bar{v}) = -i\omega\rho\,V_0 \int_{\Gamma_\mathrm{disc}} A(r)\,e^{i\ell\theta} \, \bar{v} \, ds$$

**COMSOL implementation:**
- Apply **Impedance** BC with $Z = Z_w$ on the disc.
- Separately apply a **Normal Velocity** BC with the vortex pattern expression.
- For the velocity expression, define:
  ```
  r = sqrt((x - cx)^2 + (y - cy)^2)
  theta = atan2(y - cy, x - cx)
  A_r = 0.5*(1 + cos(pi*r/R_disc))
  v_vortex = V0 * A_r * exp(i*ell*theta)
  ```
- Be careful with the interaction between impedance and velocity BCs in COMSOL — they may need to be combined into a single "Impedance with source" BC using:
  $$\frac{\partial p}{\partial n} = \frac{i\omega\rho}{Z_w}p - i\omega\rho\,v_\mathrm{vortex}$$

**Mode logic:**
- `vortex` or `combined`: vortex source ON.
- `standing`: vortex source OFF (disc is still impedance-matched, it just has no driving signal).

### 4c. Bottom Outside Disc ($z = 0$, $r > R_\mathrm{disc}$) — Rigid

$$\frac{\partial p}{\partial n} = 0$$

This is the natural Neumann BC. No term is added to either the bilinear or linear form for this boundary.

**COMSOL:** Apply **Sound Hard Wall** (or leave as default if Pressure Acoustics treats unmarked boundaries as rigid).

### 4d. Top ($z = H$) — Low-Impedance Robin (Air Interface)

$$\frac{\partial p}{\partial n} = \frac{i\omega\rho}{Z_\mathrm{top}} \, p$$

where $Z_\mathrm{top} = \epsilon \cdot Z_w$ with $\epsilon$ = `top_impedance_factor` = 0.001.

| Parameter | Value |
|-----------|-------|
| $Z_\mathrm{top}$ | $0.001 \times 1.4795 \times 10^6 = 1479.5$ Pa·s/m |

This is very close to a pressure-release boundary ($p \approx 0$) but implemented as a finite-impedance Robin BC to avoid the singular Dirichlet constraint and to better model the water–air interface.

**Bilinear form contribution:**

$$a \mathrel{+}= \alpha_\mathrm{top} \int_{\Gamma_\mathrm{top}} u\,\bar{v}\,ds, \quad \alpha_\mathrm{top} = -\frac{i\omega\rho}{Z_\mathrm{top}}$$

**COMSOL:** Apply **Impedance** BC with $Z = Z_\mathrm{top} = 1479.5$ Pa·s/m.

Alternatively, use $Z = 0.001 \times Z_w$ to keep it parametric.

---

## 5. Source Signals

| Signal | Symbol | Value | Unit |
|--------|--------|-------|------|
| Standing-wave velocity amplitude | $V_s$ | 10 × 10⁻⁶ | m/s |
| Vortex velocity amplitude | $V_0$ | 10 × 10⁻⁶ | m/s |
| Standing phase pattern | — | antiphase ($\pi$ between opposite walls) | — |
| Standing axis | — | "both" (all 4 walls active) | — |
| Vortex topological charge | $\ell$ | 1 | — |
| Vortex apodization | — | cosine taper | — |

**Relative phase between walls:**
- $x = 0$ and $y = 0$: phase 0.
- $x = L$ and $y = L$: phase $\pi$.
- The $x$ and $y$ wall pairs use the **same** amplitude and are **in phase** with each other (i.e., $x=0$ and $y=0$ both at phase 0). There is no $\pi/2$ shift between $x$ and $y$ pairs.

**No amplitude scaling or normalization:** The velocity amplitudes are applied directly as boundary conditions. No post-hoc pressure normalization is performed.

---

## 6. Frequency Settings

| Parameter | Value |
|-----------|-------|
| Nominal frequency | 500 kHz |
| Sweep range (Batch 1) | 475, 487.5, 500, 512.5, 525 kHz |
| Sweep fractions | −5%, −2.5%, 0%, +2.5%, +5% |

**COMSOL frequency sweep setup:**
- Use **Parametric Sweep** or **Frequency Domain** study.
- Define parameter `f0 = 500e3`.
- Sweep: `f0 * [0.95, 0.975, 1.0, 1.025, 1.05]`.
- All geometry, mesh, and BCs should be parametric in $f$ (recalculate $\omega$, $k$ at each step).
- The mesh does NOT change between frequency steps (reuse nominal mesh).

---

## 7. Mesh Notes

| Parameter | Recommendation |
|-----------|----------------|
| Element type | Tetrahedral (free mesh) |
| Polynomial order | P2 (quadratic Lagrange) in FEniCSx; use **Quadratic** elements in COMSOL |
| Elements per wavelength | $N \ge 6$ (FEniCSx uses $N = 6$) |
| Maximum element size | $h_\mathrm{max} \le \lambda / N = 2.968 / 6 \approx 0.49$ mm |
| $z$-direction | At least 8 elements across depth $H$ |
| Disc edge refinement | Refine near $r = R_\mathrm{disc}$ on $z = 0$ face (disc–rigid transition) |

**Batch 1 mesh (validated):** $20 \times 20 \times 8$ structured hex grid, 19200 tetrahedral cells, 3969 vertices, 28577 P2 DOFs.

For COMSOL with P2 elements: a similar resolution should give comparable results. The structured grid is not essential — COMSOL's free tetrahedral mesher is fine, but ensure:
- Boundary layer mesh is **not** needed (no viscous BL in this lossless model).
- Maximum element size does not exceed $\lambda / 6$.
- The disc edge on the bottom has a **size constraint** or **edge refinement** to resolve the boundary between disc and rigid regions.

---

## 8. Solver Settings

| Setting | FEniCSx value | COMSOL recommendation |
|---------|--------------|----------------------|
| Study type | Frequency domain | Frequency Domain |
| Solver | GMRES + ILU(0) | **MUMPS (direct)** recommended |
| Relative tolerance | $10^{-8}$ | $10^{-6}$ to $10^{-8}$ |
| Max iterations | 3000 | N/A for direct solver |

**Notes:**
- FEniCSx uses iterative GMRES+ILU because DOLFINx's LinearProblem defaults to PETSc iterative methods. For the Batch 1 mesh (~28k DOFs) this converges in seconds.
- COMSOL should use a **direct solver** (MUMPS or PARDISO) for robustness. At ~30k DOFs this is trivial.
- For larger domains (L = 50 mm, ~10⁶ DOFs), iterative solvers with appropriate preconditioners become necessary in both FEniCSx and COMSOL.

---

## 9. Fields to Export from COMSOL

For each mode (`standing`, `vortex`, `combined`), evaluate and export:

| Field | Expression | Notes |
|-------|-----------|-------|
| $\mathrm{Re}(p)$ | `real(acpr.p_t)` | Real part of phasor |
| $\mathrm{Im}(p)$ | `imag(acpr.p_t)` | Imaginary part |
| $|p|$ | `abs(acpr.p_t)` | Pressure magnitude |
| $\arg(p)$ | `arg(acpr.p_t)` | Phase angle [rad] |
| $|v_1|$ | `acpr.v_mag` or `abs(acpr.p_t)/(omega*rho)` gradient magnitude | First-order velocity magnitude |

**Gradient-based velocity:**

$$|\mathbf{v}_1| = \frac{|\nabla p|}{\omega \rho}$$

In COMSOL: use `acpr.v_mag` or compute manually from `sqrt(abs(acpr.px)^2 + abs(acpr.py)^2 + abs(acpr.pz)^2) / (omega_src * rho)`.

### Slice Planes for Comparison

| Slice | Definition | Fields to plot |
|-------|-----------|----------------|
| Mid-plane | $z = H/2$ | $|p|$, $\arg(p)$, $|v_1|$ |
| Near-bottom | $z = H/10$ | $|p|$, $|v_1|$ |

### Export Format

Export as either:
- COMSOL `.txt` with columns (x, y, z, Re(p), Im(p), ...) on a regular grid.
- VTU or XDMF for direct ParaView comparison.

---

## 10. Comparison Metrics (to compute in Python)

After exporting both FEniCSx and COMSOL fields on a common grid:

### 10a. Relative $L^2$ Error

$$\varepsilon_{L^2} = \frac{\|p_\mathrm{FEniCSx} - p_\mathrm{COMSOL}\|_{L^2}}{\|p_\mathrm{COMSOL}\|_{L^2}}$$

Compute on the mid-plane grid:
```python
err = np.sqrt(np.nanmean(np.abs(p_fenics - p_comsol)**2))
ref = np.sqrt(np.nanmean(np.abs(p_comsol)**2))
epsilon_L2 = err / ref
```

### 10b. Phase-Aligned Comparison

Because FEniCSx and COMSOL may have an arbitrary global phase offset (both solve a linear system up to a complex scalar), align phases before comparing:

$$p_\mathrm{aligned} = p_\mathrm{FEniCSx} \cdot e^{i\phi_\mathrm{corr}}$$

where $\phi_\mathrm{corr}$ minimises the $L^2$ error. In practice:

```python
# Cross-correlation phase correction
phi_corr = np.angle(np.sum(p_fenics * np.conj(p_comsol)))
p_aligned = p_fenics * np.exp(-1j * phi_corr)
```

### 10c. Trap Location Comparison

1. Compute Gor'kov potential $U(\mathbf{x})$ from both solutions.
2. Find local minima (trap positions) in both.
3. Match nearest traps and report displacement $|\Delta \mathbf{x}_\mathrm{trap}|$.
4. Compare trap depths $\Delta U$.

### 10d. Stiffness Eigenvalue Comparison

At each matched trap, compute the Hessian $H_{ij} = \partial^2 U / \partial x_i \partial x_j$ and compare eigenvalues (trap stiffness).

---

## Appendix A: Particle Properties (for Gor'kov potential comparison)

| Property | Symbol | Value | Unit |
|----------|--------|-------|------|
| Particle radius | $a$ | 5 × 10⁻⁶ | m |
| Particle density | $\rho_p$ | 1050 | kg/m³ |
| Particle compressibility | $\kappa_p$ | 2.4 × 10⁻¹⁰ | Pa⁻¹ |
| Monopole contrast $f_1$ | $1 - \kappa_p / \kappa_f$ | $\approx 0.473$ | — |
| Dipole contrast $f_2$ | $2(\rho_p - \rho)/(2\rho_p + \rho)$ | $\approx 0.034$ | — |

Gor'kov potential:

$$U = \frac{4\pi}{3} a^3 \left[ f_1 \frac{\langle p^2 \rangle}{2\rho c^2} - f_2 \frac{3\rho}{4} \langle |\mathbf{v}|^2 \rangle \right]$$

where $\langle p^2 \rangle = |p|^2/2$ and $\langle |\mathbf{v}|^2 \rangle = |\nabla p|^2 / (2\omega^2 \rho^2)$.

## Appendix B: Summary of FEniCSx Batch 1 Results (Validation Targets)

| Metric | Standing | Vortex | Combined |
|--------|----------|--------|----------|
| max $|p|$ [Pa] | 93.15 | 11.16 | 88.10 |
| max $|v_1|$ mid-plane [μm/s] | 40.53 | 6.83 | 38.40 |
| Phase winding (vortex) | — | 1.000 | — |
| Interaction metric | — | — | 0.087 |

COMSOL results should match these to within ~5% for the same mesh resolution, and converge as mesh is refined.

## Appendix C: Quick COMSOL Setup Checklist

1. **New Model** → 3D → Pressure Acoustics, Frequency Domain.
2. **Geometry** → Block: width $L_x$, depth $L_y$, height $H$.
3. **Work Plane** at $z = 0$ → Circle ($R_\mathrm{disc}$, centred at $(c_x, c_y)$).
4. **Partition Boundaries** → bottom face split into disc and rigid.
5. **Material** → Water: $\rho = 997$, $c = 1484$.
6. **Impedance BC** → top face: $Z = 0.001 \cdot \rho c$.
7. **Impedance BC** → bottom disc: $Z = \rho c$.
8. **Normal Velocity** → bottom disc (vortex pattern): `V0 * 0.5*(1+cos(pi*r/R_disc)) * exp(i*ell*atan2(y-cy,x-cx))`.
9. **Normal Velocity** → $x=0$: `Vs`, $x=L$: `-Vs`, $y=0$: `Vs`, $y=L$: `-Vs`.
10. **Sound Hard Wall** → bottom rigid (or leave as default).
11. **Mesh** → Free Tetrahedral, max element size $\le \lambda/6$.
12. **Study** → Frequency Domain at $f = 500$ kHz.
13. **Solver** → Direct (MUMPS).
14. **Export** → Mid-plane slice of $|p|$, $\arg(p)$, $|v_1|$.
