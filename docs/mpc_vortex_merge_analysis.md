# MPC Vortex-Merge Transport — Analytical Note

**Date:** 2025-03-31  
**Script:** `scripts/dev/mpc_vortex_merge.py`  
**Reference run (LG):** `results/dev/mpc_vortex_merge/run_20260331_173503/`

---

## 1. Objective

Transport a remote particle **A** (placed 2.0 λ from target **B**) toward a
stationary cluster at **B**, then merge into the standing-wave (SW) lattice.
The vortex must not disturb trapped cluster particles.

### Key changes from the prior version

| Concern | Change |
|---------|--------|
| Vortex disturbs trapped particles | Delayed MPC activation: open-loop brings A close before MPC engages cluster-aware control |
| Vortex moves too fast | Per-DOF rate limits (`du_max`): ψ free (2 π/step), x_v/y_v slow (adjustable, default 0.5–2.0 µm/step) |
| Phase delay ψ underused | `w_smooth_vec[ψ] = 1.0` (effectively free); MPC now varies ψ actively (544/800 non-zero ψ steps in reference run) |
| LG beam unrealistic for acoustics | Bessel vortex from finite aperture via ASM added (`make_bessel_vortex_field`); see §4 for findings |

---

## 2. Three-phase pipeline

### Phase I — Open-loop tethered pickup

The vortex centres on A's initial position and translates toward B at a
configurable max speed (default 5 mm/s).  A **leash** constraint clamps the
vortex: if the vortex–A distance exceeds 0.3 λ ≈ 223 µm, the vortex halts
until A catches up.

During open-loop:

- **α_ol = 3.0** (strong vortex for capture)
- **β_ol = 0.5** (weakened SW so the vortex can pull A out of its trap)
- **ψ = 0** (no optimisation — open-loop constant)

Forces are evaluated via the fast 5-point stencil (`_fast_forces_at_pts`),
which avoids the full 400 × 400 trilinear basis rebuild per step.  This keeps
memory O(1) regardless of the number of open-loop steps.

**Result:** d(A, B) reduced 1 484 → 736 µm in 1 484 steps (148 ms).

### Phase II — MPC-controlled approach

Standard receding-horizon MPC with discrete adjoint gradients (L-BFGS-B).
The optimizer controls all 5 DOFs: ψ, x_v, y_v, α, β.

| Parameter | Value |
|-----------|-------|
| K (horizon) | 10 |
| T (steps) | 800 |
| replan_every | 5 |
| n_iters | 10 |
| vxy_rate | 2.0 µm/step |
| ψ_rate | 2 π/step (free) |

Per-DOF smoothness weighting via `w_smooth_vec`:

$$\mathbf{w}_{\text{smooth}} = [1,\; w_s,\; w_s,\; 0.1\,w_s,\; 0.1\,w_s]$$

where $w_s = 10^8$.  The ψ weight is 10⁸× smaller than x/y, so the
optimizer treats ψ as essentially free.

**Result:** d(A, B) reduced 736 → 224 µm during MPC (800 steps, 80 ms).

### Phase III — Settling (α → 0)

Vortex amplitude α ramps linearly to 0 over 50 steps, then 2 950 relaxation
steps under pure SW forces.  Particles re-trap at nearest SW nodes.

**Result:** d(A, B) settled to **65.9 µm** (< capture radius 111 µm).
Merge time: **278.6 ms** from t = 0.

---

## 3. ψ usage analysis

The phase delay ψ controls the interference phase between the standing wave
and the vortex:

$$p_{\text{tot}} = \beta\, p_{\text{sw}} + \alpha\, e^{i\psi}\, p_v$$

During MPC, ψ varied between 0 and 2 π, with a mean of ~0.97 rad (≈ π/3).
544 of 800 MPC steps used non-zero ψ.  This demonstrates that the optimizer
finds value in rotating the vortex-SW interference pattern to shape the
Gor'kov potential landscape during transport.

**Key observation:** ψ jumps rapidly between 0 and 2 π at early MPC steps
(large cost sensitivity), then stabilises at ~0.27 rad during the final
approach.  This suggests ψ is most useful during transitions between SW traps,
where the interference cross-term $2 \alpha \beta \operatorname{Re}(e^{i\psi} p_v\,p_{\text{sw}}^*)$ can
break local potential barriers.

---

## 4. Bessel vortex findings

The function `make_bessel_vortex_field()` generates a uniform-amplitude source
with spiral phase $e^{i\ell\theta}$ inside a circular aperture of radius $R$,
then propagates via angular spectrum method (Rayleigh–Sommerfeld Type-I) over
distance $z$.

### Ring radius measurements

| $R$ (mm) | $z$ (mm) | Ring radius (µm) | Ring radius (λ) |
|-----------|----------|-------------------|-----------------|
| 2.0 | 5.0 | **877** | 1.18 |
| 2.0 | 2.0 | 995 | 1.34 |
| 2.0 | 1.0 | 1 010 | 1.36 |
| 1.0 | 1.0 | **460** | 0.62 |
| 1.0 | 0.5 | 464 | 0.63 |
| 3.0 | 1.0 | 980 | 1.32 |

**Minimum realistic ring radius ≈ 460 µm (0.62 λ)**, limited by diffraction.
This is 3× wider than the LG waist (150 µm = 0.20 λ), which is
sub-wavelength and therefore physically unrealistic.

### Transport performance with Bessel vortex

With the tightest Bessel ring (R = 1 mm, z = 1 mm, ring ≈ 460 µm):

- **Open-loop (α = 5, β = 0.5):**  d(A, B) 1 484 → 1 312 µm (172 µm, 12%)
- **After MPC + settle:** d(A, B) = 1 352 µm — **no merge**.
- **Cluster disturbance:** max neighbour displacement 1 349 µm.

Root cause: the Bessel beam's ring extends over the full grid (~6 mm), and
its sidelobes create forces at the cluster location.  At α = 5, these
sidelobes alone displace cluster particles by > 1 mm.  Meanwhile, the wider
gradient at the ring means weaker restoring force per unit displacement.

### Why LG succeeds where Bessel struggles

The LG amplitude envelope $A(r) = (r/w)^{|\ell|} e^{-r^2/w^2}$ falls off as
a Gaussian beyond the waist.  At $r = 3w$ the field is < 0.01% of peak — sidelobes
are negligible.  The Bessel beam $J_1(k_r r)$ decays only as $r^{-1/2}$,
maintaining significant amplitude many wavelengths from the ring.

**Force comparison at particle A, vortex offset 50 µm (α = 1.5, β = 1.0):**

| Setting | |F| | Particle v (mm/s) |
|---------|------|---------------------|
| SW only (α = 0) | 6.1 × 10⁻³ | 15.4 |
| LG vortex + SW | 6.8 × 10⁻³ | 17.0 |
| Bessel vortex + SW | 6.8 × 10⁻³ | 17.0 |
| Bessel vortex only (β = 0) | 3.9 × 10⁻⁴ | 0.96 |

The SW dominates in all cases.  The vortex restoring force (toward vortex
centre) is ~17× weaker than the SW trapping force.  Only by reducing β (weakening
SW) does the vortex gain sufficient authority — but this releases all particles.

---

## 5. Summary of metrics (LG reference run)

| Metric | Value |
|--------|-------|
| d(A, B) final | 65.9 µm |
| Merge time | 278.6 ms |
| A_success | ✓ |
| B_stable | ✓ (7.9 µm displacement) |
| neighbour_stable | ✗ (max 358 µm, mean 89 µm) |
| Classification | partial_success |
| Total steps | 5 284 (OL: 1 484, MPC: 800, settle: 3 000) |
| MPC wall-clock | 1 011 s |

---

## 6. Recommendations & next steps

### Immediate

1. **Increase T_settle or add explicit neighbour re-trapping weight** to
   improve neighbour stability during settling (currently max disp = 358 µm).

2. **Scan ψ during open-loop** — rather than fixing ψ = 0, do a 1-D line
   search over ψ ∈ [0, 2 π] each step to maximise the force component along
   the transport direction.  This leverages the cross-term that is currently
   wasted in open-loop.

3. **Reduce β_ol progressively** — instead of a fixed β = 0.5 during
   open-loop, ramp β from 1.0 → 0.3 → 1.0 (parabolic profile) so the cluster
   is only weakened when the vortex is far from it.

### Medium-term

4. **Make the Bessel beam work:**
   - Use a focused annular transducer (ring aperture) instead of a filled
     aperture — this suppresses sidelobes and can produce a tighter ring.
   - Apodize with a steeper taper (Hann or Blackman instead of cosine).
   - Consider propagating to the geometric focus of a curved aperture.

5. **Array-aware vortex:** Replace the single-source Bessel model with a
   phased array that synthesises a vortex beam with controlled sidelobe
   levels.  This is the realistic acoustic tweezer scenario.

6. **Quantify ψ contribution:** Run a control experiment with ψ clamped at
   0 and compare merge time / neighbour disturbance to the free-ψ case.
   This directly measures the value of phase-delay optimisation.

### Physics concerns documented

- The LG beam (w = 0.15 mm = 0.20 λ) achieves sub-wavelength focusing, which
  is **unrealistic for free-space acoustics at 2 MHz**.  Any experimental
  validation should use a physically achievable beam profile.

- The overdamped dynamics model ($\dot{x} = \mu F_{\text{Gor'kov}}$) ignores
  particle inertia and acoustic streaming.  Real transport at 278 ms timescale
  would include streaming-induced drift.

- The Gor'kov potential is a time-averaged quantity valid for $a \ll \lambda$.
  For 50 µm particles at λ = 742 µm, $a/\lambda \approx 0.034$, safely within
  the Rayleigh limit.
