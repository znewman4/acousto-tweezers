# Vortex Lens Investigation — Phase 1: Lens/Field Sweep + Ranking

**Date:** 2026-02-25  
**Script:** `scripts/experiments/vortex_lens_sweep.py`  
**Results:** `results/vortex_lens_sweep_20260225_152950/`  
**Runtime:** 7632 s (127 min) — 96 FEM solves (1 standing + 95 vortex)

---

## 1. Objective

Compare four vortex beam families — **Laguerre–Gaussian (LG)**, **Bessel**,
**Bessel–Gaussian (BG)**, and **Plastic lens** (baseline) — across a parameter
sweep and rank them by a composite score J that balances non-disruption of the
standing-wave trap pattern against local vortex quality.

This is a **field-level investigation only** (no particle dynamics).

---

## 2. Setup

| Parameter | Value |
|---|---|
| Preset | `CORRECTED_PRESET` |
| Frequency | 2.0 MHz |
| λ | 0.742 mm |
| k | 8467.9 rad/m |
| Domain | 6 × 6 × 5.01 mm |
| Mesh | 4 elem/λ → 232k DOFs, ~50 s/solve |
| Trap plane z* | 4.190 mm (z_mid + 0.25λ) |
| ROI | 1.5λ radius around (3.0, 3.0) mm |
| Grid | 200 × 200 |
| Standing wave | V = 10 µm/s, antiphase, both axes |
| Vortex | V = 1 µm/s (→ V_ratio = 10) |
| Combination | p_comb = p_stand + α · 10 · p_vortex |
| α values | 0.05, 0.10, 0.20, 0.40 |

### Hard constraints

| Metric | Threshold |
|---|---|
| η_out (outside-ROI relative ΔE) | ≤ 0.02 |
| corr_out (outside-ROI |p| correlation) | ≥ 0.995 |

### Composite score

J = 0.20 × (1 − η/η_thr) + 0.15 × corr + 0.25 × E_roi + 0.25 × core_ratio/10 + 0.15 × (1 − winding_std)

Configs failing hard constraints get J = −1.

---

## 3. Parameter Space (95 configs)

| Family | Swept parameters | Count |
|---|---|---|
| **LG** | ℓ ∈ {1,2,3}, w ∈ {0.4R, 0.6R, 0.8R}, f ∈ {off, 1.5, 2.0, 2.5 mm} | 36 + 2 checks = 38 |
| **Bessel** | ℓ ∈ {1,2,3}, k_r ∈ {0.5k, 1.0k, 1.5k} | 9 + 2 checks = 11 |
| **BG** | ℓ ∈ {1,2,3}, k_r ∈ {0.5k, 1.0k, 1.5k}, w ∈ {0.4R, 0.6R, 0.8R} | 27 + 1 check = 28 |
| **Plastic** | ℓ ∈ {1,2,3}, f ∈ {1.5, 2.0, 2.5 mm}, offset ∈ {0, 0.2 mm} | 18 |

All configs use cosine-taper apodization (default), with spot-checks at
uniform for LG and Bessel.

---

## 4. Results Summary

### 4.1 Pass/Fail by family (α = 0.20)

| Family | Pass | Fail | Rate |
|---|---|---|---|
| BG | 25 | 3 | **89%** |
| Plastic | 12 | 6 | 67% |
| Bessel | 3 | 8 | 27% |
| LG | 6 | 32 | **16%** |
| **Total** | **46** | **49** | 48% |

### 4.2 Top 5 overall (α = 0.20)

| Rank | Config | J | η_out | corr_out | E_roi | core_ratio |
|---|---|---|---|---|---|---|
| 1 | `bg_l1_w0.4_kr12702_cos` | 0.507 | 0.0046 | 0.99998 | 0.132 | 0.78 |
| 2 | `bg_l1_w0.6_kr12702_cos` | 0.497 | 0.0058 | 0.99997 | 0.138 | 0.81 |
| 3 | `bg_l3_w0.6_kr12702_cos` | 0.441 | 0.0016 | 1.00000 | 0.054 | 0.55 |
| 4 | `bg_l1_w0.8_kr12702_cos` | 0.420 | 0.0064 | 0.99996 | 0.139 | 0.82 |
| 5 | `bg_l1_w0.8_kr4234_cos` | 0.408 | 0.0151 | 0.99982 | 0.157 | 0.77 |

**BG dominates all top 5 positions.**

### 4.3 Best per family (α = 0.20, per ℓ)

| Family | ℓ | Best config | J | η_out | E_roi |
|---|---|---|---|---|---|
| BG | 1 | `bg_l1_w0.4_kr12702_cos` | 0.507 | 0.005 | 0.132 |
| BG | 2 | `bg_l2_w0.6_kr12702_cos` | 0.398 | 0.009 | 0.146 |
| BG | 3 | `bg_l3_w0.6_kr12702_cos` | 0.441 | 0.002 | 0.054 |
| Bessel | 1 | `bessel_l1_kr12702_uni` | 0.384 | 0.018 | 0.138 |
| Bessel | 3 | `bessel_l3_kr12702_uni` | 0.327 | 0.008 | 0.075 |
| Plastic | 1 | `plastic_l1_f2.5_off0.2_cos` | 0.376 | 0.018 | 0.144 |
| Plastic | 3 | `plastic_l3_f2.5_off0.2_cos` | 0.262 | 0.015 | 0.078 |
| LG | 1 | `lg_l1_w0.4_f1.5_cos` | 0.273 | 0.020 | 0.101 |
| LG | 3 | `lg_l3_w0.4_f1.5_cos` | 0.184 | 0.019 | 0.064 |
| LG | 2 | (all fail) | −1.0 | 0.042 | 0.143 |
| Bessel | 2 | (all fail) | −1.0 | 0.127 | 0.133 |
| Plastic | 2 | (all fail) | −1.0 | 0.044 | 0.136 |

**ℓ = 2 fails hard constraints** for LG, Bessel, and Plastic families (only
BG ℓ=2 passes). The ℓ = 2 vortex beam creates a wider core that disrupts the
standing-wave pattern more strongly.

---

## 5. Critical Finding: Evanescent Beam Bias

### Dispersion analysis

| k_r | k_r / k | k_z | Regime |
|---|---|---|---|
| 4234 rad/m | 0.5 | 7333 rad/m | **Propagating** |
| 8468 rad/m | 1.0 | ≈ 0 | Marginal / evanescent |
| 12702 rad/m | 1.5 | i × 9467 rad/m | **Evanescent** |

For Bessel/BG beams, the axial wavenumber $k_z = \sqrt{k^2 - k_r^2}$.
When $k_r > k$, the beam becomes evanescent in z, decaying exponentially
before reaching the trap plane.

**The top 4 ranked configs all use k_r = 1.5k (evanescent).** They achieve
low η_out and high corr_out trivially — their fields are too weak at z* to
disrupt anything. This is a degenerate solution that the current composite
score rewards.

### Propagating-only rankings (k_r ≤ 0.5k, or LG/Plastic)

| Rank | Config | J | η_out | E_roi | Note |
|---|---|---|---|---|---|
| 1 | `bg_l1_w0.8_kr4234_cos` | 0.408 | 0.015 | 0.157 | BG, propagating |
| 2 | `plastic_l1_f2.5_off0.2` | 0.376 | 0.018 | 0.144 | Plastic baseline |
| 3 | `plastic_l1_f1.5_off0.2` | 0.371 | 0.019 | 0.174 | Plastic, highest E_roi |
| 4 | `bg_l2_w0.4_kr4234_cos` | 0.339 | 0.015 | 0.144 | BG, ℓ=2 passes! |
| 5 | `bg_l3_w0.4_kr4234_cos` | 0.350 | 0.003 | 0.067 | BG, ℓ=3 |

Among propagating beams, **BG with k_r = 0.5k is best**, followed closely
by **Plastic** baseline. LG is consistently the weakest family.

---

## 6. Phase Topology

Winding numbers for best configs (per family, ℓ = 1, vortex-only field):

| Config | w @ 1.0λ | w @ 1.5λ | w @ 2.0λ | Stable? |
|---|---|---|---|---|
| `bg_l1_w0.4_kr12702` | 0.0 | 0.0 | 0.0 | No phase singularity |
| `bessel_l1_kr12702` | +1.0 | +1.0 | −1.0 | Sign flip at 2λ |
| `plastic_l1_f2.5_off0.2` | −1.0 | +1.0 | +1.0 | Sign flip at 1λ |
| `lg_l1_w0.4_f1.5` | 0.0 | 0.0 | −1.0 | Weak until 2λ |

**Winding numbers are unreliable for evanescent beams** (the phase structure
doesn't survive propagation). For propagating beams, the winding is noisy
at 4 elem/λ with NearestND interpolation; the r ≥ 2.0λ measurement is
more stable, consistent with pre-lens affirmation findings.

---

## 7. Alpha Sensitivity

For the overall top config (`bg_l1_w0.4_kr12702_cos`):

| α | η_out | corr_out | J | Pass |
|---|---|---|---|---|
| 0.05 | 0.001 | 1.0000 | 0.541 | Yes |
| 0.10 | 0.002 | 1.0000 | 0.530 | Yes |
| 0.20 | 0.005 | 0.9999 | 0.507 | Yes |
| 0.40 | 0.009 | 0.9999 | 0.461 | Yes |

All α values pass for this config. At α = 0.05, all families converge to
J ≈ 0.51–0.54 — the vortex contribution is too weak to differentiate.
**α = 0.20 provides the best balance** between discrimination power and
constraint satisfaction.

---

## 8. Parameter Sensitivities

### Beam waist w (LG, BG)
- Smaller w (0.4R) slightly favors low η (tighter beam, less energy spread)
- Larger w (0.8R) favors E_roi (more overlap with ROI)
- Optimal: w = 0.6R–0.8R for propagating BG

### Focal length (LG, Plastic)
- f = 1.5 mm gives highest E_roi (tightest focus)
- f = 2.5 mm gives lowest η_out (more spread)
- f = 2.0 mm is a balanced choice matching the CORRECTED_PRESET default

### Offset
- offset = 0.2 mm consistently improves Plastic family (breaks symmetry,
  moves vortex core away from standing-wave node)
- Little effect on BG/Bessel families

### Apodization
- cosine_taper preferred (default); uniform slightly increases η_out
  (sharper edge → more diffraction)

---

## 9. Outputs

| File | Description |
|---|---|
| `csv/base_metrics.csv` | All configs at α = 0.20, full metric set |
| `csv/alpha_metrics.csv` | All configs × all α values |
| `results.json` | Complete structured output (ranked, best per family, solve log) |
| `figures/family_{fam}_l{ell}_best_6panel.png` | 12 panels: |p|, phase, Δ|p|, winding for best config per family per ℓ |
| `figures/compare_families_l{ell}_panel.png` | 3 panels: side-by-side family comparison per ℓ |
| `figures/ranking_scatter.png` | η_out vs E_roi scatter with top-5 annotations |

---

## 10. Recommendations for Phase 2

1. **Exclude evanescent beams:** Restrict k_r ≤ 0.9k (or add minimum E_roi
   threshold to hard constraints). The current composite score trivially
   rewards weak fields.

2. **Focus on BG (k_r = 0.5k) and Plastic families** — these are the
   competitive propagating-beam options. Bessel and LG underperform.

3. **Refine α range:** Use α ∈ {0.15, 0.20, 0.25, 0.30} for finer
   discrimination (α = 0.05 is too weak, α = 0.40 fails many configs).

4. **Add minimum vortex strength constraint:** E.g., require E_roi ≥ 0.05
   or |p_vortex|_peak ≥ 0.5 Pa to prevent degenerate weak-field solutions.

5. **Investigate ℓ = 2 failure mode:** Only BG ℓ=2 passes constraints. The
   wider core of ℓ = 2 beams may require offset or reduced α to avoid
   standing-wave disruption. A targeted ℓ = 2 sweep with lower α could
   recover viable configs.

6. **Higher resolution for topology:** Winding number measurements need
   ≥ 6 elem/λ or bilinear phase interpolation for reliable detection (per
   pre-lens affirmation). Consider a targeted re-solve of top-5 configs at
   higher resolution for topology validation.

---

## 11. Verdict

**GREEN LIGHT for Phase 2** with the following constraints:

- Use **BG (k_r = 0.5k, w = 0.6R–0.8R)** as default vortex family
- Use **Plastic (f = 2.0 mm, offset = 0.2 mm)** as baseline comparator
- Restrict to **propagating beams only** (k_r ≤ k_water)
- Set **α = 0.20** as default mixing ratio
- Add **E_roi ≥ 0.05** to hard constraints to prevent degenerate solutions

**Overall family ranking (propagating beams):**
1. BG (k_r = 0.5k) — J = 0.41, best E_roi
2. Plastic — J = 0.38, reliable baseline
3. Bessel — J ≈ 0.32 (only ℓ=1 viable)
4. LG — J = 0.27, weakest performer
