"""
Batch rendering: orchestrate all 4 canonical views.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .loaders import load_rich, load_pair, clip_roi, list_phase_files
from .views import (
    view_2d_magnitude_slice,
    view_trap_geometry,
    view_particle_pluck,
    view_difference,
    view_trap_geometry_frame,
    _camera, _percentile_thresholds,
)


def render_all(run_dir, roi_center=None, roi_size=0.008):
    """
    Render all canonical views for a single run directory.

    Expects NPZ + XDMF files produced by generate_rich_data.py.
    
    Also computes and prints quantitative invariants:
      - Phase field validity: fraction of points with |p| ≥ threshold
      - U scaling check: validates Gor'kov formula (U ∝ |p|²)
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / 'canonical'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CANONICAL RENDERS — {run_dir.name}")
    print(f"{'='*70}")

    # --- Load data ---
    print("\nLoading data...")
    standing = load_rich(run_dir, 'standing')
    combined = load_rich(run_dir, 'combined')

    # Auto-detect ROI centre
    if roi_center is None:
        b = combined.bounds
        roi_center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2])

    # ---- Quantitative Invariants ----
    print("\n[INVARIANTS]")
    
    # Phase validity check
    clipped = clip_roi(combined, roi_center, roi_size)
    mag_clipped = clipped.point_data['magnitude']
    threshold_pa = 1.0  # Pa
    n_weak = np.sum(mag_clipped < threshold_pa)
    frac_weak = 100.0 * n_weak / len(mag_clipped)
    frac_strong = 100.0 - frac_weak
    print(f"  Phase field validity: {frac_strong:.1f}% of ROI has |p| ≥ {threshold_pa} Pa")
    
    # Gorkov scaling check (if available)
    if 'gorkov' in combined.point_data:
        u_combined = combined.point_data['gorkov']
        u_standing = standing.point_data['gorkov']
        u_comb_max = np.max(np.abs(u_combined))
        u_stand_max = np.max(np.abs(u_standing))
        # U ∝ |p|², so ratio should be ≈ (|p_comb|/|p_stand|)²
        p_comb_max = np.max(combined.point_data['magnitude'])
        p_stand_max = np.max(standing.point_data['magnitude'])
        expected_ratio = (p_comb_max / p_stand_max)**2 if p_stand_max > 0 else 0
        actual_ratio = u_comb_max / u_stand_max if u_stand_max > 0 else 0
        print(f"  Gor'kov scaling check: U_combined/U_standing ≈ {actual_ratio:.2f}")
        print(f"    (expected from |p|² scaling: {expected_ratio:.2f})")

    # ---- View 0: 2D magnitude slice ----
    print("\n[View 0] Magnitude cross-section (spatial grounding)...")
    view_2d_magnitude_slice(
        standing,
        out_dir / 'v0_magnitude_2d.png',
        roi_center=roi_center, roi_size=roi_size,
        title='Standing wave magnitude — cross-section at center z',
    )

    # ---- View 1: Trap geometry (standing) ----
    print("[View 1a] Standing wave trap geometry...")
    view_trap_geometry(
        standing,
        out_dir / 'v1_trap_standing.png',
        roi_center=roi_center, roi_size=roi_size,
        title='Standing wave — trap lattice',
    )

    # ---- View 1: Trap geometry (combined) ----
    print("[View 1b] Combined field trap geometry...")
    view_trap_geometry(
        combined,
        out_dir / 'v1_trap_combined.png',
        roi_center=roi_center, roi_size=roi_size,
        title='Combined — vortex reshapes traps',
    )

    # ---- View 1: Vortex only ----
    try:
        vortex = load_rich(run_dir, 'vortex')
        print("[View 1c] Vortex-only topology...")
        view_trap_geometry(
            vortex,
            out_dir / 'v1_trap_vortex.png',
            roi_center=roi_center, roi_size=roi_size,
            percentiles=(80, 90, 97),
            title='Vortex only — helical phase',
        )
    except FileNotFoundError:
        pass

    # ---- View 2: Particle pluck ----
    print("\n[View 2] Particle pluck (combined)...")
    view_particle_pluck(
        combined,
        out_dir / 'v2_particle_pluck.png',
        roi_center=roi_center, roi_size=roi_size,
        title='Particles at Gor\'kov minima — radiation force arrows',
    )

    # ---- View 4: Difference ----
    print("\n[View 4a] Δ|p| slice...")
    view_difference(
        standing, combined,
        out_dir / 'v4_delta_pressure.png',
        roi_center=roi_center, roi_size=roi_size,
        field='magnitude',
    )

    print("[View 4b] ΔU slice...")
    view_difference(
        standing, combined,
        out_dir / 'v4_delta_gorkov.png',
        roi_center=roi_center, roi_size=roi_size,
        field='gorkov',
        title='ΔU = U_combined − U_standing',
    )

    # ---- View 3: Phase sweep storyboard ----
    phase_files = list_phase_files(run_dir)
    if phase_files:
        print(f"\n[View 3] Phase sweep storyboard ({len(phase_files)} frames)...")
        render_phase_storyboard(
            run_dir, phase_files,
            out_dir, roi_center, roi_size,
        )

    print(f"\n{'='*70}")
    print(f"DONE — {out_dir}")
    print(f"{'='*70}\n")


def render_phase_storyboard(run_dir, phase_files, out_dir,
                            roi_center, roi_size):
    """
    View 3 — Phase sweep storyboard.

    Same camera, same iso-thresholds, multiple frames.
    Also assembles a single storyboard image.
    """
    run_dir = Path(run_dir)
    frame_dir = out_dir / 'phase_frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    # --- Compute SHARED thresholds from phase-0 data ---
    ref = load_rich(run_dir, phase_files[0][1])
    ref_clip = clip_roi(ref, roi_center, roi_size)
    mag = ref_clip.point_data['magnitude']
    thresholds = _percentile_thresholds(mag, [92, 96, 99])
    opacities  = [0.08, 0.06, 0.10]
    camera_pos = _camera(roi_center, roi_size)

    # --- Render each frame ---
    frame_paths = []
    for deg, name in phase_files:
        grid = load_rich(run_dir, name)
        fp = frame_dir / f'phase_{deg:03d}.png'
        view_trap_geometry_frame(
            grid, fp,
            roi_center=roi_center, roi_size=roi_size,
            camera_pos=camera_pos,
            thresholds=thresholds, opacities=opacities,
            resolution=(800, 800),
            label=f'φ = {deg}°',
        )
        frame_paths.append(fp)
        print(f"    frame: φ = {deg}°")

    # --- Assemble storyboard grid ---
    n = len(frame_paths)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.02, hspace=0.08)

    for i, (fp, (deg, _)) in enumerate(zip(frame_paths, phase_files)):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        img = plt.imread(str(fp))
        ax.imshow(img)
        ax.set_title(f'φ = {deg}°', fontsize=14, fontweight='bold')
        ax.axis('off')

    fig.suptitle('Phase Sweep — Trap Geometry Responds to Vortex Phase',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(str(out_dir / 'v3_phase_storyboard.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [View 3] v3_phase_storyboard.png")
