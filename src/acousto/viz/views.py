"""
Canonical views for acoustic-tweezers visualizations.

RULES (non-negotiable):
  1. Never volume-render acoustic magnitude.
  2. Render ONLY derived geometry (2-4 isosurfaces, streamlines, particles).
  3. Phase may only color thin objects (isosurfaces, curves).
  4. Aggressive spatial cropping to ROI.
  5. Explicit visual hierarchy: particles > trap surfaces > phase > context.
"""

from pathlib import Path
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .loaders import clip_roi


# ═══════════════════════════════════════════════════════════════════════
#  VORTEX PERTURBATION — HIGH-FIDELITY HERO RENDER
# ═══════════════════════════════════════════════════════════════════════

def render_vortex_perturbation_hires(standing_grid, combined_grid, out_path,
                                      roi_center=None, roi_size=0.010,
                                      percentile=97,
                                      resolution=(2400, 1400),
                                      title=None):
    """
    High-fidelity hero render of vortex–standing wave interaction.

    Visualizes the PERTURBATION field Δp = p_combined - p_standing:
      - Geometry: Single iso-surface of |Δp| at high percentile
      - Coloring: Phase arg(Δp) using cyclic twilight colormap
      - Lighting: Realistic ambient + diffuse + specular
      - Background: Dark (not white)
      - Opacity: High (0.6) for clear 3D geometry
      - Resolution: 2400×1400 or higher

    Parameters
    ----------
    standing_grid : pv.UnstructuredGrid
        Standing field (must have 'p_real', 'p_imag')
    combined_grid : pv.UnstructuredGrid
        Combined field (must have 'p_real', 'p_imag')
    out_path : str or Path
        Output PNG path
    roi_center : (3,) array, or None for auto-detect
    roi_size : float, ROI cube edge in metres
    percentile : int, iso-threshold percentile of |Δp|
    resolution : (w, h) in pixels
    title : str, optional text label
    """
    # Auto ROI
    if roi_center is None:
        bounds = combined_grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        roi_center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
    
    # Clip both grids to ROI
    standing_clip = clip_roi(standing_grid, roi_center, roi_size)
    combined_clip = clip_roi(combined_grid, roi_center, roi_size)
    
    # Compute complex perturbation: Δp = p_combined - p_standing
    p_stand = standing_clip.point_data['p_real'] + 1j * standing_clip.point_data['p_imag']
    p_comb  = combined_clip.point_data['p_real'] + 1j * combined_clip.point_data['p_imag']
    
    # Ensure same length (should be, but safety check)
    n = min(len(p_stand), len(p_comb))
    delta_p = p_comb[:n] - p_stand[:n]
    
    # Add perturbation fields to combined_clip
    combined_clip = combined_clip.extract_points(np.arange(n))
    combined_clip.point_data['delta_magnitude'] = np.abs(delta_p)
    combined_clip.point_data['delta_phase'] = np.angle(delta_p)
    
    # Extract iso-surface at high percentile of |Δp|
    mag_delta = combined_clip.point_data['delta_magnitude']
    threshold = np.percentile(mag_delta, percentile)
    
    print(f"  Δ|p| percentile {percentile}: {threshold:.1f} Pa")
    print(f"  Δ|p| range: [{np.min(mag_delta):.1f}, {np.max(mag_delta):.1f}] Pa")
    
    # Contour at threshold
    surface = combined_clip.contour(isosurfaces=[threshold],
                                     scalars='delta_magnitude')
    
    if surface.n_points == 0:
        print(f"Warning: No surface at percentile {percentile}")
        return
    
    print(f"  Iso-surface: {surface.n_points} points, {surface.n_cells} cells")
    
    # Sample phase onto surface
    surface_sampled = surface.sample(combined_clip)
    phase_surface = surface_sampled.point_data['delta_phase']
    
    # Create plotter with dark background
    pl = pv.Plotter(window_size=resolution, off_screen=True)
    pl.set_background([0.15, 0.15, 0.18])  # Dark blue-grey, not white
    
    # Remove default lights and add high-quality lighting
    pl.remove_all_lights()
    
    # Key light (dominant)
    key_light = pv.Light(
        position=(1.0, -0.8, 1.2),
        focal_point=roi_center.tolist(),
        intensity=1.0,
        light_type='scene light'
    )
    
    # Fill light (soften shadows)
    fill_light = pv.Light(
        position=(-0.8, 0.6, 0.8),
        focal_point=roi_center.tolist(),
        intensity=0.5,
        light_type='scene light'
    )
    
    # Back light (rim, separation from background)
    back_light = pv.Light(
        position=(0.0, 1.0, -0.8),
        focal_point=roi_center.tolist(),
        intensity=0.3,
        light_type='scene light'
    )
    
    pl.add_light(key_light)
    pl.add_light(fill_light)
    pl.add_light(back_light)
    
    # Add mesh with high opacity and phase coloring
    pl.add_mesh(
        surface_sampled,
        scalars='delta_phase',
        cmap='twilight',
        clim=[-np.pi, np.pi],
        opacity=0.65,  # High opacity for clear geometry
        smooth_shading=True,
        show_edges=False,
        specular=0.8,  # Shiny surface to catch light
        specular_power=25,  # Sharp highlights
    )
    
    # Set camera: fixed, intentional view of interaction
    # Position: looking at perturbation region from upper corner
    cam_dist = roi_size * 1.8
    cam_pos = roi_center + np.array([cam_dist, -cam_dist * 0.7, cam_dist * 0.9])
    pl.camera_position = [
        cam_pos.tolist(),
        roi_center.tolist(),
        [0, 0, 1]
    ]
    
    # Add text label if provided
    if title:
        pl.add_text(title, position='upper_left', font_size=18, color='white')
    
    # Add scalar bar for phase
    pl.add_scalar_bar(
        title='Phase of Δp\n(radians)',
        position_x=0.02, position_y=0.35,
        width=0.12, height=0.55,
        color='white',
    )
    
    # Render
    pl.screenshot(str(out_path), window_size=resolution)
    pl.close()
    
    print(f"  [Vortex Perturbation — Hero Render] {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  VIEW 0 — 2D MAGNITUDE SLICE (Spatial Grounding)
# ═══════════════════════════════════════════════════════════════════════

def view_2d_magnitude_slice(grid, out_path,
                             roi_center=None, roi_size=0.008,
                             resolution=(1200, 1000),
                             title='Acoustic Magnitude — Cross-section at z=center'):
    """
    View 0 — "2D magnitude slice (grounding view)"

    Simple 2D contour plot of |p| on a horizontal slice through ROI center.
    Shows dish boundary, magnitude scale, and spatial context for non-experts.

    Parameters
    ----------
    grid : pv.UnstructuredGrid  (must have 'magnitude')
    out_path : str or Path
    roi_center : (3,) array in metres, or None for auto
    roi_size : float, ROI cube edge in metres
    resolution : (w, h) pixels for matplotlib figure
    title : str, plot title
    """
    if roi_center is None:
        roi_center = np.mean(grid.bounds, axis=1)
    
    # Clip to ROI
    clipped = clip_roi(grid, roi_center, roi_size)
    
    # Extract horizontal slice at z = roi_center[2]
    z_slice = roi_center[2]
    tolerance = roi_size * 0.05
    
    # Create a horizontal plane and slice
    origin = [roi_center[0], roi_center[1], z_slice]
    normal = [0, 0, 1]  # Horizontal plane
    
    sliced = clipped.slice(normal=normal, origin=origin)
    
    if sliced.n_points == 0:
        print(f"Warning: No points in slice at z={z_slice}")
        return
    
    # Extract x, y, magnitude
    x = sliced.points[:, 0] * 1000  # Convert to mm
    y = sliced.points[:, 1] * 1000
    mag = sliced.point_data['magnitude']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    
    # Scatter plot
    scatter = ax.scatter(x, y, c=mag, cmap='viridis', s=20, alpha=0.8, edgecolors='none')
    
    # Dish outline (20mm × 20mm centered at origin in mm space)
    dish_center_mm = np.array(roi_center[:2]) * 1000
    dish_half = 10  # mm
    rect = plt.Rectangle(
        (dish_center_mm[0] - dish_half, dish_center_mm[1] - dish_half),
        2*dish_half, 2*dish_half,
        fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Dish boundary'
    )
    ax.add_patch(rect)
    
    # Labels & formatting
    ax.set_xlabel('x (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('y (mm)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='|p| (Pa)')
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Save
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  [View 0] {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _camera(roi_center, roi_size):
    """Consistent camera for all canonical views."""
    d = roi_size * 2.2
    pos   = roi_center + np.array([d, -d * 0.6, d * 0.8])
    focal = roi_center
    up    = (0, 0, 1)
    return [pos.tolist(), focal.tolist(), list(up)]


def _make_plotter(resolution=(1920, 1080)):
    """Create a clean plotter with white background and proper lighting."""
    pl = pv.Plotter(window_size=resolution, off_screen=True)
    pl.set_background('white')
    # Remove default lights, add key + fill for surface perception
    pl.remove_all_lights()
    key = pv.Light(position=(1, -0.5, 1), focal_point=(0, 0, 0),
                   intensity=0.9, light_type='scene light')
    fill = pv.Light(position=(-1, 1, 0.5), focal_point=(0, 0, 0),
                    intensity=0.3, light_type='scene light')
    pl.add_light(key)
    pl.add_light(fill)
    return pl


def _percentile_thresholds(data, percentiles):
    """Compute absolute values at given percentiles."""
    return [np.percentile(data, p) for p in percentiles]


# ═══════════════════════════════════════════════════════════════════════
#  VIEW 1 — TRAP GEOMETRY + VORTEX
# ═══════════════════════════════════════════════════════════════════════

def view_trap_geometry(grid, out_path,
                       roi_center=None, roi_size=0.008,
                       percentiles=(92, 96, 99),
                       opacities=(0.08, 0.06, 0.10),
                       resolution=(1920, 1080),
                       title=None):
    """
    View 1 — "Trap geometry + vortex"

    Shows 2-3 isosurfaces of |p| at high percentiles, colored by phase.
    The standing-wave lattice becomes visible as repeating shells;
    the vortex breaks the symmetry locally.

    Parameters
    ----------
    grid : pv.UnstructuredGrid  (must have 'magnitude' and 'phase')
    out_path : str or Path
    roi_center : (3,) array in metres, or None for auto
    roi_size : float, ROI cube edge in metres
    percentiles : tuple of percentiles for isosurfaces (e.g. 92, 96, 99)
    opacities : tuple of opacities per surface (outermost → innermost)
    resolution : (w, h) pixels
    title : optional text annotation
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Crop to ROI ---
    clipped = clip_roi(grid, roi_center, roi_size)
    mag = clipped.point_data['magnitude']

    if roi_center is None:
        b = grid.bounds
        roi_center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2])
    roi_center = np.asarray(roi_center, dtype=float)

    # --- Compute threshold values ---
    thresholds = _percentile_thresholds(mag, percentiles)

    # --- Create plotter ---
    pl = _make_plotter(resolution)

    # --- Add isosurfaces (from lowest to highest percentile) ---
    for iso_val, opa, pct in zip(thresholds, opacities, percentiles):
        try:
            surface = clipped.contour(isosurfaces=[iso_val], scalars='magnitude')
        except Exception:
            continue
        if surface.n_points == 0:
            continue

        # Transfer phase to surface by sampling from the clipped volume
        surface_sampled = surface.sample(clipped)

        pl.add_mesh(
            surface_sampled,
            scalars='phase',
            cmap='twilight',           # perceptually-uniform cyclic
            clim=[-np.pi, np.pi],
            opacity=opa,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=False,
        )

    # --- Add thin context slice (very faint) ---
    try:
        context_slice = clipped.slice(normal='z', origin=roi_center)
        pl.add_mesh(
            context_slice,
            scalars='magnitude',
            cmap='gray',
            opacity=0.10,
            show_edges=False,
            show_scalar_bar=False,
        )
    except Exception:
        pass

    # --- Phase scalar bar ---
    pl.add_scalar_bar(
        title='Phase (rad)',
        mapper=pl.mapper,
        n_labels=5,
        fmt='%.1f',
        title_font_size=14,
        label_font_size=12,
        color='black',
        position_x=0.82, position_y=0.15,
        width=0.12, height=0.7,
    )

    # --- Title annotation ---
    if title:
        pl.add_text(title, position='upper_left', font_size=14, color='black')

    # --- Camera ---
    pl.camera_position = _camera(roi_center, roi_size)

    # --- Save ---
    pl.screenshot(str(out_path))
    pl.close()
    print(f"  [View 1] {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
#  VIEW 2 — PARTICLE PLUCK
# ═══════════════════════════════════════════════════════════════════════

def view_particle_pluck(grid, out_path,
                        roi_center=None, roi_size=0.008,
                        n_particles=12, seed=42,
                        resolution=(1920, 1080),
                        title=None):
    """
    View 2 — "Particle pluck"

    Faint Gor'kov-potential slice underneath.  Particles (spheres)
    seeded at local minima of U with force arrows showing the
    radiation force direction.  No 3D volume.

    Parameters
    ----------
    grid : pv.UnstructuredGrid  (must have 'gorkov' and 'magnitude')
    out_path : str or Path
    roi_center, roi_size : ROI specification
    n_particles : number of particle markers to place
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = clip_roi(grid, roi_center, roi_size)

    if roi_center is None:
        b = grid.bounds
        roi_center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2])
    roi_center = np.asarray(roi_center, dtype=float)

    pl = _make_plotter(resolution)

    # --- Gor'kov slice (XY at z = center) ---
    has_gorkov = 'gorkov' in clipped.point_data
    if has_gorkov:
        g_slice = clipped.slice(normal='z', origin=roi_center)
        gorkov_vals = g_slice.point_data['gorkov']
        # Normalise for display
        g_min, g_max = gorkov_vals.min(), gorkov_vals.max()
        if g_max > g_min:
            g_slice.point_data['gorkov_norm'] = (gorkov_vals - g_min) / (g_max - g_min)

        pl.add_mesh(
            g_slice,
            scalars='gorkov',
            cmap='viridis',
            opacity=0.35,
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args=dict(
                title='Gor\'kov U (J)',
                color='black',
                title_font_size=12,
                label_font_size=10,
                position_x=0.82, position_y=0.15,
                width=0.12, height=0.35,
                fmt='%.1e',
            ),
        )

    # --- Also show faint magnitude slice for context ---
    mag_slice = clipped.slice(normal='z', origin=roi_center)
    pl.add_mesh(
        mag_slice,
        scalars='magnitude',
        cmap='gray',
        opacity=0.08,
        show_edges=False,
        show_scalar_bar=False,
    )

    # --- Seed particles at Gor'kov minima ---
    if has_gorkov:
        z_slice = clipped.slice(normal='z', origin=roi_center)
        pts   = z_slice.points
        g_val = z_slice.point_data['gorkov']

        # Find local minima: points where gorkov < all nearby points
        # Simple approach: take the n_particles lowest-gorkov points
        # with minimum spacing of roi_size/8
        min_spacing = roi_size / 8
        order = np.argsort(g_val)
        seeds = []
        for idx in order:
            pt = pts[idx]
            too_close = False
            for s in seeds:
                if np.linalg.norm(pt - s) < min_spacing:
                    too_close = True
                    break
            if not too_close:
                seeds.append(pt)
            if len(seeds) >= n_particles:
                break

        if seeds:
            seed_pts = np.array(seeds)
            # Particles: opaque red spheres
            particle_cloud = pv.PolyData(seed_pts)
            pl.add_mesh(
                particle_cloud,
                color='#E03030',
                point_size=14,
                render_points_as_spheres=True,
                opacity=1.0,
                show_scalar_bar=False,
            )

            # Compute crude force direction at each particle:
            # F ∝ -∇U, approximate by finite difference on the slice
            if len(seed_pts) > 0:
                # Use gradient of gorkov on the slice
                try:
                    g_grad = z_slice.compute_derivative(scalars='gorkov')
                    # Sample gradient at particle locations
                    grad_pts = pv.PolyData(seed_pts)
                    grad_sampled = grad_pts.sample(g_grad)
                    grad_key = [k for k in grad_sampled.point_data if 'gradient' in k.lower()]
                    if grad_key:
                        gvecs = grad_sampled.point_data[grad_key[0]]
                        # Force = -gradient
                        force = -gvecs
                        # Normalise for arrow length
                        norms = np.linalg.norm(force, axis=1, keepdims=True)
                        norms[norms == 0] = 1
                        force_norm = force / norms * (roi_size * 0.12)  # arrow length

                        arrows = pv.Arrow()  # glyph template
                        for i in range(len(seed_pts)):
                            start = seed_pts[i]
                            direction = force_norm[i]
                            arrow_mesh = pv.Arrow(
                                start=start,
                                direction=direction,
                                scale=np.linalg.norm(direction),
                                tip_length=0.3,
                                tip_radius=0.12,
                                shaft_radius=0.04,
                            )
                            pl.add_mesh(
                                arrow_mesh,
                                color='#202020',
                                opacity=0.85,
                                show_scalar_bar=False,
                            )
                except Exception:
                    pass  # gradient computation may fail on thin slices

    if title:
        pl.add_text(title, position='upper_left', font_size=14, color='black')

    # Camera — slightly higher for top-down view of slice
    cam_pos = roi_center + np.array([0, 0, roi_size * 2.5])
    pl.camera_position = [
        cam_pos.tolist(),
        roi_center.tolist(),
        [0, 1, 0],
    ]

    pl.screenshot(str(out_path))
    pl.close()
    print(f"  [View 2] {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
#  VIEW 4 — STANDING VS COMBINED DIFFERENCE
# ═══════════════════════════════════════════════════════════════════════

def view_difference(grid_standing, grid_combined, out_path,
                    roi_center=None, roi_size=0.008,
                    field='magnitude',
                    resolution=(1920, 1080),
                    title=None):
    """
    View 4 — "Standing vs combined difference"

    Single-slice visualisation of Δ|p| = |p_combined| - |p_standing|
    or ΔU.  High-contrast diverging colormap, clearly localised.

    Parameters
    ----------
    grid_standing, grid_combined : pv.UnstructuredGrid
    out_path : str or Path
    roi_center, roi_size : ROI spec
    field : 'magnitude' or 'gorkov'
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_s = clip_roi(grid_standing, roi_center, roi_size)
    clip_c = clip_roi(grid_combined, roi_center, roi_size)

    if roi_center is None:
        b = grid_combined.bounds
        roi_center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2])
    roi_center = np.asarray(roi_center, dtype=float)

    # --- Slices ---
    sl_s = clip_s.slice(normal='z', origin=roi_center)
    sl_c = clip_c.slice(normal='z', origin=roi_center)

    # --- Compute difference ---
    # Need to interpolate one onto the other since meshes are identical
    # but clipping may have produced slightly different triangulations.
    # Safe route: sample combined onto standing slice points.
    sl_c_on_s = sl_s.sample(sl_c)

    vals_s = sl_s.point_data[field]
    vals_c = sl_c_on_s.point_data[field]
    diff   = vals_c - vals_s

    sl_s.point_data['delta'] = diff

    # --- Symmetric colour limits ---
    vmax = np.percentile(np.abs(diff), 99)
    if vmax == 0:
        vmax = 1.0

    # --- Plot with matplotlib for cleaner 2D output ---
    pts = sl_s.points
    x, y = pts[:, 0] * 1e3, pts[:, 1] * 1e3   # mm

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.tripcolor(
        x, y, diff,
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
        shading='gouraud',
    )
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)', fontsize=13)
    ax.set_ylabel('y (mm)', fontsize=13)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    unit = 'Pa' if field == 'magnitude' else 'J'
    cbar.set_label(f'Δ{field} ({unit})', fontsize=13)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        label = '|p|' if field == 'magnitude' else 'U'
        ax.set_title(f'Δ{label}  =  {label}_combined − {label}_standing',
                     fontsize=14)

    # Annotate localisation
    max_idx = np.argmax(np.abs(diff))
    ax.annotate('vortex\nperturbation',
                xy=(x[max_idx], y[max_idx]),
                xytext=(x[max_idx] + 1.0, y[max_idx] + 1.0),
                fontsize=10, color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.8))

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [View 4] {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════
#  VIEW 3 — PHASE SWEEP STORYBOARD  (assembled by batch.py)
# ═══════════════════════════════════════════════════════════════════════

def view_trap_geometry_frame(grid, out_path,
                             roi_center, roi_size, camera_pos,
                             thresholds, opacities,
                             resolution=(960, 960),
                             label=None):
    """
    Render a single frame for the phase-sweep storyboard.

    Uses FIXED camera_pos, FIXED thresholds (for comparability).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = clip_roi(grid, roi_center, roi_size)

    pl = _make_plotter(resolution)

    for iso_val, opa in zip(thresholds, opacities):
        try:
            surface = clipped.contour(isosurfaces=[iso_val], scalars='magnitude')
        except Exception:
            continue
        if surface.n_points == 0:
            continue

        surface_sampled = surface.sample(clipped)
        pl.add_mesh(
            surface_sampled,
            scalars='phase',
            cmap='twilight',
            clim=[-np.pi, np.pi],
            opacity=opa,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=False,
        )

    if label:
        pl.add_text(label, position='upper_left', font_size=16, color='black')

    pl.camera_position = camera_pos
    pl.screenshot(str(out_path))
    pl.close()
