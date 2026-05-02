#!/usr/bin/env python3
"""
Export the final 3-D printable lens STL for the bridge IASA reconstruction.

The IASA hologram is regenerated at 100 iterations (with the corrected
non-zero-percentile normalisation) so the saved STL corresponds exactly to
the assessment in bridge_iasa_assessment.py.  The thickness map is derived
from the phase-only hologram:

    t(x,y) = h_base + h_max * φ(x,y) / 2π       (inside aperture)
    t(x,y) = 0                                    (outside aperture)

where h_max = 2π / |k_lens − k_water|.

Outputs (results/bridge_iasa_assessment/):
    bridge_iasa_100iter_lens.stl        — printable solid mesh (mm units)
    bridge_iasa_100iter_thickness.png   — top-view thickness heat-map
    bridge_iasa_100iter_stl_render.png  — 3-D pyvista oblique render

Run:
    python scripts/dev/bridge_iasa_export_stl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    make_grid,
    propagate_asm,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_IASA_DIR = (
    PROJECT_ROOT / "results" / "dev"
    / "inverse_bridge_pressure_lens_replica_scaled2x_20260324_095106"
)
BRIDGE_IASA_NPZ = _IASA_DIR / "bridge_inverse_replica_fields.npz"
OUT_DIR         = PROJECT_ROOT / "results" / "bridge_iasa_assessment"

# ── Physics ───────────────────────────────────────────────────────────────────
FREQUENCY_HZ       = 2_150_000.0
C_WATER            = 1480.0
C_LENS             = 2636.0
N_GRID             = 400
TRANSDUCER_DIAM_MM = 20.0
FOCAL_MM           = 13.21309776965029
H_BASE_MM          = 1.0
SOURCE_PRESSURE_PA = 0.05e6

OUTSIDE_SUP = 0.0
N_ITER      = 100           # final hologram iteration

CMAP_THICK  = "viridis"
DPI         = 190


# ── Config ────────────────────────────────────────────────────────────────────
def _build_cfg() -> ReplicaConfig:
    return ReplicaConfig(
        frequency_hz=FREQUENCY_HZ,
        c_water=C_WATER,
        c_lens=C_LENS,
        transducer_diameter_mm=TRANSDUCER_DIAM_MM,
        focal_distance_mm=FOCAL_MM,
        n_grid=N_GRID,
        h_base_mm=H_BASE_MM,
        n_iter=N_ITER,
        source_pressure_pa=SOURCE_PRESSURE_PA,
    )


# ── Normalisation (identical to bridge_iasa_assessment.py) ────────────────────
def _normalise_target(raw_amp: np.ndarray, clip_pct: float = 99.5, gamma: float = 0.9) -> np.ndarray:
    a = np.maximum(raw_amp, 0.0)
    nz = a[a > 1e-12]
    if nz.size == 0:
        return np.zeros_like(a)
    a_floor = float(np.percentile(nz, 1.0))
    a = np.maximum(a - a_floor, 0.0)
    nz2 = a[a > 1e-12]
    a_clip = float(np.percentile(nz2, clip_pct)) if nz2.size > 0 else 0.0
    if a_clip <= 0.0:
        return np.zeros_like(a)
    a = np.clip(a / a_clip, 0.0, 1.0)
    return np.power(a, max(gamma, 1e-6))


# ── IASA ──────────────────────────────────────────────────────────────────────
def _run_iasa(cfg: ReplicaConfig,
              aperture_mask: np.ndarray,
              target_amp: np.ndarray,
              roi_mask: np.ndarray,
              dx: float) -> np.ndarray:
    """Run N_ITER Gerchberg-Saxton iterations and return the final lens field."""
    z   = cfg.focal_distance_m
    outside_roi = ~roi_mask

    rng  = np.random.default_rng(seed=42)
    lens = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, (N_GRID, N_GRID)))
    lens[~aperture_mask] = 0.0

    for it in range(1, N_ITER + 1):
        img = propagate_asm(lens, cfg.k_water, z, dx)
        upd = img.copy()
        upd[roi_mask] = target_amp[roi_mask] * np.exp(1j * np.angle(img[roi_mask]))
        if OUTSIDE_SUP < 1.0:
            upd[outside_roi] = OUTSIDE_SUP * np.abs(img[outside_roi]) * np.exp(
                1j * np.angle(img[outside_roi])
            )
        lens = np.exp(1j * np.angle(propagate_asm(upd, cfg.k_water, -z, dx)))
        lens[~aperture_mask] = 0.0
        if it % 10 == 0:
            print(f"  iter {it:3d}/{N_ITER}")

    return lens


# ── Thickness map ─────────────────────────────────────────────────────────────
def _thickness_mm(lens_field: np.ndarray,
                  cfg: ReplicaConfig,
                  aperture_mask: np.ndarray) -> np.ndarray:
    """Convert phase-only hologram to physical lens thickness in mm."""
    phi = np.mod(np.angle(lens_field), 2.0 * np.pi)
    t   = cfg.h_base_m + cfg.h_max_m * (phi / (2.0 * np.pi))
    t[~aperture_mask] = 0.0
    return t * 1e3  # → mm


# ── Thickness heat-map ────────────────────────────────────────────────────────
def _save_thickness_png(thickness_mm: np.ndarray,
                         iasa_x: np.ndarray,
                         iasa_y: np.ndarray,
                         cfg: ReplicaConfig,
                         out_path: Path) -> None:
    r_ap = TRANSDUCER_DIAM_MM / 2.0
    theta = np.linspace(0, 2 * np.pi, 360)
    ext   = [iasa_x[0] * 1e3, iasa_x[-1] * 1e3,
             iasa_y[0] * 1e3, iasa_y[-1] * 1e3]

    t_plot = thickness_mm.copy()
    t_plot[t_plot == 0] = np.nan  # collapse outside to NaN for better display

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        t_plot, origin="lower", extent=ext,
        cmap=CMAP_THICK, aspect="equal",
        vmin=float(np.nanmin(t_plot)), vmax=float(np.nanmax(t_plot)),
    )
    plt.colorbar(im, ax=ax, label="Thickness [mm]", fraction=0.046, pad=0.02)
    ax.plot(r_ap * np.cos(theta), r_ap * np.sin(theta), "r--", lw=0.8, alpha=0.7)
    ax.set_title(
        f"Bridge IASA holographic lens — iter {N_ITER}\n"
        f"h_base={H_BASE_MM:.1f} mm,  h_max={cfg.h_max_m*1e3:.3f} mm",
        fontweight="bold",
    )
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── STL writer (pure numpy, no 3rd-party mesh lib required) ──────────────────
def _write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    """
    Write a binary STL file.

    triangles : (N, 3, 3) float64 array — each row is (v0, v1, v2) in mm.
    """
    import struct

    n = len(triangles)
    tri = triangles.astype(np.float32)

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normals = np.cross(e1, e2)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(norms > 1e-20, norms, 1.0)

    # 50-byte record per triangle
    rec = np.zeros(n, dtype=[
        ("n",  np.float32, (3,)),
        ("v0", np.float32, (3,)),
        ("v1", np.float32, (3,)),
        ("v2", np.float32, (3,)),
        ("ab", np.uint16),
    ])
    rec["n"]  = normals
    rec["v0"] = tri[:, 0]
    rec["v1"] = tri[:, 1]
    rec["v2"] = tri[:, 2]
    rec["ab"] = 0

    with open(path, "wb") as f:
        header = b"Bridge IASA holographic lens STL" + b" " * 48
        f.write(header)
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())


def _heightmap_to_triangles(xg: np.ndarray, yg: np.ndarray,
                             t_top: np.ndarray) -> np.ndarray:
    """
    Build a watertight solid mesh from a 2-D height map.

    Returns (N_tri, 3, 3) float64 array of triangles (in mm units).

    The solid consists of:
      • Top surface  — height = t_top(x,y)
      • Bottom plate — height = 0 everywhere
      • 4 perimeter walls connecting top to bottom at the grid edges

    Cells where both top and bottom are at z=0 are automatically degenerate
    (zero area) and are excluded.
    """
    Ny, Nx = t_top.shape
    assert len(yg) == Ny and len(xg) == Nx

    # ── Build vertex arrays ───────────────────────────────────────────────
    # Top vertices  shape (Ny, Nx, 3)
    XX, YY = np.meshgrid(xg, yg)   # row=y, col=x
    Vtop = np.stack([XX, YY, t_top], axis=2)         # (Ny, Nx, 3)
    Vbot = np.stack([XX, YY, np.zeros_like(t_top)], axis=2)

    tris = []

    # ── Top and bottom surface quads ─────────────────────────────────────
    # Quad (iy, ix): corners at (iy,ix),(iy+1,ix),(iy,ix+1),(iy+1,ix+1)
    iy = np.arange(Ny - 1)
    ix = np.arange(Nx - 1)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")   # (Ny-1, Nx-1)

    def _quad_tris(V, outward_up: bool):
        """Triangulate all (Ny-1)×(Nx-1) quads for a surface array V (Ny,Nx,3).
        outward_up=True  → top face (normal pointing +z)
        outward_up=False → bottom face (normal pointing -z)
        """
        a = V[IY,     IX    ]   # (Ny-1, Nx-1, 3)
        b = V[IY + 1, IX    ]
        c = V[IY,     IX + 1]
        d = V[IY + 1, IX + 1]

        if outward_up:
            t1 = np.stack([a, c, b], axis=2)   # CCW from above
            t2 = np.stack([c, d, b], axis=2)
        else:
            t1 = np.stack([a, b, c], axis=2)   # CW from above = CCW from below
            t2 = np.stack([c, b, d], axis=2)

        # t1/t2 shape: (Ny-1, Nx-1, 3_verts, 3_xyz) ← need transpose
        t1 = t1.transpose(0, 1, 3, 2)   # -> (Ny-1, Nx-1, 3_xyz, 3_verts) No…

        # build as (Ny-1, Nx-1, 3_verts, 3_xyz)
        if outward_up:
            t1 = np.stack([a, c, b], axis=3)   # (Ny-1, Nx-1, 3, 3) wrong shape
        # redo cleanly
        # shape: (Ny-1*Nx-1, 3, 3)
        flat_a = a.reshape(-1, 3)
        flat_b = b.reshape(-1, 3)
        flat_c = c.reshape(-1, 3)
        flat_d = d.reshape(-1, 3)

        if outward_up:
            t1_ = np.stack([flat_a, flat_c, flat_b], axis=1)
            t2_ = np.stack([flat_c, flat_d, flat_b], axis=1)
        else:
            t1_ = np.stack([flat_a, flat_b, flat_c], axis=1)
            t2_ = np.stack([flat_c, flat_b, flat_d], axis=1)

        quads = np.concatenate([t1_, t2_], axis=0)  # (2*(Ny-1)*(Nx-1), 3, 3)
        # Remove degenerate triangles (zero area)
        e1 = quads[:, 1] - quads[:, 0]
        e2 = quads[:, 2] - quads[:, 0]
        area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        return quads[area2 > 1e-20]

    tris.append(_quad_tris(Vtop, outward_up=True))
    tris.append(_quad_tris(Vbot, outward_up=False))

    # ── Perimeter walls (4 edges of the rectangular grid) ────────────────
    # For each edge segment, build a quad from (top_a, top_b, bot_a, bot_b)
    # Edge - bottom (iy=0): ix from 0..Nx-2, wall outward normal is -y
    def _wall_strip(top_a, top_b, bot_a, bot_b):
        """Build wall quads between two edges. All arrays shape (M, 3)."""
        t1 = np.stack([top_a, bot_a, top_b], axis=1)
        t2 = np.stack([top_b, bot_a, bot_b], axis=1)
        quads = np.concatenate([t1, t2], axis=0)
        e1 = quads[:, 1] - quads[:, 0]
        e2 = quads[:, 2] - quads[:, 0]
        area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        return quads[area2 > 1e-20]

    # bottom edge (iy=0, ix=0..Nx-2)
    tris.append(_wall_strip(Vtop[0, :-1], Vtop[0, 1:],
                             Vbot[0, :-1], Vbot[0, 1:]))
    # top edge (iy=Ny-1, ix=0..Nx-2) — outward is +y so reverse winding
    tris.append(_wall_strip(Vtop[-1, 1:], Vtop[-1, :-1],
                             Vbot[-1, 1:], Vbot[-1, :-1]))
    # left edge (ix=0, iy=0..Ny-2) — outward is -x so reverse
    tris.append(_wall_strip(Vtop[1:, 0], Vtop[:-1, 0],
                             Vbot[1:, 0], Vbot[:-1, 0]))
    # right edge (ix=Nx-1, iy=0..Ny-2) — outward is +x
    tris.append(_wall_strip(Vtop[:-1, -1], Vtop[1:, -1],
                             Vbot[:-1, -1], Vbot[1:, -1]))

    return np.concatenate(tris, axis=0)  # (N_tri, 3, 3)


# ── 3-D STL build & render ────────────────────────────────────────────────────
def _build_and_save_stl(thickness_mm: np.ndarray,
                         iasa_x: np.ndarray,
                         iasa_y: np.ndarray,
                         cfg: ReplicaConfig,
                         stl_path: Path,
                         render_path: Path,
                         stride: int = 2) -> None:
    """
    Build a watertight printable lens STL using a pure-numpy binary writer.

    stride=1 → full 400×400 grid, 0.05 mm pitch (very fine)
    stride=2 → 200×200 sub-sampled, 0.10 mm pitch (fast, adequate for SLA/FDM)
    """
    # Sub-sample
    idx_s = np.arange(0, N_GRID, stride, dtype=int)
    if idx_s[-1] != N_GRID - 1:
        idx_s = np.append(idx_s, N_GRID - 1)

    xs_mm  = iasa_x[idx_s] * 1e3   # mm
    ys_mm  = iasa_y[idx_s] * 1e3   # mm
    t_s    = thickness_mm[np.ix_(idx_s, idx_s)].copy()

    # Zero out residual corners outside circular aperture
    xg_s, yg_s = np.meshgrid(xs_mm, ys_mm)
    r_ap = 0.499 * TRANSDUCER_DIAM_MM
    t_s[(xg_s ** 2 + yg_s ** 2) > r_ap ** 2] = 0.0

    print(f"  Mesh: {len(ys_mm)}×{len(xs_mm)}, "
          f"pitch={xs_mm[1]-xs_mm[0]:.3f} mm, "
          f"thickness {t_s[t_s>0].min():.4f}–{t_s.max():.4f} mm")

    print("  Building triangles …")
    triangles = _heightmap_to_triangles(xs_mm, ys_mm, t_s)
    print(f"  Triangles built: {len(triangles):,}")

    _write_binary_stl(stl_path, triangles)
    size_kb = stl_path.stat().st_size // 1024
    print(f"  Saved STL: {stl_path.name}  ({len(triangles):,} triangles, {size_kb} KB)")

    # ── 3-D render (pyvista PolyData — much lighter than StructuredGrid) ──
    try:
        import pyvista as pv

        # Load the just-written STL back for rendering
        mesh = pv.read(str(stl_path))
        surf_all = mesh.extract_surface()
        normals  = surf_all.compute_normals(cell_normals=True, point_normals=False)
        top_mask = normals["Normals"][:, 2] > 0.5
        top_layer = surf_all.extract_cells(np.where(top_mask)[0]) if np.any(top_mask) else surf_all

        r_cam  = TRANSDUCER_DIAM_MM * 2.5
        el, az = 35.0, 45.0
        cam_x  = r_cam * np.cos(np.radians(az)) * np.cos(np.radians(el))
        cam_y  = r_cam * np.sin(np.radians(az)) * np.cos(np.radians(el))
        cam_z  = r_cam * np.sin(np.radians(el))
        focal_z = 0.5 * float(cfg.h_base_m * 1e3 + 0.5 * cfg.h_max_m * 1e3)

        try:
            pv.start_xvfb()
        except Exception:
            pass

        pl = pv.Plotter(off_screen=True, window_size=(1800, 1200))
        pl.set_background("white")
        pl.add_mesh(
            top_layer,
            scalars=top_layer.points[:, 2],
            cmap=CMAP_THICK,
            show_scalar_bar=False, smooth_shading=True,
            show_edges=False, lighting=True,
        )
        pl.add_scalar_bar(
            title="Thickness [mm]", n_labels=4, fmt="%.2f",
            position_x=0.85, position_y=0.05,
            width=0.12, height=0.60,
            label_font_size=22, title_font_size=18,
        )
        theta_r  = np.linspace(0, 2 * np.pi, 360)
        ring_pts = np.column_stack([
            r_ap * np.cos(theta_r), r_ap * np.sin(theta_r), np.zeros(360)])
        ring = pv.Spline(ring_pts, 360).tube(radius=r_ap * 0.005)
        pl.add_mesh(ring, color="black", opacity=0.6)
        pl.camera_position = [(cam_x, cam_y, cam_z), (0.0, 0.0, focal_z), (0.0, 0.0, 1.0)]
        pl.add_title(f"Bridge IASA holographic lens — {N_ITER} iterations",
                     font_size=14, color="black")
        pl.screenshot(str(render_path), return_img=True)
        pl.close()
        print(f"  Saved render: {render_path.name}")

    except Exception as exc:
        print(f"  [render skipped: {exc}]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BRIDGE IASA — STL EXPORT")
    print("=" * 70)

    cfg = _build_cfg()
    iasa_x, iasa_y, _, _, _, _, _, dx = make_grid(cfg)

    # ── Load IASA data ────────────────────────────────────────────────────
    print("\n[1] Loading IASA data …")
    ir = np.load(BRIDGE_IASA_NPZ)
    target_raw_amp = ir["target_raw_amp"].astype(float)
    roi_mask       = ir["roi_mask"].astype(bool)
    aperture_mask  = ir["aperture_mask"].astype(bool)
    target_amp     = _normalise_target(target_raw_amp) * aperture_mask.astype(float)

    print(f"  h_base  = {cfg.h_base_m*1e3:.3f} mm")
    print(f"  h_max   = {cfg.h_max_m*1e3:.3f} mm")
    print(f"  h_total = {(cfg.h_base_m + cfg.h_max_m)*1e3:.3f} mm")
    print(f"  aperture diameter = {TRANSDUCER_DIAM_MM:.1f} mm")
    print(f"  grid pitch dx = {dx*1e6:.1f} µm  ({N_GRID}×{N_GRID})")

    # ── Run IASA (or load cached result) ─────────────────────────────────
    _cache = OUT_DIR / f"_lens_cache_{N_ITER}iter.npz"
    if _cache.exists():
        print(f"\n[2] Loading cached lens field from {_cache.name} …")
        lens_final = np.load(_cache)["lens_field"]
    else:
        print(f"\n[2] Running IASA ({N_ITER} iterations) …")
        lens_final = _run_iasa(cfg, aperture_mask, target_amp, roi_mask, dx)
        np.savez_compressed(_cache, lens_field=lens_final)
        print(f"  Cached to {_cache.name} (re-run skips IASA next time)")

    # ── Thickness map ─────────────────────────────────────────────────────
    print("\n[3] Computing thickness map …")
    thick_mm = _thickness_mm(lens_final, cfg, aperture_mask)
    nz = thick_mm[thick_mm > 0]
    print(f"  Inside-aperture thickness range: {nz.min():.4f} – {nz.max():.4f} mm")
    print(f"  Pixel pitch at 1× (stride=1): {dx*1e3:.4f} mm/px")

    # ── Thickness PNG ─────────────────────────────────────────────────────
    print("\n[4] Saving thickness heat-map …")
    _save_thickness_png(
        thick_mm, iasa_x, iasa_y, cfg,
        OUT_DIR / f"bridge_iasa_{N_ITER}iter_thickness.png",
    )

    # ── STL ───────────────────────────────────────────────────────────────
    # stride=2 → 200×200 sub-sampled (0.10 mm pitch, well within SLA/FDM
    # printer resolution; avoids expensive boolean_intersection on large grid)
    print("\n[5] Building and saving STL (stride=2, 200×200, 0.10 mm pitch) …")
    _build_and_save_stl(
        thick_mm, iasa_x, iasa_y, cfg,
        stl_path   = OUT_DIR / f"bridge_iasa_{N_ITER}iter_lens.stl",
        render_path= OUT_DIR / f"bridge_iasa_{N_ITER}iter_stl_render.png",
        stride=2,
    )

    print("\n" + "=" * 70)
    print(f"Done.  All files written to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
