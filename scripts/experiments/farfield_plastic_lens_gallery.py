#!/usr/bin/env python3
"""
D3: Plastic lens gallery — generate phase/amplitude/thickness diagnostics
for each lens preset (A, B, C).

Usage:
    micromamba run -n acousto-complex python scripts/experiments/farfield_plastic_lens_gallery.py

Outputs: results/farfield_lens_gallery_<timestamp>/
"""
from __future__ import annotations

import sys, csv
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from acoustweezers.physics.acoustics.vortex_lens import (
    LENS_PRESETS, export_lens_maps,
    compute_plastic_lens_phase, compute_plastic_lens_amplitude,
    compute_plastic_lens_thickness,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path("results") / f"farfield_lens_gallery_{stamp}"
OUT.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*70}")
print("D3: PLASTIC LENS GALLERY")
print(f"{'='*70}\n")

summary_rows = []

for name, factory in LENS_PRESETS.items():
    lens_cfg = factory()
    lens_dir = OUT / f"lens_{name}"
    lens_dir.mkdir(exist_ok=True)

    print(f"  Preset {name}: l={lens_cfg.topological_charge}  "
          f"f={lens_cfg.focal_length*1e3:.1f}mm  "
          f"offset=({lens_cfg.focus_offset_x*1e3:.2f},{lens_cfg.focus_offset_y*1e3:.2f})mm")

    # Export NPY/CSV
    info = export_lens_maps(lens_cfg, lens_dir, N=200)

    # Generate plots
    R = lens_cfg.aperture_radius
    N = 200
    xg = np.linspace(-1.2 * R, 1.2 * R, N)
    yg = np.linspace(-1.2 * R, 1.2 * R, N)
    XX, YY = np.meshgrid(xg, yg)
    xf, yf = XX.ravel(), YY.ravel()

    phi_tgt, phi_pl = compute_plastic_lens_phase(xf, yf, lens_cfg)
    amp = compute_plastic_lens_amplitude(xf, yf, lens_cfg)
    thickness = compute_plastic_lens_thickness(xf, yf, lens_cfg)

    phi_2d = phi_pl.reshape(N, N)
    amp_2d = amp.reshape(N, N)
    thick_2d = thickness.reshape(N, N)
    pattern = amp * np.exp(1j * phi_pl)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    extent = [xg[0]*1e3, xg[-1]*1e3, yg[0]*1e3, yg[-1]*1e3]

    im0 = axes[0, 0].imshow(amp_2d, extent=extent, origin="lower", cmap="viridis")
    axes[0, 0].set_title("Amplitude A(r)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(phi_2d, extent=extent, origin="lower", cmap="twilight")
    axes[0, 1].set_title("φ_plastic [rad]")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(thick_2d * 1e3, extent=extent, origin="lower", cmap="copper")
    axes[0, 2].set_title("Thickness [mm]")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(np.real(pattern).reshape(N, N), extent=extent,
                              origin="lower", cmap="RdBu_r")
    axes[1, 0].set_title("Re(v_n)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(np.imag(pattern).reshape(N, N), extent=extent,
                              origin="lower", cmap="RdBu_r")
    axes[1, 1].set_title("Im(v_n)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(phi_tgt.reshape(N, N), extent=extent,
                              origin="lower", cmap="twilight")
    axes[1, 2].set_title("φ_target (unwrapped)")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        circle = plt.Circle((0, 0), R * 1e3, fill=False, ec="cyan", lw=0.8, ls="--")
        ax.add_patch(circle)

    fig.suptitle(f"Lens Preset {name}: l={lens_cfg.topological_charge}, "
                 f"f={lens_cfg.focal_length*1e3:.0f}mm, "
                 f"offset=({lens_cfg.focus_offset_x*1e3:.2f},{lens_cfg.focus_offset_y*1e3:.2f})mm",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(lens_dir / f"lens_{name}_gallery.png", dpi=150)
    plt.close(fig)

    # Individual plots
    for pname, data, cmap in [
        ("disk_amplitude", amp_2d, "viridis"),
        ("disk_phase", phi_2d, "twilight"),
        ("disk_real", np.real(pattern).reshape(N, N), "RdBu_r"),
        ("disk_imag", np.imag(pattern).reshape(N, N), "RdBu_r"),
        ("thickness_map", thick_2d * 1e3, "copper"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im = ax2.imshow(data, extent=extent, origin="lower", cmap=cmap)
        ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
        ax2.set_title(pname.replace("_", " ").title())
        ax2.set_aspect("equal")
        circle = plt.Circle((0, 0), R * 1e3, fill=False, ec="cyan", lw=0.8, ls="--")
        ax2.add_patch(circle)
        plt.colorbar(im, ax=ax2)
        fig2.tight_layout()
        fig2.savefig(lens_dir / f"{pname}.png", dpi=150)
        plt.close(fig2)

    summary_rows.append({
        "preset": name,
        "l": lens_cfg.topological_charge,
        "f_mm": f"{lens_cfg.focal_length*1e3:.1f}",
        "xf_mm": f"{lens_cfg.focus_offset_x*1e3:.2f}",
        "yf_mm": f"{lens_cfg.focus_offset_y*1e3:.2f}",
        "thickness_min_mm": f"{info['thickness_min_mm']:.3f}",
        "thickness_max_mm": f"{info['thickness_max_mm']:.3f}",
        "dk_rad_m": f"{info['dk_rad_m']:.1f}",
    })
    print(f"    thickness: [{info['thickness_min_mm']:.3f}, {info['thickness_max_mm']:.3f}] mm")

# ── Summary CSV ──
with open(OUT / "gallery_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader()
    w.writerows(summary_rows)

# ── Symlink ──
latest = Path("results") / "farfield_lens_gallery_latest"
if latest.is_symlink() or latest.exists():
    latest.unlink()
latest.symlink_to(OUT.name)

print(f"\n  Output: {OUT}")
print(f"  Files per preset: lens_<X>_gallery.png, disk_amplitude.png, disk_phase.png,")
print(f"    disk_real.png, disk_imag.png, thickness_map.png, *.npy, *.csv")
