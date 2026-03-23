#!/usr/bin/env python3
"""
Run the replica IASA lens design pipeline using the produced bridge pressure field
as the target amplitude map.

This reuses the core iterative solver from:
  scripts/dev/inverse_c_shape_lens_replica.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.inverse_c_shape_lens_replica import (
    ReplicaConfig,
    export_stl,
    make_grid,
    propagate_asm,
    run_iasa,
)


BRIDGE_FIELDS_NPZ = (
    PROJECT_ROOT / "results" / "dev" / "bridge_pressure_field_standalone_scaled" / "bridge_pressure_fields.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run replica IASA on bridge pressure field target")
    parser.add_argument("--input-npz", type=Path, default=BRIDGE_FIELDS_NPZ)
    parser.add_argument("--field-key", type=str, default="p_bridge_effective_full")
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--focal-distance-mm", type=float, default=40.0)
    parser.add_argument("--frequency-hz", type=float, default=2.44e6)
    parser.add_argument("--c-water", type=float, default=1480.0)
    parser.add_argument("--c-lens", type=float, default=2636.0)
    parser.add_argument("--h-base-mm", type=float, default=1.0)
    parser.add_argument("--source-pressure-pa", type=float, default=0.05e6)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument(
        "--stl-grid-stride",
        type=int,
        default=4,
        help="Downsample factor for STL export mesh (>=1). Higher is faster/coarser.",
    )
    parser.add_argument("--skip-stl", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _normalise_target(raw_amp: np.ndarray, clip_pct: float, gamma: float) -> np.ndarray:
    a = np.asarray(raw_amp, dtype=float)
    a = np.maximum(a, 0.0)
    a_floor = float(np.percentile(a, 1.0))
    a = np.maximum(a - a_floor, 0.0)
    a_clip = float(np.percentile(a, clip_pct))
    if a_clip <= 0.0:
        return np.zeros_like(a)
    a = np.clip(a / a_clip, 0.0, 1.0)
    g = max(float(gamma), 1.0e-6)
    return np.power(a, g)


def _save_map(data: np.ndarray, title: str, cbar: str, cmap: str, out: Path, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(data, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=cbar)
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    def _log(msg: str) -> None:
        print(msg, flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (PROJECT_ROOT / "results" / "dev" / f"inverse_bridge_pressure_lens_replica_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[start] output_dir={out_dir}")

    d = np.load(args.input_npz)
    _log(f"[load] input_npz={args.input_npz}")
    if args.field_key not in d.files:
        raise KeyError(f"Field key '{args.field_key}' not found in {args.input_npz}. Available: {list(d.files)}")

    p_target_complex = d[args.field_key].astype(complex)
    target_raw_amp = np.abs(p_target_complex)

    n = int(target_raw_amp.shape[0])
    if target_raw_amp.ndim != 2 or target_raw_amp.shape[0] != target_raw_amp.shape[1]:
        raise RuntimeError("Target field must be square 2D")

    x_full = d["x_full"].astype(float)
    diameter_m = float(x_full[-1] - x_full[0])

    cfg = ReplicaConfig(
        frequency_hz=float(args.frequency_hz),
        c_water=float(args.c_water),
        c_lens=float(args.c_lens),
        transducer_diameter_mm=float(diameter_m * 1e3),
        focal_distance_mm=float(args.focal_distance_mm),
        n_grid=int(n),
        h_base_mm=float(args.h_base_mm),
        n_iter=int(args.n_iter),
        source_pressure_pa=float(args.source_pressure_pa),
        output_dir=str(out_dir),
    )
    _log(
        f"[config] n_grid={cfg.n_grid}, n_iter={cfg.n_iter}, "
        f"focal_distance_mm={cfg.focal_distance_mm:.2f}, skip_stl={args.skip_stl}, "
        f"stl_grid_stride={max(int(args.stl_grid_stride), 1)}"
    )

    timings: dict[str, float] = {}
    t0_all = time.perf_counter()

    t0 = time.perf_counter()
    _, _, xg, yg, _, _, aperture_mask, dx = make_grid(cfg)
    timings["grid_creation"] = time.perf_counter() - t0
    _log(f"[stage] grid_creation done in {timings['grid_creation']:.3f}s")

    t0 = time.perf_counter()
    target_amp = _normalise_target(target_raw_amp, clip_pct=float(args.clip_percentile), gamma=float(args.gamma))
    target_amp = target_amp * aperture_mask.astype(float)
    timings["target_normalisation"] = time.perf_counter() - t0
    _log(f"[stage] target_normalisation done in {timings['target_normalisation']:.3f}s")

    t0 = time.perf_counter()
    _log("[stage] run_iasa starting...")
    lens_field = run_iasa(cfg, aperture_mask, target_amp, dx)
    timings["iasa_total"] = time.perf_counter() - t0
    timings["iasa_per_iteration"] = timings["iasa_total"] / max(cfg.n_iter, 1)
    _log(
        f"[stage] run_iasa done in {timings['iasa_total']:.3f}s "
        f"({timings['iasa_per_iteration']:.4f}s/iter)"
    )

    t0 = time.perf_counter()
    _log("[stage] reconstruction starting...")
    p_recon = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)
    scale = cfg.source_pressure_pa / (np.sqrt(np.mean(np.abs(lens_field[aperture_mask]) ** 2)) + 1e-12)
    recon_amp = np.abs(p_recon) * scale
    recon_amp_n = recon_amp / (np.percentile(recon_amp[aperture_mask], 99.5) + 1e-12)
    recon_amp_n = np.clip(recon_amp_n, 0.0, 1.0)
    timings["reconstruction"] = time.perf_counter() - t0
    _log(f"[stage] reconstruction done in {timings['reconstruction']:.3f}s")

    # Thickness mapping (same logic as replica script)
    phi_wrapped = np.mod(np.angle(lens_field), 2.0 * np.pi)
    thickness = cfg.h_base_m + cfg.h_max_m * (phi_wrapped / (2.0 * np.pi))
    thickness[~aperture_mask] = cfg.h_base_m

    # Correlation metric inside aperture
    ta = target_amp[aperture_mask].ravel()
    ra = recon_amp_n[aperture_mask].ravel()
    if np.std(ta) > 0 and np.std(ra) > 0:
        corr = float(np.corrcoef(ta, ra)[0, 1])
    else:
        corr = float("nan")

    _log("[stage] saving figures...")
    _save_map(target_raw_amp, f"Raw bridge target amplitude ({args.field_key})", "|p| [Pa]", "inferno", out_dir / "target_raw_bridge_amplitude.png")
    _save_map(target_amp, "Normalised target amplitude used by IASA", "a.u.", "inferno", out_dir / "target_amplitude_for_iasa.png", vmin=0.0, vmax=1.0)
    _save_map(phi_wrapped, "Lens wrapped phase", "phase [rad]", "twilight", out_dir / "lens_phase_wrapped.png", vmin=0.0, vmax=2.0 * np.pi)
    _save_map(recon_amp * 1e-3, "Reconstructed pressure amplitude at focus", "|p| [kPa]", "inferno", out_dir / "reconstructed_pressure_at_focus.png")
    _save_map(thickness * 1e3, "Lens thickness map", "thickness [mm]", "viridis", out_dir / "lens_thickness_map.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    im0 = axes[0].imshow(target_amp, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0].set_title("Target (normalised)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    im1 = axes[1].imshow(recon_amp_n, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title("Reconstruction (normalised)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    fig.suptitle(f"Target vs Reconstruction inside aperture (corr={corr:.4f})")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "target_vs_reconstruction.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    _log("[stage] figures saved")

    stl_status = "skipped"
    stl_error = None
    if not args.skip_stl:
        try:
            _log("[stage] STL export starting...")
            _, _, open_edges = export_stl(
                cfg=cfg,
                lens_field=lens_field,
                aperture_mask=aperture_mask,
                xg=xg,
                yg=yg,
                save_path=out_dir / "bridge_inverse_hologram_lens.stl",
                timings=timings,
                grid_stride=max(int(args.stl_grid_stride), 1),
            )
            stl_status = f"ok (open_edges={open_edges})"
            _log(f"[stage] STL export done: {stl_status}")
        except Exception as exc:
            stl_status = "failed"
            stl_error = str(exc)
            _log(f"[stage] STL export failed: {stl_error}")
    else:
        _log("[stage] STL export skipped (--skip-stl)")

    np.savez_compressed(
        out_dir / "bridge_inverse_replica_fields.npz",
        target_raw_amp=target_raw_amp,
        target_amp=target_amp,
        lens_field=lens_field,
        lens_phase_wrapped=phi_wrapped,
        thickness=thickness,
        recon_amp=recon_amp,
        recon_amp_n=recon_amp_n,
        aperture_mask=aperture_mask,
    )

    timings["total"] = time.perf_counter() - t0_all
    manifest = {
        "script": "scripts/dev/run_inverse_replica_on_bridge_pressure_field.py",
        "input_npz": str(args.input_npz),
        "field_key": args.field_key,
        "config": {
            "n_grid": cfg.n_grid,
            "transducer_diameter_mm": cfg.transducer_diameter_mm,
            "focal_distance_mm": cfg.focal_distance_mm,
            "frequency_hz": cfg.frequency_hz,
            "n_iter": cfg.n_iter,
            "h_base_mm": cfg.h_base_mm,
            "clip_percentile": float(args.clip_percentile),
            "gamma": float(args.gamma),
            "stl_grid_stride": max(int(args.stl_grid_stride), 1),
        },
        "metrics": {
            "target_reconstruction_corr": corr,
            "target_amp_nonzero_frac": float(np.mean(target_amp[aperture_mask] > 0.0)),
            "thickness_min_mm": float(np.min(thickness[aperture_mask]) * 1e3),
            "thickness_max_mm": float(np.max(thickness[aperture_mask]) * 1e3),
            "h_max_mm": float(cfg.h_max_m * 1e3),
        },
        "stl": {
            "status": stl_status,
            "error": stl_error,
        },
        "timings_s": timings,
        "outputs": {
            "target_raw_bridge_amplitude_png": "target_raw_bridge_amplitude.png",
            "target_amplitude_for_iasa_png": "target_amplitude_for_iasa.png",
            "lens_phase_wrapped_png": "lens_phase_wrapped.png",
            "reconstructed_pressure_at_focus_png": "reconstructed_pressure_at_focus.png",
            "lens_thickness_map_png": "lens_thickness_map.png",
            "target_vs_reconstruction_png": "target_vs_reconstruction.png",
            "fields_npz": "bridge_inverse_replica_fields.npz",
            "stl": "bridge_inverse_hologram_lens.stl" if stl_status.startswith("ok") else None,
        },
    }

    with open(out_dir / "bridge_inverse_replica_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _log("[done] manifest written")

    print(f"Output dir: {out_dir}")
    print(f"Correlation(target,recon): {corr:.4f}")
    print(f"Thickness range [mm]: {np.min(thickness[aperture_mask])*1e3:.4f} .. {np.max(thickness[aperture_mask])*1e3:.4f}")
    print(f"STL: {stl_status}")


if __name__ == "__main__":
    main()
