#!/usr/bin/env python3
"""
Strict replica-style inverse C-shape acoustic lens generator.

This script follows a direct IASA/Gerchberg-Saxton workflow with:
1) single-frequency ASM propagation,
2) hard binary C-shape focal target amplitude,
3) phase-only lens updates,
4) wrapped-phase to thickness conversion,
5) required diagnostic plots and STL export.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ReplicaConfig:
    # Physical constants (reference defaults)
    frequency_hz: float = 2.44e6
    c_water: float = 1480.0
    c_lens: float = 2636.0
    rho_water: float = 998.0

    # Lens / source geometry
    transducer_diameter_mm: float = 20.0
    focal_distance_mm: float = 40.0
    n_grid: int = 512

    # Particle / drag / scaling constants
    particle_radius_m: float = 0.045e-3
    particle_density: float = 1370.0
    particle_sound_speed: float = 2350.0
    mu: float = 1e-3
    source_pressure_pa: float = 0.05e6

    # C-shape target geometry
    arc_radius_mm: float = 1.5
    arc_width_mm: float = 0.6
    gap_angle_deg: float = 100.0
    gap_direction_deg: float = 0.0

    # Thickness mapping
    h_base_mm: float = 1.0

    # Iteration count
    n_iter: int = 100

    # Output directory
    output_dir: str = "."

    def __post_init__(self) -> None:
        if self.n_grid < 64:
            raise ValueError("n_grid must be >= 64")
        if self.n_iter < 1:
            raise ValueError("n_iter must be >= 1")

    @property
    def wavelength_m(self) -> float:
        return self.c_water / self.frequency_hz

    @property
    def k_water(self) -> float:
        return 2.0 * np.pi * self.frequency_hz / self.c_water

    @property
    def k_lens(self) -> float:
        return 2.0 * np.pi * self.frequency_hz / self.c_lens

    @property
    def h_max_m(self) -> float:
        dk = self.k_lens - self.k_water
        return 2.0 * np.pi / abs(dk)

    @property
    def diameter_m(self) -> float:
        return self.transducer_diameter_mm * 1e-3

    @property
    def focal_distance_m(self) -> float:
        return self.focal_distance_mm * 1e-3

    @property
    def arc_radius_m(self) -> float:
        return self.arc_radius_mm * 1e-3

    @property
    def arc_width_m(self) -> float:
        return self.arc_width_mm * 1e-3

    @property
    def gap_angle_rad(self) -> float:
        return np.deg2rad(self.gap_angle_deg)

    @property
    def gap_direction_rad(self) -> float:
        return np.deg2rad(self.gap_direction_deg)

    @property
    def h_base_m(self) -> float:
        return self.h_base_mm * 1e-3


def make_grid(cfg: ReplicaConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Square Cartesian grid over full lens diameter.

    Replication setting:
    - domain size = transducer diameter
    - dx = dia / N
    - x from -dia/2 to +dia/2
    """
    n = cfg.n_grid
    dia = cfg.diameter_m
    dx = dia / n

    x = np.linspace(-0.5 * dia, 0.5 * dia, n)
    y = np.linspace(-0.5 * dia, 0.5 * dia, n)
    xg, yg = np.meshgrid(x, y)

    r = np.sqrt(xg**2 + yg**2)
    theta = np.arctan2(yg, xg)
    aperture_mask = r <= (0.5 * dia)
    return x, y, xg, yg, r, theta, aperture_mask, dx


def build_c_target_amplitude(r: np.ndarray, theta: np.ndarray, cfg: ReplicaConfig) -> tuple[np.ndarray, np.ndarray]:
    arc_r = cfg.arc_radius_m
    arc_w = cfg.arc_width_m
    gap_half = 0.5 * cfg.gap_angle_rad

    in_ring = (r > (arc_r - 0.5 * arc_w)) & (r < (arc_r + 0.5 * arc_w))
    angle_local = np.angle(np.exp(1j * (theta - cfg.gap_direction_rad)))
    not_in_gap = np.abs(angle_local) > gap_half

    support_mask = in_ring & not_in_gap
    target_amp = support_mask.astype(float)
    return target_amp, support_mask


def propagate_asm(field: np.ndarray, k: float, z: float, dx: float) -> np.ndarray:
    n_y, n_x = field.shape
    if n_y != n_x:
        raise ValueError("field must be square")
    n = n_x

    fx = np.fft.fftfreq(n, d=dx)
    fy = np.fft.fftfreq(n, d=dx)
    kx, ky = np.meshgrid(2.0 * np.pi * fx, 2.0 * np.pi * fy)
    kz = np.sqrt(np.maximum(k**2 - kx**2 - ky**2, 0.0)).astype(complex)

    h = np.exp(1j * kz * z)
    return np.fft.ifft2(np.fft.fft2(field) * h)


def run_iasa(
    cfg: ReplicaConfig,
    aperture_mask: np.ndarray,
    target_amp: np.ndarray,
    dx: float,
) -> np.ndarray:
    n = cfg.n_grid
    lens_field = np.ones((n, n), dtype=complex)
    lens_field[~aperture_mask] = 0.0

    for _ in range(cfg.n_iter):
        img_field = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)
        img_field = target_amp * np.exp(1j * np.angle(img_field))
        lens_field = np.exp(1j * np.angle(propagate_asm(img_field, cfg.k_water, -cfg.focal_distance_m, dx)))
        lens_field[~aperture_mask] = 0.0

    return lens_field


def ring_radius_diagnostics(
    p_mag: np.ndarray,
    r: np.ndarray,
    support_mask: np.ndarray,
    lens_radius_m: float,
) -> dict[str, float]:
    idx_max = int(np.argmax(p_mag))
    r_global_m = float(r.ravel()[idx_max])

    n_bins = 240
    edges = np.linspace(0.0, lens_radius_m, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial_mean = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        in_bin = (r >= edges[i]) & (r < edges[i + 1]) & support_mask
        if np.any(in_bin):
            radial_mean[i] = float(np.mean(p_mag[in_bin]))

    if np.all(np.isnan(radial_mean)):
        measured_ring_radius_m = float("nan")
        radial_mean_peak = float("nan")
    else:
        best = int(np.nanargmax(radial_mean))
        measured_ring_radius_m = float(centers[best])
        radial_mean_peak = float(radial_mean[best])

    return {
        "global_max_radius_m": r_global_m,
        "measured_ring_radius_m": measured_ring_radius_m,
        "radial_support_peak_pa": radial_mean_peak,
    }


def compute_gorkov(cfg: ReplicaConfig, p_mag: np.ndarray, dx: float) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    rho_w = cfg.rho_water
    c_w = cfg.c_water
    rho_p = cfg.particle_density
    c_p = cfg.particle_sound_speed
    a = cfg.particle_radius_m

    f1 = 1.0 - (rho_w * c_w**2) / (rho_p * c_p**2)
    f2 = 2.0 * (rho_p - rho_w) / (2.0 * rho_p + rho_w)
    vp = (4.0 / 3.0) * np.pi * a**3

    u_gork = vp * (
        f1 * p_mag**2 / (2.0 * rho_w * c_w**2)
        - 3.0 * f2 * rho_w * (p_mag**2 / (rho_w * c_w) ** 2) / 4.0
    )

    fy, fx = np.gradient(-u_gork, dx, dx)
    f_peak = float(np.max(np.sqrt(fx**2 + fy**2)))
    f_drag = float(6.0 * np.pi * cfg.mu * a * 1e-3)
    return u_gork, f_peak, f_drag, fx, fy


def save_imshow(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    cbar: str,
    cmap: str,
    save_path: Path,
    zoom_half_mm: float | None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, label=cbar)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if zoom_half_mm is not None:
        ax.set_xlim(-zoom_half_mm, zoom_half_mm)
        ax.set_ylim(-zoom_half_mm, zoom_half_mm)
    fig.tight_layout()
    fig.savefig(save_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(
    target_amp: np.ndarray,
    lens_phase: np.ndarray,
    p_mag: np.ndarray,
    u_gork: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    zoom_half_mm: float,
) -> None:
    extent = [x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(target_amp, origin="lower", extent=extent, cmap="hot", aspect="equal")
    plt.colorbar(im0, ax=axes[0], label="target amplitude")
    axes[0].set_title("Target C-shape Amplitude")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_xlim(-zoom_half_mm, zoom_half_mm)
    axes[0].set_ylim(-zoom_half_mm, zoom_half_mm)

    im1 = axes[1].imshow(lens_phase, origin="lower", extent=extent, cmap="twilight", vmin=0.0, vmax=2.0 * np.pi, aspect="equal")
    plt.colorbar(im1, ax=axes[1], label="phase (rad)")
    axes[1].set_title("Hologram Phase Map")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")

    im2 = axes[2].imshow(p_mag * 1e-3, origin="lower", extent=extent, cmap="hot", aspect="equal")
    plt.colorbar(im2, ax=axes[2], label="Pressure (kPa)")
    axes[2].set_title("Pressure Magnitude at Focus")
    axes[2].set_xlabel("x (mm)")
    axes[2].set_ylabel("y (mm)")
    axes[2].set_xlim(-zoom_half_mm, zoom_half_mm)
    axes[2].set_ylim(-zoom_half_mm, zoom_half_mm)

    im3 = axes[3].imshow(u_gork * 1e18, origin="lower", extent=extent, cmap="RdBu_r", aspect="equal")
    plt.colorbar(im3, ax=axes[3], label="U (aJ)")
    axes[3].set_title("Gorkov Potential")
    axes[3].set_xlabel("x (mm)")
    axes[3].set_ylabel("y (mm)")
    axes[3].set_xlim(-zoom_half_mm, zoom_half_mm)
    axes[3].set_ylim(-zoom_half_mm, zoom_half_mm)

    fig.tight_layout()
    fig.savefig(save_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def export_stl(
    cfg: ReplicaConfig,
    lens_field: np.ndarray,
    aperture_mask: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    save_path: Path,
    timings: dict,
    grid_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    try:
        import pyvista as pv
        from pymeshfix import MeshFix
    except Exception as exc:
        raise RuntimeError(
            "STL export requires pyvista and pymeshfix. Install with: pip install pyvista pymeshfix"
        ) from exc

    phi_wrapped = np.mod(np.angle(lens_field), 2.0 * np.pi)
    thickness = cfg.h_base_m + cfg.h_max_m * (phi_wrapped / (2.0 * np.pi))
    thickness[~aperture_mask] = cfg.h_base_m

    s = max(int(grid_stride), 1)
    if s > 1:
        idx = np.unique(np.r_[np.arange(0, cfg.n_grid, s, dtype=int), cfg.n_grid - 1])
        xg_s = xg[np.ix_(idx, idx)]
        yg_s = yg[np.ix_(idx, idx)]
        thickness_s = thickness[np.ix_(idx, idx)]
    else:
        xg_s = xg
        yg_s = yg
        thickness_s = thickness

    n = int(xg_s.shape[0])
    x3 = np.repeat(xg_s[:, :, np.newaxis], 2, axis=2)
    y3 = np.repeat(yg_s[:, :, np.newaxis], 2, axis=2)
    z3 = np.zeros((n, n, 2), dtype=float)
    z3[:, :, 0] = 0.0
    z3[:, :, 1] = thickness_s

    _t = time.perf_counter()
    grid = pv.StructuredGrid(x3, y3, z3)
    grid.dimensions = [n, n, 2]
    timings["stl_structured_grid"] = time.perf_counter() - _t

    _t = time.perf_counter()
    mesh = grid.extract_surface().triangulate().clean()
    timings["stl_surface_extract"] = time.perf_counter() - _t

    clip_radius = 0.99 * 0.5 * cfg.diameter_m
    clip_height = 2.0 * (cfg.h_base_m + cfg.h_max_m)

    _t = time.perf_counter()
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.5 * clip_height),
        direction=(0.0, 0.0, 1.0),
        radius=clip_radius,
        height=clip_height,
        resolution=256,
    ).triangulate().clean()
    timings["stl_cylinder"] = time.perf_counter() - _t

    _t = time.perf_counter()
    try:
        clipped = mesh.boolean_intersection(cylinder).triangulate().clean()
    except Exception:
        # Fallback for environments where boolean intersection is less robust.
        clipped = mesh.clip_surface(cylinder, invert=False).triangulate().clean()
    timings["stl_boolean_clip"] = time.perf_counter() - _t

    if clipped.n_cells == 0:
        # Last-resort fallback: avoid writing an empty STL if clipping failed.
        clipped = mesh

    _t = time.perf_counter()
    repaired = clipped
    try:
        fixer = MeshFix(clipped)
        fixer.repair(joincomp=True, remove_smallest_components=False)
        repaired_candidate = fixer.mesh.triangulate().clean()
        if repaired_candidate.n_cells > 0:
            repaired = repaired_candidate
    except Exception:
        # Keep unclipped meshfix fallback mesh on repair errors.
        repaired = clipped
    timings["stl_meshfix_repair"] = time.perf_counter() - _t

    if repaired.n_cells == 0:
        raise RuntimeError("STL export produced an empty mesh")

    _t = time.perf_counter()
    repaired.save(str(save_path))
    timings["stl_save"] = time.perf_counter() - _t

    open_edges = int(repaired.n_open_edges)
    return phi_wrapped, thickness, open_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict replica IASA C-shape lens generator")
    parser.add_argument("--frequency-hz", type=float, default=2.44e6)
    parser.add_argument("--c-water", type=float, default=1480.0)
    parser.add_argument("--c-lens", type=float, default=2636.0)
    parser.add_argument("--rho-water", type=float, default=998.0)

    parser.add_argument("--transducer-diameter-mm", type=float, default=20.0)
    parser.add_argument("--focal-distance-mm", type=float, default=40.0)
    parser.add_argument("--n-grid", type=int, default=512)

    parser.add_argument("--particle-radius-m", type=float, default=0.045e-3)
    parser.add_argument("--particle-density", type=float, default=1370.0)
    parser.add_argument("--particle-sound-speed", type=float, default=2350.0)
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--source-pressure-pa", type=float, default=0.05e6)

    parser.add_argument("--arc-radius-mm", type=float, default=1.5)
    parser.add_argument("--arc-width-mm", type=float, default=0.6)
    parser.add_argument("--gap-angle-deg", type=float, default=100.0)
    parser.add_argument("--gap-direction-deg", type=float, default=0.0)

    parser.add_argument("--h-base-mm", type=float, default=1.0)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--skip-stl", action="store_true", help="skip STL export")
    parser.add_argument("--timings-json", type=str, default="", help="save timing results to this JSON path")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> tuple[ReplicaConfig, bool, str]:
    return ReplicaConfig(
        frequency_hz=float(args.frequency_hz),
        c_water=float(args.c_water),
        c_lens=float(args.c_lens),
        rho_water=float(args.rho_water),
        transducer_diameter_mm=float(args.transducer_diameter_mm),
        focal_distance_mm=float(args.focal_distance_mm),
        n_grid=int(args.n_grid),
        particle_radius_m=float(args.particle_radius_m),
        particle_density=float(args.particle_density),
        particle_sound_speed=float(args.particle_sound_speed),
        mu=float(args.mu),
        source_pressure_pa=float(args.source_pressure_pa),
        arc_radius_mm=float(args.arc_radius_mm),
        arc_width_mm=float(args.arc_width_mm),
        gap_angle_deg=float(args.gap_angle_deg),
        gap_direction_deg=float(args.gap_direction_deg),
        h_base_mm=float(args.h_base_mm),
        n_iter=int(args.n_iter),
        output_dir=str(args.output_dir),
    ), bool(args.skip_stl), str(args.timings_json)


def main() -> None:
    args = parse_args()
    cfg, skip_stl, timings_json_path = config_from_args(args)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t_total_start = time.perf_counter()

    _t = time.perf_counter()
    x, y, xg, yg, r, theta, aperture_mask, dx = make_grid(cfg)
    timings["grid_creation"] = time.perf_counter() - _t

    _t = time.perf_counter()
    target_amp, support_mask = build_c_target_amplitude(r, theta, cfg)
    timings["target_creation"] = time.perf_counter() - _t

    print(f"Wavelength            : {cfg.wavelength_m * 1e3:.6f} mm")
    print(f"h_max                 : {cfg.h_max_m * 1e3:.6f} mm")
    print(f"Target C-shape arc radius : {cfg.arc_radius_mm:.6f} mm")

    _t = time.perf_counter()
    lens_field = run_iasa(cfg, aperture_mask, target_amp, dx)
    timings["iasa_total"] = time.perf_counter() - _t
    timings["iasa_per_iteration"] = timings["iasa_total"] / cfg.n_iter
    print("IASA Done.")

    _t = time.perf_counter()
    p_field = propagate_asm(lens_field, cfg.k_water, cfg.focal_distance_m, dx)
    scale = cfg.source_pressure_pa / (np.sqrt(np.mean(np.abs(lens_field[aperture_mask]) ** 2)) + 1e-12)
    p_mag = np.abs(p_field) * scale
    timings["final_forward_propagation"] = time.perf_counter() - _t

    _t = time.perf_counter()
    diagnostics = ring_radius_diagnostics(
        p_mag=p_mag,
        r=r,
        support_mask=support_mask,
        lens_radius_m=0.5 * cfg.diameter_m,
    )
    timings["ring_diagnostics"] = time.perf_counter() - _t

    analytical_arc_radius_mm = cfg.arc_radius_mm
    measured_ring_radius_mm = diagnostics["measured_ring_radius_m"] * 1e3
    ring_radius_ratio = measured_ring_radius_mm / max(analytical_arc_radius_mm, 1e-12)

    _t = time.perf_counter()
    u_gork, f_peak, f_drag, _, _ = compute_gorkov(cfg, p_mag, dx)
    timings["gorkov_calculation"] = time.perf_counter() - _t
    trapped = bool(f_peak > f_drag)

    print(f"Analytical Arc Radius           : {analytical_arc_radius_mm:.6f} mm")
    print(f"Measured ring radius            : {measured_ring_radius_mm:.6f} mm")
    print(f"Measured ring radius/particle ratio : {ring_radius_ratio:.6f}")
    print(f"Peak Gor'kov force              : {f_peak:.6e} N")
    print(f"Stokes drag                     : {f_drag:.6e} N")
    print("TRAPPED" if trapped else "NOT TRAPPED")

    lens_phase_wrapped = np.mod(np.angle(lens_field), 2.0 * np.pi)
    zoom_half_mm = cfg.arc_radius_mm * 3.0

    _t = time.perf_counter()
    save_summary_figure(
        target_amp=target_amp,
        lens_phase=lens_phase_wrapped,
        p_mag=p_mag,
        u_gork=u_gork,
        x=x,
        y=y,
        save_path=out_dir / "c_shape_results.png",
        zoom_half_mm=zoom_half_mm,
    )
    timings["summary_figure"] = time.perf_counter() - _t

    _t = time.perf_counter()
    save_imshow(
        data=target_amp,
        x=x,
        y=y,
        title="Target C-shape Amplitude",
        cbar="target amplitude",
        cmap="hot",
        save_path=out_dir / "target_c_shape.png",
        zoom_half_mm=zoom_half_mm,
        vmin=0.0,
        vmax=1.0,
    )
    timings["save_target_c_shape"] = time.perf_counter() - _t

    _t = time.perf_counter()
    save_imshow(
        data=lens_phase_wrapped,
        x=x,
        y=y,
        title="Hologram Phase Map",
        cbar="phase (rad)",
        cmap="twilight",
        save_path=out_dir / "hologram_phase_map.png",
        zoom_half_mm=None,
        vmin=0.0,
        vmax=2.0 * np.pi,
    )
    timings["save_hologram_phase_map"] = time.perf_counter() - _t

    _t = time.perf_counter()
    save_imshow(
        data=p_mag * 1e-3,
        x=x,
        y=y,
        title="Pressure Magnitude at Focus",
        cbar="Pressure (kPa)",
        cmap="hot",
        save_path=out_dir / "pressure_at_focus.png",
        zoom_half_mm=zoom_half_mm,
    )
    timings["save_pressure_at_focus"] = time.perf_counter() - _t

    _t = time.perf_counter()
    save_imshow(
        data=u_gork * 1e18,
        x=x,
        y=y,
        title="Gorkov Potential",
        cbar="U (aJ)",
        cmap="RdBu_r",
        save_path=out_dir / "gorkov_potential.png",
        zoom_half_mm=zoom_half_mm,
    )
    timings["save_gorkov_potential"] = time.perf_counter() - _t

    if skip_stl:
        print("STL export skipped (--skip-stl).")
        timings["stl_total"] = 0.0
    else:
        _t = time.perf_counter()
        _, _, open_edges = export_stl(
            cfg=cfg,
            lens_field=lens_field,
            aperture_mask=aperture_mask,
            xg=xg,
            yg=yg,
            save_path=out_dir / "c_shape_hologram_lens.stl",
            timings=timings,
        )
        timings["stl_total"] = time.perf_counter() - _t
        print(f"STL saved: {out_dir / 'c_shape_hologram_lens.stl'} | Open edges: {open_edges}")

    timings["total"] = time.perf_counter() - t_total_start

    _w = 36
    print()
    print("=" * (_w + 14))
    print("  TIMING SUMMARY")
    print("=" * (_w + 14))
    timing_keys = [
        "grid_creation",
        "target_creation",
        "iasa_total",
        "iasa_per_iteration",
        "final_forward_propagation",
        "ring_diagnostics",
        "gorkov_calculation",
        "summary_figure",
        "save_target_c_shape",
        "save_hologram_phase_map",
        "save_pressure_at_focus",
        "save_gorkov_potential",
        "stl_total",
        "stl_structured_grid",
        "stl_surface_extract",
        "stl_cylinder",
        "stl_boolean_clip",
        "stl_meshfix_repair",
        "stl_save",
        "total",
    ]
    for key in timing_keys:
        if key in timings:
            print(f"  {key:<{_w}} {timings[key]:>8.3f} s")
    print("=" * (_w + 14))

    if timings_json_path:
        with open(timings_json_path, "w", encoding="utf-8") as _f:
            json.dump(timings, _f, indent=2)
        print(f"Timings saved: {timings_json_path}")


if __name__ == "__main__":
    main()
