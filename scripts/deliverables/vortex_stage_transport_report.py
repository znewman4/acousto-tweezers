#!/usr/bin/env python3
"""
Part 8 - Final report for rebuilt staged translated-vortex transport study.

Outputs:
  results/deliverables/vortex_stage_transport/INDEX.md
  results/deliverables/vortex_stage_transport/report/INDEX.md
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT_OUT = PROJECT_ROOT / "results" / "deliverables" / "vortex_stage_transport"
REF_SUM = ROOT_OUT / "reference" / "reference_summary.json"
CAL_SUM = ROOT_OUT / "calibration" / "selected_vortex_summary.json"
TR_SUM = ROOT_OUT / "transport_summary.json"
OUT_MD_ROOT = ROOT_OUT / "INDEX.md"
OUT_DIR = ROOT_OUT / "report"
OUT_MD_REPORT = OUT_DIR / "INDEX.md"


def _f(v: float, n: int = 3) -> str:
    return f"{float(v):.{n}f}"


def _yn(v: bool) -> str:
    return "YES" if bool(v) else "NO"


def main() -> None:
    if not REF_SUM.exists() or not CAL_SUM.exists() or not TR_SUM.exists():
        raise FileNotFoundError("Missing summaries. Run reference/calibration/transport first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = json.loads(REF_SUM.read_text())
    cal = json.loads(CAL_SUM.read_text())
    tr = json.loads(TR_SUM.read_text())

    sel = cal["selected"]
    fp = cal.get("footprint", {})
    feas = cal.get("feasibility_statement", {})
    ctl = tr.get("selected_control")
    if ctl is None:
        raise KeyError("transport_summary.json is missing 'selected_control'. Re-run vortex_stage_transport_transport.py.")
    q = tr.get("key_questions", {})

    final_row = ctl["final_row"]
    alpha_min = q.get("minimum_alpha_for_A_to_reach_B", ctl.get("minimum_alpha_with_hit"))

    lines: list[str] = []
    lines.append("# Symmetric Vortex Staged Transport Study")
    lines.append("")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append("")
    lines.append("Scope:")
    lines.append("- Cached FEM standing-wave field only")
    lines.append("- Trap plane z* from cache utilities")
    lines.append("- No FEM reruns")
    lines.append("- Objective: minimum alpha that moves A to B starting trap")
    lines.append("- Staged mechanism: SW start -> ramp-on -> translation -> ramp-off -> SW release")
    lines.append("")

    lines.append("## 1. Scenario")
    lines.append(f"- Selected trap pair A/B: indices {ref['pair']['idx_A']} / {ref['pair']['idx_B']}")
    lines.append(f"- Initial A-B distance: {_f(ref['pair']['distance_mm'])} mm")
    lines.append(f"- Neighbor traps monitored: {int(ref['neighbors']['count'])}")
    lines.append(f"- z*: {_f(ref['z_star_mm'])} mm")
    lines.append("")

    lines.append("## 2. Smallest Feasible Symmetric Vortex at z*")
    lines.append(f"- Selected family: {sel['family']}")
    lines.append(f"- Aperture radius: {_f(sel['aperture_mm'])} mm")
    lines.append(f"- Effective source distance: {_f(sel['source_distance_mm'])} mm")
    lines.append(f"- Cone angle: {_f(sel['cone_deg']) if sel['cone_deg'] == sel['cone_deg'] else 'N/A'}")
    lines.append(f"- Ring radius at z*: {_f(fp.get('ring_radius_mm', sel['ring_radius_mm']))} mm")
    lines.append(f"- Ring diameter at z*: {_f(fp.get('ring_diameter_mm', 2.0 * sel['ring_radius_mm']))} mm")
    lines.append("")
    lines.append("Footprint context:")
    lines.append(f"- Ring / A-B spacing: {_f(fp.get('ring_over_A_B', float('nan')), 3)}")
    lines.append(f"- Ring / nearest-neighbour spacing: {_f(fp.get('ring_over_nearest_neighbor', float('nan')), 3)}")
    lines.append(f"- Ring / wavelength: {_f(fp.get('ring_over_lambda', float('nan')), 3)}")
    lines.append(f"- Ring diameter / nearest-neighbour spacing: {_f(fp.get('ring_diameter_over_nearest_neighbor', float('nan')), 3)}")
    lines.append(f"- Spans too many local traps? {_yn(feas.get('spans_too_many_traps', False))}")
    lines.append(f"- Calibration statement: {feas.get('statement', 'N/A')}")
    lines.append("")

    lines.append("## 3. A->B Minimum-Alpha Result")
    if alpha_min is None:
        lines.append("- No tested alpha achieved A-in-B hit during translation.")
    else:
        lines.append(f"- Minimum alpha achieving A-in-B hit: {_f(alpha_min, 4)}")
    lines.append(f"- Selected psi: {_f(float(ctl['psi']), 4)} rad")
    lines.append(f"- Final translation endpoint s_end: {_f(float(ctl['final_schedule']['s_end']), 4)}")
    lines.append(f"- A reached B during translation: {_yn(final_row['A_hits_B_during_translation'])}")
    lines.append(f"- Minimum A-B distance during translation: {_f(final_row['min_A_dist_to_B_translate_mm'], 4)} mm")
    lines.append("")

    lines.append("## 4. Ramp-Off and Release")
    lines.append("- Ramp-off strategy: slow ramp-off begins once A reaches B trap in the probe schedule")
    lines.append(f"- A release outcome: {final_row['A_post_release_outcome']}")
    lines.append(f"- A remains in B after ramp-off: {_yn(final_row['A_release_to_B'])}")
    lines.append(f"- A release status: {final_row['A_release_status']}")
    lines.append("")

    lines.append("## 5. Notes")
    lines.append("- This rebuild optimizes only A transport to B start location.")
    lines.append("- B and neighbor stability are not constraints in the optimization objective.")
    lines.append("")

    lines.append("## 6. Required Outputs")
    lines.append("- start_end_standing_wave_comparison.png")
    lines.append("- transport_full_domain_sequence.png")
    lines.append("- transport_roi_sequence.png")
    lines.append("- moving_vortex_transport.gif")
    lines.append("- moving_vortex_release_success_or_failure.gif")
    lines.append("- transport_metrics.csv")
    lines.append("- transport_summary.json")
    lines.append("- INDEX.md")

    body = "\n".join(lines)
    OUT_MD_ROOT.write_text(body)
    OUT_MD_REPORT.write_text(body)
    print(f"Saved {OUT_MD_ROOT}")
    print(f"Saved {OUT_MD_REPORT}")


if __name__ == "__main__":
    main()
