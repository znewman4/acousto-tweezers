#!/usr/bin/env python3
"""
Part 8 - Write final summary for symmetric-vortex limit study.

Output:
  results/deliverables/vortex_limit/INDEX.md
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

ROOT_OUT = PROJECT_ROOT / "results" / "deliverables" / "vortex_limit"
REF_SUM = ROOT_OUT / "reference" / "reference_summary.json"
CAL_SUM = ROOT_OUT / "vortex_calibration" / "calibration_summary.json"
ANA_SUM = ROOT_OUT / "limit_plots" / "analysis_summary.json"
OUT_MD = ROOT_OUT / "INDEX.md"


def _f(v: float, n: int = 3) -> str:
    return f"{v:.{n}f}"


def _fmt_or_na(v: float, n: int = 3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.{n}f}"


def _build_report(ref: dict, cal: dict, ana: dict) -> str:
    pair_d = float(ref["pair"]["distance_mm"])

    selected = cal["selected"]
    selected_waist = float(selected["waist_mm"])
    smallest_tested = float(cal["smallest_waist_tested_mm"])

    safe = ana.get("max_safe_case")
    fail = ana.get("failure_case")

    if safe is not None:
        safe_a = float(safe["A_disp_toward_B_mm"])
        safe_b = float(safe["B_disp_norm_mm"])
        safe_ab = float(safe["AB_distance_mm"])
        safe_reduction = pair_d - safe_ab
    else:
        safe_a = float("nan")
        safe_b = float("nan")
        safe_ab = float("nan")
        safe_reduction = float("nan")

    lines: list[str] = []
    lines.append("# Symmetric Vortex Limit Study")
    lines.append("")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append("")
    lines.append("Scope:")
    lines.append("- Cached FEM standing-wave field only")
    lines.append("- Trap plane z* from cache utilities")
    lines.append("- No FEM reruns")
    lines.append("")

    lines.append("## 1. Scenario")
    lines.append(f"- Selected trap pair A/B: indices {ref['pair']['idx_A']} / {ref['pair']['idx_B']}")
    lines.append(f"- Initial A-B distance: {_f(pair_d)} mm")
    lines.append(f"- Neighbor traps monitored: {ref['neighbors']['count']}")
    lines.append(f"- z*: {_f(float(ref['z_star_mm']))} mm")
    lines.append("")

    lines.append("## 2. Vortex Parameters Used")
    lines.append(f"- Selected family: {selected['family']}")
    lines.append(f"- Aperture radius: {_f(float(selected['aperture_mm']))} mm")
    lines.append(f"- Effective source distance: {_f(float(selected['source_distance_mm']))} mm")
    lines.append(f"- Waist: {_fmt_or_na(float(selected_waist))} mm")
    lines.append(f"- Cone angle: {_fmt_or_na(float(selected['cone_deg']))} deg")
    lines.append(f"- Focal distance: {_fmt_or_na(float(selected['focal_mm']))} mm")
    lines.append(f"- Ring radius at z*: {_f(float(selected['ring_radius_mm']))} mm")
    lines.append("")

    lines.append("## 3. Smallest Waist Achieved")
    lines.append(f"- Smallest waist tested: {_f(smallest_tested)} mm")
    if math.isnan(selected_waist):
        lines.append("- Waist in selected best-case vortex: N/A (selected family uses no Gaussian waist parameter)")
    else:
        lines.append(f"- Waist in selected best-case vortex: {_f(selected_waist)} mm")
    lines.append("")

    lines.append("## 4. Maximum Safe Displacement")
    if safe is None:
        lines.append("- No safe case found where B returned to its original trap and A remained trapped after release.")
    else:
        lines.append(f"- alpha: {_f(float(safe['alpha']), 2)}")
        lines.append(f"- psi: {_f(float(safe['psi']) / 3.141592653589793, 2)} pi")
        lines.append(f"- A displacement toward B: {_f(safe_a)} mm")
        lines.append(f"- B displacement magnitude: {_f(safe_b)} mm")
        lines.append(f"- New A-B distance: {_f(safe_ab)} mm")
        lines.append(f"- A-B distance reduction: {_f(safe_reduction)} mm")
    lines.append("")

    lines.append("## 5. Failure Case")
    if fail is None:
        lines.append("- No B snap-back failure threshold was reached in the tested alpha range.")
    else:
        lines.append(f"- First failure alpha: {_f(float(fail['alpha']), 2)}")
        lines.append(f"- psi: {_f(float(fail['psi']) / 3.141592653589793, 2)} pi")
        lines.append(f"- A displacement toward B at failure: {_f(float(fail['A_disp_toward_B_mm']))} mm")
        lines.append(f"- B displacement at failure: {_f(float(fail['B_disp_norm_mm']))} mm")
        lines.append(f"- B release status: {fail['B_release_status']}")
        lines.append(f"- B returns original: {bool(fail['B_returns_original'])}")
        lines.append(f"- B jumps other trap: {bool(fail['B_jumps_other'])}")
        lines.append(f"- B untrapped: {bool(fail['B_untrapped'])}")
    lines.append("")

    lines.append("## 6. Interpretation")
    if safe is None:
        lines.append(
            "- In this tested best-case symmetric setup, the vortex did not provide a robust regime "
            "for A transport with reliable B snap-back."
        )
        lines.append("- Symmetric vortex alone is not suitable for binding in this operating window.")
    else:
        if fail is None:
            lines.append(
                "- Symmetric vortex achieved measurable pre-positioning while preserving B return "
                "across the scanned strengths."
            )
            lines.append(
                "- This supports pre-positioning; selective binding by symmetric vortex alone remains unproven."
            )
        else:
            lines.append(
                "- Symmetric vortex can pre-position A toward B up to a limited threshold, but B "
                "stability eventually fails as strength increases."
            )
            lines.append(
                "- Result: symmetric vortex is suitable for pre-positioning only, not selective binding on its own."
            )
    lines.append("")

    lines.append("## 7. Artifacts")
    lines.append("- reference/: standing-wave map, trap markers, ROI reference")
    lines.append("- vortex_calibration/: candidate sweep, selected best-case vortex")
    lines.append("- overlay/: full-domain/ROI/Gorkov overlays for safe and failure points")
    lines.append("- limit_plots/: displacement/failure/disturbance curves and full metrics")
    lines.append("- gifs/: push-and-release animations")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    if not REF_SUM.exists():
        raise FileNotFoundError("Missing reference summary; run vortex_limit_reference.py")
    if not CAL_SUM.exists():
        raise FileNotFoundError("Missing calibration summary; run vortex_limit_calibration.py")
    if not ANA_SUM.exists():
        raise FileNotFoundError("Missing analysis summary; run vortex_limit_analysis.py")

    ref = json.loads(REF_SUM.read_text())
    cal = json.loads(CAL_SUM.read_text())
    ana = json.loads(ANA_SUM.read_text())

    OUT_MD.write_text(_build_report(ref, cal, ana))
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
