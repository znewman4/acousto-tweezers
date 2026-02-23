#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  run_linux_all.sh — one-shot "everything run" for the Linux box
# ════════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash scripts/run_linux_all.sh                          # defaults
#    bash scripts/run_linux_all.sh --tag nightly             # tag the run
#    bash scripts/run_linux_all.sh --threads 4 --elem-per-lambda 6
#    bash scripts/run_linux_all.sh --case configs/cases/canonical_farfield.json
#
#  Run root layout (audit-grade, self-contained):
#    results/<run_id>/
#      logs/                 (combined logs)
#      config/               (copies of case files + presets)
#      csv/                  (reserved for combined CSVs)
#      fields/               (exported XDMF/HDF5 fields)
#      figures_2d/           (2D figure suite)
#      figures_3d/           (3D visualisation)
#      comsol_compare/       (COMSOL comparison outputs)
#      audit/                (MANIFEST.json, AUDIT_REPORT.md)
#      corrected_sweep/      (corrected_model_sweep output)
#      production/           (production_farfield_run output)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Parse wrapper-level args ─────────────────────────────────────────
TAG=""
CASE_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"; shift 2 ;;
        --case)
            CASE_FILE="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Activate environment ─────────────────────────────────────────────
CONDA_ENV="acousto-complex"

# Try module system first (Bristol Linux desktops)
if type module &>/dev/null; then
    module load anaconda/3-2025 2>/dev/null || true
fi

if command -v conda &>/dev/null; then
    set +u  # conda activation scripts may reference unbound vars
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    : # already inside the env
else
    echo "ERROR: conda not found and CONDA_PREFIX not set."
    echo "       Please activate the '$CONDA_ENV' environment first."
    exit 1
fi

# ── Create unified run root ──────────────────────────────────────────
STAMP="$(date +%Y%m%d_%H%M%S)"
TAG_SUFFIX=""
[[ -n "$TAG" ]] && TAG_SUFFIX="_${TAG}"
RUN_ID="run_${STAMP}${TAG_SUFFIX}"
RUN_ROOT="results/${RUN_ID}"

mkdir -p "${RUN_ROOT}"/{logs,config,csv,fields,figures_2d,figures_3d,comsol_compare,audit}

START_TIME="$(date -Iseconds)"

echo ""
echo "################################################################"
echo "  acousto-tweezers  —  Linux full run"
echo "  Environment : $CONDA_ENV"
echo "  Python      : $(python --version 2>&1)"
echo "  Working dir : $REPO_ROOT"
echo "  Run root    : $RUN_ROOT"
echo "  Tag         : ${TAG:-<none>}"
echo "  Case file   : ${CASE_FILE:-<presets>}"
echo "  Extra args  : ${EXTRA_ARGS[*]:-<none>}"
echo "################################################################"
echo ""

# Copy case file into config/ for archival
CASE_ARGS=()
if [[ -n "$CASE_FILE" ]]; then
    cp "$CASE_FILE" "${RUN_ROOT}/config/$(basename "$CASE_FILE")"
    CASE_ARGS=(--case "$CASE_FILE")
fi

# ── 1. Corrected-model sweep ────────────────────────────────────────
echo ">>> Starting corrected_model_sweep.py …"
python scripts/experiments/corrected_model_sweep.py \
    --out "${RUN_ROOT}/corrected_sweep" \
    --overwrite \
    "${CASE_ARGS[@]+"${CASE_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "${RUN_ROOT}/logs/corrected_sweep.log"
echo ""

# ── 2. Production verification pipeline ────────────────────────────
echo ">>> Starting production_farfield_run.py …"
python scripts/experiments/production_farfield_run.py \
    --out "${RUN_ROOT}/production" \
    --overwrite \
    "${CASE_ARGS[@]+"${CASE_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "${RUN_ROOT}/logs/production.log"
echo ""

END_TIME="$(date -Iseconds)"

# ── 3. Generate MANIFEST.json ───────────────────────────────────────
SWEEP_CMD="python scripts/experiments/corrected_model_sweep.py --out ${RUN_ROOT}/corrected_sweep --overwrite ${CASE_ARGS[*]+"${CASE_ARGS[*]}"} ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
PROD_CMD="python scripts/experiments/production_farfield_run.py --out ${RUN_ROOT}/production --overwrite ${CASE_ARGS[*]+"${CASE_ARGS[*]}"} ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"

python -c "
import sys, json; sys.path.insert(0, 'src')
from acoustweezers.core.audit import generate_manifest
from pathlib import Path
generate_manifest(
    Path('${RUN_ROOT}'),
    $(python -c "import json; print(json.dumps(['${SWEEP_CMD}', '${PROD_CMD}']))"),
    start_time='${START_TIME}',
    end_time='${END_TIME}',
    extra={'run_id': '${RUN_ID}', 'tag': '${TAG}'},
)
print('  Wrote ${RUN_ROOT}/audit/MANIFEST.json')
"

# ── 4. Summary ──────────────────────────────────────────────────────
echo ""
echo "################################################################"
echo "  ALL RUNS COMPLETE  —  $(date -Iseconds)"
echo ""
echo "  Run root: ${RUN_ROOT}"
echo ""
echo "  Contents:"
ls -1 "${RUN_ROOT}/" 2>/dev/null | sed 's/^/    /'
echo ""
echo "  Check for FAILED.txt:"
find "${RUN_ROOT}" -name FAILED.txt -print 2>/dev/null || echo "    None — all good."
echo "################################################################"