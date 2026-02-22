#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  run_linux_all.sh — one-shot "everything run" for the Linux box
# ════════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash scripts/run_linux_all.sh                # defaults
#    bash scripts/run_linux_all.sh --tag nightly   # tag the output dirs
#    bash scripts/run_linux_all.sh --threads 4 --elem-per-lambda 6
#
#  What it does:
#    1. Activates the fenicsx conda environment
#    2. Runs the corrected-model sweep
#    3. Runs the production verification pipeline
#    4. Prints a summary of results in results/
#
#  All CLI flags are forwarded to BOTH scripts unchanged, so:
#    --out, --tag, --elem-per-lambda, --threads, --overwrite
#  all work.  (--out is generally not passed here since each script
#  picks its own timestamped directory.)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── activate environment ─────────────────────────────────────────────
CONDA_ENV="fenicsx"

# Try conda activate; fall back to source activate for older setups.
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    : # already inside the env
else
    echo "ERROR: conda not found and CONDA_PREFIX not set."
    echo "       Please activate the '$CONDA_ENV' environment first."
    exit 1
fi

echo ""
echo "################################################################"
echo "  acousto-tweezers  —  Linux full run"
echo "  Environment : $CONDA_ENV"
echo "  Python      : $(python --version 2>&1)"
echo "  Working dir : $REPO_ROOT"
echo "  Extra args  : $*"
echo "################################################################"
echo ""

# ── 1. Corrected-model sweep ────────────────────────────────────────
echo ">>> Starting corrected_model_sweep.py …"
python scripts/experiments/corrected_model_sweep.py "$@"
echo ""

# ── 2. Production verification pipeline ────────────────────────────
echo ">>> Starting production_farfield_run.py …"
python scripts/experiments/production_farfield_run.py "$@"
echo ""

# ── 3. Summary ──────────────────────────────────────────────────────
echo "################################################################"
echo "  ALL RUNS COMPLETE  —  $(date -Iseconds)"
echo ""
echo "  Results:"
echo "  -------"
ls -1d results/corrected_model_* results/farfield_production_* 2>/dev/null \
    | tail -4 || echo "  (none found)"
echo ""
echo "  Check for FAILED.txt:"
find results/ -name FAILED.txt -print 2>/dev/null || echo "  None — all good."
echo "################################################################"
