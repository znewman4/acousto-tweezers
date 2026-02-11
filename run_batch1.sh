#!/bin/bash
# Wrapper to run batch-1 script with signal protection
set -e
cd /home/znewman4/projects/acousto-tweezers
rm -rf results/latest/batch1_2026-02-11 2>/dev/null || true
exec micromamba run -n acousto-complex python -u scripts/analysis/run_batch1_outputs.py 2>&1
