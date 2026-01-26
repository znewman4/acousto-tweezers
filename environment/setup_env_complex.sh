#!/bin/bash
# Setup script for complex PETSc + FEniCSx environment
# Required for time-harmonic acoustics with PML
#
# Usage:
#   chmod +x environment/setup_env_complex.sh
#   ./environment/setup_env_complex.sh
#
# After installation, activate with:
#   micromamba activate acousto-complex

set -e

echo "============================================================"
echo "Setting up Complex PETSc + FEniCSx Environment"
echo "============================================================"

# Check for micromamba or conda
if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: No conda/mamba/micromamba found!"
    echo "Install micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    exit 1
fi

echo "Using: $CONDA_CMD"

# Remove existing env if present
echo ""
echo "Removing existing acousto-complex environment (if any)..."
$CONDA_CMD env remove -n acousto-complex -y 2>/dev/null || true

# Create environment from yml
echo ""
echo "Creating environment from complex-fenicsx.yml..."
$CONDA_CMD env create -f environment/complex-fenicsx.yml

echo ""
echo "============================================================"
echo "Environment created! Activate with:"
echo "  $CONDA_CMD activate acousto-complex"
echo ""
echo "Verify complex PETSc with:"
echo '  python -c "from petsc4py import PETSc; import numpy as np; print(f\"Complex: {np.issubdtype(PETSc.ScalarType, np.complexfloating)}\")"'
echo ""
echo "Expected output: Complex: True"
echo "============================================================"
