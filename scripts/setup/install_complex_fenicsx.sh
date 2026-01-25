#!/bin/bash
# Install Complex PETSc + DOLFINx from source
#
# This script builds PETSc with --with-scalar-type=complex and then
# builds DOLFINx against it.
#
# Prerequisites:
#   - cmake, make, gcc/g++, gfortran
#   - mpi (openmpi or mpich)
#   - Python 3.10+ with pip
#
# Usage:
#   ./scripts/setup/install_complex_fenicsx.sh
#
# This takes 30-60 minutes depending on hardware.
#
# Author: Acousto-Tweezers Project
# Date: January 2026

set -e  # Exit on error

echo "=============================================="
echo "Installing Complex PETSc + DOLFINx"
echo "=============================================="

# Configuration
INSTALL_PREFIX="${HOME}/.local/fenicsx-complex"
PETSC_VERSION="3.21.1"
DOLFINX_VERSION="0.9.0"
NUM_PROCS=$(nproc)

mkdir -p "${INSTALL_PREFIX}"
mkdir -p /tmp/fenicsx-build
cd /tmp/fenicsx-build

# ==========================================
# 1. Build PETSc with complex scalars
# ==========================================
echo ""
echo "[1/4] Building PETSc ${PETSC_VERSION} with complex scalars..."
echo ""

if [ ! -d "petsc-${PETSC_VERSION}" ]; then
    wget -q https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz
    tar xzf petsc-${PETSC_VERSION}.tar.gz
fi

cd petsc-${PETSC_VERSION}

export PETSC_DIR=$(pwd)
export PETSC_ARCH=complex-opt

./configure \
    --with-scalar-type=complex \
    --with-fortran-bindings=0 \
    --with-debugging=0 \
    --with-shared-libraries=1 \
    --download-hypre \
    --download-mumps \
    --download-scalapack \
    --download-metis \
    --download-parmetis \
    --download-superlu_dist \
    --prefix="${INSTALL_PREFIX}/petsc"

make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all -j${NUM_PROCS}
make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} install

export PETSC_DIR="${INSTALL_PREFIX}/petsc"
unset PETSC_ARCH

cd /tmp/fenicsx-build

# ==========================================
# 2. Install petsc4py
# ==========================================
echo ""
echo "[2/4] Installing petsc4py..."
echo ""

pip install --no-cache-dir petsc4py==${PETSC_VERSION}

# Verify complex
python3 -c "from petsc4py import PETSc; assert PETSc.ScalarType == complex, 'PETSc not complex!'; print('✓ PETSc is complex')"

# ==========================================
# 3. Build Basix, UFL, FFCx
# ==========================================
echo ""
echo "[3/4] Installing FEniCSx components..."
echo ""

pip install --no-cache-dir fenics-basix==${DOLFINX_VERSION}
pip install --no-cache-dir fenics-ufl==${DOLFINX_VERSION}
pip install --no-cache-dir fenics-ffcx==${DOLFINX_VERSION}

# ==========================================
# 4. Build DOLFINx
# ==========================================
echo ""
echo "[4/4] Building DOLFINx..."
echo ""

if [ ! -d "dolfinx-${DOLFINX_VERSION}" ]; then
    wget -q https://github.com/FEniCS/dolfinx/archive/refs/tags/v${DOLFINX_VERSION}.tar.gz -O dolfinx.tar.gz
    tar xzf dolfinx.tar.gz
    mv dolfinx-${DOLFINX_VERSION} dolfinx-src
fi

cd dolfinx-src/cpp
mkdir -p build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}/dolfinx" \
    -DCMAKE_BUILD_TYPE=Release

make -j${NUM_PROCS}
make install

cd ../../python
pip install --no-cache-dir .

cd /tmp/fenicsx-build

# ==========================================
# Final verification
# ==========================================
echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="

python3 << 'EOF'
from petsc4py import PETSc
import dolfinx
import numpy as np

print(f"PETSc ScalarType: {PETSc.ScalarType}")
print(f"DOLFINx version: {dolfinx.__version__}")

assert PETSc.ScalarType == complex, "PETSc is not complex!"
print("✓ Complex backend verified!")

# Quick test: create a complex function
from dolfinx import mesh, fem
from mpi4py import MPI

msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
V = fem.functionspace(msh, ("Lagrange", 1))
u = fem.Function(V, dtype=complex)
u.x.array[:] = 1.0 + 2.0j
print(f"✓ Complex function created: max = {np.max(u.x.array)}")

print("\n✓ Installation successful!")
EOF

# ==========================================
# Setup environment script
# ==========================================
cat > "${INSTALL_PREFIX}/activate.sh" << 'ENVSCRIPT'
export PETSC_DIR="${HOME}/.local/fenicsx-complex/petsc"
export LD_LIBRARY_PATH="${PETSC_DIR}/lib:${LD_LIBRARY_PATH}"
export PATH="${HOME}/.local/fenicsx-complex/dolfinx/bin:${PATH}"
echo "Complex FEniCSx environment activated"
ENVSCRIPT

echo ""
echo "=============================================="
echo "Installation complete!"
echo ""
echo "To activate, run:"
echo "  source ${INSTALL_PREFIX}/activate.sh"
echo "=============================================="
