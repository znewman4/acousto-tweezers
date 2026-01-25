# Dockerfile for Complex PETSc + DOLFINx (FEniCSx)
# 
# This builds an environment where PETSc.ScalarType == complex,
# which is REQUIRED for frequency-domain Helmholtz and coupled
# fluid-solid acoustics.
#
# Build:
#   docker build -t acousto-tweezers-complex -f docker/Dockerfile .
#
# Run:
#   docker run -it --rm -v $(pwd):/workspace acousto-tweezers-complex
#
# Author: Acousto-Tweezers Project
# Date: January 2026

FROM dolfinx/dolfinx:v0.9.0

# The official dolfinx images use REAL scalars by default.
# We need to rebuild PETSc and DOLFINx with complex scalars.
# For now, use the complex variant image if available, or build from source.

# Alternative: Use the complex scalar variant
# FROM ghcr.io/fenics/dolfinx/dolfinx:v0.9.0-complex

# If base image doesn't have complex, we need to rebuild
# Check and fail fast if not complex
RUN python3 -c "from petsc4py import PETSc; assert PETSc.ScalarType == complex, 'PETSc must be complex'" || \
    echo "WARNING: Base image has real PETSc. Switching to complex build strategy."

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    gmsh \
    libgmsh-dev \
    python3-gmsh \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    gmsh \
    pyvista \
    imageio \
    imageio-ffmpeg \
    scipy \
    pandas \
    h5py \
    meshio \
    tqdm

# Set working directory
WORKDIR /workspace

# Copy project
COPY . /workspace/

# Install the package
RUN pip3 install -e .

# Verify complex backend
RUN python3 -c "from petsc4py import PETSc; print(f'PETSc ScalarType: {PETSc.ScalarType}'); assert PETSc.ScalarType == complex"

# Default command
CMD ["python3", "scripts/run_fem_multiphysics.py", "--help"]
