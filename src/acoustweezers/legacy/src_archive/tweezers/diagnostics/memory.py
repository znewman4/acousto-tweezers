"""Memory instrumentation and diagnostics."""
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import resource
import os


def get_rss_mb():
    """Return resident set size in MB."""
    if HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1e6
    else:
        # Fallback: use rusage (less accurate but always available)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # ru_maxrss is in KB on Linux


def array_bytes(arr):
    """Estimate memory footprint of a numpy array in bytes."""
    return arr.nbytes


def array_summary(arr, name=""):
    """Return a summary string of array shape/dtype/bytes."""
    nb = array_bytes(arr)
    mb = nb / 1e6
    return f"{name:20s} | shape {str(arr.shape):30s} dtype {str(arr.dtype):12s} | {mb:8.2f} MB ({nb:.2e} bytes)"


def memory_checkpoint(label=""):
    """Print current RSS usage with optional label."""
    rss = get_rss_mb()
    print(f"[MEM CHECKPOINT] {label:40s} | RSS = {rss:.1f} MB")
    return rss


class MemoryTracker:
    """Track memory usage across a workflow."""
    
    def __init__(self):
        self.checkpoints = {}
        self.start_rss = get_rss_mb()
    
    def checkpoint(self, label):
        """Record memory at this stage."""
        rss = get_rss_mb()
        self.checkpoints[label] = rss
        delta = rss - self.start_rss
        print(f"[MEM] {label:40s} | RSS = {rss:7.1f} MB | Δ = {delta:+7.1f} MB")
        return rss
    
    def report(self):
        """Print summary report."""
        if not self.checkpoints:
            print("[MEM] No checkpoints recorded.")
            return
        print("\n" + "="*80)
        print("[MEMORY REPORT]")
        print("="*80)
        sorted_cp = sorted(self.checkpoints.items(), key=lambda x: x[1])
        min_rss = sorted_cp[0][1]
        max_rss = sorted_cp[-1][1]
        print(f"  Min RSS: {min_rss:.1f} MB at '{sorted_cp[0][0]}'")
        print(f"  Max RSS: {max_rss:.1f} MB at '{sorted_cp[-1][0]}'")
        print(f"  Peak Δ from start: {max_rss - self.start_rss:.1f} MB")
        print("="*80 + "\n")


def estimate_sparse_matrix_bytes(n_rows, n_cols, nnz, value_dtype=np.complex128, index_dtype=np.int32):
    """
    Rough estimate of sparse matrix memory (CSR format).
    nnz: number of non-zeros
    Includes: data (nnz * value_bytes), indices (nnz * index_bytes), indptr ((n_rows+1) * index_bytes)
    """
    value_bytes = np.dtype(value_dtype).itemsize
    index_bytes = np.dtype(index_dtype).itemsize
    data_mem = nnz * value_bytes
    indices_mem = nnz * index_bytes
    indptr_mem = (n_rows + 1) * index_bytes
    total = data_mem + indices_mem + indptr_mem
    return total


def print_memory_banner(description, Nx, Ny, Nz, omega, dtype_precision="single"):
    """Print a diagnostic banner with estimated memory usage."""
    print("\n" + "="*80)
    print(f"[RUN BANNER] {description}")
    print("="*80)
    print(f"  Grid: Nx={Nx}, Ny={Ny}, Nz={Nz}, total_points={Nx*Ny*Nz}")
    print(f"  Omega: {omega:.2e} rad/s")
    print(f"  Dtype precision: {dtype_precision}")
    
    # Estimate memory for key arrays
    if dtype_precision == "single":
        p_dtype = np.complex64
        U_dtype = np.float32
    else:
        p_dtype = np.complex128
        U_dtype = np.float64
    
    p_bytes = Nx * Ny * Nz * np.dtype(p_dtype).itemsize
    U_bytes = Nx * Ny * Nz * np.dtype(U_dtype).itemsize
    F_bytes = 3 * Nx * Ny * Nz * np.dtype(U_dtype).itemsize
    
    print(f"\nEstimated memory for single copy:")
    print(f"  p (complex pressure):     {p_bytes/1e6:8.1f} MB")
    print(f"  U (Gor'kov potential):    {U_bytes/1e6:8.1f} MB")
    print(f"  F (Fx, Fy, Fz):           {F_bytes/1e6:8.1f} MB")
    
    # Rough matrix estimate: for 3D Laplacian, ~7 stencil points per interior node
    nnz_est = 7 * Nx * Ny * Nz
    A_bytes = estimate_sparse_matrix_bytes(Nx*Ny*Nz, Nx*Ny*Nz, nnz_est, value_dtype=p_dtype)
    print(f"  A (sparse matrix, 7-pt stencil): {A_bytes/1e6:8.1f} MB")
    
    total_est = p_bytes + U_bytes + F_bytes + A_bytes
    print(f"\nTotal single-instance estimate: {total_est/1e6:8.1f} MB")
    print("="*80 + "\n")
