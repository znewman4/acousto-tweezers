"""
Audit manifest generation.

Captures runtime environment metadata into a JSON manifest
suitable for reproducibility and audit of simulation runs.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _git_commit_hash() -> str:
    """Return short git commit hash or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _dolfinx_version() -> str:
    try:
        import dolfinx
        return getattr(dolfinx, "__version__", str(dolfinx))
    except ImportError:
        return "not installed"


def _petsc_version() -> str:
    try:
        from petsc4py import PETSc
        return PETSc.Sys.getVersion().__str__() if hasattr(PETSc.Sys, "getVersion") else str(PETSc.Sys.getVersionInfo()) if hasattr(PETSc.Sys, "getVersionInfo") else "unknown"
    except Exception:
        return "not installed"


def _petsc_scalar_type() -> str:
    try:
        from petsc4py import PETSc
        import numpy as np
        st = PETSc.ScalarType
        if np.issubdtype(st, np.complexfloating):
            return f"complex ({st.__name__})"
        else:
            return f"real ({st.__name__})"
    except Exception:
        return "unknown"


def generate_manifest(
    run_dir: Path,
    command_lines: List[str],
    *,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """
    Write ``audit/MANIFEST.json`` with full environment metadata.

    Parameters
    ----------
    run_dir : Path
        Root of the run directory. ``audit/`` subdirectory will be created.
    command_lines : list of str
        The command lines that were (or will be) executed.
    start_time : str, optional
        ISO-format start time.
    end_time : str, optional
        ISO-format end time.
    extra : dict, optional
        Additional metadata entries.

    Returns
    -------
    Path
        Path to the written MANIFEST.json.
    """
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    thread_vars = {}
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        thread_vars[var] = os.environ.get(var, "(unset)")

    manifest = {
        "git_commit": _git_commit_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "dolfinx_version": _dolfinx_version(),
        "petsc_version": _petsc_version(),
        "petsc_scalar_type": _petsc_scalar_type(),
        "thread_env_vars": thread_vars,
        "cpu_count": os.cpu_count(),
        "command_lines": command_lines,
        "start_time": start_time or datetime.now().isoformat(),
        "end_time": end_time,
    }

    if extra:
        manifest.update(extra)

    out = audit_dir / "MANIFEST.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return out


def write_solver_info(
    output_path: Path,
    *,
    ksp_type: str = "preonly",
    pc_type: str = "lu",
    lu_backend: str = "mumps",
    dofs: int = 0,
    walltime_s: float = 0.0,
    mesh_time_s: float = 0.0,
    ksp_iterations: int = 0,
    ksp_converged_reason: str = "",
    ksp_residual_norm: float = 0.0,
    max_pressure: float = 0.0,
    extra: Optional[Dict] = None,
) -> Path:
    """
    Write ``solver_info.json`` for a single solve.

    Parameters
    ----------
    output_path : Path
        Directory to write into.

    Returns
    -------
    Path
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out = output_path / "solver_info.json"

    info = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "lu_backend": lu_backend,
        "dofs": dofs,
        "walltime_s": walltime_s,
        "mesh_time_s": mesh_time_s,
        "ksp_iterations": ksp_iterations,
        "ksp_converged_reason": ksp_converged_reason,
        "ksp_residual_norm": ksp_residual_norm,
        "max_pressure_Pa": max_pressure,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        info.update(extra)

    with open(out, "w") as f:
        json.dump(info, f, indent=2, default=str)

    return out
