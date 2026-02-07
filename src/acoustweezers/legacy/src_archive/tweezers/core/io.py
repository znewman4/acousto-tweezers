"""
I/O utilities for the acousto-tweezers package.

Provides a single authoritative function for creating run directories
with the correct structure.

CRITICAL: All runs MUST go to:
    results/fem_multiphysics/run_YYYYMMDD_HHMMSS/

This module enforces this discipline.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List


def get_repo_root(reference_file: Optional[str] = None) -> Path:
    """
    Get the repository root directory.
    
    Uses multiple strategies:
    1. If reference_file provided, walk up to find pyproject.toml
    2. Look for ACOUSTO_TWEEZERS_ROOT environment variable
    3. Walk up from current working directory
    
    Parameters
    ----------
    reference_file : str, optional
        A file path within the repository (e.g., __file__ from caller)
        
    Returns
    -------
    Path
        Absolute path to repository root
        
    Raises
    ------
    RuntimeError
        If repository root cannot be determined
    """
    # Strategy 1: Walk up from reference file
    if reference_file is not None:
        path = Path(reference_file).resolve()
        for parent in [path] + list(path.parents):
            if (parent / "pyproject.toml").exists():
                return parent
    
    # Strategy 2: Environment variable
    env_root = os.environ.get("ACOUSTO_TWEEZERS_ROOT")
    if env_root is not None:
        root = Path(env_root)
        if root.exists():
            return root
    
    # Strategy 3: Walk up from CWD
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    raise RuntimeError(
        "Cannot determine repository root. "
        "Please run from within the acousto-tweezers directory "
        "or set ACOUSTO_TWEEZERS_ROOT environment variable."
    )


def make_run_dir(
    reference_file: Optional[str] = None,
    timestamp: Optional[str] = None,
    subdirs: Optional[List[str]] = None,
    base_name: str = "fem_multiphysics"
) -> Path:
    """
    Create a run directory with the correct structure.
    
    MANDATORY structure:
        results/fem_multiphysics/run_YYYYMMDD_HHMMSS/
            config.json
            diagnostics/
                sanity_report.txt
                mesh_report.txt
                solver_report.txt
                acoustics_report.txt
                interface_residuals.txt (if solids enabled)
                pml_report.txt (if PML enabled)
                actuation_report.txt
            mesh/
            figures/
            fields/
            logs/
                run.log
    
    Parameters
    ----------
    reference_file : str, optional
        Reference file to determine repo root (typically __file__)
    timestamp : str, optional
        Custom timestamp (default: YYYYMMDD_HHMMSS)
    subdirs : list, optional
        Additional subdirectories to create
    base_name : str
        Base name for run folder type (default: fem_multiphysics)
        
    Returns
    -------
    Path
        Path to the created run directory
    """
    # Get repo root
    repo_root = get_repo_root(reference_file)
    
    # Generate timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build path: results/fem_multiphysics/run_TIMESTAMP/
    run_dir = repo_root / "results" / base_name / f"run_{timestamp}"
    
    # Required subdirectories
    required_subdirs = [
        "diagnostics",
        "mesh", 
        "figures",
        "fields",
        "logs",
    ]
    
    # Add any custom subdirs
    if subdirs is not None:
        required_subdirs.extend(subdirs)
    
    # Create all directories
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in required_subdirs:
        (run_dir / subdir).mkdir(exist_ok=True)
    
    return run_dir


def validate_run_dir(run_dir: Path) -> List[str]:
    """
    Validate that a run directory has all required artifacts.
    
    Parameters
    ----------
    run_dir : Path
        Path to run directory to validate
        
    Returns
    -------
    list
        List of missing required files/directories
    """
    required = [
        "config.json",
        "logs/run.log",
        "diagnostics/sanity_report.txt",
        "diagnostics/mesh_report.txt",
        "diagnostics/solver_report.txt",
    ]
    
    missing = []
    for item in required:
        if not (run_dir / item).exists():
            missing.append(item)
    
    return missing
