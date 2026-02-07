"""
Logging utilities for the acousto-tweezers package.

Provides consistent logging setup across all modules.

Author: Acousto-Tweezers Project
Date: January 2026
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    output_dir: Optional[Path] = None,
    log_filename: str = "run.log",
    verbose: bool = True,
    quiet: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logging to file and/or console.
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory for log file. If None, no file logging.
    log_filename : str
        Name of log file (default: run.log)
    verbose : bool
        If True, also log to console
    quiet : bool
        If True, suppress console output (overrides verbose)
    level : int
        Logging level
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Get root logger for the package
    logger = logging.getLogger("acousto_tweezers")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if output_dir is not None:
        log_path = Path(output_dir) / "logs" / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if verbose and not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the acousto_tweezers prefix.
    
    Parameters
    ----------
    name : str
        Module name (e.g., 'solver', 'acoustics')
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(f"acousto_tweezers.{name}")
