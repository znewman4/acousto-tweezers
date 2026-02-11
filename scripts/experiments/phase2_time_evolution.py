#!/usr/bin/env python3
"""
Phase 2: Time Evolution with Particle Dynamics
Thin entrypoint - main logic in acoustweezers.experiments.square_dish.time_evolution
"""
import sys
from pathlib import Path

# Ensure src is in path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Import and run main from module
from acoustweezers.experiments.square_dish.time_evolution import main

if __name__ == "__main__":
    main()
