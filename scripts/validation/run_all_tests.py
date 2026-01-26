#!/usr/bin/env python3
"""
Validation Test Suite Runner
=============================
Runs all validation micro-tests and reports summary results.

Tests are run in order:
0. test_env_complex_petsc.py - Environment gate (MUST PASS FIRST)
1. test_acoustics_smoke.py - Quick smoke test for Level 1
2. test_pml_smoke.py - PML validation smoke test
3. test_acoustics_only.py - Full acoustic solver stack
4. test_pml_simple.py - PML wave absorption
5. test_interface_continuity.py - Solution smoothness
6. test_fluid_solid_coupled.py - Coupled solver

Run with: python scripts/validation/run_all_tests.py
"""

import subprocess
import sys
from pathlib import Path


def run_test(script_path: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run a test script and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: Test exceeded {timeout}s limit"
    except Exception as e:
        return False, f"ERROR: {e}"


def run_env_gate(script_dir: Path) -> bool:
    """Run environment gate test. Returns True if passed."""
    env_test = script_dir / "test_env_complex_petsc.py"
    
    print("=" * 70)
    print("ENVIRONMENT GATE: Complex PETSc Check")
    print("=" * 70)
    print()
    
    if not env_test.exists():
        print("FATAL: Environment test not found!")
        print(f"Expected: {env_test}")
        return False
    
    success, output = run_test(env_test, timeout=30)
    print(output)
    
    if not success:
        print()
        print("=" * 70)
        print("ENVIRONMENT GATE FAILED")
        print("=" * 70)
        print()
        print("Cannot proceed with validation tests.")
        print("PETSc must be compiled with complex scalar support.")
        print()
        print("FIX: Install complex PETSc environment:")
        print()
        print("  micromamba env create -f environment/complex-fenicsx.yml")
        print("  micromamba activate acousto-complex")
        print()
        print("Then re-run this test suite.")
        print("=" * 70)
        return False
    
    print()
    return True


def main():
    print("=" * 70)
    print("ACOUSTO-TWEEZERS VALIDATION TEST SUITE")
    print("=" * 70)
    
    # Get validation scripts directory
    script_dir = Path(__file__).parent
    
    # ENVIRONMENT GATE - Must pass before other tests
    if not run_env_gate(script_dir):
        print()
        print("Passed: 0/1 (environment gate failed)")
        print("Failed: 1/1")
        return 1
    
    # Define tests (env gate already passed)
    tests = [
        ("Acoustics Smoke Test", script_dir / "test_acoustics_smoke.py", 180),
        ("PML Smoke Test", script_dir / "test_pml_smoke.py", 180),
        ("Acoustic Solver Stack", script_dir / "test_acoustics_only.py", 120),
        ("PML Absorption", script_dir / "test_pml_simple.py", 120),
        ("Interface Continuity", script_dir / "test_interface_continuity.py", 120),
        ("Fluid-Solid Coupling", script_dir / "test_fluid_solid_coupled.py", 180),
    ]
    
    results = []
    
    for name, path, timeout in tests:
        print(f"\n{'─' * 70}")
        print(f"Running: {name}")
        print(f"Script: {path.name}")
        print(f"{'─' * 70}\n")
        
        if not path.exists():
            print(f"⚠ SKIP: Script not found: {path.name}")
            results.append((name, None, "Script not found"))
            continue
        
        success, output = run_test(path, timeout=timeout)
        
        # Print abbreviated output (last 30 lines)
        lines = output.strip().split('\n')
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} lines omitted) ...\n")
            print('\n'.join(lines[-30:]))
        else:
            print(output)
        
        results.append((name, success, ""))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")
    
    # Count env gate as passed
    passed = 1  # env gate
    failed = 0
    skipped = 0
    total = len(results) + 1  # +1 for env gate
    
    print(f"  ✓ PASS  Environment Gate (Complex PETSc)")
    
    for name, success, note in results:
        if success is None:
            status = "⚠ SKIP"
            skipped += 1
        elif success:
            status = "✓ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"  {status}  {name}")
        if note:
            print(f"         {note}")
    
    print(f"\n{'─' * 70}")
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"  Skipped: {skipped}/{total}")
    print("=" * 70)
    
    # Exit with appropriate code
    if failed > 0:
        return 1
    elif passed == 0:
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
