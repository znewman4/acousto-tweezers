#!/usr/bin/env python3
"""
Validation Test Suite Runner
=============================
Runs all validation micro-tests and reports summary results.

Tests:
1. test_acoustics_only.py - Full acoustic solver stack
2. test_pml_simple.py - PML wave absorption
3. test_interface_continuity.py - Solution smoothness

Run with: python scripts/validation/run_all_tests.py
"""

import subprocess
import sys
from pathlib import Path


def run_test(script_path: Path) -> tuple[bool, str]:
    """Run a test script and capture output."""
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per test
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: Test exceeded 2 minute limit"
    except Exception as e:
        return False, f"ERROR: {e}"


def main():
    print("=" * 70)
    print("ACOUSTO-TWEEZERS VALIDATION TEST SUITE")
    print("=" * 70)
    
    # Get validation scripts directory
    script_dir = Path(__file__).parent
    
    # Define tests
    tests = [
        ("Acoustic Solver Stack", script_dir / "test_acoustics_only.py"),
        ("PML Absorption", script_dir / "test_pml_simple.py"),
        ("Interface Continuity", script_dir / "test_interface_continuity.py"),
        ("Fluid-Solid Coupling", script_dir / "test_fluid_solid_coupled.py"),
    ]
    
    results = []
    
    for name, path in tests:
        print(f"\n{'─' * 70}")
        print(f"Running: {name}")
        print(f"Script: {path.name}")
        print(f"{'─' * 70}\n")
        
        if not path.exists():
            print(f"⚠ SKIP: Script not found")
            results.append((name, None, "Script not found"))
            continue
        
        success, output = run_test(path)
        
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
    
    passed = 0
    failed = 0
    skipped = 0
    
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
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Skipped: {skipped}/{len(results)}")
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
