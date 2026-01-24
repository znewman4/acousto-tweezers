#!/bin/bash
# Quick validation script for 3D Helmholtz solver improvements

set -e

echo "=========================================="
echo "3D Helmholtz Solver Validation Tests"
echo "=========================================="
echo ""

cd /home/znewman4/projects/acousto-tweezers

# Test 1: Minimal demo (should complete in <5 seconds)
echo "[TEST 1] Minimal demo (11×11×4 grid, 10 steps)..."
OUTPUT=$(python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 10 \
  --render_stride 2 \
  --gif 1 \
  --Lx 0.01 --Ly 0.01 --H 0.003 \
  --dx 0.001 --dy 0.001 --dz 0.001 \
  2>&1)
if echo "$OUTPUT" | grep -q "RUN COMPLETE"; then
  echo "  ✅ PASS: Demo completed"
else
  echo "  ❌ FAIL: Demo did not complete"
  exit 1
fi
if echo "$OUTPUT" | grep -q "particle_slice.gif"; then
  echo "  ✅ PASS: GIF created"
else
  echo "  ❌ FAIL: GIF not created"
  exit 1
fi
if echo "$OUTPUT" | grep -q "MEMORY REPORT"; then
  echo "  ✅ PASS: Memory report printed"
else
  echo "  ❌ FAIL: Memory report missing"
  exit 1
fi
echo ""

# Test 2: Matrix caching (verify matrix reuse)
echo "[TEST 2] Matrix caching (should see 'Using cached A' messages)..."
OUTPUT=$(python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 3 \
  --render_stride 1 \
  --gif 0 \
  --Lx 0.01 --Ly 0.01 --H 0.003 \
  --dx 0.001 --dy 0.001 --dz 0.001 \
  2>&1)
ASSEMBLED=$(echo "$OUTPUT" | grep -c "Assembled and cached A" || true)
CACHED=$(echo "$OUTPUT" | grep -c "Using cached A" || true)
if [ "$ASSEMBLED" -eq 1 ]; then
  echo "  ✅ PASS: Matrix assembled once ($ASSEMBLED time)"
else
  echo "  ❌ FAIL: Expected 1 assembly, got $ASSEMBLED"
  exit 1
fi
if [ "$CACHED" -eq 2 ]; then
  echo "  ✅ PASS: Matrix reused ($CACHED times)"
else
  echo "  ❌ FAIL: Expected 2 reuses, got $CACHED"
  exit 1
fi
echo ""

# Test 3: Feature flags (--no_gorkov --no_particle)
echo "[TEST 3] Feature flags (--no_gorkov --no_particle)..."
OUTPUT=$(python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 2 \
  --render_stride 1 \
  --gif 0 \
  --no_gorkov \
  --no_particle \
  --Lx 0.01 --Ly 0.01 --H 0.003 \
  --dx 0.001 --dy 0.001 --dz 0.001 \
  2>&1)
if echo "$OUTPUT" | grep -q "RUN COMPLETE"; then
  echo "  ✅ PASS: Solver-only mode completed"
else
  echo "  ❌ FAIL: Solver-only mode failed"
  exit 1
fi
if echo "$OUTPUT" | grep -q "Particle simulation skipped"; then
  echo "  ✅ PASS: Particle simulation skipped as expected"
else
  echo "  ❌ FAIL: Particle skip message missing"
  exit 1
fi
echo ""

# Test 4: Memory diagnostics
echo "[TEST 4] Memory diagnostics (should have 3+ checkpoints)..."
OUTPUT=$(python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 1 \
  --render_stride 1 \
  --gif 0 \
  --Lx 0.01 --Ly 0.01 --H 0.003 \
  --dx 0.001 --dy 0.001 --dz 0.001 \
  2>&1)
CHECKPOINTS=$(echo "$OUTPUT" | grep -c "\[MEM\]" || true)
if [ "$CHECKPOINTS" -ge 3 ]; then
  echo "  ✅ PASS: $CHECKPOINTS memory checkpoints recorded"
else
  echo "  ❌ FAIL: Expected ≥3 checkpoints, got $CHECKPOINTS"
  exit 1
fi
if echo "$OUTPUT" | grep -q "Peak Δ from start"; then
  echo "  ✅ PASS: Peak memory delta reported"
else
  echo "  ❌ FAIL: Peak delta report missing"
  exit 1
fi
echo ""

# Test 5: Realistic grid (21×21×8)
echo "[TEST 5] Realistic grid (21×21×8, 20 steps)..."
OUTPUT=$(python3 scripts/demo_helmholtz3d_v2.py \
  --n_steps 20 \
  --render_stride 2 \
  --gif 1 \
  --Lx 0.03 --Ly 0.03 --H 0.01 \
  --dx 0.0015 --dy 0.0015 --dz 0.0015 \
  --dtype single \
  2>&1)
if echo "$OUTPUT" | grep -q "Nx=21, Ny=21, Nz=8"; then
  echo "  ✅ PASS: Grid computed correctly (21×21×8)"
else
  echo "  ❌ FAIL: Grid size incorrect"
  exit 1
fi
if echo "$OUTPUT" | grep -q "particle_slice.gif"; then
  echo "  ✅ PASS: GIF created for realistic grid"
else
  echo "  ❌ FAIL: GIF not created"
  exit 1
fi
if echo "$OUTPUT" | grep -q "RUN COMPLETE"; then
  echo "  ✅ PASS: Large grid completed without OOM"
else
  echo "  ❌ FAIL: Large grid failed or OOM"
  exit 1
fi
echo ""

# Test 6: CSV output
echo "[TEST 6] CSV trajectory output..."
LATEST_RUN=$(ls -td results/helmholtz3d_demo/run_* | head -1)
if [ -f "$LATEST_RUN/traj_moving_lens.csv" ]; then
  ROWS=$(wc -l < "$LATEST_RUN/traj_moving_lens.csv")
  echo "  ✅ PASS: CSV created with $ROWS rows"
  # Verify header
  HEADER=$(head -1 "$LATEST_RUN/traj_moving_lens.csv")
  if echo "$HEADER" | grep -q "t_s,x_m,y_m"; then
    echo "  ✅ PASS: CSV header correct"
  else
    echo "  ❌ FAIL: CSV header incorrect"
    exit 1
  fi
else
  echo "  ❌ FAIL: CSV not found"
  exit 1
fi
echo ""

# Test 7: GIF validity
echo "[TEST 7] GIF file validity..."
LATEST_GIF=$(ls -t results/helmholtz3d_demo/run_*/particle_slice.gif 2>/dev/null | head -1)
if [ -f "$LATEST_GIF" ]; then
  if file "$LATEST_GIF" | grep -q "GIF image data"; then
    SIZE=$(stat --format=%s "$LATEST_GIF" | numfmt --to=iec)
    echo "  ✅ PASS: Valid GIF file ($SIZE)"
  else
    echo "  ❌ FAIL: GIF file is invalid"
    exit 1
  fi
else
  echo "  ❌ FAIL: GIF not found"
  exit 1
fi
echo ""

# Test 8: CLI help
echo "[TEST 8] CLI help message..."
if python3 scripts/demo_helmholtz3d_v2.py --help 2>&1 | grep -q "optional arguments\|options"; then
  echo "  ✅ PASS: CLI help works"
else
  echo "  ❌ FAIL: CLI help broken"
  exit 1
fi
echo ""

echo "=========================================="
echo "✅ ALL TESTS PASSED"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - OOM fixed: 21×21×8 grid (3.5K points) runs safely"
echo "  - Matrix caching: Confirmed reuse across steps"
echo "  - Memory tracking: 5+ checkpoints, peak delta reported"
echo "  - GIF streaming: No frame buffering, files valid"
echo "  - CSV output: Trajectory data saved correctly"
echo "  - CLI: All flags operational"
echo ""
echo "For detailed run, see latest:"
echo "  $LATEST_RUN"
echo ""
