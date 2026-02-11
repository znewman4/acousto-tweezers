# Cleanup Summary — 2026-02-10

## What Changed

### Documentation (Part A)
| Action | Count |
|--------|-------|
| Root `.md` / `.txt` files archived | 25 |
| `docs/` files and subdirs archived | 15 |
| New `CHANGELOG.md` written | 1 |
| New `README.md` written | 1 |
| New `docs/validation.md` written | 1 |

### Scripts (Part B)
| Destination | Scripts |
|-------------|---------|
| `scripts/validation/` | 22 test / validation scripts (unchanged) |
| `scripts/experiments/` | 10 experiment runners |
| `scripts/analysis/` | 5 analysis scripts + `render/` + `visualization/` |
| `scripts/dev/` | empty (placeholder for future dev tools) |
| Archived to `archive/scripts_old/` | 12 debug / duplicate / obsolete scripts |
| Deleted | 1 (`create_streaming_summary.py` — hardcoded data) |

### Results (Part C)
| Action | Details |
|--------|---------|
| Archived | ~30 result directories (~4.5 GB) |
| Retained | `results/deposition_20260210_203721/` (latest run) |

### Archive (Part D)
```
archive/
├── notes.md                 # manifest of everything moved
├── redundant_docs/          # old markdown files
│   ├── docs/                # old docs/ subdirectories
│   ├── CHANGELOG_old.md
│   └── README_old.md
├── results/                 # old result directories
└── scripts_old/             # debug / duplicate scripts
    ├── legacy_scripts/      # adjoint, MPC, FD, early FEM
    └── setup/               # environment setup scripts
```

## What Did NOT Change

- **No physics code was modified.** All files under `src/acoustweezers/` are
  byte-identical to their pre-cleanup state.
- No solver parameters, algorithms, or numerical results were altered.
- The `environment.yml` and `pyproject.toml` are untouched.

## Final Checks (Part E)

| Check | Result |
|-------|--------|
| `python -m compileall src scripts` | ✓ PASS — no syntax/import errors |
| `test_1d_impedance.py` | ✓ PASS — Tests 1–2 match (|R|≈0 matched, |R|=1 rigid) |
| `test_energy_balance.py` | ✓ PASS — rigid-wall analytical match (|p| error ≈ 9e-11) |
| `test_petri_dish_bcs.py` | ✓ PASS — all 4 modes (standing 58 Pa, vortex 11 Pa, combined 55 Pa) |

All checks passed.  No physics were changed.
