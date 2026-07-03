# CI and Validation Policy

## Overview

jaxfne uses a two-tier validation scope:

1. **Fast CI** (runs on every push/PR): Smoke tests, quick examples, build validation (~1 min)
2. **Extended validation** (manual/release): Full tutorials, large simulations (~5-10 min)

This keeps development fast while ensuring comprehensive testing before release.

## Fast CI Gate

### What runs (`.github/workflows/ci.yml` - Fast PR/Dev CI)

- Python 3.10, 3.12 matrix (optimized routine development matrix)
- Compilation check: `python -m compileall -q jaxfne tests examples`
- Core tests: 2759 pytest tests collected (`python -m pytest tests/ --collect-only -q`), run with
  `-m "not slow" --ignore=tests/test_multi_area_source_projector.py --ignore=tests/test_multi_area_spectrolaminar_objective.py`
  on non-`main` pushes/PRs (`.github/workflows/ci.yml:34`); the two ignored files and the `slow`-marked
  tests still run on pushes to `main` (`ci.yml:41`)
- Examples: **all** `examples/00-08` scripts listed in the "Run examples" step (`ci.yml:46-54`) are
  executed on every fast-CI run, not a subset
- Build: wheel + sdist
- Smoke test: fresh venv wheel install + minimal workflow

### What runs (`.github/workflows/release_ci.yml` - Release & Scheduled CI)

- Python 3.10, 3.11, 3.12 matrix (comprehensive version-compatibility verification)
- Full core/slow test runs and build validations
- Triggers on pushes/PRs to `main`, manual dispatch, and nightly scheduled cron


### What is excluded from fast CI

As of this writing, `.github/workflows/ci.yml`'s "Run examples" step runs every example listed
there (`examples/00-08`, including the spectrolaminar, jaxley-bridge, and 100-neuron examples) on
every fast-CI run — there is currently **no** large-example exclusion list in fast CI. The two
`--ignore`d test files above (`test_multi_area_source_projector.py`,
`test_multi_area_spectrolaminar_objective.py`) and `slow`-marked tests are the only things skipped
on non-`main` runs; notebook execution (`notebook`-marked tests) is excluded from `ci.yml`
entirely and instead handled by the separate `notebook_execution.yml` workflow.

### Why some things are still excluded

- `notebook`-marked tests (28 release-facing notebooks) are handled by the dedicated
  `notebook_execution.yml` workflow instead of every push — they can take well over an hour combined
- The two `--ignore`d test files above are excluded from every `ci.yml` run (not gated by branch)
- `slow`-marked tests run only on `main`, keeping non-`main` iteration fast

### Known subprocess test behaviors

Two tests spawn large examples as subprocesses:

- `test_network_100_ei_colab_v0210.py::test_example_script_runs`
- `test_two_neuron_ei_colab_v029.py::test_example_script_runs`

These fail in local dev without release-validation environment setup. They should be marked `@pytest.mark.release_validation` to exclude from default pytest runs. See [Extended validation](#extended-validation-manualrelease) for manual execution.

## Extended Validation (Manual/Release)

### Run all large examples

```bash
python scripts/run_all_tutorials.py --smoke --write-figures --out-root outputs/
python scripts/validate_tutorial_outputs.py outputs/
```

This:
- Executes all large examples (02-07)
- Generates static PNG figures and metadata
- Validates output contracts (manifest.json, metrics.json, etc.)
- Checks validation gates (computational_scaffold, amplitude_status=False)
- Regenerates artifact hashes

Expected runtime: 5-10 minutes (CPU-only, JAX on CPU).

### Release validation requirements

Before tagging a release:

1. Run core validation: `python -m pytest tests/ -q --tb=line` (must pass)
2. Run extended validation: `scripts/run_all_tutorials.py && scripts/validate_tutorial_outputs.py`
3. Verify build: `python -m build && twine check dist/*`
4. Verify wheel smoke: fresh venv install + minimal workflow (checked in CI)
5. Update CHANGELOG.md with version and summary
6. Tag and push

## Status Status and Statement Gates

All outputs maintain strict validation gates:

```yaml
run_status: tutorial_scaffold
computational_scaffold: true
amplitude_status: false
field_solver_status: linear_solver (or not_computed)
```

These gates are enforced in validation scripts and cannot be escalated without explicit approval and corresponding evidence.

## Performance Benchmarking (v0.2.30+)

### Benchmark scope

jaxfne v0.2.30 includes deterministic performance benchmarking scripts (not CI gates):

- `scripts/benchmark_jaxfne.py`: Measures 7 computational phases with hardware metadata
- `scripts/validate_json_safe.py`: Detects NaN/Infinity in JSON outputs

### Benchmark statement boundaries

**Important:** Benchmarks are `local_environment_receipt_only`. No universal performance statements are made:

- ✓ Local timing receipts (CPU/platform specific)
- ✓ Timing model and scaling analysis
- ✗ Comparative performance vs. other simulators
- ✗ GPU/TPU acceleration factors
- ✗ Real-time performance or biological correspondence

See [Performance Baseline](performance_baseline.md) for detailed statements and reproduction instructions.

### Running benchmarks manually

```bash
# Generate benchmark report
python scripts/benchmark_jaxfne.py
# Output: outputs/benchmarks_v030/benchmark_report.json

# Validate JSON safety
python scripts/validate_json_safe.py
# Output: outputs/json_validation_report.json
```

No automatic CI gates are applied to benchmark results. Measurements serve as reproducible documentation, not performance requirements.

## Documentation and Policy Updates

CI policy documentation:
- This file explains smoke-safe CI and the current test/example scope
- `README.md` does not currently link to this file (verified via `grep -in ci_policy README.md`,
  no match); if a stable cross-reference is wanted, add one under README's docs links

See the "Extended Validation (Manual/Release)" section above for manual validation commands.

## Reserved Improvements

Potential enhancements (not yet implemented):

- Mark long subprocess tests with `@pytest.mark.release_validation`
- Add `--release-validation` flag to pytest to run excluded markers
- Link validation GitHub Actions workflow for scheduled extended testing
- Automated artifact comparison (before/after) for regression detection
