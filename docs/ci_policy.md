# CI and Validation Policy

## Overview

jaxfne uses a two-tier validation strategy:

1. **Fast CI** (runs on every push/PR): Smoke tests, quick examples, build validation (~1 min)
2. **Extended validation** (manual/release): Full tutorials, large simulations (~5-10 min)

This keeps development fast while ensuring comprehensive testing before release.

## Fast CI Gate

### What runs (`.github/workflows/ci.yml`)

- Python 3.10, 3.11, 3.12 matrix
- Compilation check: `python -m compileall -q jaxfne tests examples`
- Core tests: 903 pytest tests (5 skipped, expected)
- Fast examples only: `examples/00-06` (smoke tests, ~1 min)
- Build: wheel + sdist
- Smoke test: fresh venv wheel install + minimal workflow

### What is excluded from fast CI

Large, long-running examples:

- `examples/02_spectrolaminar_oddball_scaffold.py` (~3 min)
- `examples/03_single_neuron_multimodal_probe.py` (~3 min)
- `examples/04_two_neuron_ei_multimodal.py` (~3 min)
- `examples/05_network_100_ei_multimodal.py` (~5 min)
- `examples/07_jaxley_trace_bridge.py` (synthetic, <10 sec but grouped with large examples)

### Why excluded

Large examples are not blockers for fast iteration:

- Developers often work on small focal changes
- Large simulations don't parallelize with other CI jobs
- Testing all large examples on every push would add 10+ min to CI
- These examples validate tutorial outputs, not core functionality

### Known subprocess test behaviors

Two tests spawn large examples as subprocesses:

- `test_network_100_ei_colab_v0210.py::test_example_script_runs`
- `test_two_neuron_ei_colab_v029.py::test_example_script_runs`

These fail in local dev without release-validation environment setup. They should be marked `@pytest.mark.release_validation` to exclude from default pytest runs. See [Release validation](#release-validation) for manual execution.

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
- Checks validation gates (computational_scaffold, physical_amplitude_claim_allowed=False)
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

## Truth Status and Claim Gates

All outputs maintain strict validation gates:

```yaml
truth_mode: truth_safe_unverified
computational_scaffold: true
physical_amplitude_claim_allowed: false
field_solver_status: laminar_proxy_no_pde (or not_computed)
```

These gates are enforced in validation scripts and cannot be escalated without explicit approval and corresponding evidence.

## Performance Benchmarking (v0.2.30+)

### Benchmark scope

jaxfne v0.2.30 includes deterministic performance benchmarking scripts (not CI gates):

- `scripts/benchmark_jaxfne.py`: Measures 7 computational phases with hardware metadata
- `scripts/validate_json_safe.py`: Detects NaN/Infinity in JSON outputs

### Benchmark claim boundaries

**Important:** Benchmarks are `local_environment_receipt_only`. No universal performance claims are made:

- ✓ Local timing receipts (CPU/platform specific)
- ✓ Timing model and scaling analysis
- ✗ Comparative performance vs. other simulators
- ✗ GPU/TPU acceleration factors
- ✗ Real-time performance or biological correspondence

See [Performance Baseline](performance_baseline.md) for detailed claims and reproduction instructions.

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
- This file explains smoke-safe CI and large example exclusion
- `README.md` links users to this page
- Large examples are documented as manual-validation in README

See the "Extended Validation (Manual/Release)" section above for manual validation commands.

## Future Improvements

Potential enhancements (not yet implemented):

- Mark long subprocess tests with `@pytest.mark.release_validation`
- Add `--release-validation` flag to pytest to run excluded markers
- Link validation GitHub Actions workflow for scheduled extended testing
- Automated artifact comparison (before/after) for regression detection
