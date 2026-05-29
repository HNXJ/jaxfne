# v0.3.14 Release Preflight Validation Report

**Date**: 2026-05-29  
**Branch**: agy  
**Status**: ✅ VALIDATED  
**Decision Label**: `jaxfne_v0_3_14_release_ready_after_three_branch_validation`

## Phase v0.3.14 Summary

This phase implements the final stabilization and release preflight gates for `v0.3.14`, including package-native population ablations and synchrony controls, targeted unit test suites, PyPI package building, and twine compliance validation.

### Commit 13: Package-Native Nulls & Ablations Configuration

**Status**: ✅ COMPLETE

**Implementation**:
- `Simulation(ablation=...)` support added in `jaxfne/core.py` and `jaxfne/emitters.py`
- Supported configurations:
  - `E_silence`: Clamps excitatory neuron membrane potentials to the reset level `c` and silences their spikes.
  - `I_silence`: Clamps inhibitory neuron membrane potentials to the reset level `c` and silences their spikes.
  - `disconnected_null`: Dynamically severs all recurrent connections by zeroing out the weights matrix `W` (for dense networks) or edge list weights (for sparse networks).
  - `shuffled_timing`: Destroys spike coordination by shuffling the driving schedule array along the temporal axis independently for each neuron using vectorized, PRNGKey-driven `jax.random.permutation` inside a deterministic `jax.vmap`.

### Commit 14: Ablation & Control Validation Tests

**Status**: ✅ 4/4 TESTS PASS

**Test Breakdown**:
- `test_ablation_e_silence`: Verifies all excitatory neurons emit exactly zero spikes, while leaving inhibitory neurons free to fire under the same drive schedule.
- `test_ablation_i_silence`: Verifies all inhibitory neurons emit exactly zero spikes, while leaving excitatory neurons active.
- `test_ablation_disconnected_null`: Confirms that the network runs cleanly and all outputs remain finite when synaptic weights are zeroed.
- `test_ablation_shuffled_timing`: Verifies that stimulus drive schedules are correctly shuffled along the time axis to destroy temporal synchrony.

### Commit 15: PyPI Rehearsal Packaging Validation

**Status**: ✅ COMPLETE

**Verification**:
- Sdist and Wheel successfully compiled:
  - `jaxfne-0.3.14.tar.gz`
  - `jaxfne-0.3.14-py3-none-any.whl`
- `twine check dist/*` returned `PASSED` for all builds.
- Editable virtual environment installation verified with fresh imports.

### Commit 16: Launch, Tagging, and final Report

**Status**: ✅ COMPLETE

- Package version bumped to `"0.3.14"` in `pyproject.toml` and `jaxfne/core.py` (`_JAXFNE_VERSION`).
- Completed preflight reports and merged validated workspace across branches.

### Architectural Separation: paradigm.py Migration

**Status**: ✅ COMPLETE

- Extracted `ParadigmEvent`, `ParadigmCondition`, and `Paradigm` classes, along with `paradigm()`, `evoked_l4_drive_paradigm()`, and `omission_oddball_paradigm()` functions, from `jaxfne/core.py` into a dedicated `jaxfne/paradigm.py` module.
- Retained seamless backward-compatibility by importing all paradigm classes and creator functions back into `jaxfne/core.py` from `.paradigm`.
- Verified 100% green status on all 29 targeted paradigm, evoked-drive, and omission test suites.

---

## Quality Gates

* ✅ **Package-Native API**: Ablation features are implemented natively inside the core simulation loops.
* ✅ **Architectural Decoupling**: Large task-script and paradigm containers have been clean-decoupled out of `jaxfne/core.py` into `jaxfne/paradigm.py`.
* ✅ **Proxy-Safe Wording**: Maintained uncalibrated, unphysical status defaults throughout. Wording avoids any biological active inference, real EEG/MEG, or biophysical metabolism claims.
* ✅ **JSON Safety**: All reports/manifests are strictly serializable with `allow_nan=False`.
* ✅ **Finite Outputs**: Simulation outputs contain no `NaN` or `Inf` values under all ablation controls.

## Files Modified

**New Files**:
- `jaxfne/paradigm.py`
- `tests/test_v0314_ablation_controls.py`
- `phase_validation/v0_3_14_release_preflight_report.md`
- `phase_validation/v0_3_14_release_preflight_report.json`

**Modified Files**:
- `pyproject.toml`
- `jaxfne/core.py`
- `jaxfne/emitters.py`
