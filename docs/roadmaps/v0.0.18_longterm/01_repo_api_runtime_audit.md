# Repo, API, and Runtime Audit (v0.0.18)

## 1. README versus Current API
- **Stale:** README references v0.0.5 API surface but lacks clear examples for `RunReceipt`, `ReadoutSpec`, and `.jcfg.json` loading introduced in v0.0.15–v0.0.18.
- **Drift:** `model.manifest()` is used in README but `Model.run_receipt()` is the newer canonical method for gathering signals and readouts.

## 2. Pyproject versus Runtime Version
- **Status:** Synchronized at `0.0.18`.
- **Location:** `pyproject.toml` and `jaxfne/__init__.py`.

## 3. Editable Install Metadata
- **Status:** Stale. Reports `0.0.3`.
- **Reason:** Local environment drift; requires `pip install -e .` to refresh.

## 4. Optional Dependency Guards
- **Verified:** `require_optax()` and `require_jaxley()` are implemented and correctly guarded in `jaxfne/bridges.py` and `jaxfne/optim.py`.
- **Usage:** Correctly used in `core.py` and examples.

## 5. Package Layout
- **Status:** Preserved minimal grammar.
- **Modules:** `core.py`, `emitters.py`, `fields.py`, `objectives.py`, `optim.py`, `runtime.py`, `io.py`, `validation.py`, `bridges.py`.

## 6. Public API Exports
- **Verified:** `__init__.py` exports key classes and factory functions.
- **Consistency:** High.

## 7. Examples
- **Verified:** 7 examples in `examples/` directory.
- **Status:** `minimal_eig_column.py` and `06_edge_list_recurrent_backend.py` are the primary smoke targets.

## 8. Tests
- **Status:** 178 tests total.
- **Result:** 176 passed, 2 failed (environment-related import issues in sub-processes for examples).

## 9. Manifest and Runtime Reporting
- **Verified:** `JaxFNEConfig` and `RunReceipt` provide JSON-safe manifests.
- **Metadata:** `truth_mode`, `claim_level`, and `source_calibration_status` are correctly propagated.

## 10. Known Risks before Merge
- **Metadata Drift:** Stale pip metadata might cause confusion during deployment.
- **Test Failures:** The 2 failing example tests should be resolved by setting `PYTHONPATH` in the test runner.
- **Documentation:** README needs a minimal patch to reflect the `v0.0.18` state.
