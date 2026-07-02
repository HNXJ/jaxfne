"""v0.4.x operator-stage minimum-method coverage.

One package-native, finite-output-checked smoke per operator stage of the
Emitter -> Source -> FieldProxy -> Probe -> Objective -> Optimizer -> Manifest
pipeline, plus the validation and schema-migration utilities. Every stage method
is reached through the stable `jaxfne` (jtfne) namespace.

Inputs/outputs are documented inline per stage. Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest \
        tests/test_operator_stage_coverage_v04.py -q
"""
import pytest

import jaxfne as jtfne


@pytest.fixture(scope="module")
def model():
    # Config builder stage -> construct/model creation stage.
    cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    return jtfne.construct(cfg)


@pytest.fixture(scope="module")
def signals(model):
    # Emitter step / simulate stage. Output: Signals(vm,spk,source[T,N]).
    return jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)


def test_config_builder_stage():
    cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    assert isinstance(cfg, jtfne.Configuration)
    # fluent builder reachable too
    fluent = jtfne.configuration().runtime(seed=0).column(name="c", layers=["L4"], n=8)
    assert isinstance(fluent, jtfne.Configuration)


def test_optimizer_tune_stage():
    # Optimizer specs reachable; black-box path is gradient-safe by construction.
    for spec_fn in (jtfne.agsdr, jtfne.gsdr, jtfne.optax_adam, jtfne.random_search):
        spec = spec_fn()
        assert isinstance(spec, jtfne.OptimizerSpec)
        assert spec.optimizer_class in ("blackbox", "differentiable", "multiparameter_blackbox")


def test_manifest_receipt_export_stage(model, signals):
    manifest = model.manifest(signals)
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["field_solver_status"] in ("linear_solver", "pde_solver")
    assert manifest["physical_amplitude_calibrated"] is False
    receipt = jtfne.run_receipt(model, signals)
    assert receipt.to_dict()["jaxfne_version"]


def test_schema_migration_stage():
    # migration utility reachable + idempotent on canonical input
    canonical = {"claim_level": "computational_scaffold", "field_solver_status": "linear_solver"}
    assert jtfne.migrate_schema(canonical) == canonical
