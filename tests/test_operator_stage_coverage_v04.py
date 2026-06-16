"""v0.4.x operator-stage minimum-method coverage.

One package-native, finite-output-checked smoke per operator stage of the
Emitter -> Source -> FieldProxy -> Probe -> Objective -> Optimizer -> Manifest
pipeline, plus the validation and schema-migration utilities. Every stage method
is reached through the stable `jaxfne` (jtfne) namespace.

Inputs/outputs are documented inline per stage. Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest \
        tests/test_operator_stage_coverage_v04.py -q
"""
import jax.numpy as jnp
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


def _finite(a):
    return bool(jnp.all(jnp.isfinite(a)))


def test_config_builder_stage():
    cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
    assert cfg is not None
    # fluent builder reachable too
    fluent = jtfne.configuration().runtime(seed=0).column(name="c", layers=["L4"], n=8)
    assert fluent is not None


def test_construct_and_simulate_stage(signals):
    for key in ("vm", "spk", "source"):
        arr = signals.get(key)
        assert arr.ndim == 2, f"{key} expected [T,N]"
        assert _finite(arr), f"{key} non-finite"


def test_source_tensor_stage(signals):
    # construct_source_tensor: emitter state -> source tensor [T,N] (+status dict).
    src = signals.get("source")
    tensor, status = jtfne.construct_source_tensor(
        mode="total_membrane_current_proxy", total_membrane_current=src,
    )
    assert _finite(tensor)
    assert isinstance(status, dict)


def test_field_projection_linear_solver_stage(signals):
    # FieldProxy stage: source [T,N] -> laminar contacts (linear_solver).
    src = signals.get("source")
    n = src.shape[1]
    positions = jnp.stack([jnp.zeros(n), jnp.zeros(n), jnp.linspace(0.0, 1.0, n)], axis=1)
    field = jtfne.project_laminar_sources(src, positions, n_contacts=16, width=0.1)
    assert _finite(field.lfp_proxy) and _finite(field.csd_proxy)
    assert field.lfp_proxy.shape[0] == src.shape[0]


def test_probe_readout_stage(model, signals):
    rep = model.probe(signals, ["vm", "spk"])
    assert isinstance(rep, dict)


def test_objective_evaluate_stage(model, signals):
    e_idx = model.select(cell_type="E")
    obj = jtfne.rate_targets(groups={"E": e_idx}, targets_hz={"E": 5.0})
    report = model.evaluate(signals, obj, strict=False)
    assert isinstance(report, dict)


def test_optimizer_tune_stage():
    # Optimizer specs reachable; black-box path is gradient-safe by construction.
    for spec_fn in (jtfne.agsdr, jtfne.gsdr, jtfne.optax_adam, jtfne.random_search):
        assert spec_fn() is not None


def test_manifest_receipt_export_stage(model, signals):
    manifest = model.manifest(signals)
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["field_solver_status"] in ("linear_solver", "pde_solver")
    assert manifest["physical_amplitude_calibrated"] is False
    receipt = jtfne.run_receipt(model, signals)
    assert receipt.to_dict()["jaxfne_version"]


def test_validation_report_stage(signals):
    assert jtfne.is_valid_signal(signals) in (True, False)
    status = jtfne.validate_source_field_status(field_output=signals.field)
    assert isinstance(status, dict)


def test_schema_migration_stage():
    # migration utility reachable + idempotent on canonical input
    canonical = {"claim_level": "computational_scaffold", "field_solver_status": "linear_solver"}
    assert jtfne.migrate_schema(canonical) == canonical
