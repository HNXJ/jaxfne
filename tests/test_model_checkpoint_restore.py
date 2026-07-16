"""Regression tests for Model.checkpoint()/Model.restore() (plans.json:P3b).

Direct dataclass reconstruction (not JAX treedef unflatten) -- the same
verified-safe pattern as
scripts/cortical_column_localized_workflow.py::save_column/load_column, now
promoted to real Model methods. See skills/jaxfne-neural-tensor/SKILL.md's
"Construct-once / checkpoint / reload" section for the two landmines this
avoids: a differently-sized dummy treedef silently producing a structurally
mismatched model, and reusing a pre-construct cfg.metadata that construct()
has since mutated in place.
"""
import numpy as np
import jaxfne as jtfne
from jaxfne._model import Model


def _build_cfg():
    return (
        jtfne.configuration()
        .network(n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )


def test_checkpoint_restore_round_trip_is_bit_identical(tmp_path):
    cfg = _build_cfg()
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=0)
    signals_before = model.simulate(sim)

    path = str(tmp_path / "model_checkpoint")
    returned_path = model.checkpoint(path)
    assert returned_path is not None

    fresh_cfg = _build_cfg()  # cheap, no construct() call
    restored = Model.restore(path, fresh_cfg)

    np.testing.assert_array_equal(
        np.asarray(restored.params["emitter"].W), np.asarray(model.params["emitter"].W)
    )
    np.testing.assert_array_equal(
        np.asarray(restored.params["edge_list"].pre), np.asarray(model.params["edge_list"].pre)
    )
    assert restored.cfg.metadata == model.cfg.metadata
    assert restored.static == model.static

    signals_after = restored.simulate(sim)
    np.testing.assert_array_equal(np.asarray(signals_before.V_m), np.asarray(signals_after.V_m))
    np.testing.assert_array_equal(np.asarray(signals_before.spikes), np.asarray(signals_after.spikes))


def test_restore_rejects_unknown_schema(tmp_path):
    cfg = _build_cfg()
    model = jtfne.construct(cfg)
    path = tmp_path / "model_checkpoint"
    model.checkpoint(str(path))

    import json
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["schema"] = "some_other_schema_v99"
    meta_path.write_text(json.dumps(meta))

    try:
        Model.restore(str(path), _build_cfg())
        assert False, "expected ValueError for unknown schema"
    except ValueError as e:
        assert "schema" in str(e).lower()
