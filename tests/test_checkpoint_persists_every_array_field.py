"""A dynamics-consuming field silently dropped by checkpoint is a correctness defect.

``edge_list.delay_steps`` was omitted from the persisted key set, so every restored
model came back with zero delays and different dynamics, with no error. The
structural test below fails for *any* future array field that is added to EdgeList
or IzhikevichParams and not persisted -- repairing the class of defect, not the
instance.
"""

from __future__ import annotations

from dataclasses import fields, replace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._model import Model
from jaxfne.emitters import EdgeList, IzhikevichParams

N = 120
DELAY = 3
# Persisted through the JSON sidecar rather than the npz, by design.
_META_PERSISTED = {"source_calibration_status", "labels", "layer_labels"}


def _config():
    return (
        jtfne.Configuration()
        .runtime(seed=1, dtype="float32", duration_ms=30.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=N)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(p_connect=0.0)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .mechanisms(name="ampa", kind="exponential", params={"tau_ms": 2.0, "receptor": "AMPA"})
        .connections(name="rec", source={}, target={}, mechanism="ampa", weight=0.03,
                     max_in_degree=10, spatial_sigma=0.1)
    )


def _delayed_model(cfg):
    model = jtfne.construct(cfg)
    edge_list = model.params["edge_list"]
    n_edges = int(edge_list.n_edges)
    return replace(model, params={**model.params, "edge_list": replace(
        edge_list, delay_steps=jnp.full(n_edges, DELAY, dtype=jnp.int32))})


def _saved_keys(tmp_path):
    cfg = _config()
    _delayed_model(cfg).checkpoint(str(tmp_path / "ck"))
    with np.load(tmp_path / "ck.npz") as archive:
        return set(archive.files)


@pytest.mark.parametrize(
    ("dataclass_type", "prefix"),
    [(EdgeList, "edge_"), (IzhikevichParams, "emitter_")],
)
def test_every_array_field_is_persisted(tmp_path, dataclass_type, prefix):
    """Structural: any array field not in the npz would silently restore as a default."""
    saved = _saved_keys(tmp_path)
    expected = {
        f"{prefix}{f.name}" for f in fields(dataclass_type)
        if f.name not in _META_PERSISTED
    }
    missing = sorted(expected - saved)
    assert not missing, (
        f"{dataclass_type.__name__} field(s) {missing} are not persisted by "
        "checkpoint(); a restored model would silently receive dataclass defaults"
    )


def test_delays_survive_a_checkpoint_round_trip(tmp_path):
    cfg = _config()
    model = _delayed_model(cfg)
    before = np.asarray(model.params["edge_list"].delay_steps)
    v_before = np.asarray(jtfne.simulate(model, duration_ms=30.0, dt_ms=0.5, seed=0).V_m)

    model.checkpoint(str(tmp_path / "ck"))
    restored = Model.restore(str(tmp_path / "ck"), _config())

    np.testing.assert_array_equal(before, np.asarray(restored.params["edge_list"].delay_steps))
    v_after = np.asarray(jtfne.simulate(restored, duration_ms=30.0, dt_ms=0.5, seed=0).V_m)
    np.testing.assert_array_equal(v_before, v_after)


def test_v1_checkpoints_still_restore(tmp_path):
    """v1 predates delay persistence: restore must accept it, defaulting to zeros."""
    import json

    cfg = _config()
    _delayed_model(cfg).checkpoint(str(tmp_path / "ck"))
    meta_path = tmp_path / "ck.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema"] = "model_checkpoint_v1"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with np.load(tmp_path / "ck.npz") as archive:
        without_delays = {k: archive[k] for k in archive.files if k != "edge_delay_steps"}
    np.savez(tmp_path / "ck.npz", **without_delays)

    restored = Model.restore(str(tmp_path / "ck"), _config())
    assert int(np.asarray(restored.params["edge_list"].delay_steps).max()) == 0


def test_unknown_schema_is_refused(tmp_path):
    import json

    cfg = _config()
    _delayed_model(cfg).checkpoint(str(tmp_path / "ck"))
    meta_path = tmp_path / "ck.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema"] = "model_checkpoint_v99"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown checkpoint schema"):
        Model.restore(str(tmp_path / "ck"), _config())
