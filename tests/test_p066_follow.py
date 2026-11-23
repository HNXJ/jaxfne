"""P-066 follow-up (T-P066-FOLLOW): connect() height carry and refusals, the PV<->E gate's depth
unit, and simulate(record_fields=True) forwarding column_height_mm to the field projection.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.fields import project_laminar_sources

RUNTIME = dict(seed=1, duration_ms=1.0, dt_ms=0.5)


def _finish(cfg):
    return (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .runtime(**RUNTIME)
    )


def _laminar(name, n, height_mm):
    return _finish(jtfne.build_laminar_column(name, n, geometry="laminar", height_mm=height_mm))


def _canonical(canonical_on, height_mm=2.0):
    cfg = _finish(jtfne.build_laminar_column("V1", 160, ei_profile="canonical", height_mm=height_mm))
    return cfg.runtime(canonical_biophysics=canonical_on)


def _uniform(canonical_on):
    cfg = _finish(jtfne.build_laminar_column("V1", 160, geometry="uniform3d"))
    return cfg.runtime(canonical_biophysics=canonical_on)


def _pv_e_mask(model):
    lab = np.asarray(model.params["emitter"].labels)
    pre = np.asarray(model.params["edge_list"].pre)
    post = np.asarray(model.params["edge_list"].post)
    pc, qc = lab[pre], lab[post]
    return pre, post, ((pc == "PV") & (qc == "E")) | ((pc == "E") & (qc == "PV"))


def _check_pv_e_gate(model_on, model_off, scale):
    """Weights: on = off * (1 + 2*exp(-((|dz|/scale)/0.15)^2)) on PV<->E edges, off elsewhere."""
    pre, post, pv_e = _pv_e_mask(model_on)
    pre_o, post_o, _ = _pv_e_mask(model_off)
    assert np.array_equal(pre, pre_o) and np.array_equal(post, post_o), "fixture: on/off edge lists differ"
    assert int(pv_e.sum()) > 0, "fixture must have PV<->E edges; the gate check would pass empty"
    pos = np.asarray(model_on.params["positions"], dtype=float)
    assert np.array_equal(pos, np.asarray(model_off.params["positions"], dtype=float)), "fixture: positions differ"
    z = pos[:, 2]
    w_on = np.asarray(model_on.params["edge_list"].weight, dtype=float)
    w_off = np.asarray(model_off.params["edge_list"].weight, dtype=float)
    dz = np.abs(z[pre] - z[post])
    factor = 1.0 + 2.0 * np.exp(-(((dz / scale) / 0.15) ** 2))
    np.testing.assert_allclose(w_on[pv_e], w_off[pv_e] * factor[pv_e], rtol=1e-5)
    np.testing.assert_array_equal(w_on[~pv_e], w_off[~pv_e])
    return int(pv_e.sum())


# (1) connect() carries the member height; mismatched and mixed members are refused.


def test_connect_carries_equal_member_height():
    m1 = jtfne.construct(_laminar("V1", 20, 2.0))
    m2 = jtfne.construct(_laminar("V2", 20, 2.0))
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    assert ens.params["column_height_mm"] == 2.0


def test_connect_refuses_mismatched_member_heights():
    m1 = jtfne.construct(_laminar("V1", 20, 2.0))
    m2 = jtfne.construct(_laminar("V2", 20, 8.0))
    with pytest.raises(ValueError, match="different column_height_mm"):
        jtfne.connect(m1, m2, namespace=("A", "B"))


def test_connect_refuses_laminar_with_height_free_member():
    m1 = jtfne.construct(_laminar("V1", 20, 2.0))
    m2 = jtfne.construct(_laminar("V2", 20, 2.0))
    bare = dataclasses.replace(
        m2, params={k: v for k, v in m2.params.items() if k != "column_height_mm"}
    )
    assert "column_height_mm" not in bare.params, "fixture: member must carry no height"
    with pytest.raises(ValueError, match="refusing to merge laminar"):
        jtfne.connect(m1, bare, namespace=("A", "B"))


# (2) PV<->E gate: layered z is divided by the column height; uniform_3d z is used raw.


def test_pv_e_gate_divides_layered_z_by_column_height():
    on = jtfne.construct(_canonical(True, height_mm=2.0))
    off = jtfne.construct(_canonical(False, height_mm=2.0))
    assert on.cfg.metadata.get("canonical_biophysics") is True, "fixture must enable canonical biophysics"
    assert float(on.params["column_height_mm"]) == 2.0
    z = np.asarray(on.params["positions"], dtype=float)[:, 2]
    assert z.max() > 1.0, "fixture must be in mm"
    n_pv_e = _check_pv_e_gate(on, off, scale=2.0)
    assert n_pv_e > 0


def test_pv_e_gate_uses_raw_z_for_uniform_3d():
    on = jtfne.construct(_uniform(True))
    off = jtfne.construct(_uniform(False))
    assert on.cfg.metadata.get("canonical_biophysics") is True, "fixture must enable canonical biophysics"
    assert set(np.asarray(on.params["emitter"].layer_labels).tolist()) == {"uniform_3d"}
    n_pv_e = _check_pv_e_gate(on, off, scale=1.0)
    assert n_pv_e > 0


def _mixed(canonical_on):
    cfg = (
        jtfne.Configuration()
        .column("A", layers=["uniform_3d"], n=60)
        .column("B", layers=["L2/3", "L5"], n=60)
        .cell_types({"E": 0.75, "PV": 0.25})
        .layer_fractions(
            {"L2/3": (0.0, 0.5), "L5": (0.5, 1.0)},
            {"L2/3": {"E": 0.75, "PV": 0.25}, "L5": {"E": 0.75, "PV": 0.25}},
        )
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45)
    )
    return _finish(cfg).runtime(canonical_biophysics=canonical_on)


def test_pv_e_gate_divides_every_z_by_height_in_a_mixed_model():
    """Decision (T-P066-FOLLOW, 2026-11-23): a model with uniform_3d and layered columns takes one
    depth unit, z / column_height_mm, for every neuron; only an all-uniform_3d model keeps raw z."""
    on = jtfne.construct(_mixed(True))
    off = jtfne.construct(_mixed(False))
    labels = set(np.asarray(on.params["emitter"].layer_labels).tolist())
    assert "uniform_3d" in labels and len(labels) > 1, "fixture must mix uniform_3d and layered labels"
    height = float(on.cfg.metadata.get("column_height_mm", 1.60))
    assert height != 1.0, "fixture: a height of 1 cannot tell divided from raw z"
    assert _check_pv_e_gate(on, off, scale=height) > 0


# (3) simulate(record_fields=True) forwards the column height to the projection.


def test_simulate_record_fields_forwards_column_height():
    H = 2.0
    model = jtfne.construct(_laminar("V1", 60, H))
    assert float(model.params["column_height_mm"]) == H
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.5, seed=0, record_fields=True, record_sources=True)
    sig = model.simulate(sim)
    assert sig.field is not None and sig.sources is not None, "fixture must record field and sources"
    sources = jnp.asarray(sig.sources)
    assert float(jnp.abs(sources).sum()) > 0.0, "fixture must drive the sources"
    positions = jnp.asarray(model.params["positions"])
    assert float(positions[:, 2].max()) > 1.0, "fixture must be in mm"

    with_height = project_laminar_sources(sources, positions, n_contacts=8, column_height_mm=H)
    without_height = project_laminar_sources(sources, positions, n_contacts=8)
    recorded = np.asarray(sig.field.phi_e_proxy, dtype=float)
    np.testing.assert_allclose(recorded, np.asarray(with_height.phi_e_proxy, dtype=float), rtol=1e-5, atol=1e-7)
    diff = np.max(np.abs(np.asarray(with_height.phi_e_proxy, dtype=float) - np.asarray(without_height.phi_e_proxy, dtype=float)))
    assert diff > 1e-3, "height must change the projection, or the forwarding check cannot fail"
