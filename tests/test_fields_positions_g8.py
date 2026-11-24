"""Lane 7 (G8): model-aware field positions.

``project_model_sources`` (``jaxfne/fields/proxy.py``) projects sources at
``model.params["positions"]`` unless an explicit override is given; the
override must carry the model's neuron count. A Pose3D z-translation moves
the constructed positions and the projection with them.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import jaxfne as jtfne

N = 8
T = 64


def _model(*, dz=0.0):
    tensor = jtfne.make_minimal_ei_tensor(n=N)
    if dz:
        area = dataclasses.replace(
            tensor.areas[0], pose=jtfne.Pose3D(translation=(0.0, 0.0, dz))
        )
        tensor = dataclasses.replace(tensor, areas=[area])
    return jtfne.construct_neuronal_tensor(tensor, seed=0)


def _sources(n=N, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, n)).astype(np.float64)


def test_model_positions_used_by_default():
    """No override: identical arrays to the delegate at model positions, with
    positions_source/model-match provenance in diagnostics."""
    model = _model()
    sources = _sources()
    out = jtfne.project_model_sources(model, sources, n_contacts=8)
    # P-066: model positions are mm; the delegate needs the column height to read depth.
    ref = jtfne.project_laminar_sources(
        sources, np.asarray(model.params["positions"]), n_contacts=8,
        column_height_mm=float(model.params["column_height_mm"]),
    )
    np.testing.assert_array_equal(np.asarray(out.lfp_proxy), np.asarray(ref.lfp_proxy))
    np.testing.assert_array_equal(np.asarray(out.kernel), np.asarray(ref.kernel))
    assert out.lfp_proxy.shape == (T, 8)
    assert bool(np.isfinite(np.asarray(out.lfp_proxy)).all())
    assert out.diagnostics["positions_source"] == "model"
    assert out.diagnostics["positions_match_model"] is True
    assert "observation" in out.diagnostics
    assert out.diagnostics["physical_amplitude_calibrated"] is False


def test_pose_z_translation_moves_positions_and_projection():
    """A 0.5 z-translation shifts constructed z by 0.5, changes the laminar
    projection, and stays consistent with the delegate at the new layout."""
    model_a = _model()
    model_b = _model(dz=0.5)
    pos_a = np.asarray(model_a.params["positions"])
    pos_b = np.asarray(model_b.params["positions"])
    assert pos_a.shape == (N, 3)
    np.testing.assert_allclose(pos_b[:, 2] - pos_a[:, 2], 0.5, rtol=1e-6, atol=1e-6)
    sources = _sources()
    out_a = jtfne.project_model_sources(model_a, sources, n_contacts=8)
    out_b = jtfne.project_model_sources(model_b, sources, n_contacts=8)
    assert not np.allclose(
        np.asarray(out_a.lfp_proxy), np.asarray(out_b.lfp_proxy), rtol=1e-6, atol=1e-9
    ), "z-shifted layout must move the projection"
    ref_b = jtfne.project_laminar_sources(
        sources, pos_b, n_contacts=8, column_height_mm=float(model_b.params["column_height_mm"])
    )
    np.testing.assert_array_equal(
        np.asarray(out_b.lfp_proxy), np.asarray(ref_b.lfp_proxy)
    )


@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_posed_plane_keeps_the_laminar_projection(plane):
    """P-067: a pose plane moves the column, not its laminar readout. The same area
    posed "xy" and posed ``plane`` projects identical sources to the same field."""
    base = _model()
    tensor = jtfne.make_minimal_ei_tensor(n=N)
    area = dataclasses.replace(tensor.areas[0], pose=jtfne.Pose3D(plane=plane))
    posed = jtfne.construct_neuronal_tensor(dataclasses.replace(tensor, areas=[area]), seed=0)
    pos_xy = np.asarray(base.params["positions"])
    pos_p = np.asarray(posed.params["positions"])
    assert np.ptp(pos_xy[:, 2]) > 0, "fixture must spread laminar depth"
    assert not np.allclose(pos_p, pos_xy), "fixture pose must move the positions"
    sources = _sources()
    out_xy = jtfne.project_model_sources(base, sources, n_contacts=8)
    out_p = jtfne.project_model_sources(posed, sources, n_contacts=8)
    np.testing.assert_allclose(
        np.asarray(out_p.lfp_proxy), np.asarray(out_xy.lfp_proxy), rtol=1e-6, atol=1e-9
    )


def _adex_model(plane):
    from jaxfne import neuronal_tensor as nt

    static = nt.StaticParams(dT_ms=5.0, reversal_potentials_mV={"AMPA": 0.0})
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=static, delay_ms=1.0,
    )
    types = [nt.NeuronType.make(name, emitter="adex") for name in ("E", "PV")]
    tensor = nt.NeuronalTensor(name="p067", areas=[nt.Area(
        name="V1", layers=[nt.Layer(name="L4", n_neurons=N, neuron_types=types)],
        inter_connections=[conn], pose=jtfne.Pose3D(plane=plane),
    )])
    runtime = nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=0.5, emitter="adex")
    return jtfne.construct(tensor, runtime)


@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_posed_plane_keeps_the_laminar_projection_on_adex(plane):
    """P-067 on the adex route, which builds its own configuration."""
    base, posed = _adex_model("xy"), _adex_model(plane)
    pos_xy = np.asarray(base.params["positions"])
    assert np.ptp(pos_xy[:, 2]) > 0, "fixture must spread laminar depth"
    assert not np.allclose(np.asarray(posed.params["positions"]), pos_xy), "pose must move positions"
    sources = _sources()
    np.testing.assert_allclose(
        np.asarray(jtfne.project_model_sources(posed, sources, n_contacts=8).lfp_proxy),
        np.asarray(jtfne.project_model_sources(base, sources, n_contacts=8).lfp_proxy),
        rtol=1e-6, atol=1e-9,
    )


@pytest.mark.parametrize("plane", ["xz", "yz"])
def test_posed_plane_keeps_recorded_fields(plane):
    """P-067 on ``simulate(record_fields=True)``: same sources, same recorded field."""
    base = _model()
    tensor = jtfne.make_minimal_ei_tensor(n=N)
    area = dataclasses.replace(tensor.areas[0], pose=jtfne.Pose3D(plane=plane))
    posed = jtfne.construct_neuronal_tensor(dataclasses.replace(tensor, areas=[area]), seed=0)
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=0.1, record_fields=True, record_sources=True)
    out_xy, out_p = base.simulate(sim), posed.simulate(sim)
    src = np.asarray(out_xy.sources)
    assert np.abs(src).max() > 0, "fixture must produce sources"
    np.testing.assert_array_equal(np.asarray(out_p.sources), src)
    np.testing.assert_allclose(
        np.asarray(out_p.field.lfp_proxy), np.asarray(out_xy.field.lfp_proxy), rtol=1e-5, atol=1e-7
    )


def test_explicit_positions_override_and_match_flag():
    """Explicit model-identical positions agree and flag a match; shifted
    ones differ and flag a mismatch."""
    model = _model()
    sources = _sources()
    pos = np.asarray(model.params["positions"])
    same = jtfne.project_model_sources(model, sources, positions=pos, n_contacts=8)
    default = jtfne.project_model_sources(model, sources, n_contacts=8)
    np.testing.assert_array_equal(
        np.asarray(same.lfp_proxy), np.asarray(default.lfp_proxy)
    )
    assert same.diagnostics["positions_source"] == "explicit"
    assert same.diagnostics["positions_match_model"] is True
    shifted = np.array(pos)
    shifted[:, 2] = shifted[:, 2] + 0.25
    moved = jtfne.project_model_sources(model, sources, positions=shifted, n_contacts=8)
    assert moved.diagnostics["positions_source"] == "explicit"
    assert moved.diagnostics["positions_match_model"] is False
    assert not np.allclose(
        np.asarray(moved.lfp_proxy), np.asarray(default.lfp_proxy), rtol=1e-6, atol=1e-9
    )


def test_model_unmatched_inputs_refused():
    """Wrong-N sources, wrong-N explicit positions, a model without
    positions, and non-(N, 3) positions all raise ValueError."""
    model = _model()
    with pytest.raises(ValueError, match="N ="):
        jtfne.project_model_sources(model, _sources(n=N + 1), n_contacts=8)
    with pytest.raises(ValueError, match="N ="):
        jtfne.project_model_sources(
            model, _sources(), positions=np.zeros((N + 1, 3)), n_contacts=8
        )
    with pytest.raises(ValueError, match="model.params\\['positions'\\]"):
        jtfne.project_model_sources(object(), _sources(), n_contacts=8)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        jtfne.project_model_sources(
            model, _sources(), positions=np.zeros((N,)), n_contacts=8
        )


def test_new_surface_is_advanced_not_public():
    """project_model_sources is root-reachable without widening __all__."""
    assert hasattr(jtfne, "project_model_sources")
    assert "project_model_sources" not in jtfne.__all__
