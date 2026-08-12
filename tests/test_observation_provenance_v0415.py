"""O1/O3 observation provenance and authority tests (0.4.15).

Additive metadata only: no FieldOutput layout change, no neural/source
numeric change. Uses public post-hoc ``project_laminar_sources`` and
``LinearReadout`` on a frozen ``Q``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def _tiny_signals():
    cfg = (
        jtfne.configuration()
        .network(
            name="V1",
            kind="cortical_column",
            n=8,
            cell_types={"E": 0.75, "PV": 0.125, "SST": 0.125},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    signals = model.simulate(jtfne.simulation(duration_ms=8.0, dt_ms=0.1, seed=0))
    return model, signals


def test_laminar_observation_chain_on_existing_fieldoutput():
    _, signals = _tiny_signals()
    field = signals.field
    assert field is not None
    obs = field.diagnostics["observation"]
    assert obs["execution_form"] == "fused"
    chain = obs["operator_chain"]
    assert chain["source"]["identity"] == "canonical_relative_source"
    assert chain["field"]["identity"] == "gaussian_laminar_projection"
    assert chain["probe"]["identity"] == "contact_sampling_compiled_into_kernel"
    assert obs["amplitude_semantics"] == "relative"
    assert obs["validation_status"] == "computational"
    assert obs["physical_claim"] == "proxy_readout"
    assert obs["output_identities"]["lfp_proxy"] == "alias_of_source_proxy"
    csd = obs["csd"]
    assert csd["execution_form"] == "fused"
    assert csd["operator_chain"]["probe"]["identity"] == "laminar_second_derivative"
    assert csd["operator_chain"]["field"]["identity"] == chain["field"]["identity"]


def test_eeg_linear_readout_is_fused_p_circ_f():
    W = jnp.ones((3, 8), dtype=jnp.float32)
    report = jtfne.fields.LinearReadout(name="eeg_proxy", W=W).report()
    obs = report["observation"]
    assert obs["execution_form"] == "fused"
    assert obs["operator_chain"]["probe"]["identity"] == "linear_leadfield"
    assert obs["amplitude_semantics"] == "relative"
    assert report["physical_amplitude_calibrated"] is False
    assert obs["physical_claim"] == "proxy_readout"


def test_meg_linear_readout_has_no_orientation_claim():
    W = jnp.ones((3, 8), dtype=jnp.float32)
    obs = jtfne.fields.LinearReadout(name="meg_proxy", W=W).report()["observation"]
    assert obs["execution_form"] == "fused"
    assert obs["operator_chain"]["probe"]["identity"] == "relative_linear_map"
    assert obs["operator_chain"]["probe"]["orientation_claim"] == "none"
    assert obs["output_identity"] == "meg_relative_proxy"


def test_eeg_meg_probe_wrappers_carry_observation_chain():
    arr = jnp.zeros((4, 2), dtype=jnp.float32)
    eeg = jtfne.fields.eeg_proxy_probe(arr)
    meg = jtfne.fields.meg_proxy_probe(arr)
    assert eeg.report["observation"]["operator_chain"]["probe"]["identity"] == "linear_leadfield"
    assert meg.report["observation"]["operator_chain"]["probe"]["orientation_claim"] == "none"


def test_o3_same_operator_same_q_identical_y():
    model, signals = _tiny_signals()
    q = signals.sources
    pos = jnp.asarray(model.params["positions"])
    a = jtfne.project_laminar_sources(q, pos, n_contacts=8, width=0.10)
    b = jtfne.project_laminar_sources(q, pos, n_contacts=8, width=0.10)
    np.testing.assert_array_equal(np.asarray(a.lfp_proxy), np.asarray(b.lfp_proxy))
    np.testing.assert_array_equal(np.asarray(a.csd_proxy), np.asarray(b.csd_proxy))
    np.testing.assert_array_equal(np.asarray(a.kernel), np.asarray(b.kernel))


def test_o3_changed_observation_operator_changes_y_frozen_q():
    model, signals = _tiny_signals()
    q = np.asarray(signals.sources)
    vm = np.asarray(signals.V_m)
    pos = jnp.asarray(model.params["positions"])
    ya = jtfne.project_laminar_sources(signals.sources, pos, n_contacts=8, width=0.10)
    yb = jtfne.project_laminar_sources(signals.sources, pos, n_contacts=8, width=0.25)
    np.testing.assert_array_equal(q, np.asarray(signals.sources))
    np.testing.assert_array_equal(vm, np.asarray(signals.V_m))
    assert not np.allclose(np.asarray(ya.lfp_proxy), np.asarray(yb.lfp_proxy))
    assert ya.diagnostics["observation"]["operator_chain"]["field"]["kernel_width_relative"] == pytest.approx(0.10)
    assert yb.diagnostics["observation"]["operator_chain"]["field"]["kernel_width_relative"] == pytest.approx(0.25)


def test_o3_linear_projection_superposition_and_zero():
    model, signals = _tiny_signals()
    q = signals.sources
    pos = jnp.asarray(model.params["positions"])
    q1 = q
    q2 = 0.5 * q
    y1 = jtfne.project_laminar_sources(q1, pos, n_contacts=8, width=0.10)
    y2 = jtfne.project_laminar_sources(q2, pos, n_contacts=8, width=0.10)
    ysum = jtfne.project_laminar_sources(0.3 * q1 + 0.7 * q2, pos, n_contacts=8, width=0.10)
    np.testing.assert_allclose(
        np.asarray(ysum.lfp_proxy),
        0.3 * np.asarray(y1.lfp_proxy) + 0.7 * np.asarray(y2.lfp_proxy),
        rtol=1e-5,
        atol=1e-5,
    )
    y0 = jtfne.project_laminar_sources(jnp.zeros_like(q), pos, n_contacts=8, width=0.10)
    np.testing.assert_allclose(np.asarray(y0.lfp_proxy), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(y0.csd_proxy), 0.0, atol=1e-6)


def test_o3_linear_readout_superposition_and_zero():
    _, signals = _tiny_signals()
    q = signals.sources
    n = int(q.shape[1])
    W = jnp.arange(2 * n, dtype=jnp.float32).reshape(2, n) * 0.01
    op = jtfne.fields.LinearReadout(name="eeg_proxy", W=W)
    y1 = op.apply(q)
    y2 = op.apply(0.5 * q)
    ysum = op.apply(0.3 * q + 0.7 * (0.5 * q))
    np.testing.assert_allclose(
        np.asarray(ysum),
        0.3 * np.asarray(y1) + 0.7 * np.asarray(y2),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(op.apply(jnp.zeros_like(q))), 0.0, atol=1e-6)


def test_o1_does_not_change_laminar_numerics():
    """Same public call; arrays match a second invocation (zero numeric delta)."""
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.normal(size=(20, 6)).astype(np.float32))
    pos = jnp.asarray(rng.uniform(0.0, 1.0, size=(6, 3)).astype(np.float32))
    a = jtfne.project_laminar_sources(q, pos, n_contacts=5, width=0.12)
    b = jtfne.project_laminar_sources(q, pos, n_contacts=5, width=0.12)
    np.testing.assert_array_equal(np.asarray(a.source_proxy), np.asarray(b.source_proxy))
    np.testing.assert_array_equal(np.asarray(a.lfp_proxy), np.asarray(a.source_proxy))
    np.testing.assert_array_equal(np.asarray(a.phi_e_proxy), np.asarray(a.source_proxy))
