"""P-048 (Hamm ruling 2026-10-28): the Izhikevich initial recovery state defaults to u0 = b * v0.

``b`` is the realized per-cell value after ``cell_params()`` and ``v0`` the realized per-cell
value after the ``random_v0`` draw. A ``u0`` declared in ``cell_params()`` still wins. Every
assertion reads ``model.params["emitter"]``, never the declared input.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import izhikevich_params_from_labels

N = 60
B_E = 0.30  # override of the E preset b (0.20)
U0_E = -7.0  # declared u0 for the E cells


def _cfg(*, random_v0=True, **runtime):
    return (
        jtfne.Configuration()
        .runtime(seed=42, dtype="float64", duration_ms=10.0, dt_ms=0.5, random_v0=random_v0, **runtime)
        .column(name="v1_col", layers=["L4"], n=N)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity()
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
    )


def _emitter(cfg):
    return jtfne.construct(cfg).params["emitter"]


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _assert_u0_follows_b_and_v0(em, mask=None):
    b, v0, u0 = _f64(em.b), _f64(em.v0), _f64(em.u0)
    sel = np.ones(u0.shape, dtype=bool) if mask is None else mask
    assert sel.any(), "fixture must select at least one cell"
    np.testing.assert_allclose(u0[sel], b[sel] * v0[sel], rtol=1e-12)


@pytest.mark.x64
def test_b_override_with_fixed_v0_sets_u0_to_b_times_v0():
    em = _emitter(_cfg(random_v0=False).cell_params({"cell_type": "E"}, {"b": B_E}))
    e = np.asarray(em.labels) == "E"
    assert e.any() and np.all(_f64(em.b)[e] == B_E), "fixture must realize the b override"
    np.testing.assert_array_equal(_f64(em.v0), -65.0)
    _assert_u0_follows_b_and_v0(em)
    np.testing.assert_allclose(_f64(em.u0)[e], B_E * -65.0, rtol=1e-12)


@pytest.mark.x64
def test_default_u0_follows_the_drawn_v0():
    em = _emitter(_cfg())
    assert np.ptp(_f64(em.v0)) > 0.0, "fixture must draw v0 (random_v0 is on by default)"
    _assert_u0_follows_b_and_v0(em)


@pytest.mark.x64
def test_b_override_follows_the_realized_b_and_the_drawn_v0():
    em = _emitter(_cfg().cell_params({"cell_type": "E"}, {"b": B_E}))
    e = np.asarray(em.labels) == "E"
    assert e.any() and np.all(_f64(em.b)[e] == B_E), "fixture must realize the b override"
    assert np.ptp(_f64(em.v0)) > 0.0, "fixture must draw v0"
    _assert_u0_follows_b_and_v0(em)


@pytest.mark.x64
def test_declared_u0_still_wins_over_b_times_v0():
    em = _emitter(_cfg().cell_params({"cell_type": "E"}, {"b": B_E, "u0": U0_E}))
    e = np.asarray(em.labels) == "E"
    assert e.any(), "fixture must hold E cells"
    np.testing.assert_array_equal(_f64(em.u0)[e], np.float64(U0_E))
    _assert_u0_follows_b_and_v0(em, mask=~e)


@pytest.mark.x64
def test_b_override_under_canonical_biophysics_follows_b_times_v0():
    cfg = _cfg(canonical_biophysics=True).cell_params({"cell_type": "E"}, {"b": B_E})
    em = _emitter(cfg)
    e = np.asarray(em.labels) == "E"
    assert e.any() and np.all(_f64(em.b)[e] == B_E), "fixture must realize the b override"
    assert not np.allclose(_f64(em.a)[e], 0.02), "fixture must apply the canonical deep-E grading"
    _assert_u0_follows_b_and_v0(em)


def test_b_override_on_the_plain_route_follows_b_times_v0():
    cfg = (
        jtfne.configuration()
        .network(name="gap", kind="balanced_ei_population", n=40, cell_types={"E": 0.8, "I": 0.2})
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 5.0, "reversal_mV": 0.0})
        .cell_params({"cell_type": "E"}, {"b": B_E})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    em = jtfne.construct(cfg).params["emitter"]
    e = np.asarray(em.labels) == "E"
    assert e.any() and np.all(_f64(em.b)[e] == np.float32(B_E)), "fixture must realize the b override"
    assert np.ptp(_f64(em.v0)) > 0.0, "fixture must draw v0"
    np.testing.assert_allclose(_f64(em.u0), _f64(em.b) * _f64(em.v0), rtol=1e-6)


@pytest.mark.x64
def test_preset_default_is_unchanged_with_fixed_v0():
    """Guard: no override and v0 fixed at -65 leave u0 at the preset b * -65."""
    em = _emitter(_cfg(random_v0=False))
    preset = izhikevich_params_from_labels(
        tuple(np.asarray(em.labels).tolist()), dtype="float64", build_dense_connectivity=False
    )
    np.testing.assert_array_equal(_f64(em.u0), _f64(preset.u0))
