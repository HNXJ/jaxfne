"""P-054 (Hamm ruling 2026-11-02): the VIP and I presets integrate Izhikevich 2004's class-1 equation.

Source: izhikevich.org/publications/figure1.m, panel G (class 1 excitable): a 0.02, b -0.1, c -55, d 6 with
V' = 0.04 V^2 + 4.1 V + 108 - u + I. Rheobase (4.1 - b)^2 / 0.16 - 108 = 2.25; under the standard
0.04 V^2 + 5 V + 140 it was 22.56, so a VIP cell at its preset drive 3.0 was silent. All-standard
populations keep ``v_lin``/``v_const`` None and the exact standard arithmetic.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import izhikevich_params_from_labels, simulate_eig_izhikevich

DT = 0.1
STEPS = 20000  # 2000 ms


def _spikes(params, drive=None):
    if drive is not None:
        params = replace(params, drive=jnp.full_like(params.drive, drive))
    _, spikes, _ = simulate_eig_izhikevich(
        params, STEPS, DT, jax.random.PRNGKey(0), dtype="float64", noise_scale=0.0)
    return np.asarray(spikes)


@pytest.mark.x64
@pytest.mark.parametrize("label", ["VIP", "I"])
def test_class1_coefficients_are_realized(label):
    p = izhikevich_params_from_labels((label,), dtype="float64")
    np.testing.assert_array_equal(np.asarray(p.v_lin), [4.1])
    np.testing.assert_array_equal(np.asarray(p.v_const), [108.0])


def test_all_standard_population_keeps_the_standard_equation():
    p = izhikevich_params_from_labels(("E", "PV", "SST", "Inl", "Ing"))
    assert p.v_lin is None and p.v_const is None


@pytest.mark.x64
@pytest.mark.parametrize("label", ["VIP", "I"])
def test_class1_cell_fires_at_its_preset_drive(label):
    p = izhikevich_params_from_labels((label,), dtype="float64")
    assert float(p.drive[0]) == 3.0, "fixture must use the preset drive"
    assert float(np.asarray(p.W).sum()) == 0.0, "fixture must be one unconnected cell"
    assert _spikes(p).sum() > 0


@pytest.mark.x64
def test_class1_rheobase_lies_between_2_and_2p5():
    p = izhikevich_params_from_labels(("VIP",), dtype="float64")
    assert _spikes(p, drive=2.0).sum() == 0
    assert _spikes(p, drive=2.5).sum() > 0


@pytest.mark.x64
def test_standard_cell_beside_a_class1_cell_is_unchanged():
    alone = izhikevich_params_from_labels(("E",), dtype="float64")
    mixed = izhikevich_params_from_labels(("E", "VIP"), dtype="float64")
    mixed = replace(mixed, W=jnp.zeros_like(mixed.W))
    np.testing.assert_array_equal(np.asarray(mixed.v_lin), [5.0, 4.1])
    e_alone = _spikes(alone)[:, 0]
    assert e_alone.sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(_spikes(mixed)[:, 0], e_alone)


def _cfg_vip():
    return (
        jtfne.Configuration()
        .runtime(seed=1, dtype="float64", duration_ms=10.0, dt_ms=0.5)
        .column(name="c", layers=["L4"], n=60)
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.05, "VIP": 0.05})
        .connectivity()
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
    )


@pytest.mark.x64
def test_construct_realizes_class1_on_vip_cells_only():
    em = jtfne.construct(_cfg_vip()).params["emitter"]
    vip = np.asarray(em.labels) == "VIP"
    assert vip.any() and (~vip).any(), "fixture must mix VIP and standard cells"
    lin, const = np.asarray(em.v_lin), np.asarray(em.v_const)
    assert np.all(lin[vip] == 4.1) and np.all(const[vip] == 108.0)
    assert np.all(lin[~vip] == 5.0) and np.all(const[~vip] == 140.0)


@pytest.mark.x64
def test_cell_params_off_the_class1_set_restore_the_standard_equation():
    # Izhikevich 2003 and the E1 library path label their inhibitory cells "I" with their own a, b, c, d.
    cfg = _cfg_vip().cell_params({"cell_type": "VIP"}, {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0})
    em = jtfne.construct(cfg).params["emitter"]
    assert np.any(np.asarray(em.labels) == "VIP"), "fixture must hold VIP cells"
    assert em.v_lin is None and em.v_const is None


@pytest.mark.x64
def test_cell_params_drive_only_keeps_the_class1_equation():
    cfg = _cfg_vip().cell_params({"cell_type": "VIP"}, {"drive": 4.0})
    em = jtfne.construct(cfg).params["emitter"]
    vip = np.asarray(em.labels) == "VIP"
    assert vip.any() and np.all(np.asarray(em.v_lin)[vip] == 4.1)


@pytest.mark.x64
def test_with_emitter_parameters_b_override_restores_the_standard_equation():
    model = jtfne.construct(_cfg_vip())
    assert model.params["emitter"].v_lin is not None, "fixture must start with class-1 cells"
    assert model.with_emitter_parameters(b=0.2).params["emitter"].v_lin is None


@pytest.mark.x64
def test_checkpoint_round_trip_keeps_the_v_equation(tmp_path):
    cfg = _cfg_vip()
    model = jtfne.construct(cfg)
    model.checkpoint(str(tmp_path / "ck"))
    got = jtfne.Model.restore(str(tmp_path / "ck"), cfg).params["emitter"]
    em = model.params["emitter"]
    assert np.any(np.asarray(em.v_lin) != 5.0), "fixture must carry class-1 cells"
    np.testing.assert_array_equal(np.asarray(got.v_lin), np.asarray(em.v_lin))
    np.testing.assert_array_equal(np.asarray(got.v_const), np.asarray(em.v_const))


def test_sanity_runtime_vip_cells_use_class1():
    from jaxfne import sanity_runtime as sr

    lin, const = (np.asarray(x) for x in sr._build_v_coefs(500))
    b = np.asarray(sr._build_area_index(500)[1])
    vip = b == -0.1
    assert vip.sum() == 40, "fixture must find the 8 VIP cells of each of 5 areas"
    assert np.all(lin[vip] == 4.1) and np.all(const[vip] == 108.0)
    assert np.all(lin[~vip] == 5.0) and np.all(const[~vip] == 140.0)
