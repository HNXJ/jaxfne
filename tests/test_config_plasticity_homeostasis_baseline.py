"""``Configuration.plasticity`` / ``Configuration.homeostasis`` baseline verbs.

``relative_baseline=1.0`` is the identity/neutral setting for both: it must
not change ``simulate()`` output relative to a Configuration that never calls
either verb. Deviating from ``1.0`` (or passing an explicit override kwarg)
activates the existing homeostasis kernel; plasticity stays declaration-only
(no STDP kernel is wired into ``Model.simulate()`` today).
"""
import numpy as np
import pytest

import jaxfne as jtfne


def _base_cfg():
    return (
        jtfne.Configuration()
        .runtime(duration_ms=20.0, dt_ms=0.5, seed=0)
        .column("V1", layers=["L2/3", "L4"], n=20)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(["spikes", "V_m"])
    )


def _signals(cfg):
    model = jtfne.construct(cfg)
    return jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)


# --- declarative metadata -----------------------------------------------------

def test_plasticity_default_is_identity_baseline_and_declared_not_wired():
    cfg = _base_cfg().plasticity()
    spec = cfg.metadata["plasticity"]
    assert spec["relative_baseline"] == 1.0
    assert spec["status"] == "declared_not_wired_to_simulate"


def test_homeostasis_default_is_identity_baseline_and_inert():
    cfg = _base_cfg().homeostasis()
    assert cfg.metadata["homeostasis"] == {"relative_baseline": 1.0}
    assert cfg.metadata["enable_homeostasis"] is False
    assert "homeostasis_params" not in cfg.metadata


def test_homeostasis_deviation_resolves_k_gain_and_activates():
    cfg = _base_cfg().homeostasis(relative_baseline=2.0)
    assert cfg.metadata["enable_homeostasis"] is True
    assert cfg.metadata["homeostasis_params"]["k_gain"] == pytest.approx(1.0)


def test_homeostasis_explicit_k_gain_override_activates_at_baseline():
    cfg = _base_cfg().homeostasis(k_gain=0.3)
    assert cfg.metadata["enable_homeostasis"] is True
    assert cfg.metadata["homeostasis_params"]["k_gain"] == pytest.approx(0.3)
    assert cfg.metadata["homeostasis"]["relative_baseline"] == 1.0


# --- manifest visibility ------------------------------------------------------

def test_manifest_surfaces_plasticity_and_homeostasis_specs():
    cfg = _base_cfg().plasticity(relative_baseline=1.0).homeostasis(relative_baseline=1.0)
    man = jtfne.manifest(cfg)
    assert man["plasticity"]["relative_baseline"] == 1.0
    assert man["homeostasis"]["relative_baseline"] == 1.0


def test_manifest_omits_keys_gracefully_when_verbs_never_called():
    cfg = _base_cfg()
    man = jtfne.manifest(cfg)
    assert man["plasticity"] is None
    assert man["homeostasis"] is None


# --- behavior parity (the hard guarantee) -------------------------------------

def test_baseline_simulate_output_unchanged_vs_no_verb_call():
    sig_plain = _signals(_base_cfg())
    sig_baseline = _signals(_base_cfg().plasticity().homeostasis())
    assert np.array_equal(np.asarray(sig_plain.get("V_m")), np.asarray(sig_baseline.get("V_m")))
    assert np.array_equal(np.asarray(sig_plain.get("spikes")), np.asarray(sig_baseline.get("spikes")))


def test_homeostasis_deviation_changes_simulate_output():
    sig_plain = _signals(_base_cfg())
    sig_active = _signals(_base_cfg().homeostasis(relative_baseline=2.0))
    assert not np.array_equal(np.asarray(sig_plain.get("V_m")), np.asarray(sig_active.get("V_m")))


# --- Configuration.hdp() (mirrors Configuration.homeostasis()) ---------------

def test_hdp_default_is_identity_baseline_and_inert():
    cfg = _base_cfg().hdp()
    assert cfg.metadata["hdp"] == {"relative_baseline": 1.0}
    assert cfg.metadata["enable_hdp"] is False
    assert "hdp_params" not in cfg.metadata


def test_hdp_deviation_resolves_K_HDP_and_activates():
    cfg = _base_cfg().hdp(relative_baseline=2.0)
    assert cfg.metadata["enable_hdp"] is True
    assert cfg.metadata["hdp_params"]["K_HDP"] == pytest.approx(1.0)


def test_hdp_explicit_K_HDP_override_activates_at_baseline():
    cfg = _base_cfg().hdp(K_HDP=0.01)
    assert cfg.metadata["enable_hdp"] is True
    assert cfg.metadata["hdp_params"]["K_HDP"] == pytest.approx(0.01)
    assert cfg.metadata["hdp"]["relative_baseline"] == 1.0


def test_manifest_surfaces_hdp_spec():
    cfg = _base_cfg().hdp(relative_baseline=1.0)
    man = jtfne.manifest(cfg)
    assert man["hdp"]["relative_baseline"] == 1.0


def test_manifest_omits_hdp_key_when_verb_never_called():
    man = jtfne.manifest(_base_cfg())
    assert man["hdp"] is None


def test_hdp_baseline_simulate_output_unchanged_vs_no_verb_call():
    sig_plain = _signals(_base_cfg())
    sig_baseline = _signals(_base_cfg().hdp())
    assert np.array_equal(np.asarray(sig_plain.get("V_m")), np.asarray(sig_baseline.get("V_m")))
    assert np.array_equal(np.asarray(sig_plain.get("spikes")), np.asarray(sig_baseline.get("spikes")))


def test_hdp_and_homeostasis_both_active_raises_at_simulate():
    cfg = _base_cfg().homeostasis(relative_baseline=2.0).hdp(relative_baseline=2.0)
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError, match="mutually exclusive"):
        jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)
