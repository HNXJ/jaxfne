"""Tests for v0.0.20 semantic correctness hardening.

Covers all 9 behavioral bugs fixed:
A. receipt_completeness       — duration_ms, dt_ms, n_steps, record flags in receipt
B. receipt_id_determinism     — same payload → same id
C. receipt_id_specificity     — different duration → different id
D. manifest_backend           — used vs available backend separation
E. manifest_no_signals        — manifest without signals reports unknown_not_run
F. n_contacts_honored         — probe n_contacts propagates to model.static
G. n_contacts_default         — absent n_contacts defaults to 16
H. n_contacts_invalid         — n_contacts < 2 raises ValueError
I. field_readout_time_window  — CSD readout sliced by time_window_ms
J. field_readout_no_window    — CSD readout full array when no window given
K. empty_window_status        — t0 >= t1 → status empty_time_window, value None
L. negative_window_status     — negative start → empty_time_window, not NaN
M. zero_width_window_status   — identical start/end → empty_time_window
N. simulation_negative_dur    — Simulation(duration_ms < 0) raises ValueError
O. simulation_zero_dur        — Simulation(duration_ms = 0) raises ValueError
P. simulation_negative_dt     — Simulation(dt_ms < 0) raises ValueError
Q. simulation_zero_dt         — Simulation(dt_ms = 0) raises ValueError
R. simulation_valid_small     — Simulation(duration_ms=1.0, dt_ms=0.5) is valid
S. record_sources_in_metadata — record_sources present in signals.metadata
T. duration_in_metadata       — duration_ms present in signals.metadata
U. dt_in_metadata             — dt_ms present in signals.metadata
V. schema_constants           — _MANIFEST_SCHEMA_VERSION etc. are defined and non-empty
W. receipt_schema_v020        — _RECEIPT_SCHEMA_VERSION contains "0.0.20"
"""

import json
import math

import pytest

import jaxfne
from jaxfne.core import (
    _JAXFNE_VERSION,
    _RECEIPT_SCHEMA_VERSION,
    _MANIFEST_SCHEMA_VERSION,
    _OBJECTIVE_REPORT_SCHEMA_VERSION,
)

# ─── helpers ──────────────────────────────────────────────────────────────────


def _cfg(n_contacts: int = 16) -> jaxfne.Configuration:
    return (
        jaxfne.Configuration()
        .network(n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="lp", n_contacts=n_contacts)
    )


def _model_and_signals(n_contacts: int = 16):
    model = jaxfne.construct(_cfg(n_contacts=n_contacts))
    sim = jaxfne.Simulation(duration_ms=50.0, dt_ms=0.5)
    signals = model.simulate(sim)
    return model, signals


# ─── A. receipt completeness ──────────────────────────────────────────────────


def test_a_receipt_duration_ms_not_none():
    model, signals = _model_and_signals()
    r = model.run_receipt(signals)
    assert r.simulation["duration_ms"] == pytest.approx(50.0)


def test_a_receipt_dt_ms_not_none():
    model, signals = _model_and_signals()
    r = model.run_receipt(signals)
    assert r.simulation["dt_ms"] == pytest.approx(0.5)


def test_a_receipt_n_steps_matches():
    model, signals = _model_and_signals()
    r = model.run_receipt(signals)
    assert r.simulation["n_steps"] == 100


def test_a_receipt_record_sources_present():
    model, signals = _model_and_signals()
    r = model.run_receipt(signals)
    assert "record_sources" in r.simulation
    assert r.simulation["record_sources"] is True


def test_a_receipt_record_fields_present():
    model, signals = _model_and_signals()
    r = model.run_receipt(signals)
    assert "record_fields" in r.simulation
    assert r.simulation["record_fields"] is True


# ─── B/C. receipt_id determinism and specificity ──────────────────────────────


def test_b_receipt_id_stable_same_run():
    model, signals = _model_and_signals()
    r1 = model.run_receipt(signals)
    r2 = model.run_receipt(signals)
    assert r1.receipt_id == r2.receipt_id


def test_c_receipt_id_changes_on_duration():
    model = jaxfne.construct(_cfg())
    s50 = model.simulate(jaxfne.Simulation(duration_ms=50.0, dt_ms=0.5))
    s100 = model.simulate(jaxfne.Simulation(duration_ms=100.0, dt_ms=0.5))
    r50 = model.run_receipt(s50)
    r100 = model.run_receipt(s100)
    assert r50.receipt_id != r100.receipt_id


def test_c_receipt_id_changes_on_dt():
    model = jaxfne.construct(_cfg())
    s05 = model.simulate(jaxfne.Simulation(duration_ms=50.0, dt_ms=0.5))
    s025 = model.simulate(jaxfne.Simulation(duration_ms=50.0, dt_ms=0.25))
    r05 = model.run_receipt(s05)
    r025 = model.run_receipt(s025)
    assert r05.receipt_id != r025.receipt_id


def test_c_receipt_id_is_16_chars():
    model, signals = _model_and_signals()
    r = model.run_receipt(signals)
    assert isinstance(r.receipt_id, str)
    assert len(r.receipt_id) == 16


def test_c_receipt_json_roundtrip():
    model, signals = _model_and_signals()
    r = model.run_receipt(signals)
    d = r.to_dict()
    json.dumps(d, allow_nan=False)  # must not raise


# ─── D/E. manifest backend metadata ──────────────────────────────────────────


def test_d_manifest_used_recurrent_backend_from_signals():
    model, signals = _model_and_signals()
    mf = model.manifest(signals=signals)
    bm = mf["backend_metadata"]
    assert bm["used_recurrent_backend"] == "dense"


def test_d_manifest_available_edge_list_reported():
    model, signals = _model_and_signals()
    mf = model.manifest(signals=signals)
    bm = mf["backend_metadata"]
    assert "available_edge_list" in bm


def test_e_manifest_no_signals_reports_unknown():
    model = jaxfne.construct(_cfg())
    mf = model.manifest()
    bm = mf["backend_metadata"]
    assert bm["used_recurrent_backend"] == "unknown_not_run"


def test_d_manifest_dense_not_reported_as_edge_list():
    model, signals = _model_and_signals()
    mf = model.manifest(signals=signals)
    bm = mf["backend_metadata"]
    assert bm["used_recurrent_backend"] != "edge_list"


# ─── F/G/H. probe n_contacts ─────────────────────────────────────────────────


def test_f_n_contacts_4_honored():
    model = jaxfne.construct(_cfg(n_contacts=4))
    assert model.static["n_contacts"] == 4


def test_f_n_contacts_32_honored():
    model = jaxfne.construct(_cfg(n_contacts=32))
    assert model.static["n_contacts"] == 32


def test_g_n_contacts_default_16():
    cfg = (
        jaxfne.Configuration()
        .network(n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="lp")  # no n_contacts key
    )
    model = jaxfne.construct(cfg)
    assert model.static["n_contacts"] == 16


def test_h_n_contacts_invalid_zero_raises():
    cfg = (
        jaxfne.Configuration()
        .network(n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="lp", n_contacts=0)
    )
    with pytest.raises(ValueError, match="n_contacts must be >= 2"):
        jaxfne.construct(cfg)


def test_h_n_contacts_invalid_one_raises():
    cfg = (
        jaxfne.Configuration()
        .network(n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="lp", n_contacts=1)
    )
    with pytest.raises(ValueError, match="n_contacts must be >= 2"):
        jaxfne.construct(cfg)


# ─── I/J. field readout time-window slicing ──────────────────────────────────


def test_i_field_readout_window_different_from_full():
    model, signals = _model_and_signals()
    spec_full = jaxfne.ReadoutSpec(name="full", metric="csd_abs_mean")
    spec_half = jaxfne.ReadoutSpec(name="half", metric="csd_abs_mean",
                                   time_window_ms=(0.0, 25.0))
    res = model.compute_readout(signals, [spec_full, spec_half])
    assert res[0].status == "computed"
    assert res[1].status == "computed"
    assert res[0].value != res[1].value


def test_j_field_readout_no_window_uses_full_array():
    model, signals = _model_and_signals()
    spec = jaxfne.ReadoutSpec(name="full", metric="csd_abs_mean")
    res = model.compute_readout(signals, [spec])
    assert res[0].status == "computed"
    assert res[0].value is not None
    assert not math.isnan(res[0].value)


# ─── K/L/M. empty / negative / zero-width window ─────────────────────────────


def test_k_empty_window_status():
    model, signals = _model_and_signals()
    spec = jaxfne.ReadoutSpec(name="emp", metric="spike_rate_hz",
                              time_window_ms=(25.0, 25.0))
    res = model.compute_readout(signals, [spec])
    assert res[0].status == "empty_time_window"
    assert res[0].value is None


def test_l_negative_window_no_nan():
    model, signals = _model_and_signals()
    spec = jaxfne.ReadoutSpec(name="baseline", metric="spike_rate_hz",
                              time_window_ms=(-500.0, 0.0))
    res = model.compute_readout(signals, [spec])
    assert res[0].status == "empty_time_window"
    assert res[0].value is None


def test_m_zero_width_window_status():
    model, signals = _model_and_signals()
    spec = jaxfne.ReadoutSpec(name="zw", metric="mean_V_m",
                              time_window_ms=(10.0, 10.0))
    res = model.compute_readout(signals, [spec])
    assert res[0].status == "empty_time_window"
    assert res[0].value is None


def test_l_negative_window_json_safe():
    model, signals = _model_and_signals()
    spec = jaxfne.ReadoutSpec(name="b", metric="csd_abs_mean",
                              time_window_ms=(-500.0, 0.0))
    res = model.compute_readout(signals, [spec])
    # ReadoutResult must be JSON-safe (no NaN)
    from jaxfne.io import json_safe
    d = {"value": res[0].value, "status": res[0].status}
    json.dumps(json_safe(d), allow_nan=False)


# ─── N–R. Simulation validation ──────────────────────────────────────────────


def test_n_simulation_negative_duration_raises():
    with pytest.raises(ValueError, match="duration_ms"):
        jaxfne.Simulation(duration_ms=-10.0)


def test_o_simulation_zero_duration_raises():
    with pytest.raises(ValueError, match="duration_ms"):
        jaxfne.Simulation(duration_ms=0.0)


def test_p_simulation_negative_dt_raises():
    with pytest.raises(ValueError, match="dt_ms"):
        jaxfne.Simulation(dt_ms=-0.5)


def test_q_simulation_zero_dt_raises():
    with pytest.raises(ValueError, match="dt_ms"):
        jaxfne.Simulation(dt_ms=0.0)


def test_r_simulation_valid_small():
    sim = jaxfne.Simulation(duration_ms=1.0, dt_ms=0.5)
    assert sim.n_steps == 2


def test_r_simulation_default_is_valid():
    sim = jaxfne.Simulation()
    assert sim.n_steps > 0


# ─── S/T/U. metadata completeness ────────────────────────────────────────────


def test_s_record_sources_in_metadata():
    model, signals = _model_and_signals()
    assert "record_sources" in signals.metadata
    assert signals.metadata["record_sources"] is True


def test_t_duration_ms_in_metadata():
    model, signals = _model_and_signals()
    assert signals.metadata["duration_ms"] == pytest.approx(50.0)


def test_u_dt_ms_in_metadata():
    model, signals = _model_and_signals()
    assert signals.metadata["dt_ms"] == pytest.approx(0.5)


# ─── V/W. schema/version constants ───────────────────────────────────────────


def test_v_manifest_schema_version_defined():
    assert isinstance(_MANIFEST_SCHEMA_VERSION, str)
    assert len(_MANIFEST_SCHEMA_VERSION) > 0


def test_v_objective_report_schema_version_defined():
    assert isinstance(_OBJECTIVE_REPORT_SCHEMA_VERSION, str)
    assert len(_OBJECTIVE_REPORT_SCHEMA_VERSION) > 0


def test_w_receipt_schema_version_contains_version():
    assert "0.0.21" in _RECEIPT_SCHEMA_VERSION


def test_v_jaxfne_version_is_028():
    """jaxfne runtime version must match pyproject.toml version (active version check)."""
    import tomllib
    from pathlib import Path
    pyproject_version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    assert _JAXFNE_VERSION == pyproject_version, \
        f"Module version {_JAXFNE_VERSION} does not match pyproject.toml {pyproject_version}"
