"""Canary chains: one seeded, fast run per core data path, summarised into a
frozen register (``tests/canary_register.json``).

A red canary means "this data path changed its output"; the unit tests then
localise the change. Exact values are spike counts per population and a sha256
of the sorted spike pairs: ``(step index, cell id)`` for the clocked chains and
``(spike time in us, cell id)`` for the event-scan chain, whose events are not
on the step grid. Floats are summaries compared with ``rtol=1e-5``; keys in mV
also get ``atol=1e-4`` mV, the B3 cross-platform contract. Several floats are
float32 summaries (the objective chain), so ``rtol=1e-6`` would risk a
Linux-vs-Windows false red.

Each chain asserts its precondition (spikes, a moved weight sum or weights, a
moved source scale) before it compares. Write the register once with
``JAXFNE_CANARY_WRITE=1`` (run serially); otherwise the tests only read it, and
a missing chain entry fails.

Each chain reuses the smallest fixture of an existing test module (named in its
docstring). A jdna develop chain (``jaxfne.jdna``) is not included yet.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.hdp_network import DEFAULT_HDP

pytestmark = pytest.mark.canary

REGISTER = Path(__file__).resolve().parent / "canary_register.json"
WRITE = os.environ.get("JAXFNE_CANARY_WRITE") == "1"
RTOL = 1e-5
ATOL_MV = 1e-4


# --- register and comparison -------------------------------------------------


def _read_register() -> dict:
    if not REGISTER.exists():
        return {"chains": {}}
    return json.loads(REGISTER.read_text(encoding="utf-8"))


def _write_register(reg: dict) -> None:
    text = json.dumps(reg, indent=2, sort_keys=True) + "\n"
    REGISTER.write_bytes(text.encode("utf-8"))


def _float_atol(key: str) -> float:
    return ATOL_MV if "_mV" in key else 0.0


def _check(chain: str, exact: dict, floats: dict) -> None:
    """Compare one chain to its register entry, or write it in write mode."""
    reg = _read_register()
    if WRITE:
        reg.setdefault("chains", {})[chain] = {"exact": exact, "floats": floats}
        _write_register(reg)
        return
    entry = reg.get("chains", {}).get(chain)
    if entry is None:
        pytest.fail(
            f"canary chain {chain!r}: no register entry in {REGISTER.name}; "
            "run once with JAXFNE_CANARY_WRITE=1")
    for key in sorted(set(exact) | set(entry["exact"])):
        got = exact.get(key, "<absent>")
        want = entry["exact"].get(key, "<absent>")
        if got != want:
            raise AssertionError(
                f"canary chain {chain!r}: first differing key {key!r}: "
                f"got {got!r}, register {want!r}")
    for key in sorted(set(floats) | set(entry["floats"])):
        if key not in floats or key not in entry["floats"]:
            raise AssertionError(
                f"canary chain {chain!r}: first differing key {key!r}: "
                "present on one side only")
        try:
            np.testing.assert_allclose(
                floats[key], entry["floats"][key], rtol=RTOL, atol=_float_atol(key))
        except AssertionError as err:
            raise AssertionError(
                f"canary chain {chain!r}: first differing key {key!r}: {err}") from None


def _summary(spikes, labels, populations=None) -> dict:
    """Exact summary of a (steps, cells) spike array.

    ``populations`` maps a population name to a boolean cell mask; when None,
    the per-label counts come from ``labels`` (one label per cell).
    """
    s = np.asarray(spikes).astype(bool)
    assert s.ndim == 2, f"spikes must be (steps, cells), got {s.shape}"
    pairs = np.argwhere(s).astype("<i8")
    pairs = np.ascontiguousarray(pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))])
    out = {
        "spikes/total": int(s.sum()),
        "raster_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
    }
    if populations is None:
        assert labels is not None, "fixture must name its populations"
        lab = np.asarray([str(x) for x in labels])
        populations = {name: lab == name for name in sorted(set(lab.tolist()))}
    for name, mask in populations.items():
        out[f"spikes/{name}"] = int(s[:, np.asarray(mask, dtype=bool)].sum())
    assert out["spikes/total"] > 0, "fixture must spike"
    return out


def _labels(model):
    return getattr(model.params["emitter"], "labels", None)


def _finite(name: str, value) -> float:
    arr = np.asarray(value, dtype=float)
    assert bool(np.isfinite(arr).all()), f"{name} must be finite"
    return float(np.mean(arr)) if arr.ndim else float(arr)


# --- chain 1: NeuronalTensor -> construct -> simulate, LIF edge route -------
# Fixture: tests/test_lif_tensor_route.py::_tensor and _spiking_lif_model.


def _lif_tensor():
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
        delay_ms=1.0,
    )
    layer = nt.Layer(
        name="L4", n_neurons=6,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")])
    return nt.NeuronalTensor(
        name="canary-lif",
        areas=[nt.Area(name="V1", layers=[layer], inter_connections=[conn])],
    )


def test_chain1_lif_edge_route_construct_simulate():
    model = jtfne.construct(
        _lif_tensor(),
        nt.RuntimeConfiguration(seed=0, dt_ms=0.5, duration_ms=20.0, emitter="lif"))
    sig = model.simulate(jtfne.Simulation(
        duration_ms=20.0, dt_ms=0.5, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
    exact = _summary(sig.spikes, _labels(model))
    floats = {"mean_V_m_mV": _finite("V_m", sig.V_m)}
    _check("lif_edge", exact, floats)


# --- chain 2: NeuronalTensor -> construct -> simulate, Izhikevich route -----
# Fixture: tests/test_neuronal_tensor.py::_single_area_tensor.


def _izhikevich_tensor():
    layer = nt.Layer(
        name="L4", n_neurons=20,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")])
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(
            g_mech={"AMPA": 1.0}, reversal_potentials_mV={"AMPA": 0.0}, dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=2.0, H=0.0),
    )
    return nt.NeuronalTensor(
        areas=[nt.Area(name="V1", layers=[layer], inter_connections=[conn])],
        name="canary_izhikevich")


def test_chain2_izhikevich_route_construct_simulate():
    model = nt.construct_neuronal_tensor(
        _izhikevich_tensor(), seed=0, duration_ms=50.0, dt_ms=0.5)
    sig = jtfne.simulate(model, duration_ms=50.0, dt_ms=0.5, seed=0)
    exact = _summary(sig.spikes, _labels(model))
    floats = {"mean_V_m_mV": _finite("V_m", sig.V_m)}
    _check("izhikevich_route", exact, floats)


# --- chain 3: HDP run with plastic W (izhikevich laminar column) -----------
# Fixture: tests/test_laminar_1000n_lfp_csd_hdp.py::_build_model at N=8, shorter.


def _laminar(n: int, duration_ms: float):
    return (
        jtfne.build_laminar_column(n=n, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=4)
        .runtime(seed=0, duration_ms=duration_ms, dt_ms=0.1)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )


def test_chain3_hdp_run_with_plastic_w():
    model = jtfne.construct(_laminar(8, 200.0))
    w0 = np.asarray(model.params["edge_list"].weight, dtype=float)
    sig = jtfne.simulate(
        model, duration_ms=200.0, dt_ms=0.1, seed=0,
        runtime=jtfne.RuntimeConfig(
            enable_hdp=True,
            hdp_params=dict(DEFAULT_HDP, record_weight_trace=False)))
    diag = model.last_hdp_diagnostics()
    assert diag is not None, "HDP must be active on the canary run"
    w_final = np.asarray(diag["w_final"], dtype=float)
    assert w_final.shape == w0.shape, "w_final must align with the edge list"
    # Precondition: W is plastic on this run, so its final sum differs from the initial sum.
    assert not np.isclose(w_final.sum(), w0.sum(), rtol=1e-9, atol=0.0), (
        "fixture must move the weight sum")
    assert not np.allclose(w_final, w0), "fixture must move W"
    exact = _summary(sig.spikes, _labels(model))
    floats = {
        "mean_V_m_mV": _finite("V_m", sig.V_m),
        "final_weight_sum": _finite("w_final", w_final.sum()),
    }
    _check("hdp_plastic_w", exact, floats)


# --- chain 4: event-scan replay with a declared H and W update -------------
# Fixture: tests/test_emitters_lif.py::_complex_chain_fixture: two simple depths
# and one complex cell, events at 100 and 200 us, so the cascade spikes four
# times. The declared H update is _interval_h() with a 100 us interval, widened
# to three cells: it fires at the second event, after that event's spikes. The
# declared W update is _w_update(nd=2, complex_mode="excitatory") from the same
# module. EventScanDrive carries both declarations; the tensor route forwards
# them to simulate_lif_edge (jaxfne/_model_simulate.py). Event times are in
# microseconds, so the raster pairs are (spike time us, cell id).


def _interval_h3():
    """The H update of tests/test_emitters_lif.py::_interval_h, three cells, 100 us interval."""
    return dict(
        measurement_interval_us=100.0, rate_alpha=0.6,
        target_rate_hz=np.array([0.75, 0.0, 0.0]),
        threshold_eta_mV=np.array([4.0, 0.0, 0.0]),
        threshold_min_mV=np.array([4.0, 0.0, 0.0]),
        last_update_us=0,
    )


def _w_update_two_depths():
    """The W declaration of tests/test_emitters_lif.py::_w_update(nd=2, complex_mode="excitatory")."""

    def population(mode):
        return {
            "mode": mode, "eLTP": 2.0, "eLTD": 1.0, "iLTP": 0.5, "iLTD": 0.5,
            "tau_ltp_us": 40000.0, "tau_ltd_us": 40000.0,
            "norm_factor": 50.0, "eta_inh": 1500.0,
        }

    return {
        "rule": "pcl_buffer_stdp", "buffer_capacity": 8,
        "wi_s": np.zeros((2, 2)), "wi_c": np.zeros((1, 1)),
        "decay": np.ones((3,)),
        "simple": population("excitatory"), "complex": population("excitatory"),
    }


def _scan_drive(*, declared: bool):
    from jaxfne.pcl_topology import (
        EventScanDrive,
        build_s1_scan_tensor,
        event_scan_events,
        event_scan_tables,
        local_inhibition_blocks,
    )

    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    w_s[0, 0, 0, 0] = 40.0
    w_s[0, 1, 0, 0] = 4.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=2, cx=1, cy=1, cd=1,
        origins_x=[0], origins_y=[0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.array([[[[40.0, 4.0]]]], dtype=np.float64),
        vw=1, vh=1, inhib_range=(0, 0), lateral_on=False,
    )
    zeros = np.zeros(2, dtype=np.int64)
    events = event_scan_events(
        np.array([100, 200]), zeros, zeros, zeros,
        dt_ms=0.5, n_steps=1, vw=1, vh=1,
    )
    groups = np.array([0, 0, 1], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups,
        li_pre=np.zeros(0, dtype=np.int64), li_post=np.zeros(0, dtype=np.int64),
        li_weight=np.zeros(0, dtype=np.float64),
    )
    drive = EventScanDrive(
        events, tables, blocks, groups, dt_ms=0.5, n_steps=1,
        h_update=_interval_h3() if declared else None,
        w_update=_w_update_two_depths() if declared else None,
    )
    return drive, build_s1_scan_tensor(ns=2, nc=1)


def test_chain4_event_scan_replay_with_declared_h_and_w_update():
    from jaxfne.emitters_lif import make_lif_params, simulate_lif_edge

    drive, tensor = _scan_drive(declared=True)
    params = make_lif_params(["simple", "simple", "complex"])
    kw = dict(
        n_steps=1, dt_ms=0.5,
        local_inhibition_groups=drive.groups,
        local_inhibition_resolution="event_scan",
        local_inhibition_blocks=drive.blocks,
        event_scan_events=drive.events,
        event_scan_tables=drive.tables,
    )
    ref = simulate_lif_edge(
        params, event_scan_h_update=drive.h_update,
        event_scan_w_update=drive.w_update, **kw)
    frozen = simulate_lif_edge(params, **kw)

    model = jtfne.construct(
        tensor,
        nt.RuntimeConfiguration(seed=0, dt_ms=0.5, duration_ms=0.5, emitter="lif"))
    sig = model.simulate(
        jtfne.Simulation(
            duration_ms=0.5, dt_ms=0.5, seed=1,
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")),
        paradigm=drive)
    assert np.array_equal(np.asarray(sig.spikes), np.asarray(ref["spikes"])), (
        "the tensor route must replay the declared kernel")

    # Preconditions: more than two spikes; the declared H moves a threshold and
    # the declared W moves both weight matrices, each relative to the frozen run.
    times = np.asarray(ref["spike_times_us"], dtype=np.int64)
    cells = np.asarray(ref["spike_cells"], dtype=np.int64)
    assert times.size > 2, "fixture must spike more than twice"
    th = np.asarray(ref["threshold_mV"], dtype=float)
    assert not np.array_equal(th, np.asarray(frozen["threshold_mV"], dtype=float)), (
        "the declared H update must move a threshold")
    for name in ("w_s", "w_c"):
        got_w = np.asarray(ref["weights_final"][name], dtype=float)
        frozen_w = np.asarray(frozen["weights_final"][name], dtype=float)
        assert not np.array_equal(got_w, frozen_w), (
            f"the declared W update must move {name}")

    order = np.lexsort((cells, times))
    pairs = np.ascontiguousarray(
        np.stack([times[order], cells[order]], axis=1).astype("<i8"))
    exact = {
        "spikes/total": int(times.size),
        "raster_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
        "spikes/simple": int(np.isin(cells, [0, 1]).sum()),
        "spikes/complex": int(np.isin(cells, [2]).sum()),
    }
    floats = {
        "w_s_sum": float(np.asarray(ref["weights_final"]["w_s"], dtype=float).sum()),
        "w_c_sum": float(np.asarray(ref["weights_final"]["w_c"], dtype=float).sum()),
        "threshold_sum_mV": float(th.sum()),
    }
    _check("event_scan_replay", exact, floats)


# --- chain 5: field probe, LFP and CSD proxies from a simulated run --------
# Fixture: tests/test_laminar_1000n_lfp_csd_hdp.py (field readout), without HDP.


def test_chain5_field_probe_lfp_and_csd():
    model = jtfne.construct(_laminar(8, 200.0))
    sig = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.1, seed=0, record_fields=True)
    assert sig.field is not None, "record_fields=True must produce a FieldOutput"
    lfp = np.asarray(sig.field.lfp_proxy, dtype=float)
    csd = np.asarray(sig.field.csd_proxy, dtype=float)
    assert lfp.shape[1] == 4 and csd.shape == lfp.shape
    exact = _summary(sig.spikes, _labels(model))
    floats = {
        "lfp_rms": float(np.sqrt(np.mean(lfp ** 2))) if np.isfinite(lfp).all() else float("nan"),
        "csd_rms": float(np.sqrt(np.mean(csd ** 2))) if np.isfinite(csd).all() else float("nan"),
    }
    assert np.isfinite(floats["lfp_rms"]) and floats["lfp_rms"] > 0.0, "fixture must carry LFP"
    assert np.isfinite(floats["csd_rms"]), "CSD must be finite"
    _check("field_lfp_csd", exact, floats)


# --- chain 6: objective and one optimizer step ------------------------------
# Fixture: tests/test_differentiable_source_scale_tune.py::_model, steps=1.


def _tune_model():
    cfg = (
        jtfne.configuration()
        .network(n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def test_chain6_objective_and_one_optimizer_step():
    model = _tune_model()
    sim = jtfne.simulation(duration_ms=20.0, dt_ms=0.1, seed=0)
    baseline = model.simulate(sim)
    baseline_val = float(abs(baseline.field.source_proxy).mean())
    exact = _summary(baseline.spikes, _labels(model))
    objective = jtfne.objective().loss(
        "hit_target", metric="source_proxy_abs_mean", target=baseline_val * 2.0)
    optimizer = jtfne.optax_adam(
        learning_rate=0.05, differentiability_status="declared_surrogate")
    result = model.tune(objectives=objective, optimizer=optimizer, simulation=sim, steps=1)
    assert result.summary["tuning_status"] == "differentiable_source_scale_loop_v0.1"
    history = result.summary["gradient_history"]
    exact["gradient_steps"] = len(history)
    assert len(history) == 1, "one optimizer step expected"
    # best_parameters only moves when a loss beats the start, so after one step
    # it can stay at the initial scale; the stepped value is in the history.
    stepped = float(history[0]["source_scale"])
    assert stepped != 1.0, "fixture must move source_scale in its one step"
    floats = {
        "baseline_source_proxy_abs_mean": baseline_val,
        "loss_step1": float(history[0]["loss"]),
        "gradient_step1": float(history[0]["gradient"]),
        "source_scale_after_step": stepped,
    }
    assert all(np.isfinite(v) for v in floats.values()), "tune outputs must be finite"
    _check("objective_one_step", exact, floats)
