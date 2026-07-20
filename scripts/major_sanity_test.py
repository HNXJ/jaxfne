"""Minimal (100ms/dt=0.1ms) max-API-coverage HDP/TFNE-Izhikevich E/I sanity test.

Built action-by-action per plans.json:major-sanity-test-20-action-breakdown --
each ACTION_N function is independently runnable and independently verified
before the next one is added. Do not add ACTION_2+ without a real
command/output receipt on the current last action first.

Run: PYTHONPATH=. python3 scripts/major_sanity_test.py
"""
from __future__ import annotations

import tempfile

import numpy as np

import jaxfne as jtfne


def action_1_minimal_tensor() -> jtfne.NeuronalTensor:
    """8 neurons (6E/2I), one flat layer, plastic=PlasticParams(H=1.0) on every
    connection -- regression-checks the PlasticParams.H=0.0 default footgun
    (an unset H seeds that neuron's initial HDP state outside the valid
    [H_min, H_max] range once HDP is enabled; see skills/FRICTIONS_STACK.md F-031)."""
    return jtfne.make_minimal_ei_tensor()


def action_2_bridge_to_configuration(tensor: jtfne.NeuronalTensor) -> jtfne.Configuration:
    """Bridge to Configuration via neuronal_tensor_to_configuration, then null
    construct()'s default dense same-area connectivity generator
    (within_gain=0.0) -- regression-checks the double-wiring landmine (F-031):
    without this, construct() layers its own default random recurrent wiring
    on top of the tensor's explicit InterConnection edges, silently doubling
    edge count. Edge-count verification itself happens at Action 3 (construct())."""
    return jtfne.neuronal_tensor_to_configuration(
        tensor, seed=0, duration_ms=100.0, dt_ms=0.1
    ).connectivity(within_area="all_to_all_uniform_random", within_gain=0.0)


def action_3_construct(cfg: jtfne.Configuration) -> jtfne.Model:
    """construct() -> Model; inspect neuron_table() + emitter params for sane
    Izhikevich ranges, and confirm the edge_list is NOT double-wired (Action
    2's within_gain=0.0 fix) -- 8 neurons (6E/2I), all-to-all-minus-self-loops
    across 4 InterConnection rules should yield exactly 6*5 + 2*1 + 6*2 + 2*6
    = 56 edges, not roughly double that."""
    return jtfne.construct(cfg)


def action_4_enable_hdp(model: jtfne.Model, sim):
    """Enable HDP via an explicit RuntimeConfig(enable_hdp=True,
    hdp_params=DEFAULT_HDP) on Simulation.runtime; confirm Model.simulate()
    needs it explicit -- regression-checks the Model.simulate()/jtfne.simulate()
    inheritance gap (F-031/AGENTS.md): unlike top-level jtfne.simulate(), which
    derives runtime from Configuration.hdp(...)/.metadata automatically,
    Model.simulate() ignores cfg.metadata entirely and needs an explicit
    runtime= kwarg on Simulation. Returns
    (signals_without_hdp, diag_without_hdp, signals_with_hdp, diag_with_hdp) --
    diagnostics are captured immediately after each call since
    Model.last_hdp_diagnostics() reflects only the MOST RECENT simulate() call."""
    from dataclasses import replace as _replace

    signals_without_hdp = model.simulate(sim)
    diag_without_hdp = model.last_hdp_diagnostics()

    sim_with_hdp = _replace(sim, runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=dict(jtfne.DEFAULT_HDP)))
    signals_with_hdp = model.simulate(sim_with_hdp)
    diag_with_hdp = model.last_hdp_diagnostics()

    return signals_without_hdp, diag_without_hdp, signals_with_hdp, diag_with_hdp


def action_5_check_baseline_signals(signals) -> None:
    """Baseline simulate() (100ms/dt=0.1ms), HDP off -- finite V_m/spikes, sane
    physiological ranges (|Vm| > 150 mV or NaN/Inf = blowup, per
    skills/neuro-biophysics-units-sanity's own sanity gate)."""
    V_m = np.asarray(signals.V_m)
    spikes = np.asarray(signals.spikes)
    assert V_m.shape == (1000, 8), V_m.shape  # 100ms / 0.1ms = 1000 steps, 8 neurons
    assert np.all(np.isfinite(V_m)), "V_m must be finite -- NaN/Inf indicates a blowup"
    assert np.all(np.abs(V_m) < 150.0), f"|V_m| > 150mV indicates a blowup, got max {np.abs(V_m).max()}"
    assert np.all(np.isfinite(spikes))
    assert set(np.unique(spikes).tolist()) <= {0.0, 1.0}, "spikes must be a 0/1 boolean-like array"


def action_6_check_hdp_signals(diag_hdp: dict) -> None:
    """Same simulate, HDP on -- last_hdp_diagnostics() populated, H starts at
    1.0, stays in [H_min, H_max] = [0.1, 10.0] (jaxfne/hdp_network.py's
    BASE_HDP_KWARGS_DEFAULT) for the full 1000-step trace, not just the final
    step."""
    H_trace = np.asarray(diag_hdp["H_trace"])
    assert H_trace.shape == (1000, 8)
    assert np.all(np.isfinite(H_trace))
    assert np.allclose(H_trace[0], 1.0, atol=1e-6), f"H must start at equilibrium 1.0, got {H_trace[0]}"
    H_MIN, H_MAX = 0.1, 10.0
    assert np.all(H_trace >= H_MIN) and np.all(H_trace <= H_MAX), (
        f"H left [{H_MIN}, {H_MAX}] -- min={H_trace.min()}, max={H_trace.max()}"
    )


def action_7_check_target_indices(model: jtfne.Model, sim) -> None:
    """Inject a StimulusSchedule with target_indices on a neuron subset --
    confirms per-neuron targeting works. target_indices lives on the EVENT
    dict, not the schedule constructor (see AGENTS.md's "Per-event layer
    targeting" note). Targets only the 2 PV neurons (indices 6, 7)."""
    schedule = jtfne.StimulusSchedule(
        events=({"onset_ms": 10.0, "duration_ms": 50.0, "amplitude": 20.0,
                 "is_drive_event": True, "target_indices": [6, 7]},),
        n_neurons=8,
    )
    arr = np.asarray(schedule.to_array(n_steps=sim.n_steps, dt_ms=sim.dt_ms))
    nonzero_cols = set(np.where(np.any(arr != 0, axis=0))[0].tolist())
    assert nonzero_cols == {6, 7}, f"expected drive only on neurons 6,7 (PV), got {nonzero_cols}"


def action_8_check_three_read_paths(signals) -> None:
    """Read Signals 3 ways (.get('vm'), attribute access, selector filter) --
    all agree."""
    via_get = np.asarray(signals.get("vm"))
    via_attr = np.asarray(signals.V_m)
    assert np.array_equal(via_get, via_attr), ".get('vm') must match .V_m exactly"

    via_selector = np.asarray(signals.get("vm", cell_type="PV"))
    assert via_selector.shape == (1000, 2)  # 2 PV neurons out of 8
    np.testing.assert_array_equal(via_selector, via_attr[:, 6:8])


def action_9_check_rate_and_kappa(signals) -> tuple:
    """Compute rate_hz/kappa_synchrony via real free functions (not invented
    sig.rate()) -- jtfne.tutorial_utils.population_rate_hz and
    jtfne.kappa_synchrony."""
    spikes = np.asarray(signals.spikes)
    rate_hz = jtfne.tutorial_utils.population_rate_hz(spikes, dt_ms=0.1)
    kappa = jtfne.kappa_synchrony(spikes, dt_ms=0.1)
    assert isinstance(rate_hz, float) and np.isfinite(rate_hz)
    assert rate_hz >= 0.0
    assert np.isfinite(kappa)
    return rate_hz, kappa


def action_10_check_probe_modes(model: jtfne.Model, signals) -> dict:
    """model.probe(sig, modes=[...]) across spikes/V_m/CSD/LFP."""
    out = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
    assert set(out["requested_modes"]) == {"spikes", "V_m", "CSD", "LFP"}
    assert np.asarray(out["spikes"]).shape == (1000, 8)
    assert np.asarray(out["V_m"]).shape == (1000, 8)
    assert "csd_proxy" in out and np.all(np.isfinite(np.asarray(out["csd_proxy"])))
    assert "lfp_proxy" in out and np.all(np.isfinite(np.asarray(out["lfp_proxy"])))
    return out


def action_11_check_eeg_proxy_manual(signals):
    """Derive EEG proxy manually via jtfne.fields.eeg_proxy_probe(lfp) --
    confirms it's not auto-produced by simulate()/model.probe()."""
    try:
        signals.get("eeg")
        raise AssertionError("expected KeyError -- eeg_proxy must not be auto-produced")
    except KeyError:
        pass

    lfp = signals.get("lfp")
    eeg_readout = jtfne.fields.eeg_proxy_probe(lfp)
    assert eeg_readout.data is not None
    assert np.all(np.isfinite(np.asarray(eeg_readout.data)))
    return eeg_readout


def action_12_check_objective_evaluate(model: jtfne.Model, signals):
    """Build an Objective (rate target), model.evaluate(sig, obj) --
    losses/gates report correctly."""
    objective = jtfne.objective().loss("rate_loss", metric="spike_rate_hz_mean", target=15.0)
    report = model.evaluate(signals, objective)
    assert report["evaluation_status"] == "objective_evaluate_v0.0.5"
    assert len(report["losses"]) == 1
    assert report["losses"][0]["status"] == "ok"
    assert report["losses"][0]["metric"] == "spike_rate_hz_mean"
    assert report["total_loss"] is not None
    assert report["physical_amplitude_calibrated"] is False
    assert report["claim_level"] == "computational_scaffold"
    return report


def action_13_check_tune_blackbox_and_stub(model: jtfne.Model, sim):
    """model.tune(obj, optimizer='AGSDR', steps=2) -- blackbox loop runs for
    real; confirms differentiable-path stub still correctly no-ops for an
    optax optimizer with the default (not_checked) differentiability status."""
    objective = jtfne.objective().loss("rate_loss", metric="spike_rate_hz_mean", target=15.0)

    agsdr_result = model.tune(objectives=objective, optimizer="AGSDR", simulation=sim, steps=2)
    assert agsdr_result.summary["tuning_status"] == "blackbox_loop_v0.0.6"
    assert agsdr_result.best_parameters, "AGSDR must produce a real best candidate, not an empty dict"

    optax_result = model.tune(objectives=objective, optimizer=jtfne.optax_adam(), simulation=sim, steps=2)
    assert optax_result.summary["tuning_status"] == "blocked_non_differentiable_path"
    assert optax_result.best_parameters == {}
    return agsdr_result, optax_result


def action_14_check_wee_wei_wie_wii(model: jtfne.Model) -> dict:
    """Extract edge_list, compute Wee/Wei/Wie/Wii aggregates -- confirms edge
    counts (30/12/12/2, matching Action 3's no-double-wiring count) and signs
    (E excitatory positive, PV inhibitory negative)."""
    edge_list = model.params["edge_list"]
    table = model.neuron_table()
    cell_types = np.array([row["cell_type"] for row in table])
    pre = np.asarray(edge_list.pre)
    post = np.asarray(edge_list.post)
    weight = np.asarray(edge_list.weight)

    aggregates = {}
    expected_counts = {"E_E": 30, "E_PV": 12, "PV_E": 12, "PV_PV": 2}
    for src in ("E", "PV"):
        for tgt in ("E", "PV"):
            mask = (cell_types[pre] == src) & (cell_types[post] == tgt)
            key = f"{src}_{tgt}"
            n = int(mask.sum())
            mean_w = float(weight[mask].mean()) if n > 0 else float("nan")
            aggregates[key] = {"n_edges": n, "mean_weight": mean_w}
            assert n == expected_counts[key], f"{key}: expected {expected_counts[key]} edges, got {n}"
            expected_sign = 1.0 if src == "E" else -1.0
            assert np.sign(mean_w) == expected_sign, f"{key}: expected sign {expected_sign}, got mean_weight={mean_w}"
    return aggregates


def action_15_check_h_per_cell_type(model: jtfne.Model, diag_hdp: dict) -> dict:
    """last_hdp_diagnostics()'s H per cell-type -- respects H_min/H_max =
    [0.1, 10.0] and both E and PV groups stay near equilibrium under
    DEFAULT_HDP."""
    table = model.neuron_table()
    cell_types = np.array([row["cell_type"] for row in table])
    H_final = np.asarray(diag_hdp["H_final"])
    H_MIN, H_MAX = 0.1, 10.0

    per_cell_type = {}
    for ct in ("E", "PV"):
        mask = cell_types == ct
        vals = H_final[mask]
        assert vals.size > 0
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= H_MIN) and np.all(vals <= H_MAX)
        per_cell_type[ct] = {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}
    return per_cell_type


def action_16_check_manifest_save_json(model: jtfne.Model, signals, eval_report: dict, tmp_path: str):
    """jtfne.manifest(cfg, signals=sig, objective=obj) + save_json -- JSON-safe,
    zero NaN."""
    import json

    m = jtfne.manifest(model.cfg, signals=signals, evaluation=eval_report)
    jtfne.io.save_json(m, tmp_path)
    with open(tmp_path) as f:
        loaded = json.load(f)
    assert set(loaded.keys()) == set(m.keys())
    serialized = json.dumps(m)
    assert "NaN" not in serialized and "Infinity" not in serialized
    assert m["physical_amplitude_calibrated"] is False
    assert m["claim_level"] == "computational_scaffold"
    return m


def action_17_check_run_receipt_write_once(model: jtfne.Model, signals, receipt_path: str):
    """model.run_receipt() + save_receipt(), then attempt a second save
    without overwrite=True -- write-once guard actually raises (real
    ValueError, not silently overwriting or a FileExistsError)."""
    receipt = model.run_receipt(signals)
    jtfne.io.save_receipt(receipt, receipt_path)

    try:
        jtfne.io.save_receipt(receipt, receipt_path)
        raise AssertionError("expected ValueError -- write-once guard must raise on re-save")
    except ValueError as e:
        assert "receipt_file_exists" in str(e)
    return receipt


def action_18_check_config_hash_determinism():
    """config_hash on the manifest -- deterministic given identical config
    (rebuild the tensor/cfg from scratch twice, same hash both times)."""
    tensor1 = action_1_minimal_tensor()
    cfg1 = action_2_bridge_to_configuration(tensor1)
    model1 = action_3_construct(cfg1)

    tensor2 = action_1_minimal_tensor()
    cfg2 = action_2_bridge_to_configuration(tensor2)
    model2 = action_3_construct(cfg2)

    h1 = jtfne.io.config_hash(model1.cfg)
    h2 = jtfne.io.config_hash(model2.cfg)
    assert h1 == h2, f"config_hash must be deterministic for identical configs: {h1} != {h2}"

    m1 = jtfne.manifest(model1.cfg)
    assert m1.get("config_hash") == h1
    return h1


def action_19_check_vis_functions(model: jtfne.Model, signals) -> list:
    """jtfne.vis.raster/vm/rate/lfp/psd/connectivity(sig) -- every plot
    function returns cleanly (a real matplotlib Figure, not an exception)."""
    import matplotlib
    matplotlib.use("Agg")

    figs = []
    for name in ("raster", "vm", "rate", "lfp", "psd"):
        fig = getattr(jtfne.vis, name)(signals)
        assert fig is not None
        figs.append((name, fig))
    conn_fig = jtfne.vis.connectivity(model)
    assert conn_fig is not None
    figs.append(("connectivity", conn_fig))
    return figs


def action_20_check_recompilation_guard(model: jtfne.Model) -> dict:
    """Run simulate() twice with identical shapes, check
    make_recompilation_guard/CompilationRegistry -- N_compile<=1, ties
    directly to this session's Factor 1 fix (a module-level
    CompilationRegistry singleton with no per-test reset)."""
    compilation_registry = jtfne.compilation_registry
    compilation_registry.reset()
    compilation_registry.set_mode("exception")
    try:
        sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0, runtime=jtfne.RuntimeConfig(jit=True))
        model.simulate(sim)
        model.simulate(sim)  # identical shapes -- must not trigger a recompile alert
        traced = dict(compilation_registry.traced_signatures)
    finally:
        compilation_registry.reset()
        compilation_registry.set_mode("warning")

    for (name, signature), count in traced.items():
        assert count <= 1, f"N_compile > 1 for {name}{signature}: traced {count} times across 2 identical-shape calls"
    return traced


if __name__ == "__main__":
    tensor = action_1_minimal_tensor()
    layer = tensor.areas[0].layers[0]
    assert layer.n_neurons == 8
    assert [nt.name for nt in layer.neuron_types] == ["E", "PV"]
    assert abs(layer.neuron_types[0].fraction - 0.75) < 1e-9  # 6/8
    assert abs(layer.neuron_types[1].fraction - 0.25) < 1e-9  # 2/8
    assert len(tensor.areas[0].inter_connections) == 4
    assert all(c.plastic.H == 1.0 for c in tensor.areas[0].inter_connections)
    print("ACTION 1 OK: 8 neurons (6E/2I), 4 InterConnections, all H=1.0")

    cfg = action_2_bridge_to_configuration(tensor)
    assert cfg.metadata["connectivity"]["within_gain"] == 0.0
    assert cfg.metadata["connectivity"]["within_area"] == "all_to_all_uniform_random"
    assert cfg.networks and cfg.networks[0]["connectivity"]["within_gain"] == 0.0
    print("ACTION 2 OK: bridged to Configuration, within_gain nulled to 0.0")

    model = action_3_construct(cfg)
    table = model.neuron_table()
    assert len(table) == 8
    assert [row["cell_type"] for row in table] == ["E"] * 6 + ["PV"] * 2
    emitter = model.params["emitter"]
    # Sane Izhikevich regular-spiking (E) / fast-spiking (PV) parameter ranges,
    # not NaN/blowup: a in (0, 1), c around -65mV, d > 0.
    a, c, d = np.asarray(emitter.a), np.asarray(emitter.c), np.asarray(emitter.d)
    assert np.all((a > 0.0) & (a < 1.0)), a
    assert np.all((c > -100.0) & (c < -50.0)), c
    assert np.all(d > 0.0), d
    n_edges = len(model.params["edge_list"].pre)
    expected_edges = 6 * 5 + 2 * 1 + 6 * 2 + 2 * 6  # E->E + PV->PV (no self-loops) + E->PV + PV->E
    assert n_edges == expected_edges, f"expected {expected_edges} edges (no double-wiring), got {n_edges}"
    print(f"ACTION 3 OK: construct() -> Model, 8-row neuron_table, sane Izhikevich params, "
          f"{n_edges} edges (matches {expected_edges}, confirms no double-wiring)")

    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0)
    sig_no_hdp, diag_no_hdp, sig_hdp, diag_hdp = action_4_enable_hdp(model, sim)
    assert diag_no_hdp is None, "HDP must be OFF by default without an explicit runtime= kwarg"
    assert diag_hdp is not None, "HDP must be ON with an explicit RuntimeConfig(enable_hdp=True)"
    H_final = np.asarray(diag_hdp["H_final"])
    assert H_final.shape == (8,)
    assert np.all(np.isfinite(H_final))
    assert np.allclose(H_final, 1.0, atol=0.01), f"H should start near equilibrium 1.0, got {H_final}"
    print(f"ACTION 4 OK: HDP off by default (diag=None), on with explicit runtime= "
          f"(H_final near 1.0: {H_final})")

    action_5_check_baseline_signals(sig_no_hdp)
    print("ACTION 5 OK: baseline simulate() (HDP off) -- finite V_m/spikes, sane physiological ranges")

    action_6_check_hdp_signals(diag_hdp)
    print("ACTION 6 OK: HDP-on run -- H starts at 1.0, full 1000-step trace stays in [0.1, 10.0]")

    action_7_check_target_indices(model, sim)
    print("ACTION 7 OK: target_indices restricts drive to exactly the targeted neuron subset (6, 7)")

    action_8_check_three_read_paths(sig_no_hdp)
    print("ACTION 8 OK: .get('vm'), attribute access, and cell_type selector filter all agree")

    rate_hz, kappa = action_9_check_rate_and_kappa(sig_no_hdp)
    print(f"ACTION 9 OK: rate_hz={rate_hz:.2f}Hz, kappa_synchrony={kappa:.4f} via real free functions")

    probe_out = action_10_check_probe_modes(model, sig_no_hdp)
    print(f"ACTION 10 OK: model.probe() across spikes/V_m/CSD/LFP -- keys: {sorted(probe_out.keys())}")

    eeg_readout = action_11_check_eeg_proxy_manual(sig_no_hdp)
    print(f"ACTION 11 OK: EEG proxy derived manually via eeg_proxy_probe(lfp), shape "
          f"{tuple(eeg_readout.data.shape)}, not auto-produced")

    eval_report = action_12_check_objective_evaluate(model, sig_no_hdp)
    print(f"ACTION 12 OK: Objective(rate_loss target=15.0) -> total_loss={eval_report['total_loss']:.4f}")

    agsdr_result, optax_result = action_13_check_tune_blackbox_and_stub(model, sim)
    print(f"ACTION 13 OK: AGSDR blackbox loop real (best_score={agsdr_result.best_score:.4f}), "
          f"optax stub correctly blocked ({optax_result.summary['tuning_status']})")

    wagg = action_14_check_wee_wei_wie_wii(model)
    print(f"ACTION 14 OK: Wee/Wei/Wie/Wii aggregates -- {wagg}")

    h_per_ct = action_15_check_h_per_cell_type(model, diag_hdp)
    print(f"ACTION 15 OK: H per cell-type within [0.1, 10.0] -- {h_per_ct}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        manifest_path = tf.name
    m = action_16_check_manifest_save_json(model, sig_no_hdp, eval_report, manifest_path)
    print(f"ACTION 16 OK: manifest saved+reloaded JSON-safe, zero NaN, {len(m)} keys")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tf:
        receipt_path = tf.name
    receipt = action_17_check_run_receipt_write_once(model, sig_no_hdp, receipt_path)
    print(f"ACTION 17 OK: run_receipt saved, second save without overwrite=True correctly raised "
          f"ValueError (write-once guard), receipt_id={receipt.receipt_id[:12]}...")

    config_hash = action_18_check_config_hash_determinism()
    print(f"ACTION 18 OK: config_hash deterministic across 2 independently-rebuilt identical configs: "
          f"{config_hash}")

    figs = action_19_check_vis_functions(model, sig_no_hdp)
    print(f"ACTION 19 OK: all {len(figs)} vis functions returned cleanly -- {[n for n, _ in figs]}")

    traced = action_20_check_recompilation_guard(model)
    print(f"ACTION 20 OK: N_compile<=1 across 2 identical-shape simulate() calls -- {traced}")
    print()
    print("ALL 20 ACTIONS COMPLETE.")
