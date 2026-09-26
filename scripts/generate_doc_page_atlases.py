#!/usr/bin/env python3
"""Generate one dark-theme Plotly atlas per doc-simulation page (docs style rollout).

Each spec builds the page's own circuit at a stated smoke scale, simulates it,
and emits the 7-panel atlas via ``jaxfne.vis.build_atlas`` (which applies the
dark presentation-only theme) under ``docs/_static/atlas/<slug>/``.

Every spec records its provenance label honestly: circuits/runs that differ
from the page's full-scale run say so (``smoke-scale``). Nothing here changes
package semantics; figure styling is presentation-only.

Usage:
    python scripts/generate_doc_page_atlases.py --list
    python scripts/generate_doc_page_atlases.py --slug single_neuron,two_neuron_ei
    python scripts/generate_doc_page_atlases.py            # all atlas specs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ATLAS_ROOT = ROOT / "docs" / "_static" / "atlas"


def _build_atlas(slug: str, title: str, model, signals, **manifest_kw):
    from jaxfne.vis import build_atlas

    out_dir = ATLAS_ROOT / slug
    manifest = build_atlas(model, signals, out_dir=str(out_dir), title=title, **manifest_kw)
    print(f"[{slug}] {title}: sha256={manifest['sha256']}", flush=True)
    for p in manifest["panels"]:
        print(f"    {p['file']}: {p['evidence']} {p['status']} {p['bytes']}B", flush=True)
    return manifest


def spec_single_neuron():
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(n=1)
        .emitter(family="izhikevich", preset="regular_spiking")
        .field(domain="point")
        .probe(name="single_neuron", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    return ("Single neuron (1n)", model, signals, dict(duration_ms=100.0, dt_ms=0.1, seed=0))


def spec_two_neuron_ei():
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(
            n=2,
            cell_types={"E": 1, "PV": 1},
            connectivity={"E→E": 0.1, "E→PV": 0.2, "PV→E": -0.3, "PV→PV": -0.1},
        )
        .emitter(family="izhikevich", preset="regular_spiking")
        .field(domain="point")
        .probe(name="two_neuron_ei", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    # dt 0.5 (binary-exact grid; dt 0.1 float32 grids drift non-uniform over
    # long runs and are rejected by the atlas time-grid gate).
    signals = model.simulate(jtfne.simulation(duration_ms=500.0, dt_ms=0.5, seed=0))
    return ("Two-neuron E/I (2n)", model, signals, dict(duration_ms=500.0, dt_ms=0.5, seed=0))


def spec_network_100_ei():
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(
            name="network_100_ei",
            kind="balanced_ei_population",
            n=100,
            cell_types={"E": 0.75, "PV": 0.25},
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="declared_proxy",
            gauge="mean_zero",
        )
        .probe(
            name="multimodal_100_ei",
            modes=["spikes", "V_m", "source", "phi_e", "J_e", "CSD", "LFP"],
        )
    )
    model = jtfne.construct(cfg)
    signals = model.simulate(jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=42))
    return (
        "Balanced E/I population (100n)",
        model,
        signals,
        dict(duration_ms=100.0, dt_ms=0.1, seed=42),
    )


def spec_v1_column():
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(
            n=600,
            layers=["L1", "L2/3", "L4", "L5", "L6"],
            cell_types={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
            connectivity="layer_structured",
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column", conductivity="proxy", depths=[0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
        )
        .probe(name="v1_column", n_contacts=6, modes=["spikes", "V_m", "LFP", "CSD"])
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
    return (
        "V1 six-layer column (600n)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=0),
    )


def spec_v1_pfc_dual():
    # Single-trial, no-HDP-carryover smoke run of the page's two-area tensor
    # (scripts/v1_pfc_continuous_aaab_smoke_test.py builders).
    import jaxfne as jtfne
    from scripts.v1_pfc_continuous_aaab_smoke_test import (
        build_tensor,
        build_trial_schedule,
        tuning_group_indices,
    )

    tensor = build_tensor()
    model = jtfne.construct_neuronal_tensor(tensor, seed=0, duration_ms=1000.0, dt_ms=0.1)
    groups = tuning_group_indices(model)
    schedule = build_trial_schedule(len(model.neuron_table()), groups)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0, paradigm=schedule)
    return (
        "V1-PFC dual column, single AAAB trial (200n, no HDP carryover)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=0),
    )


def spec_ei_population_100():
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        .column(name="L2/3_column", layers=["L2/3"], n=100)
        .cell_types({"E": 0.75, "PV": 0.25})
        .connectivity()
        .set_emitter(family="izhikevich", preset="cortical_eig")
        .probes(["SPK", "Vm", "source", "LFP-proxy", "CSD-proxy"])
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=42)
    return (
        "Chainable E/I population (100n)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=42),
    )


def spec_suite1_column():
    # Smoke-scale (1000 ms) of the notebook's 48n laminar column (notebook: 5000 ms).
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=44, dtype="float32", duration_ms=1000.0, dt_ms=0.5)
        .column(name="laminar_column", layers=["L2/3", "L4", "L5", "L6"], n=48)
        .cell_types({"E": 0.75, "PV": 0.12, "SST": 0.08, "VIP": 0.05})
        .connectivity(kind="laminar_signed_metadata", recurrent=True)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=16)
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=44)
    return (
        "Suite No. 1 laminar column, smoke-scale (48n, 1000 ms of 5000 ms)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=44),
    )


def spec_suite2_net1():
    import jaxfne as jtfne

    cfg = jtfne.suite2_net1_config(seed=7, n=100, duration_ms=1000.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    bundle = jtfne.suite2_run_bundle(model, seed=7, duration_ms=1000.0, dt_ms=0.5)
    return (
        "Suite No. 2 net1 (100n)",
        model,
        bundle["signals"],
        dict(duration_ms=1000.0, dt_ms=0.5, seed=7),
    )


def spec_evoked_l4():
    # Page's evoked condition; the page's `cfg.paradigm(...)` has no such
    # method, so the paradigm rides on simulate() (the supported path).
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=7, dtype="float32", duration_ms=1500.0, dt_ms=0.5)
        .column("V1_reduced", layers=["L2/3", "L4", "L5"], n=100)
        .drive(baseline_drive_by_cell_type={"E": 8.0, "PV": 4.0})
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "LFP-proxy", "CSD-proxy"])
    )
    paradigm = jtfne.evoked_l4_drive_paradigm(
        l4_onset_ms=500.0,
        l4_duration_ms=200.0,
        l4_amplitude=1.0,
        pre_stimulus_buffer_ms=200.0,
        post_stimulus_buffer_ms=500.0,
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, seed=7, duration_ms=1500.0, dt_ms=0.5, paradigm=paradigm)
    return (
        "Evoked L4 drive, evoked condition (100n)",
        model,
        signals,
        dict(duration_ms=1500.0, dt_ms=0.5, seed=7),
    )


def spec_scale_100():
    # Notebook scale N=100 (seed 2303) with the notebook's async patch.
    import dataclasses

    import jax
    import jaxfne as jtfne

    n, seed = 100, 2303
    n_e = int(round(0.75 * n))
    cfg = (
        jtfne.Configuration()
        .runtime(seed=seed, dtype="float32")
        .column(name="scale_100", layers=["L2/3"], n=n)
        .cell_types({"E": n_e, "PV": n - n_e})
        .connectivity(kind="dense_signed_ei_metadata")
        .emitter(family="izhikevich", preset="cortical_eig")
        .probes(["spk", "vm", "source", "lfp_proxy"])
    )
    model = jtfne.construct(cfg)
    emitter_params = model.params["emitter"]
    n_neurons = emitter_params.n_neurons
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    new_drive = emitter_params.drive + jax.random.normal(k1, shape=(n_neurons,)) * 1.5
    new_v0 = jax.random.uniform(k2, shape=(n_neurons,), minval=-70.0, maxval=-50.0)
    model.params["emitter"] = dataclasses.replace(
        emitter_params,
        drive=new_drive,
        v0=new_v0,
        u0=emitter_params.b * new_v0,
        W=emitter_params.W * 0.05,
    )
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=seed)
    return (
        "Suite No. 3 scale N=100 (async patch)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=seed),
    )


def spec_source_column_48():
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.5)
        .column(name="tutorial_column", layers=["L2/3", "L4", "L5", "L6"], n=48)
        .cell_types({"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05})
        .connectivity(kind="laminar_signed_metadata", recurrent=True)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=16)
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=42)
    return (
        "Source-bookkeeping column (48n)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=42),
    )


def spec_lfp_csd_12():
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.5)
        .column(name="laminar_lfp_csd", layers=["L2/3", "L4", "L5", "L6"], n=12)
        .cell_types({"E": 0.75, "PV": 0.15, "SST": 0.05, "VIP": 0.05})
        .connectivity(kind="laminar_signed_metadata", recurrent=True)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=16)
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=42)
    return (
        "LFP/CSD laminar column (12n)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=42),
    )


def spec_omission_60():
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
        .column("V1_column", layers=["L2/3", "L4", "L5"], n=60)
        .drive(baseline_drive_by_cell_type={"E": 6.5, "PV": 3.0})
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "LFP-proxy", "CSD-proxy"])
    )
    model = jtfne.construct(cfg)
    # Plain-drive run; the page's omission/oddball conditions are declared
    # via jtfne.omission_oddball_paradigm on top of this circuit.
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=42)
    return (
        "Omission-oddball column, plain drive (60n)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=42),
    )


def spec_v1v4_80():
    import jaxfne as jtfne

    cfg = jtfne.suite2_v1_v4_config(seed=42, n_per_area=80, duration_ms=1000.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=42)
    return (
        "V1-V4 scaffold (80/area)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=42),
    )


def spec_canonical_etude_1000():
    import jaxfne as jtfne

    jtfne.enable_x64()
    layers = ["L1", "L2", "L3", "L4", "L5", "L6"]
    zbands = {
        "L1": (0.00, 0.10),
        "L2": (0.10, 0.35),
        "L3": (0.35, 0.55),
        "L4": (0.55, 0.65),
        "L5": (0.65, 0.85),
        "L6": (0.85, 1.00),
    }
    fractions = {
        "L1": {"E": 0.50, "PV": 0.05, "SST": 0.10, "VIP": 0.35},
        "L2": {"E": 0.50, "PV": 0.25, "SST": 0.10, "VIP": 0.15},
        "L3": {"E": 0.50, "PV": 0.25, "SST": 0.15, "VIP": 0.10},
        "L4": {"E": 0.70, "PV": 0.20, "SST": 0.05, "VIP": 0.05},
        "L5": {"E": 0.85, "PV": 0.05, "SST": 0.05, "VIP": 0.05},
        "L6": {"E": 0.95, "PV": 0.00, "SST": 0.05, "VIP": 0.00},
    }
    cfg = (
        jtfne.laminar_cortex_config(
            seed=0,
            duration_ms=1000.0,
            dt_ms=0.5,
            areas=["V1"],
            layers=layers,
            n=1000,
            emitter="izhikevich",
            baseline_drive_by_cell_type={"E": 5.0, "PV": 5.0, "SST": 5.0, "VIP": 5.0},
        )
        .layer_fractions(layer_fractions=zbands)
        .area_layer_cell_types("V1", fractions)
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, jtfne.simulation(duration_ms=1000.0, dt_ms=0.5, seed=0))
    return (
        "Canonical-column etude, uniform drive (1000n)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=0),
    )


def spec_hdp_10():
    # Small-mechanistic full-recording run: HDP on with weight-trace
    # recording (tiny edge count, so full T×E fits the plot budget).
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=10, cell_types={"E": 0.5, "PV": 0.5})
        .drive(baseline_drive_by_cell_type={"E": 8.0, "PV": 8.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    runtime = jtfne.RuntimeConfig(
        enable_hdp=True,
        recurrent_backend="edge_list",
        jit=False,
        hdp_params={
            "K_HDP": 0.01,
            "tau_0_ms": 200.0,
            "K_ctrl": 5.0,
            "barrier_c": 0.01,
            "barrier_d": 0.01,
            "H_min": 0.1,
            "H_max": 10.0,
            "w_min": -10.0,
            "w_max": 10.0,
        },
    )
    sim = jtfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0, runtime=runtime)
    signals = model.simulate(sim)
    assert model.last_hdp_diagnostics() is not None
    assert model.last_hdp_diagnostics().get("w_trace") is not None
    return (
        "HDP circuit, full recording (10n, 100 ms)",
        model,
        signals,
        dict(duration_ms=100.0, dt_ms=0.5, seed=0),
    )


def spec_hdp_1000():
    # Short HDP run; weight-trace recording off (memory: steps x edges).
    import jaxfne as jtfne

    jtfne.enable_x64()
    cfg = (
        jtfne.build_laminar_column(n=1000)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"])
        .field(domain="laminar_column", conductivity="proxy")
        .runtime(
            duration_ms=200.0,
            dt_ms=0.5,
            seed=0,
            enable_hdp=True,
            hdp_params={
                "K_HDP": 0.01,
                "tau_0_ms": 200.0,
                "K_ctrl": 5.0,
                "barrier_c": 0.01,
                "barrier_d": 0.01,
                "record_weight_trace": False,
            },
        )
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model)
    assert model.last_hdp_diagnostics() is not None
    return (
        "HDP column, short run (1000n, 200 ms)",
        model,
        signals,
        dict(duration_ms=200.0, dt_ms=0.5, seed=0),
    )


def spec_homeostasis_1000():
    import jaxfne as jtfne

    jtfne.enable_x64()
    cfg = (
        jtfne.build_laminar_column(n=1000)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"])
        .field(domain="laminar_column", conductivity="proxy")
        .runtime(
            duration_ms=1000.0,
            dt_ms=0.5,
            seed=0,
            enable_homeostasis=True,
            homeostasis_params={"k_gain": 1.0, "r_star": 0.01},
        )
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model)
    return (
        "Homeostasis column (1000n)",
        model,
        signals,
        dict(duration_ms=1000.0, dt_ms=0.5, seed=0),
    )


def spec_calibration_100():
    # Page's simulate call is a placeholder (`...`); completed minimally here.
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(n=100)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            depths=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(
            name="calibration_ready", n_contacts=6, contact_depths=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        )
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)
    return (
        "Calibration-ready column, minimal run (100n, 200 ms)",
        model,
        signals,
        dict(duration_ms=200.0, dt_ms=0.5, seed=0),
    )


def spec_objective_60():
    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=200.0, dt_ms=0.5)
        .column(name="V1", layers=["L1", "L4", "L6"], n=60)
        .cell_types({"E": 0.8, "PV": 0.2})
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=8)
    )
    paradigm = jtfne.omission_oddball_paradigm(standard_onset_ms=50.0, standard_duration_ms=20.0)
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=1, paradigm=paradigm)
    return (
        "Objective-grammar chain, pre-tune (60n)",
        model,
        signals,
        dict(duration_ms=200.0, dt_ms=0.5, seed=1),
    )


def spec_operator_chain_40():
    import jaxfne as jtfne

    cfg = jtfne.laminar_cortex_config(
        seed=7,
        duration_ms=100.0,
        dt_ms=0.5,
        areas=("V1",),
        layers=("L1", "L4", "L6"),
        cell_types={"E": 0.6, "PV": 0.25, "SST": 0.15},
        n=40,
        emitter="izhikevich",
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.5, seed=3)
    return (
        "Operator-composition column (40n)",
        model,
        signals,
        dict(duration_ms=100.0, dt_ms=0.5, seed=3),
    )


def spec_probe_32():
    # The guide's snippet is aspirational (`...`); completed minimally here.
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(n=32)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy")
        .probe(name="probe_32", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)
    return (
        "Probe-operators circuit, minimal run (32n, 200 ms)",
        model,
        signals,
        dict(duration_ms=200.0, dt_ms=0.5, seed=0),
    )


def spec_config_grammar_1000():
    # Smoke-scale (200 ms) of the guide's 1000n head config (1000 ms).
    import jaxfne as jtfne

    jtfne.enable_x64()
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=200.0, dt_ms=0.5)
        .column(name="V1", layers=["L1", "L2/3", "L4", "L5", "L6"], n=1000)
        .geometry(layer_thickness={"L1": 0.1, "L2/3": 0.3, "L4": 0.2, "L5": 0.3, "L6": 0.3})
        .cell_types({"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
        .connectivity(mode="sparse")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy")
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)
    return (
        "Configuration-grammar column, smoke-scale (1000n, 200 ms)",
        model,
        signals,
        dict(duration_ms=200.0, dt_ms=0.5, seed=0),
    )


SPECS = {
    "single_neuron": spec_single_neuron,
    "two_neuron_ei": spec_two_neuron_ei,
    "network_100_ei": spec_network_100_ei,
    "v1_column": spec_v1_column,
    "v1_pfc_dual": spec_v1_pfc_dual,
    "ei_population_100": spec_ei_population_100,
    "suite1_column": spec_suite1_column,
    "suite2_net1": spec_suite2_net1,
    "evoked_l4": spec_evoked_l4,
    "scale_100": spec_scale_100,
    "source_column_48": spec_source_column_48,
    "lfp_csd_12": spec_lfp_csd_12,
    # eeg_meg_emm_100 removed: byte-identical dynamics to ei_population_100
    # (extra probe modes change no panel); tutorial 09 links the shared slug.
    "omission_60": spec_omission_60,
    "v1v4_80": spec_v1v4_80,
    "canonical_etude_1000": spec_canonical_etude_1000,
    "hdp_10": spec_hdp_10,
    "hdp_1000": spec_hdp_1000,
    "homeostasis_1000": spec_homeostasis_1000,
    "calibration_100": spec_calibration_100,
    "objective_60": spec_objective_60,
    "operator_chain_40": spec_operator_chain_40,
    "probe_32": spec_probe_32,
    "config_grammar_1000": spec_config_grammar_1000,
}


def spec_jaxley_single():
    """Standalone dark vm panel for the Jaxley bridge (no Model -> no atlas)."""
    import jaxley as jx
    from jaxley.channels import HH

    import jaxfne as jtfne
    from jaxfne.vis import canonical as C

    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.insert(HH())
    cell.record("v")
    cell.stimulate(jx.step_current(i_delay=10, i_dur=50, i_amp=0.1, delta_t=0.025, t_max=100))
    sig = jtfne.JaxleyBridge(model=cell).simulate(duration_ms=100.0, dt_ms=0.025)
    fig = C.plot_membrane_potentials(
        sig, backend="plotly", title="Jaxley HH single compartment — V_m proxy"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
    )
    out = ROOT / "docs" / "_static" / "jaxley_interop" / "vm_dark.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")
    print(
        f"[jaxley_single] wrote {out.relative_to(ROOT)} "
        f"({out.stat().st_size}B) V_m={tuple(sig.V_m.shape)}",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--slug", default="", help="comma-separated subset (plus 'jaxley_single')")
    args = parser.parse_args(argv)

    names = list(SPECS) + ["jaxley_single"]
    if args.list:
        print("\n".join(names))
        return 0

    wanted = (
        [s.strip() for s in args.slug.split(",") if s.strip()]
        if args.slug
        else [s for s in names if s != "jaxley_single"]
    )
    unknown = [s for s in wanted if s not in names]
    if unknown:
        raise SystemExit(f"unknown slugs: {unknown}\nchoose from: {names}")

    failures = []
    for slug in wanted:
        if slug == "jaxley_single":
            try:
                spec_jaxley_single()
            except Exception as exc:  # noqa: BLE001
                failures.append((slug, repr(exc)))
            continue
        try:
            title, model, signals, kw = SPECS[slug]()
            _build_atlas(slug, title, model, signals, **kw)
        except Exception as exc:  # noqa: BLE001
            print(f"[{slug}] FAILED: {exc!r}", flush=True)
            failures.append((slug, repr(exc)))
    if failures:
        print("failures:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
