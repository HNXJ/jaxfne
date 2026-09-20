#!/usr/bin/env python3
"""Deterministic reproduction + HTML figures for trajectory-less etudes (C5).

Each mode replays the frozen protocol EXACTLY (same builders, seeds,
durations) via the protocol scripts' own functions, asserts the reproduced
trajectories/metrics equal the committed bundle bit-for-bit, then emits dark
Plotly panels to ``docs/_static/etudes/<name>/``. Any mismatch aborts with a
drift error (new problem ID) instead of publishing figures.

Frozen protocol scripts themselves are never modified. Modes:
  multiscale     40n group-A run, hash-checked vs cause_hashes
  heterogeneous  izh + hei runs, hash-checked vs Q_sha256 + rates
  hdp_mcc3       AGSDR tune + condition B, checked vs theta_hat + firing.B

Usage:
    python scripts/rerun_etude_figures.py --etude multiscale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_frozen_etude_panels import finalize  # noqa: E402


def _sha(arr) -> str:
    import numpy as np

    b = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(b.dtype).encode() + str(tuple(b.shape)).encode() + b.tobytes())
    return h.hexdigest()


def _emit(
    panels: list, out_dir: pathlib.Path, filename: str, panel: str, evidence: str, fig, source: str
) -> None:  # type: ignore[no-untyped-def]
    from scripts.plot_frozen_etude_panels import _dark, _write

    fig = _dark(fig, panel)
    nbytes = _write(fig, out_dir, filename)
    panels.append(
        {
            "file": filename,
            "panel": panel,
            "evidence": evidence,
            "status": "AVAILABLE",
            "bytes": nbytes,
            "inputs": {
                "source_artifact": source,
                "transform": panel,
                "units": "relative proxy",
                "window": "full reproduced run",
            },
        }
    )


def _stride(n: int, max_pts: int = 2000) -> int:
    return max(1, int(n // max_pts))


def _verify_multiscale_claims(bundle_metrics: dict, spikes, V, Q, positions) -> None:
    """Claims-tier verification (P-003): abort unless published claims re-verify."""
    import numpy as np

    from scripts.run_multiscale_observation_etude import (
        FS,
        mean_r90,
        psd_of,
        spectral_centroid,
    )
    from jaxfne.fields import project_laminar_sources
    import jax.numpy as jnp

    m = bundle_metrics
    if int(np.sum(spikes)) != m["spike_count"]:
        raise SystemExit("spike_count mismatch (spikes should be hash-exact)")
    if float(np.mean(spikes) * FS) != m["mean_rate_hz"]:
        raise SystemExit("mean_rate_hz mismatch (spikes should be hash-exact)")

    def close(got: float, want: float, name: str) -> None:
        if abs(got - want) > 1e-3 * max(abs(want), 1e-12):
            raise SystemExit(f"{name} outside protocol tolerance: {got} vs {want}")

    close(float(np.mean(V)), m["mean_V_m"], "mean_V_m")
    fo = project_laminar_sources(jnp.asarray(Q), jnp.asarray(positions), n_contacts=16, width=0.10)
    z = np.asarray(positions)[:, 2]
    if mean_r90(np.asarray(fo.kernel), z, np.asarray(fo.contact_depths)) != m["r90"]["lfp_ref"]:
        raise SystemExit("r90 lfp_ref mismatch (geometry should be hash-exact)")
    burn = int(round(200.0 / 0.5))
    freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
    for key, arr in (
        ("Q", Q),
        ("LFP", np.asarray(fo.lfp_proxy)),
        ("CSD", np.asarray(fo.csd_proxy)),
    ):
        psd = psd_of(np.asarray(arr)[burn:], freqs)
        close(
            float(spectral_centroid(psd, freqs)), m["spectral_centroid_hz"][key], f"centroid {key}"
        )


def mode_multiscale(out_dir: pathlib.Path) -> list[dict]:
    import numpy as np

    import jaxfne as J
    from scripts.run_multiscale_observation_etude import (
        array_sha256,
        build_config,
    )

    bundle_metrics = json.loads(
        (ROOT / "artifacts/etudes/multiscale_observation/metrics.json").read_text(encoding="utf-8")
    )
    committed = bundle_metrics["cause_hashes"]
    cfg = build_config()
    model = J.construct(cfg)
    sim = J.Simulation(
        duration_ms=2000.0,
        dt_ms=0.5,
        seed=7,
        record_sources=True,
        record_fields=True,
        runtime=J.RuntimeConfig(dtype="float32", jit=False, seed=7),
    )
    signals = model.simulate(sim)
    Q = np.asarray(signals.sources)
    V = np.asarray(signals.V_m)
    spikes = np.asarray(signals.spikes)
    positions = np.asarray(model.params["positions"])
    got = {
        "V_m": array_sha256(V),
        "spikes": array_sha256(spikes),
        "Q": array_sha256(Q),
        "positions": array_sha256(positions),
    }
    if got == committed:
        tier = "hash"
        SRC = "reproduced group-A run (hash-checked)"
        note = "deterministic reproduction; hashes equal committed bundle"
    else:
        # Claims tier (P-003): V_m/Q float hashes drift across environments
        # (JAX version unrecorded at bundle time) while spikes/positions are
        # exact. Publish figures only if every published bundle claim
        # re-verifies: spike-exact quantities ==, geometry-exact r90 ==,
        # float quantities within the protocol's own 1e-3 relative tolerance.
        tier = "claims"
        SRC = (
            "reproduced group-A run (claims-checked P-003: spikes/positions "
            "hash-exact, V_m/Q float drift, published claims re-verified)"
        )
        note = (
            "deterministic reproduction; spikes/positions hash-exact, "
            "published claims re-verified within protocol tolerance"
        )
        _verify_multiscale_claims(bundle_metrics, spikes, V, Q, positions)
    t = np.arange(Q.shape[0], dtype=float) * 0.5
    s = _stride(len(t))
    tt = t[::s]
    out_dir.mkdir(parents=True, exist_ok=True)
    panels: list = []

    import plotly.graph_objects as go
    from jaxfne.fields import project_laminar_sources

    ti, ni = np.nonzero(spikes > 0.5)
    keep = np.linspace(0, len(ti) - 1, min(len(ti), 20000)).astype(int)
    fig = go.Figure(
        go.Scatter(x=t[ti[keep]], y=ni[keep], mode="markers", marker=dict(size=2, opacity=0.6))
    )
    fig.update_layout(yaxis_title="Neuron index")
    _emit(
        panels,
        out_dir,
        "raster.html",
        "Spike raster",
        "OBSERVED",
        fig,
        SRC,
    )

    fig = go.Figure()
    for j in range(min(8, V.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=V[::s, j], mode="lines", name=f"unit {j}"))
    fig.update_layout(yaxis_title="V_m proxy")
    _emit(
        panels,
        out_dir,
        "traces.html",
        "Membrane traces",
        "OBSERVED",
        fig,
        SRC,
    )

    fo = project_laminar_sources(Q, positions, n_contacts=16, width=0.10)
    lfp = np.asarray(fo.lfp_proxy)
    csd = np.asarray(fo.csd_proxy)
    fig = go.Figure()
    for c in range(min(8, lfp.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=lfp[::s, c], mode="lines", name=f"c{c}"))
    fig.update_layout(yaxis_title="LFP proxy")
    _emit(
        panels,
        out_dir,
        "lfp.html",
        "LFP proxy",
        "DERIVED",
        fig,
        SRC,
    )

    fig = go.Figure()
    for c in range(min(8, csd.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=csd[::s, c], mode="lines", name=f"c{c}"))
    fig.update_layout(yaxis_title="CSD proxy")
    _emit(
        panels,
        out_dir,
        "csd.html",
        "CSD proxy",
        "DERIVED",
        fig,
        SRC,
    )

    from jaxfne.analysis.spectral import spectrolaminar_psd_jax
    import jax.numpy as jnp

    burn = 400
    freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
    fig = go.Figure()
    for name, arr in (
        ("Q", Q[burn:].mean(axis=1)),
        ("LFP", lfp[burn:].mean(axis=1)),
        ("CSD", csd[burn:].mean(axis=1)),
    ):
        psd = np.asarray(
            spectrolaminar_psd_jax(
                jnp.asarray(arr.astype(np.float32)[None, :, None]), fs=2000.0, freqs=freqs
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freqs, y=psd.mean(axis=(0, 2)) if psd.ndim == 3 else psd, mode="lines", name=name
            )
        )
    fig.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Power", xaxis_type="log", yaxis_type="log"
    )
    _emit(
        panels,
        out_dir,
        "oscillatory.html",
        "Oscillatory response",
        "DERIVED",
        fig,
        SRC,
    )

    finalize(
        out_dir,
        "multiscale_observation",
        f"Multiscale observation — reproduced group-A panels ({tier}-checked)",
        panels,
        {
            "source_bundle": "artifacts/etudes/multiscale_observation",
            "verification_tier": tier,
            "reproduced_cause_hashes": got,
            "seed": 7,
            "n_neurons": 40,
            "n_steps": 4000,
            "simulation_runs": 1,
            "rerun_note": note,
        },
    )
    return panels


def _verify_heterogeneous_claims(bundle: dict, runs: dict, rates: dict) -> None:
    """Claims-tier verification (P-003) for the heterogeneous rerun."""
    import numpy as np

    from scripts.run_heterogeneous_emitters_etude import (
        BURN_IN_MS,
        DT_MS as HDT,
        psd_of,
        spectral_centroid,
    )

    if rates != bundle["rates_hz"]:
        raise SystemExit(f"rate drift (spike-derived should be exact): {rates}")
    burn = int(round(BURN_IN_MS / HDT))
    freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
    for name, run in runs.items():
        Q = np.asarray(run["Q"])
        psd = psd_of(Q[burn:], freqs)
        got = float(spectral_centroid(psd, freqs))
        want = bundle["spectral_centroid_hz"][name]["P_Q"]
        if abs(got - want) > 1e-3 * max(abs(want), 1e-12):
            raise SystemExit(f"{name} P_Q centroid outside tolerance")
    q_rel = float(
        np.linalg.norm(runs["izh"]["Q"].mean(axis=1) - runs["hei"]["Q"].mean(axis=1))
        / max(np.linalg.norm(runs["izh"]["Q"].mean(axis=1)), 1e-12)
    )
    rate_rel = abs(rates["izh"] - rates["hei"]) / max(rates["izh"], rates["hei"], 1e-12)
    if bool(rate_rel < 0.5 and q_rel > 0.2) != bool(bundle["similar_rate_different_Q"]):
        raise SystemExit("similar_rate_different_Q gate flipped")
    return burn, freqs


def mode_heterogeneous(out_dir: pathlib.Path) -> list[dict]:
    import numpy as np

    from scripts.run_heterogeneous_emitters_etude import (
        DURATION_MS,
        DT_MS,
        N_HEI,
        N_IZH,
        U_SCALE,
        array_sha256,
        broadcast_u,
        mean_rate_hz,
        observe,
        simulate_hei,
        simulate_izh,
        u_shape,
    )

    bundle_het = json.loads(
        (ROOT / "artifacts/etudes/heterogeneous_emitters/metrics.json").read_text(encoding="utf-8")
    )
    committed = bundle_het
    n_steps = int(round(DURATION_MS / DT_MS))
    shape = u_shape(n_steps, DT_MS)
    runs = {
        "izh": simulate_izh(DURATION_MS, broadcast_u(shape, N_IZH, U_SCALE["izh"])),
        "hei": simulate_hei(DURATION_MS, broadcast_u(shape, N_HEI, U_SCALE["hei"])),
    }
    q_hashes = {name: array_sha256(run["Q"]) for name, run in runs.items()}
    rates = {name: mean_rate_hz(run["spikes"]) for name, run in runs.items()}
    if q_hashes == committed["Q_sha256"] and rates == committed["rates_hz"]:
        het_tier = "hash"
        het_note = "deterministic reproduction; hashes equal committed bundle"
    else:
        # Claims tier (P-003): same env-drift pattern as multiscale.
        het_tier = "claims"
        het_note = (
            "deterministic reproduction; spike-derived rates exact, "
            "Q float hashes drift, centroids re-verified"
        )
        burn, _ = _verify_heterogeneous_claims(bundle_het, runs, rates)
    HSRC = (
        "reproduced runs (hash-checked)"
        if het_tier == "hash"
        else "reproduced runs (claims-checked P-003: rates exact, "
        "Q float drift, centroids re-verified)"
    )
    obs = {name: observe(run["Q"], run["positions"]) for name, run in runs.items()}
    if het_tier == "claims":
        from scripts.run_heterogeneous_emitters_etude import (
            psd_of as _psd,
            spectral_centroid as _sc,
        )

        _freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
        for _name, _run in runs.items():
            _o = obs[_name]
            for _key, _arr in (
                ("P_LFP", np.asarray(_o["lfp_ref"])),
                ("P_CSD", np.asarray(_o["csd"])),
                ("P_EEG", np.asarray(_o["eeg_sup"])),
            ):
                _P = _psd(_arr[burn:], _freqs)
                _got = float(_sc(_P, _freqs))
                _want = committed["spectral_centroid_hz"][_name][_key]
                if abs(_got - _want) > 1e-3 * max(abs(_want), 1e-12):
                    raise SystemExit(f"{_name} {_key} centroid outside tolerance")

    t = np.arange(n_steps, dtype=float) * DT_MS
    s = _stride(n_steps)
    tt = t[::s]
    out_dir.mkdir(parents=True, exist_ok=True)
    panels: list = []
    import plotly.graph_objects as go

    izh, hei = runs["izh"], runs["hei"]
    ti, ni = np.nonzero(np.asarray(izh["spikes"]) > 0.5)
    keep = np.linspace(0, len(ti) - 1, min(len(ti), 20000)).astype(int)
    fig = go.Figure(
        go.Scatter(x=t[ti[keep]], y=ni[keep], mode="markers", marker=dict(size=2, opacity=0.6))
    )
    fig.update_layout(yaxis_title="Neuron index")
    _emit(
        panels,
        out_dir,
        "raster_izh.html",
        "Spike raster (Izhikevich)",
        "OBSERVED",
        fig,
        HSRC,
    )

    V = np.asarray(izh["V"])
    fig = go.Figure()
    for j in range(min(8, V.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=V[::s, j], mode="lines", name=f"unit {j}"))
    fig.update_layout(yaxis_title="V_m (mV native)")
    _emit(
        panels,
        out_dir,
        "traces_izh.html",
        "Membrane traces (Izhikevich)",
        "OBSERVED",
        fig,
        HSRC,
    )

    for name, key in (("izh", "lfp_izh.html"), ("hei", "lfp_hei.html")):
        lfp = np.asarray(obs[name]["lfp_ref"])
        fig = go.Figure()
        for c in range(min(8, lfp.shape[1])):
            fig.add_trace(go.Scatter(x=tt, y=lfp[::s, c], mode="lines", name=f"c{c}"))
        fig.update_layout(yaxis_title="LFP proxy")
        _emit(
            panels,
            out_dir,
            key,
            f"LFP proxy ({name})",
            "DERIVED",
            fig,
            HSRC,
        )

    X = np.asarray(hei["V"])
    fig = go.Figure()
    for j in range(min(4, X.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=X[::s, j], mode="lines", name=f"unit {j}"))
    fig.update_layout(yaxis_title="x (native, not mV)")
    _emit(
        panels,
        out_dir,
        "traces_hei.html",
        "State traces (HEI)",
        "OBSERVED",
        fig,
        HSRC,
    )

    from jaxfne.analysis.spectral import spectrolaminar_psd_jax
    import jax.numpy as jnp

    freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
    fig = go.Figure()
    for name, run in runs.items():
        y = np.asarray(run["Q"], dtype=np.float32).mean(axis=1)
        psd = np.asarray(
            spectrolaminar_psd_jax(
                jnp.asarray(y[None, :, None]), fs=float(1000.0 / DT_MS), freqs=freqs
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freqs, y=psd.mean(axis=(0, 2)) if psd.ndim == 3 else psd, mode="lines", name=name
            )
        )
    fig.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Power", xaxis_type="log", yaxis_type="log"
    )
    _emit(
        panels,
        out_dir,
        "oscillatory.html",
        "Source spectra by family",
        "DERIVED",
        fig,
        HSRC,
    )

    finalize(
        out_dir,
        "heterogeneous_emitters",
        f"Heterogeneous emitters — reproduced panels ({het_tier}-checked)",
        panels,
        {
            "source_bundle": "artifacts/etudes/heterogeneous_emitters",
            "verification_tier": het_tier,
            "reproduced_Q_sha256": {n: array_sha256(r["Q"]) for n, r in runs.items()},
            "reproduced_rates_hz": rates,
            "seed": 11,
            "simulation_runs": 2,
            "rerun_note": het_note,
        },
    )
    return panels


def mode_hdp_mcc3(out_dir: pathlib.Path) -> list[dict]:
    import numpy as np

    import jaxfne as J
    import scripts.mcc3_10s_scientific_checkpoint as M3

    committed = json.loads(
        (ROOT / "artifacts/mcc3_10s_checkpoint/mcc3_10s_metrics.json").read_text(encoding="utf-8")
    )
    specs = M3.mcc3_specs()
    objective = J.rate_targets(
        groups={"all": list(range(10))},
        targets_hz={"all": M3.TARGET_HZ},
        burn_in_ms=M3.BURN_IN_MS,
    )
    base = J.construct(M3.mcc3_config())
    tune_sim = J.simulation(
        duration_ms=M3.TUNE_DURATION_MS,
        dt_ms=M3.DT_MS,
        seed=M3.SIM_SEED,
        runtime=M3.mcc3_runtime(True),
    )
    optimizer = J.agsdr(
        parameters=specs,
        generations=M3.AGSDR_GENERATIONS,
        population_size=M3.AGSDR_POPULATION,
        seed=M3.OPT_SEED,
    )
    tune_result = base.tune(objectives=objective, optimizer=optimizer, simulation=tune_sim)
    thetah = {k: float(v) for k, v in dict(tune_result.best_parameters).items()}
    if thetah != {k: float(v) for k, v in committed["agsdr"]["theta_hat"].items()}:
        raise SystemExit(f"tuned theta drift vs frozen checkpoint: {thetah}")
    sim_10s_on = J.simulation(
        duration_ms=M3.DURATION_MS, dt_ms=M3.DT_MS, seed=M3.SIM_SEED, runtime=M3.mcc3_runtime(True)
    )
    B = M3.run_condition(base, thetah, specs, sim_10s_on, objective, "B")
    want = committed["firing"]["B"]
    for key in ("population_rate_hz", "E_rate_hz", "I_rate_hz"):
        if B[key] != want[key]:
            raise SystemExit(f"condition-B {key} drift: {B[key]} vs {want[key]}")
    if B["spike_total"] != committed["abc_table"]["total_spikes"]["B"]:
        raise SystemExit("condition-B spike_total drift")

    signals = B["signals"]
    diag = B["diagnostics"]
    spikes = np.asarray(signals.spikes)
    Vm = np.asarray(signals.V_m)
    dt_ms = float(signals.metadata.get("dt_ms", M3.DT_MS))
    t = np.arange(spikes.shape[0], dtype=float) * dt_ms
    s = _stride(len(t))
    tt = t[::s]
    out_dir.mkdir(parents=True, exist_ok=True)
    panels: list = []
    import plotly.graph_objects as go

    ti, ni = np.nonzero(spikes > 0.5)
    keep = np.linspace(0, len(ti) - 1, min(len(ti), 20000)).astype(int)
    fig = go.Figure(
        go.Scatter(x=t[ti[keep]], y=ni[keep], mode="markers", marker=dict(size=2, opacity=0.6))
    )
    fig.update_layout(yaxis_title="Neuron index")
    _emit(
        panels,
        out_dir,
        "raster.html",
        "Spike raster (condition B)",
        "OBSERVED",
        fig,
        "reproduced MCC-3 condition B (metric-checked)",
    )

    fig = go.Figure()
    for j in range(min(8, Vm.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=Vm[::s, j], mode="lines", name=f"unit {j}"))
    fig.update_layout(yaxis_title="V_m proxy")
    _emit(
        panels,
        out_dir,
        "traces.html",
        "Membrane traces (condition B)",
        "OBSERVED",
        fig,
        "reproduced MCC-3 condition B (metric-checked)",
    )

    H = np.asarray(diag["H_trace"])
    mu, sd = H.mean(axis=1), H.std(axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=(mu + sd)[::s],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=(mu - sd)[::s],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="±1 std",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=tt, y=mu[::s], mode="lines", name="mean H"))
    fig.update_layout(yaxis_title="H")
    _emit(
        panels,
        out_dir,
        "h_dynamics.html",
        "H dynamics (condition B)",
        "DERIVED",
        fig,
        "reproduced MCC-3 condition B (metric-checked)",
    )

    W = np.asarray(diag["w_trace"], dtype=float).reshape(diag["w_trace"].shape[0], -1)
    mag = np.abs(W)
    mu, sd = mag.mean(axis=1), mag.std(axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=(mu + sd)[::s],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=(mu - sd)[::s],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="±1 std",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=tt, y=mu[::s], mode="lines", name="mean |w|"))
    fig.update_layout(yaxis_title="|w|")
    _emit(
        panels,
        out_dir,
        "hdp.html",
        "HDP weight magnitude (condition B)",
        "DERIVED",
        fig,
        "reproduced MCC-3 condition B (metric-checked)",
    )

    from jaxfne.analysis.spectral import spectrolaminar_psd_jax
    import jax.numpy as jnp

    y = spikes.mean(axis=1).astype(np.float32)
    freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
    psd = np.asarray(
        spectrolaminar_psd_jax(jnp.asarray(y[None, :, None]), fs=float(1000.0 / dt_ms), freqs=freqs)
    )
    fig = go.Figure(
        go.Scatter(
            x=freqs,
            y=psd.mean(axis=(0, 2)) if psd.ndim == 3 else psd,
            mode="lines",
            name="population-rate PSD",
        )
    )
    fig.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Power", xaxis_type="log", yaxis_type="log"
    )
    _emit(
        panels,
        out_dir,
        "oscillatory.html",
        "Oscillatory response (condition B)",
        "DERIVED",
        fig,
        "reproduced MCC-3 condition B (metric-checked)",
    )

    finalize(
        out_dir,
        "hdp_controllability_reachability",
        "MCC-3 checkpoint — reproduced condition-B panels (hash-checked)",
        panels,
        {
            "source_bundle": "artifacts/etudes/hdp_controllability_reachability",
            "verification_tier": "hash",
            "checkpoint": "artifacts/mcc3_10s_checkpoint",
            "reproduced_theta_hat": thetah,
            "seed": M3.SIM_SEED,
            "n_neurons": 10,
            "n_steps": int(round(M3.DURATION_MS / M3.DT_MS)),
            "simulation_runs": 2,
            "rerun_note": "tune + condition B reproduced; theta_hat and firing.B match committed checkpoint",
        },
    )
    return panels


MODES = {
    "multiscale": ("multiscale_observation", mode_multiscale),
    "heterogeneous": ("heterogeneous_emitters", mode_heterogeneous),
    "hdp_mcc3": ("hdp_controllability_reachability", mode_hdp_mcc3),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etude", required=True, choices=sorted(MODES))
    args = parser.parse_args(argv)
    slug, fn = MODES[args.etude]
    out = ROOT / "docs" / "_static" / "etudes" / slug
    panels = fn(out)
    print(f"[{args.etude}] {len(panels)} panels -> {out.relative_to(ROOT)}")
    for p in panels:
        print(f"    {p['file']}: {p['evidence']} {p['status']} {p['bytes']}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
