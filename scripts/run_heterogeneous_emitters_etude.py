#!/usr/bin/env python3
"""Heterogeneous-emitter TFNE composition Etude — frozen protocol, no package mutation.

See docs/etudes/heterogeneous_emitters.md.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.analysis.spectral import spectrolaminar_psd_jax
from jaxfne.emitters_homeostatic_ei import simulate_homeostatic_ei
from jaxfne.fields import LinearReadout, project_laminar_sources
from jaxfne.io import json_safe
from jaxfne.vis.evidence_export import save_matplotlib_evidence_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "etudes" / "heterogeneous_emitters"
PROTOCOL = ROOT / "docs" / "etudes" / "heterogeneous_emitters.md"

SEED = 11
DURATION_MS = 1000.0
DT_MS = 0.5
FS = 1000.0 / DT_MS
BURN_IN_MS = 100.0
N_IZH = 10
N_HEI = 2
U_SCALE = {"izh": 6.0, "hei": 0.2}
FI_AMPS = {
    "izh": (0.0, 2.0, 4.0, 8.0, 12.0),
    "hei": (0.0, 0.05, 0.1, 0.2, 0.3),
}
FI_DURATION_MS = 400.0
FI_SCORE_MS = 300.0


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(x: Any) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def u_shape(n_steps: int, dt_ms: float) -> np.ndarray:
    t = np.arange(n_steps, dtype=np.float32) * np.float32(dt_ms)
    y = np.zeros(n_steps, dtype=np.float32)
    y[(t >= 100.0) & (t < 600.0)] = 1.0
    y[(t >= 700.0) & (t < 750.0)] = 1.0
    return y


def broadcast_u(shape_1d: np.ndarray, n: int, scale: float) -> np.ndarray:
    return (np.float32(scale) * shape_1d)[:, None] * np.ones((1, n), dtype=np.float32)


def psd_of(y: np.ndarray, freqs: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.asarray(spectrolaminar_psd_jax(jnp.asarray(arr[None, ...]), fs=float(FS), freqs=freqs))


def spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    p = np.asarray(psd, dtype=np.float64)
    if p.ndim == 2:
        p = p.mean(axis=1)
    mass = float(p.sum())
    if mass <= 0.0:
        return float("nan")
    return float(np.dot(np.asarray(freqs, dtype=np.float64), p) / mass)


def mean_rate_hz(spikes: np.ndarray) -> float:
    return float(np.mean(np.asarray(spikes)) * FS)


def izh_config() -> Any:
    return (
        jtfne.configuration()
        .runtime(seed=SEED, duration_ms=DURATION_MS, dt_ms=DT_MS, dtype="float32", jit=False)
        .population(N_IZH, neurons={"E": 0.7, "I": 0.3}, layers=["L2/3", "L4"], name="V1")
        .cell_types({"E": 0.7, "PV": 0.3})
        .geometry(layer_thickness={"L2/3": 0.5, "L4": 0.5})
        .cell_type_drives({"E": 8.0, "PV": 8.0})
        .set_emitter("izhikevich", "cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="etude_probe", modes=["spikes", "V_m"])
    )


def hei_config() -> Any:
    return (
        jtfne.configuration()
        .runtime(seed=SEED, duration_ms=DURATION_MS, dt_ms=DT_MS, dtype="float32", jit=False)
        .network(name="hei", n=N_HEI)
        .set_emitter("homeostatic_ei", bound_mode="stable")
        .field(domain="none")
        .probe(modes=["vm"])
    )


def simulate_izh(duration_ms: float, u: np.ndarray | None) -> dict[str, Any]:
    model = jtfne.construct(izh_config())
    sim = jtfne.Simulation(
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        seed=SEED,
        record_sources=True,
        record_fields=False,
        runtime=jtfne.RuntimeConfig(dtype="float32", jit=False, seed=SEED),
    )
    paradigm = None
    if u is not None:
        n_steps = int(round(duration_ms / DT_MS))
        events = []
        # Encode piecewise-constant U as contiguous runs (native current).
        arr = np.asarray(u[:n_steps, 0], dtype=np.float32)
        i = 0
        while i < n_steps:
            j = i + 1
            while j < n_steps and arr[j] == arr[i]:
                j += 1
            amp = float(arr[i])
            if amp != 0.0:
                events.append(
                    {
                        "label": f"u_{i}",
                        "onset_ms": float(i * DT_MS),
                        "duration_ms": float((j - i) * DT_MS),
                        "amplitude": amp,
                    }
                )
            i = j
        paradigm = jtfne.stimulus_schedule(tuple(events), n_neurons=int(u.shape[1]))
    signals = model.simulate(sim, paradigm=paradigm)
    return {
        "family": "izhikevich",
        "V": np.asarray(signals.V_m),
        "spikes": np.asarray(signals.spikes),
        "Q": np.asarray(signals.sources),
        "positions": np.asarray(model.params["positions"]),
        "source_calibration_status": str(
            signals.metadata.get("source_calibration_status")
            or "uncalibrated_izhikevich_native_current"
        ),
        "emitter_family": str(signals.metadata.get("emitter_family", "izhikevich")),
        "native_state": "izhikevich_V_m_mV_native",
        "metadata": json_safe(
            {k: signals.metadata[k] for k in ("source_calibration_status", "representation", "field_claim_level") if k in signals.metadata}
        ),
    }


def simulate_hei(duration_ms: float, u: np.ndarray | None) -> dict[str, Any]:
    model = jtfne.construct(hei_config())
    n_steps = int(round(duration_ms / DT_MS))
    params = model.params["emitter"]
    drive = None if u is None else jnp.asarray(u[:n_steps], dtype=jnp.float32)
    voltages, spikes, sources, _G, _H, diag = simulate_homeostatic_ei(
        params,
        n_steps=n_steps,
        dt_ms=DT_MS,
        key=jax.random.PRNGKey(SEED),
        activation_rule=params.activation_rule_name,
        conductance_rule=params.conductance_rule_name,
        homeostasis_rule=params.homeostasis_rule_name,
        drive_schedule=drive,
        dtype="float32",
        bound_mode=params.bound_mode,
    )
    return {
        "family": "homeostatic_ei",
        "V": np.asarray(voltages),
        "spikes": np.asarray(spikes),
        "Q": np.asarray(sources),
        "positions": np.asarray(model.params["positions"]),
        "source_calibration_status": str(params.source_calibration_status),
        "emitter_family": "homeostatic_ei",
        "native_state": "homeostatic_ei_x_activity_not_mV",
        "hei_error": bool(diag["error"]),
        "metadata": {
            "source_calibration_status": str(params.source_calibration_status),
            "representation": "relative",
            "physical_amplitude_calibrated": False,
        },
    }


def observe(Q: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    z = np.asarray(positions)[:, 2]
    fo_ref = project_laminar_sources(jnp.asarray(Q), jnp.asarray(positions), n_contacts=16, width=0.10)
    fo_wide = project_laminar_sources(jnp.asarray(Q), jnp.asarray(positions), n_contacts=16, width=0.25)
    W_sup = np.exp(
        -0.5
        * ((np.array([0.25, 0.25, 0.25], dtype=np.float32)[:, None] - z[None, :].astype(np.float32))
           / np.array([0.18, 0.20, 0.22], dtype=np.float32)[:, None]) ** 2
    ).astype(np.float32)
    eeg = LinearReadout(name="eeg_superficial", W=jnp.asarray(W_sup), leadfield_status="toy_or_declared_proxy")
    Y_eeg = np.asarray(eeg.apply(jnp.asarray(Q)))
    return {
        "lfp_ref": np.asarray(fo_ref.lfp_proxy),
        "lfp_wide": np.asarray(fo_wide.lfp_proxy),
        "csd": np.asarray(fo_ref.csd_proxy),
        "eeg_sup": Y_eeg,
        "observation_ref": json_safe(fo_ref.diagnostics.get("observation")),
        "observation_eeg": json_safe(eeg.report()),
        "q_hash_after": array_sha256(Q),
    }


def fi_curve(family: str) -> list[dict[str, float]]:
    n_steps = int(round(FI_DURATION_MS / DT_MS))
    score_from = int(round((FI_DURATION_MS - FI_SCORE_MS) / DT_MS))
    rows = []
    n = N_IZH if family == "izh" else N_HEI
    for amp in FI_AMPS[family]:
        u = np.full((n_steps, n), np.float32(amp), dtype=np.float32)
        run = simulate_izh(FI_DURATION_MS, u) if family == "izh" else simulate_hei(FI_DURATION_MS, u)
        rate = mean_rate_hz(run["spikes"][score_from:])
        rows.append({"amplitude_native": float(amp), "rate_hz": rate, "finite_Q": bool(np.isfinite(run["Q"]).all())})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    head = _git_head()
    n_steps = int(round(DURATION_MS / DT_MS))
    shape = u_shape(n_steps, DT_MS)
    runs = {
        "izh": simulate_izh(DURATION_MS, broadcast_u(shape, N_IZH, U_SCALE["izh"])),
        "hei": simulate_hei(DURATION_MS, broadcast_u(shape, N_HEI, U_SCALE["hei"])),
    }
    obs = {}
    for name, run in runs.items():
        q_hash = array_sha256(run["Q"])
        o = observe(run["Q"], run["positions"])
        if o["q_hash_after"] != q_hash:
            raise RuntimeError(f"{name} Q mutated under observation")
        run["Q_sha256"] = q_hash
        obs[name] = o

    fi = {name: fi_curve(name) for name in ("izh", "hei")}
    burn = int(round(BURN_IN_MS / DT_MS))
    freqs = jnp.linspace(1.0, 150.0, 96, dtype=jnp.float32)
    freqs_np = np.asarray(freqs)
    spectra = {}
    for name, run in runs.items():
        spectra[name] = {
            "P_Q": psd_of(run["Q"][burn:], freqs),
            "P_LFP": psd_of(obs[name]["lfp_ref"][burn:], freqs),
            "P_CSD": psd_of(obs[name]["csd"][burn:], freqs),
            "P_EEG": psd_of(obs[name]["eeg_sup"][burn:], freqs),
        }

    rates = {name: mean_rate_hz(run["spikes"]) for name, run in runs.items()}
    q_rel = float(
        np.linalg.norm(runs["izh"]["Q"].mean(axis=1) - runs["hei"]["Q"].mean(axis=1))
        / max(np.linalg.norm(runs["izh"]["Q"].mean(axis=1)), 1e-12)
    )
    py_rel = float(
        np.linalg.norm(spectra["izh"]["P_LFP"].mean(axis=1) - spectra["hei"]["P_LFP"].mean(axis=1))
        / max(np.linalg.norm(spectra["izh"]["P_LFP"].mean(axis=1)), 1e-12)
    )
    rate_rel = abs(rates["izh"] - rates["hei"]) / max(rates["izh"], rates["hei"], 1e-12)
    similar_rate_different_q = bool(rate_rel < 0.5 and q_rel > 0.2)

    gate_a = runs["izh"]["family"] != runs["hei"]["family"]
    gate_b = (
        runs["izh"]["source_calibration_status"] != runs["hei"]["source_calibration_status"]
        and np.isfinite(runs["izh"]["Q"]).all()
        and np.isfinite(runs["hei"]["Q"]).all()
        and not bool(runs["hei"].get("hei_error"))
    )
    gate_c = all(
        obs[name]["observation_ref"]["execution_form"] == "fused" and obs[name]["eeg_sup"].shape[-1] == 3
        for name in ("izh", "hei")
    )
    gate_d = (
        "izhikevich" in runs["izh"]["source_calibration_status"]
        and "homeostatic_ei" in runs["hei"]["source_calibration_status"]
        and obs["izh"]["observation_eeg"]["leadfield_status"] == "toy_or_declared_proxy"
    )
    gate_e = bool(q_rel > 0.05 or py_rel > 0.05 or similar_rate_different_q)

    metrics = {
        "protocol": "heterogeneous_emitters_v0415b",
        "package_head": head,
        "n": {"izh": N_IZH, "hei": N_HEI},
        "dt_ms": DT_MS,
        "duration_ms": DURATION_MS,
        "u_scale_native": U_SCALE,
        "rates_hz": rates,
        "rate_relative_difference": rate_rel,
        "Q_mean_relative_difference": q_rel,
        "P_LFP_relative_difference": py_rel,
        "similar_rate_different_Q": similar_rate_different_q,
        "Q_sha256": {name: run["Q_sha256"] for name, run in runs.items()},
        "source_calibration_status": {name: run["source_calibration_status"] for name, run in runs.items()},
        "native_state": {name: run["native_state"] for name, run in runs.items()},
        "spectral_centroid_hz": {
            name: {k: spectral_centroid(v, freqs_np) for k, v in spectra[name].items()}
            for name in spectra
        },
        "fi": fi,
        "hei_error": bool(runs["hei"].get("hei_error")),
        "gates": {"A": bool(gate_a), "B": bool(gate_b), "C": bool(gate_c), "D": bool(gate_d), "E": bool(gate_e)},
        "jaxley_primary": False,
        "agsdr_fit": False,
        "amplitude_semantics": "relative",
        "physical_claim": "proxy_readout",
        "cross_family_physical_equivalence": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n")
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                name: {
                    "emitter_family": runs[name]["emitter_family"],
                    "source_calibration_status": runs[name]["source_calibration_status"],
                    "native_state": runs[name]["native_state"],
                    "observation": obs[name]["observation_ref"],
                    "eeg": obs[name]["observation_eeg"],
                }
                for name in runs
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(n_steps) * DT_MS
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, name in enumerate(("izh", "hei")):
        spk_t, spk_i = np.nonzero(runs[name]["spikes"])
        axes[row, 0].scatter(spk_t * DT_MS, spk_i, s=2, c="k", linewidths=0)
        axes[row, 0].set_title(f"{name} spikes")
        axes[row, 1].imshow(runs[name]["Q"].T, aspect="auto", origin="lower", cmap="magma")
        axes[row, 1].set_title(f"{name} Q (declared S)")
        axes[row, 2].plot(t[::2], obs[name]["lfp_ref"][::2, 8], lw=0.8)
        axes[row, 2].set_title(f"{name} LFP-proxy mid")
    fig.suptitle("Heterogeneous emitters, same O_k (relative; Q not equated)")
    fig.tight_layout()
    save_matplotlib_evidence_figure(fig, OUT / "figure.png", dpi=140)

    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs_np, spectra["izh"]["P_LFP"].mean(axis=1), label="izh LFP")
    ax.plot(freqs_np, spectra["hei"]["P_LFP"].mean(axis=1), label="hei LFP")
    ax.plot(freqs_np, spectra["izh"]["P_Q"].mean(axis=1), label="izh Q", ls="--")
    ax.plot(freqs_np, spectra["hei"]["P_Q"].mean(axis=1), label="hei Q", ls="--")
    ax.set_xlabel("Hz")
    ax.set_title("Declared sources through the same spectral observation")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    save_matplotlib_evidence_figure(fig2, OUT / "figure_spectra.png", dpi=140)

    gap = _gap_review(metrics)
    (OUT / "gap_review.md").write_text(gap)
    manifest = {
        "etude": "heterogeneous_emitters_v0415b",
        "package_head": head,
        "protocol_sha256": _sha256_bytes(PROTOCOL.read_bytes()),
        "metrics_sha256": _sha256_bytes((OUT / "metrics.json").read_bytes()),
        "figure_sha256": _sha256_bytes((OUT / "figure.png").read_bytes()),
        "gates": metrics["gates"],
        "representation": "relative_proxy",
        "cross_family_physical_equivalence": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": metrics["gates"], "rates_hz": rates, "head": head}, indent=2))
    return 0 if all(metrics["gates"].values()) else 1


def _gap_review(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Gap review — Heterogeneous emitters Etude",
            "",
            f"- Gates: `{json.dumps(metrics['gates'])}`",
            f"- Rates (Hz): `{json.dumps(metrics['rates_hz'])}`",
            f"- N: `{json.dumps(metrics['n'])}`",
            f"- Similar-rate / different-Q control (null if False): `{metrics['similar_rate_different_Q']}`",
            f"- v0415b revision: finite/stable declared Q, not figure appearance.",
            "",
            "| item | class | note |",
            "|------|-------|------|",
            "| Two mature distinct F_X (Izhikevich vs homeostatic_ei) | NO_GAP | Smallest implemented pair. |",
            "| LIF/GLIF unused | NO_GAP | Placeholders; not silent Izhikevich substitutes. |",
            "| HEI Model.simulate ignores StimulusSchedule | ANALYSIS_GAP | Live Model.simulate docstring is the Izhikevich vertical slice; HEI returns before paradigm resolution. Supported HEI drive is simulate_homeostatic_ei(..., drive_schedule=). docs/api/core.md overgeneralizes injection without an HEI caveat (closure docs, not this étude). Silent ignore is an interface limitation, not CORRECTNESS_DEFECT of the Izhikevich-scoped method. |",
            "| Jaxley omitted | NO_GAP | Not a capability claim; deferred to avoid an integration étude. |",
            "| AGSDR omitted | NO_GAP | Not a capability claim; would need a cross-family observable contract. |",
            "| Family-native U scales | NO_GAP | Common shape, not calibrated current equivalence. |",
            "| Q not physically equated | NO_GAP | Declared source semantics preserved. |",
            "| HEI n=10 minimal extra-drive divergence | NO_GAP | Preserved as failed v0415; not a missing observation operator. |",
            "",
        ]
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
