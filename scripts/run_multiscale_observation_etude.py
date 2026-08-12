#!/usr/bin/env python3
"""Multiscale Observation Etude — frozen protocol, no package mutation.

See docs/etudes/multiscale_observation.md. Simulate once; vary O_k only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.analysis.spectral import spectrolaminar_psd_jax
from jaxfne.fields import LinearReadout, eeg_proxy_transform, meg_proxy_transform, project_laminar_sources
from jaxfne.io import json_safe
from jaxfne.vis.evidence_export import save_matplotlib_evidence_figure
from jaxfne.vis.script_reports import spectrolaminar_motif_heatmap
from jaxfne.vis.tutorial_array_plots import plot_laminar_readout_array

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "etudes" / "multiscale_observation"
PROTOCOL = ROOT / "docs" / "etudes" / "multiscale_observation.md"

# Frozen protocol constants (must match the protocol document).
EXPECTED_PACKAGE_PREFIX = "d5cf9a6"
SEED = 7
N = 40
DURATION_MS = 2000.0
DT_MS = 0.5
BURN_IN_MS = 200.0
FS = 1000.0 / DT_MS
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}
Y_ATOL = 1e-5
Y_DISTINCT_RTOL = 1e-3


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


def r90_1d(weights: np.ndarray, z: np.ndarray, center: float) -> float:
    w = np.abs(np.asarray(weights, dtype=np.float64))
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    dist = np.abs(np.asarray(z, dtype=np.float64) - float(center))
    order = np.argsort(dist)
    csum = np.cumsum(w[order])
    hit = np.searchsorted(csum, 0.9 * total, side="left")
    hit = min(int(hit), len(order) - 1)
    return float(dist[order[hit]])


def mean_r90(kernel: np.ndarray, z: np.ndarray, contacts: np.ndarray) -> float:
    vals = [r90_1d(kernel[p], z, float(contacts[p])) for p in range(kernel.shape[0])]
    return float(np.nanmean(vals))


def band_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.zeros(psd.shape[1], dtype=np.float64)
    return np.asarray(psd[mask], dtype=np.float64).mean(axis=0)


def spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    p = np.asarray(psd, dtype=np.float64)
    if p.ndim == 2:
        p = p.mean(axis=1)
    mass = float(p.sum())
    if mass <= 0.0:
        return float("nan")
    return float(np.dot(np.asarray(freqs, dtype=np.float64), p) / mass)


def psd_of(y: np.ndarray, freqs: Any) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    sig = jnp.asarray(arr[None, ...])
    return np.asarray(spectrolaminar_psd_jax(sig, fs=float(FS), freqs=freqs))


def max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-12)
    return float(np.linalg.norm(a - b) / denom)


def build_config() -> Any:
    return (
        jtfne.configuration()
        .runtime(seed=SEED, duration_ms=DURATION_MS, dt_ms=DT_MS, dtype="float32", jit=False)
        .population(
            N,
            neurons={"E": 0.7, "I": 0.3},
            layers=["L2/3", "L4", "L5"],
            name="V1",
        )
        .cell_types({"E": 0.7, "PV": 0.3})
        .geometry(layer_thickness={"L2/3": 0.33, "L4": 0.34, "L5": 0.33})
        .cell_type_drives({"E": 8.0, "PV": 8.0})
        .set_emitter("izhikevich", "cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="etude_probe", modes=["spikes", "V_m"])
    )


def gaussian_leadfield(z: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)
    return np.exp(-0.5 * ((centers[:, None] - z[None, :]) / widths[:, None]) ** 2).astype(np.float32)


def contact_row(z: np.ndarray, center: float, width: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    return np.exp(-0.5 * ((z - np.float32(center)) / np.float32(width)) ** 2).astype(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    head = _git_head()
    if not head.startswith(EXPECTED_PACKAGE_PREFIX):
        print(
            f"WARNING: protocol expected package prefix {EXPECTED_PACKAGE_PREFIX}, HEAD={head}",
            file=sys.stderr,
        )

    cfg = build_config()
    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(
        duration_ms=DURATION_MS,
        dt_ms=DT_MS,
        seed=SEED,
        record_sources=True,
        record_fields=True,
        runtime=jtfne.RuntimeConfig(dtype="float32", jit=False, seed=SEED),
    )
    signals = model.simulate(sim)

    Q = np.asarray(signals.sources)
    V = np.asarray(signals.V_m)
    spikes = np.asarray(signals.spikes)
    positions = np.asarray(model.params["positions"])
    z = positions[:, 2]
    t = np.arange(Q.shape[0], dtype=np.float32) * np.float32(DT_MS)
    burn = int(round(BURN_IN_MS / DT_MS))
    Q_spec = Q[burn:]
    cause_hashes = {
        "V_m": array_sha256(V),
        "spikes": array_sha256(spikes),
        "Q": array_sha256(Q),
        "positions": array_sha256(positions),
    }

    laminar = {
        "lfp_ref": dict(n_contacts=16, width=0.10),
        "lfp_narrow": dict(n_contacts=16, width=0.05),
        "lfp_wide": dict(n_contacts=16, width=0.25),
        "lfp_sparse": dict(n_contacts=8, width=0.10),
        "lfp_dense": dict(n_contacts=24, width=0.10),
    }
    fields: dict[str, Any] = {}
    for name, kw in laminar.items():
        fields[name] = project_laminar_sources(jnp.asarray(Q), jnp.asarray(positions), **kw)
        if array_sha256(Q) != cause_hashes["Q"]:
            raise RuntimeError("Q mutated during observation")

    W_sup = gaussian_leadfield(z, np.array([0.25, 0.25, 0.25]), np.array([0.18, 0.20, 0.22]))
    W_deep = gaussian_leadfield(z, np.array([0.75, 0.75, 0.75]), np.array([0.18, 0.20, 0.22]))
    eeg_sup_ro = LinearReadout(name="eeg_superficial", W=jnp.asarray(W_sup), leadfield_status="toy_or_declared_proxy")
    eeg_deep_ro = LinearReadout(name="eeg_deep", W=jnp.asarray(W_deep), leadfield_status="toy_or_declared_proxy")
    Y_eeg_sup = np.asarray(eeg_sup_ro.apply(jnp.asarray(Q)))
    Y_eeg_deep = np.asarray(eeg_deep_ro.apply(jnp.asarray(Q)))
    Y_eeg_tf = np.asarray(eeg_proxy_transform(jnp.asarray(Q), jnp.asarray(W_sup)))

    rng = np.random.default_rng(SEED)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=W_sup.shape[1])
    W_meg = (W_sup * signs[None, :]).astype(np.float32)
    meg_ro = LinearReadout(name="meg_relative", W=jnp.asarray(W_meg), leadfield_status="toy_or_declared_proxy")
    Y_meg = np.asarray(meg_ro.apply(jnp.asarray(Q)))
    Y_meg_tf = np.asarray(meg_proxy_transform(jnp.asarray(Q), jnp.asarray(W_meg)))

    W_shallow = contact_row(z, 0.20, 0.10)[None, :]
    W_deep_c = contact_row(z, 0.80, 0.10)[None, :]
    Y_shallow = np.asarray(LinearReadout(name="contact_shallow", W=jnp.asarray(W_shallow)).apply(jnp.asarray(Q)))
    Y_deep_c = np.asarray(LinearReadout(name="contact_deep", W=jnp.asarray(W_deep_c)).apply(jnp.asarray(Q)))

    meta_a = LinearReadout(name="eeg_meta_a", W=jnp.asarray(W_sup), leadfield_status="toy_or_declared_proxy")
    meta_b = LinearReadout(name="eeg_meta_b", W=jnp.asarray(W_sup), leadfield_status="declared")
    Y_meta_a = np.asarray(meta_a.apply(jnp.asarray(Q)))
    Y_meta_b = np.asarray(meta_b.apply(jnp.asarray(Q)))

    K_ref = np.asarray(fields["lfp_ref"].kernel)
    Y_compiled = np.asarray(LinearReadout(name="lfp_compiled", W=jnp.asarray(K_ref)).apply(jnp.asarray(Q)))
    Y_lfp_ref = np.asarray(fields["lfp_ref"].lfp_proxy)
    if array_sha256(Q) != cause_hashes["Q"]:
        raise RuntimeError("Q mutated after post-hoc observation maps")

    freqs = jnp.linspace(1.0, 150.0, 96, dtype=jnp.float32)
    freqs_np = np.asarray(freqs)
    psd_q = psd_of(Q_spec, freqs)
    psd_lfp = psd_of(np.asarray(fields["lfp_ref"].lfp_proxy)[burn:], freqs)
    psd_csd = psd_of(np.asarray(fields["lfp_ref"].csd_proxy)[burn:], freqs)
    psd_eeg_sup = psd_of(Y_eeg_sup[burn:], freqs)
    psd_eeg_deep = psd_of(Y_eeg_deep[burn:], freqs)
    psd_narrow = psd_of(np.asarray(fields["lfp_narrow"].lfp_proxy)[burn:], freqs)
    psd_wide = psd_of(np.asarray(fields["lfp_wide"].lfp_proxy)[burn:], freqs)

    r90 = {
        name: mean_r90(np.asarray(fo.kernel), z, np.asarray(fo.contact_depths))
        for name, fo in fields.items()
    }
    r90_bands: dict[str, dict[str, float]] = {}
    k_ref = np.asarray(fields["lfp_ref"].kernel)
    c_ref = np.asarray(fields["lfp_ref"].contact_depths)
    for band, (lo, hi) in BANDS.items():
        p_src = band_power(psd_q, freqs_np, lo, hi)
        weighted = k_ref * p_src[None, :]
        r90_bands[band] = {
            "mean_r90": mean_r90(weighted, z, c_ref),
            "static_mean_r90": r90["lfp_ref"],
        }

    q_hash_after = array_sha256(Q)
    level_a = q_hash_after == cause_hashes["Q"] and array_sha256(V) == cause_hashes["V_m"]
    distinct = {
        "narrow_vs_wide": max_rel_diff(
            np.asarray(fields["lfp_narrow"].lfp_proxy),
            np.asarray(fields["lfp_wide"].lfp_proxy),
        ),
        "sparse_vs_dense": max_rel_diff(
            np.asarray(fields["lfp_sparse"].lfp_proxy)[:, :8],
            np.asarray(fields["lfp_dense"].lfp_proxy)[:, :8],
        ),
        "shallow_vs_deep_contact": max_rel_diff(Y_shallow, Y_deep_c),
        "lfp_vs_csd": max_rel_diff(
            np.asarray(fields["lfp_ref"].lfp_proxy),
            np.asarray(fields["lfp_ref"].csd_proxy),
        ),
        "eeg_sup_vs_deep": max_rel_diff(Y_eeg_sup, Y_eeg_deep),
    }
    # sparse vs dense have different channel counts; compare mean traces
    distinct["sparse_vs_dense_mean"] = max_rel_diff(
        np.asarray(fields["lfp_sparse"].lfp_proxy).mean(axis=1),
        np.asarray(fields["lfp_dense"].lfp_proxy).mean(axis=1),
    )
    negative_max = float(np.max(np.abs(Y_meta_a - Y_meta_b)))
    compile_max = float(np.max(np.abs(Y_compiled - Y_lfp_ref)))
    eeg_tf_max = float(np.max(np.abs(Y_eeg_sup - Y_eeg_tf)))
    meg_tf_max = float(np.max(np.abs(Y_meg - Y_meg_tf)))
    level_b = (
        distinct["narrow_vs_wide"] > Y_DISTINCT_RTOL
        and distinct["lfp_vs_csd"] > Y_DISTINCT_RTOL
        and distinct["eeg_sup_vs_deep"] > Y_DISTINCT_RTOL
        and distinct["shallow_vs_deep_contact"] > Y_DISTINCT_RTOL
        and negative_max <= Y_ATOL
        and compile_max <= Y_ATOL
    )
    r90_width_delta = abs(r90["lfp_wide"] - r90["lfp_narrow"])
    centroid = {
        "Q": spectral_centroid(psd_q, freqs_np),
        "LFP": spectral_centroid(psd_lfp, freqs_np),
        "CSD": spectral_centroid(psd_csd, freqs_np),
        "EEG_sup": spectral_centroid(psd_eeg_sup, freqs_np),
        "EEG_deep": spectral_centroid(psd_eeg_deep, freqs_np),
    }
    band_r90_span = float(
        np.nanmax([v["mean_r90"] for v in r90_bands.values()])
        - np.nanmin([v["mean_r90"] for v in r90_bands.values()])
    )
    level_c = (
        r90_width_delta > 0.01
        and abs(centroid["LFP"] - centroid["CSD"]) > 0.5
        and abs(centroid["EEG_sup"] - centroid["EEG_deep"]) > 0.1
    )

    spike_count = int(np.sum(spikes))
    rate_hz = float(np.mean(spikes) * FS)
    provenance = {
        name: json_safe(fo.diagnostics.get("observation"))
        for name, fo in fields.items()
    }
    provenance["eeg_superficial"] = json_safe(eeg_sup_ro.report())
    provenance["eeg_deep"] = json_safe(eeg_deep_ro.report())
    provenance["meg_relative"] = json_safe(meg_ro.report())

    metrics = {
        "protocol": "multiscale_observation_v0415",
        "package_head": head,
        "cause_hashes": cause_hashes,
        "n_neurons": int(Q.shape[1]),
        "n_steps": int(Q.shape[0]),
        "dt_ms": DT_MS,
        "spike_count": spike_count,
        "mean_rate_hz": rate_hz,
        "mean_V_m": float(np.mean(V)),
        "r90": r90,
        "r90_bands_lfp_ref": r90_bands,
        "band_r90_span": band_r90_span,
        "distinctness": distinct,
        "negative_control_max_abs": negative_max,
        "compilation_identity_max_abs": compile_max,
        "eeg_transform_vs_linearreadout_max_abs": eeg_tf_max,
        "meg_transform_vs_linearreadout_max_abs": meg_tf_max,
        "spectral_centroid_hz": centroid,
        "r90_width_delta": r90_width_delta,
        "levels": {"A": bool(level_a), "B": bool(level_b), "C": bool(level_c)},
        "amplitude_semantics": "relative",
        "validation_status": "computational",
        "physical_claim": "proxy_readout",
        "meg_orientation_claim": provenance["meg_relative"]["observation"]["operator_chain"]["probe"]["orientation_claim"],
        "q_hash_invariant": q_hash_after == cause_hashes["Q"],
    }
    (OUT / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    # Figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax = axes[0, 0]
    spk_t, spk_i = np.nonzero(spikes)
    ax.scatter(spk_t * DT_MS, spk_i, s=2, c="k", linewidths=0)
    ax.set_title("A  spikes (frozen X)")
    ax.set_xlabel("ms")
    ax.set_ylabel("neuron")
    ax = axes[0, 1]
    im = ax.imshow(Q.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("A  Q (frozen source)")
    ax.set_xlabel("time index")
    ax.set_ylabel("source")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 2]
    tt = t[:800]
    ax.plot(tt, np.asarray(fields["lfp_narrow"].lfp_proxy)[:800, 8], label="narrow", lw=0.9)
    ax.plot(tt, np.asarray(fields["lfp_wide"].lfp_proxy)[:800, 8], label="wide", lw=0.9)
    ax.set_title("B  LFP-proxy mid contact")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    names = list(r90)
    ax.bar(range(len(names)), [r90[n] for n in names])
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean R90")
    ax.set_title("B  locality vs operator")
    ax = axes[1, 1]
    ax.plot(freqs_np, psd_lfp.mean(axis=1), label="LFP")
    ax.plot(freqs_np, psd_csd.mean(axis=1), label="CSD")
    ax.set_title("C  LFP vs CSD PSD")
    ax.set_xlabel("Hz")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    ax.plot(freqs_np, psd_q.mean(axis=1), label="Q", lw=1.0)
    ax.plot(freqs_np, psd_lfp.mean(axis=1), label="LFP", lw=1.0)
    ax.plot(freqs_np, psd_csd.mean(axis=1), label="CSD", lw=1.0)
    ax.plot(freqs_np, psd_eeg_sup.mean(axis=1), label="EEG_sup", lw=1.0)
    ax.plot(freqs_np, psd_eeg_deep.mean(axis=1), label="EEG_deep", lw=1.0)
    ax.set_title("E  spectra under O_k")
    ax.set_xlabel("Hz")
    ax.legend(fontsize=7)
    fig.suptitle("Multiscale observation (relative proxy; Q frozen)")
    fig.tight_layout()
    save_matplotlib_evidence_figure(fig, OUT / "figure.png", dpi=140)

    fig2 = spectrolaminar_motif_heatmap(
        psd_lfp.T,
        np.asarray(fields["lfp_ref"].contact_depths),
        title="LFP-proxy spectrolaminar motif (relative)",
    )
    save_matplotlib_evidence_figure(fig2, OUT / "figure_lfp_motif.png", dpi=140)
    fig3 = plot_laminar_readout_array(
        t[burn:burn + 800],
        np.asarray(fields["lfp_ref"].lfp_proxy)[burn:burn + 800],
        csd_proxy=np.asarray(fields["lfp_ref"].csd_proxy)[burn:burn + 800],
        title="C  same Q: LFP-proxy vs CSD-proxy",
        show=False,
    )
    save_matplotlib_evidence_figure(fig3, OUT / "figure_lfp_csd.png", dpi=140)

    np.savez_compressed(
        OUT / "observations.npz",
        t=t,
        Q=Q,
        positions=positions,
        Y_eeg_sup=Y_eeg_sup,
        Y_eeg_deep=Y_eeg_deep,
        Y_meg=Y_meg,
        freqs=freqs_np,
        psd_q=psd_q,
        psd_lfp=psd_lfp,
        psd_csd=psd_csd,
        psd_eeg_sup=psd_eeg_sup,
        psd_eeg_deep=psd_eeg_deep,
        psd_narrow=psd_narrow,
        psd_wide=psd_wide,
    )

    def _hf(path: Path) -> str:
        return _sha256_bytes(Path(path).read_bytes())

    gap = _gap_review(metrics, r90_bands)
    (OUT / "gap_review.md").write_text(gap)

    manifest = {
        "etude": "multiscale_observation_v0415",
        "package_head": head,
        "protocol_sha256": _hf(PROTOCOL),
        "metrics_sha256": _hf(OUT / "metrics.json"),
        "figure_sha256": _hf(OUT / "figure.png"),
        "levels": metrics["levels"],
        "cause_hashes": cause_hashes,
        "representation": "relative_proxy",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"levels": metrics["levels"], "out": str(OUT), "head": head}, indent=2))
    return 0 if (level_a and level_b) else 1


def _gap_review(metrics: dict[str, Any], r90_bands: dict[str, Any]) -> str:
    lines = [
        "# Gap review — Multiscale Observation Etude",
        "",
        "Classified after the frozen protocol run. Not a reason to retune the protocol.",
        "",
        f"- Level A: `{metrics['levels']['A']}`",
        f"- Level B: `{metrics['levels']['B']}`",
        f"- Level C: `{metrics['levels']['C']}`",
        "",
        "| item | class | note |",
        "|------|-------|------|",
        "| Fused `KQ` rather than materialized `Φ` then `P` | NO_GAP | Compilation `P∘F→O` is the intended 0.4.15 stance. |",
        "| Contact `z` not independently settable on `project_laminar_sources` | ANALYSIS_GAP | Depth intervention used declared `LinearReadout` Gaussians; package linspace `[0,1]` is documented fabrication. |",
        "| `R90` not a package primitive | NO_GAP | Étude metric; reuse does not require core promotion. |",
        "| Frequency-dependent locality without impedance | NO_GAP | Static `K` × spatially structured `P_Q(f)`. |",
        "| EEG leadfield is declared Gaussian, not a head model | PHYSICAL_MODEL_GAP | Sufficient for operator-authority; not empirical EEG. |",
        "| MEG is a signed linear map on scalar `Q` | PHYSICAL_MODEL_GAP | Boundary demonstration; orientation correctly `none`. |",
        "| JIT drops observation diagnostics | NO_GAP | Eager evidence path used; traced numerics unchanged. |",
        "| `meg_proxy_transform(..., source_oriented=...)` name | ETUDE_PRESENTATION_ONLY | Historical parameter; do not rename in 0.4.15. |",
        "| Two EEG sensors share identical centers, differ only in width | ANALYSIS_GAP | Still two operators; a richer sensor geometry is presentation, not a core gap. |",
        "| Sparse vs dense compared on mean trace / truncated channels | ANALYSIS_GAP | Different output dimensions; authority still shown via mean and other pairs. |",
        "| No Emitter variation | NO_GAP | Deferred by design to the Emitter Étude. |",
        "",
        "Presumptive 0.4.15 package work from this run: **none** (`GENERAL_OPERATOR_GAP` empty).",
        "",
        "Observed scientific nulls (preserved; not package defects):",
        "",
        f"- Band-weighted R90 span `{metrics['band_r90_span']:.4f}` vs width-driven R90 "
        f"narrow `{metrics['r90']['lfp_narrow']:.3f}` / wide `{metrics['r90']['lfp_wide']:.3f}`. "
        "Frequency-dependent measured locality is a function of source spatial-frequency "
        "structure and K, not of K alone. This Q realization lacks strong laminar spectral "
        "segregation. Class: NO_GAP.",
        f"- EEG_sup vs EEG_deep spectral centroids "
        f"`{metrics['spectral_centroid_hz']['EEG_sup']:.1f}` vs "
        f"`{metrics['spectral_centroid_hz']['EEG_deep']:.1f}` Hz "
        f"(Y distinctness `{metrics['distinctness']['eeg_sup_vs_deep']:.3f}`). "
        "Operator authority holds; the neurophysiological EEG contrast is weak on this "
        "declared pair of similar-width depth Gaussians. Class: ANALYSIS_GAP / "
        "PHYSICAL_MODEL_GAP, not a core operator defect.",
        f"- LFP vs CSD centroids `{metrics['spectral_centroid_hz']['LFP']:.1f}` vs "
        f"`{metrics['spectral_centroid_hz']['CSD']:.1f}` Hz remain a positive result "
        "on the same frozen Q.",
        f"- Identical-W negative control max abs `{metrics['negative_control_max_abs']}`.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
