#!/usr/bin/env python3
"""Dark Plotly panels from FROZEN etude trajectories (Batch C5, B5-safe).

Consumes only committed bundle artifacts (arrays + receipts); imports no
simulate/construct path and runs no simulation (Δsimulation = 0). Currently
serves Experiment A (`canonical_source.npz`: X_V_m, X_spikes, H, Q,
positions, time_ms).

Output: ``docs/_static/etudes/<name>/*.html`` + ``manifest.json``
(figure_state GENERATED; promote with scripts/promote_atlas_state.py).

Usage:
    python scripts/plot_frozen_etude_panels.py --etude experiment_a
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9"),
)


def _dark(fig, title: str):  # type: ignore[no-untyped-def]
    fig.update_layout(title=f"<b>{title}</b>", xaxis_title="Time (ms)", **DARK)
    return fig


def _write(fig, out_dir: pathlib.Path, name: str) -> int:  # type: ignore[no-untyped-def]
    p = out_dir / name
    p.write_text(fig.to_html(include_plotlyjs="cdn", full_html=False), encoding="utf-8")
    return p.stat().st_size


def panels_experiment_a(bundle: pathlib.Path, out_dir: pathlib.Path) -> list[dict]:
    import numpy as np
    import plotly.graph_objects as go
    from jaxfne.fields import project_laminar_sources

    d = np.load(bundle / "canonical_source.npz")
    t = np.asarray(d["time_ms"], dtype=float)
    Vm = np.asarray(d["X_V_m"], dtype=float)
    spk = np.asarray(d["X_spikes"], dtype=float)
    H = np.asarray(d["H"], dtype=float)
    Q = np.asarray(d["Q"], dtype=float)
    pos = np.asarray(d["positions"], dtype=float)
    seed = int(np.asarray(d["seed"]))
    src_hash = hashlib.sha256()
    for a in (Vm, spk, H, Q):
        b = np.ascontiguousarray(a)
        src_hash.update(str(b.dtype).encode() + str(tuple(b.shape)).encode() + b.tobytes())

    out_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    stride = max(1, len(t) // 2000)
    tt = t[::stride]

    def emit(filename: str, panel: str, evidence: str, fig) -> None:  # type: ignore[no-untyped-def]
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
                    "source_artifact": "canonical_source.npz",
                    "transform": panel,
                    "units": "relative proxy",
                    "window": "full frozen run",
                },
            }
        )

    # Raster (event scatter, cap points).
    ti, ni = np.nonzero(spk > 0.5)
    order = np.linspace(0, len(ti) - 1, min(len(ti), 20000)).astype(int)
    fig = go.Figure(
        go.Scatter(
            x=t[ti[order]],
            y=ni[order],
            mode="markers",
            marker=dict(size=2, opacity=0.6),
            name="spikes",
        )
    )
    fig.update_layout(yaxis_title="Neuron index")
    emit("raster.html", "Spike raster", "OBSERVED", fig)

    # Membrane traces (subset, time-downsampled like atlas panels).
    fig = go.Figure()
    for j in range(min(8, Vm.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=Vm[::stride, j], mode="lines", name=f"unit {j}"))
    fig.update_layout(yaxis_title="V_m proxy")
    emit("traces.html", "Membrane traces", "OBSERVED", fig)

    # H dynamics (population mean ± std, time-downsampled).
    mu, sd = H.mean(axis=1), H.std(axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=(mu + sd)[::stride],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tt,
            y=(mu - sd)[::stride],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="±1 std",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=tt, y=mu[::stride], mode="lines", name="mean H"))
    fig.update_layout(yaxis_title="H (identity RBS container)")
    emit("h_dynamics.html", "H dynamics", "OBSERVED", fig)

    # LFP from frozen Q (declared Gaussian projection, same as protocol).
    fo = project_laminar_sources(Q, pos, n_contacts=16, width=0.10)
    lfp = np.asarray(fo.lfp_proxy)
    fig = go.Figure()
    for c in range(min(8, lfp.shape[1])):
        fig.add_trace(go.Scatter(x=tt, y=lfp[::stride, c], mode="lines", name=f"contact {c}"))
    fig.update_layout(yaxis_title="LFP proxy")
    emit("lfp.html", "LFP proxy", "DERIVED", fig)

    # Oscillatory response (Welch PSD of the aggregate readout).
    from jaxfne.analysis.spectral import spectrolaminar_psd_jax
    import jax.numpy as jnp

    y = lfp.mean(axis=1).astype(np.float32)
    freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
    psd = np.asarray(
        spectrolaminar_psd_jax(
            jnp.asarray(y[None, :, None]), fs=float(1000.0 / float(t[1] - t[0])), freqs=freqs
        )
    )
    fig = go.Figure(
        go.Scatter(
            x=freqs, y=psd.mean(axis=(0, 2)) if psd.ndim == 3 else psd, mode="lines", name="PSD"
        )
    )
    fig.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Power", xaxis_type="log", yaxis_type="log"
    )
    emit("oscillatory.html", "Oscillatory response", "DERIVED", fig)

    finalize(
        out_dir,
        "experiment_a",
        "Experiment A — frozen-trajectory panels (no resimulation)",
        panels,
        {
            "source_bundle": "artifacts/etudes/experiment_a",
            "verification_tier": "frozen",
            "source_trajectory_sha256": src_hash.hexdigest()[:16],
            "seed": seed,
            "n_neurons": int(Vm.shape[1]),
            "n_steps": int(Vm.shape[0]),
            "simulation_runs": 0,
        },
    )
    return panels


def finalize(out_dir: pathlib.Path, etude: str, title: str, panels: list, extra: dict) -> dict:
    """Write manifest.json + dark index.html for one étude panel set."""
    manifest = {
        "suite": "etude_posthoc.v1",
        "etude": etude,
        "figure_state": "GENERATED",
        "jaxfne_version": __import__("jaxfne").__version__,
        "panels": panels,
        "sha256": hashlib.sha256(
            json.dumps(panels, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    manifest.update(extra)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Index page (dark).
    cards = "".join(
        f'<article class="card"><div class="card-header"><h2>{p["panel"]}</h2></div>'
        f'<div class="card-body"><p class="meta">{p["evidence"]} · {p["bytes"] / 1024:.1f} KB</p>'
        f'<a class="btn" href="{p["file"]}" target="_blank" rel="noopener">Open</a></div></article>'
        for p in panels
    )
    (out_dir / "index.html").write_text(
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:sans-serif;"
        "background:#0d1117;color:#c9d1d9;padding:2rem}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:1.5rem}"
        ".card{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden}"
        ".card-header{background:#21262d;color:#f0f6fc;padding:1.25rem}"
        ".card-body{padding:1.25rem}.meta{color:#8b949e;font-size:.85rem;margin-bottom:1rem}"
        ".btn{display:inline-block;padding:.6rem 1.2rem;background:#1f6feb;color:#fff;"
        "text-decoration:none;border-radius:6px}</style></head><body>"
        f"<h1>{title}</h1>"
        f"<div class='grid'>{cards}</div></div></body></html>",
        encoding="utf-8",
    )
    return manifest


ETUDES = {"experiment_a": panels_experiment_a}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etude", required=True, choices=sorted(ETUDES))
    args = parser.parse_args(argv)
    out = ROOT / "docs" / "_static" / "etudes" / args.etude
    panels = ETUDES[args.etude](ROOT / "artifacts" / "etudes" / args.etude, out)
    print(f"[{args.etude}] {len(panels)} panels -> {out.relative_to(ROOT)}")
    for p in panels:
        print(f"    {p['file']}: {p['evidence']} {p['status']} {p['bytes']}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
