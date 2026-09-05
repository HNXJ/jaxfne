"""Atlas suite — generic visualization atlas for ANY jaxfne model (N>=1).

Portable by construction: only the jaxfne public surface is used
(Model.neuron_table / edge_table / summary, Signals.time_ms/V_m/spikes/
sources/field, jaxfne.vis.canonical plotters, jaxfne.simulate). No jomission
imports. Copy verbatim to ``jaxfne/jaxfne/vis/atlas_suite.py`` as the standard
suite (see docs/ATLAS_SUITE_HANDOUT.md + docs/jaxfne_atlas_suite_dropin/).

Six fixed panels (always emitted, even for a single neuron):
  1. network_3d.html      — plot_network_3d(model)                    OBSERVED
  2. connectivity.html    — plot_connectivity(model)                  OBSERVED
  3. raster.html          — plot_raster(signals, model)               OBSERVED
  4. traces.html          — plot_membrane_potentials(signals, model)  OBSERVED
  5. spectral.html        — plot_psd (+spectrogram when feasible)     DERIVED
  6. state_summary.html   — summary + per-cell-type rates (custom)    DERIVED
Optional:
  field.html             — plot_lfp (+plot_csd) when field non-empty  DERIVED

N=1 degradation rules: no panel is skipped. Empty edge lists render the
backend's empty-matrix figure; short runs render PSD-only spectral; missing
field skips only the optional field.html. Every file carries a provenance
card (config_hash, N, edges, steps, dt, jaxfne version, evidence level).

Entry point:
    build_atlas(model, signals=None, out_dir=..., simulate_fn=None, ...) -> dict
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("network_3d.html", "Network 3D", "OBSERVED"),
    ("connectivity.html", "Connectivity", "OBSERVED"),
    ("raster.html", "Spike raster", "OBSERVED"),
    ("traces.html", "Membrane traces", "OBSERVED"),
    ("spectral.html", "Spectral (PSD)", "DERIVED"),
    ("state_summary.html", "State summary", "DERIVED"),
)

OPTIONAL_FIELD_FILE = "field.html"


def _np(arr: Any) -> np.ndarray | None:
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
        return a if a.size else None
    except Exception:
        return None


def _model_counts(model: Any) -> Dict[str, Any]:
    try:
        neurons = list(model.neuron_table())
    except Exception:
        neurons = []
    try:
        edges = list(model.edge_table())
    except Exception:
        edges = []
    try:
        summary = dict(model.summary())
    except Exception:
        summary = {}
    return {"neurons": neurons, "edges": edges, "summary": summary}


def _signals_arrays(signals: Any) -> Dict[str, Any]:
    def _get(name: str) -> np.ndarray | None:
        return _np(getattr(signals, name, None))

    return {
        "time_ms": _get("time_ms"),
        "V_m": _get("V_m"),
        "spikes": _get("spikes"),
        "sources": _get("sources"),
        "field": _get("field"),
    }


def _provenance(
    *,
    config_hash: str,
    n_neurons: int,
    n_edges: int,
    n_steps: int,
    dt_ms: float,
    evidence: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    try:
        import jaxfne as _J

        jaxfne_version = getattr(_J, "__version__", "unknown")
    except Exception:
        jaxfne_version = "unknown"
    prov: Dict[str, str] = {
        "config_hash": str(config_hash),
        "neurons": str(n_neurons),
        "edges": str(n_edges),
        "steps": str(n_steps),
        "dt_ms": str(dt_ms),
        "jaxfne": str(jaxfne_version),
        "evidence": str(evidence),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "calibration": "relative_proxy_readout (never calibrated physical units)",
    }
    for k, v in (extra or {}).items():
        prov[str(k)] = str(v)
    return prov


def _wrap_html(title: str, fig_html: str, caption: str, provenance: Dict[str, str], evidence: str) -> str:
    badge = "#238636" if evidence == "OBSERVED" else "#1f6feb"
    rows = "".join(
        f"<tr><td style='color:#8b949e;padding:3px 12px 3px 0;'><b>{k}</b></td>"
        f"<td style='color:#c9d1d9;font-family:monospace;'>{v}</td></tr>"
        for k, v in provenance.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — atlas_suite</title>
<style>body{{background:#0d1117;color:#c9d1d9;font-family:sans-serif;margin:0;padding:24px;display:flex;flex-direction:column;align-items:center}}
.container{{width:100%;max-width:1280px}}.box{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px 20px;margin-top:16px;font-size:13px;line-height:1.6}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;color:#fff;background:{badge};margin-bottom:8px}}
table{{border-collapse:collapse;margin-top:8px}}</style></head>
<body><div class="container">{fig_html}
<div class="box"><span class="badge">{evidence}</span><div>{caption}</div></div>
<div class="box"><b>Provenance</b><table>{rows}</table></div></div></body></html>"""


def _empty_note_fig(text: str):  # type: ignore[no-untyped-def]
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False)
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font=dict(color="#c9d1d9"))
    return fig


def _build_state_summary_fig(model: Any, signals: Any):  # type: ignore[no-untyped-def]
    """Rates per cell-type + summary counts. Pure numpy/plotly, N>=1 safe."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    counts = _model_counts(model)
    neurons = counts["neurons"]
    arr = _signals_arrays(signals)
    spikes = arr["spikes"]
    time_ms = arr["time_ms"]
    dur_ms = float(time_ms[-1] - time_ms[0]) if time_ms is not None and len(time_ms) > 1 else 0.0

    cell_types: List[str] = []
    for n in neurons:
        ct = str(n.get("cell_type", "?"))
        if ct not in cell_types:
            cell_types.append(ct)
    if not cell_types:
        cell_types = ["?"]

    rates, silence = [], []
    if spikes is not None and spikes.ndim == 2 and dur_ms > 0:
        n_units = spikes.shape[1]
        for ct in cell_types:
            idx = [i for i, n in enumerate(neurons)
                   if str(n.get("cell_type", "?")) == ct and i < n_units]
            if not idx:
                rates.append(0.0)
                silence.append(100.0)
                continue
            sub = np.asarray(spikes[:, idx], dtype=float)
            per_unit_hz = sub.sum(axis=0) / (dur_ms / 1000.0)
            rates.append(float(per_unit_hz.mean()))
            silence.append(float((per_unit_hz == 0).mean() * 100.0))
    else:
        rates = [0.0] * len(cell_types)
        silence = [100.0] * len(cell_types)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("<b>Mean rate by cell type</b>",
                                        "<b>Silence fraction by cell type</b>"))
    fig.add_trace(go.Bar(x=cell_types, y=rates, name="Hz",
                         text=[f"{r:.2f} Hz" for r in rates], textposition="auto"), row=1, col=1)
    fig.add_trace(go.Bar(x=cell_types, y=silence, name="% silent",
                         text=[f"{s:.0f}%" for s in silence], textposition="auto"), row=1, col=2)
    fig.update_xaxes(title_text="Cell type", row=1, col=1)
    fig.update_yaxes(title_text="Hz", row=1, col=1)
    fig.update_xaxes(title_text="Cell type", row=1, col=2)
    fig.update_yaxes(title_text="% silent", range=[0, 100], row=1, col=2)
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font=dict(color="#c9d1d9"), width=1180, height=520,
                      title="<b>State summary</b> (rates from spike counts; DERIVED)")
    return fig, rates, silence, dur_ms


def build_atlas(
    model: Any,
    signals: Any | None = None,
    *,
    out_dir: str = "docs/_static/atlas",
    simulate_fn: Callable[[Any], Any] | None = None,
    duration_ms: float = 500.0,
    dt_ms: float = 0.1,
    seed: int = 0,
    title: str = "Model atlas",
) -> Dict[str, Any]:
    """Build the 6-panel atlas for any model. Never skips a fixed panel.

    Provide ``signals`` directly, or ``simulate_fn(model) -> signals``; else a
    default ``jaxfne.simulate(model, Simulation(...))`` run is attempted.
    Returns the manifest dict and writes ``<out_dir>/*.html`` + manifest.json.
    """
    from jaxfne.vis import canonical as C

    if signals is None:
        if simulate_fn is not None:
            signals = simulate_fn(model)
        else:
            import jaxfne as _J

            sim = _J.Simulation(duration_ms=float(duration_ms), dt_ms=float(dt_ms), seed=int(seed))
            signals = _J.simulate(model, sim)

    counts = _model_counts(model)
    n_neurons = len(counts["neurons"]) or int(counts["summary"].get("n_units", 0) or 0)
    n_edges = len(counts["edges"])
    config_hash = str(counts["summary"].get("config_hash", "unknown"))
    arr = _signals_arrays(signals)
    n_steps = int(arr["time_ms"].shape[0]) if arr["time_ms"] is not None else (
        int(arr["spikes"].shape[0]) if arr["spikes"] is not None else 0)

    os.makedirs(out_dir, exist_ok=True)
    manifest_panels: List[Dict[str, Any]] = []

    def _emit(filename: str, panel: str, evidence: str, caption: str,
              make_fig: Callable[[], Any], extra: Dict[str, Any] | None = None) -> None:
        degradation_status = "AVAILABLE"
        try:
            fig = make_fig()
            fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        except Exception as exc:  # never skip: emit placeholder with reason
            degradation_status = "ERROR"
            fig_html = _empty_note_fig(f"{panel}: unavailable ({type(exc).__name__})").to_html(
                include_plotlyjs="cdn", full_html=False)
            caption = caption + f" [placeholder: {type(exc).__name__}: {exc}]"
            extra = dict(extra or {}, placeholder=str(exc)[:200])
        prov = _provenance(config_hash=config_hash, n_neurons=n_neurons, n_edges=n_edges,
                           n_steps=n_steps, dt_ms=dt_ms, evidence=evidence, extra=extra)
        html = _wrap_html(f"{title} — {panel}", fig_html, caption, prov, evidence)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        manifest_panels.append({"file": filename, "panel": panel, "evidence": evidence,
                                "status": degradation_status,
                                "bytes": os.path.getsize(path)})

    _emit("network_3d.html", "Network 3D", "OBSERVED",
          "Realized geometry + edges from the model (single marker when N=1).",
          lambda: C.plot_network_3d(model, backend="plotly"))
    _emit("connectivity.html", "Connectivity", "OBSERVED",
          "Realized weight matrix (empty-matrix card when the model has no edges).",
          lambda: C.plot_connectivity(model, backend="plotly"))
    _emit("raster.html", "Spike raster", "OBSERVED",
          "Spike times vs neuron index from simulated signals.",
          lambda: C.plot_raster(signals, model, backend="plotly"))
    _emit("traces.html", "Membrane traces", "OBSERVED",
          "Membrane potential proxy traces for a subset of neurons.",
          lambda: C.plot_membrane_potentials(signals, model, backend="plotly"))

    def _spectral_fig():  # type: ignore[no-untyped-def]
        from plotly.subplots import make_subplots

        psd_fig = C.plot_psd(signals, backend="plotly")
        try:
            spec_fig = C.plot_spectrogram(signals, backend="plotly")
            combo = make_subplots(rows=1, cols=2,
                                  subplot_titles=("<b>PSD</b>", "<b>Spectrogram</b>"))
            for tr in psd_fig.data:
                combo.add_trace(tr, row=1, col=1)
            for tr in spec_fig.data:
                combo.add_trace(tr, row=1, col=2)
            combo.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                                font=dict(color="#c9d1d9"), width=1180, height=520)
            return combo
        except Exception:
            return psd_fig

    _emit("spectral.html", "Spectral", "DERIVED",
          "Welch PSD (plus spectrogram when the run is long enough).",
          _spectral_fig)

    def _op_fig():  # type: ignore[no-untyped-def]
        fig, rates, silence, dur = _build_state_summary_fig(model, signals)
        _op_fig.info = {"rates": rates, "silence": silence, "dur_ms": dur}  # type: ignore[attr-defined]
        return fig

    _emit("state_summary.html", "State summary", "DERIVED",
          "Mean rate + silence fraction per cell type from spike counts; counts from model.summary().",
          _op_fig, extra={"summary": json.dumps(counts["summary"], default=str)[:500]})

    # Optional field panel — only when a non-empty field proxy exists.
    field = arr["field"]
    if field is not None:
        def _field_fig():  # type: ignore[no-untyped-def]
            try:
                return C.plot_lfp(signals, backend="plotly")
            except Exception:
                return C.plot_csd(signals, backend="plotly")

        _emit(OPTIONAL_FIELD_FILE, "Field proxy", "DERIVED",
              "LFP/CSD proxy traces when the model records a field.",
              _field_fig)

    # Index atlas page.
    cards = []
    for p in manifest_panels:
        cards.append(
            f'<article class="card"><div class="card-header"><h2>{p["panel"]}</h2></div>'
            f'<div class="card-body"><p class="meta">{p["evidence"]} · {p["bytes"]/1024:.1f} KB</p>'
            f'<a class="btn" href="{p["file"]}" target="_blank" rel="noopener">Open</a></div></article>')
    index_html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} — atlas_suite</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:sans-serif;background:#f5f5f5;padding:2rem}}
.container{{max-width:1200px;margin:0 auto}}header{{text-align:center;margin-bottom:2rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:1.5rem}}
.card{{background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.card-header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:1.25rem}}
.card-body{{padding:1.25rem}}.meta{{color:#888;font-size:.85rem;margin-bottom:1rem}}
.btn{{display:inline-block;padding:.6rem 1.2rem;background:#667eea;color:#fff;text-decoration:none;border-radius:6px}}</style>
</head><body><div class="container"><header><h1>{title}</h1>
<p>N={n_neurons} · edges={n_edges} · steps={n_steps} · config {config_hash[:12]}</p></header>
<div class="grid">{''.join(cards)}</div></div></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    manifest = {
        "suite": "atlas_suite.v1",
        "title": title,
        "config_hash": config_hash,
        "n_neurons": n_neurons,
        "n_edges": n_edges,
        "n_steps": n_steps,
        "dt_ms": dt_ms,
        "panels": manifest_panels,
        "sha256": hashlib.sha256(
            json.dumps(manifest_panels, sort_keys=True, default=str).encode()).hexdigest()[:16],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
