"""Atlas suite — generic visualization atlas for ANY jaxfne model (N>=1).

Portable by construction: only the jaxfne public surface is used
(Model.neuron_table / edge_table / summary, Signals.time_ms/V_m/spikes/
sources/field, jaxfne.vis.canonical plotters, jaxfne.simulate). No jomission
imports. Copy verbatim to ``jaxfne/jaxfne/vis/atlas_suite.py`` as the standard
suite (see docs/ATLAS_SUITE_HANDOUT.md + docs/jaxfne_atlas_suite_dropin/).

Seven fixed panels (always emitted, even for a single neuron; a panel whose
data contract cannot be met is emitted as an explicit omission card, never
skipped and never substituted):
  1. schema.html         — network_hspice_plotly(model)                OBSERVED
  2. network_3d.html      — plot_network_3d(model)                    OBSERVED
  3. raster.html          — plot_raster(signals, model)               OBSERVED
  4. lfp.html             — plot_lfp (+plot_csd) from recorded field  DERIVED
  5. h_dynamics.html      — recorded H (HDP H_trace / homeostasis)    DERIVED
  6. hdp.html             — mutable weight diagnostics (w_trace)      DERIVED
  7. oscillatory.html     — plot_psd (+spectrogram when feasible)     DERIVED

N=1 degradation rules: no panel is skipped. Empty edge lists render no
arrows on the schema; short runs render PSD-only oscillatory response;
unrecorded field/H/HDP render explicit omission cards stating what was not
recorded. Every file carries a provenance
card (config_hash, N, edges, steps, dt, jaxfne version, evidence level).

Entry point:
    build_atlas(model, signals=None, out_dir=..., simulate_fn=None, ...) -> dict
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

DT_SOURCE_INFERRED = "INFERRED_FROM_TIME_GRID"
DT_SOURCE_FALLBACK = "FALLBACK_EXPLICIT_DT"
DT_SOURCE_INVALID = "INVALID_TIME_GRID"


PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("schema.html", "Circuit schematic", "OBSERVED"),
    ("network_3d.html", "Network 3D", "OBSERVED"),
    ("raster.html", "Spike raster", "OBSERVED"),
    ("lfp.html", "LFP proxy", "DERIVED"),
    ("h_dynamics.html", "H dynamics", "DERIVED"),
    ("hdp.html", "HDP plasticity", "DERIVED"),
    ("oscillatory.html", "Oscillatory response", "DERIVED"),
)


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


def _live_jaxfne_version() -> str:
    try:
        import jaxfne as _J

        return str(getattr(_J, "__version__", "unknown"))
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class DtInference:
    """Classified outcome of deriving ``dt_ms`` from a realized time grid."""

    dt_ms: float
    dt_source: str
    fallback_reason: str | None = None


def _raise_invalid_time_grid(reason: str) -> None:
    raise ValueError(f"INVALID_TIME_GRID ({reason})")


def classify_dt_ms(time_ms: np.ndarray | None, fallback: float) -> DtInference:
    """Classify how ``dt_ms`` was chosen from ``time_ms`` and an explicit fallback.

    Outcomes:
      - ``INFERRED_FROM_TIME_GRID``: uniform, finite, strictly increasing grid.
      - ``FALLBACK_EXPLICIT_DT``: missing or insufficient time information.

    A present but invalid grid raises ``ValueError`` with reason
    ``INVALID_TIME_GRID (...)``. Unexpected errors propagate unchanged.
    """
    fb = float(fallback)
    if time_ms is None:
        return DtInference(fb, DT_SOURCE_FALLBACK, "missing_time")
    t = np.asarray(time_ms, dtype=float).ravel()
    if t.size < 2:
        return DtInference(fb, DT_SOURCE_FALLBACK, "insufficient_samples")
    if not np.all(np.isfinite(t)):
        _raise_invalid_time_grid("non_finite")
    diffs = np.diff(t)
    if diffs.size == 0 or not np.all(diffs > 0):
        _raise_invalid_time_grid("non_monotonic_or_nonpositive")
    median = float(np.median(diffs))
    if median <= 0:
        _raise_invalid_time_grid("nonpositive_median")
    # Realized simulation grids are float32-accumulated; 1e-4 rejects genuine
    # non-uniformity (e.g. mixed step sizes) while accepting uniform sim output.
    if not np.allclose(diffs, median, rtol=1e-4, atol=0.0):
        _raise_invalid_time_grid("non_uniform")
    return DtInference(median, DT_SOURCE_INFERRED, None)


def _infer_dt_ms(time_ms: np.ndarray | None, fallback: float) -> float:
    """Return the classified ``dt_ms`` value (see ``classify_dt_ms``)."""
    return classify_dt_ms(time_ms, fallback).dt_ms


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




class _OmitPanel(Exception):
    """A panel whose data contract cannot be met: emit an explicit omission card."""


def _stride(n: int, max_pts: int = 2000) -> int:
    return max(1, int(np.ceil(int(n) / max_pts)))


def _mean_band(x: Any):  # type: ignore[no-untyped-def]
    """Mean ± std over units (last axes); 1-D input returns the trace thrice."""
    a = np.asarray(x, dtype=float)
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if a.ndim == 1:
        return a, a, a
    mu = a.mean(axis=1)
    sd = a.std(axis=1)
    return mu, mu - sd, mu + sd


def _hdp_diagnostics_of(model: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for meth in ("last_hdp_diagnostics", "last_homeostasis_diagnostics"):
        fn = getattr(model, meth, None)
        if not callable(fn):
            continue
        try:
            d = fn()
        except Exception:
            d = None
        if isinstance(d, dict):
            out[meth] = d
    return out


def _band_fig(title: str, mu: Any, lo: Any, hi: Any, ylabel: str):  # type: ignore[no-untyped-def]
    import plotly.graph_objects as go

    n = len(np.asarray(mu))
    s = _stride(n)
    idx = np.arange(0, n, s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=np.asarray(hi)[idx], mode="lines",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=np.asarray(lo)[idx], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             name="±1 std", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=idx, y=np.asarray(mu)[idx], mode="lines",
                             name="mean"))
    fig.update_layout(title=f"<b>{title}</b>", xaxis_title="step",
                      yaxis_title=ylabel)
    return fig


def _h_dynamics_fig(model: Any):  # type: ignore[no-untyped-def]
    """Recorded hidden-state trajectory, or an explicit omission."""
    diags = _hdp_diagnostics_of(model)
    hdp = diags.get("last_hdp_diagnostics") or {}
    if hdp.get("H_trace") is not None:
        mu, lo, hi = _mean_band(hdp["H_trace"])
        fig = _band_fig("H dynamics (HDP H_trace, population mean ± std)", mu, lo, hi, "H")
        fig.update_layout(annotations=[dict(text="H from the HDP run",
                                            xref="paper", yref="paper", x=0.5, y=1.08,
                                            showarrow=False)])
        return fig
    home = diags.get("last_homeostasis_diagnostics") or {}
    if home.get("r_trace") is not None:
        mu, lo, hi = _mean_band(home["r_trace"])
        return _band_fig("H dynamics (homeostasis rate trace, mean ± std)", mu, lo, hi, "r")
    raise _OmitPanel("no H recorded — run with enable_hdp or enable_homeostasis")


def _hdp_fig(model: Any):  # type: ignore[no-untyped-def]
    """Mutable weight diagnostics, or an explicit omission (never inferred)."""
    diags = _hdp_diagnostics_of(model)
    hdp = diags.get("last_hdp_diagnostics")
    if not hdp:
        raise _OmitPanel("HDP not enabled on this run")
    W = hdp.get("w_trace")
    if W is None:
        raise _OmitPanel("weight trace not recorded (recording_budget: terminal weights only)")
    a = np.asarray(W, dtype=float)
    if a.size == 0:
        raise _OmitPanel("weight trace empty")
    if a.nbytes > 64_000_000:
        raise _OmitPanel("weight trace exceeds the 64 MB plot budget; "
                         "rerun with a sparse/aggregate recording_budget")
    mag = np.abs(a.reshape(a.shape[0], -1))
    mu, lo, hi = _mean_band(mag)
    return _band_fig("HDP weight magnitude (population mean ± std)", mu, lo, hi, "|w|")


def _lfp_fig(signals: Any):  # type: ignore[no-untyped-def]
    """Recorded field proxy, or an explicit omission (never substituted)."""
    from jaxfne.vis import canonical as C

    try:
        return C.plot_lfp(signals, backend="plotly")
    except Exception:
        pass
    try:
        return C.plot_csd(signals, backend="plotly")
    except Exception as exc:
        raise _OmitPanel(f"no field proxy recorded ({type(exc).__name__})") from exc


def _schema_fig(model: Any):  # type: ignore[no-untyped-def]
    from jaxfne.vis import network_inspect as NI

    return NI.network_hspice_plotly(model, theme="dark")["fig"]


def _discover_upstream_lineage(model: Any, upstream: Dict[str, Any]) -> tuple[Any, Any]:
    """Return (tfne_digest, k_d), preferring caller-supplied values.

    Auto-discovery inspects tensor provenance stashed on the model; anything
    undiscoverable stays ``None`` (recorded as ``null``, never guessed).
    """
    tfne_digest = upstream.get("tfne_digest")
    k_d = upstream.get("k_d")
    if tfne_digest is None or k_d is None:
        try:
            params = getattr(model, "params", None)
            prov = None
            if isinstance(params, dict):
                prov = params.get("tensor_provenance")
            if not isinstance(prov, dict):
                prov = getattr(model, "provenance", None)
            if isinstance(prov, dict):
                if tfne_digest is None:
                    tfne_digest = (prov.get("genome_sha256")
                                   or prov.get("tfne_digest"))
                if k_d is None:
                    k_d = (prov.get("development_seed")
                           if "development_seed" in prov else prov.get("k_d"))
        except Exception:
            pass
    return tfne_digest, k_d


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
    provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the 7-panel atlas for any model. Never skips a fixed panel.

    Provide ``signals`` directly, or ``simulate_fn(model) -> signals``; else a
    default ``jaxfne.simulate(model, Simulation(...))`` run is attempted.
    ``provenance`` optionally carries upstream lineage the atlas cannot
    observe itself (``tfne_digest``, ``k_d``, ``recording_class``,
    ``recording_budget``); undiscoverable fields record ``null``, never a
    guess. Every emitted manifest starts at figure state ``GENERATED``.
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
    # The realized signals' time grid is authoritative for dt_ms when inferable.
    # The parameter default (0.1) only describes the fallback simulation below.
    dt_info = classify_dt_ms(arr["time_ms"], dt_ms)
    dt_ms = dt_info.dt_ms
    dt_provenance = {
        "dt_source": dt_info.dt_source,
        **(
            {"dt_fallback_reason": dt_info.fallback_reason}
            if dt_info.fallback_reason is not None
            else {}
        ),
    }
    jaxfne_version = _live_jaxfne_version()
    upstream = dict(provenance or {})
    tfne_digest, k_d = _discover_upstream_lineage(model, upstream)
    rec_class = upstream.get("recording_class")
    rec_budget = upstream.get("recording_budget")
    if rec_class is None:
        # Recording-budget policy (Batch A3): the limiting object is
        # T×(H + W + recorded), not neuron count alone.
        if n_neurons <= 64 and n_edges <= 20000:
            rec_class = "small-mechanistic"
            rec_budget = rec_budget or "full H, HDP targets, representative W, diagnostics"
        elif n_neurons >= 1000 or n_edges > 200000:
            rec_class = "large"
            rec_budget = rec_budget or ("population H, selected traces, summaries; "
                                        "no full T×E")
        else:
            rec_class = "medium"
            rec_budget = rec_budget or "population H, selected traces, summaries"

    os.makedirs(out_dir, exist_ok=True)
    manifest_panels: List[Dict[str, Any]] = []

    def _apply_dark_theme(fig: Any) -> Any:
        # Presentation only (never information/semantics): match the slate
        # docs chrome so Plotly panels read as one dark system. Trace data,
        # counts, and provenance are untouched.
        try:
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                              plot_bgcolor="#161b22",
                              font=dict(color="#c9d1d9"))
        except Exception:
            pass
        return fig

    def _emit(filename: str, panel: str, evidence: str, caption: str,
              make_fig: Callable[[], Any], extra: Dict[str, Any] | None = None,
              lineage: Dict[str, str] | None = None) -> None:
        degradation_status = "AVAILABLE"
        try:
            fig = _apply_dark_theme(make_fig())
            fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        except _OmitPanel as omit:
            degradation_status = "OMITTED"
            fig_html = _empty_note_fig(f"{panel}: omitted — {omit}").to_html(
                include_plotlyjs="cdn", full_html=False)
            caption = caption + f" [omitted: {omit}]"
            extra = dict(extra or {}, omitted=str(omit)[:200])
        except Exception as exc:  # never skip: emit placeholder with reason
            degradation_status = "ERROR"
            fig_html = _empty_note_fig(f"{panel}: unavailable ({type(exc).__name__})").to_html(
                include_plotlyjs="cdn", full_html=False)
            caption = caption + f" [placeholder: {type(exc).__name__}: {exc}]"
            extra = dict(extra or {}, placeholder=str(exc)[:200])
        prov = _provenance(config_hash=config_hash, n_neurons=n_neurons, n_edges=n_edges,
                           n_steps=n_steps, dt_ms=dt_ms, evidence=evidence,
                           extra={**dt_provenance, **(extra or {})})
        html = _wrap_html(f"{title} — {panel}", fig_html, caption, prov, evidence)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        manifest_panels.append({"file": filename, "panel": panel, "evidence": evidence,
                                "status": degradation_status,
                                "bytes": os.path.getsize(path),
                                "inputs": dict(lineage or {})})

    _emit("schema.html", "Circuit schematic", "OBSERVED",
          "Block schematic from realized metadata: areas, layers, classes, area-pair links.",
          lambda: _schema_fig(model),
          lineage={"source_artifact": "model",
                   "variable": "neuron_table + params.edge_list",
                   "transform": "network_hspice_plotly",
                   "units": "counts + relative weights", "window": "static"})
    _emit("network_3d.html", "Network 3D", "OBSERVED",
          "Realized geometry + edges from the model (single marker when N=1).",
          lambda: C.plot_network_3d(model, backend="plotly"),
          lineage={"source_artifact": "model",
                   "variable": "params.positions + params.edge_list",
                   "transform": "plot_network_3d",
                   "units": "relative geometry", "window": "static"})
    _emit("raster.html", "Spike raster", "OBSERVED",
          "Spike times vs neuron index from simulated signals.",
          lambda: C.plot_raster(signals, model, backend="plotly"),
          lineage={"source_artifact": "signals",
                   "variable": "spikes [T,N] + neuron_table",
                   "transform": "plot_raster",
                   "units": "spike indicators", "window": "full run"})
    _emit("lfp.html", "LFP proxy", "DERIVED",
          "LFP/CSD proxy traces from the recorded field (omitted explicitly when unrecorded).",
          lambda: _lfp_fig(signals),
          lineage={"source_artifact": "signals",
                   "variable": "field proxy",
                   "transform": "plot_lfp/plot_csd",
                   "units": "relative proxy", "window": "full run"})
    _emit("h_dynamics.html", "H dynamics", "DERIVED",
          "Recorded hidden-state trajectory: HDP H_trace or homeostasis trace "
          "(omitted explicitly when unrecorded).",
          lambda: _h_dynamics_fig(model),
          lineage={"source_artifact": "model diagnostics",
                   "variable": "H_trace / r_trace",
                   "transform": "population mean ± std band",
                   "units": "relative state", "window": "full run"})
    _emit("hdp.html", "HDP plasticity", "DERIVED",
          "Mutable weight diagnostics from the HDP run (omitted explicitly when "
          "HDP is off or the trace is unrecorded; never inferred from activity).",
          lambda: _hdp_fig(model),
          lineage={"source_artifact": "model diagnostics",
                   "variable": "w_trace",
                   "transform": "population |w| mean ± std band",
                   "units": "relative weights", "window": "full run"})

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

    _emit("oscillatory.html", "Oscillatory response", "DERIVED",
          "Welch PSD (plus spectrogram when the run is long enough) over the declared window.",
          _spectral_fig,
          lineage={"source_artifact": "signals",
                   "variable": "field traces",
                   "transform": "plot_psd (+plot_spectrogram)",
                   "units": "relative power", "window": "full run"})

    # Index atlas page.
    cards = []
    for p in manifest_panels:
        cards.append(
            f'<article class="card"><div class="card-header"><h2>{p["panel"]}</h2></div>'
            f'<div class="card-body"><p class="meta">{p["evidence"]} · {p["bytes"]/1024:.1f} KB</p>'
            f'<a class="btn" href="{p["file"]}" target="_blank" rel="noopener">Open</a></div></article>')
    index_html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} — atlas_suite</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:sans-serif;background:#0d1117;color:#c9d1d9;padding:2rem}}
.container{{max-width:1200px;margin:0 auto}}header{{text-align:center;margin-bottom:2rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:1.5rem}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden}}.card-header{{background:#21262d;color:#f0f6fc;padding:1.25rem}}
.card-body{{padding:1.25rem}}.meta{{color:#8b949e;font-size:.85rem;margin-bottom:1rem}}
.btn{{display:inline-block;padding:.6rem 1.2rem;background:#1f6feb;color:#fff;text-decoration:none;border-radius:6px}}</style>
</head><body><div class="container"><header><h1>{title}</h1>
<p>N={n_neurons} · edges={n_edges} · steps={n_steps} · config {config_hash[:12]}</p></header>
<div class="grid">{''.join(cards)}</div></div></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    manifest = {
        "suite": "atlas_suite.v2",
        "title": title,
        "figure_state": "GENERATED",
        "config_hash": config_hash,
        "tfne_digest": tfne_digest,
        "k_d": k_d,
        "sim_identity": (f"seed={int(seed)} duration_ms={float(duration_ms)} "
                         f"dt_ms={dt_ms}"),
        "recording_class": rec_class,
        "recording_budget": rec_budget,
        "jaxfne_version": jaxfne_version,
        "seed": int(seed),
        "duration_ms": float(duration_ms),
        "n_neurons": n_neurons,
        "n_edges": n_edges,
        "n_steps": n_steps,
        "dt_ms": dt_ms,
        "dt_source": dt_info.dt_source,
        **(
            {"dt_fallback_reason": dt_info.fallback_reason}
            if dt_info.fallback_reason is not None
            else {}
        ),
        "panels": manifest_panels,
        "sha256": hashlib.sha256(
            json.dumps(manifest_panels, sort_keys=True, default=str).encode()).hexdigest()[:16],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
