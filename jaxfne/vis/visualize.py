"""Standard 8-panel simulation visualization bundle (optional, zero-overhead).

`sources = jtfne.simulate(model)` remains the sole simulation contract.
`jtfne.visualize(model, signals)` is an additive, optional post-hoc layer
that reuses the existing jaxfne.vis surface — it adds no numerics, no new
field solver, and no physical-amplitude claim. All panels are proxy
readouts (relative units) backed by the same Izhikevich scaffold.

8 panels
--------
1. Network structure (positions/layers/E-I)
2. Connectivity (matrix / sparse)
3. Parameter summary (weights/delays/state ownership)
4. Raster
5. Population rates
6. State traces (V, H, W)
7. Source Q (source proxy)
8. Field/probe (LFP-like)

Backend
-------
* ``backend="static"``  — matplotlib (default, Agg-safe, no GUI)
* ``backend="plotly"``  — interactive Plotly (requires ``pip install jaxfne[viz]``)
* ``backend="both"``    — return dict with ``*_static`` + ``*_plotly`` keys

Large-network handling
----------------------
All panels downsample deterministically for N > thresholds (no OOM, no
dense NxN allocation). See ``LARGE_N_*`` constants below.

Zero-overhead guarantee
-----------------------
This module is *not* imported by ``jaxfne.core`` or ``jaxfne._model``.
It is loaded lazily on first ``jtfne.visualize`` / ``jtfne.vis.visualize``
attribute access (see :class:`jaxfne._RuntimeModuleWrapper`). A fresh
``import jaxfne.core`` must not attach matplotlib/plotly — covered by
``test_simulation_engine_has_zero_graphics_overhead``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

# Large-network thresholds — deterministic, artifact-visible in receipt.
LARGE_N_RASTER = 2000
LARGE_N_CONNECTIVITY = 600
LARGE_N_GEOMETRY = 5000
MAX_RASTER_POINTS = 200_000
MAX_CONNECTIVITY_EDGES = 8000
MAX_TRACE_NEURONS = 8
DEFAULT_SAMPLE_SEED = 0

PANEL_KEYS: tuple[str, ...] = (
    "01_network_structure",
    "02_connectivity",
    "03_parameter_summary",
    "04_raster",
    "05_population_rates",
    "06_state_traces",
    "07_source",
    "08_field_probe",
)


# ── small helpers (no graphics import) ─────────────────────────────────────

def _to_numpy(arr: Any) -> Any:
    """Device-to-host safe conversion without importing matplotlib/plotly."""
    if arr is None:
        return None
    try:
        import jax

        if hasattr(arr, "device") or hasattr(arr, "device_buffer"):
            return __import__("numpy").asarray(jax.device_get(arr))
    except Exception:
        pass
    try:
        import numpy as _np

        return _np.asarray(arr)
    except Exception:
        return arr


def _get_time_ms(signals: Any) -> Any:
    raw = getattr(signals, "time_ms", None)
    if raw is None and isinstance(signals, dict):
        raw = signals.get("time_ms")
    arr = _to_numpy(raw)
    if arr is None:
        return None
    return arr


def _get_spikes(signals: Any) -> Any:
    raw = getattr(signals, "spikes", None)
    if raw is None and isinstance(signals, dict):
        raw = signals.get("spikes")
    return _to_numpy(raw)


def _get_sources(signals: Any) -> Any:
    raw = getattr(signals, "sources", None)
    if raw is None and isinstance(signals, dict):
        raw = signals.get("sources")
    return _to_numpy(raw)


def _get_vm(signals: Any) -> Any:
    raw = getattr(signals, "V_m", None)
    if raw is None and isinstance(signals, dict):
        raw = signals.get("V_m")
    return _to_numpy(raw)


def _get_field(signals: Any) -> Any:
    field = getattr(signals, "field", None)
    if field is None and isinstance(signals, dict):
        field = signals.get("field")
    return field


def _neuron_rows(model: Any, signals: Any) -> list[dict[str, Any]]:
    """Resolve neuron metadata rows, preferring model.neuron_table, then signals.metadata."""
    # model path
    if model is not None and hasattr(model, "neuron_table"):
        try:
            rows = model.neuron_table()
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass
    # also handle model.static["neuron_metadata"]
    if model is not None and hasattr(model, "static"):
        try:
            rows = (model.static or {}).get("neuron_metadata")
            if rows:
                return [dict(r) for r in rows]
        except Exception:
            pass
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    if isinstance(meta, dict):
        rows = meta.get("neuron_metadata")
        if rows:
            return [dict(r) for r in rows]
    return []


def _edge_info(model: Any) -> Optional[dict[str, Any]]:
    """Return edge_list info if present, else None."""
    if model is None or not hasattr(model, "params"):
        return None
    try:
        edge = model.params.get("edge_list") if isinstance(model.params, dict) else getattr(model.params, "edge_list", None)
    except Exception:
        edge = None
    if edge is None:
        return None
    try:

        pre = _to_numpy(edge.pre)
        post = _to_numpy(edge.post)
        w = _to_numpy(edge.weight)
        tau = _to_numpy(edge.tau_ms) if hasattr(edge, "tau_ms") else None
        delay = _to_numpy(edge.delay_steps) if hasattr(edge, "delay_steps") else None
        return {"pre": pre, "post": post, "weight": w, "tau_ms": tau, "delay_steps": delay, "n_edges": int(pre.shape[0]) if pre is not None else 0}
    except Exception:
        return None


def _dense_weight_matrix(model: Any) -> Any:
    if model is None or not hasattr(model, "params"):
        return None
    try:
        params = model.params
        emitter = None
        if isinstance(params, dict):
            emitter = params.get("emitter")
        else:
            emitter = getattr(params, "emitter", None)
        if emitter is None:
            return None
        W = getattr(emitter, "W", None)
        if W is None:
            W = getattr(emitter, "G", None)
        arr = _to_numpy(W)
        if arr is None or arr.size == 0 or arr.shape == (0, 0):
            return None
        return arr
    except Exception:
        return None


def _hdp_diagnostics(model: Any, signals: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # model.last_hdp_diagnostics()
    if model is not None and hasattr(model, "last_hdp_diagnostics"):
        try:
            d = model.last_hdp_diagnostics()
            if isinstance(d, dict):
                out.update(d)
        except Exception:
            pass
        # also direct attribute
        try:
            d2 = getattr(model, "_last_hdp_diag", None)
            if isinstance(d2, dict):
                for k, v in d2.items():
                    out.setdefault(k, v)
        except Exception:
            pass
    # signals.metadata["hdp"]
    meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
    if isinstance(meta, dict) and "hdp" in meta and isinstance(meta["hdp"], dict):
        for k, v in meta["hdp"].items():
            out.setdefault(k, _to_numpy(v) if hasattr(v, "shape") else v)
    # convert any JAX arrays
    for k, v in list(out.items()):
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            out[k] = _to_numpy(v)
    return out


# ── per-panel builders (import matplotlib/plotly lazily inside each) ────────

def _panel_network_structure(model: Any, signals: Any, *, backend: str = "static") -> Any:
    if backend == "plotly":
        try:
            # Prefer canonical vis entry point (adds consistent layout override handling).
            from jaxfne.vis.canonical import plot_network_3d

            fig = plot_network_3d(model, signals, backend="plotly")
            return fig
        except Exception:
            # fallback: direct visualize_network_3d
            try:
                from jaxfne.vis.network3d import visualize_network_3d

                # Cap for very large networks — caller-visible sample
                rows = _neuron_rows(model, signals)
                if len(rows) > LARGE_N_GEOMETRY:
                    import numpy as _np

                    rng = _np.random.default_rng(DEFAULT_SAMPLE_SEED)
                    idx = rng.choice(len(rows), size=LARGE_N_GEOMETRY, replace=False)
                    idx = sorted(idx)
                    rows_sub = [rows[i] for i in idx]
                    # build a DataFrame-like list passed to visualize_network_3d
                    fig = visualize_network_3d(rows_sub, show_layers=False)
                    # annotate sampling
                    try:
                        fig.add_annotation(text=f"sampled {LARGE_N_GEOMETRY}/{len(rows)} neurons", showarrow=False)
                    except Exception:
                        pass
                    return fig
                return visualize_network_3d(model if hasattr(model, "neuron_table") else signals)
            except Exception as exc:
                from jaxfne.vis.core import require_matplotlib

                require_matplotlib()
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, f"Network structure (plotly unavailable):\n{exc}", ha="center", va="center")
                ax.set_title("Network structure — fallback")
                return fig
    # static
    from jaxfne.vis.core import require_matplotlib

    require_matplotlib()
    import matplotlib.pyplot as plt

    rows = _neuron_rows(model, signals)
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "neuron_metadata not available\n(declared geometry proxy requires model with positions)", ha="center", va="center")
        ax.set_title("Network structure — no geometry metadata")
        return fig
    # Delegate to existing geometry3d (handles subsampling internally) but ensure cap
    try:
        from jaxfne.vis.network3d import geometry3d

        # geometry3d can take either signals or model; pass whichever has rows
        obj = signals if (getattr(signals, "metadata", {}).get("neuron_metadata") if isinstance(getattr(signals, "metadata", None), dict) else False) else model
        # large-N cap: pass a truncated row list via transient object
        if len(rows) > LARGE_N_GEOMETRY:
            # Build a lightweight signals-like holder with truncated rows
            import types

            import numpy as _np

            rng = _np.random.default_rng(DEFAULT_SAMPLE_SEED)
            idx = sorted(rng.choice(len(rows), size=LARGE_N_GEOMETRY, replace=False).tolist())
            rows_sub = [rows[i] for i in idx]
            holder = types.SimpleNamespace(metadata={"neuron_metadata": rows_sub})
            fig = geometry3d(holder)
            # add small annotation about sampling
            try:
                fig.text(0.5, 0.01, f"sampled {LARGE_N_GEOMETRY}/{len(rows)} neurons for display", ha="center", fontsize=8, color="gray")
            except Exception:
                pass
            return fig
        return geometry3d(obj)
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"geometry3d failed:\n{exc}", ha="center", va="center", fontsize=9)
        ax.set_title("Network structure — error fallback")
        return fig


def _panel_connectivity(model: Any, signals: Any, *, backend: str = "static") -> Any:
    if backend == "plotly":
        try:
            from jaxfne.vis.canonical import plot_connectivity

            # cap edges for plotly: sampling inside plot_connectivity via max_neurons
            n_neurons = 0
            try:
                n_neurons = len(_neuron_rows(model, signals)) or 0
            except Exception:
                pass
            if n_neurons > LARGE_N_CONNECTIVITY:
                return plot_connectivity(model, backend="plotly", max_neurons=LARGE_N_CONNECTIVITY)
            return plot_connectivity(model, backend="plotly")
        except Exception as exc:
            from jaxfne.vis.core import require_matplotlib

            require_matplotlib()
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Plotly connectivity unavailable:\n{exc}", ha="center", va="center")
            return fig
    # static — handle dense vs sparse without O(N^2) alloc
    from jaxfne.vis.core import require_matplotlib

    require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as _np

    edge = _edge_info(model)
    W = _dense_weight_matrix(model)

    n_neurons = 0
    try:
        rows = _neuron_rows(model, signals)
        n_neurons = len(rows)
        if n_neurons == 0 and W is not None:
            n_neurons = int(W.shape[0])
        elif n_neurons == 0 and edge is not None:
            # infer from max index
            m = int(max(int(_np.max(edge["pre"])), int(_np.max(edge["post"]))) + 1) if edge["n_edges"] > 0 else 0
            n_neurons = m
    except Exception:
        pass

    # Choose rendering path
    if edge is not None and edge["n_edges"] > 0:
        n_edges = edge["n_edges"]
        # Large-edge cap — deterministic subsample
        if n_edges > MAX_CONNECTIVITY_EDGES:
            rng = _np.random.default_rng(DEFAULT_SAMPLE_SEED)
            sel = rng.choice(n_edges, size=MAX_CONNECTIVITY_EDGES, replace=False)
            sel = _np.sort(sel)
            pre = edge["pre"][sel]
            post = edge["post"][sel]
            w = edge["weight"][sel]
            tau = edge["tau_ms"][sel] if edge["tau_ms"] is not None else None
            delay = edge["delay_steps"][sel] if edge["delay_steps"] is not None else None
            subtitle = f"sparse scatter — sampled {MAX_CONNECTIVITY_EDGES}/{n_edges} edges (N={n_neurons})"
        else:
            pre, post, w = edge["pre"], edge["post"], edge["weight"]
            tau, delay = edge["tau_ms"], edge["delay_steps"]
            subtitle = f"sparse scatter — {n_edges} edges (N={n_neurons})"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
        ax0, ax1 = axes
        sc = ax0.scatter(pre, post, c=w, s=6, cmap="RdBu_r", alpha=0.8, edgecolors="none")
        # symmetric v limits around zero
        try:
            vmax = float(_np.nanmax(_np.abs(w))) or 1.0
            sc.set_clim(-vmax, vmax)
        except Exception:
            pass
        ax0.set_xlabel("Pre neuron")
        ax0.set_ylabel("Post neuron")
        ax0.set_title(f"Connectivity (edge-list) — {subtitle}", fontsize=10)
        fig.colorbar(sc, ax=ax0, label="Weight (proxy, native units)")
        ax0.grid(True, alpha=0.2)
        # weight histogram
        ax1.hist(w, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        ax1.set_title("Weight distribution", fontsize=10)
        ax1.set_xlabel("Weight")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.2)
        # annotate tau/delay summaries if present
        extra = []
        if tau is not None:
            try:
                extra.append(f"tau_ms: mean={float(_np.mean(tau)):.2f} [{float(_np.min(tau)):.1f},{float(_np.max(tau)):.1f}]")
            except Exception:
                pass
        if delay is not None:
            try:
                extra.append(f"delay_steps: max={int(_np.max(delay))}  nonzero={int(_np.count_nonzero(delay))}")
            except Exception:
                pass
        if extra:
            ax1.text(0.5, -0.22, "\n".join(extra), transform=ax1.transAxes, ha="center", fontsize=8, color="dimgray")
        fig.tight_layout()
        return fig

    if W is not None:
        n = W.shape[0]
        if n > LARGE_N_CONNECTIVITY:
            # downsample matrix via strided sampling for display
            step = max(1, n // LARGE_N_CONNECTIVITY)
            idx = _np.arange(0, n, step)[:LARGE_N_CONNECTIVITY]
            W_disp = W[_np.ix_(idx, idx)]
            subtitle = f"dense heatmap — downsampled {W_disp.shape[0]}x{W_disp.shape[0]} of {n}x{n} (stride {step})"
        else:
            W_disp = W
            subtitle = f"dense heatmap — {n}x{n}"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
        ax0, ax1 = axes
        vmax = float(_np.nanmax(_np.abs(W_disp))) or 1.0
        im = ax0.imshow(W_disp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="upper")
        ax0.set_title(f"Connectivity — {subtitle}", fontsize=10)
        ax0.set_xlabel("Pre (sampled)" if n > LARGE_N_CONNECTIVITY else "Pre")
        ax0.set_ylabel("Post (sampled)" if n > LARGE_N_CONNECTIVITY else "Post")
        fig.colorbar(im, ax=ax0, label="Weight (proxy)")
        ax1.hist(W_disp.ravel(), bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        ax1.set_title("Weight distribution", fontsize=10)
        ax1.set_xlabel("Weight")
        fig.tight_layout()
        return fig

    # fallback
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "No connectivity payload found\n(edge_list and emitter.W absent)", ha="center", va="center")
    ax.set_title("Connectivity — unavailable")
    return fig


def _panel_parameter_summary(model: Any, signals: Any, *, backend: str = "static") -> Any:
    # summary is intentionally identical across backends — text/table is a static concept;
    # for backend=plotly we return a Plotly Table figure.
    import numpy as _np

    rows = _neuron_rows(model, signals)
    edge = _edge_info(model)
    W = _dense_weight_matrix(model)
    time_ms = _get_time_ms(signals)
    meta_model = {}
    try:
        meta_model = getattr(model, "cfg", None).metadata if model is not None and hasattr(model, "cfg") else {}
        if meta_model is None:
            meta_model = {}
    except Exception:
        meta_model = {}
    try:
        static = getattr(model, "static", {}) or {}
    except Exception:
        static = {}

    # counts
    n_neurons = len(rows) if rows else (int(W.shape[0]) if W is not None else (int(_get_vm(signals).shape[1]) if _get_vm(signals) is not None and _get_vm(signals).ndim > 1 else 0))
    n_steps = int(time_ms.shape[0]) if time_ms is not None else 0
    try:
        dt_ms = float((getattr(signals, "metadata", {}) or {}).get("dt_ms", 0.05)) if not isinstance(signals, dict) else float((signals.get("metadata", {}) or {}).get("dt_ms", 0.05))
    except Exception:
        dt_ms = 0.05
    n_contacts = int(static.get("n_contacts", 16)) if isinstance(static, dict) else 16
    n_edges = int(edge["n_edges"]) if edge is not None else (int(_np.count_nonzero(W)) if W is not None else 0)

    # weight stats
    weight_stats = None
    if edge is not None and edge["weight"] is not None and edge["weight"].size:
        w = edge["weight"]
        weight_stats = {"mean": float(_np.mean(w)), "std": float(_np.std(w)), "min": float(_np.min(w)), "max": float(_np.max(w)), "source": "edge_list"}
    elif W is not None and W.size:
        weight_stats = {"mean": float(_np.mean(W)), "std": float(_np.std(W)), "min": float(_np.min(W)), "max": float(_np.max(W)), "source": "dense W"}

    # delay / tau
    tau_summary = None
    delay_summary = None
    if edge is not None:
        if edge["tau_ms"] is not None and edge["tau_ms"].size:
            tau_summary = f"{float(_np.mean(edge['tau_ms'])):.2f} ms (range {float(_np.min(edge['tau_ms'])):.1f}..{float(_np.max(edge['tau_ms'])):.1f})"
        if edge["delay_steps"] is not None and edge["delay_steps"].size:
            delay_summary = f"max {int(_np.max(edge['delay_steps']))}, nonzero {int(_np.count_nonzero(edge['delay_steps']))}, mean {float(_np.mean(edge['delay_steps'])):.2f} steps"

    # cell-type / layer breakdown
    ct_counts: dict[str, int] = {}
    layer_counts: dict[str, int] = {}
    for r in rows:
        ct = str(r.get("cell_type", "unknown"))
        layer = str(r.get("layer", "unspecified"))
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    # state ownership
    param_keys = []
    static_keys = []
    try:
        if model is not None and hasattr(model, "params") and isinstance(model.params, dict):
            param_keys = sorted(model.params.keys())
        if isinstance(static, dict):
            static_keys = sorted(static.keys())
    except Exception:
        pass

    # calibration / truth
    src_calib = None
    claim_level = None
    try:
        if isinstance(meta_model, dict):
            src_calib = meta_model.get("source_calibration_status")
            claim_level = meta_model.get("claim_level")
    except Exception:
        pass
    try:
        sig_meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
        if isinstance(sig_meta, dict):
            src_calib = sig_meta.get("source_calibration_status") or src_calib
    except Exception:
        pass

    if backend == "plotly":
        try:
            import plotly.graph_objects as go

            header = ["Field", "Value"]
            cells_a: list[str] = []
            cells_b: list[str] = []
            def add(k: str, v: str) -> None:
                cells_a.append(k)
                cells_b.append(v)
            add("n_neurons", str(n_neurons))
            add("n_steps / dt_ms / duration", f"{n_steps} / {dt_ms} ms / {n_steps*dt_ms:.1f} ms")
            add("n_contacts", str(n_contacts))
            add("n_edges", str(n_edges))
            add("weight", f"{weight_stats}" if weight_stats else "—")
            add("tau_ms", tau_summary or "—")
            add("delay_steps", delay_summary or "—")
            add("cell_types", ", ".join(f"{k}:{v}" for k, v in sorted(ct_counts.items())) or "—")
            add("layers", ", ".join(f"{k}:{v}" for k, v in sorted(layer_counts.items())) or "—")
            add("params keys", ", ".join(param_keys) or "—")
            add("static keys", ", ".join(static_keys) or "—")
            add("source_calibration", str(src_calib) if src_calib else "—")
            add("claim_level", str(claim_level) if claim_level else "—")
            add("field", "proxy_readout (uncalibrated)")
            fig = go.Figure(data=[go.Table(header=dict(values=header, fill_color="black", font=dict(color="white")), cells=dict(values=[cells_a, cells_b], fill_color="lavender"))])
            fig.update_layout(title="Parameter summary — state ownership & calibration (proxy, no physical claim)")
            return fig
        except Exception:
            backend = "static"  # fall through to matplotlib path

    # static matplotlib path
    from jaxfne.vis.core import require_matplotlib

    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], hspace=0.35, wspace=0.25)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.axis("off")
    title = "Parameter summary — weights / delays / state ownership (proxy, no physical claim)"
    ax_top.set_title(title, fontsize=11, fontweight="bold", pad=12)
    # Build text block
    lines: list[str] = []
    lines.append(f"Network:  N={n_neurons}  |  edges={n_edges}  |  steps={n_steps}  dt={dt_ms} ms  |  contacts={n_contacts}")
    if weight_stats:
        lines.append(f"Weights ({weight_stats['source']}):  mean {weight_stats['mean']:+.3f}  std {weight_stats['std']:.3f}  range [{weight_stats['min']:+.2f}, {weight_stats['max']:+.2f}]")
    else:
        lines.append("Weights: — (no payload)")
    if tau_summary:
        lines.append(f"Tau: {tau_summary}")
    if delay_summary:
        lines.append(f"Delays: {delay_summary}")
    else:
        lines.append("Delays: all zero / not declared")
    if ct_counts:
        lines.append("Cell types: " + ", ".join(f"{k}={v}" for k, v in sorted(ct_counts.items())))
    if layer_counts:
        lines.append("Layers: " + ", ".join(f"{k}={v}" for k, v in sorted(layer_counts.items())))
    lines.append(f"State ownership — params: {', '.join(param_keys) or '—'}")
    lines.append(f"State ownership — static: {', '.join(static_keys) or '—'}")
    calib_line = f"Calibration: source={src_calib or 'uncalibrated_izhikevich_native_current'}  claim_level={claim_level or 'computational_scaffold'}  field=proxy_readout"
    lines.append(calib_line)
    lines.append("Note: all values are relative proxy units; no physical-amplitude claim.")
    ax_top.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=9, family="monospace", transform=ax_top.transAxes)

    # Bottom panels — weight hist + layer/cell-type bar (if available)
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])
    if weight_stats is not None:
        if edge is not None and edge["weight"].size:
            w = edge["weight"]
        else:
            w = W.ravel() if W is not None else _np.array([])
        if w.size:
            ax_bl.hist(w, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
            ax_bl.set_title("Weight distribution", fontsize=10)
            ax_bl.set_xlabel("Weight (proxy)")
            ax_bl.grid(True, alpha=0.2)
        else:
            ax_bl.text(0.5, 0.5, "no weights", ha="center", va="center")
    else:
        ax_bl.text(0.5, 0.5, "no weight payload", ha="center", va="center")
        ax_bl.set_title("Weight distribution")
    if ct_counts:
        keys = sorted(ct_counts)
        vals = [ct_counts[k] for k in keys]
        ax_br.bar(keys, vals, color="darkorange", alpha=0.85)
        ax_br.set_title("Neurons per cell type", fontsize=10)
        ax_br.set_ylabel("Count")
        ax_br.tick_params(axis="x", rotation=15)
        for i, v in enumerate(vals):
            ax_br.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    elif layer_counts:
        keys = sorted(layer_counts)
        vals = [layer_counts[k] for k in keys]
        ax_br.bar(keys, vals, color="seagreen", alpha=0.85)
        ax_br.set_title("Neurons per layer", fontsize=10)
    else:
        ax_br.text(0.5, 0.5, "no layer/cell-type breakdown", ha="center", va="center")
        ax_br.set_title("Composition")
    fig.tight_layout()
    return fig


def _panel_raster(model: Any, signals: Any, *, backend: str = "static") -> Any:
    if backend == "plotly":
        try:
            from jaxfne.vis.canonical import plot_raster

            return plot_raster(signals, model, backend="plotly")
        except Exception:
            backend = "static"
    # static
    from jaxfne.vis.core import require_matplotlib

    require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as _np

    spikes = _get_spikes(signals)
    time_ms = _get_time_ms(signals)
    if spikes is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No spikes in signals", ha="center", va="center")
        ax.set_title("Raster — no data")
        return fig
    # Large handling — random cap on rendered points (spike dots), deterministic
    # Keep the underlying rate truth intact; only rendering is subsampled.
    if spikes.size > 0:
        n_neurons = spikes.shape[1] if spikes.ndim > 1 else 1
        total_dots = int(_np.count_nonzero(spikes))
        if total_dots > MAX_RASTER_POINTS or n_neurons > LARGE_N_RASTER:
            t_idx, n_idx = _np.where(spikes > 0)
            if len(t_idx) > MAX_RASTER_POINTS:
                rng = _np.random.default_rng(DEFAULT_SAMPLE_SEED)
                sel = rng.choice(len(t_idx), size=MAX_RASTER_POINTS, replace=False)
                sel = _np.sort(sel)
                t_idx, n_idx = t_idx[sel], n_idx[sel]
                note = f"sampled {MAX_RASTER_POINTS}/{total_dots} spikes for display (N={n_neurons})"
            else:
                note = f"N={n_neurons} (large), all {total_dots} spikes shown" if total_dots else ""
            # synthesize a lightweight signals holder that raster() understands — reuse raw arrays
            # Build a minimal holder and call raster with subsampled spikes matrix reconstructed as
            # sparse scatter here rather than reconstructing dense matrix.
            fig, ax = plt.subplots(figsize=(10, 4))
            if time_ms is not None and len(time_ms) == spikes.shape[0]:
                x = time_ms[t_idx]
            else:
                x = t_idx
            rows = _neuron_rows(model, signals)
            if len(rows) == n_neurons:
                order = _np.argsort([float(r.get("z", i)) for i, r in enumerate(rows)])
                rank = _np.empty_like(order)
                rank[order] = _np.arange(order.shape[0])
                y = rank[n_idx]
                ylabel = "Neuron rank (by z)"
            else:
                y = n_idx
                ylabel = "Neuron index"
            ax.scatter(x, y, s=2, marker="|", alpha=0.6)
            ax.set_title(f"Spike raster (proxy) — {note}" if note else "Spike raster (proxy)")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.2)
            return fig
    # small — use existing vis
    try:
        from jaxfne.vis.rasters import raster

        return raster(signals)
    except Exception as exc:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"raster failed: {exc}", ha="center", va="center")
        return fig


def _panel_population_rates(model: Any, signals: Any, *, backend: str = "static") -> Any:
    if backend == "plotly":
        try:
            from jaxfne.vis.canonical import plot_population_rate

            return plot_population_rate(signals, model, backend="plotly")
        except Exception:
            backend = "static"
    from jaxfne.vis.core import require_matplotlib

    require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as _np

    spikes = _get_spikes(signals)
    time_ms = _get_time_ms(signals)
    if spikes is None or time_ms is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "spikes/time_ms not available", ha="center", va="center")
        ax.set_title("Population rates — no data")
        return fig
    try:
        meta = getattr(signals, "metadata", {}) if not isinstance(signals, dict) else signals.get("metadata", {})
        dt_ms = float(meta.get("dt_ms", 0.05)) if isinstance(meta, dict) else 0.05
    except Exception:
        dt_ms = 0.05
    # Try per-cell-type rates if small and metadata present
    rows = _neuron_rows(model, signals)
    if rows and len(rows) == spikes.shape[1] and len(rows) <= LARGE_N_RASTER:
        # Use vis.traces.rate for the mean (handles binning), then overlay per-type if needed
        try:
            from jaxfne.vis.traces import rate

            fig = rate(signals)
            ax = fig.axes[0] if fig.axes else None
            if ax is not None and len(rows) <= 800:
                ct_vals = [str(r.get("cell_type", "unknown")) for r in rows]
                uniq = sorted(set(ct_vals))
                for ct in uniq:
                    mask = _np.asarray([v == ct for v in ct_vals])
                    if not _np.any(mask):
                        continue
                    r = _np.mean(spikes[:, mask], axis=1) * (1000.0 / dt_ms)
                    ax.plot(time_ms, r, label=ct, alpha=0.7, linewidth=1.0)
                ax.legend(fontsize=8, title="Cell type")
                ax.set_title("Population rates — mean + per-type (proxy Hz)")
            return fig
        except Exception:
            pass
    # fallback — mean rate only
    try:
        from jaxfne.vis.traces import rate

        return rate(signals)
    except Exception:
        fig, ax = plt.subplots()
        mean_rate = _np.mean(spikes, axis=1) * (1000.0 / dt_ms) if spikes.ndim > 1 else spikes * (1000.0 / dt_ms)
        ax.plot(time_ms, mean_rate, color="#c0392b")
        ax.set_title("Population mean rate (Hz, proxy)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Hz")
        return fig


def _panel_state_traces(model: Any, signals: Any, *, backend: str = "static") -> Any:
    # State traces panel is static-first; plotly path reuses the same matplotlib grid
    # exported as Plotly via mpl-to-plotly would be opaque — instead we produce
    # plotly traces explicitly when backend==plotly.
    diag = _hdp_diagnostics(model, signals)
    has_H = any(k in diag for k in ("H_trace", "H_final", "H"))
    has_W = any(k in diag for k in ("w_trace", "w_final", "G_trace"))
    Vm = _get_vm(signals)
    time_ms = _get_time_ms(signals)

    if backend == "plotly":
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            rows_plot = 3 if (has_H or has_W) else 1
            fig = make_subplots(rows=rows_plot, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("V_m (proxy mV)", "H (control state, proxy)" if has_H else "H — not enabled", "W (edge weights, proxy)" if has_W else "W — not enabled"))
            if Vm is not None and time_ms is not None:
                import numpy as _np
                t = _np.asarray(time_ms)
                cols = min(MAX_TRACE_NEURONS, Vm.shape[1] if Vm.ndim > 1 else 1)
                idx = _np.linspace(0, Vm.shape[1] - 1, cols, dtype=int) if Vm.ndim > 1 else [0]
                for j, c in enumerate(idx):
                    y = Vm[:, c] if Vm.ndim > 1 else Vm
                    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"V[{c}]"), row=1, col=1)
            # H trace
            if has_H and rows_plot >= 2:
                import numpy as _np
                Ht = diag.get("H_trace")
                if Ht is None:
                    Ht = diag.get("H")
                if Ht is not None:
                    Ht = _to_numpy(Ht)
                    if Ht.ndim == 2:
                        # (steps, neurons) -> mean
                        fig.add_trace(go.Scatter(x=_np.asarray(time_ms)[:Ht.shape[0]], y=_np.mean(Ht, axis=1), mode="lines", name="H mean"), row=2, col=1)
                    elif Ht.ndim == 3:
                        fig.add_trace(go.Scatter(x=_np.asarray(time_ms)[:Ht.shape[0]], y=_np.mean(Ht, axis=(1,2)), mode="lines", name="H mean"), row=2, col=1)
            # W trace — histogram of final weights
            if has_W and rows_plot >= 2:
                w = diag.get("w_final")
                if w is None:
                    w = diag.get("W")
                if w is None:
                    w = diag.get("G_trace")
                if w is not None:
                    import numpy as _np
                    wf = _np.asarray(_to_numpy(w)).ravel()
                    fig.add_trace(go.Histogram(x=wf, name="W final"), row=rows_plot, col=1)
            fig.update_layout(title="State traces — V, H, W (proxy, uncalibrated)", height=260 * rows_plot)
            fig.update_xaxes(title_text="Time (ms)", row=rows_plot, col=1)
            return fig
        except Exception:
            backend = "static"

    # static matplotlib
    from jaxfne.vis.core import require_matplotlib

    require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as _np

    # Decide grid
    show_H = has_H
    show_W = has_W
    n_rows = 1 + int(show_H) + int(show_W)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.0 * n_rows), sharex=True, squeeze=False)
    axes = axes[:, 0].tolist()

    # Row 0 — V traces
    ax = axes[0]
    if Vm is not None and time_ms is not None:
        if Vm.ndim == 1:
            ax.plot(_np.asarray(time_ms), _np.asarray(Vm), linewidth=0.8)
        else:
            cols = min(MAX_TRACE_NEURONS, Vm.shape[1])
            idx = _np.linspace(0, Vm.shape[1] - 1, cols, dtype=int)
            for j, c in enumerate(idx):
                ax.plot(_np.asarray(time_ms), Vm[:, c], label=f"n{c}", alpha=0.85, linewidth=0.8)
            if cols <= 8:
                ax.legend(fontsize=7, ncol=4)
            else:
                ax.text(0.02, 0.95, f"showing {cols}/{Vm.shape[1]} neurons (sampled)", transform=ax.transAxes, fontsize=8, va="top", color="dimgray")
        ax.set_ylabel("V_m (proxy mV)")
        ax.grid(True, alpha=0.2)
        ax.set_title("State traces — V / H / W (proxy, uncalibrated)", fontsize=11)
    else:
        ax.text(0.5, 0.5, "V_m not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("State traces — V / H / W (proxy)")

    # Row 1 — H
    if show_H:
        ax_h = axes[1]
        Ht = diag.get("H_trace")
        if Ht is None:
            Ht = diag.get("H")
        if Ht is None:
            Ht = diag.get("H_final")
        if Ht is not None:
            Ht = _to_numpy(Ht)
            t = _np.asarray(time_ms)[:Ht.shape[0]] if Ht.ndim >= 1 else _np.asarray(time_ms)
            try:
                if Ht.ndim == 1:
                    ax_h.plot(t, Ht, color="teal")
                elif Ht.ndim == 2:
                    # (steps, neurons) -> mean + std band
                    mean = _np.mean(Ht, axis=1)
                    std = _np.std(Ht, axis=1)
                    ax_h.plot(t, mean, color="teal", label="H mean")
                    ax_h.fill_between(t, mean - std, mean + std, color="teal", alpha=0.18)
                    ax_h.legend(fontsize=8)
                elif Ht.ndim == 3:
                    mean = _np.mean(Ht, axis=(1, 2))
                    ax_h.plot(t, mean, color="teal", label="H mean (pop)")
                    ax_h.legend(fontsize=8)
                ax_h.set_ylabel("H (proxy)")
                ax_h.grid(True, alpha=0.2)
            except Exception as exc:
                ax_h.text(0.5, 0.5, f"H plot failed: {exc}", ha="center", va="center", transform=ax_h.transAxes)
        else:
            ax_h.text(0.5, 0.5, "H_trace not available (enable_hdp=True to record)", ha="center", va="center", transform=ax_h.transAxes)
            ax_h.set_ylabel("H (proxy)")

    # Row 2 — W
    if show_W:
        ax_w = axes[-1]
        w_final = diag.get("w_final")
        w_trace = diag.get("w_trace")
        # Prefer trace mean evolution if available; otherwise histogram of final
        if w_trace is not None:
            wt = _to_numpy(w_trace)
            try:
                if wt.ndim == 2:
                    # (steps, edges)
                    mean = _np.mean(wt, axis=1)
                    ax_w.plot(_np.asarray(time_ms)[:mean.shape[0]], mean, color="purple")
                    ax_w.set_ylabel("W mean (proxy)")
                    ax_w.grid(True, alpha=0.2)
                    ax_w.set_title("W trace — mean weight evolution", fontsize=9)
                else:
                    ax_w.hist(_np.asarray(wt).ravel(), bins=40, color="purple", alpha=0.8, edgecolor="white")
                    ax_w.set_title("W — distribution (trace)", fontsize=9)
            except Exception as exc:
                ax_w.text(0.5, 0.5, f"W trace plot failed: {exc}", ha="center", va="center", transform=ax_w.transAxes)
        elif w_final is not None:
            wf = _np.asarray(_to_numpy(w_final)).ravel()
            ax_w.hist(wf, bins=40, color="purple", alpha=0.8, edgecolor="white")
            ax_w.set_title("W — final weight distribution", fontsize=9)
            ax_w.set_xlabel("Weight (proxy)")
            ax_w.grid(True, alpha=0.2)
        else:
            ax_w.text(0.5, 0.5, "W trace/final not available (enable_hdp/record_weight_trace)", ha="center", va="center", transform=ax_w.transAxes)
            ax_w.set_ylabel("W (proxy)")
    # x label
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    return fig


def _panel_source(model: Any, signals: Any, *, backend: str = "static") -> Any:
    if backend == "plotly":
        try:
            # Plotly path: build time series traces explicitly
            import plotly.graph_objects as go
            import numpy as _np

            src = _get_sources(signals)
            t = _to_numpy(_get_time_ms(signals))
            if src is None or t is None:
                fig = go.Figure()
                fig.add_annotation(text="sources not recorded (record_sources=True)", showarrow=False)
                return fig
            # large-N: plot mean +/- std rather than per-neuron spaghetti
            if src.ndim > 1 and src.shape[1] > MAX_TRACE_NEURONS:
                mean = _np.mean(src, axis=1)
                std = _np.std(src, axis=1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=mean, mode="lines", name="mean source"))
                fig.add_trace(go.Scatter(x=t, y=mean + std, mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=t, y=mean - std, mode="lines", fill="tonexty", line=dict(width=0), name="±1 std"))
                fig.update_layout(title=f"Source proxy — mean ± std (N={src.shape[1]}, sampled band)", xaxis_title="Time (ms)", yaxis_title="Source (proxy)")
                return fig
            # small: per-neuron
            fig = go.Figure()
            cols = min(src.shape[1] if src.ndim > 1 else 1, MAX_TRACE_NEURONS)
            for i in range(cols):
                y = src[:, i] if src.ndim > 1 else src
                fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=f"src[{i}]"))
            fig.update_layout(title="Source proxy Q(t) — per-neuron (proxy, uncalibrated)", xaxis_title="Time (ms)", yaxis_title="Source (proxy)")
            return fig
        except Exception:
            backend = "static"
    # static
    from jaxfne.vis.core import require_matplotlib

    require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as _np

    src = _get_sources(signals)
    t = _to_numpy(_get_time_ms(signals))
    if src is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "sources not recorded\n(run with record_sources=True)", ha="center", va="center")
        ax.set_title("Source Q — unavailable")
        return fig
    if t is None:
        t = _np.arange(src.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax_top, ax_bot = axes
    # Top — traces (mean +/- std for large, sampled for small large)
    if src.ndim > 1 and src.shape[1] > MAX_TRACE_NEURONS:
        if src.shape[1] > 64:
            mean = _np.mean(src, axis=1)
            std = _np.std(src, axis=1)
            ax_top.plot(t, mean, color="darkgreen", linewidth=1.0, label="mean")
            ax_top.fill_between(t, mean - std, mean + std, color="darkgreen", alpha=0.18, label="±1 std")
            ax_top.legend(fontsize=8)
            ax_top.set_title(f"Source Q(t) — mean ± std (N={src.shape[1]}, proxy, uncalibrated)", fontsize=11)
        else:
            cols = min(src.shape[1], MAX_TRACE_NEURONS)
            idx = _np.linspace(0, src.shape[1] - 1, cols, dtype=int)
            for c in idx:
                ax_top.plot(t, src[:, c], alpha=0.75, linewidth=0.8, label=f"n{c}")
            ax_top.legend(fontsize=7, ncol=4)
            ax_top.set_title(f"Source Q(t) — {cols}/{src.shape[1]} neurons sampled (proxy)", fontsize=11)
    else:
        if src.ndim == 1:
            ax_top.plot(t, src, color="darkgreen", linewidth=1.0)
        else:
            cols = min(src.shape[1], MAX_TRACE_NEURONS)
            idx = _np.linspace(0, src.shape[1] - 1, cols, dtype=int)
            for c in idx:
                ax_top.plot(t, src[:, c], alpha=0.8, linewidth=0.8, label=f"n{c}")
            if cols > 1:
                ax_top.legend(fontsize=7, ncol=4)
        ax_top.set_title("Source Q(t) — per-neuron traces (proxy, uncalibrated)", fontsize=11)
    ax_top.set_ylabel("Source (proxy)")
    ax_top.grid(True, alpha=0.2)
    # Bottom — absolute-mean evolution as Q summary (same semantics as _compute_all_metrics)
    try:
        q_abs_mean = _np.mean(_np.abs(src), axis=1) if src.ndim > 1 else _np.abs(src)
        ax_bot.plot(t, q_abs_mean, color="teal", linewidth=1.0)
        ax_bot.set_title("Source |Q| — absolute-mean (proxy)", fontsize=9)
        ax_bot.set_ylabel("|Q|")
        ax_bot.grid(True, alpha=0.2)
    except Exception:
        pass
    ax_bot.set_xlabel("Time (ms)")
    fig.tight_layout()
    return fig


def _panel_field_probe(model: Any, signals: Any, *, backend: str = "static") -> Any:
    field = _get_field(signals)
    t = _to_numpy(_get_time_ms(signals))
    if field is None:
        # Try to provide a helpful static placeholder
        if backend == "plotly":
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_annotation(text="field not recorded (record_fields=True) — no LFP/CSD proxy available<br>Probe readout requires a laminar source projection", showarrow=False)
                fig.update_layout(title="Field / probe (LFP-like) — unavailable")
                return fig
            except Exception:
                pass
        from jaxfne.vis.core import require_matplotlib

        require_matplotlib()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "field not recorded (record_fields=True)\nNo LFP/CSD proxy available.\nProbe readout requires laminar source projection.", ha="center", va="center", fontsize=11)
        ax.set_title("Field / probe (LFP-like) — unavailable (proxy)")
        return fig
    if backend == "plotly":
        try:
            from jaxfne.vis.canonical import plot_lfp

            return plot_lfp(signals, backend="plotly")
        except Exception:
            # Fall back to matplotlib -> still return something
            backend = "static"
    # static — delegate to existing vis, but also handle stacking for readability
    try:
        from jaxfne.vis.traces import lfp as _lfp_static

        fig = _lfp_static(signals)
        # Upgrade title to make proxy claim explicit if not already
        try:
            ax = fig.axes[0] if fig.axes else None
            if ax is not None and "proxy" not in ax.get_title().lower():
                ax.set_title(ax.get_title() + " (proxy)")
        except Exception:
            pass
        return fig
    except Exception:
        from jaxfne.vis.core import require_matplotlib

        require_matplotlib()
        import matplotlib.pyplot as plt
        import numpy as _np

        # manual fallback: heatmap of lfp_proxy
        try:
            lfp_arr = _to_numpy(field.lfp_proxy) if hasattr(field, "lfp_proxy") else _to_numpy(field.get("lfp_proxy") if isinstance(field, dict) else None)
            if lfp_arr is None:
                raise ValueError("no lfp_proxy")
            fig, ax = plt.subplots(figsize=(10, 4))
            if t is None:
                t = _np.arange(lfp_arr.shape[0])
            im = ax.imshow(lfp_arr.T, cmap="viridis", aspect="auto", origin="upper", extent=[float(t[0]), float(t[-1]), lfp_arr.shape[1], 0])
            ax.set_title("Field / probe — LFP proxy heatmap (proxy, uncalibrated)", fontsize=11)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Contact (depth)")
            fig.colorbar(im, ax=ax, label="LFP proxy")
            return fig
        except Exception as exc:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"field panel failed: {exc}", ha="center", va="center")
            return fig


# ── public bundle ──────────────────────────────────────────────────────────

def visualize(
    model: Any,
    signals: Any,
    *,
    backend: str = "static",
    save_dir: Optional[str] = None,
    formats: Sequence[str] = ("png",),
    dpi: int = 150,
    show: bool = False,
) -> Any:
    """Build the standard 8-panel post-hoc visualization bundle.

    Parameters
    ----------
    model : Model | None
        The :class:`jaxfne.Model` used to produce ``signals``. May be
        ``None`` when only ``signals``-only panels (raster/rates/…) are
        desired — panels 1–3 will render as "unavailable" placeholders.
    signals : Signals
        The :class:`jaxfne.Signals` returned by :func:`jaxfne.simulate`
        (or ``model.simulate``). Must carry ``time_ms`` and ``spikes`` at
        minimum; ``sources`` and ``field`` are optional (their panels fall
        back to placeholders).
    backend : {"static","plotly","both"}
        Rendering backend. ``static`` (default) returns matplotlib
        ``Figure`` objects (``Agg``-safe). ``plotly`` returns
        ``plotly.graph_objects.Figure``. ``both`` returns a dict with both
        variants keyed ``"<panel>_static"`` / ``"<panel>_plotly"``.
    save_dir : str | None
        If provided, figures are written to this directory (and the manifest
        is returned alongside). ``formats`` controls file extensions.
    formats : Sequence[str]
        Export formats for ``save_dir`` (e.g. ``("png","pdf","html")``
        — ``html`` is only written for plotly figures).
    dpi : int
        Matplotlib export DPI.
    show : bool
        If ``True``, call ``plt.show()`` / ``fig.show()`` for interactive
        use. Default ``False`` (headless-safe).

    Returns
    -------
    FigureBundle | dict
        When ``save_dir`` is ``None``, a :class:`jaxfne.vis.exporters.FigureBundle`
        whose keys are the 8 panel names (or 16 keys when ``backend="both"``).
        When ``save_dir`` is set, the bundle is still returned and files are
        written as a side-effect; an additional ``.manifest`` attribute
        (or second return element) is not needed — inspect ``bundle.figures``.

    Notes
    -----
    * This function adds *no* simulation numerics. It is safe to call many
      times on the same ``(model, signals)`` without re-running the kernel.
    * Large networks are handled by deterministic downsampling (seed 0) —
      panels never allocate ``N×N`` dense matrices for ``N > 600``.
    * All panels that show physical-like axes are explicitly labelled
      "proxy" / "uncalibrated" — no physical-amplitude claim.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> model = jtfne.construct(jtfne.configuration(n_neurons=80))
    >>> signals = jtfne.simulate(model, jtfne.simulation(duration_ms=200, dt_ms=0.5))
    >>> bundle = jtfne.visualize(model, signals)            # 8 static figs
    >>> bundle.export("artifacts/viz_run0")                 # write pngs
    >>> bundle_plotly = jtfne.visualize(model, signals, backend="plotly")
    >>> bundle_both = jtfne.visualize(model, signals, backend="both")
    """
    if backend not in ("static", "plotly", "both"):
        raise ValueError(f"backend must be 'static', 'plotly', or 'both', got {backend!r}")

    # Build dispatch
    if backend == "both":
        static_figs = _visualize_static(model, signals)
        plotly_figs = _visualize_plotly(model, signals)
        figures: Dict[str, Any] = {}
        for k in PANEL_KEYS:
            figures[f"{k}_static"] = static_figs[k]
            figures[f"{k}_plotly"] = plotly_figs[k]
    elif backend == "plotly":
        figures = _visualize_plotly(model, signals)
    else:
        figures = _visualize_static(model, signals)

    from jaxfne.vis.exporters import FigureBundle

    bundle = FigureBundle(figures)

    if save_dir is not None:
        # Export may raise if formats include "html" for static figs — export_figures
        # already skips html for matplotlib figures, so this is safe.
        bundle.export(save_dir, formats=tuple(formats), dpi=int(dpi))

    if show:
        if backend in ("static", "both"):
            try:
                import matplotlib.pyplot as plt

                plt.show()
            except Exception:
                pass
        if backend in ("plotly", "both"):
            for fig in figures.values():
                try:
                    fig.show()
                except Exception:
                    pass

    return bundle


def _visualize_static(model: Any, signals: Any) -> Dict[str, Any]:
    return {
        "01_network_structure": _panel_network_structure(model, signals, backend="static"),
        "02_connectivity": _panel_connectivity(model, signals, backend="static"),
        "03_parameter_summary": _panel_parameter_summary(model, signals, backend="static"),
        "04_raster": _panel_raster(model, signals, backend="static"),
        "05_population_rates": _panel_population_rates(model, signals, backend="static"),
        "06_state_traces": _panel_state_traces(model, signals, backend="static"),
        "07_source": _panel_source(model, signals, backend="static"),
        "08_field_probe": _panel_field_probe(model, signals, backend="static"),
    }


def _visualize_plotly(model: Any, signals: Any) -> Dict[str, Any]:
    return {
        "01_network_structure": _panel_network_structure(model, signals, backend="plotly"),
        "02_connectivity": _panel_connectivity(model, signals, backend="plotly"),
        "03_parameter_summary": _panel_parameter_summary(model, signals, backend="plotly"),
        "04_raster": _panel_raster(model, signals, backend="plotly"),
        "05_population_rates": _panel_population_rates(model, signals, backend="plotly"),
        "06_state_traces": _panel_state_traces(model, signals, backend="plotly"),
        "07_source": _panel_source(model, signals, backend="plotly"),
        "08_field_probe": _panel_field_probe(model, signals, backend="plotly"),
    }


# Backwards-tick alias — some notebooks used `visualise` spelling historically.
# Not advertised, but do not break search-and-replace across siloed copies.
visualise = visualize

__all__ = ["visualize", "visualise", "PANEL_KEYS", "LARGE_N_RASTER", "LARGE_N_CONNECTIVITY", "LARGE_N_GEOMETRY", "MAX_RASTER_POINTS", "MAX_CONNECTIVITY_EDGES"]
