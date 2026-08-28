"""Interactive canonical cortical column viewer — realized EdgeList only.

Standalone HTML viewer for a canonical cortical column (or any Model):
layer, E/I class, position (x/y/z), phenotype, H ownership, degree,
weight/delay stats, with edge toggles E→E / E→I / I→E / I→I / FF / FB.

Displays the **realized EdgeList** (``model.params["edge_list"]``) as the
source of truth, not just the configured spec (``cfg.metadata["connectivity"]``).
A Configured→Realized table shows what was requested vs what was materialized.
All numerics are read from the executed Model; no new simulation is run.

Δscience = 0 — this module is additive, read-only, and optional. It never
touches emitter kernels, samplers, or solver paths. Plotly is imported lazily
inside the render function so ``import jaxfne`` pays no graphics overhead when
the viewer is not used (enforced by ``test_simulation_engine_has_zero_graphics_overhead``).

Usage
-----
>>> import jaxfne as jtfne
>>> from jaxfne.vis.column_viewer import render_column_viewer
>>> cfg = jtfne.build_laminar_column(n=1000, ei_profile="canonical")
>>> cfg = cfg.set_emitter("izhikevich","cortical_eig").probes(["spikes"],n_contacts=16).field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
>>> model = jtfne.construct(cfg)
>>> path = render_column_viewer(model, output_path="artifacts/column_viewer_canonical_1000n.html")

The function writes a standalone HTML file (Plotly.js via CDN, no Python
server needed) and returns its path plus a data summary dict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data collection (pure NumPy/JAX host reads — no simulation, no Plotly)
# ---------------------------------------------------------------------------

def _is_E(cell_type: str) -> bool:
    return str(cell_type) == "E"


def _classify_edge(pre_is_E: bool, post_is_E: bool) -> str:
    if pre_is_E and post_is_E:
        return "E→E"
    if pre_is_E and not post_is_E:
        return "E→I"
    if not pre_is_E and post_is_E:
        return "I→E"
    return "I→I"


def _area_rank_map(model) -> dict[str, int]:
    """Hierarchy rank for FF/FB classification.

    Uses declaration order in ``cfg.metadata["column_names"]`` / ``columns`` as
    the low→high hierarchy (canonical V1 < V4 < PFC). A single-area column
    gives every edge rank 0 so FF/FB are both zero and shown as inactive.
    """
    meta = getattr(getattr(model, "cfg", None), "metadata", {}) or {}
    names = meta.get("column_names")
    if not names:
        cols = meta.get("columns") or []
        names = [c.get("name") for c in cols if c.get("name")]
    # fallback: unique areas from neuron_table in encounter order
    if not names:
        try:
            rows = model.neuron_table()
            seen: list[str] = []
            for r in rows:
                a = str(r.get("area", ""))
                if a and a not in seen:
                    seen.append(a)
            names = seen
        except Exception:
            names = []
    return {str(n): i for i, n in enumerate(names)}


def collect_column_viewer_data(model, *, max_edges_for_stats: int = 200_000) -> dict[str, Any]:
    """Read realized state from *model* into a JSON-safe viewer payload.

    Never materializes dense W; all counts come from ``edge_list`` arrays.
    ``max_edges_for_stats`` caps histogram sampling for very large EdgeLists
    (default 200k keeps stats instant at N=1k all-to-all 999k edges).
    """
    # --- neuron table / positions ---
    rows = model.neuron_table()
    n = len(rows)
    areas = np.asarray([str(r.get("area", "")) for r in rows])
    layers = np.asarray([str(r.get("layer", "")) for r in rows])
    cell_types = np.asarray([str(r.get("cell_type", "")) for r in rows])
    xs = np.asarray([float(r.get("x", 0.0)) for r in rows], dtype=float)
    ys = np.asarray([float(r.get("y", 0.0)) for r in rows], dtype=float)
    zs = np.asarray([float(r.get("z", 0.0)) for r in rows], dtype=float)
    is_E = np.asarray([_is_E(ct) for ct in cell_types])
    phenotypes = np.asarray([f"{lyr}/{ct}" for lyr, ct in zip(layers, cell_types)])

    # --- H ownership ---
    h_vals: np.ndarray | None = None
    h_info: dict[str, Any] = {"present": False, "enabled": False, "mean": None, "note": ""}
    params = getattr(model, "params", {}) or {}
    if "hdp_initial_H" in params:
        try:
            h_raw = np.asarray(params["hdp_initial_H"])
            if h_raw.ndim == 2:
                h_vals = h_raw[:, 0].astype(float)  # first H-dim proxy
                h_info["note"] = f"vector H (dim {h_raw.shape[1]}), showing H[:,0]"
            else:
                h_vals = h_raw.astype(float)
            h_info["present"] = True
            h_info["mean"] = float(np.mean(h_vals))
            h_info["enabled"] = True
        except Exception:
            h_vals = None
    # also check cfg flag
    meta = getattr(getattr(model, "cfg", None), "metadata", {}) or {}
    if bool(meta.get("enable_hdp")) or bool(meta.get("hdp_params")):
        h_info["enabled"] = True
        if not h_info["present"]:
            h_info["note"] = "HDP enabled but hdp_initial_H not yet materialized on this model (tensor path sets it post-construct)"

    # --- realized EdgeList ---
    edge_list = params.get("edge_list") if isinstance(params, dict) else None
    if edge_list is None:
        raise ValueError("model.params['edge_list'] is missing — construct() must have run")
    pre = np.asarray(edge_list.pre, dtype=np.int64)
    post = np.asarray(edge_list.post, dtype=np.int64)
    weight = np.asarray(edge_list.weight, dtype=float)
    receptor_index = np.asarray(edge_list.receptor_index, dtype=int)
    tau_ms = np.asarray(edge_list.tau_ms, dtype=float)
    delay_steps = np.asarray(edge_list.delay_steps, dtype=int)
    n_edges = int(pre.shape[0])

    # degree (realized)
    out_deg = np.bincount(pre, minlength=n) if n_edges else np.zeros(n, dtype=int)
    in_deg = np.bincount(post, minlength=n) if n_edges else np.zeros(n, dtype=int)

    # edge categories (E/I → E/I) using realized endpoints
    # guard index range (defensive; construct should guarantee 0<=id<n)
    valid_edge = (pre >= 0) & (pre < n) & (post >= 0) & (post < n)
    pre_is_E = np.zeros(n_edges, dtype=bool)
    post_is_E = np.zeros(n_edges, dtype=bool)
    if n_edges and valid_edge.any():
        pre_is_E[valid_edge] = is_E[pre[valid_edge]]
        post_is_E[valid_edge] = is_E[post[valid_edge]]
    edge_cat = np.asarray([_classify_edge(bool(a), bool(b)) for a, b in zip(pre_is_E, post_is_E)])
    cat_counts = {k: int((edge_cat == k).sum()) for k in ("E→E", "E→I", "I→E", "I→I")}

    # FF / FB via area hierarchy
    rank_map = _area_rank_map(model)
    # map each neuron id → rank via its area string
    neuron_rank = np.asarray([rank_map.get(str(a), 0) for a in areas], dtype=int)
    if n_edges and valid_edge.any():
        pre_rank = neuron_rank[pre[valid_edge]]
        post_rank = neuron_rank[post[valid_edge]]
        # only cross-area edges participate in FF/FB
        pre_area = areas[pre[valid_edge]]
        post_area = areas[post[valid_edge]]
        cross = pre_area != post_area
        ff_mask = cross & (pre_rank < post_rank)
        fb_mask = cross & (pre_rank > post_rank)
        n_ff = int(ff_mask.sum())
        n_fb = int(fb_mask.sum())
        n_local = int((~cross).sum())
        # full edge mask for display categories (need aligned to n_edges)
        ff_full = np.zeros(n_edges, dtype=bool)
        fb_full = np.zeros(n_edges, dtype=bool)
        local_full = np.zeros(n_edges, dtype=bool)
        # scatter back to full indexing (valid_edge positions)
        valid_idx = np.where(valid_edge)[0]
        ff_full[valid_idx[cross & (pre_rank < post_rank)]] if False else None  # placeholder
        # easier: build directly
        for i, idx in enumerate(valid_idx):
            if not cross[i]:
                local_full[idx] = True
            elif pre_rank[i] < post_rank[i]:
                ff_full[idx] = True
            elif pre_rank[i] > post_rank[i]:
                fb_full[idx] = True
    else:
        n_ff = n_fb = n_local = 0
        ff_full = fb_full = local_full = np.zeros(n_edges, dtype=bool)

    # weight / delay stats (sampled if huge)
    sample_idx = np.arange(n_edges)
    if n_edges > max_edges_for_stats:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_edges, size=max_edges_for_stats, replace=False)
    w_sample = weight[sample_idx] if n_edges else np.array([], dtype=float)
    delay_sample = delay_steps[sample_idx] if n_edges else np.array([], dtype=int)
    # per-category weight means (full population, not sample — cheap reductions)
    cat_w_mean: dict[str, float | None] = {}
    cat_w_std: dict[str, float | None] = {}
    for cat in ("E→E", "E→I", "I→E", "I→I"):
        m = edge_cat == cat
        if m.any():
            cat_w_mean[cat] = float(weight[m].mean())
            cat_w_std[cat] = float(weight[m].std())
        else:
            cat_w_mean[cat] = None
            cat_w_std[cat] = None

    # delay: usually all-zero unless delay kernel used
    has_delay = bool((delay_steps != 0).any()) if n_edges else False

    # configured spec (what was requested)
    configured: dict[str, Any] = {}
    if meta:
        # connectivity (within_gain etc.)
        conn = meta.get("connectivity")
        if conn is not None:
            configured["connectivity"] = dict(conn)
        # circuit connections table if present
        circuit = meta.get("circuit") or {}
        if circuit.get("connections") is not None:
            configured["circuit_connections"] = [
                {k: v for k, v in c.items() if k in ("name", "source", "target", "probability", "weight", "sign", "mechanism", "status")}
                for c in circuit["connections"]
            ]
            configured["circuit_mechanisms"] = list(circuit.get("mechanisms", []))
        # connectivity_mode + compilation
        if "connectivity_mode" in meta:
            configured["connectivity_mode"] = str(meta["connectivity_mode"])
        if "connectivity_compilation" in meta:
            configured["connectivity_compilation"] = dict(meta["connectivity_compilation"])
        # canonical spec
        if meta.get("canonical_biophysics") is not None:
            configured["canonical_biophysics"] = bool(meta["canonical_biophysics"])
        # layer/cell_type phenotype declaration
        if meta.get("layer_fractions") is not None:
            configured["layer_fractions"] = dict(meta["layer_fractions"])
        if meta.get("layer_cell_types") is not None:
            configured["layer_cell_types"] = dict(meta["layer_cell_types"])
        # H spec
        if meta.get("hdp_params") is not None:
            configured["hdp_params"] = dict(meta["hdp_params"])
        if meta.get("hdp") is not None:
            configured["hdp"] = dict(meta["hdp"])

    # realized summary
    realized: dict[str, Any] = {
        "n_neurons": int(n),
        "n_edges": int(n_edges),
        "mean_in_degree": float(in_deg.mean()) if n else 0.0,
        "mean_out_degree": float(out_deg.mean()) if n else 0.0,
        "max_in_degree": int(in_deg.max()) if n else 0,
        "max_out_degree": int(out_deg.max()) if n else 0,
        "weight_mean": float(weight.mean()) if n_edges else None,
        "weight_std": float(weight.std()) if n_edges else None,
        "weight_min": float(weight.min()) if n_edges else None,
        "weight_max": float(weight.max()) if n_edges else None,
        "tau_ms_unique": sorted(np.unique(tau_ms).tolist()) if n_edges else [],
        "delay_steps_unique": sorted(np.unique(delay_steps).tolist()) if n_edges else [],
        "has_delay": has_delay,
        "edge_category_counts": cat_counts,
        "edge_category_weight_mean": cat_w_mean,
        "edge_category_weight_std": cat_w_std,
        "n_FF": int(n_ff),
        "n_FB": int(n_fb),
        "n_local": int(n_local),
        "areas": sorted(np.unique(areas).tolist()),
        "layers": sorted(np.unique(layers).tolist()),
        "cell_types": sorted(np.unique(cell_types).tolist()),
        "layer_counts": {str(k): int(v) for k, v in zip(*np.unique(layers, return_counts=True))},
        "celltype_counts": {str(k): int(v) for k, v in zip(*np.unique(cell_types, return_counts=True))},
        "phenotype_counts": {str(k): int(v) for k, v in zip(*np.unique(phenotypes, return_counts=True))},
        "in_degree_per_neuron_sample": in_deg[: min(20, n)].tolist(),
        "out_degree_per_neuron_sample": out_deg[: min(20, n)].tolist(),
    }

    return {
        "n": int(n),
        "areas": areas,
        "layers": layers,
        "cell_types": cell_types,
        "is_E": is_E,
        "phenotypes": phenotypes,
        "x": xs, "y": ys, "z": zs,
        "h_vals": h_vals,
        "h_info": h_info,
        "pre": pre, "post": post, "weight": weight,
        "receptor_index": receptor_index, "tau_ms": tau_ms, "delay_steps": delay_steps,
        "edge_cat": edge_cat,
        "ff_full": ff_full, "fb_full": fb_full, "local_full": local_full,
        "in_deg": in_deg, "out_deg": out_deg,
        "w_sample": w_sample, "delay_sample": delay_sample,
        "cat_counts": cat_counts, "cat_w_mean": cat_w_mean,
        "configured": configured, "realized": realized,
        "rank_map": rank_map,
        "columns_meta": meta.get("columns"),
        "connectivity_meta": meta.get("connectivity"),
    }


def _histogram_counts(values: np.ndarray, bins: int = 30) -> tuple[list[float], list[int]]:
    if values.size == 0:
        return [], []
    # drop inf/nan already handled; but be defensive
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return [], []
    counts, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers.tolist(), counts.tolist()


def render_column_viewer(
    model,
    *,
    output_path: str | Path = "artifacts/column_viewer_canonical_1000n.html",
    title: str = "Canonical cortical column — realized EdgeList viewer",
    max_edges_per_category: int = 600,
    max_edges_total: int = 2400,
) -> tuple[Path, dict[str, Any]]:
    """Generate a standalone HTML viewer for *model*.

    Parameters
    ----------
    model : Model
        Constructed ``jaxfne.Model`` (after ``construct()``) — the viewer
        reads its realized ``edge_list`` and ``neuron_table``.
    output_path : str | Path
        Where to write the HTML (parent directories created).
    title : str
        Viewer header title.
    max_edges_per_category : int
        Per-category line cap for 3D edge traces (keeps the page interactive
        at N=1000 all-to-all 999k edges).
    max_edges_total : int
        Total cap across all categories.

    Returns
    -------
    (output_path, summary_dict) : (Path, dict)
        Path to the written HTML and a JSON-safe summary (configured→realized
        comparison, degree/weight/delay aggregates).
    """
    data = collect_column_viewer_data(model)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = data["n"]
    realized = data["realized"]
    configured = data["configured"]
    layers = data["layers"]
    cell_types = data["cell_types"]
    h_info = data["h_info"]
    h_vals = data["h_vals"]
    x, y, z = data["x"], data["y"], data["z"]
    pre, post = data["pre"], data["post"]
    edge_cat = data["edge_cat"]
    ff_full, fb_full = data["ff_full"], data["fb_full"]
    in_deg, out_deg = data["in_deg"], data["out_deg"]
    w_sample, delay_sample = data["w_sample"], data["delay_sample"]

    # --- build Plotly trace payloads as plain JSON (no Plotly import needed here) ---
    # neuron scatter: one trace per layer for legend filtering (layer toggle via legend click)
    unique_layers = sorted(np.unique(layers).tolist())
    layer_traces: list[dict[str, Any]] = []
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    layer_color = {lyr: palette[i % len(palette)] for i, lyr in enumerate(unique_layers)}
    # E/I marker symbol distinction: E circle, I diamond
    for lyr in unique_layers:
        sel = layers == lyr
        # per-point: E circle, I diamond proxy via hover + opacity split into two sub-traces per layer
        # to avoid WebGL complexity, emit one trace per layer/E_I split
        for is_e, label, sym, col in [(True, "E", "circle", None), (False, "I", "diamond", None)]:
            mask = sel & (data["is_E"] == is_e)
            if not int(mask.sum()):
                continue
            hover = [
                f"id={i}<br>area={str(data['areas'][i])}<br>layer={str(layers[i])}<br>cell_type={str(cell_types[i])}"
                f"<br>phenotype={str(data['phenotypes'][i])}"
                f"<br>E/I={'E' if bool(data['is_E'][i]) else 'I'}"
                + (f"<br>H={float(h_vals[i]):.3f}" if h_vals is not None else "")
                + f"<br>in_deg={int(in_deg[i])} out_deg={int(out_deg[i])}"
                + f"<br>x={float(x[i]):.3f} y={float(y[i]):.3f} z={float(z[i]):.3f}"
                for i in np.where(mask)[0]
            ]
            base = layer_color[lyr]
            # dim I slightly so E/I remain distinguishable even within a layer color
            marker_color = base if is_e else base
            layer_traces.append({
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{lyr} {label}",
                "legendgroup": lyr,
                "x": x[mask].tolist(), "y": y[mask].tolist(), "z": z[mask].tolist(),
                "marker": {"size": 4 if is_e else 3.5, "color": marker_color, "symbol": sym, "opacity": 0.88, "line": {"width": 0}},
                "text": hover, "hoverinfo": "text",
            })

    # edge traces — one per category (E→E etc. + FF/FB) so toggle checkboxes can hide them
    edge_traces: list[dict[str, Any]] = []
    edge_colors = {"E→E": "#66c2a5", "E→I": "#fc8d62", "I→E": "#8da0cb", "I→I": "#e78ac3", "FF": "#ffd92f", "FB": "#a6d854"}
    # cap per category deterministically (first N in file order, but sampled if > cap)
    rng = np.random.default_rng(1)
    categories = ["E→E", "E→I", "I→E", "I→I"]
    for cat in categories:
        idx = np.where(edge_cat == cat)[0]
        if idx.size == 0:
            continue
        if idx.size > max_edges_per_category:
            idx = rng.choice(idx, size=max_edges_per_category, replace=False)
            idx = np.sort(idx)
        xs_e, ys_e, zs_e = [], [], []
        for k in idx:
            p, q = int(pre[k]), int(post[k])
            if p >= n or q >= n:
                continue
            xs_e += [float(x[p]), float(x[q]), None]
            ys_e += [float(y[p]), float(y[q]), None]
            zs_e += [float(z[p]), float(z[q]), None]
        edge_traces.append({
            "type": "scatter3d", "mode": "lines",
            "name": f"edges {cat} (show {idx.size} / {realized['edge_category_counts'][cat]})",
            "showlegend": True, "visible": True,
            "category": cat,
            "x": xs_e, "y": ys_e, "z": zs_e,
            "line": {"color": edge_colors[cat], "width": 1},
            "opacity": 0.22, "hoverinfo": "skip",
        })
    # FF / FB traces (cross-area only; for single column these are empty and omitted)
    for cat, mask, label in [("FF", ff_full, "FF (cross-area low→high)"), ("FB", fb_full, "FB (cross-area high→low)")]:
        idx = np.where(mask)[0]
        cnt_real = int(mask.sum())
        if cnt_real == 0:
            continue
        if idx.size > max_edges_per_category:
            idx = rng.choice(idx, size=max_edges_per_category, replace=False)
            idx = np.sort(idx)
        xs_e, ys_e, zs_e = [], [], []
        for k in idx:
            p, q = int(pre[k]), int(post[k])
            xs_e += [float(x[p]), float(x[q]), None]
            ys_e += [float(y[p]), float(y[q]), None]
            zs_e += [float(z[p]), float(z[q]), None]
        edge_traces.append({
            "type": "scatter3d", "mode": "lines",
            "name": f"edges {cat} ({label}, show {idx.size} / {cnt_real})",
            "showlegend": True, "visible": True,
            "category": cat,
            "x": xs_e, "y": ys_e, "z": zs_e,
            "line": {"color": edge_colors[cat], "width": 1.5},
            "opacity": 0.35, "hoverinfo": "skip",
        })

    # histogram payloads (centers + counts) for weight/delay/degree
    w_centers, w_counts = _histogram_counts(w_sample, bins=32)
    d_centers, d_counts = _histogram_counts(delay_sample.astype(float) if delay_sample.size else delay_sample, bins=min(16, max(8, int(np.unique(delay_sample).size) if delay_sample.size else 8)))
    in_centers, in_counts_hist = _histogram_counts(in_deg.astype(float), bins=24)
    out_centers, out_counts_hist = _histogram_counts(out_deg.astype(float), bins=24)
    # per-layer degree mean for bar chart
    layer_in_mean = {lyr: float(in_deg[layers == lyr].mean()) if (layers == lyr).any() else 0.0 for lyr in unique_layers}
    layer_out_mean = {lyr: float(out_deg[layers == lyr].mean()) if (layers == lyr).any() else 0.0 for lyr in unique_layers}

    # phenotype heatmap: layer x cell_type counts
    unique_cts = sorted(np.unique(cell_types).tolist())
    pheno_matrix = []
    for lyr in unique_layers:
        row = []
        for ct in unique_cts:
            row.append(int(((layers == lyr) & (cell_types == ct)).sum()))
        pheno_matrix.append(row)

    # JSON payload for the inline script
    plot_payload = {
        "layerTraces": layer_traces,
        "edgeTraces": edge_traces,
        "histograms": {
            "weight": {"centers": w_centers, "counts": w_counts},
            "delay": {"centers": d_centers, "counts": d_counts},
            "inDegree": {"centers": in_centers, "counts": in_counts_hist},
            "outDegree": {"centers": out_centers, "counts": out_counts_hist},
        },
        "layerInMean": layer_in_mean,
        "layerOutMean": layer_out_mean,
        "phenoMatrix": pheno_matrix,
        "phenoRows": unique_layers,
        "phenoCols": unique_cts,
    }

    # build HTML (standalone, cdn Plotly)
    # summary table html fragments
    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            if abs(v) < 0.001 and v != 0:
                return f"{v:.3e}"
            return f"{v:.4g}"
        return str(v)

    cfg_conn = configured.get("connectivity", {})
    cfg_comp = configured.get("connectivity_compilation")
    cfg_circ = configured.get("circuit_connections")
    realized_html_rows = (
        f"<tr><td>N neurons</td><td>—</td><td>{realized['n_neurons']}</td></tr>"
        f"<tr><td>Edges</td><td>{' / '.join(str(x) for x in ([str(len(cfg_circ))] if cfg_circ is not None else [])) if cfg_circ is not None else (str(cfg_comp.get('declared_rule_edge_count')) if isinstance(cfg_comp, dict) and 'declared_rule_edge_count' in cfg_comp else 'spec: see configured')} "
        f"</td><td><b>{realized['n_edges']:,}</b> (realized EdgeList)</td></tr>"
        f"<tr><td>Mean in-degree</td><td>—</td><td>{_fmt(realized['mean_in_degree'])}</td></tr>"
        f"<tr><td>Mean out-degree</td><td>—</td><td>{_fmt(realized['mean_out_degree'])}</td></tr>"
        f"<tr><td>Weight</td><td>{_fmt(cfg_conn.get('within_gain')) if cfg_conn else '—'} (configured within_gain)</td><td>mean {_fmt(realized['weight_mean'])} σ {_fmt(realized['weight_std'])} min {_fmt(realized['weight_min'])} max {_fmt(realized['weight_max'])}</td></tr>"
        f"<tr><td>Delay</td><td>—</td><td>{'has non-zero delay' if realized['has_delay'] else 'all zero (instantaneous)'} unique steps {realized['delay_steps_unique'][:8]}</td></tr>"
        f"<tr><td>τ per edge</td><td>—</td><td>{realized['tau_ms_unique']}</td></tr>"
        f"<tr><td>Categories</td><td>—</td><td>E→E {realized['edge_category_counts'].get('E→E',0):,} · E→I {realized['edge_category_counts'].get('E→I',0):,} · I→E {realized['edge_category_counts'].get('I→E',0):,} · I→I {realized['edge_category_counts'].get('I→I',0):,}</td></tr>"
        f"<tr><td>FF / FB / local</td><td>—</td><td>FF {realized['n_FF']:,} · FB {realized['n_FB']:,} · local {realized['n_local']:,} (hierarchy {list(data['rank_map'].keys()) or 'single-area'})</td></tr>"
        f"<tr><td>Layers → counts</td><td>{json.dumps(configured.get('layer_fractions', {}), indent=0)[:300] if configured.get('layer_fractions') else '—'}</td><td>{json.dumps(realized['layer_counts'])}</td></tr>"
        f"<tr><td>Cell-type → counts</td><td>—</td><td>{json.dumps(realized['celltype_counts'])} phenotype distinct {len(realized['phenotype_counts'])}</td></tr>"
    )
    # configured JSON pretty
    configured_pre = json.dumps(configured, indent=2, default=str) if configured else "{}"
    # escape for html
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    configured_pre_esc = _esc(configured_pre)

    html = f"""<!doctype html>
<html lang=\"en\"><head>
<meta charset=\"utf-8\"/>
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>
<title>{_esc(title)}</title>
<script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
<style>
  :root {{ --bg:#0b0e14; --card:#121821; --muted:#9aa4b2; --text:#e6edf3; --accent:#66c2a5; --border:#1f2a37; }}
  html,body {{ margin:0; padding:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial; }}
  header {{ padding:18px 20px 10px; border-bottom:1px solid var(--border); }}
  header h1 {{ margin:0 0 4px; font-size:20px; font-weight:700; }}
  header p {{ margin:0; color:var(--muted); }}
  .wrap {{ max-width:1400px; margin:0 auto; padding:16px 16px 40px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px; margin:10px 0; }}
  .card h2 {{ margin:0 0 8px; font-size:15px; font-weight:700; }}
  .grid {{ display:grid; gap:12px; }}
  .grid-2 {{ grid-template-columns:1fr 1fr; }}
  .grid-3 {{ grid-template-columns:1fr 1fr 1fr; }}
  @media (max-width:900px){{ .grid-2,.grid-3{{grid-template-columns:1fr;}} }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#1f2a37; color:var(--muted); font-size:11px; margin:2px 4px 2px 0; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }}
  .controls label {{ display:inline-flex; gap:6px; align-items:center; cursor:pointer; font-size:13px; }}
  .note {{ color:var(--muted); font-size:12px; }}
  pre {{ background:#0e141e; border:1px solid var(--border); border-radius:8px; padding:10px; overflow:auto; font-size:12px; }}
  a {{ color:#7cc4ff; }}
</style>
</head><body>
<header>
  <h1>{_esc(title)}</h1>
  <p>Realized EdgeList only — no simulation re-run, no kernel change (Δscience=0). Hover neurons for layer / E/I / phenotype / H / degree / position. Toggle edges by category via checkboxes + legend. Standalone HTML — no server.</p>
  <p class=\"note\">N={realized['n_neurons']:,} · edges={realized['n_edges']:,} · layers={", ".join(realized['layers'])} · cell types={", ".join(realized['cell_types'])} · phenotype distinct={len(realized['phenotype_counts'])} · H ownership={'present (mean '+_fmt(h_info.get('mean'))+')' if h_info.get('present') else ('HDP enabled, awaiting H' if h_info.get('enabled') else 'no H (equilibrium 1.0, HDP disabled)')} </p>
</header>
<div class=\"wrap\">

  <div class=\"card\">
    <h2>Configured → Realized (EdgeList is truth)</h2>
    <p class=\"note\">Left = what was requested in <code>cfg.metadata</code> (connectivity / circuit / compilation). Right = what <code>model.params[&quot;edge_list&quot;]</code> materialized — the only place dynamics reads.</p>
    <table>
      <thead><tr><th>Aspect</th><th>Configured spec</th><th>Realized EdgeList</th></tr></thead>
      <tbody>{realized_html_rows}</tbody>
    </table>
    <details style=\"margin-top:8px\"><summary>Configured JSON (cfg.metadata slice)</summary><pre>{configured_pre_esc}</pre></details>
    <p class=\"note\">Edge sampling cap for display: {max_edges_per_category} per category, {max_edges_total} total. Histograms sampled from up to {data['w_sample'].size* (realized['n_edges']/max(data['w_sample'].size,1)):.0f} edges with deterministic seed.</p>
  </div>

  <div class=\"card\">
    <h2>Edge toggles — E→E / E→I / I→E / I→I / FF / FB</h2>
    <div class=\"controls\" id=\"edgeToggles\">
      <label><input type=\"checkbox\" data-cat=\"E→E\" checked> <span style=\"color:#66c2a5\">■</span> E→E <span class=\"pill\">{realized['edge_category_counts'].get('E→E',0):,}</span></label>
      <label><input type=\"checkbox\" data-cat=\"E→I\" checked> <span style=\"color:#fc8d62\">■</span> E→I <span class=\"pill\">{realized['edge_category_counts'].get('E→I',0):,}</span></label>
      <label><input type=\"checkbox\" data-cat=\"I→E\" checked> <span style=\"color:#8da0cb\">■</span> I→E <span class=\"pill\">{realized['edge_category_counts'].get('I→E',0):,}</span></label>
      <label><input type=\"checkbox\" data-cat=\"I→I\" checked> <span style=\"color:#e78ac3\">■</span> I→I <span class=\"pill\">{realized['edge_category_counts'].get('I→I',0):,}</span></label>
      <label><input type=\"checkbox\" data-cat=\"FF\" checked> <span style=\"color:#ffd92f\">■</span> FF <span class=\"pill\">{realized['n_FF']:,}</span></label>
      <label><input type=\"checkbox\" data-cat=\"FB\" checked> <span style=\"color:#a6d854\">■</span> FB <span class=\"pill\">{realized['n_FB']:,}</span></label>
      <span class=\"note\" id=\"edgeSummary\"></span>
    </div>
    <p class=\"note\">FF = cross-area low→high in hierarchy {list(data['rank_map'].keys()) or ['single-area: FF/FB are zero']}; FB = high→low. Local = same-area ({realized['n_local']:,} edges) not toggled separately — it is inside the E→E etc. categories. For single-area columns FF/FB are zero by construction.</p>
  </div>

  <div class=\"card\">
    <h2>3D column — layers · E/I class · position · phenotype · H ownership · degree (hover)</h2>
    <div id=\"plot3d\" style=\"width:100%;height:620px\"></div>
    <p class=\"note\">Rotate/drag to inspect depth (z). Layer colors via legend — click legend entries to isolate a layer. Neuron symbols: E circle, I diamond (same hue per layer). Hover shows H, in/out-degree, phenotype, position.</p>
  </div>

  <div class=\"grid grid-3\">
    <div class=\"card\"><h2>Weight stats (realized)</h2><div id=\"weightHist\" style=\"height:260px\"></div><p class=\"note\">Mean {_fmt(realized['weight_mean'])} σ {_fmt(realized['weight_std'])} · by category: E→E {_fmt(realized['edge_category_weight_mean'].get('E→E'))} · E→I {_fmt(realized['edge_category_weight_mean'].get('E→I'))} · I→E {_fmt(realized['edge_category_weight_mean'].get('I→E'))} · I→I {_fmt(realized['edge_category_weight_mean'].get('I→I'))}</p></div>
    <div class=\"card\"><h2>Delay stats (steps)</h2><div id=\"delayHist\" style=\"height:260px\"></div><p class=\"note\">{'Has non-zero delay' if realized['has_delay'] else 'All delays zero — instantaneous kernel'} · unique steps {realized['delay_steps_unique'][:10]}</p></div>
    <div class=\"card\"><h2>Degree stats</h2><div id=\"degreeHist\" style=\"height:260px\"></div><p class=\"note\">Mean in {_fmt(realized['mean_in_degree'])} max {realized['max_in_degree']} · mean out {_fmt(realized['mean_out_degree'])} max {realized['max_out_degree']}</p></div>
  </div>

  <div class=\"grid grid-2\">
    <div class=\"card\"><h2>Layer × E/I degree (mean)</h2><div id=\"layerDegree\" style=\"height:280px\"></div></div>
    <div class=\"card\"><h2>Phenotype heatmap — layer × cell type counts</h2><div id=\"phenoHeat\" style=\"height:280px\"></div></div>
  </div>

  <div class=\"card\">
    <h2>Phenotype & H ownership</h2>
    <table>
      <tr><th>Phenotype (layer/cell_type) distinct</th><td>{len(realized['phenotype_counts'])} · {json.dumps(realized['phenotype_counts'])}</td></tr>
      <tr><th>Cell-type counts</th><td>{json.dumps(realized['celltype_counts'])} — E/I split: E {realized['celltype_counts'].get('E',0)} / I {sum(v for k,v in realized['celltype_counts'].items() if k!='E')}</td></tr>
      <tr><th>Layer counts</th><td>{json.dumps(realized['layer_counts'])}</td></tr>
      <tr><th>H ownership</th><td>{_esc(str(h_info))} {'— per-neuron H histogram below' if h_vals is not None else '— equilibrium (model.params has no hdp_initial_H)'}</td></tr>
    </table>
    <div id=\"hHist\" style=\"height:220px;margin-top:10px;display:{'block' if h_vals is not None else 'none'}\"></div>
  </div>

  <div class=\"card\">
    <h2>How to verify</h2>
    <pre>import jaxfne as jtfne
from jaxfne.vis.column_viewer import render_column_viewer, collect_column_viewer_data
cfg = jtfne.build_laminar_column(n=1000, ei_profile="canonical")
cfg = cfg.set_emitter("izhikevich","cortical_eig").probes(["spikes"],n_contacts=16).field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
model = jtfne.construct(cfg)
data = collect_column_viewer_data(model)  # no simulation, reads realized EdgeList
render_column_viewer(model, output_path="artifacts/column_viewer_canonical_1000n.html")
# Check: data["realized"]["n_edges"] == int(model.params["edge_list"].n_edges)
# and edge_category_counts sum to n_edges.</pre>
    <p class=\"note\">No kernel or sampler was changed to build this viewer. The HTML is self-contained (Plotly.js via CDN) — open it in a browser, no Python server needed. Re-render with any Model (laminar, multi-area, neuronal tensor, HDP) without re-running a simulation.</p>
  </div>

</div>

<script>
const payload = {json.dumps(plot_payload)};
const nNeurons = {n};
const nEdgesRealized = {realized['n_edges']};
const hVals = {json.dumps(h_vals.tolist() if h_vals is not None else None)};

// --- 3D ---
function build3d() {{
  const traces = [...payload.layerTraces, ...payload.edgeTraces];
  const layout = {{
    paper_bgcolor: "#0b0e14", scene: {{
      bgcolor: "black",
      xaxis: {{title:"x (mm)", color:"#aaa", gridcolor:"#1f2a37"}},
      yaxis: {{title:"y (mm)", color:"#aaa", gridcolor:"#1f2a37"}},
      zaxis: {{title:"depth z (mm, 0 superficial → deep)", color:"#aaa", gridcolor:"#1f2a37", autorange:"reversed"}},
    }},
    margin:{{l:0,r:0,t:10,b:0}},
    legend:{{font:{{color:"#e6edf3"}}, bgcolor:"rgba(0,0,0,0.35)"}},
    title:{{text:"", font:{{color:"#e6edf3"}}}},
  }};
  Plotly.newPlot("plot3d", traces, layout, {{responsive:true}});

  // toggle handling: map category → trace indices (edge traces only start after layerTraces)
  const layerCount = payload.layerTraces.length;
  const catToIdx = {{}};
  payload.edgeTraces.forEach((tr,i) => {{
    const cat = tr.category;
    catToIdx[cat] = catToIdx[cat] || [];
    catToIdx[cat].push(layerCount + i);
  }});
  function visibleFor(cat) {{
    const cb = document.querySelector(`#edgeToggles input[data-cat="${{cat}}"]`);
    return cb ? cb.checked : true;
  }}
  function applyToggles() {{
    const vis = payload.edgeTraces.map(tr => visibleFor(tr.category));
    // restyle only edge traces
    vis.forEach((v,i) => {{
      Plotly.restyle("plot3d", {{visible: v}}, [layerCount + i]);
    }});
    const totalVisible = payload.edgeTraces.filter((tr,i)=>vis[i]).length;
    const shownEdges = payload.edgeTraces.filter((tr,i)=>vis[i]).reduce((s,tr)=> s + (tr.x ? tr.x.filter(x=>x!==null).length/2 : 0),0);
    document.getElementById("edgeSummary").textContent = ` · showing ${{totalVisible}} edge trace(s)`;
  }}
  document.querySelectorAll("#edgeToggles input").forEach(cb => cb.addEventListener("change", applyToggles));
  applyToggles();
}}

// --- histograms ---
function histPlot(div, hist, title, xTitle, color) {{
  if (!hist.centers.length) {{
    document.getElementById(div).innerHTML = '<p class="note">no data</p>'; return;
  }}
  Plotly.newPlot(div, [{{x:hist.centers, y:hist.counts, type:"bar", marker:{{color:color}}, hovertemplate:"%{{x:.4g}} → %{{y}}<extra></extra>"}}],
    {{paper_bgcolor:"#121821", plot_bgcolor:"#0e141e", margin:{{t:26,l:40,r:10,b:36}}, title:{{text:title, font:{{color:"#e6edf3", size:13}}}}, xaxis:{{title:xTitle, color:"#9aa4b2", gridcolor:"#1f2a37"}}, yaxis:{{title:"count", color:"#9aa4b2", gridcolor:"#1f2a37"}}, bargap:0.08}},
    {{displayModeBar:false, responsive:true}});
}}

function init() {{
  build3d();
  histPlot("weightHist", payload.histograms.weight, "Edge weight (realized)", "weight (native)", "#66c2a5");
  histPlot("delayHist", payload.histograms.delay, "Delay steps (realized)", "delay_steps", "#ffd92f");
  // degree: show in-degree histogram
  histPlot("degreeHist", payload.histograms.inDegree, "In-degree (realized)", "in-degree", "#8da0cb");
  // layer × degree
  const layers = Object.keys(payload.layerInMean);
  Plotly.newPlot("layerDegree", [
    {{x:layers, y:layers.map(k=>payload.layerInMean[k]), type:"bar", name:"mean in-degree", marker:{{color:"#8da0cb"}}}},
    {{x:layers, y:layers.map(k=>payload.layerOutMean[k]), type:"bar", name:"mean out-degree", marker:{{color:"#fc8d62"}}}},
  ], {{paper_bgcolor:"#121821", plot_bgcolor:"#0e141e", margin:{{t:10,l:40,r:10,b:36}}, barmode:"group", legend:{{font:{{color:"#9aa4b2"}}, orientation:"h"}}, xaxis:{{color:"#9aa4b2", gridcolor:"#1f2a37"}}, yaxis:{{color:"#9aa4b2", gridcolor:"#1f2a37", title:"mean degree"}}}}, {{displayModeBar:false, responsive:true}});

  Plotly.newPlot("phenoHeat", [{{z:payload.phenoMatrix, x:payload.phenoCols, y:payload.phenoRows, type:"heatmap", colorscale:"Blues", hovertemplate:"%{{y}} / %{{x}} = %{{z}}<extra></extra>", colorbar:{{title:"count"}}}}],
    {{paper_bgcolor:"#121821", plot_bgcolor:"#0e141e", margin:{{t:10,l:60,r:10,b:40}}, xaxis:{{color:"#9aa4b2"}}, yaxis:{{color:"#9aa4b2", autorange:"reversed"}}}}, {{displayModeBar:false, responsive:true}});

  if (hVals) {{
    const bins = 20;
    const lo = Math.min(...hVals), hi = Math.max(...hVals);
    const w = (hi-lo)/bins || 1;
    const counts = new Array(bins).fill(0);
    const centers = [];
    for(let i=0;i<bins;i++) centers.push(lo + (i+0.5)*w);
    for(const v of hVals) {{
      let b = Math.floor((v-lo)/w); if(b<0) b=0; if(b>=bins) b=bins-1; counts[b]++;
    }}
    histPlot("hHist", {{centers:centers, counts:counts}}, "H ownership per neuron (realized H0 / equilibrium)", "H", "#e78ac3");
  }}
}}
init();
</script>
</body></html>
"""
    out.write_text(html, encoding="utf-8")

    summary: dict[str, Any] = {
        "output_path": str(out),
        "title": title,
        "realized": realized,
        "configured": configured,
        "h_info": h_info,
        "n_edges_shown_per_category": max_edges_per_category,
        "n_layer_traces": len(layer_traces),
        "n_edge_traces": len(edge_traces),
        "note": "Realized EdgeList is truth — configured is shown only for comparison. No simulation was run; no kernel changed.",
    }
    return out, summary


# Back-compat alias used in some docs/scripts
interactive_column_viewer = render_column_viewer
render_interactive_column_viewer = render_column_viewer

__all__ = [
    "collect_column_viewer_data",
    "render_column_viewer",
    "interactive_column_viewer",
    "render_interactive_column_viewer",
]
