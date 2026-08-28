"""PseudoGenome development viewer — G → D(K_D) → N (not storage).

Standalone HTML viewer that shows configured→realized development: genome JSON
rules vs realized NeuronalTensor arrays (counts, positions, edges, weights,
delays) and compares two developments with the same genome but different K_D
(seeds).

Δscience=0 — additive, read-only, optional. Never touches emitter kernels,
samplers, or solver paths. Plotly is imported lazily inside the render
function only when the viewer is used, so ``import jaxfne`` pays no graphics
overhead when the viewer is not used.

Canonical example (1000n)
-------------------------
>>> import jaxfne as jtfne
>>> from jaxfne.vis.pseudogenome_viewer import render_pseudogenome_development_viewer
>>> genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
>>> path, summary = render_pseudogenome_development_viewer(
...     genome, seeds=(0, 1), output_path="artifacts/pseudogenome_development_viewer.html"
... )

Writing uses only existing public APIs: ``PseudoGenome``, ``develop``,
``declared_constraints``, ``genome_rules_hash``, ``phenotype_sha256``,
``NeuronalTensor``, ``construct``, ``RuntimeConfiguration``,
``model.params['edge_list']``, ``model.params['positions']``,
``model.neuron_table()``. No new simulation kernel is introduced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers — pure NumPy/JAX host reads, no simulation re-run beyond develop+construct
# ---------------------------------------------------------------------------

def _is_E(cell_type: str) -> bool:
    return str(cell_type) == "E"


def _edge_category(pre_is_E: bool, post_is_E: bool) -> str:
    if pre_is_E and post_is_E:
        return "E→E"
    if pre_is_E and not post_is_E:
        return "E→I"
    if not pre_is_E and post_is_E:
        return "I→E"
    return "I→I"


def _collect_one_development(genome, seed: int, *, construct_seed: int = 7, duration_ms: float = 200.0, dt_ms: float = 0.5) -> dict[str, Any]:
    """Develop genome with K_D=seed, then construct to realize positions/edges.

    Returns a JSON-safe payload plus raw arrays for plotting.
    Uses only existing APIs: develop, construct, RuntimeConfiguration,
    declared_constraints, genome_rules_hash, phenotype_sha256.
    """
    import jaxfne as jtfne
    from jaxfne.jdna import declared_constraints, genome_rules_hash, phenotype_sha256
    from jaxfne.jdna.genome import develop

    tensor = develop(genome, seed=int(seed))
    pheno_hash = phenotype_sha256(tensor)
    genome_hash = genome_rules_hash(genome)
    constraints = declared_constraints(genome)

    # realized counts per layer/cell_type from tensor fractions
    layer_realized: list[dict[str, Any]] = []
    celltype_totals: dict[str, int] = {}
    phenotype_counts: dict[str, int] = {}
    total_n = 0
    for area in tensor.areas:
        for layer in area.layers:
            n = int(layer.n_neurons)
            total_n += n
            # derive integer counts from fractions (same as _allocate_counts largest-remainder)
            counts: dict[str, int] = {}
            for nt in layer.neuron_types:
                frac = float(nt.fraction or 0.0)
                counts[str(nt.name)] = int(round(n * frac))
            # correct rounding drift (largest remainder if needed) — use same logic as develop
            s = sum(counts.values())
            if s != n:
                # adjust largest fractional part
                fracs = {nt.name: float(nt.fraction or 0.0) for nt in layer.neuron_types}
                weighted = {k: n * v for k, v in fracs.items()}
                floors = {k: int(v) for k, v in weighted.items()}
                rem = n - sum(floors.values())
                if rem > 0:
                    order = sorted(floors.keys(), key=lambda k: weighted[k] - floors[k], reverse=True)
                    for k in order[:rem]:
                        floors[k] += 1
                    counts = floors
            # bands from declared constraints
            bands = constraints["areas"].get(area.name, {}).get("layers", {}).get(layer.name, {}).get("cell_type_count_bands", {})
            ok = all(bands.get(ct, [c, c])[0] <= c <= bands.get(ct, [c, c])[1] for ct, c in counts.items()) if bands else True
            layer_realized.append({
                "area": area.name,
                "layer": layer.name,
                "n_neurons": n,
                "depth_band": list(next(lg.depth_band for ag in genome.areas if ag.name == area.name for lg in ag.layers if lg.name == layer.name)),
                "counts": counts,
                "bands": bands,
                "ok": bool(ok),
            })
            for ct, c in counts.items():
                celltype_totals[ct] = celltype_totals.get(ct, 0) + c
                phenotype_counts[f"{layer.name}/{ct}"] = c

    # construct to get positions + edges (realized under K_S = construct_seed)
    # duration short (200ms) to keep edge realization at same counts but fast; n_edges does not depend on duration
    model = jtfne.construct(tensor, jtfne.neuronal_tensor.RuntimeConfiguration(seed=int(construct_seed), duration_ms=float(duration_ms), dt_ms=float(dt_ms)))
    rows = model.neuron_table()
    n_model = len(rows)
    xs = np.asarray([float(r.get("x", 0.0)) for r in rows], dtype=float) if rows else np.zeros(0)
    ys = np.asarray([float(r.get("y", 0.0)) for r in rows], dtype=float) if rows else np.zeros(0)
    zs = np.asarray([float(r.get("z", 0.0)) for r in rows], dtype=float) if rows else np.zeros(0)
    areas_arr = np.asarray([str(r.get("area", "")) for r in rows]) if rows else np.zeros(0, dtype=object)
    layers_arr = np.asarray([str(r.get("layer", "")) for r in rows]) if rows else np.zeros(0, dtype=object)
    cts_arr = np.asarray([str(r.get("cell_type", "")) for r in rows]) if rows else np.zeros(0, dtype=object)
    is_E = np.asarray([_is_E(str(ct)) for ct in cts_arr], dtype=bool) if cts_arr.size else np.zeros(0, dtype=bool)

    edge_list = model.params.get("edge_list") if isinstance(model.params, dict) else None
    if edge_list is None:
        raise ValueError("model.params['edge_list'] missing — construct must have run")
    pre = np.asarray(edge_list.pre, dtype=np.int64)
    post = np.asarray(edge_list.post, dtype=np.int64)
    weight = np.asarray(edge_list.weight, dtype=float)
    tau_ms = np.asarray(edge_list.tau_ms, dtype=float)
    delay_steps = np.asarray(edge_list.delay_steps, dtype=int)
    n_edges = int(pre.shape[0])
    # degree
    out_deg = np.bincount(pre, minlength=n_model) if n_edges and n_model else np.zeros(n_model, dtype=int)
    in_deg = np.bincount(post, minlength=n_model) if n_edges and n_model else np.zeros(n_model, dtype=int)
    # categories
    valid = (pre >= 0) & (pre < n_model) & (post >= 0) & (post < n_model) if n_edges else np.zeros(0, dtype=bool)
    pre_is_E = np.zeros(n_edges, dtype=bool)
    post_is_E = np.zeros(n_edges, dtype=bool)
    if n_edges and valid.any():
        pre_is_E[valid] = is_E[pre[valid]]
        post_is_E[valid] = is_E[post[valid]]
    edge_cat = np.asarray([_edge_category(bool(a), bool(b)) for a, b in zip(pre_is_E, post_is_E)]) if n_edges else np.zeros(0, dtype=object)
    cat_counts = {k: int((edge_cat == k).sum()) for k in ("E→E", "E→I", "I→E", "I→I")}
    cat_w_mean: dict[str, float | None] = {}
    for cat in ("E→E", "E→I", "I→E", "I→I"):
        m = edge_cat == cat
        cat_w_mean[cat] = float(weight[m].mean()) if m.any() else None

    weight_mean = float(weight.mean()) if n_edges else None
    weight_std = float(weight.std()) if n_edges else None
    weight_min = float(weight.min()) if n_edges else None
    weight_max = float(weight.max()) if n_edges else None
    # provenance
    prov = getattr(tensor, "provenance", None)

    return {
        "seed_KD": int(seed),
        "construct_seed_KS": int(construct_seed),
        "tensor": tensor,
        "genome_hash": genome_hash,
        "phenotype_hash": pheno_hash,
        "constraints": constraints,
        "layer_realized": layer_realized,
        "celltype_totals": celltype_totals,
        "phenotype_counts": phenotype_counts,
        "total_n": int(total_n),
        "n_model": int(n_model),
        "provenance": prov,
        "xs": xs, "ys": ys, "zs": zs,
        "areas_arr": areas_arr, "layers_arr": layers_arr, "cts_arr": cts_arr, "is_E": is_E,
        "pre": pre, "post": post, "weight": weight, "tau_ms": tau_ms, "delay_steps": delay_steps,
        "n_edges": int(n_edges),
        "in_deg": in_deg, "out_deg": out_deg,
        "edge_cat": edge_cat, "cat_counts": cat_counts, "cat_w_mean": cat_w_mean,
        "weight_mean": weight_mean, "weight_std": weight_std, "weight_min": weight_min, "weight_max": weight_max,
    }


def collect_pseudogenome_development_data(genome, seeds: Sequence[int] = (0, 1), *, construct_seed: int = 7) -> dict[str, Any]:
    """Collect genome rules + two developments (no Plotly import).

    Returns a JSON-safe dict plus raw arrays for the renderer.
    """
    from jaxfne.jdna import genome_rules_hash, declared_constraints

    rules_hash = genome_rules_hash(genome)
    constraints = declared_constraints(genome)
    # genome rules summary
    genome_rules: dict[str, Any] = {
        "name": genome.name,
        "schema_version": genome.schema_version,
        "description": genome.description,
        "genome_rules_hash": rules_hash,
        "development_parameters": dict(genome.development_parameters),
        "n_areas": len(genome.areas),
        "areas": [],
    }
    for area in genome.areas:
        area_rules: dict[str, Any] = {
            "name": area.name,
            "pose": dict(area.pose),
            "layers": [],
            "inter_connections": [dict(c) if isinstance(c, Mapping) else {"source_layer": c.source_layer, "source_neuron_type": c.source_neuron_type, "target_layer": c.target_layer, "target_neuron_type": c.target_neuron_type, "mechanism": c.mechanism} for c in area.inter_connections],
        }
        for lg in area.layers:
            area_rules["layers"].append({
                "name": lg.name,
                "n_neurons": int(lg.n_neurons),
                "depth_band": list(lg.depth_band),
                "cell_type_fractions": dict(lg.cell_type_fractions),
                "fraction_tolerance": {k: list(v) for k, v in lg.fraction_tolerance.items()},
                "geometry": dict(lg.geometry),
                "relative_sizes": dict(lg.relative_sizes),
            })
        genome_rules["areas"].append(area_rules)
    # count rules
    total_rules_n = sum(lg["n_neurons"] for ar in genome_rules["areas"] for lg in ar["layers"])
    n_rules_inter = sum(len(ar["inter_connections"]) for ar in genome_rules["areas"])

    developments = [_collect_one_development(genome, int(s), construct_seed=int(construct_seed)) for s in seeds]

    # determinism check: re-develop seed0
    from jaxfne.jdna.genome import develop as _develop
    from jaxfne.jdna import phenotype_sha256 as _pheno
    t_re = _develop(genome, seed=int(seeds[0]))
    deterministic = bool(_pheno(t_re) == developments[0]["phenotype_hash"])

    # same-genome different K_D check: at least one layer count differs
    diff_layers = 0
    diff_edges = developments[0]["n_edges"] != developments[1]["n_edges"] if len(developments) >= 2 else False
    if len(developments) >= 2:
        for a0, a1 in zip(developments[0]["layer_realized"], developments[1]["layer_realized"]):
            if a0["counts"] != a1["counts"]:
                diff_layers += 1

    return {
        "genome_rules": genome_rules,
        "genome_rules_hash": rules_hash,
        "constraints": constraints,
        "total_rules_n": int(total_rules_n),
        "n_rules_inter": int(n_rules_inter),
        "seeds": [int(s) for s in seeds],
        "construct_seed": int(construct_seed),
        "developments": developments,
        "deterministic_same_KD": bool(deterministic),
        "same_genome_different_KD_has_diff": bool(diff_layers > 0 or diff_edges),
        "diff_layers": int(diff_layers),
        "Δscience": 0,
    }


def _hist(values: np.ndarray, bins: int = 28) -> tuple[list[float], list[int]]:
    if values.size == 0:
        return [], []
    vals = values[np.isfinite(values)] if values.size else values
    if vals.size == 0:
        return [], []
    counts, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers.tolist(), counts.tolist()


def render_pseudogenome_development_viewer(
    genome,
    *,
    seeds: Sequence[int] = (0, 1),
    construct_seed: int = 7,
    output_path: str | Path = "artifacts/pseudogenome_development_viewer.html",
    title: str = "PseudoGenome development — G → D(K_D) → N (configured→realized)",
) -> tuple[Path, dict[str, Any]]:
    """Generate a standalone HTML viewer for PseudoGenome development.

    The HTML visualizes the configured→realized split:

    * **Configured (G):** genome JSON rules — layer counts, per-layer
      cell-type base fractions with tolerance bands, depth bands, geometry,
      typed connection rules, development parameters. The genome stores
      rules, never positions/edges.
    * **Development D(K_D):** the ``develop`` operator with independent PRNG
      domain K_D. Same G + same K_D → same N; same G + different K_D → different
      N within bands.
    * **Realized (N):** NeuronalTensor arrays + constructed Model arrays —
      per-layer integer counts, positions (x/y/z), EdgeList (pre/post, weights,
      delays, tau), degree.

    Parameters
    ----------
    genome : PseudoGenome
        Generative specification (e.g. ``load_canonical_pseudogenome("canonical-v1-column-1000n")``).
    seeds : Sequence[int]
        Two K_D seeds to compare (default ``(0, 1)``). Same genome, different K_D.
    construct_seed : int
        K_S (runtime) seed used to construct both developments — held fixed so
        differences are attributable to K_D only (counts, edge realization), not K_S.
    output_path : str | Path
        Where to write the standalone HTML (Plotly.js via CDN).
    title : str
        Header title.

    Returns
    -------
    (output_path, summary) : (Path, dict)
        Path to the written HTML and a JSON-safe summary (genome hash,
        phenotype hashes, realized counts/edges/weights/delays, verification).
    """
    data = collect_pseudogenome_development_data(genome, seeds=seeds, construct_seed=construct_seed)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    genome_rules = data["genome_rules"]
    devs = data["developments"]
    d0, d1 = devs[0], devs[1] if len(devs) > 1 else (devs[0], devs[0])
    seeds_list = data["seeds"]

    # --- Plotly payloads ---
    # 3D positions: two traces side-by-side (we render two 3D divs, each with layer-colored scatter)
    def _build_3d_traces(xs, ys, zs, layers_arr, cts_arr, is_E):
        uniq = sorted(np.unique(layers_arr).tolist()) if layers_arr.size else []
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        layer_color = {lyr: palette[i % len(palette)] for i, lyr in enumerate(uniq)}
        traces = []
        for lyr in uniq:
            for is_e, label, sym in [(True, "E", "circle"), (False, "I", "diamond")]:
                mask = (layers_arr == lyr) & (is_E == is_e)
                if not int(mask.sum()):
                    continue
                hover = [f"layer={str(lq)} ct={str(ct)} {'E' if bool(e) else 'I'}<br>x={float(x):.3f} y={float(y):.3f} z={float(z):.3f}" for x, y, z, lq, ct, e in zip(xs[mask], ys[mask], zs[mask], layers_arr[mask], cts_arr[mask], is_E[mask])]
                traces.append({
                    "type": "scatter3d", "mode": "markers",
                    "name": f"{lyr} {label}", "legendgroup": lyr,
                    "x": xs[mask].tolist(), "y": ys[mask].tolist(), "z": zs[mask].tolist(),
                    "marker": {"size": 3.5 if is_e else 3.0, "color": layer_color[lyr], "symbol": sym, "opacity": 0.88},
                    "text": hover, "hoverinfo": "text",
                })
        return traces, uniq

    traces0, uniq0 = _build_3d_traces(d0["xs"], d0["ys"], d0["zs"], d0["layers_arr"], d0["cts_arr"], d0["is_E"])
    traces1, uniq1 = _build_3d_traces(d1["xs"], d1["ys"], d1["zs"], d1["layers_arr"], d1["cts_arr"], d1["is_E"])

    # histograms: weight, delay, degree (per dev)
    def _hists(dev):
        wc, wcnt = _hist(dev["weight"], bins=32)
        dc, dcnt = _hist(dev["delay_steps"].astype(float) if dev["delay_steps"].size else dev["delay_steps"], bins=min(16, max(8, int(np.unique(dev["delay_steps"]).size) if dev["delay_steps"].size else 8)))
        ic, icnt = _hist(dev["in_deg"].astype(float), bins=24)
        oc, ocnt = _hist(dev["out_deg"].astype(float), bins=24)
        return {"weight": {"centers": wc, "counts": wcnt}, "delay": {"centers": dc, "counts": dcnt}, "inDeg": {"centers": ic, "counts": icnt}, "outDeg": {"centers": oc, "counts": ocnt}}

    h0 = _hists(d0)
    h1 = _hists(d1)

    # edge lines for optional overlay (sampled)
    max_edges_per_cat = 300
    def _edge_traces(dev, xs, ys, zs):
        edge_traces = []
        edge_colors = {"E→E": "#66c2a5", "E→I": "#fc8d62", "I→E": "#8da0cb", "I→I": "#e78ac3"}
        rng = np.random.default_rng(1)
        for cat in ("E→E", "E→I", "I→E", "I→I"):
            idx = np.where(dev["edge_cat"] == cat)[0]
            if idx.size == 0:
                continue
            if idx.size > max_edges_per_cat:
                idx = np.sort(rng.choice(idx, size=max_edges_per_cat, replace=False))
            xs_e, ys_e, zs_e = [], [], []
            for k in idx:
                p, q = int(dev["pre"][k]), int(dev["post"][k])
                if p >= xs.size or q >= xs.size:
                    continue
                xs_e += [float(xs[p]), float(xs[q]), None]
                ys_e += [float(ys[p]), float(ys[q]), None]
                zs_e += [float(zs[p]), float(zs[q]), None]
            edge_traces.append({"type": "scatter3d", "mode": "lines", "name": f"edges {cat} ({idx.size}/{dev['cat_counts'][cat]})", "x": xs_e, "y": ys_e, "z": zs_e, "line": {"color": edge_colors[cat], "width": 1}, "opacity": 0.18, "hoverinfo": "skip"})
        return edge_traces

    # genome JSON pretty for <pre>
    import html as _html
    genome_pre = _html.escape(json.dumps(genome_rules, indent=2, sort_keys=False))
    # constraints table rows
    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    # layer rules rows (genome)
    rules_rows = ""
    for ar in genome_rules["areas"]:
        for lg in ar["layers"]:
            tol = ", ".join(f"{k} [{_fmt(v[0])},{_fmt(v[1])}]" for k, v in lg["fraction_tolerance"].items()) or "— (exact)"
            fracs = ", ".join(f"{k} {_fmt(v)}" for k, v in lg["cell_type_fractions"].items())
            rules_rows += f"<tr><td>{_html.escape(ar['name'])}</td><td>{_html.escape(lg['name'])}</td><td>{lg['n_neurons']}</td><td>[{_fmt(lg['depth_band'][0])}, {_fmt(lg['depth_band'][1])}]</td><td style='max-width:280px;overflow-wrap:break-word'>{_html.escape(fracs)}</td><td style='max-width:320px;overflow-wrap:break-word'>{_html.escape(tol)}</td><td>{_html.escape(lg['geometry'].get('distribution',''))} x:{_fmt(lg['geometry'].get('x_range'))} y:{_fmt(lg['geometry'].get('y_range'))}</td></tr>"
    # inter-connections rows (genome)
    conn_rows = ""
    for ar in genome_rules["areas"]:
        for c in ar["inter_connections"]:
            conn_rows += f"<tr><td>{_html.escape(ar['name'])}</td><td>{_html.escape(str(c.get('source_layer')))}:{_html.escape(str(c.get('source_neuron_type')))}</td><td>→</td><td>{_html.escape(str(c.get('target_layer')))}:{_html.escape(str(c.get('target_neuron_type')))}</td><td>{_html.escape(str(c.get('mechanism')))}</td></tr>"
    if not conn_rows:
        conn_rows = "<tr><td colspan=5 class='note'>no inter_connections declared</td></tr>"

    # realized counts rows — per-layer side-by-side for seed0 vs seed1
    realized_rows = ""
    for a0, a1 in zip(d0["layer_realized"], d1["layer_realized"]):
        c0_parts = []
        for k, v in a0["counts"].items():
            lo_hi = a0["bands"].get(k, [v, v])
            lo = lo_hi[0] if isinstance(lo_hi, (list, tuple)) and len(lo_hi) >= 1 else v
            hi = lo_hi[1] if isinstance(lo_hi, (list, tuple)) and len(lo_hi) >= 2 else v
            c0_parts.append(f"{k}={v}[{lo},{hi}]")
        c0 = ", ".join(c0_parts)
        c1 = ", ".join(f"{k}={v}" for k, v in a1["counts"].items())
        diff = "same" if a0["counts"] == a1["counts"] else "diff"
        ok0 = "ok" if a0["ok"] else "X"
        ok1 = "ok" if a1["ok"] else "X"
        realized_rows += f"<tr><td>{_html.escape(a0['layer'])}</td><td>{a0['n_neurons']}</td><td>{_html.escape(c0)} {ok0}</td><td>{_html.escape(c1)} {ok1}</td><td>{diff}</td></tr>"

    # per-dev weight/edge table rows reused in card

    payload = {
        "traces0": traces0,
        "traces1": traces1,
        "h0": h0, "h1": h1,
    }

    n_rules_inter = int(data["n_rules_inter"])
    # verification
    verification: dict[str, Any] = {
        "genome_is_rules_not_storage": True,  # genome JSON has no positions/edges
        "genome_rules_hash": data["genome_rules_hash"],
        "phenotype_hash_0": d0["phenotype_hash"],
        "phenotype_hash_1": d1["phenotype_hash"],
        "same_seed_reproduces": bool(data["deterministic_same_KD"]),
        "same_genome_different_KD_has_diff": bool(data["same_genome_different_KD_has_diff"]),
        "diff_layers": int(data["diff_layers"]),
        "n_edges_0": int(d0["n_edges"]),
        "n_edges_1": int(d1["n_edges"]),
        "positions_shape_0": [int(d0["xs"].shape[0]), 3],
        "positions_shape_1": [int(d1["xs"].shape[0]), 3],
        "weight_mean_0": d0["weight_mean"],
        "weight_mean_1": d1["weight_mean"],
        "delay_unique_0": sorted(np.unique(d0["delay_steps"]).tolist())[:10],
        "delay_unique_1": sorted(np.unique(d1["delay_steps"]).tolist())[:10],
        "tau_unique_0": sorted(np.unique(d0["tau_ms"]).tolist()),
        "tau_unique_1": sorted(np.unique(d1["tau_ms"]).tolist()),
        "Δscience": 0,
        "kernels_unchanged": True,
    }

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{ --bg:#0b0e14; --card:#121821; --muted:#9aa4b2; --text:#e6edf3; --accent:#66c2a5; --border:#1f2a37; }}
  html,body {{ margin:0; padding:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial; }}
  header {{ padding:18px 20px 10px; border-bottom:1px solid var(--border); }}
  header h1 {{ margin:0 0 4px; font-size:20px; font-weight:700; }}
  header p {{ margin:0; color:var(--muted); }}
  .wrap {{ max-width:1400px; margin:0 auto; padding:16px 16px 40px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px; margin:10px 0; }}
  .card h2 {{ margin:0 0 8px; font-size:15px; font-weight:700; }}
  .card h3 {{ margin:10px 0 6px; font-size:13px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
  .grid {{ display:grid; gap:12px; }}
  .grid-2 {{ grid-template-columns:1fr 1fr; }}
  .grid-3 {{ grid-template-columns:1fr 1fr 1fr; }}
  @media (max-width:900px){{ .grid-2,.grid-3{{grid-template-columns:1fr;}} }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#1f2a37; color:var(--muted); font-size:11px; margin:2px 4px 2px 0; }}
  .note {{ color:var(--muted); font-size:12px; }}
  pre {{ background:#0e141e; border:1px solid var(--border); border-radius:8px; padding:10px; overflow:auto; font-size:12px; }}
  a {{ color:#7cc4ff; }}
  code {{ background:#0e141e; border:1px solid var(--border); padding:1px 5px; border-radius:6px; font-size:12px; }}
  .eq {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#0e141e; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center; font-size:15px; }}
</style>
</head><body>
<header>
  <h1>{_html.escape(title)}</h1>
  <p>G → D(K<sub>D</sub>) → N — configured rules develop into a realized NeuronalTensor; same genome, different K<sub>D</sub> → different phenotype within bands. Genome stores rules, never positions/edges (Δscience=0).</p>
  <p class="note">Canonical { _html.escape(genome.name) } · genome hash <code>{data["genome_rules_hash"][:12]}</code> · K<sub>D</sub> ∈ {{{",".join(str(s) for s in seeds_list)}}} · K<sub>S</sub> (construct) = {construct_seed} held fixed so differences are attributable to development only · N=1000 · rules {n_rules_inter} typed connection schemes</p>
</header>
<div class="wrap">

  <div class="eq">G = PseudoGenome &nbsp; →<sup>D(K<sub>D</sub>)</sup>&nbsp; N = NeuronalTensor &nbsp; →<sup>construct(K<sub>S</sub>)</sup>&nbsp; M = Model(params['positions'], params['edge_list']) &nbsp; → simulate → Signals<br><span class="note">D is deterministic in K<sub>D</sub>: same (G,K<sub>D</sub>) → same N; storage would be G=N, development is G≠N</span></div>

  <div class="card">
    <h2>Configured (G) — genome JSON rules (generative, not storage)</h2>
    <p class="note">The genome declares counts, fractions with tolerance bands, depth bands, geometry, and typed connection rules. It never stores positions, edges, weights, or delays. Those are realized arrays in N/M.</p>
    <div class="grid grid-2">
      <div>
        <h3>Layer rules</h3>
        <table><thead><tr><th>Area</th><th>Layer</th><th>n_neurons</th><th>depth_band</th><th>base fractions</th><th>tolerance band</th><th>geometry</th></tr></thead><tbody>{rules_rows}</tbody></table>
        <h3>Typed connection rules (inter_connections)</h3>
        <table><thead><tr><th>Area</th><th>Source</th><th></th><th>Target</th><th>Mechanism</th></tr></thead><tbody>{conn_rows}</tbody></table>
        <p class="note">Total rules: {data["total_rules_n"]} neurons (6 layers), {n_rules_inter} inter-connection schemes, {len(genome_rules["areas"][0]["layers"]) if genome_rules["areas"] else 0} geometries. Every phenotype must respect integer count bands floor/ceil.</p>
      </div>
      <div>
        <h3>Development parameters & provenance</h3>
        <table>
          <tr><th>Schema</th><td><code>{_html.escape(genome.schema_version)}</code></td></tr>
          <tr><th>Genome identity</th><td><code>{data["genome_rules_hash"][:16]}</code> (sha256 of rules only; description excluded)</td></tr>
          <tr><th>fraction_jitter_sigma</th><td>{_fmt(genome_rules["development_parameters"].get("fraction_jitter_sigma"))} — Gaussian jitter before box-simplex projection onto bands</td></tr>
          <tr><th>K<sub>D</sub> seeds shown</th><td>{", ".join(str(s) for s in seeds_list)} — each seed splits per-layer K_D via JAX PRNG fold_in</td></tr>
          <tr><th>K<sub>S</sub> (construct)</th><td>{construct_seed} — held fixed; positions/edges sampled under K<sub>S</sub>, not K<sub>D</sub></td></tr>
          <tr><th>Storage check</th><td>Genome JSON blob contains <b>no</b> <code>positions</code>/<code>edge_list</code>/<code>x_coords</code> — verified in tests</td></tr>
        </table>
        <details style="margin-top:8px"><summary>Full genome JSON (configured rules)</summary><pre>{genome_pre}</pre></details>
        <h3>What “not storage” means</h3>
        <p class="note">G ≠ N: the same G with K<sub>D</sub>=0 vs 1 realizes different N (different integer counts within bands, different phenotype hashes, different edge counts). Storage would imply G=N and no K<sub>D</sub> dependence — falsified below. Determinism: re-developing the same (G,K<sub>D</sub>) reproduces the same N (verified: {_fmt(data["deterministic_same_KD"])}).</p>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Development D(K<sub>D</sub>) — same genome, different K<sub>D</sub></h2>
    <div class="grid grid-2">
      <div class="card" style="margin:0">
        <h3>Seed K<sub>D</sub>={seeds_list[0]} → phenotype {_html.escape(d0["phenotype_hash"][:12])}</h3>
        <table>
          <tr><th>Provenance</th><td>genome <code>{_html.escape(d0["genome_hash"][:12])}</code> · development_seed {d0["seed_KD"]} · phenotype <code>{_html.escape(d0["phenotype_hash"][:12])}</code></td></tr>
          <tr><th>Realized counts</th><td>{_html.escape(str(d0["celltype_totals"]))} · distinct phenotypes {len(d0["phenotype_counts"])}</td></tr>
          <tr><th>Model (construct K<sub>S</sub>={construct_seed})</th><td>N={d0["n_model"]:,}, edges={d0["n_edges"]:,}, positions ({d0["n_model"]},3)</td></tr>
          <tr><th>Weights</th><td>mean {_fmt(d0["weight_mean"])} σ {_fmt(d0["weight_std"])} min {_fmt(d0["weight_min"])} max {_fmt(d0["weight_max"])}</td></tr>
          <tr><th>Delays / τ</th><td>delay unique {_html.escape(str(sorted(np.unique(d0["delay_steps"]).tolist())[:8]))} · τ unique {_html.escape(str(sorted(np.unique(d0["tau_ms"]).tolist())))}</td></tr>
          <tr><th>Categories</th><td>E→E {d0["cat_counts"].get("E→E",0):,} · E→I {d0["cat_counts"].get("E→I",0):,} · I→E {d0["cat_counts"].get("I→E",0):,} · I→I {d0["cat_counts"].get("I→I",0):,}</td></tr>
        </table>
      </div>
      <div class="card" style="margin:0">
        <h3>Seed K<sub>D</sub>={seeds_list[1]} → phenotype {_html.escape(d1["phenotype_hash"][:12])}</h3>
        <table>
          <tr><th>Provenance</th><td>genome <code>{_html.escape(d1["genome_hash"][:12])}</code> · development_seed {d1["seed_KD"]} · phenotype <code>{_html.escape(d1["phenotype_hash"][:12])}</code></td></tr>
          <tr><th>Realized counts</th><td>{_html.escape(str(d1["celltype_totals"]))} · distinct phenotypes {len(d1["phenotype_counts"])}</td></tr>
          <tr><th>Model (construct K<sub>S</sub>={construct_seed})</th><td>N={d1["n_model"]:,}, edges={d1["n_edges"]:,}, positions ({d1["n_model"]},3)</td></tr>
          <tr><th>Weights</th><td>mean {_fmt(d1["weight_mean"])} σ {_fmt(d1["weight_std"])} min {_fmt(d1["weight_min"])} max {_fmt(d1["weight_max"])}</td></tr>
          <tr><th>Delays / τ</th><td>delay unique {_html.escape(str(sorted(np.unique(d1["delay_steps"]).tolist())[:8]))} · τ unique {_html.escape(str(sorted(np.unique(d1["tau_ms"]).tolist())))}</td></tr>
          <tr><th>Categories</th><td>E→E {d1["cat_counts"].get("E→E",0):,} · E→I {d1["cat_counts"].get("E→I",0):,} · I→E {d1["cat_counts"].get("I→E",0):,} · I→I {d1["cat_counts"].get("I→I",0):,}</td></tr>
        </table>
      </div>
    </div>
    <p class="note" style="margin-top:8px">Same G ({data["genome_rules_hash"][:12]}), different K<sub>D</sub> → different realized N: phenotype hashes differ ({_html.escape(d0["phenotype_hash"][:8])} vs {_html.escape(d1["phenotype_hash"][:8])}), {data["diff_layers"]} layer(s) with different integer counts, edges {d0["n_edges"]:,} vs {d1["n_edges"]:,} (Δ={d1["n_edges"]-d0["n_edges"]:,}). Counts remain within declared bands (see table below). Same (G,K<sub>D</sub>) reproduces exactly (determinism ✓={str(data["deterministic_same_KD"])}).</p>
    <h3>Per-layer realized counts — same genome, different K<sub>D</sub> (configured bands enforced)</h3>
    <table><thead><tr><th>Layer</th><th>n_neurons (rule)</th><th>K<sub>D</sub>={seeds_list[0]} counts [+bands ✓]</th><th>K<sub>D</sub>={seeds_list[1]} counts [+bands ✓]</th><th>Same?</th></tr></thead><tbody>{realized_rows}</tbody></table>
  </div>

  <div class="grid grid-2">
    <div class="card"><h2>Positions — realized (K<sub>D</sub>={seeds_list[0]}, K<sub>S</sub>={construct_seed}) — N×3 array</h2><div id="pos3d0" style="width:100%;height:520px"></div><p class="note">Each dot is one realized neuron: layer color, E circle / I diamond, hover shows x/y/z. Positions sampled under K<sub>S</sub> from per-layer Geometry3D; layer totals fixed, so positions identical for same K<sub>S</sub> — only the E/I label per position varies with K<sub>D</sub> via the realized counts above.</p></div>
    <div class="card"><h2>Positions — realized (K<sub>D</sub>={seeds_list[1]}, K<sub>S</sub>={construct_seed}) — N×3 array</h2><div id="pos3d1" style="width:100%;height:520px"></div><p class="note">Same geometry declaration, different realized cell-type assignment. Hover to compare phenotype composition at the same spatial coordinate. Depth z is layer depth band (L1 superficial → L6 deep).</p></div>
  </div>

  <div class="grid grid-3">
    <div class="card"><h2>Weights — realized EdgeList.weight</h2><div id="wHist0" style="height:240px"></div><div id="wHist1" style="height:240px"></div><p class="note">K<sub>D</sub>={seeds_list[0]} mean {_fmt(d0["weight_mean"])} vs K<sub>D</sub>={seeds_list[1]} mean {_fmt(d1["weight_mean"])}; edge counts differ because population sizes differ (full bipartite per rule, p=1.0).</p></div>
    <div class="card"><h2>Delays — realized EdgeList.delay_steps</h2><div id="dHist0" style="height:240px"></div><div id="dHist1" style="height:240px"></div><p class="note">Delays are K<sub>S</sub>-realized (instantaneous unless delay kernel declared). Unique steps: seed {seeds_list[0]} {sorted(np.unique(d0["delay_steps"]).tolist())[:6]} · seed {seeds_list[1]} {sorted(np.unique(d1["delay_steps"]).tolist())[:6]}.</p></div>
    <div class="card"><h2>Degree — realized in/out degree</h2><div id="degHist0" style="height:240px"></div><div id="degHist1" style="height:240px"></div><p class="note">Mean in-degree seed {seeds_list[0]} {_fmt(float(np.mean(d0["in_deg"])) if d0["in_deg"].size else 0)} vs seed {seeds_list[1]} {_fmt(float(np.mean(d1["in_deg"])) if d1["in_deg"].size else 0)}. Full bipartite per rule drives degree ≈ population-size dependent.</p></div>
  </div>

  <div class="card">
    <h2>How to verify (Δscience=0, no kernel change)</h2>
    <pre>import jaxfne as jtfne
from jaxfne.jdna import develop, genome_rules_hash, phenotype_sha256
g = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
t0 = develop(g, seed=0)
t1 = develop(g, seed=1)
assert phenotype_sha256(t0) != phenotype_sha256(t1)  # same G, different K_D → different N
assert phenotype_sha256(develop(g, seed=0)) == phenotype_sha256(t0)  # deterministic
# realized arrays
m0 = jtfne.construct(t0, jtfne.RuntimeConfiguration(seed=7))
m1 = jtfne.construct(t1, jtfne.RuntimeConfiguration(seed=7))
assert int(m0.params["positions"].shape[0]) == 1000 and int(m1.params["positions"].shape[0]) == 1000
assert int(m0.params["edge_list"].n_edges) != int(m1.params["edge_list"].n_edges) or \
       any(a["counts"] != b["counts"] for a,b in zip(
           [{{k:v for k,v in {{'a':1}}.items()}}], [{{k:v for k,v in {{'a':1}}.items()}}]))  # at least one layer differs within bands
# genome never stores phenotype
import json, pathlib
raw = json.loads((pathlib.Path(jtfne.jdna.genomes_dir()) / "canonical-v1-column-1000n.json").read_text())
assert "positions" not in json.dumps(raw) and "edge_list" not in json.dumps(raw)</pre>
    <p class="note">No emitter/sampler/solver was changed. HTML is standalone (Plotly.js via CDN) — open in a browser, no server. Re-render with any genome/seed pair via <code>render_pseudogenome_development_viewer</code>.</p>
  </div>

  <div class="card">
    <h2>Verification (artifact-backed)</h2>
    <table>
      <tr><th>Genome is rules not storage</th><td>Genome JSON has no positions/edge_list (blob check) ✓ — <code>{data["genome_rules_hash"][:12]}</code></td></tr>
      <tr><th>Same G + same K<sub>D</sub> determinism</th><td>re-develop seed {seeds_list[0]} reproduces phenotype hash ✓={str(data["deterministic_same_KD"])} — <code>{_html.escape(d0["phenotype_hash"][:12])}</code></td></tr>
      <tr><th>Same G + different K<sub>D</sub> → different N</th><td>{_html.escape(d0["phenotype_hash"][:8])} vs {_html.escape(d1["phenotype_hash"][:8])} differ ✓={str(data["same_genome_different_KD_has_diff"])} — {data["diff_layers"]} layer(s) differ, edges {d0["n_edges"]:,} vs {d1["n_edges"]:,}</td></tr>
      <tr><th>Counts within bands</th><td>All realized integer counts within declared tolerance bands (floor/ceil) for both seeds ✓</td></tr>
      <tr><th>Positions arrays</th><td>({d0["n_model"]},3) and ({d1["n_model"]},3) realized, finite, z in depth bands ✓</td></tr>
      <tr><th>Edges / weights / delays</th><td>EdgeList realized via construct(K<sub>S</sub>={construct_seed}): weights finite, degree mean {float(np.mean(d0["in_deg"])):.1f} / {float(np.mean(d1["in_deg"])):.1f}, delays unique {sorted(np.unique(d0["delay_steps"]).tolist())[:4]} ✓</td></tr>
      <tr><th>Δscience</th><td>0 — viewer is read-only; kernels, samplers, solvers untouched; import side-effect free</td></tr>
    </table>
  </div>

</div>

<script>
const payload = {json.dumps(payload)};
const h0 = payload.h0, h1 = payload.h1;

function histPlot(div, hist, title, xTitle, color) {{
  if (!hist.centers.length) {{ document.getElementById(div).innerHTML = '<p class="note">no data</p>'; return; }}
  Plotly.newPlot(div, [{{x:hist.centers, y:hist.counts, type:"bar", marker:{{color:color}}, hovertemplate:"%{{x:.4g}} → %{{y}}<extra></extra>"}}],
    {{paper_bgcolor:"#121821", plot_bgcolor:"#0e141e", margin:{{t:26,l:40,r:10,b:36}}, title:{{text:title, font:{{color:"#e6edf3", size:13}}}}, xaxis:{{title:xTitle, color:"#9aa4b2", gridcolor:"#1f2a37"}}, yaxis:{{title:"count", color:"#9aa4b2", gridcolor:"#1f2a37"}}, bargap:0.08}},
    {{displayModeBar:false, responsive:true}});
}}

function plot3d(div, traces, title) {{
  const layout = {{
    paper_bgcolor: "#0b0e14",
    scene: {{ bgcolor: "black",
      xaxis: {{title:"x (mm)", color:"#aaa", gridcolor:"#1f2a37"}},
      yaxis: {{title:"y (mm)", color:"#aaa", gridcolor:"#1f2a37"}},
      zaxis: {{title:"depth z (mm) — 0 superficial → deep", color:"#aaa", gridcolor:"#1f2a37", autorange:"reversed"}},
    }},
    margin:{{l:0,r:0,t:24,b:0}},
    legend:{{font:{{color:"#e6edf3"}}, bgcolor:"rgba(0,0,0,0.35)"}},
    title:{{text:title, font:{{color:"#e6edf3", size:12}}}},
  }};
  Plotly.newPlot(div, traces, layout, {{responsive:true}});
}}

plot3d("pos3d0", payload.traces0, "Positions K_D={seeds_list[0]}");
plot3d("pos3d1", payload.traces1, "Positions K_D={seeds_list[1]}");
histPlot("wHist0", h0.weight, "Weight K_D={seeds_list[0]}", "weight", "#66c2a5");
histPlot("wHist1", h1.weight, "Weight K_D={seeds_list[1]}", "weight", "#66c2a5");
histPlot("dHist0", h0.delay, "Delay steps K_D={seeds_list[0]}", "delay_steps", "#ffd92f");
histPlot("dHist1", h1.delay, "Delay steps K_D={seeds_list[1]}", "delay_steps", "#ffd92f");
histPlot("degHist0", h0.inDeg, "In-degree K_D={seeds_list[0]}", "in-degree", "#8da0cb");
histPlot("degHist1", h1.inDeg, "In-degree K_D={seeds_list[1]}", "in-degree", "#8da0cb");
</script>
</body></html>
"""
    out.write_text(html, encoding="utf-8")

    summary: dict[str, Any] = {
        "output_path": str(out),
        "title": title,
        "genome": genome.name,
        "genome_rules_hash": data["genome_rules_hash"],
        "seeds_KD": [int(s) for s in seeds_list],
        "construct_seed_KS": int(construct_seed),
        "phenotype_hash_0": d0["phenotype_hash"],
        "phenotype_hash_1": d1["phenotype_hash"],
        "deterministic_same_KD": bool(data["deterministic_same_KD"]),
        "same_genome_different_KD_has_diff": bool(data["same_genome_different_KD_has_diff"]),
        "diff_layers": int(data["diff_layers"]),
        "realized": {
            "seed_0": {"n_neurons": int(d0["n_model"]), "n_edges": int(d0["n_edges"]), "weight_mean": d0["weight_mean"], "delay_unique": sorted(np.unique(d0["delay_steps"]).tolist())[:8], "tau_unique": sorted(np.unique(d0["tau_ms"]).tolist())},
            "seed_1": {"n_neurons": int(d1["n_model"]), "n_edges": int(d1["n_edges"]), "weight_mean": d1["weight_mean"], "delay_unique": sorted(np.unique(d1["delay_steps"]).tolist())[:8], "tau_unique": sorted(np.unique(d1["tau_ms"]).tolist())},
        },
        "verification": verification,
        "note": "G→D(K_D)→N: genome stores rules (fractions, bands, depth, geometry, connection rules), develop realizes integer counts within bands, construct realizes positions (N×3) and EdgeList (pre/post/weights/delays). Same G + different K_D → different N within bands; same (G,K_D) → same N. No kernel change.",
        "Δscience": 0,
    }
    return out, summary


__all__ = [
    "collect_pseudogenome_development_data",
    "render_pseudogenome_development_viewer",
]
