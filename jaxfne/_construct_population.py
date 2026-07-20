"""Suite No. 2 population construction: per-layer/per-area neuron sampling,
cell-type composition, and within-area connectivity weight building.

Split out of ``jaxfne/_construct.py`` (Phase 2 defragmentation, 2026-07-20,
part of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1). ``jaxfne/_construct.py``
re-exports every symbol here for backward compatibility -- import from
``jaxfne.core`` or ``jaxfne._construct``, not this module, unless you are
working on this split itself.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from .emitters import EdgeList, IzhikevichParams, izhikevich_params_from_labels
from ._config import Configuration, _counts_from_fractions
from ._construct_connectivity import _interarea_W


_SUITE2_LAYER_FRACTIONS = {
    "L1": (0.00, 0.10),
    "L2": (0.10, 0.25),
    "L3": (0.25, 0.45),
    "L4": (0.45, 0.55),
    "L5": (0.55, 0.85),
    "L6": (0.85, 1.00),
}

_SUITE2_LAYER_CELL_TYPES_V1 = {
    "L1": {"E": 0.75, "PV": 0.00, "SST": 0.00, "VIP": 0.25},
    "L2": {"E": 0.75, "PV": 0.05, "SST": 0.05, "VIP": 0.15},
    "L3": {"E": 0.75, "PV": 0.10, "SST": 0.10, "VIP": 0.05},
    "L4": {"E": 0.25, "PV": 0.45, "SST": 0.15, "VIP": 0.15},
    "L5": {"E": 0.15, "PV": 0.25, "SST": 0.30, "VIP": 0.30},
    "L6": {"E": 0.10, "PV": 0.20, "SST": 0.20, "VIP": 0.50},
}

_SUITE2_LAYER_CELL_TYPES_V4 = {
    "L1": {"E": 0.70, "PV": 0.00, "SST": 0.00, "VIP": 0.30},
    "L2": {"E": 0.70, "PV": 0.08, "SST": 0.06, "VIP": 0.16},
    "L3": {"E": 0.70, "PV": 0.12, "SST": 0.12, "VIP": 0.06},
    "L4": {"E": 0.30, "PV": 0.40, "SST": 0.15, "VIP": 0.15},
    "L5": {"E": 0.20, "PV": 0.25, "SST": 0.30, "VIP": 0.25},
    "L6": {"E": 0.15, "PV": 0.20, "SST": 0.25, "VIP": 0.40},
}

_SUITE2_PROXY_MODES = (
    "spikes", "V_m", "source", "LFP", "CSD", "EEG-proxy", "MEG-proxy", "EMM-proxy"
)


def _default_layer_cell_types() -> dict[str, dict[str, float]]:
    return {k: dict(v) for k, v in _SUITE2_LAYER_CELL_TYPES_V1.items()}


def _layer_ranges_for(layers: Sequence[str], metadata: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    declared = metadata.get("layer_fractions") or _SUITE2_LAYER_FRACTIONS
    out: dict[str, tuple[float, float]] = {}
    missing = [str(layer) for layer in layers if str(layer) not in declared]
    if missing:
        n = len(layers)
        for idx, layer in enumerate(layers):
            out[str(layer)] = (idx / max(n, 1), (idx + 1) / max(n, 1))
        return out
    for layer in layers:
        z0, z1 = declared[str(layer)]
        out[str(layer)] = (float(z0), float(z1))
    return out


def _area_layer_cell_type_map(metadata: Mapping[str, Any], area: str) -> dict[str, dict[str, float]]:
    per_area = metadata.get("area_layer_cell_types", {}) or {}
    if area in per_area:
        return {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in per_area[area].items()}
    base = metadata.get("layer_cell_types", None)
    if base:
        return {str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in base.items()}
    # A flat cell_types() declaration (metadata["cell_types"]) was previously
    # invisible here -- Configuration.cell_types() writes that key, but this
    # resolver only ever checked area_layer_cell_types/layer_cell_types, so a
    # bare .cell_types({...}) after .population()/.geometry() silently fell
    # through to the hardcoded _default_layer_cell_types() table below
    # instead (confirmed regression: {"E":0.5,"PV":0.5} at N=2 on layer "L4"
    # produced ["PV","VIP"], not ["E","PV"], because L4's hardcoded default is
    # {"E":0.25,"PV":0.45,"SST":0.15,"VIP":0.15}). Broadcasting the flat
    # declaration to every layer here closes that gap without touching
    # cell_types()'s own write behavior.
    flat = metadata.get("cell_types", None)
    if flat:
        clean = {str(k): float(v) for k, v in flat.items()}
        layers = metadata.get("layer_fractions") or _SUITE2_LAYER_FRACTIONS
        return {str(layer): dict(clean) for layer in layers}
    return _default_layer_cell_types()


def _area_layer_count_frac(metadata: Mapping[str, Any], area: str) -> dict[str, float] | None:
    """Per-layer neuron *population* fractions (count budget, NOT thickness).

    These are set by ``Configuration.population(...)`` and decouple how many
    neurons a layer gets from how thick it is (e.g. a thin but dense L2 holding
    more neurons than a thick but sparse L5). Resolution order: per-area override
    (``area_layer_count_frac``) -> global ``layer_count_frac`` -> ``None`` (caller
    falls back to the historical thickness-proportional allocation).
    """
    per_area = metadata.get("area_layer_count_frac", {}) or {}
    if area in per_area:
        return {str(k): float(v) for k, v in per_area[area].items()}
    base = metadata.get("layer_count_frac", None)
    if base:
        return {str(k): float(v) for k, v in base.items()}
    return None


def _neuron_population_from_config(cfg: "Configuration", *, dtype: str = "float32") -> tuple[IzhikevichParams, jax.Array, dict[str, Any]]:
    """Build explicit Suite No. 2 neuron metadata and reduced emitter arrays."""

    metadata = cfg.metadata
    net = cfg.networks[0]
    columns = [dict(c) for c in metadata.get("columns", [])]
    if not columns:
        columns = [{"name": str(net.get("name", "net1")), "layers": ["uniform_3d"], "n": int(net.get("n", 100)), "start_index": 0, "stop_index": int(net.get("n", 100))}]
    seed = int(metadata.get("seed", 0))
    uniform_3d = bool(metadata.get("uniform_3d", False))
    global_cell_types = {str(k): float(v) for k, v in net.get("cell_types", {"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}).items()}

    labels: list[str] = []
    layer_labels: list[str] = []
    area_labels: list[str] = []
    neuron_rows: list[dict[str, Any]] = []
    position_chunks: list[jax.Array] = []
    jdtype = jnp.float64 if dtype == "float64" and bool(jax.config.read("jax_enable_x64")) else jnp.float32
    neuron_id = 0

    for area_idx, column in enumerate(columns):
        area = str(column.get("name", f"area_{area_idx}"))
        n_col = int(column.get("n", 0))
        if n_col <= 0:
            continue
        layers = [str(x) for x in column.get("layers", ["uniform_3d"])] or ["uniform_3d"]
        key = jax.random.PRNGKey(seed + 1009 * (area_idx + 1))

        if uniform_3d or layers == ["uniform_3d"]:
            counts_by_type = _counts_from_fractions(n_col, global_cell_types)
            col_labels: list[str] = []
            for cell_type, count in counts_by_type.items():
                col_labels.extend([cell_type] * count)
            col_labels = col_labels[:n_col] + ["E"] * max(0, n_col - len(col_labels))
            x_key, y_key, z_key = jax.random.split(key, 3)
            radius = float(metadata.get("column_radius_mm", 0.25))
            height = float(metadata.get("column_height_mm", 1.60))
            x = jax.random.uniform(x_key, (n_col,), minval=-radius, maxval=radius, dtype=jdtype) + jnp.asarray(area_idx * 2.0, dtype=jdtype)
            y = jax.random.uniform(y_key, (n_col,), minval=-radius, maxval=radius, dtype=jdtype)
            z = jax.random.uniform(z_key, (n_col,), minval=0.0, maxval=height, dtype=jdtype)
            position_chunks.append(jnp.stack([x, y, z], axis=1))
            for local_idx, cell_type in enumerate(col_labels[:n_col]):
                labels.append(cell_type)
                layer_labels.append("uniform_3d")
                area_labels.append(area)
                neuron_rows.append({"neuron_id": neuron_id, "area": area, "layer": "uniform_3d", "cell_type": cell_type, "x": float(x[local_idx]), "y": float(y[local_idx]), "z": float(z[local_idx])})
                neuron_id += 1
            continue

        layer_ranges = _layer_ranges_for(layers, metadata)
        count_frac = _area_layer_count_frac(metadata, area)
        if count_frac is not None and sum(count_frac.get(layer, 0.0) for layer in layers) > 0.0:
            # Population-fraction allocation (decoupled from thickness).
            alloc = {layer: float(count_frac.get(layer, 0.0)) for layer in layers}
        else:
            # Historical fallback: allocate neurons proportional to layer thickness.
            alloc = {layer: max(0.0, layer_ranges[layer][1] - layer_ranges[layer][0]) for layer in layers}
        layer_counts = _counts_from_fractions(n_col, alloc)
        area_layer_map = _area_layer_cell_type_map(metadata, area)
        for layer_idx, layer in enumerate(layers):
            n_layer = int(layer_counts.get(layer, 0))
            if n_layer <= 0:
                continue
            z0, z1 = layer_ranges[layer]
            type_fracs = area_layer_map.get(layer, global_cell_types)
            type_counts = _counts_from_fractions(n_layer, type_fracs)
            layer_cell_labels: list[str] = []
            for cell_type, count in type_counts.items():
                layer_cell_labels.extend([cell_type] * count)
            layer_cell_labels = layer_cell_labels[:n_layer] + ["E"] * max(0, n_layer - len(layer_cell_labels))
            x_key, y_key, z_key = jax.random.split(jax.random.fold_in(key, layer_idx), 3)
            radius = float(metadata.get("column_radius_mm", 0.25))
            x = jax.random.uniform(x_key, (n_layer,), minval=-radius, maxval=radius, dtype=jdtype) + jnp.asarray(area_idx * 2.0, dtype=jdtype)
            y = jax.random.uniform(y_key, (n_layer,), minval=-radius, maxval=radius, dtype=jdtype)
            z = jax.random.uniform(z_key, (n_layer,), minval=float(z0), maxval=float(z1), dtype=jdtype)
            position_chunks.append(jnp.stack([x, y, z], axis=1))
            for local_idx, cell_type in enumerate(layer_cell_labels[:n_layer]):
                labels.append(cell_type)
                layer_labels.append(layer)
                area_labels.append(area)
                neuron_rows.append({"neuron_id": neuron_id, "area": area, "layer": layer, "cell_type": cell_type, "x": float(x[local_idx]), "y": float(y[local_idx]), "z": float(z[local_idx])})
                neuron_id += 1

    # Apply baseline_drive_by_cell_type from drive specification if present
    drive_spec = metadata.get("drive", {})
    baseline_drive = drive_spec.get("baseline_drive_by_cell_type") if isinstance(drive_spec, dict) else None

    params = izhikevich_params_from_labels(
        labels,
        layer_labels=layer_labels,
        dtype=dtype,
        drive_overrides=baseline_drive,
        # _apply_connectivity (called a few lines below) unconditionally
        # replaces W in every branch (sparse-direct placeholder or a freshly
        # rebuilt dense matrix) -- the dense W built here would be pure waste,
        # confirmed as a real 40GB OOM at N=100,000 before this fix.
        build_dense_connectivity=False,
    )
    
    # Compile and apply declared cell_params overrides
    circuit = metadata.get("circuit", {})
    cell_param_decls = circuit.get("cell_params", [])
    if cell_param_decls:
        import numpy as np
        a_list = np.array(params.a)
        b_list = np.array(params.b)
        c_list = np.array(params.c)
        d_list = np.array(params.d)
        drive_list = np.array(params.drive)
        for decl in cell_param_decls:
            selector = decl.get("selector", {})
            param_overrides = decl.get("params", {})
            for i in range(len(labels)):
                match = True
                if "cell_type" in selector and labels[i] != selector["cell_type"]:
                    match = False
                if "layer" in selector and layer_labels[i] != selector["layer"]:
                    match = False
                if match:
                    if "a" in param_overrides:
                        a_list[i] = float(param_overrides["a"])
                    if "b" in param_overrides:
                        b_list[i] = float(param_overrides["b"])
                    if "c" in param_overrides:
                        c_list[i] = float(param_overrides["c"])
                    if "d" in param_overrides:
                        d_list[i] = float(param_overrides["d"])
                    if "drive" in param_overrides:
                        drive_list[i] = float(param_overrides["drive"])
        params = replace(
            params,
            a=jnp.asarray(a_list, dtype=jdtype),
            b=jnp.asarray(b_list, dtype=jdtype),
            c=jnp.asarray(c_list, dtype=jdtype),
            d=jnp.asarray(d_list, dtype=jdtype),
            drive=jnp.asarray(drive_list, dtype=jdtype),
        )

    positions = jnp.concatenate(position_chunks, axis=0) if position_chunks else jnp.zeros((0, 3), dtype=jdtype)
    params, _prebuilt_edges = _apply_connectivity(params, area_labels, layer_labels, labels, metadata, seed=seed, dtype=dtype)
    geometry_meta = {
        "neuron_rows": neuron_rows,
        "area_labels": area_labels,
        "layer_labels": layer_labels,
        "cell_type_labels": labels,
        "geometry_mode": metadata.get("geometry_mode", "declared_metadata_not_solved_3d_pde_grid"),
        "position_units": "mm_declared_metadata",
        "uniform_3d": bool(uniform_3d),
    }
    return params, positions, geometry_meta, _prebuilt_edges


# Above this neuron count, a dense all-to-all within-area weight matrix is flagged
# as an O(N^2) cost (memory + edges). Sparse connectivity (p_connect<1) avoids it.
_DENSE_CONNECTIVITY_WARN_N = 5000


# At/above this neuron count, a SPARSE within-area connectivity request (p_connect<1
# with no special-structure specs) is built as an edge list DIRECTLY, skipping the
# dense (n,n) weight matrix entirely. Below it, the dense-then-mask path is kept so
# small-N behavior (and the W-inspecting tests) stay exact.
_SPARSE_DIRECT_N = 5000


def _make_sparse_within_area_edges(area_labels, sign, n, *, within_gain, p_connect, seed, jdtype):
    """Build a within-area sparse EdgeList directly (no dense (n,n) materialization).

    Erdos-Renyi-style: each within-area ordered pair (j->i, i!=j) is included with
    probability ``p_connect``. Edge weight matches the dense path's connected-edge
    magnitude ``within_gain * U(0.25,1) * sign[pre] / (sqrt(n) * p_connect)`` so the
    expected total input is preserved. Edges are excitatory/inhibitory by presyn
    sign (receptor_index + tau follow make_edge_list_from_dense). Deterministic from
    ``seed``. Statistically equivalent to the dense path; NOT bit-identical.
    """
    import numpy as _np
    rng = _np.random.default_rng(int(seed) + 4242)
    codes = _np.unique(_np.asarray(area_labels), return_inverse=True)[1]
    sign_np = _np.asarray(sign, dtype=_np.float64)
    sqrt_n = float(max(n, 1)) ** 0.5
    p = float(p_connect)
    pre_chunks, post_chunks, w_chunks = [], [], []
    for code in _np.unique(codes):
        idx = _np.where(codes == code)[0]
        m = idx.size
        if m < 2:
            continue
        n_edges = int(rng.binomial(m * (m - 1), p))
        if n_edges == 0:
            continue
        pre_loc = rng.integers(0, m, n_edges)
        post_loc = rng.integers(0, m, n_edges)
        self_loop = pre_loc == post_loc
        while self_loop.any():                      # resample i==j (no self-edges)
            post_loc[self_loop] = rng.integers(0, m, int(self_loop.sum()))
            self_loop = pre_loc == post_loc
        pre_g = idx[pre_loc]
        post_g = idx[post_loc]
        rnd = rng.uniform(0.25, 1.0, n_edges)
        w = within_gain * rnd * sign_np[pre_g] / (sqrt_n * p)
        pre_chunks.append(pre_g)
        post_chunks.append(post_g)
        w_chunks.append(w)
    if pre_chunks:
        pre = _np.concatenate(pre_chunks)
        post = _np.concatenate(post_chunks)
        w = _np.concatenate(w_chunks)
    else:
        pre = _np.zeros(0, int)
        post = _np.zeros(0, int)
        w = _np.zeros(0, float)
    receptor = (w < 0).astype(_np.int32)
    tau = _np.where(receptor == 0, 2.0, 5.0)
    return EdgeList(
        pre=jnp.asarray(pre, jnp.int32),
        post=jnp.asarray(post, jnp.int32),
        weight=jnp.asarray(w, jdtype),
        receptor_index=jnp.asarray(receptor, jnp.int32),
        tau_ms=jnp.asarray(tau, jdtype),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )


def _apply_connectivity(params: IzhikevichParams, area_labels: Sequence[str], layer_labels: Sequence[str], cell_labels: Sequence[str], metadata: Mapping[str, Any], *, seed: int, dtype: str):
    n = len(cell_labels)
    if n == 0:
        return params, None
    jdtype = jnp.float64 if dtype == "float64" and bool(jax.config.read("jax_enable_x64")) else jnp.float32
    connectivity_spec = metadata.get("connectivity", {}) or {}
    p_connect = connectivity_spec.get("p_connect")
    base_gain = float(connectivity_spec.get("within_gain", 0.45))

    # ── Sparse-direct path ────────────────────────────────────────────────────
    # For a sparse within-area request at scale (p_connect<1, n>=threshold) with no
    # special-structure specs (TCM / suite2-interarea / inter-column), build the
    # EdgeList DIRECTLY and skip the dense (n,n) weight matrix entirely. Returns a
    # (0,0) placeholder W; the model is run on the edge_list backend. This is the
    # O(N^2)-memory escape for large sparse columns. Small-N / all-to-all / special
    # specs keep the exact dense path below.
    _tcm = bool(metadata.get("tcm_v1_6pop", False)) or bool(connectivity_spec.get("tcm_v1_6pop", False))
    _interarea = bool(metadata.get("suite2_interarea", False))
    _inter_col = bool(metadata.get("inter_column_connectivity"))
    if (p_connect is not None and 0.0 < float(p_connect) < 1.0
            and n >= _SPARSE_DIRECT_N and not (_tcm or _interarea or _inter_col)):
        edges = _make_sparse_within_area_edges(
            area_labels, params.sign, n,
            within_gain=base_gain, p_connect=float(p_connect), seed=seed, jdtype=jdtype)
        placeholder_W = jnp.zeros((0, 0), dtype=jdtype)
        return replace(params, W=placeholder_W), edges

    key = jax.random.PRNGKey(seed + 4242)
    rnd = jax.random.uniform(key, (n, n), minval=0.25, maxval=1.0, dtype=jdtype)
    sign = params.sign.astype(jdtype)
    # Vectorized same-area mask: O(N^2) numpy broadcast instead of an O(N^2)
    # pure-Python double comprehension (~100M ops at N=10k dominated construct cost).
    import numpy as _np
    _area_codes = _np.unique(_np.asarray(area_labels), return_inverse=True)[1]
    same_area = jnp.asarray(_area_codes[:, None] == _area_codes[None, :], dtype=jdtype)
    eye = jnp.eye(n, dtype=jdtype)
    W = base_gain * rnd * sign[None, :] * same_area * (1.0 - eye) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))

    # Transparency: dense all-to-all within-area connectivity materializes an (n,n)
    # weight matrix and ~n^2 edges. That is inherent to all-to-all topology, not a
    # bug, but it is O(N^2) in memory and edge count. Make the cost visible at scale
    # and point to the sparse lever (p_connect < 1) rather than failing silently.
    if n >= _DENSE_CONNECTIVITY_WARN_N and (p_connect is None or float(p_connect) >= 1.0):
        import warnings as _warnings
        _mb = (n * n * (8 if jdtype == jnp.float64 else 4)) / 1e6
        _warnings.warn(
            f"dense all-to-all within-area connectivity at N={n} materializes an "
            f"({n}x{n}) weight matrix (~{_mb:.0f} MB) and ~{n*n} edges (O(N^2)). "
            f"For large columns set connectivity p_connect<1 (or a sparse rule) to "
            f"reduce memory and edge count.",
            RuntimeWarning,
            stacklevel=2,
        )
    if p_connect is not None:
        p_val = float(p_connect)
        if 0.0 < p_val < 1.0:
            mask_key = jax.random.fold_in(key, 999)
            mask = jax.random.bernoulli(mask_key, p_val, (n, n)).astype(jdtype)
            W = W * mask / p_val

    # Apply TCM_V1_6POP specific population-based connection mask if enabled
    tcm_active = bool(metadata.get("tcm_v1_6pop", False)) or bool(connectivity_spec.get("tcm_v1_6pop", False))
    if tcm_active:
        import numpy as np
        pop = []
        for idx in range(n):
            lyr = layer_labels[idx]
            cell = cell_labels[idx]
            is_e = (cell == "E")
            if lyr in {"L2/3", "L2", "L3"}:
                pop.append("SP" if is_e else "SI")
            elif lyr == "L4":
                pop.append("SS" if is_e else "other")
            elif lyr == "L5":
                pop.append("DP" if is_e else "DI")
            elif lyr == "L6":
                pop.append("TP" if is_e else "other")
            else:
                pop.append("other")
        allowed_pairs = {
            ("SS", "SP"),
            ("SS", "SI"),
            ("SP", "SP"),
            ("SP", "SI"),
            ("SI", "SP"),
            ("DP", "DP"),
            ("DP", "DI"),
            ("DI", "DP"),
            ("SP", "DP"),
            ("DP", "SP"),
            ("SP", "TP"),
            ("TP", "DP")
        }
        # Vectorized population-pair mask (bit-identical to the prior O(N^2) Python
        # double loop, which was ~N^2 Python-level iterations — ~1e8 at N=1e4).
        # tcm_mask[i, j] = 1 iff (pop[j] (pre), pop[i] (post)) in allowed_pairs.
        uniq = list(dict.fromkeys(pop))
        code = {p: c for c, p in enumerate(uniq)}
        K = len(uniq)
        allowed_code = np.zeros((K, K), dtype=float)   # [src_code, dst_code]
        for src, dst in allowed_pairs:
            if src in code and dst in code:
                allowed_code[code[src], code[dst]] = 1.0
        codes = np.fromiter((code[p] for p in pop), dtype=np.intp, count=n)
        tcm_mask = allowed_code[codes[None, :], codes[:, None]]  # [i,j] = allowed[codes[j], codes[i]]
        W = W * jnp.asarray(tcm_mask, dtype=jdtype)

    if bool(metadata.get("suite2_interarea", False)):
        pre_v1_l23 = jnp.asarray([area_labels[j] == "V1" and layer_labels[j] in {"L2", "L3", "L2/3"} and cell_labels[j] == "E" for j in range(n)], dtype=jdtype)
        post_v4_l4 = jnp.asarray([area_labels[i] == "V4" and layer_labels[i] == "L4" and cell_labels[i] == "E" for i in range(n)], dtype=jdtype)
        pre_v4_l23 = jnp.asarray([area_labels[j] == "V4" and layer_labels[j] in {"L2", "L3", "L2/3"} and cell_labels[j] == "E" for j in range(n)], dtype=jdtype)
        post_v1_deep_l1 = jnp.asarray([area_labels[i] == "V1" and layer_labels[i] in {"L1", "L5", "L6"} and cell_labels[i] == "E" for i in range(n)], dtype=jdtype)
        ff_gain = float((metadata.get("connectivity", {}) or {}).get("feedforward_gain", 0.65)) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))
        fb_gain = float((metadata.get("connectivity", {}) or {}).get("feedback_gain", 0.50)) / jnp.sqrt(jnp.asarray(max(n, 1), dtype=jdtype))
        W = W + ff_gain * (post_v4_l4[:, None] * pre_v1_l23[None, :])
        W = W + fb_gain * (post_v1_deep_l1[:, None] * pre_v4_l23[None, :])

    # Declarative inter-column connectivity (from Configuration.inter_column_connectivity).
    # Materializes real cross-area edges using anatomical routing: feedforward
    # from source L2/3 E-cells -> target L4; feedback from source L6 -> target
    # L1/L5. An explicit layer_to_layer_map overrides these defaults. Proxy
    # scaffold connectivity, not a calibrated axonal projection.
    inter = metadata.get("inter_column_connectivity")
    if inter:
        specs = inter if isinstance(inter, list) else [inter]
        for spec in specs:
            W = W + _interarea_W(spec, area_labels, layer_labels, cell_labels, n, jdtype, seed)

    return replace(params, W=W), None

