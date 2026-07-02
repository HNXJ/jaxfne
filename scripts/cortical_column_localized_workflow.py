"""N-parametrized large cortical column: capped/localized connectivity, full
readout+vis+statistics workflow.

Not tied to any specific N ("100k") -- N is a plain parameter. Timing/memory
verified separately this session:
  - N=100,000: construct() ~216s, simulate() (100 steps) ~5s, 10.0M edges
    (max_in_degree=100, spatial_sigma=0.15mm), CPU-only (jax-metal ruled out:
    incompatible with the installed JAX version, confirmed 2026-07-01).
  - N=100k+ or faster iteration: run this same script in a Colab notebook
    with a real GPU/TPU runtime instead of this machine's CPU.

HDP is deliberately OFF (K_HDP=0, the null control) here -- this workflow
predates the HDP fix. K_ctrl is now a live, validated two-sided restoring
term (see skills/FRICTIONS_STACK.md F-017, resolved 2026-07-01); enabling HDP
on the real Configuration-based column path (as opposed to the standalone
HDPColumnConfig path already validated) is tracked separately as
plans.json item hdp-100k-100step-validation-run -- not done in this script.

construct()-once / save / reload (added 2026-07-01): `construct()` is the
expensive step (2.9s at N=2000, 231s at N=100k -- connectivity sampling
dominates, not population init). Static parameters (positions, per-neuron
Izhikevich params, edge list) don't change between runs of the same config,
so `save_column`/`load_column` below persist them and reload without paying
construct() again. IMPORTANT correction: an earlier in-session attempt used
a "tiny dummy construct() + jax.tree_util.tree_unflatten" trick assuming
treedef is N-independent -- this is FALSE and was caught by a real
bit-identical-output check: `IzhikevichParams.labels`/`layer_labels` are aux
data (length N) baked into the treedef itself, so a dummy-N treedef silently
corrupts the reloaded model (no error, just wrong simulate() output).
`save_column`/`load_column` instead reconstruct the dataclasses directly
(bypassing tree_unflatten's structural-matching requirement entirely) --
verified bit-identical V_m at N=2000 with zero construct() calls on reload.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.io import json_safe


def build_config(
    n: int,
    *,
    radius_mm: float = 1.0,
    height_mm: float = 2.0,
    max_in_degree: int = 100,
    spatial_sigma: float = 0.15,
    duration_ms: float = 100.0,
    dt_ms: float = 0.5,
    seed: int = 0,
) -> "jtfne.Configuration":
    """6-layer, 4-cell-type cylindrical column, N is just a parameter.

    geometry='uniform3d' is required for the cylindrical (radius_mm, height_mm)
    footprint -- but that mode discards the per-neuron 'layer' label (confirmed
    2026-07-01; the alternative geometry='laminar' mode preserves the label but
    ignores radius_mm/height_mm entirely -- the two are mutually exclusive in
    the current builder, a real API gap, not a flag you can just pick around).
    The per-layer E:I cell-type COMPOSITION is still correctly banded by z
    either way (layer_fractions/layer_cell_type_fractions apply regardless of
    which geometry mode assigns positions) -- only the descriptive label is
    lost, and is reconstructed from z-position + layer_fractions in
    `layer_statistics_table` below.
    """
    return (
        jtfne.build_laminar_column(
            "V1", n=n,
            layers=list(jtfne.CANONICAL_LAYERS_6L),
            ei_profile="canonical",
            geometry="uniform3d", radius_mm=radius_mm, height_mm=height_mm,
        )
        # Negligible auto-connectivity: MUST be far smaller than 1/n^2 at large
        # n or it silently adds thousands of stray edges on top of the capped
        # rule below (confirmed 2026-07-01: p_connect=1e-6 leaked ~10,028 extra
        # edges at n=100,000, nudging a few neurons' in-degree above the cap).
        .connectivity(p_connect=1e-10)
        .runtime(seed=seed, duration_ms=duration_ms, dt_ms=dt_ms)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .mechanisms(name="ampa", kind="synapse", params={"tau_ms": 5.0, "sign": 1.0})
        .connections(
            name="local", source={}, target={},
            max_in_degree=max_in_degree, spatial_sigma=spatial_sigma,
            weight=0.3, mechanism="ampa",
        )
    )


def save_column(model: "jtfne.Model", path: str) -> None:
    """Persist the expensive construct() output: emitter/edge_list arrays +
    their aux data (labels, layer_labels, calibration strings) + positions +
    model.static + model.cfg.metadata.

    model.cfg.metadata IS persisted (confirmed 2026-07-01 to be required,
    not optional): construct() mutates it in place (adds a
    `recurrent_backend` key, flips `circuit.connections[*].status` etc.) and
    simulate() reads that mutated state -- reusing a pre-construct cfg's
    metadata silently reproduces a DIFFERENT (still finite, no error) V_m
    trace. Caught by a real bit-identical-output check across two calls to
    `build_config(same kwargs)`, not assumed.
    """
    p = Path(path)
    emitter, edge_list, positions = model.params["emitter"], model.params["edge_list"], model.params["positions"]
    np.savez(
        p.with_suffix(".npz"),
        **{f"emitter_{f}": np.asarray(getattr(emitter, f))
           for f in ("a", "b", "c", "d", "drive", "sign", "W", "v0", "u0", "source_scale")},
        **{f"edge_{f}": np.asarray(getattr(edge_list, f))
           for f in ("pre", "post", "weight", "receptor_index", "tau_ms")},
        positions=np.asarray(positions),
    )
    meta = {
        "schema": "column_checkpoint_v1",
        "emitter_labels": list(emitter.labels),
        "emitter_layer_labels": list(emitter.layer_labels) if emitter.layer_labels is not None else None,
        "emitter_source_calibration_status": emitter.source_calibration_status,
        "edge_source_calibration_status": edge_list.source_calibration_status,
        "static": json_safe(model.static),
        "cfg_metadata": json_safe(model.cfg.metadata),
    }
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_column(path: str, cfg: "jtfne.Configuration") -> "jtfne.Model":
    """Inverse of `save_column`. `cfg` must come from a fresh (cheap,
    <1ms, no construct() call) `build_config(**same kwargs used to build the
    saved model)` -- its `metadata` field is overwritten in place with the
    saved post-construct metadata (see `save_column`), then the params
    dataclasses are reconstructed directly (bypassing tree_unflatten's
    fragile treedef-structural-matching requirement entirely).
    """
    import dataclasses
    p = Path(path)
    meta = json.loads(p.with_suffix(".json").read_text())
    if meta["schema"] != "column_checkpoint_v1":
        raise ValueError(f"Unknown checkpoint schema: {meta['schema']!r}")
    with np.load(p.with_suffix(".npz")) as z:
        emitter = IzhikevichParams(
            a=jnp.array(z["emitter_a"]), b=jnp.array(z["emitter_b"]), c=jnp.array(z["emitter_c"]),
            d=jnp.array(z["emitter_d"]), drive=jnp.array(z["emitter_drive"]), sign=jnp.array(z["emitter_sign"]),
            W=jnp.array(z["emitter_W"]), v0=jnp.array(z["emitter_v0"]), u0=jnp.array(z["emitter_u0"]),
            source_scale=jnp.array(z["emitter_source_scale"]),
            labels=tuple(meta["emitter_labels"]),
            layer_labels=tuple(meta["emitter_layer_labels"]) if meta["emitter_layer_labels"] is not None else None,
            source_calibration_status=meta["emitter_source_calibration_status"],
        )
        edge_list = EdgeList(
            pre=jnp.array(z["edge_pre"]), post=jnp.array(z["edge_post"]), weight=jnp.array(z["edge_weight"]),
            receptor_index=jnp.array(z["edge_receptor_index"]), tau_ms=jnp.array(z["edge_tau_ms"]),
            source_calibration_status=meta["edge_source_calibration_status"],
        )
        positions = jnp.array(z["positions"])
    params = {"emitter": emitter, "positions": positions, "edge_list": edge_list}
    cfg = dataclasses.replace(cfg, metadata=meta["cfg_metadata"])
    return jtfne.Model(cfg=cfg, params=params, static=meta["static"])


def layer_statistics_table(model: "jtfne.Model", height_mm: float) -> str:
    """Area/layer/cell-type rows: neuron count + average synapse (in-degree) count."""
    layer_bands = sorted(model.cfg.metadata["layer_fractions"].items(), key=lambda kv: kv[1][0])
    layer_order = [name for name, _ in layer_bands]

    def z_to_layer(z: float) -> str:
        frac = z / height_mm
        for name, (z0, z1) in layer_bands:
            if z0 <= frac < z1 or (name == layer_order[-1] and frac >= z0):
                return name
        return layer_order[0]

    rows = model.neuron_table()
    post = np.asarray(model.params["edge_list"].post)
    in_degree = np.bincount(post, minlength=len(rows))

    by_key = defaultdict(list)
    for i, r in enumerate(rows):
        by_key[(r["area"], z_to_layer(r["z"]), r["cell_type"])].append(in_degree[i])

    lines = [f'{"area":<6}{"layer":<6}{"type":<6}{"n":>8}{"avg_synapses":>14}']
    for area, layer, ct in sorted(by_key, key=lambda k: (k[0], layer_order.index(k[1]), k[2])):
        degs = by_key[(area, layer, ct)]
        lines.append(f"{area:<6}{layer:<6}{ct:<6}{len(degs):>8}{np.mean(degs):>14.2f}")
    return "\n".join(lines)


def run(n: int = 2000) -> dict:
    cfg_ltfnn = build_config(n)
    jnt = jtfne.construct(cfg_ltfnn)                                      # NeuronalTensor/Model

    sig = jtfne.simulate(jnt, duration_ms=100.0, dt_ms=0.5, seed=0)       # 1. simulate
    fig_lfp = jtfne.vis.lfp(sig)                                         # 2. LFP
    fig_raster = jtfne.vis.raster(sig)                                   # 3. Raster
    fig_csd = jtfne.vis.csd(sig)                                         # 4. CSD
    fig_spectro = jtfne.vis.spectrolaminar_suite(sig)                    # 5. spectrolaminar suite
    stats_table = layer_statistics_table(jnt, height_mm=2.0)             # 6. statistics

    assert bool(jnp.all(jnp.isfinite(sig.V_m))), "non-finite V_m -- do not trust this run"
    print(stats_table)
    return {
        "model": jnt, "signals": sig,
        "fig_lfp": fig_lfp, "fig_raster": fig_raster, "fig_csd": fig_csd, "fig_spectro": fig_spectro,
        "stats_table": stats_table,
    }


if __name__ == "__main__":
    run()
