"""N-parametrized large cortical column: capped/localized connectivity, full
readout+vis+statistics workflow.

Not tied to any specific N ("100k") -- N is a plain parameter. Timing/memory
verified separately this session:
  - N=100,000: construct() ~216s, simulate() (100 steps) ~5s, 10.0M edges
    (max_in_degree=100, spatial_sigma=0.15mm), CPU-only (jax-metal ruled out:
    incompatible with the installed JAX version, confirmed 2026-07-01).
  - N=100k+ or faster iteration: run this same script in a Colab notebook
    with a real GPU/TPU runtime instead of this machine's CPU.

HDP is deliberately OFF (K_HDP=0, the null control) -- the custom "spend more,
lose more" dH/dt formula requested for this column has not yet been designed
or small-scale-validated. Both of jaxfne's existing HDP restoring mechanisms
were confirmed non-functional this session (K_ctrl is dead code; the full
rho_passive sweep fails at all 15 candidates -- see skills/FRICTIONS_STACK.md
F-017). Enabling an unvalidated third formula directly at scale would repeat
that mistake at far higher cost. See plans.json item
hdp-universal-default-kernel-consolidation for the follow-on design.
"""
from __future__ import annotations

from collections import defaultdict

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne


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
