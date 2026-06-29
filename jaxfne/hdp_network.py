"""Generic, size-agnostic builder for TFNE-Izhikevich+HDP laminar columns.

Single reusable network maker for any neuron count. No function here is
named after a specific N -- structural size, layer geometry, cell-type
mix, drive correction, layer-size scaling, and HDP gains are all config
fields on `HDPColumnConfig`, analogous to a torch `nn.Module` taking
`in_channels`/`out_channels` etc. as constructor args rather than being
redefined per width. This consolidates the near-duplicate `build_model`/
`apply_drive_correction` pairs previously copy-pasted across
`scripts/hdp_1000_laminar_column_boosted.py` and
`scripts/hdp_100_stability_sweep.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne

LAYERS_DEFAULT = ["L1", "L2", "L3", "L4", "L5", "L6"]

# Canonical z-bands (width prop. to neuron count): 100,250,200,100,200,150.
ZBANDS_DEFAULT = {
    "L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}

# Canonical PV-peak-at-L4-only cell-type mix (E:I gradient deep->E,
# superficial->I; see project memory `canonical-cortical-column-template`).
LAYER_CELL_TYPE_FRAC_DEFAULT = {
    "L1": {"E": 0.50, "PV": 0.00, "SST": 0.15, "VIP": 0.35},
    "L2": {"E": 0.65, "PV": 0.20, "SST": 0.10, "VIP": 0.05},
    "L3": {"E": 0.80, "PV": 0.08, "SST": 0.08, "VIP": 0.04},
    "L4": {"E": 0.75, "PV": 0.18, "SST": 0.04, "VIP": 0.03},
    "L5": {"E": 0.88, "PV": 0.06, "SST": 0.04, "VIP": 0.02},
    "L6": {"E": 0.90, "PV": 0.05, "SST": 0.03, "VIP": 0.02},
}

BASE_DRIVE_BY_CELL_TYPE_DEFAULT = {"E": 4.0, "PV": 4.0, "SST": 4.0, "VIP": 4.0}

# Re-derived (N=250, see scripts/hdp_100_stability_sweep.py docstring) and
# verified STATIONARY over 20s at K_HDP=0 -- the prerequisite fixed point
# any HDP plasticity setting is layered on top of. Caveat: a drive
# correction found at one N is a starting point, not guaranteed to hold
# exactly at every N -- re-verify stationarity before trusting a new size.
DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT = {
    "E": 1.0, "PV": 0.8757734374999999, "SST": 0.06492187500000002, "VIP": 5.7509375,
}

# Verified long-term stable (20s, 5-seed gate, flat rates in-band, H pinned
# ~1.0000-1.0029, weight_saturation_fraction~0.105). Freeze candidate for
# 0.4.6 per project memory; tau_0_ms multiplies the per-neuron size**3 term
# (cube law, 0.4.7: size=2.0 -> 8x tau_0_ms, larger neurons adapt slower).
# Note (0.4.7 HDP v2): K_ctrl is now a no-op; use rho_passive for passive-income
# restoring instead. The verified configuration above used K_ctrl=5.0; this has
# not yet been re-tuned for the new passive-income mechanism (rho_passive/H^2).
# For backward compatibility, K_ctrl=5.0 is preserved here but inactive; set
# rho_passive > 0 to use the new restoring mechanism.
DEFAULT_HDP = dict(
    K_HDP=0.01,
    tau_0_ms=200.0,
    K_ctrl=5.0,
    rho_passive=0.0,
    barrier_c=0.01,
    barrier_d=0.01,
)

BASE_HDP_KWARGS_DEFAULT = dict(
    H_min=0.1, H_max=10.0,
    alpha=0.01, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0,
    barrier_eps=1.0e-3, w_floor=0.01, w_ceiling=10.0, H_boost_gain=4.0,
)

# Faster, less-overdamped H dynamics: DEFAULT_HDP's tau_0_ms=200 combined
# with size**3 scaling (E size=5 -> tau_i=25000ms) makes H_i nearly static
# (H_std~0.0006), which in turn keeps every neuron's spiking near-regular
# ("ECG-like") -- low population-level rate variance even though overall
# rate is in-band. This profile trades that overdamping for faster H
# integration (tau_0_ms=5, 40x faster) plus a rate-drain term (gamma>0,
# previously 0.0) so H genuinely responds to each neuron's own activity
# instead of being pinned by K_ctrl alone.
#
# Second-pass refinement (gamma/alpha/K_ctrl re-swept around the first
# candidate, N=500, 2000ms, drive_scale=1.2x BASE_DRIVE_BY_CELL_TYPE_DEFAULT):
# raising gamma from 0.3 to 0.5 tightened BOTH tails (the rate-drain term
# is rate-bounded, not weight-multiplicative, so it doesn't amplify the
# H>1 weight-growth feedback loop the way alpha does); gamma>=0.55 hits a
# stability cliff (weight runaway, H pinned to the H_max clamp) at every
# K_ctrl/alpha tried. Lowering alpha from 0.07 to 0.05 at gamma=0.5 then
# tightened the upper tail further (alpha>=0.08 also hits the same
# runaway cliff). The lower tail (~0.95) did not move further down across
# the K_ctrl range tried (0.08-0.15) -- it appears structurally bottlenecked
# by E neurons' large tau_i (tau_0_ms*size**3, size=5 for E) keeping the
# majority population's H sluggish, not by these three gains. Verified
# 5-seed stable (N=500, 2000ms): rate=12.6Hz, H=1.028+-0.023 in
# [0.953,1.227] (vs the first-pass [0.96,1.47]), rate_std~6.0Hz (vs 0.99Hz
# at DEFAULT_HDP), kappa~0.044 (still async-irregular). Tighter and more
# symmetric than the first pass but still does not reach the requested
# [0.8,1.2] floor -- treat as the current best candidate, not a finished/
# frozen point like DEFAULT_HDP.
# Note (0.4.7 HDP v2): K_ctrl is now a no-op; rho_passive=0.0 by default.
DEFAULT_HDP_DESYNC = dict(
    K_HDP=0.01,
    tau_0_ms=5.0,
    K_ctrl=0.15,
    rho_passive=0.0,
    barrier_c=0.01,
    barrier_d=0.01,
    alpha=0.05,
    gamma=0.5,
    C_spike=0.0,
)

# Drive scale paired with DEFAULT_HDP_DESYNC: x1.2 on
# BASE_DRIVE_BY_CELL_TYPE_DEFAULT raises overall rate from ~7.3Hz to
# ~12.6Hz while keeping H pinned (H_mean=1.001) at DEFAULT_HDP -- found by
# sweeping drive_scale in {1.0,1.2,1.4,...} and picking the point with the
# best variance/rate tradeoff before H itself was made faster.
DRIVE_SCALE_DESYNC = 1.2

# Deep-layer size inflation, per user spec: makes deep layers appear
# visually less dense and (via tau_i = tau_0_ms * size_i**3 in
# jaxfne.emitters) makes deep-layer H integrate slower. Multiplies
# whatever cell-type size the neuron already has; 1.0 = no change.
LAYER_SIZE_SCALE_DEFAULT = {
    "L1": 1.0, "L2": 1.0, "L3": 1.0, "L4": 1.0, "L5": 1.5, "L6": 2.0,
}


@dataclass
class HDPColumnConfig:
    """Fully-flexible spec for a TFNE-Izhikevich+HDP laminar column.

    `n_neurons` is just a field, not part of any function/file name --
    the same builder produces a 100-neuron smoke test or a 10000-neuron
    production run from the same code path.
    """

    n_neurons: int
    duration_ms: float
    dt_ms: float = 0.5
    seed: int = 0
    areas: list[str] = field(default_factory=lambda: ["V1"])
    layers: list[str] = field(default_factory=lambda: list(LAYERS_DEFAULT))
    layer_fractions: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(ZBANDS_DEFAULT))
    layer_cell_type_frac: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in LAYER_CELL_TYPE_FRAC_DEFAULT.items()})
    base_drive_by_cell_type: Mapping[str, float] = field(
        default_factory=lambda: dict(BASE_DRIVE_BY_CELL_TYPE_DEFAULT))
    drive_correction_by_cell_type: Mapping[str, float] = field(
        default_factory=lambda: dict(DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT))
    layer_size_scale: "Mapping[str, float] | None" = field(
        default_factory=lambda: dict(LAYER_SIZE_SCALE_DEFAULT))
    probe_name: str = "hdp_column_probe"


def build_model(cfg: HDPColumnConfig) -> "jtfne.core.Model":
    """Construct an HDP-ready laminar column Model from `cfg`. The single
    network maker -- vary `cfg.n_neurons` (or any other field) to scale up
    or down; never copy this function to add a new size."""
    builder = (
        jtfne.laminar_cortex_config(
            areas=cfg.areas, layers=cfg.layers, n=cfg.n_neurons,
            duration_ms=cfg.duration_ms, dt_ms=cfg.dt_ms, seed=cfg.seed,
            emitter="izhikevich",
            baseline_drive_by_cell_type=dict(cfg.base_drive_by_cell_type),
        )
        .layer_fractions(layer_fractions=dict(cfg.layer_fractions))
        .area_layer_cell_types(cfg.areas[0], {k: dict(v) for k, v in cfg.layer_cell_type_frac.items()})
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name=cfg.probe_name, modes=["spikes", "V_m"])
    )
    return jtfne.construct(builder)


def apply_drive_correction(model: "jtfne.core.Model", cfg: HDPColumnConfig) -> "jtfne.core.Model":
    """Per-cell-type multiplicative drive correction, applied post-hoc
    via with_emitter_parameters (no rebuild needed)."""
    labels = np.asarray(model.params["emitter"].labels)
    base_drive = np.asarray(model.params["emitter"].drive)
    corrected = base_drive.copy()
    for ct, factor in cfg.drive_correction_by_cell_type.items():
        corrected[labels == ct] = base_drive[labels == ct] * factor
    return model.with_emitter_parameters(drive_per_neuron=jnp.asarray(corrected))


def layer_size_scale_override(model: "jtfne.core.Model", cfg: HDPColumnConfig) -> "np.ndarray | None":
    """Per-neuron size array combining the kernel's existing per-cell-type
    size table (jaxfne.emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE) with
    `cfg.layer_size_scale` (a per-layer multiplier, e.g. deep layers
    inflated to read less dense and integrate H more slowly via
    tau_i = tau_0_ms * size_i**3). Returns None if cfg.layer_size_scale is
    unset, so callers can pass it straight through to
    `simulate_edge_recurrent_izhikevich_hdp(size_scale_override=...)`.
    """
    if not cfg.layer_size_scale:
        return None
    from jaxfne.emitters import DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE

    rows = model.neuron_table()
    out = np.empty(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        base = DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE.get(row["cell_type"], 1.0)
        layer_mult = cfg.layer_size_scale.get(row["layer"], 1.0)
        out[i] = base * layer_mult
    return out


def run(
    model: "jtfne.core.Model",
    cfg: HDPColumnConfig,
    duration_ms: float,
    seed: int,
    hdp_kwargs: "Mapping[str, Any] | None" = None,
    size_scale_override: "np.ndarray | None" = None,
):
    """Run simulate_edge_recurrent_izhikevich_hdp for `duration_ms` with the
    DEFAULT_HDP+BASE_HDP_KWARGS_DEFAULT defaults, overridable via `hdp_kwargs`."""
    import jax
    from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp

    kw = dict(BASE_HDP_KWARGS_DEFAULT)
    kw.update(DEFAULT_HDP)
    if hdp_kwargs:
        kw.update(hdp_kwargs)
    n_steps = int(round(duration_ms / cfg.dt_ms))
    key = jax.random.PRNGKey(seed)
    voltages, spikes, sources, diagnostics = simulate_edge_recurrent_izhikevich_hdp(
        model.params["emitter"], model.params["edge_list"], n_steps, cfg.dt_ms, key,
        size_scale_override=size_scale_override,
        **kw,
    )
    return {"voltages": voltages, "spikes": spikes, "sources": sources,
            "diagnostics": diagnostics, "n_steps": n_steps, "hdp_kwargs": kw}
