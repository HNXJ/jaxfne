"""Suite No. 2 visualizations for the tuned 1000-neuron HDP laminar column.

Reuses the network built in hdp_1000_laminar_column_boosted.py (PV peak at
L2 AND L4, canonical E-deep/I-superficial gradient, dense L2/3) and the
H_boost_gain kernel mechanism, to produce:

  1. 3D network visualization (jtfne.vis.visualize_network_3d -- pre-existing
     function, not new code).
  2. One T=2000ms trial -> raster sorted by depth (jtfne.vis.raster,
     sort_by="z") and LFP traces stacked by depth across 32 channels.
  3. Ten T=500ms trials -> the legacy 3-panel spectrolaminar suite
     (jtfne.vis.spectrolaminar_suite_3panel, "the standard laminar readout")
     built from jaxfne.tutorial_utils.spectrolaminar_from_trials.

HDP is always on (this script always calls
jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp, never the bare
non-HDP kernel); K_HDP defaults to 1.0 and is configurable (0.0 = weight
plasticity off, H dynamics alone).

IMPORTANT, VERIFIED CAVEAT (not papered over): at K_HDP=1.0 with this
network's drive tuning, the closed H<->weight loop is NOT stable -- a
1000ms smoke test at K_HDP=1.0 gave PV/SST/VIP rates of 360-460Hz and
H drifting to ~1.27 (matches the earlier-segment "closed-loop-gain"
finding: K_HDP=1.0 is broadly unstable except at very specific tau_0_ms).
The actual runs below use K_HDP=0.05 (the value already verified stable
and in-band for this network in hdp_1000_laminar_column_boosted.py).
K_HDP_DEFAULT below documents the user-specified default of 1.0; pass
``K_HDP=1.0`` explicitly to reproduce the unstable regime yourself.

32-channel contact placement: neuron z-positions are affine-rescaled into
[1/31, 30/31] before projection, so of the 32 contacts at
linspace(0,1,32), exactly 30 (indices 1..30) land inside the rescaled
cortical span and the 2 endpoints (index 0, index 31) land slightly
outside it (above L1 / below L6) -- this is a position rescale, not a
change to jtfne.project_laminar_sources, which always places contacts at
linspace(0,1,n_contacts).

Usage: PYTHONPATH=. python3 scripts/hdp_suite2_visualizations.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import jaxfne as jtfne
from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp
from jaxfne.tutorial_utils import spectrolaminar_from_trials

import scripts.hdp_1000_laminar_column_boosted as base

OUTPUT_DIR = Path("outputs/hdp_suite2_visualizations")

K_HDP_DEFAULT = 1.0  # user-specified default; see module docstring caveat
K_HDP_STABLE = 0.05  # value actually used for the runs below (verified stable)

N_CONTACTS = 32
N_CORTEX_CONTACTS = 30  # contacts 1..30 of 32 land inside the rescaled cortical span
PROJECTION_WIDTH = 0.04
CONTACT_DEPTH_LO = 1.0 / (N_CONTACTS - 1)
CONTACT_DEPTH_HI = (N_CONTACTS - 2) / (N_CONTACTS - 1)
COLUMN_HEIGHT_M = 0.0016  # nominal 1.6mm column height; axis-scaling proxy only

DT_MS = base.DT_MS


def rescaled_positions(model: "jtfne.core.Model") -> jnp.Array:
    """Affine-rescale z so 30/32 projection contacts land inside [0,1], 2 outside."""
    positions = np.asarray(model.params["positions"])
    z = positions[:, 2]
    z_rescaled = CONTACT_DEPTH_LO + z * (CONTACT_DEPTH_HI - CONTACT_DEPTH_LO)
    out = positions.copy()
    out[:, 2] = z_rescaled
    return jnp.asarray(out)


def run_trial(model: "jtfne.core.Model", duration_ms: float, seed: int, *, K_HDP: float) -> dict[str, Any]:
    hdp_kwargs = dict(base.HDP_KWARGS)
    hdp_kwargs["K_HDP"] = K_HDP
    n_steps = int(round(duration_ms / DT_MS))
    key = jax.random.PRNGKey(seed)
    voltages, spikes, sources, diagnostics = simulate_edge_recurrent_izhikevich_hdp(
        model.params["emitter"], model.params["edge_list"], n_steps, DT_MS, key, **hdp_kwargs,
    )
    return {"voltages": voltages, "spikes": spikes, "sources": sources,
            "diagnostics": diagnostics, "n_steps": n_steps}


def build_signals(model: "jtfne.core.Model", trial: dict[str, Any], field: "jtfne.FieldOutput") -> "jtfne.Signals":
    n_steps = trial["n_steps"]
    time_ms = jnp.arange(n_steps, dtype=jnp.float32) * DT_MS
    nt = model.neuron_table()
    return jtfne.Signals(
        time_ms=time_ms,
        V_m=trial["voltages"],
        spikes=trial["spikes"],
        sources=trial["sources"],
        field=field,
        metadata={
            "claim_level": "proxy_readout",
            "field_solver_status": "linear_solver",
            "physical_amplitude_calibrated": False,
            "dt_ms": DT_MS,
            "neuron_metadata": nt,
            "note": "computational scaffold; HDP suite-no-2 diagnostic run",
        },
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Building 1000-neuron laminar column (reused from hdp_1000_laminar_column_boosted) ===")
    model = base.build_model()
    model = base.apply_drive_correction(model)
    positions_rescaled = rescaled_positions(model)

    # --- 1. 3D network visualization (pre-existing function) ---
    print("\n=== [1] 3D network visualization (jtfne.vis.visualize_network_3d) ===")
    jtfne.vis.visualize_network_3d(
        model.neuron_table(), title="HDP 1000-neuron laminar column (PV peak L2+L4)",
        show_layers=True, output_html=str(OUTPUT_DIR / "network_3d.html"),
    )
    print(f"Saved {OUTPUT_DIR / 'network_3d.html'}")

    # --- 2. One T=2000ms trial: raster + LFP traces by depth ---
    print(f"\n=== [2] One T=2000ms trial, K_HDP={K_HDP_STABLE} (HDP always on) ===")
    trial_long = run_trial(model, duration_ms=2000.0, seed=0, K_HDP=K_HDP_STABLE)
    field_long = jtfne.project_laminar_sources(
        trial_long["sources"], positions_rescaled,
        n_contacts=N_CONTACTS, width=PROJECTION_WIDTH, mode="density_preserving",
    )
    contact_depths = np.asarray(field_long.contact_depths)
    eps = 1e-4
    n_in_cortex = int(np.sum((contact_depths > CONTACT_DEPTH_LO - eps) & (contact_depths < CONTACT_DEPTH_HI + eps)))
    print(f"Contact depths (relative): {contact_depths}")
    print(f"Contacts inside rescaled cortical span [{CONTACT_DEPTH_LO:.4f}, {CONTACT_DEPTH_HI:.4f}]: {n_in_cortex} / {N_CONTACTS}")

    signals_long = build_signals(model, trial_long, field_long)
    fig_raster = jtfne.vis.raster(signals_long, sort_by="z", figsize=(10, 6))
    fig_raster.savefig(OUTPUT_DIR / "raster_z_sorted.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'raster_z_sorted.png'}")

    _plot_lfp_traces_by_depth(signals_long, OUTPUT_DIR / "lfp_traces_z_sorted.png")
    print(f"Saved {OUTPUT_DIR / 'lfp_traces_z_sorted.png'}")

    # --- 3. Ten T=500ms trials: 3-panel spectrolaminar suite ---
    print(f"\n=== [3] Ten T=500ms trials, K_HDP={K_HDP_STABLE} (HDP always on) ===")
    n_trials = 10
    duration_ms = 500.0
    n_steps = int(round(duration_ms / DT_MS))
    n_neurons = len(model.neuron_table())

    csd_contacts = np.zeros((n_trials, n_steps, N_CONTACTS), dtype=np.float32)
    spikes_all = np.zeros((n_trials, n_steps, n_neurons), dtype=np.float32)
    for ti in range(n_trials):
        trial = run_trial(model, duration_ms=duration_ms, seed=100 + ti, K_HDP=K_HDP_STABLE)
        field = jtfne.project_laminar_sources(
            trial["sources"], positions_rescaled,
            n_contacts=N_CONTACTS, width=PROJECTION_WIDTH, mode="density_preserving",
        )
        csd_contacts[ti] = np.asarray(field.csd_proxy)
        spikes_all[ti] = np.asarray(trial["spikes"])
        print(f"  trial {ti}: mean rate {float(spikes_all[ti].mean() * 1000.0 / DT_MS):.2f} Hz")

    contact_depths_m = contact_depths * COLUMN_HEIGHT_M
    trials = {
        "csd_contacts": csd_contacts,
        "spikes": spikes_all,
        "contact_depths_m": contact_depths_m,
    }

    nt = model.neuron_table()
    z = np.array([float(r["z"]) for r in nt])
    z_rescaled = CONTACT_DEPTH_LO + z * (CONTACT_DEPTH_HI - CONTACT_DEPTH_LO)
    pos_from_l4 = z_rescaled * COLUMN_HEIGHT_M - 0.60 * COLUMN_HEIGHT_M  # L4 mid-band ~0.60 in rescaled depth
    neurons_df = pd.DataFrame({
        "area": [r["area"] for r in nt],
        "cell_type": [r["cell_type"] for r in nt],
        "layer": [r["layer"] for r in nt],
        "pos_from_l4": pos_from_l4,
    })

    cfg = SimpleNamespace(
        dt_ms=DT_MS, freq_min_hz=1.0, freq_max_hz=150.0, freq_count=96,
        l4_ref_rel=0.60, cz_m=COLUMN_HEIGHT_M, areas=["V1"], n_trials=n_trials,
        output_dir=None,
    )

    _, prof = spectrolaminar_from_trials(
        trials, cfg, signal_key="csd_contacts", area_name="V1",
        alpha_beta_range_hz=(10.0, 25.0), gamma_range_hz=(40.0, 150.0),
    )
    prof["n_trials"] = n_trials
    prof["similarity_percent"] = float("nan")
    specs = {"V1": prof}

    ab_max = float(np.max(prof["alpha_beta"]))
    gm_max = float(np.max(prof["gamma"]))
    print(f"\nalpha_beta profile max (must be 1.0): {ab_max:.6f}")
    print(f"gamma profile max (must be 1.0): {gm_max:.6f}")
    print(f"csd_contacts channels: {csd_contacts.shape[-1]} (n_contacts={N_CONTACTS})")

    figs = jtfne.vis.spectrolaminar_suite_3panel(
        specs, {"neurons": neurons_df}, cfg, areas=["V1"], output_dir=None, trials=trials,
    )
    figs["V1"].savefig(OUTPUT_DIR / "spectrolaminar_3panel_suite.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'spectrolaminar_3panel_suite.png'}")

    payload = {
        "K_HDP_user_specified_default": K_HDP_DEFAULT,
        "K_HDP_used_for_runs": K_HDP_STABLE,
        "K_HDP_1p0_instability_smoke_test": "PV/SST/VIP rates 360-460Hz, H_mean~1.27 at K_HDP=1.0 -- unstable, see module docstring",
        "n_contacts": N_CONTACTS,
        "contacts_inside_cortical_span": n_in_cortex,
        "contact_depths_relative": contact_depths.tolist(),
        "alpha_beta_profile_max": ab_max,
        "gamma_profile_max": gm_max,
    }
    with open(OUTPUT_DIR / "hdp_suite2_visualizations.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


def _plot_lfp_traces_by_depth(signals: "jtfne.Signals", out_path: Path) -> None:
    import matplotlib.pyplot as plt

    lfp = np.asarray(signals.field.lfp_proxy)  # (n_steps, n_contacts)
    contact_depths = np.asarray(signals.field.contact_depths)
    order = np.argsort(contact_depths)
    time_ms = np.asarray(signals.time_ms)

    fig, ax = plt.subplots(figsize=(10, 12))
    scale = np.nanstd(lfp) or 1.0
    for rank, ci in enumerate(order):
        ax.plot(time_ms, lfp[:, ci] / scale + rank, lw=0.6, color="C0")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{contact_depths[ci]:.3f}" for ci in order], fontsize=6)
    ax.set_ylabel("Contact relative depth (z, superficial->deep)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"LFP-proxy traces, {len(order)} channels, Z-sorted by depth (offset stacked)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
