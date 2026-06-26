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
from jaxfne.hdp_network import build_model as _hdp_build_model
from jaxfne.hdp_network import apply_drive_correction as _hdp_apply_drive_correction
from jaxfne.tutorial_utils import spectrolaminar_from_trials

import scripts.hdp_1000_laminar_column_boosted as base


def build_model() -> "jtfne.core.Model":
    """Thin wrapper: hdp_1000_laminar_column_boosted.py's build_model/
    apply_drive_correction were consolidated into the generic
    jaxfne.hdp_network builder (base.CFG carries the same config this
    script previously hardcoded)."""
    return _hdp_build_model(base.CFG)


def apply_drive_correction(model: "jtfne.core.Model") -> "jtfne.core.Model":
    return _hdp_apply_drive_correction(model, base.CFG)

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
    model = build_model()
    model = apply_drive_correction(model)
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

    # --- 4. Extended A-F panel grid (cell density, heatmap, crossing line,
    # LFP/raster last-1000ms, E/I rate table); G is the depth-corrected 3D
    # scatter from step [1] (kept as its own interactive HTML, not embedded
    # -- Plotly 3D and a static matplotlib grid can't share one figure). ---
    print("\n=== [4] Extended A-F panel grid ===")
    build_etude2_panel_grid(
        nt=nt, prof=prof, signals_long=signals_long,
        last_window_ms=1000.0, out_path=OUTPUT_DIR / "etude2_panel_grid_AF.png",
    )
    print(f"Saved {OUTPUT_DIR / 'etude2_panel_grid_AF.png'}")

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
    lfp = np.asarray(signals.field.lfp_proxy)  # (n_steps, n_contacts)
    contact_depths = np.asarray(signals.field.contact_depths)
    time_ms = np.asarray(signals.time_ms)

    fig = jtfne.vis.lfp_traces_stacked_by_depth(time_ms, lfp, contact_depths)
    fig.savefig(out_path, dpi=150)
    jtfne.vis.close_all()


LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6"]
_CT_COLORS = {"E": "#1f77b4", "PV": "#d62728", "SST": "#2ca02c", "VIP": "#9467bd"}


def build_etude2_panel_grid(
    *, nt: list[dict], prof: dict, signals_long: "jtfne.Signals",
    last_window_ms: float, out_path: Path,
) -> None:
    """Etude No. 2 extended A-F static panel grid (dark theme).

    A: cell-type relative density per layer (stacked, fractions sum to 1
       per layer; L1 top -> L6 bottom, matching the package's depth
       convention -- see jaxfne.vis.visualize_network_3d's z-axis fix).
    B: spectrolaminar relative power, depth x frequency heatmap (Gaussian-
       smoothed), reusing the same `prof` spec the 3-panel suite uses.
    C: alpha-beta (blue) / gamma (red) band-power vs depth crossing line.
    D: 32-channel LFP-proxy traces, last `last_window_ms` of the long trial,
       offset-stacked by depth (superficial top -> deep bottom).
    E: spike raster, last `last_window_ms`, z-sorted by depth.
    F: E/I mean firing-rate table per layer (2 rows x 6 layers); I = PV+
       SST+VIP combined.

    G (3D column scatter, E/PV/SST/VIP colored) is NOT part of this grid --
    it is the separate interactive Plotly HTML from visualize_network_3d
    (step [1] of main()); an interactive 3D scene can't be embedded as a
    static matplotlib subplot.
    """
    from scipy.ndimage import gaussian_filter1d as gauss1d

    dt_ms = float(np.asarray(signals_long.time_ms)[1] - np.asarray(signals_long.time_ms)[0])
    win_steps = int(round(last_window_ms / dt_ms))
    spikes_full = np.asarray(signals_long.spikes)         # (T, N)
    lfp_full = np.asarray(signals_long.field.lfp_proxy)   # (T, n_contacts)
    spikes_win = spikes_full[-win_steps:]
    lfp_win = lfp_full[-win_steps:]
    time_win_ms = np.asarray(signals_long.time_ms)[-win_steps:]

    layers = [r["layer"] for r in nt]
    cell_types = [r["cell_type"] for r in nt]
    z = np.array([float(r["z"]) for r in nt])
    contact_depths = np.asarray(signals_long.field.contact_depths)
    ct_order = sorted(set(cell_types), key=lambda ct: ["E", "PV", "SST", "VIP"].index(ct) if ct in ("E", "PV", "SST", "VIP") else 99)

    # ── A: cell-type relative density per layer (stacked, L1 top -> L6 bottom) ──
    layer_density_fracs: dict[str, np.ndarray] = {}
    for ct in ct_order:
        fracs = np.zeros(len(LAYER_ORDER))
        for li, layer in enumerate(LAYER_ORDER):
            layer_mask = [lyr == layer for lyr in layers]
            n_layer = sum(layer_mask)
            if n_layer == 0:
                continue
            n_ct = sum(1 for lyr, c in zip(layers, cell_types) if lyr == layer and c == ct)
            fracs[li] = n_ct / n_layer
        layer_density_fracs[ct] = fracs

    # ── B/C: depth x frequency relative-power heatmap + band-power crossing line ──
    freq_hz = np.asarray(prof["freq_hz"], dtype=float)
    relative_power = np.asarray(prof["relative_power"], dtype=float)  # (n_freq, n_contacts)
    pos_from_l4 = np.asarray(prof["pos_from_l4"], dtype=float) * 1.0e3  # m -> mm
    heat = gauss1d(gauss1d(relative_power, sigma=1.0, axis=0), sigma=1.0, axis=1)
    ab_profile = gauss1d(np.asarray(prof["alpha_beta"], dtype=float), sigma=1.0)
    gm_profile = gauss1d(np.asarray(prof["gamma"], dtype=float), sigma=1.0)

    # ── E: spike raster, last window, z-sorted by depth ─────────────────────────
    neuron_order = np.argsort(z)
    spk_sorted = spikes_win[:, neuron_order]
    t_idx, n_idx = np.where(spk_sorted)

    # ── F: E/I mean firing-rate table per layer (2 x 6) ──────────────────────────
    dur_s = win_steps * dt_ms / 1000.0
    rate_table = np.zeros((2, len(LAYER_ORDER)))  # row0=E, row1=I (PV+SST+VIP)
    for li, layer in enumerate(LAYER_ORDER):
        e_idx = [i for i, (lyr, ct) in enumerate(zip(layers, cell_types)) if lyr == layer and ct == "E"]
        i_idx = [i for i, (lyr, ct) in enumerate(zip(layers, cell_types)) if lyr == layer and ct != "E"]
        rate_table[0, li] = float(spikes_win[:, e_idx].sum() / (len(e_idx) * dur_s)) if e_idx else float("nan")
        rate_table[1, li] = float(spikes_win[:, i_idx].sum() / (len(i_idx) * dur_s)) if i_idx else float("nan")

    fig = jtfne.vis.etude2_extended_panel_grid(
        layer_order=LAYER_ORDER,
        cell_types_present=ct_order,
        ct_colors=_CT_COLORS,
        layer_density_fracs=layer_density_fracs,
        freq_hz=freq_hz,
        relative_power=heat,
        pos_from_l4_mm=pos_from_l4,
        alpha_beta_profile=ab_profile,
        gamma_profile=gm_profile,
        time_win_ms=time_win_ms,
        lfp_win=lfp_win,
        contact_depths=contact_depths,
        spike_times_ms=time_win_ms[t_idx],
        spike_neuron_rank=n_idx,
        n_neurons_zsorted=len(neuron_order),
        rate_table_hz=rate_table,
        k_hdp_used=K_HDP_STABLE,
        last_window_ms=last_window_ms,
    )
    fig.savefig(out_path, dpi=150, facecolor="#1a1a1a", bbox_inches="tight")
    jtfne.vis.close_all()


if __name__ == "__main__":
    main()
