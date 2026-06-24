"""Channel-resolved synaptic-income trace across the K_ctrl=0/tau_0_ms=20 bifurcation.

hdp_dH_component_trace.py isolated dH_income (alpha*I_syn) as the term that
departs first (t=282.5ms) -- dH_rate stays ~1e-4 and dH_weight is identically
zero (delta=0). This script decomposes I_syn itself by connection class
(E->E, E->PV, PV->E, ...) using the new record_edge_current=True kernel
output (per-edge w*syn_state, the summand segment_sum aggregates into I_syn),
to find which synaptic pathway is responsible for the income-term runaway.

Usage: PYTHONPATH=. python3 scripts/hdp_synaptic_channel_trace.py
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from jaxfne.emitters import simulate_edge_recurrent_izhikevich_hdp
from spectrolaminar_tfne_izhikevich_pipeline import build_config, build_model

OUTPUT_DIR = Path("outputs/hdp_spectrolaminar_pipeline")
BIN_MS = 5.0
WINDOW_START_MS = 250.0
WINDOW_END_MS = 500.0
TAU_0_MS = 20.0


def main() -> None:
    import jax

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()

    print("=== Building laminar column Model (one-time construct) ===")
    model = build_model(cfg)

    hdp_kwargs = dict(cfg["hdp_defaults"])
    hdp_kwargs.update({"K_HDP": 1.0, "tau_0_ms": TAU_0_MS, "K_ctrl": 0.0, "barrier_c": 0.0, "barrier_d": 0.0})
    print(f"hdp_kwargs: {hdp_kwargs}")

    dt_ms = cfg["dt_ms"]
    n_steps = int(round(WINDOW_END_MS / dt_ms))
    key = jax.random.PRNGKey(cfg["seed"])

    edge_list = model.params["edge_list"]
    labels = np.asarray(model.params["emitter"].labels)
    pre = np.asarray(edge_list.pre)
    post = np.asarray(edge_list.post)
    pre_labels = labels[pre]
    post_labels = labels[post]
    cell_types = sorted(set(labels.tolist()))
    classes = list(product(cell_types, cell_types))  # (pre_type, post_type) e.g. ("E", "PV") = E->PV

    class_masks = {f"{a}->{b}": (pre_labels == a) & (post_labels == b) for a, b in classes}
    class_masks = {k: v for k, v in class_masks.items() if v.any()}
    print(f"Connection classes present: {sorted(class_masks.keys())}")

    _, spikes, _, diag = simulate_edge_recurrent_izhikevich_hdp(
        model.params["emitter"], edge_list, n_steps, dt_ms, key,
        record_edge_current=True, **hdp_kwargs,
    )

    edge_current = np.asarray(diag["edge_current_trace"])  # (n_steps, n_edges)

    bin_steps = max(1, int(round(BIN_MS / dt_ms)))
    start_step = int(round(WINDOW_START_MS / dt_ms))
    n_bins = (n_steps - start_step) // bin_steps

    points = []
    for b in range(n_bins):
        s0 = start_step + b * bin_steps
        s1 = s0 + bin_steps
        t_ms = (s0 + s1) / 2.0 * dt_ms
        bin_slice = edge_current[s0:s1]  # (bin_steps, n_edges)
        point: dict[str, Any] = {"t_ms": t_ms}
        for cls, mask in class_masks.items():
            # Total (not per-neuron-averaged) channel contribution to network-wide I_syn this bin.
            point[cls] = float(bin_slice[:, mask].sum() / bin_slice.shape[0])
        points.append(point)
        print(f"t={t_ms:>6.1f}ms  " + "  ".join(f"{k}={v:+.4f}" for k, v in point.items() if k != "t_ms"))

    baseline_n = 6
    def first_departure(cls, n_sigma=3.0):
        baseline = np.array([p[cls] for p in points[:baseline_n]])
        mu, sigma = baseline.mean(), baseline.std()
        sigma = sigma if sigma > 0 else 1e-12
        for i, p in enumerate(points):
            if abs(p[cls] - mu) > n_sigma * sigma:
                return i
        return None

    departures = {cls: first_departure(cls) for cls in class_masks}
    print("\n=== First-departure bin per connection class (>3 sigma vs first 6 bins' baseline) ===")
    ranked = sorted(((idx, cls) for cls, idx in departures.items() if idx is not None))
    for idx, cls in ranked:
        print(f"  {cls}: t={points[idx]['t_ms']}ms (bin {idx})")
    for cls, idx in departures.items():
        if idx is None:
            print(f"  {cls}: no departure detected")

    with open(OUTPUT_DIR / "hdp_synaptic_channel_trace.json", "w") as f:
        json.dump({"points": points, "first_departure": departures, "classes": sorted(class_masks.keys())}, f, indent=2, allow_nan=False)

    _plot(points, departures, sorted(class_masks.keys()), OUTPUT_DIR / "hdp_synaptic_channel_trace.png")
    print(f"\nDone. Results written to {OUTPUT_DIR.resolve()}")


def _plot(points: list[dict[str, Any]], departures: dict[str, int | None], classes: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    t = [p["t_ms"] for p in points]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = cm.tab20(np.linspace(0, 1, len(classes)))
    for cls, color in zip(classes, colors):
        ax.plot(t, [p[cls] for p in points], label=cls, color=color)
        idx = departures.get(cls)
        if idx is not None:
            ax.axvline(points[idx]["t_ms"], color=color, ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.0, color="k", ls="--", lw=0.5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("mean per-step channel current contribution")
    ax.set_title(f"Synaptic-channel decomposition, {WINDOW_START_MS:.0f}-{WINDOW_END_MS:.0f}ms, bin={BIN_MS:.0f}ms")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
