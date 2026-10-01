"""Measure baseline dynamics observables of canonical-v1-column-1000n.

Reports firing rates per cell type (E, PV, SST, VIP), irregularity (CV_ISI),
pairwise spike correlation (20 ms bin r_sc), and LFP spectrolaminar peak.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jaxfne as jtfne
assert "site-packages" not in jtfne.__file__, jtfne.__file__

import numpy as np


def measure_canonical_observables(duration_ms: float = 1000.0, dt_ms: float = 0.5, seed: int = 0) -> dict[str, float]:
    """Simulate canonical-v1-column-1000n and compute reference observables."""
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
    model = jtfne.construct_neuronal_tensor(tensor, seed=seed, duration_ms=duration_ms, dt_ms=dt_ms)
    signals = jtfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)

    spikes = np.asarray(signals.spikes)
    if spikes.shape[0] != int(duration_ms / dt_ms) and spikes.shape[1] == int(duration_ms / dt_ms):
        spikes = spikes.T

    table = model.neuron_table()
    cell_types = np.array([row["cell_type"] for row in table])
    dt_s = dt_ms / 1000.0
    rates = spikes.sum(axis=0) / (duration_ms / 1000.0)

    # CV_ISI on neurons with at least 5 spikes
    cv_list = []
    for i in range(spikes.shape[1]):
        st = np.where(spikes[:, i] > 0)[0] * dt_s
        if len(st) >= 5:
            isis = np.diff(st)
            if np.mean(isis) > 0:
                cv_list.append(float(np.std(isis) / np.mean(isis)))
    mean_cv = float(np.mean(cv_list)) if cv_list else float("nan")

    # 20 ms binned spike count correlation
    steps_20ms = int(20.0 / dt_ms)
    n_bins = spikes.shape[0] // steps_20ms
    b20 = spikes[: n_bins * steps_20ms].reshape(n_bins, steps_20ms, -1).sum(axis=1)
    act = np.where(b20.std(axis=0) > 0)[0]
    if len(act) >= 2:
        corr_mat = np.corrcoef(b20[:, act].T)
        triu_idx = np.triu_indices(len(act), k=1)
        r_sc = float(np.nanmean(corr_mat[triu_idx]))
    else:
        r_sc = float("nan")

    return {
        "rate_E": float(np.mean(rates[cell_types == "E"])),
        "rate_PV": float(np.mean(rates[cell_types == "PV"])),
        "rate_SST": float(np.mean(rates[cell_types == "SST"])),
        "rate_VIP": float(np.mean(rates[cell_types == "VIP"])),
        "cv_isi": mean_cv,
        "r_sc_20ms": r_sc,
    }


def main() -> None:
    res = measure_canonical_observables()
    print("Canonical 1000n reference observables:")
    print(f"  E rate:       {res['rate_E']:.2f} Hz")
    print(f"  PV rate:      {res['rate_PV']:.2f} Hz")
    print(f"  SST rate:     {res['rate_SST']:.2f} Hz")
    print(f"  VIP rate:     {res['rate_VIP']:.2f} Hz")
    print(f"  CV_ISI:       {res['cv_isi']:.3f}")
    print(f"  r_sc (20 ms): {res['r_sc_20ms']:.3f}")


if __name__ == "__main__":
    main()
