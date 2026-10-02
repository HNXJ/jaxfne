#!/usr/bin/env python3
"""Still figure for docs/etudes/experiment_a.md from the FROZEN bundle.

Reads only ``artifacts/etudes/experiment_a/canonical_source.npz`` (plus the
declared post-hoc probe ``project_laminar_sources`` applied to frozen Q);
runs no simulation (simulation_runs = 0). Output:
``docs/assets/etudes/experiment_a.png``.

Usage:
    python scripts/plot_experiment_a_still.py
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUNDLE = ROOT / "artifacts" / "etudes" / "experiment_a" / "canonical_source.npz"
OUT = ROOT / "docs" / "assets" / "etudes" / "experiment_a.png"


def main() -> int:
    import numpy as np

    d = np.load(BUNDLE)
    t = np.asarray(d["time_ms"], dtype=float)
    vm = np.asarray(d["X_V_m"], dtype=float)
    spk = np.asarray(d["X_spikes"], dtype=float)
    H = np.asarray(d["H"], dtype=float)
    Q = np.asarray(d["Q"], dtype=float)
    pos = np.asarray(d["positions"], dtype=float)
    seed = int(np.asarray(d["seed"]))
    dt_ms = float(t[1] - t[0])

    from jaxfne.fields import project_laminar_sources

    lfp = np.asarray(
        project_laminar_sources(Q, pos, n_contacts=16, width=0.10).lfp_proxy
    )
    mid = lfp.shape[1] // 2
    y = lfp.mean(axis=1)
    freqs = np.fft.rfftfreq(y.size, d=dt_ms / 1000.0)
    psd = (np.abs(np.fft.rfft(y - y.mean())) ** 2) / y.size
    band = (freqs >= 1.0) & (freqs <= 150.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jaxfne.vis.evidence_export import save_matplotlib_evidence_figure

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax = axes[0, 0]
    ti, ni = np.nonzero(spk > 0.5)
    ax.scatter(t[ti], ni, s=2, c="k", linewidths=0)
    ax.set_title("Spikes (frozen X)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax = axes[0, 1]
    stride = max(1, len(t) // 2000)
    for j in range(min(8, vm.shape[1])):
        ax.plot(t[::stride], vm[::stride, j], lw=0.8)
    ax.set_title("V_m traces (native)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V_m (mV)")
    ax = axes[0, 2]
    ax.plot(t[::stride], H.mean(axis=1)[::stride], lw=1.0)
    ax.set_title("H mean (identity RBS)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("H (relative)")
    ax = axes[1, 0]
    im = ax.imshow(Q.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Q (frozen source)")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Source index")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Q (relative native current)")
    ax = axes[1, 1]
    ax.plot(t[::stride], lfp[::stride, mid], lw=0.8)
    ax.set_title(f"LFP-proxy contact {mid} from frozen Q")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("LFP (relative proxy)")
    ax = axes[1, 2]
    ax.loglog(freqs[band], psd[band], lw=1.0)
    ax.set_title("Aggregate LFP-proxy spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (relative)")
    fig.suptitle(
        f"Experiment A frozen bundle (seed {seed}, N={vm.shape[1]}, "
        "2000 ms; relative proxy, no resimulation)"
    )
    fig.tight_layout()
    save_matplotlib_evidence_figure(fig, OUT, dpi=140)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]
    print(f"wrote {OUT.relative_to(ROOT)} sha256:{sha} simulation_runs:0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
