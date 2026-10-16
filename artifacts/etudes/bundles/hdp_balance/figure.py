"""Figure for the balance étude: learning curves (10 s bins) and paired test in-band per seed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"full": "#1f77b4", "frozen": "#7f7f7f", "shuffled": "#d62728", "wired": "#2ca02c"}


def main(path: Path):
    res = json.loads(path.read_text())
    c = np.array(res["conditions"])
    band = np.array(res["train_band"])  # (agents, seconds)
    test = np.array(res["test_band"])
    rate = np.array(res["train_rate_E"])
    n = int(res["params"]["n_seeds"])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    t = np.arange(band.shape[1] // 10) * 10 + 5
    for k, col in COLORS.items():
        m = c == k
        if not m.any():
            continue
        b = band[m][:, : len(t) * 10].reshape(m.sum(), len(t), 10).mean(-1)
        mu, se = b.mean(0), b.std(0) / np.sqrt(m.sum())
        ax[0].plot(t, mu, color=col, label=k)
        ax[0].fill_between(t, mu - se, mu + se, color=col, alpha=0.2, lw=0)
        r = rate[m][:, : len(t) * 10].reshape(m.sum(), len(t), 10).mean(-1).mean(0)
        ax[2].plot(t, r, color=col, label=k)
    ax[0].set(xlabel="training time (s)", ylabel="fraction of time |θ| < 0.25", title="Learning (mean ± s.e.m.)", ylim=(0, 1))
    ax[0].legend(frameon=False, fontsize=8)
    x = np.arange(3)
    for i in range(n):
        ax[1].plot(x, [test[i], test[n + i], test[2 * n + i]], color="0.75", lw=0.8, zorder=1)
    for j, k in enumerate(("full", "frozen", "shuffled")):
        ax[1].scatter(np.full(n, j), test[j * n:(j + 1) * n], color=COLORS[k], s=14, zorder=2)
    if (c == "wired").any():
        ax[1].axhline(test[c == "wired"].mean(), color=COLORS["wired"], ls="--", lw=1, label="wired control")
        ax[1].legend(frameon=False, fontsize=8)
    ax[1].set(xticks=x, xticklabels=["full", "frozen", "shuffled"], ylabel="test fraction in band", title="Test (learning off), paired by seed", ylim=(0, 1))
    ax[2].axhline(5.0, color="k", ls=":", lw=1)
    ax[2].set(xlabel="training time (s)", ylabel="E rate (Hz)", title="Inner loop: E activity (target 5 Hz)")
    out = path.with_name(path.stem + "_figure.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
