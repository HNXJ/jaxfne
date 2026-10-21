"""Plot learned simple-cell kernels (ON minus OFF, 7x7) from a pcl_column weights file."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(path: Path):
    K = np.load(path)["s_exc"].reshape(16, 7, 7, 2)
    fig, ax = plt.subplots(2, 8, figsize=(10, 2.8), constrained_layout=True)
    for f, a in enumerate(ax.ravel()):
        d = K[f, ..., 0] - K[f, ..., 1]
        m = np.abs(d).max() or 1.0
        a.imshow(d, cmap="RdBu_r", vmin=-m, vmax=m)
        a.set(xticks=[], yticks=[], title=f"f{f}")
    fig.suptitle("Simple-cell kernels after training (ON − OFF; red ON, blue OFF)")
    out = path.with_name(path.stem + "_kernels.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
