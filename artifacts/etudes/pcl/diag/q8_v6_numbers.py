"""Unscored: PCL T2 v6 candidate numbers on the v4 runs, even phases.

Complex: per point j, sd_col_j (ddof 1) over the four columns' own plot_fqcy curves; tolerance
eps + z*sqrt(1.25)*sd_col_j; guard SSE <= 0.5*SSE_flat + 1.25*sum sd_col_j^2.
Simple: point test (v5) fraction and level |mean ours - mean fig| over scored points present in ours.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(sys.argv[1])))
import fig3_analysis as fa  # noqa: E402
import fig3_v5_t2 as v5  # noqa: E402

run_dir = sys.argv[2]
ref = v5.load_reference()
sets = fa.cell_sets()
counts_s, _, counts_c, _ = fa._load_grid(run_dir, sets["simple"], sets["complex"])
ph = v5.EVEN_PHASES

pool = v5.score_set(counts_c, "complex", v5.PANEL_COMPLEX, ph, ref)
n_col = counts_c.shape[4] // len(fa.COMPLEX_COLS)
cols = [v5.tuning_curve(counts_c[..., k * n_col:(k + 1) * n_col], "complex", ph)[0]
        for k in range(len(fa.COMPLEX_COLS))]
sse = flat = extra = 0.0
ok = 0
pts = pool["points"]
fig_vals = [p["fig"] for p in pts]
m = float(np.mean(fig_vals))
for p in pts:
    key = v5._key(p["j"]) if hasattr(v5, "_key") else str(p["j"])
    vals = [c[key]["mean"] for c in cols if key in c]
    sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
    ours = p["ours"]
    tol = v5.EPS + v5.Z * np.sqrt(1.25) * sd
    good = ours is not None and abs(ours - p["fig"]) <= tol
    ok += good
    sse += (0.0 if ours is None else ours - p["fig"]) ** 2 if ours is not None else p["fig"] ** 2
    flat += (m - p["fig"]) ** 2
    extra += 1.25 * sd ** 2
    print(f"j {p['j']:.1f} ours {ours if ours is None else round(ours, 3)} fig {p['fig']:.3f} "
          f"sd_col {sd:.3f} (n {len(vals)}) tol {tol:.3f} ok {good}")
print(f"complex: frac {ok}/{len(pts)} = {ok/len(pts):.3f}; SSE {sse:.4f}; "
      f"0.5*flat {0.5*flat:.4f}; 1.25*sum sd^2 {extra:.4f}; limit {0.5*flat+extra:.4f}")

simple = v5.score_set(counts_s, "simple", v5.PANEL_SIMPLE, ph, ref)
sp = [p for p in simple["points"] if p["ours"] is not None]
lvl = abs(np.mean([p["ours"] for p in sp]) - np.mean([p["fig"] for p in sp]))
print(f"simple: frac {simple['frac']:.3f}; level |mean ours - mean fig| {lvl:.3f} (eps {v5.EPS})")
