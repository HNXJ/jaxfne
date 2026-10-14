"""Q4: does the sf axis span the cells' range? Fraction of cells with sf* at index 0 or 17.

Two preference sets, both from fig3_analysis and reused:
  - T1/T2 fold (fit on even phases, cells with >= 20 fit spikes): diag_common.fold_state
  - all-phase preference (T3/T4): fig3_analysis.fit_prefs_all (all cells in the set)
Index 0 and index 17 are the lowest and highest sf index (N_SF = 18). Run with -I.
"""
import sys

sys.path.insert(0, "C:/workspace/pcl_diag")

import numpy as np  # noqa: E402

import diag_common as dc  # noqa: E402

fa = dc.fa
data = dc.load_grid()

for name, counts, times, cells in dc.sets_of(data):
    keep, o_star, f_star, flat = dc.fold_state(counts)
    n = keep.size
    e0 = int(np.sum(f_star == 0))
    e17 = int(np.sum(f_star == fa.N_SF - 1))
    print(f"=== {name} ===")
    print(f"T1/T2 fold prefs: n {n}; sf* at index 0: {e0}; at index 17: {e17}; edge fraction {(e0 + e17) / n:.4f}")
    print("T1/T2 fold sf* histogram (index: count):",
          " ".join(f"{i}:{c}" for i, c in enumerate(np.bincount(f_star, minlength=fa.N_SF))))

    o_all, f_all, p_all = fa.fit_prefs_all(counts)
    m = len(f_all)
    a0 = int(np.sum(f_all == 0))
    a17 = int(np.sum(f_all == fa.N_SF - 1))
    print(f"all-phase prefs (T3/T4): n {m}; sf* at index 0: {a0}; at index 17: {a17}; edge fraction {(a0 + a17) / m:.4f}")
    print("all-phase sf* histogram (index: count):",
          " ".join(f"{i}:{c}" for i, c in enumerate(np.bincount(f_all, minlength=fa.N_SF))))
    print("")

print("q4_sf_edges DONE")
