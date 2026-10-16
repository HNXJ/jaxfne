"""Q3: distribution of per-cell R_sf in each set.

Per-cell R_sf is DERIVED here (not a scorer output): the scored-fold response of
the cell at its own (ori*, sf*) divided by that cell's own flank (mean over
|dsf| >= 6, from fig3_analysis._sf_flank_per_cell), using fig3_analysis._ratio
for the 0/0 and x/0 rules. Fold and preferences from diag_common.fold_state
(the scorer's T1/T2 fold). Run with -I.
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
    cols = np.arange(n)
    peak_i = flat[o_star * fa.N_SF + f_star, cols]
    flank_i = fa._sf_flank_per_cell(flat, o_star, f_star)
    r_i = np.array([fa._ratio(peak_i[i], flank_i[i]) for i in range(n)])
    n_inf = int(np.sum(np.isinf(r_i)))
    n_nan = int(np.sum(np.isnan(r_i)))
    fin = r_i[np.isfinite(r_i)]
    q1, med, q3 = np.percentile(fin, [25, 50, 75])
    n_ge2_all = int(np.sum(r_i >= 2.0))
    n_ge2_fin = int(np.sum(fin >= 2.0))
    print(f"=== {name}: n_cells {n}, nonfinite inf={n_inf} nan={n_nan} ===")
    print(f"per-cell R_sf (finite): median {med:.4f}  Q1 {q1:.4f}  Q3 {q3:.4f}  min {fin.min():.4f}  max {fin.max():.4f}")
    print(f"cells with R_sf >= 2: {n_ge2_all} of {n} (inf counted), {n_ge2_fin} of {n} (finite only)")
    print(f"per-cell peak (scored fold, own ori*/sf*) median {np.median(peak_i):.4f}; flank median {np.nanmedian(flank_i):.4f}")
    for lab, sel in (("f*=0", f_star == 0), ("f*=17", f_star == 17), ("interior", (f_star > 0) & (f_star < 17))):
        if np.any(sel):
            rr = r_i[sel]
            print(f"  {lab}: n {int(sel.sum())}, median R_sf {np.nanmedian(np.where(np.isfinite(rr), rr, np.nan)):.4f}, n>=2 {int(np.sum(rr >= 2.0))}")
        else:
            print(f"  {lab}: n 0")
    print("")

print("q3_rsf_dist DONE")
