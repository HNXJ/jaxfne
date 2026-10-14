"""Q2: population sf curve at ori*, aligned to sf*, folded as the scorer does.

Fit preferences on even phases, score on odd phases, trials pooled (the scorer's
T1/T2 fold). Uses fig3_analysis._sf_curve and _sf_flank_per_cell unchanged.
Checks the recomputed R_sf against fig3_analysis.t1_t2 and the scored JSON.
Run with -I.
"""
import sys

sys.path.insert(0, "C:/workspace/pcl_diag")

import numpy as np  # noqa: E402

import diag_common as dc  # noqa: E402

fa = dc.fa
data = dc.load_grid()
sc = dc.scored()
all_ok = True

for name, counts, times, cells in dc.sets_of(data):
    keep, o_star, f_star, flat = dc.fold_state(counts)
    curve = fa._sf_curve(flat, o_star, f_star)
    flank = fa._sf_flank_per_cell(flat, o_star, f_star)
    fl_mean = float(np.nanmean(flank))
    peak = float(curve[fa.N_SF - 1])
    r_sf = fa._ratio(peak, fl_mean)

    chk = fa.t1_t2(counts, np.random.default_rng(0), n_null=1)
    same_pref = chk["o_star"] == o_star.tolist() and chk["f_star"] == f_star.tolist()
    scored_r = float(sc[f"t2_r_sf_{name}"])
    d_t1 = abs(r_sf - chk["r_sf"])
    d_sc = abs(r_sf - scored_r)
    ok = bool(same_pref and d_t1 < 1e-12 and d_sc < 1e-9)
    all_ok = all_ok and ok

    print(f"=== {name}: kept cells {keep.size} of {len(cells)}; t1_t2 n_excluded {chk['n_excluded']} ===")
    print(f"peak (dsf=0) mean response = {peak:.6f}")
    print(f"per-cell flank |dsf|>=6, mean over cells = {fl_mean:.6f}; median = {float(np.nanmedian(flank)):.6f}")
    print(f"R_sf recomputed = {r_sf:.6f}; t1_t2 r_sf = {chk['r_sf']:.6f}; scored JSON = {scored_r:.6f}")
    print(f"check same_pref={same_pref} |recomp-t1_t2|={d_t1:.2e} |recomp-scored|={d_sc:.2e} ok={ok}")
    print("dsf   mean_resp   n_cells_contributing")
    for k, d in enumerate(range(-(fa.N_SF - 1), fa.N_SF)):
        n_ok = int(np.sum((f_star + d >= 0) & (f_star + d < fa.N_SF)))
        print(f"{d:+3d}   {curve[k]:.6f}   {n_ok}")
    for dd in (1, 2, 3, 6, 9, 12, 17):
        left = curve[(fa.N_SF - 1) - dd]
        right = curve[(fa.N_SF - 1) + dd]
        print(f"|dsf|={dd:2d}: dsf=-{dd} {left:.6f}  dsf=+{dd} {right:.6f}  ratio_to_peak(mean L/R) {((left + right) / 2) / peak:.4f}")
    print("")

print(f"q2_sf_curve DONE checks_ok={all_ok}")
