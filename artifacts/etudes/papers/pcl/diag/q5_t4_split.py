"""Q5: T4 amplitudes A(w), A(2w) and L split by pooled count at (ori*, sf*, phase*).

Reuses fig3_analysis: fit_prefs_all (all-phase prefs), _omega_map (reads the
stimulus files read-only, allow_pickle=False), _cell_Ls (L, n_spikes, used omega,
inclusion), _amp_at (A(f)). A(w) and A(2w) use the same pooled spikes (both
trials) that _cell_Ls uses for L. Included cells = the scorer's (count_ok,
band_ok, finite). Checks against the scored JSON. Run with -I.
"""
import os
import sys

DIAG_DIR = os.environ.get("PCL_DIAG_DIR", "")
if not DIAG_DIR:
    sys.exit("PCL_DIAG_DIR is unset: set it to the PCL diagnostic workspace folder")
sys.path.insert(0, DIAG_DIR)

import numpy as np  # noqa: E402

import diag_common as dc  # noqa: E402

fa = dc.fa
data = dc.load_grid()
sc = dc.scored()

counts = {"simple": data["counts_s"], "complex": data["counts_c"]}
times = {"simple": data["times_s"], "complex": data["times_c"]}
prefs = {k: fa.fit_prefs_all(counts[k]) for k in counts}
omap = fa._omega_map(dc.INPUT, prefs["simple"], prefs["complex"], None)
oms = np.array(list(omap.values()), dtype=float)
print(f"stimulus files read for omega: {len(omap)}; omega min {oms.min():.4f} max {oms.max():.4f} median {np.median(oms):.4f}")
print("")

all_ok = True
for name in ("simple", "complex"):
    o_s, f_s, p_s = prefs[name]
    n_cells = len(o_s)
    res = fa._cell_Ls(times[name], prefs[name], omap)
    inc = res["included"]
    n_inc = int(inc.sum())
    w = res["used"]
    aw = np.full(n_cells, np.nan)
    a2w = np.full(n_cells, np.nan)
    for i in range(n_cells):
        o, f, p = int(o_s[i]), int(f_s[i]), int(p_s[i])
        parts = []
        for s in range(fa.N_TRIAL):
            pos, t = times[name][(o, f, p, s)]
            parts.append(np.asarray(t)[pos == i])
        pooled = np.concatenate(parts)
        aw[i] = fa._amp_at(pooled, w[i])
        a2w[i] = fa._amp_at(pooled, 2.0 * w[i])

    L_re = np.log(a2w[inc] / aw[inc])
    max_dev = float(np.max(np.abs(L_re - res["Ls"][inc]))) if n_inc else float("nan")
    med_L = float(np.median(res["Ls"][inc])) if n_inc else float("nan")
    scored_medL = float(sc[f"median_L_{name}"])
    n_excl_scored = int(sc[f"n_excluded_t4_{name}"])
    chk_excl = (n_cells - n_inc) == n_excl_scored
    chk_L = abs(med_L - scored_medL) < 1e-9 and max_dev < 1e-9
    all_ok = all_ok and chk_excl and chk_L

    nsp = res["n_spikes"]
    med_n = float(np.median(nsp[inc])) if n_inc else float("nan")
    above = inc & (nsp > med_n)
    below = inc & (nsp <= med_n)

    print(f"=== {name}: cells {n_cells}, included {n_inc}, excluded {n_cells - n_inc} (scored {n_excl_scored}) ===")
    print(f"median A(w)  over included = {float(np.median(aw[inc])):.4f}")
    print(f"median A(2w) over included = {float(np.median(a2w[inc])):.4f}")
    print(f"median L over included = {med_L:.6f}; scored median_L_{name} = {scored_medL:.6f}; max|L recomputed - L scorer| = {max_dev:.2e}")
    print(f"check excl={chk_excl} L={chk_L}")
    print(f"pooled count at (ori*,sf*,phase*) over included: median {med_n:.1f}")
    for lab, sel in (("above median", above), ("at or below median", below)):
        if np.any(sel):
            print(f"  {lab}: n {int(sel.sum())}, median L {float(np.median(res['Ls'][sel])):.6f}, "
                  f"median A(w) {float(np.median(aw[sel])):.4f}, median A(2w) {float(np.median(a2w[sel])):.4f}")
        else:
            print(f"  {lab}: n 0")
    print("")

print(f"q5_t4_split DONE checks_ok={all_ok}")
