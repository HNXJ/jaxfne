"""Unscored PCL T2 sweep over analysis choices on one run directory.

Usage: pcl_t2_sweep.py PORT_DIR RUN_DIR OUT_JSON. Phase subsets x simple column; complex as the
four-column pool and each column alone. Criterion and reference are fig3_v5_t2's, unchanged.
"""
import json
import sys
from pathlib import Path

port = Path(sys.argv[1])
sys.path.insert(0, str(port))
import fig3_analysis as fa  # noqa: E402
import fig3_v5_t2 as v5  # noqa: E402

run_dir, out = sys.argv[2], Path(sys.argv[3])
ref = v5.load_reference()
PHASES = {"even": (0, 2, 4, 6, 8), "odd": (1, 3, 5, 7, 9), "all": tuple(range(10)),
          "0-4": (0, 1, 2, 3, 4), "5-9": (5, 6, 7, 8, 9)}
SIMPLE_COLS = ((4, 4), (3, 4), (4, 3), (4, 5), (5, 4))


def brief(res):
    return {"frac": round(res["frac"], 3), "sse": round(res["sse"], 4),
            "guard_limit": round(v5.GUARD * res["sse_flat"], 4),
            "pass_frac": res["pass_frac"], "guard": res["guard"], "t2": res["t2_pass"]}


rows = []
grids = {}
for col in SIMPLE_COLS:
    try:
        sets = fa.cell_sets(simple_col=col)
    except ValueError as e:
        print("skip", col, e)
        continue
    counts_s, _, counts_c, _ = fa._load_grid(run_dir, sets["simple"], sets["complex"])
    grids[col] = (counts_s, counts_c)
for pname, ph in PHASES.items():
    for col, (counts_s, counts_c) in grids.items():
        r = brief(v5.score_set(counts_s, "simple", v5.PANEL_SIMPLE, ph, ref))
        rows.append({"set": f"simple {col}", "phases": pname, **r})
    counts_c = grids[(4, 4)][1]
    rows.append({"set": "complex pool", "phases": pname,
                 **brief(v5.score_set(counts_c, "complex", v5.PANEL_COMPLEX, ph, ref))})
    n_col = counts_c.shape[4] // len(fa.COMPLEX_COLS)
    for k, c in enumerate(fa.COMPLEX_COLS):
        sub = counts_c[..., k * n_col:(k + 1) * n_col]
        rows.append({"set": f"complex {c}", "phases": pname,
                     **brief(v5.score_set(sub, "complex", v5.PANEL_COMPLEX, ph, ref))})
for r in rows:
    print(f"{r['set']:16s} {r['phases']:5s} frac {r['frac']:.3f} sse {r['sse']:.4f} "
          f"(limit {r['guard_limit']:.4f}) t2 {r['t2']}")
out.write_text(json.dumps({"run_dir": run_dir, "rows": rows}, indent=1) + "\n", encoding="utf-8")
