"""Score the GSDR P3 fit evaluations (PREREG_P3_fit_signed.md). Usage: python score_p3_fit.py <run_dir> [<out_json>].

G1 best loss at gen 30 vs gen 0 (manifest trajectory); G2 mean kappa_500_1000 over trials 14-18 < 0.1;
G3 mean rate over trials in [5, 50] Hz, shown for both rate fields the JSON carries (mean_rate_hz and
e_stim_window_mean_hz) because the prereg names "mean rate" without a field.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
rows = []
for s in (9, 10, 11, 12, 13):
    m = json.loads((root / f"seed{s}" / "manifest.json").read_text())
    traj = m["trajectory"]
    l0 = traj[0]["best_loss"]
    l30 = traj[-1]["best_loss"]
    ks, r_all, r_e, morph, shas = [], [], [], set(), set()
    for t in (14, 15, 16, 17, 18):
        p = root / f"eval_seed{s}_trial{t}.json"
        d = json.loads(p.read_text())
        ks.append(d["kappa_500_1000"])
        r_all.append(d["mean_rate_hz"])
        r_e.append(d["e_stim_window_mean_hz"])
        morph.add(d["morphology"])
        shas.add(d["git_sha"][:8])
    k, ra, re_ = float(np.mean(ks)), float(np.mean(r_all)), float(np.mean(r_e))
    g2 = k < 0.1
    g3a, g3e = 5.0 <= ra <= 50.0, 5.0 <= re_ <= 50.0
    rows.append((s, l0, l30, k, min(ks), max(ks), ra, re_, g2, g3a, g3e, sorted(morph), sorted(shas),
                 m.get("es_clipped_coords")))
print("seed | loss g0 | loss g30 | kappa mean [min,max] | rate all | rate E stim | G2 | G3(all) | G3(E) | morph | sha | clipped")
for r in rows:
    print(f"{r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.4f} [{r[4]:.4f},{r[5]:.4f}] | {r[6]:.2f} | {r[7]:.2f} | "
          f"{r[8]} | {r[9]} | {r[10]} | {r[11]} | {r[12]} | {r[13]}")
passes = sum(1 for r in rows if r[8] and r[9])
print("seeds passing G2 and G3(all):", passes, "of 5; verdict:", "PASS" if passes >= 4 else "FAIL")
if len(sys.argv) > 2:
    files = sorted(root.glob("eval_seed*_trial*.json")) + sorted(root.glob("seed*/manifest.json")) + \
        sorted(root.glob("seed*/g_star_uS_float32.npy"))
    out = {"prereg": "artifacts/etudes/papers/gsdr2026/PREREG_P3_fit_signed.md", "code_sha": "8516a005",
           "receipts_dir": "F:/warehouse/gsdr/P3_fit_8516a005",
           "rate_field_G3": "mean_rate_hz (e_stim_window_mean_hz also shown)",
           "rows": [{"seed": r[0], "loss_g0": r[1], "loss_g30": r[2], "kappa_mean": r[3], "kappa_min": r[4],
                     "kappa_max": r[5], "rate_mean_hz": r[6], "e_stim_rate_mean_hz": r[7], "G2": r[8],
                     "G3": r[9], "morphology": r[11], "es_clipped_coords": r[13]} for r in rows],
           "seeds_passing": passes, "verdict": "PASS" if passes >= 4 else "FAIL",
           "sha256": {f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", sys.argv[2], len(out["sha256"]), "hashes")
