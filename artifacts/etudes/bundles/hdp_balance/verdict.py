"""Apply the README pass criteria P1-P4 to a result JSON."""
import json
import sys

import numpy as np

res = json.load(open(sys.argv[1]))
c = np.array(res["conditions"])
b = np.array(res["test_band"])
r = np.array(res["test_rate_E"])
n = int(res["params"]["n_seeds"])
full, frozen, shuf = b[:n], b[n:2 * n], b[2 * n:3 * n]
learn = np.isin(c, ["full", "shuffled"])
checks = {
    "P1 full>frozen": (int((full > frozen).sum()), round(float((full - frozen).mean()), 3)),
    "P2 full>shuffled": (int((full > shuf).sum()), round(float((full - shuf).mean()), 3)),
}
for k, (w, d) in checks.items():
    print(f"{k}: wins {w}/{n} (need >=13), mean diff {d:+.3f} (need >=0.10) -> {'PASS' if w >= 13 and d >= 0.10 else 'FAIL'}")
ok = (r[learn] >= 2.5) & (r[learn] <= 10.0)
print(f"P3 E rate in [2.5,10] Hz: {int(ok.sum())}/{int(learn.sum())} learning agents, range {r[learn].min():.2f}-{r[learn].max():.2f} -> {'PASS' if ok.all() else 'FAIL'}")
w = b[c == "wired"].mean()
print(f"P4 wired in-band {w:.3f} (need >=0.6) -> {'PASS' if w >= 0.6 else 'FAIL'}")
