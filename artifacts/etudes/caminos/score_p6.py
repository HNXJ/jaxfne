"""Score CAMINOS P6 v2 (signed 2026-10-13): Q1-Q5 and C1 from two caminos_analysis score files.

Usage: score_p6.py MAIN_SCORE_JSON C1_SCORE_JSON OUT_JSON. Margins are option C of PREREG_P6.md.
"""
import json
import sys
from collections import defaultdict

P5 = {"f_pk": 33.8, "lag": 7.04, "E": 1.106, "PV": 11.37, "SOM": 7.05}
MARGIN = {"f_pk": 1.0, "lag": 0.34, "E": 0.061, "PV": 0.012, "SOM": 0.082}  # Hz, ms, relative
P5_Q4 = {"r1": True, "r2": True, "r3": True, "r4": True}
P5_ONSET = 0.04
SWEEP_P = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)


def _mean(xs):
    return sum(xs) / len(xs)


def score(main, c1):
    runs = main["per_run"]
    reasons = []
    shas = {r["git_sha"] for r in runs + c1["per_run"]}
    if len(shas) != 1:
        reasons.append(f"code SHAs {sorted(shas)}")
    if {r["T"] for r in runs + c1["per_run"]} != {12.0}:
        reasons.append("a run is not T = 12 s")
    p07 = sorted((r for r in runs if abs(r["p"] - 0.07) < 1e-9), key=lambda r: r["seed"])
    if [r["seed"] for r in p07] != [1, 2, 3, 4, 5]:
        reasons.append("p = 0.07 seeds are not 1-5")
    by_p = defaultdict(dict)
    for r in runs:
        by_p[round(r["p"], 2)][r["seed"]] = r
    for p in SWEEP_P:
        if not {1, 2, 3} <= set(by_p[round(p, 2)]):
            reasons.append(f"p = {p} lacks seeds 1-3")
    if len(c1["per_run"]) != 1:
        reasons.append("C1 needs exactly one run")
    if reasons:
        return {"verdict": "incomplete", "reasons": reasons}

    f = _mean([r["peak_freq_E_Hz"] for r in p07])
    lag = _mean([r["lag_PV_SOM_ms"] for r in p07])
    rates = {k: _mean([r["rates_Hz"][k] for r in p07]) for k in ("E", "PV", "SOM")}
    q = {
        "Q1": {"f_pk_Hz": f, "pass": abs(f - P5["f_pk"]) <= MARGIN["f_pk"]},
        "Q2": {"lag_ms": lag, "pass": abs(lag - P5["lag"]) <= MARGIN["lag"]},
        "Q3": {k: {"Hz": v, "rel": abs(v - P5[k]) / P5[k], "pass": abs(v - P5[k]) / P5[k] <= MARGIN[k]}
               for k, v in rates.items()},
        "Q4": {k: {"pass_p6": bool(main[k]["pass"]), "pass": bool(main[k]["pass"]) == P5_Q4[k]}
               for k in P5_Q4},
    }
    q["Q3"]["pass"] = all(v["pass"] for v in q["Q3"].values() if isinstance(v, dict))
    q["Q4"]["pass"] = all(v["pass"] for v in q["Q4"].values() if isinstance(v, dict))
    onset = next((p for p in SWEEP_P
                  if all(by_p[round(p, 2)][s]["oscillates"] for s in (1, 2, 3))), None)
    q["Q5"] = {"onset_p": onset, "pass": onset is not None and abs(onset - P5_ONSET) <= 0.01 + 1e-12}
    run = c1["per_run"][0]
    q["C1"] = {"oscillates": bool(run["oscillates"]), "f_pk_Hz": run["peak_freq_E_Hz"],
               "peak_excess": run["peak_excess_E"], "pass": bool(run["oscillates"])}
    verdict = "pass" if all(q[k]["pass"] for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "C1")) else "fail"
    return {"verdict": verdict, "code_sha": shas.pop(), "criteria": q}


if __name__ == "__main__":
    main_s = json.load(open(sys.argv[1], encoding="utf-8"))
    c1_s = json.load(open(sys.argv[2], encoding="utf-8"))
    out = {"prereg": "PREREG_P6.md signed v2, margins C", "p6": score(main_s, c1_s),
           "main_score": main_s, "c1_score": c1_s}
    with open(sys.argv[3], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("P6", out["p6"]["verdict"], "->", sys.argv[3])
