"""P3_v2 signed scorer: five seed manifests (seeds 4-8) to the lane verdict.

Reads the ``manifest.json`` files under one directory and refuses a missing,
duplicate, non-signed or mixed-SHA set. z is the mean over scored seeds per
estimand (prereg line 34: "z is the mean over scored seeds"); each primary
criterion is applied exactly as the prereg table states (R rows with
e = |z-y|/y <= 0.20, prereg lines 39-42; O-IE, prereg line 43; O-ORD, prereg
line 44; CV, prereg line 45; all on the seed-mean values). The verdict is
"closes" only when every primary row holds (prereg line 48). Secondary rows
are reported, never scored (prereg line 46).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCORED_SEEDS = (4, 5, 6, 7, 8)
CLOSE_E = 0.20  # prereg lines 39-42: "close: e <= 0.20 (proposal)"
CV_FLOOR = 0.8  # prereg line 45: "above 0.8 in every population (Fig. 6C)"
LAYERS = ("L23", "L4", "L5", "L6")


def _load_manifests(run_dir):
    """(path, doc) for every manifest.json under run_dir, sorted by path."""
    files = sorted(Path(run_dir).rglob("manifest.json"))
    if not files:
        raise SystemExit("error: no manifest.json under %s" % run_dir)
    docs = []
    for path in files:
        try:
            docs.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError) as exc:
            raise SystemExit("error: cannot read %s: %s" % (path, exc))
    return docs


def _select_entries(docs):
    """One signed entry per scored seed 4-8 with a single code SHA."""
    by_seed = {}
    sha_by_seed = {}
    for path, doc in docs:
        for entry in doc.get("runs", []):
            try:
                seed = int(entry.get("seed"))
            except (TypeError, ValueError):
                continue
            if seed not in SCORED_SEEDS:
                continue
            if entry.get("status") != "signed":
                raise SystemExit(
                    "error: seed %d in %s is %r, not signed"
                    % (seed, path, entry.get("status")))
            if seed in by_seed:
                raise SystemExit("error: duplicate scored seed %d" % seed)
            by_seed[seed] = entry
            sha_by_seed[seed] = doc.get("git_sha")
    missing = [s for s in SCORED_SEEDS if s not in by_seed]
    if missing:
        raise SystemExit("error: missing scored seeds %s" % (missing,))
    shas = {sha_by_seed[s] for s in SCORED_SEEDS}
    if len(shas) != 1:
        raise SystemExit("error: mixed code SHA across seed manifests: %s"
                         % sorted(str(s) for s in shas))
    return [by_seed[s] for s in SCORED_SEEDS], shas.pop()


def _mean(values):
    return float(sum(values)) / float(len(values))


def _score_entries(entries):
    """Primary rows on the seed-mean values, plus the reported secondary."""
    first = entries[0]["estimands_p3v2"]
    pops = sorted(first["rates_hz"])
    mean_rate = {p: _mean([e["estimands_p3v2"]["rates_hz"][p] for e in entries])
                 for p in pops}
    mean_cv = {p: _mean([e["estimands_p3v2"]["CV"]["cv_isi"][p] for e in entries])
               for p in pops}
    rows = {}
    for row in sorted(first["R"]):
        ys = {e["estimands_p3v2"]["R"][row]["y_hz"] for e in entries}
        if len(ys) != 1:
            raise SystemExit("error: mixed y_hz for %s across seeds" % row)
        y = ys.pop()
        z = _mean([e["estimands_p3v2"]["R"][row]["z_hz"] for e in entries])
        e = abs(z - y) / abs(y)
        rows[row] = {"y_hz": y, "z_hz": z, "e": e, "pass": bool(e <= CLOSE_E)}
    oi_layers = {}
    for layer in LAYERS:
        e_rate, i_rate = mean_rate[layer + "E"], mean_rate[layer + "PV"]
        oi_layers[layer] = {"e_hz": e_rate, "i_hz": i_rate,
                            "pass": bool(i_rate > e_rate)}
    rows["O-IE"] = {"pass": bool(all(v["pass"] for v in oi_layers.values())),
                    "layers": oi_layers}
    e_rates = {layer: mean_rate[layer + "E"] for layer in LAYERS}
    order = {"L5E_highest": bool(
        e_rates["L5"] > max(e_rates[layer] for layer in ("L23", "L4", "L6"))),
        "L23E_below_L4E": bool(e_rates["L23"] < e_rates["L4"]),
        "L6E_below_L4E": bool(e_rates["L6"] < e_rates["L4"])}
    order["pass"] = bool(all(order.values()))
    rows["O-ORD"] = order
    rows["CV"] = {"cv_isi": {p: mean_cv[p] for p in sorted(mean_cv)},
                  "pass": bool(all(v > CV_FLOOR for v in mean_cv.values()))}
    secondary = {"SYN_fano_3ms": {
        p: _mean([e["estimands_p3v2"]["SYN_fano_3ms"][p] for e in entries])
        for p in sorted(first["SYN_fano_3ms"])}}
    primary = [rows[k] for k in sorted(first["R"])]
    primary += [rows["O-IE"], rows["O-ORD"], rows["CV"]]
    verdict = "closes" if all(r["pass"] for r in primary) else "off"
    return rows, secondary, verdict


def _inputs_sha256(files):
    """sha256 over the input manifest bytes in sorted path order."""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    """Score one directory of seed manifests into the verdict JSON."""
    parser = argparse.ArgumentParser(prog="score_p3.py")
    parser.add_argument("run_dir")
    parser.add_argument("out_json")
    args = parser.parse_args(argv)
    docs = _load_manifests(args.run_dir)
    entries, sha = _select_entries(docs)
    rows, secondary, verdict = _score_entries(entries)
    doc = {"git_sha": sha,
           "seeds": list(SCORED_SEEDS),
           "rows": rows,
           "secondary": secondary,
           "verdict": verdict,
           "inputs": [str(p) for p, _ in docs],
           "inputs_sha256": _inputs_sha256([p for p, _ in docs])}
    out = Path(args.out_json)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print("verdict: %s" % verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
