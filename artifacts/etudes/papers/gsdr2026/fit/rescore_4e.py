"""Lane 4e fidelity rescoring of held-out seeds 4-8 (CPU-only, stored V traces).

For each (seed, tag) stim trial computes z two ways against the y labels
(``y_labels.json``, data outside the repo):

- paper path: stored mean-voltage PSD (``[100, 1400]`` ms, max-only,
  ``main_fig6.py:407-421``) scored with the paper ``psd_loss``
  (``main_fig6.py:427``) and whole-trial rate penalty (``:429-440``);
- like-for-like path: ``run_gsdr.spike_trace_psd`` (model onsets through the
  label recipe ``main_fig6.py:109-126``), stim-window rates
  (``run_gsdr.window_rates_hz`` on ``[500, 1000]`` ms, the functional match of
  the y ``[600, 1200)`` window), kappa on both windows.

Per-estimand ``e = |z - y| / |y|``; band is close iff every e <= 0.05
(``RESIDUAL_PROTOCOL.md``). Paper verdict context: the Fig. 6 loss targets
the 5-50 Hz band (``main_fig6.py:176-177,432-436``), not the point values
2.14 Hz / 0.0156; kappa is metric-only there (no kappa term).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RUN_DIR = _HERE.parent / "run"

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

Y_LABELS_DEFAULT = "/home/nejath/gsdr_src/y_labels.json"
HELD_OUT_SEEDS = (4, 5, 6, 7, 8)


def rescore(d, y, rg, dl):
    """Score one stored stim-trial dir; returns a row dict."""
    import numpy as np

    grid = np.asarray(y["grid"], dtype=np.float64)
    y_stim = np.asarray(y["stim_psd"], dtype=np.float64)
    manifest = json.loads((d / "manifest.json").read_text())
    entry = manifest["trials"][0]
    v = np.load(d / entry["file"])
    n_steps = v.shape[0]

    z_volt = np.asarray(entry["psd"], dtype=np.float64)
    z_spike = np.asarray(rg.spike_trace_psd(v), dtype=np.float64)

    spikes = rg.detect_spikes(v)
    i0 = min(int(500.0 / rg.DT_MS), n_steps - 2)
    i1 = min(int(1000.0 / rg.DT_MS), n_steps - 1)
    j0 = min(int(600.0 / rg.DT_MS), n_steps - 2)
    j1 = min(int(1200.0 / rg.DT_MS), n_steps - 1)

    stim_rates = rg.window_rates_hz(v, 500.0, 1000.0)
    z = {
        "volt_peak_hz": float(grid[int(np.argmax(z_volt))]),
        "volt_bands": dl.band_powers(z_volt, grid),
        "volt_psd_loss": float(rg.psd_loss(z_volt, y_stim)),
        "spike_peak_hz": float(grid[int(np.argmax(z_spike))]),
        "spike_bands": dl.band_powers(z_spike, grid),
        "spike_psd_loss": float(rg.psd_loss(z_spike, y_stim)),
        "rate_whole_hz": float(np.asarray(entry["rates_hz"]).mean()),
        "rate_penalty": float(entry["rate_penalty"]),
        "rate_stimwin_hz": float(np.asarray(stim_rates).mean()),
        "kappa_500_1000": float(rg.kappa_of_onsets(spikes[i0:i1, :])),
        "kappa_600_1200": float(rg.kappa_of_onsets(spikes[j0:j1, :])),
    }
    yrow = {
        "spike_peak_hz": float(y["stim_peak_hz"]),
        "spike_gamma": float(y["stim_bands"]["gamma"]),
        "spike_beta": float(y["stim_bands"]["beta"]),
        "rate_stimwin_hz": float(y["stim_rates_hz"]["mean"]),
        "kappa": float(y["kappa_stim"]),
    }
    zrow = {"spike_peak_hz": z["spike_peak_hz"],
            "spike_gamma": z["spike_bands"]["gamma"],
            "spike_beta": z["spike_bands"]["beta"],
            "rate_stimwin_hz": z["rate_stimwin_hz"],
            "kappa": z["kappa_500_1000"]}
    erow = {k: abs(zrow[k] - yrow[k]) / abs(yrow[k]) for k in yrow}
    return {"dir": str(d), "y": yrow, "z": zrow, "e": erow,
            "band": ("close" if all(v <= 0.05 for v in erow.values())
                     else "off"),
            "paper_path": {
                "volt_peak_hz": z["volt_peak_hz"],
                "volt_gamma": z["volt_bands"]["gamma"],
                "volt_beta": z["volt_bands"]["beta"],
                "volt_psd_loss": z["volt_psd_loss"],
                "spike_psd_loss": z["spike_psd_loss"],
                "rate_whole_hz": z["rate_whole_hz"],
                "rate_penalty": z["rate_penalty"],
                "kappa_500_1000": z["kappa_500_1000"],
                "kappa_600_1200": z["kappa_600_1200"]}}


def main(argv=None):
    import numpy as np  # noqa: F401 (kept for argv-parity with score_heldout)

    ap = argparse.ArgumentParser(description="Lane 4e fidelity rescoring (CPU).")
    ap.add_argument("--traces-root",
                    default=str(_repo_root / "scratch" / "gsdr2026"))
    ap.add_argument("--seeds", type=int, nargs="*",
                    default=list(HELD_OUT_SEEDS))
    ap.add_argument("--tags", nargs="*", default=["main", "gstar"])
    ap.add_argument("--y-labels", default=Y_LABELS_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(_HERE))
    sys.path.insert(0, str(_RUN_DIR))
    import run_gsdr as rg
    from fit_gsdr import load_y_labels

    import importlib.util

    dl_path = _RUN_DIR / "data_labels.py"
    spec = importlib.util.spec_from_file_location("gsdr2026_dl_rescore",
                                                  dl_path)
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)

    y = load_y_labels(args.y_labels)
    root = Path(args.traces_root)
    rows = []
    for tag in args.tags:
        for s in args.seeds:
            d = root / ("bistab_s%d_%s" % (s, tag))
            if not (d / "manifest.json").exists():
                raise SystemExit("missing trace dir: %s" % d)
            row = rescore(d, y, rg, dl)
            row["seed"] = int(s)
            row["tag"] = tag
            rows.append(row)
    for r in rows:
        print("s%(seed)d/%(tag)s band %(band)s | "
              "spike-peak y %(yp).2f z %(zp).2f e %(ep).3f | "
              "gamma y %(yg).4f z %(zg).4f e %(eg).3f | "
              "beta y %(yb).4f z %(zb).4f e %(eb).3f | "
              "rate y %(yr).2f z %(zr).2f e %(er).3f | "
              "kappa y %(yk).5f z %(zk).4f e %(ek).3f | "
              "paper volt-loss %(vl).4f spike-loss %(sl).4f pen %(pen).4f "
              "k500 %(k5).3f k600 %(k6).3f"
              % {"seed": r["seed"], "tag": r["tag"], "band": r["band"],
                 "yp": r["y"]["spike_peak_hz"],
                 "zp": r["z"]["spike_peak_hz"],
                 "ep": r["e"]["spike_peak_hz"],
                 "yg": r["y"]["spike_gamma"], "zg": r["z"]["spike_gamma"],
                 "eg": r["e"]["spike_gamma"],
                 "yb": r["y"]["spike_beta"], "zb": r["z"]["spike_beta"],
                 "eb": r["e"]["spike_beta"],
                 "yr": r["y"]["rate_stimwin_hz"],
                 "zr": r["z"]["rate_stimwin_hz"],
                 "er": r["e"]["rate_stimwin_hz"],
                 "yk": r["y"]["kappa"], "zk": r["z"]["kappa"],
                 "ek": r["e"]["kappa"],
                 "vl": r["paper_path"]["volt_psd_loss"],
                 "sl": r["paper_path"]["spike_psd_loss"],
                 "pen": r["paper_path"]["rate_penalty"],
                 "k5": r["paper_path"]["kappa_500_1000"],
                 "k6": r["paper_path"]["kappa_600_1200"]}, flush=True)
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps({"y_labels": args.y_labels, "rows": rows},
                                  indent=2))
        print("wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
