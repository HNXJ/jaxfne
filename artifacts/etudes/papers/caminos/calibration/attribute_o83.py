"""O-83 attribution: which parameter family makes SOM silencing lower gamma in the cellrep set but not in default.

Usage: python attribute_o83.py <out json> [group ...]
  groups: sizes, connectivity, delay, syn_tau, weights, adaptation, leak_threshold (default: all, in that order)
Each hybrid set is the PLoS ``default`` set with one family taken from ``cellrep``. Per hybrid: SOM silenced at
fraction 0 and 0.9, seed 1, T 12 s (env FIGA_T_S overrides), metrics as in ``score_plos_figs.py`` (f_pk, band
power 25-45 Hz, peak |X| 5-60 Hz, max E rate, population rates); the 0.9/0 band-power ratio is the attribution
readout (default 1.00 seed mean, cellrep 0.44; RESULT_PLOS_FIGS.md, RESULT_FIGURES.md). Rows stream to the JSON
after each hybrid. Exit 0, or 2 on usage.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PAPER = _HERE.parent
_repo_root = Path(__file__).resolve().parents[5]  # P-025: the checkout's jaxfne, never site-packages
for p in (str(_repo_root), str(_PAPER)):
    if p not in sys.path:
        sys.path.insert(0, p)

import caminos_jax as cj  # noqa: E402
import cellrep_figures as cf  # noqa: E402
import run_figA as rf  # noqa: E402  (keep_spectrum: the runner's spectrum trim)

assert "site-packages" not in cf.__file__, "attribute_o83 must run the checkout's cellrep_figures"

GROUPS = {
    "sizes": ("NPV", "NSOM"),
    "connectivity": ("p", "p_E_PV_mult", "ext_sources"),
    "delay": ("tau_d_s",),
    "syn_tau": ("tauE_E_s", "tauE_PV_s", "tauE_SOM_s", "tauI_s"),
    "weights": ("Q_E_E_S", "Q_E_PV_S", "Q_E_SOM_S", "Q_PV_E_S", "Q_PV_PV_S", "Q_SOM_E_S", "Q_SOM_PV_S",
                "Q_PV_SOM_S", "Q_SOM_SOM_S", "Q_ext_S"),
    "adaptation": ("E_b_A", "PV_b_A", "SOM_b_A"),
    "leak_threshold": ("E_EL_V", "PV_EL_V", "SOM_EL_V", "E_Delta_V", "PV_Delta_V", "SOM_Delta_V"),
}


def hybrid(group):
    """The default set with ``group``'s keys from cellrep; syn_tau drops the uniform tauE_s."""
    P = dict(cj.PARAMS)
    if group == "syn_tau":
        P.pop("tauE_s")
    if group == "connectivity" and "p_E_PV_mult" not in P:
        P["p_E_PV_mult"] = 1.0
    for k in GROUPS[group]:
        P[k] = cj.CELLREP_PARAMS[k]
    return P


def peak_mag(row, lo=5.0, hi=60.0):
    f = np.asarray(row["psd_freqs_hz"])
    m = np.asarray(row["psd_mag"])
    return float(m[(f >= lo) & (f <= hi)].max())


def summarize(rows):
    base = next(r for r in rows if r["fraction"] == 0.0)
    out = []
    for r in rows:
        out.append({
            "fraction": r["fraction"],
            "f_pk_hz": float(r["f_pk_hz"]),
            "band_power_ratio": float(r["band_power_25_45"] / base["band_power_25_45"]),
            "peak_mag_ratio": peak_mag(r) / peak_mag(base),
            "max_e_rate_hz": float(r["max_e_rate_hz"]),
            "rates_hz": {k: float(v) for k, v in r["rates_Hz"].items()},
        })
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out_path = sys.argv[1]
    groups = sys.argv[2:] or list(GROUPS)
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        print("unknown groups: %s" % unknown)
        return 2
    t_s = float(os.environ.get("FIGA_T_S", "12"))
    result = {"T_s": t_s, "seed": 1, "fractions": [0.0, 0.9], "groups": {}}
    for g in groups:
        name = "o83_" + g
        cj.PARAM_SETS[name] = hybrid(g)
        cf.PARAM_SET = name
        t0 = time.time()
        rows = rf.keep_spectrum(cf.silence_sweep("SOM", [0.0, 0.9], [1], T_s=t_s), fmax_hz=200.0)
        result["groups"][g] = {"swapped": {k: cj.CELLREP_PARAMS[k] for k in GROUPS[g]},
                               "rows": summarize(rows), "elapsed_s": round(time.time() - t0, 1)}
        s = result["groups"][g]["rows"][1]
        print("group %s band_ratio %.3f f_pk %.1f max_e %.1f elapsed %.0f"
              % (g, s["band_power_ratio"], s["f_pk_hz"], s["max_e_rate_hz"], time.time() - t0), flush=True)
        with open(out_path, "w", newline="\n") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
