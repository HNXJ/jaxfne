"""Tests for ``score_plos_rasters.py`` (CAMINOS PLoS Figs 2C-F / 3C-F scorer).

Builds synthetic run directories in the real on-disk format (small N, 500
1 ms bins at dt 1e-4, raster + rate_hz + scalars + mean-field traces +
manifest with correct sha256) and checks the scorer's estimands, criteria
and figures. Each test asserts its fixture's precondition.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_plos_rasters as spr  # noqa: E402

NE, NPV, NSOM = 40, 10, 10
N_STEPS, BIN_STEPS = 5000, 10
N_BINS = N_STEPS // BIN_STEPS
BIN_S = 1e-3
T_S = 12.0

CONDITIONS = [
    "baseline",
    "pv_silenced_0.4",
    "pv_silenced_0.9",
    "som_silenced_0.4",
    "som_silenced_0.9",
    "pv_randomized_0.4",
    "pv_randomized_0.9",
    "som_randomized_0.4",
    "som_randomized_0.9",
]
MF_CONDITIONS = [
    "baseline",
    "pv_silenced_0.4",
    "pv_silenced_0.9",
    "som_silenced_0.4",
    "som_silenced_0.9",
]
POP_INDEX = {"E": 0, "PV": 1, "SOM": 2}


def _trace(mean, rhythm_hz=0.0, depth=0.8):
    """Per-bin Hz-per-cell trace, optionally carrying a sine rhythm."""
    t = (np.arange(N_BINS) + 0.5) * BIN_S
    out = np.full(N_BINS, float(mean))
    if rhythm_hz:
        out = out * (1.0 + depth * np.sin(2.0 * np.pi * rhythm_hz * t))
    return out


def _raster_from_traces(e_rate, pv_rate, som_rate):
    """Deterministic raster whose per-bin sums follow the given rate traces."""
    raster = np.zeros((N_STEPS, NE + NPV + NSOM), bool)
    ptr = 0
    for rates, n_cells, off in ((e_rate, NE, 0), (pv_rate, NPV, NE),
                                (som_rate, NSOM, NE + NPV)):
        acc = 0.0
        for b in range(N_BINS):
            acc += float(rates[b]) * n_cells * BIN_S
            count = int(np.floor(acc))
            acc -= count
            for j in range(count):
                flat = (ptr * 11 + j * 17 + b * 5) % (BIN_STEPS * n_cells)
                raster[b * BIN_STEPS + flat % BIN_STEPS, off + flat // BIN_STEPS] = True
                ptr += 1
    return raster


def _condition_npz(e=8.0, pv=9.0, som=5.0, rhythm_hz=0.0, f_pk=30.0,
                   band=1.0, maxe=12.0):
    raster = _raster_from_traces(_trace(e, rhythm_hz), _trace(pv), _trace(som))
    rate = np.empty((N_BINS, 3))
    for i, (s0, s1) in enumerate(((0, NE), (NE, NE + NPV), (NE + NPV, NE + NPV + NSOM))):
        n_cells = s1 - s0
        per_bin = raster[:, s0:s1].reshape(N_BINS, BIN_STEPS, n_cells).sum(axis=(1, 2))
        rate[:, i] = per_bin / (n_cells * BIN_S)
    return {
        "raster": raster,
        "type_slices": np.array([[0, NE], [NE, NE + NPV], [NE + NPV, NE + NPV + NSOM]]),
        "rate_hz": rate,
        "f_pk_hz": np.asarray(float(f_pk)),
        "band_power_25_45": np.asarray(float(band)),
        "max_e_rate_hz": np.asarray(float(maxe)),
    }


def _meanfield_npz(e=8.0, pv=9.0, som=5.0, e_rhythm_hz=0.0, e_mod=0.0,
                   pv_sil=0.0, som_sil=0.0):
    t = (T_S - 0.5) + (np.arange(N_STEPS) + 0.5) * 1e-4
    nu = np.empty((N_STEPS, 3))
    nu[:, 0] = e * (1.0 + e_mod * np.sin(2.0 * np.pi * e_rhythm_hz * t)) if e_rhythm_hz \
        else e
    nu[:, 1] = pv
    nu[:, 2] = som
    return {
        "t_s": t,
        "nu_hz": nu,
        "w": np.zeros(N_STEPS),
        "pv_silence": np.asarray(float(pv_sil)),
        "som_silence": np.asarray(float(som_sil)),
    }


def make_run(root, seeds=(1, 2), net=None, mf=None):
    """Write a synthetic run directory in the `run_plos_rasters.py` format."""
    net = net or {}
    mf = mf or {}
    root = Path(root)
    conditions, mfields = {}, {}
    for s in seeds:
        sdir = root / ("seed%d" % s)
        sdir.mkdir(parents=True)
        conditions[str(s)] = {}
        for name in CONDITIONS:
            arrays = _condition_npz(**net.get(name, {}))
            rel = "seed%d/%s.npz" % (s, name)
            np.savez(root / rel, **arrays)
            conditions[str(s)][name] = {
                "file": rel,
                "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest(),
            }
    for name in MF_CONDITIONS:
        arrays = _meanfield_npz(**mf.get(name, {}))
        rel = "meanfield_%s.npz" % name
        np.savez(root / rel, **arrays)
        mfields[name] = {
            "file": rel,
            "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest(),
        }
    manifest = {
        "seeds": list(seeds),
        "T_s": T_S,
        "conditions": conditions,
        "meanfield": mfields,
        "meanfield_absent": [c for c in CONDITIONS if c not in MF_CONDITIONS],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return root


def _score(run, tmp_path, name, extra=()):
    out = tmp_path / name
    rc = spr.main([str(run), str(out)] + list(extra))
    assert rc == 0, "scorer exit %r" % rc
    return json.loads(out.read_text(encoding="utf-8"))


def test_f_recovers_known_30hz_rhythm(tmp_path):
    """Network and mean-field f recover a planted 30 Hz E rhythm (2 Hz grid)."""
    run = make_run(tmp_path / "run", net={"baseline": {"e": 20.0, "rhythm_hz": 30.0}},
                   mf={"baseline": {"e": 20.0, "e_rhythm_hz": 30.0, "e_mod": 0.5}})
    net_e = np.load(run / "seed1" / "baseline.npz")["rate_hz"][:, 0]
    assert net_e.max() - net_e.min() > 0, "precondition: network E trace must carry the rhythm"
    with np.load(run / "meanfield_baseline.npz") as z:
        t, nu = np.asarray(z["t_s"]), np.asarray(z["nu_hz"])
    _, binned = spr.bin_meanfield(t, nu)
    assert binned[:, 0].max() - binned[:, 0].min() > 0, \
        "precondition: mean-field E trace must carry the rhythm"
    d = _score(run, tmp_path, "o.json")
    assert abs(d["per_seed"]["1"]["baseline"]["f_hz"] - 30.0) <= 2.0
    assert abs(d["meanfield"]["baseline"]["f_hz"] - 30.0) <= 2.0


def test_silenced_population_correction(tmp_path):
    """PV silenced 0.4 with N_PV 10 reports the per-unsilenced-cell rate."""
    assert int(round(0.4 * NPV)) == 4, "precondition: 4 of 10 PV cells masked"
    run = make_run(tmp_path / "run",
                   net={"pv_silenced_0.4": {"e": 12.0, "pv": 9.0, "som": 5.0}})
    d = _score(run, tmp_path, "o.json")
    got = d["per_seed"]["1"]["pv_silenced_0.4"]
    raw = got["raw_mean_rate_hz"]["PV"]
    assert raw > 0, "precondition: raw PV rate must be nonzero"
    assert got["mean_rate_hz"]["PV"] == pytest.approx(raw * NPV / (NPV - 4))
    assert got["mean_rate_hz"]["E"] == pytest.approx(got["raw_mean_rate_hz"]["E"])


def test_tampered_file_exits_2_naming_it(tmp_path, capsys):
    """One flipped byte in an input exits 2 and names the file."""
    run = make_run(tmp_path / "run")
    assert spr.main([str(run), str(tmp_path / "ok.json")]) == 0, \
        "precondition: untampered run must score 0"
    target = run / "seed1" / "baseline.npz"
    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 1
    target.write_bytes(bytes(raw))
    rc = spr.main([str(run), str(tmp_path / "bad.json")])
    assert rc == 2
    assert "seed1/baseline.npz" in capsys.readouterr().err


def test_2e_pass_and_mf_cv_fail(tmp_path):
    """2E passes on 15x network E with a flat 150 Hz mean field; CV 0.2 fails it."""
    net = {"baseline": {"e": 10.0}, "pv_silenced_0.9": {"e": 150.0}}
    mf_ok = {"baseline": {"e": 8.0}, "pv_silenced_0.9": {"e": 150.0}}
    d = _score(make_run(tmp_path / "ok", net=net, mf=mf_ok), tmp_path, "ok.json")
    panel = d["panels"]["2E"]
    assert panel["criteria"][0]["values"]["network_e_ratio"] == pytest.approx(15.0), \
        "precondition: network E must sit at 15x baseline"
    assert panel["criteria"][1]["values"]["mf_e_mean_hz"] == pytest.approx(150.0)
    assert panel["criteria"][1]["values"]["mf_e_cv"] < 0.05, \
        "precondition: flat mean field must have CV below 0.05"
    assert panel["verdict"] == "pass"
    mf_bad = {"baseline": {"e": 8.0},
              "pv_silenced_0.9": {"e": 150.0, "e_rhythm_hz": 30.0, "e_mod": 0.2 * np.sqrt(2)}}
    d2 = _score(make_run(tmp_path / "bad", net=net, mf=mf_bad), tmp_path, "bad.json")
    panel2 = d2["panels"]["2E"]
    assert panel2["criteria"][1]["values"]["mf_e_cv"] == pytest.approx(0.2, abs=0.02), \
        "precondition: modulated mean field must carry CV near 0.2"
    assert panel2["criteria"][0]["pass"] is True and panel2["criteria"][1]["pass"] is False
    assert panel2["verdict"] == "partial", "the two 2E claims split: network seizure holds, fixed point fails"


def test_3d_direction_matches_2d(tmp_path):
    """3D passes when both ratios move the same way as 2D's; one flip fails it."""
    base = {"baseline": {"f_pk": 30.0, "band": 1.0},
            "som_silenced_0.4": {"f_pk": 25.0, "band": 0.8}}
    d = _score(make_run(tmp_path / "ok", net=dict(
        base, **{"som_randomized_0.4": {"f_pk": 26.0, "band": 0.85}})),
        tmp_path, "ok.json")
    assert d["seed_means"]["som_silenced_0.4"]["f_pk_ratio"] < 1.0, \
        "precondition: 2D f_pk ratio must sit below 1"
    assert d["seed_means"]["som_silenced_0.4"]["band_ratio"] < 1.0, \
        "precondition: 2D band ratio must sit below 1"
    assert d["panels"]["3D"]["verdict"] == "pass"
    d2 = _score(make_run(tmp_path / "bad", net=dict(
        base, **{"som_randomized_0.4": {"f_pk": 35.0, "band": 0.85}})),
        tmp_path, "bad.json")
    assert d2["seed_means"]["som_randomized_0.4"]["f_pk_ratio"] > 1.0, \
        "precondition: flipped f_pk ratio must sit above 1"
    assert d2["panels"]["3D"]["verdict"] == "fail"


def test_figdir_writes_both_pngs(tmp_path):
    """--figdir writes non-empty fig2_CF.png and fig3_CF.png."""
    run = make_run(tmp_path / "run",
                   net={"baseline": {"e": 20.0, "rhythm_hz": 30.0}},
                   mf={"baseline": {"e": 20.0, "e_rhythm_hz": 30.0, "e_mod": 0.5}})
    raster = np.load(run / "seed1" / "baseline.npz")["raster"]
    assert int(raster.sum()) > 0, "precondition: fixture raster must hold spikes"
    figdir = tmp_path / "figs"
    d = _score(run, tmp_path, "o.json", extra=("--figdir", str(figdir)))
    assert d["figures"]["fig2_CF.png"].endswith("fig2_CF.png")
    for name in ("fig2_CF.png", "fig3_CF.png"):
        assert (figdir / name).is_file(), name
        assert (figdir / name).stat().st_size > 0, name
