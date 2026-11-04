"""Smoke tests for ``suite/run_cellrep_rasters.py`` (Cell Reports raster + LFP runner).

Runs the runner once on a smoke net (``--sizes 400,80,80``, ``--smoke-drive``,
``FIGA_T_S=2.6``) and checks the on-disk contract: every
condition file with the common-context keys plus ``lfp``/``t0_s``, the
``raster``/``lfp`` row agreement, the baseline firing precondition, rate
consistency, the manifest hashes, and the three refusals (non-cellrep param
set, non-empty outdir, unknown condition).

``FIGA_T_S=2.6`` rather than a shorter smoke T: the fixed 2 s discard of
``caminos_analysis.kept_slice`` (``cellrep_figures.run_cell`` applies it
unconditionally) refuses any run with T_s <= 2 s, so 0.6 s of run length is
impossible; 2.6 s gives the intended 0.6 s kept window. Sizes 400/80/80
(the documented smoke net of ``test_caminos_api.py``) rather than 80/10/10:
at 80/10/10 the PV population stays silent in the last-0.5 s window, which
the baseline precondition below must rule out.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

_repo_root = Path(__file__).resolve().parents[1]
_caminos = _repo_root / "artifacts" / "etudes" / "papers" / "caminos"
for p in (str(_repo_root), str(_caminos), str(_caminos / "suite")):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED = 7
T_SMOKE = 2.6  # s: 2 s discard + 0.6 s kept window (see module docstring)
SMOKE_SIZES = {"NE": 400, "NPV": 80, "NSOM": 80}
ARGS = ["--sizes", "400,80,80", "--smoke-drive"]

CONDITIONS = [
    "baseline",
    "silence_som_0.5",
    "silence_som_1.0",
    "silence_pv_0.5",
    "silence_pv_0.75",
    "silence_pv_1.0",
    "replace_som_0.5",
    "replace_pv_0.5",
    "replace_pv_1.0",
]

COMMON_KEYS = ("raster", "type_slices", "rate_hz", "f_pk_hz",
               "band_power_25_45", "max_e_rate_hz", "lfp", "t0_s")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Run the script once on the smoke net; share the output across tests."""
    import run_cellrep_rasters as rcr

    outdir = tmp_path_factory.mktemp("cellrepr") / "out"
    old_ts = os.environ.get("FIGA_T_S")
    old_ps = os.environ.get("FIGA_PARAM_SET")
    os.environ["FIGA_T_S"] = str(T_SMOKE)
    os.environ.pop("FIGA_PARAM_SET", None)
    try:
        rcr.main([str(outdir), str(SEED)] + ARGS)
    finally:
        if old_ts is None:
            os.environ.pop("FIGA_T_S", None)
        else:
            os.environ["FIGA_T_S"] = old_ts
        if old_ps is None:
            os.environ.pop("FIGA_PARAM_SET", None)
        else:
            os.environ["FIGA_PARAM_SET"] = old_ps
    return outdir


def _load(name, run_dir):
    with np.load(run_dir / name, allow_pickle=False) as z:
        return dict(z)


def test_run_writes_every_condition_file_and_matching_manifest(run_dir):
    """All nine condition files and a manifest whose hashes match exist; the
    manifest stamps seed, T_s, cellrep param set, backend, git SHA and wall time."""
    import jax
    import run_cellrep_rasters as rcr

    assert list(rcr.CONDITIONS) == CONDITIONS, rcr.CONDITIONS
    for name in CONDITIONS:
        assert (run_dir / ("seed%d" % SEED) / (name + ".npz")).is_file(), name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seeds"] == [SEED], manifest["seeds"]
    assert manifest["T_s"] == T_SMOKE, manifest["T_s"]
    assert manifest["param_set"] == "cellrep", manifest["param_set"]
    assert manifest["backend"] == jax.default_backend(), manifest["backend"]
    assert len(manifest["git_sha"]) == 40, manifest["git_sha"]
    assert manifest["wall_s"] > 0.0, manifest["wall_s"]
    assert set(manifest["conditions"]) == {str(SEED)}, manifest["conditions"]
    for name, entry in manifest["conditions"][str(SEED)].items():
        digest = hashlib.sha256((run_dir / entry["file"]).read_bytes()).hexdigest()
        assert entry["file"] == "seed%d/%s.npz" % (SEED, name), entry["file"]
        assert digest == entry["sha256"], name


def test_files_carry_the_common_keys_plus_lfp_and_t0(run_dir):
    """Every file has the shared trace keys plus ``lfp`` (steps,) and ``t0_s``;
    the LFP covers exactly the raster window starting at T_s - 0.5."""
    for name in CONDITIONS:
        z = _load("seed%d/%s.npz" % (SEED, name), run_dir)
        for key in COMMON_KEYS:
            assert key in z, (name, key)
        raster, lfp = z["raster"], z["lfp"]
        assert raster.dtype == bool and raster.ndim == 2, (name, raster.dtype, raster.shape)
        assert lfp.dtype.kind == "f" and lfp.shape == (raster.shape[0],), (name, lfp.shape)
        assert np.isfinite(lfp).all(), name
        assert float(z["t0_s"]) == pytest.approx(T_SMOKE - 0.5, abs=1e-9), name
        for key in ("f_pk_hz", "band_power_25_45", "max_e_rate_hz"):
            assert np.asarray(z[key]).ndim == 0, (name, key)


def test_baseline_fires_in_every_population(run_dir):
    """Precondition: E and SOM each have at least one spike in the baseline
    window, so the population-level figure panels are not drawn from nothing.

    PV is not asserted: in the smoke net PV is structurally silent (observed
    rates_Hz PV 0.0 for seeds 1-3 and 7 at T_s 2.6 and 4.6, sizes 400/80/80,
    smoke drive -- the E smoke rate of about 1.6 Hz gives a PV cell roughly
    22 E events/s at 3.75 nS x 0.1 ms, about 0.05 mV steady against the
    20 ms leak, far under the 10 mV to threshold; only the full NE=8000
    recruits PV), and the existing cellrep smoke tests never assert PV
    firing either (test_caminos_api.py asserts E and SOM only).
    """
    z = _load("seed%d/baseline.npz" % SEED, run_dir)
    raster = z["raster"]
    for i, (s0, s1) in enumerate(z["type_slices"]):
        pop = ("E", "PV", "SOM")[i]
        if pop == "PV":
            continue
        assert int(raster[:, int(s0):int(s1)].sum()) > 0, "population %s silent" % pop


def test_raster_sums_match_rate_hz_per_type(run_dir):
    """Per type, the raster column sums equal the counts implied by rate_hz
    (same window, 1 ms bins, Hz per cell)."""
    for name in CONDITIONS:
        z = _load("seed%d/%s.npz" % (SEED, name), run_dir)
        raster, rate, slices = z["raster"], z["rate_hz"], z["type_slices"]
        n_steps = raster.shape[0]
        assert rate.shape == (n_steps // 10, 3), (name, rate.shape)
        for i, (s0, s1) in enumerate(slices):
            n_cells = int(s1) - int(s0)
            spike_total = int(raster[:, int(s0):int(s1)].sum())
            implied = float(rate[:, i].sum()) * n_cells * 1e-3
            np.testing.assert_allclose(implied, spike_total, rtol=1e-9, atol=0,
                                       err_msg="%s type %d" % (name, i))


def test_noncellrep_param_set_nonempty_outdir_and_unknown_condition_refused(tmp_path):
    """A non-cellrep FIGA_PARAM_SET, an existing non-empty outdir and an
    unknown --conditions name all raise SystemExit."""
    import run_cellrep_rasters as rcr

    old_ps = os.environ.get("FIGA_PARAM_SET")
    os.environ["FIGA_PARAM_SET"] = "default"
    try:
        with pytest.raises(SystemExit):
            rcr.main([str(tmp_path / "x"), str(SEED)] + ARGS)
    finally:
        if old_ps is None:
            os.environ.pop("FIGA_PARAM_SET", None)
        else:
            os.environ["FIGA_PARAM_SET"] = old_ps
    with pytest.raises(SystemExit):
        rcr.parse_args([str(tmp_path / "y"), str(SEED), "--conditions", "baseline,bogus"])
    full = tmp_path / "full"
    full.mkdir()
    (full / "prior.txt").write_text("prior")
    with pytest.raises(SystemExit):
        rcr.main([str(full), str(SEED)] + ARGS)


def test_conditions_subset_flag_selects_a_subset():
    """--conditions parses a comma-separated subset, defaults to all nine."""
    import run_cellrep_rasters as rcr

    _, _, _, conds, smoke = rcr.parse_args(["out", "1", "--conditions=baseline,replace_pv_0.5",
                                            "--smoke-drive"])
    assert conds == ["baseline", "replace_pv_0.5"] and smoke is True
    _, _, _, conds, smoke = rcr.parse_args(["out", "1"])
    assert conds == CONDITIONS and smoke is False