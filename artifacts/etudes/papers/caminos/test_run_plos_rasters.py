"""Smoke tests for ``run_plos_rasters.py`` (CAMINOS PLoS Figs 2C-F, 3C-F data runner).

Runs the runner on a small smoke net (``--sizes`` + ``--smoke-drive``,
``FIGA_T_S=2.5``) and checks the on-disk contract: every condition file plus a
manifest whose hashes match, raster/rate consistency, the SOM-silencing mask
effect, and the two refusals.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # P-025: repo root before the covered sibling imports

import cellrep_figures as cf

SEEDS = (7, 8)
SEED = SEEDS[0]
T_SMOKE = 2.5
SMOKE_SIZES = {"NE": 400, "NPV": 80, "NSOM": 80}
ARGS = ["--sizes", "400,80,80", "--smoke-drive"]

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

SEED_ARG = ",".join(str(s) for s in SEEDS)

@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Run the script once on the smoke net; share the output across tests."""
    import run_plos_rasters as rpr

    outdir = tmp_path_factory.mktemp("plosr") / "out"
    old_ts = os.environ.get("FIGA_T_S")
    old_ps = os.environ.get("FIGA_PARAM_SET")
    old_param = cf.PARAM_SET
    os.environ["FIGA_T_S"] = str(T_SMOKE)
    os.environ["FIGA_PARAM_SET"] = "default"
    try:
        rpr.main([str(outdir), SEED_ARG] + ARGS)
    finally:
        if old_ts is None:
            os.environ.pop("FIGA_T_S", None)
        else:
            os.environ["FIGA_T_S"] = old_ts
        if old_ps is None:
            os.environ.pop("FIGA_PARAM_SET", None)
        else:
            os.environ["FIGA_PARAM_SET"] = old_ps
        cf.PARAM_SET = old_param
    return outdir


def _load(name, run_dir):
    with np.load(run_dir / name, allow_pickle=False) as z:
        return dict(z)


def test_run_writes_every_condition_file_and_matching_manifest(run_dir):
    """All nine condition files, five mean-field files and a manifest with
    matching sha256 exist; the manifest stamps seed, T_s, param set and backend."""
    import jax
    import run_plos_rasters as rpr

    assert set(rpr.CONDITIONS) == set(CONDITIONS), rpr.CONDITIONS
    for s in SEEDS:
        for name in CONDITIONS:
            assert (run_dir / ("seed%d" % s) / (name + ".npz")).is_file(), (s, name)
    for name in MF_CONDITIONS:
        assert (run_dir / ("meanfield_" + name + ".npz")).is_file(), name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seeds"] == list(SEEDS), manifest["seeds"]
    assert manifest["T_s"] == T_SMOKE, manifest["T_s"]
    assert manifest["param_set"] == "default", manifest["param_set"]
    assert manifest["backend"] == jax.default_backend(), manifest["backend"]
    assert len(manifest["git_sha"]) == 40, manifest["git_sha"]
    assert set(manifest["conditions"]) == {str(s) for s in SEEDS}, manifest["conditions"]
    assert set(manifest["meanfield"]) == set(MF_CONDITIONS), manifest["meanfield"]
    entries = [(name, entry, "seed%s/%s.npz" % (s, name))
               for s, conds in manifest["conditions"].items() for name, entry in conds.items()]
    entries += [(name, entry, "meanfield_%s.npz" % name)
                for name, entry in manifest["meanfield"].items()]
    assert len(entries) == len(SEEDS) * len(CONDITIONS) + len(MF_CONDITIONS), len(entries)
    for name, entry, rel in entries:
        digest = hashlib.sha256((run_dir / entry["file"]).read_bytes()).hexdigest()
        assert entry["file"] == rel, entry["file"]
        assert digest == entry["sha256"], name
    assert manifest["meanfield_absent"] == [
        "pv_randomized_0.4",
        "pv_randomized_0.9",
        "som_randomized_0.4",
        "som_randomized_0.9",
    ], manifest["meanfield_absent"]


def test_randomization_rate_is_the_paired_baseline_mean(run_dir):
    """The group rate is the mean over all seeds' baselines (run_figA.py
    protocol), not one seed's own baseline; the seeds' baselines differ, so
    the check can fail."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    per_seed = manifest["baseline_rate_hz"]
    assert set(per_seed) == {str(s) for s in SEEDS}, per_seed
    for pop in ("PV", "SOM"):
        rates = [per_seed[str(s)][pop] for s in SEEDS]
        assert rates[0] > 0.0 and rates[0] != rates[1], (pop, rates)
        np.testing.assert_allclose(manifest["group_rate_hz"][pop], np.mean(rates),
                                   rtol=1e-12, err_msg=pop)


def test_raster_sums_match_rate_hz_per_type(run_dir):
    """Per type, the raster column sums equal the counts implied by rate_hz
    (same window, 1 ms bins, Hz per cell)."""
    for name in ["seed%d/%s" % (s, c) for s in SEEDS for c in CONDITIONS]:
        z = _load(name + ".npz", run_dir)
        raster = z["raster"]
        rate = z["rate_hz"]
        slices = z["type_slices"]
        assert raster.dtype == bool, (name, raster.dtype)
        assert raster.shape[0] == 5000, (name, raster.shape)
        assert rate.shape == (500, 3), (name, rate.shape)
        for i, (s0, s1) in enumerate(slices):
            n_cells = int(s1) - int(s0)
            spike_total = int(raster[:, s0:s1].sum())
            implied = float(rate[:, i].sum()) * n_cells * 1e-3
            np.testing.assert_allclose(implied, spike_total, rtol=1e-9, atol=0,
                                       err_msg="%s type %d" % (name, i))
        for key in ("f_pk_hz", "band_power_25_45", "max_e_rate_hz"):
            assert np.asarray(z[key]).ndim == 0, (name, key)


def test_som_silenced_09_leaves_no_spikes_on_masked_cells(run_dir):
    """The exact silenced SOM subset (the simulation mask) is all-zero in the
    raster window; the baseline SOM fires, so the check is not vacuous."""
    import jax

    import caminos_jax as cj

    base = _load("seed%d/baseline.npz" % SEED, run_dir)
    ne, npv, nsom = (int(x) for x in
                     (SMOKE_SIZES["NE"], SMOKE_SIZES["NPV"], SMOKE_SIZES["NSOM"]))
    som = base["raster"][:, ne + npv:]
    assert int(som.sum()) > 0, "precondition: baseline SOM must spike"
    mask = np.asarray(cj.manip_mask(jax.random.PRNGKey(SEED), {"SOM": 0.9},
                                    {"E": ne, "PV": npv, "SOM": nsom}, "silence"))
    assert int(mask.sum()) == int(round(0.9 * nsom)), int(mask.sum())
    sil = _load("seed%d/som_silenced_0.9.npz" % SEED, run_dir)
    sil_som = sil["raster"][:, ne + npv:]
    assert not bool(sil_som[:, mask[ne + npv:]].any()), "silenced SOM cells spiked"


def test_nondefault_param_set_and_nonempty_outdir_are_refused(tmp_path):
    """A non-default FIGA_PARAM_SET and an existing non-empty outdir raise SystemExit."""
    import run_plos_rasters as rpr

    old_ts = os.environ.get("FIGA_T_S")
    old_ps = os.environ.get("FIGA_PARAM_SET")
    os.environ["FIGA_T_S"] = str(T_SMOKE)
    try:
        os.environ["FIGA_PARAM_SET"] = "cellrep"
        with pytest.raises(SystemExit):
            rpr.main([str(tmp_path / "x"), SEED_ARG] + ARGS)
        os.environ["FIGA_PARAM_SET"] = "default"
        full = tmp_path / "full"
        full.mkdir()
        (full / "prior.txt").write_text("prior")
        with pytest.raises(SystemExit):
            rpr.main([str(full), SEED_ARG] + ARGS)
    finally:
        if old_ts is None:
            os.environ.pop("FIGA_T_S", None)
        else:
            os.environ["FIGA_T_S"] = old_ts
        if old_ps is None:
            os.environ.pop("FIGA_PARAM_SET", None)
        else:
            os.environ["FIGA_PARAM_SET"] = old_ps
