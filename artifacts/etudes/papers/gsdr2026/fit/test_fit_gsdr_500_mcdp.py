"""--mcdp mode of fit_gsdr_500 (antithetic-ES gradient + core MCDP update).

Stub ``candidate_loss`` (monkeypatched): no HH simulation runs here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fit_gsdr_500 as f500


def _stub_loss(g, rg, y, *, fit_seeds=(1, 2, 3), morphology="single"):
    comp = {"mean_psd_term": 1.0, "mean_penalty": 0.5,
            "mean_kappa_term": 0.2,
            "trials": [{"active_frac": 0.9}
                       for _ in range(1 + len(tuple(fit_seeds)))]}
    return float(np.asarray(g, dtype=np.float64).sum()), comp


@pytest.fixture
def stubbed(monkeypatch):
    import fit_gsdr

    monkeypatch.setattr(fit_gsdr, "candidate_loss", _stub_loss)
    monkeypatch.setattr(fit_gsdr, "load_y_labels", lambda path=None: {})
    return f500


def test_es_gradient_recovers_quadratic():
    rng = np.random.default_rng(1)
    n = 2
    center = rng.uniform(0.0, 1.0, n)
    target = rng.uniform(0.0, 1.0, n)
    s = 0.1
    eps = [rng.standard_normal(n) for _ in range(64)]
    lp = [float(np.sum((center + s * e - target) ** 2)) for e in eps]
    lm = [float(np.sum((center - s * e - target) ** 2)) for e in eps]
    got = f500.es_gradient(eps, lp, lm, s)
    want = 2.0 * (center - target)
    assert (np.linalg.norm(got - want) / np.linalg.norm(want)) < 0.10


def test_mcdp_factors_change_realized_update(stubbed, tmp_path):
    centers = {}
    for name, val in (("ones", 1.0), ("zeros", 0.0)):
        import unittest.mock as mock

        with mock.patch.object(
                stubbed, "center_mcdp_factors",
                lambda c, rg, fs, morphology="single", _v=val:
                np.full(int(np.asarray(c).shape[0]), _v)):
            _, manifest = stubbed.run_fit_500(
                str(tmp_path / name), generations=10, population=2, seed=0,
                mcdp=True, morphology="single")
        assert manifest["optimizer"]["mcdp_applied"] is True
        z = np.load(tmp_path / name / "ckpt_gen0009.npz")
        centers[name] = np.asarray(z["center"], dtype=np.float64)
    assert not np.allclose(centers["ones"], centers["zeros"])


def test_off_mode_manifest_and_trajectory_unchanged(stubbed, tmp_path):
    _, manifest = stubbed.run_fit_500(
        str(tmp_path / "off"), generations=1, population=3, seed=0,
        mcdp=False, morphology="single")
    opt = manifest["optimizer"]
    assert set(opt) == {"name", "alpha", "exploration", "mcdp_applied",
                        "deselect_factor_consumed",
                        "authors_optimizer_settings_not_used"}
    assert opt["mcdp_applied"] is False
    assert "gradient_estimator" not in manifest
    assert "a_trajectory" not in manifest
    assert [t["generation"] for t in manifest["trajectory"]] == [-1, 0]


def test_odd_population_refused(stubbed, tmp_path):
    with pytest.raises(SystemExit):
        stubbed.run_fit_500(str(tmp_path / "odd"), generations=1,
                            population=3, seed=0, mcdp=True,
                            morphology="single")


def _run_mcdp_base(path, **kw):
    import unittest.mock as mock

    with mock.patch.object(
            f500, "center_mcdp_factors",
            lambda c, rg, fs, morphology="single":
            np.zeros(int(np.asarray(c).shape[0]))):
        return f500.run_fit_500(str(path), generations=10, population=2,
                                seed=0, mcdp=True, morphology="single", **kw)


@pytest.mark.parametrize("field,changed", [
    ("seed", {"seed": 1}),
    ("population", {"population": 4}),
    ("es_sigma", {"es_sigma": 0.25}),
])
def test_resume_refuses_changed_sampling_state(stubbed, tmp_path, field,
                                              changed):
    import shutil

    _run_mcdp_base(tmp_path / "base")
    assert (tmp_path / "base" / "ckpt_gen0009.npz").is_file(), \
        "fixture must write a checkpoint"
    dst = tmp_path / ("resume_" + field)
    shutil.copytree(tmp_path / "base", dst)
    call_kw = {"generations": 10, "population": 2, "seed": 0,
               "es_sigma": None}
    call_kw.update(changed)
    with pytest.raises(SystemExit, match=field):
        f500.run_fit_500(str(dst), mcdp=True, morphology="single",
                         resume=True, **call_kw)


def test_resume_same_sampling_state_continues(stubbed, tmp_path):
    import shutil

    _, manifest = _run_mcdp_base(tmp_path / "base")
    dst = tmp_path / "resumed"
    shutil.copytree(tmp_path / "base", dst)
    _, manifest2 = f500.run_fit_500(str(dst), generations=10, population=2,
                                    seed=0, mcdp=True, morphology="single",
                                    resume=True)
    assert len(manifest2["trajectory"]) >= len(manifest["trajectory"]), \
        "resumed run must keep the prior trajectory"


def test_clipped_pairs_counted(stubbed, tmp_path):
    _, manifest = _run_mcdp_base(tmp_path / "clip", es_sigma=1e6)
    gens = [t for t in manifest["trajectory"] if t["generation"] >= 0]
    assert gens, "fixture must run generations"
    assert all(t["es_clipped_coords"] > 0 for t in gens), \
        "a center at a bound with a huge sigma must clip every generation"
    assert manifest["es_clipped_coords"] == sum(
        t["es_clipped_coords"] for t in gens), "manifest holds the total"
    assert manifest["es_clipped_coords"] > 0


@pytest.mark.parametrize("key", ["seed", "population", "es_sigma"])
def test_resume_refuses_legacy_checkpoint_missing_key(stubbed, tmp_path,
                                                      key):
    """A checkpoint predating the sampling-state record is refused, naming it."""
    import os
    import shutil

    _run_mcdp_base(tmp_path / "base")
    ckpt = tmp_path / "base" / "ckpt_gen0009.npz"
    assert ckpt.is_file(), "fixture must write a checkpoint"
    dst = tmp_path / ("legacy_" + key)
    shutil.copytree(tmp_path / "base", dst)
    legacy = dst / "ckpt_gen0009.npz"
    z = np.load(legacy)
    assert key in z.files, "fixture checkpoint must carry %s to strip it" % key
    keep = {k: z[k] for k in z.files if k != key}
    z.close()
    tmp = dst / "ckpt_gen0009.npz.tmp.npz"
    np.savez(tmp, **keep)
    os.replace(tmp, legacy)
    with pytest.raises(SystemExit, match=key):
        f500.run_fit_500(str(dst), generations=10, population=2, seed=0,
                         mcdp=True, morphology="single", resume=True)
    try:
        err = None
        f500.run_fit_500(str(dst), generations=10, population=2, seed=0,
                         mcdp=True, morphology="single", resume=True)
    except SystemExit as e:
        err = str(e)
    assert err is not None and "predates" in err, \
        "the refusal must say the checkpoint predates the record"


def test_resumed_clip_total_covers_all_generations(stubbed, tmp_path):
    """After a resume the manifest clip total still covers every generation."""
    import shutil

    _run_mcdp_base(tmp_path / "base", es_sigma=1e6)
    assert (tmp_path / "base" / "ckpt_gen0009.npz").is_file(), \
        "fixture must write the gen-10 checkpoint"
    dst = tmp_path / "resumed"
    shutil.copytree(tmp_path / "base", dst)
    _, manifest = f500.run_fit_500(str(dst), generations=20, population=2,
                                   seed=0, es_sigma=1e6, mcdp=True,
                                   morphology="single", resume=True)
    gens = [t["generation"] for t in manifest["trajectory"]
            if t["generation"] >= 0]
    assert gens == list(range(20)), \
        "resumed run must hold generations 0-19, got %r" % (gens,)
    keyed = [t for t in manifest["trajectory"] if "es_clipped_coords" in t]
    assert len(keyed) == 20, "every generation entry must carry the count"
    assert manifest["es_clipped_coords"] == sum(
        t["es_clipped_coords"] for t in keyed), \
        "manifest total must cover pre- and post-resume generations"


def test_nonfinite_epoch_is_noop(stubbed, tmp_path, monkeypatch):
    import jax

    import fit_gsdr
    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    # The driver keeps this call for the all-nonfinite generation; the core
    # contract it relies on (jaxfne/optim/gsdr.py:299-300) is pinned here.
    src = Path(f500.__file__).read_text()
    assert "gsdr.py:299-300" in src, \
        "the driver must cite the core no-op line it relies on"
    transform = gsdr_mcdp_transform(
        deselection_threshold=3.0, a_init=0.0, a_dynamic=True,
        lambda_d=1.0, checkpoint_n=5)
    params = jax.numpy.asarray(np.ones(4), dtype=jax.numpy.float32)
    state = transform.init(params)
    updates, state2 = transform.update(
        jax.numpy.zeros_like(params), state, params=params,
        loss=float("nan"), key=jax.random.PRNGKey(0), mcdp=None)
    assert bool((np.asarray(updates) == 0.0).all()), "voided epoch: zero updates"
    assert state2[0] == state[0], "voided epoch: optimizer state unchanged"
    assert state2[0].step == 0, "voided epoch: not even the step counter moves"

    comp = {"mean_psd_term": 1.0, "mean_penalty": 0.5,
            "mean_kappa_term": 0.2,
            "trials": [{"active_frac": 0.0}, {"active_frac": 0.0},
                       {"active_frac": 0.0}, {"active_frac": 0.0}]}
    monkeypatch.setattr(fit_gsdr, "candidate_loss",
                        lambda g, rg, y, **kw: (float("inf"), comp))
    _run_mcdp_base(tmp_path / "void")
    got = np.asarray(np.load(tmp_path / "void" / "ckpt_gen0009.npz")["center"],
                     dtype=np.float64)
    import run_gsdr as rg

    want = np.clip(np.asarray(
        rg.init_conductances(jax.random.PRNGKey(1)), dtype=np.float64)
        / rg.US_TO_MSCM2, 0.0, 10.0)
    assert np.array_equal(got, want), \
        "all-nonfinite generations must leave the center at init"
