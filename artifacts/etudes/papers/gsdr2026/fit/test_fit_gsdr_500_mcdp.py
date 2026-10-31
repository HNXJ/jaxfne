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
