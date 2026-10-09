"""F2 divergence: verdict logic on synthetic series, and the lockstep base arm against pcl_column_hdp.block."""
from __future__ import annotations

import json
import math
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne  # noqa: F401

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

import artifacts.etudes.theory_f2.f2_divergence as F  # noqa: E402

N = 600


def _write(tmp_path, seed, dw_nudge, dw_ref, ds_frozen):
    zeros = [0.0] * N
    pairs = {
        "base_nudge": {"D_w": [float(x) for x in dw_nudge], "D_s": zeros},
        "base_ref": {"D_w": [float(x) for x in dw_ref], "D_s": zeros},
        "frozen": {"D_w": zeros, "D_s": [float(x) for x in ds_frozen]},
    }
    path = tmp_path / f"synthetic_seed{seed}.json"
    path.write_text(json.dumps(dict(seed=seed, n_seq=N, pairs=pairs)))
    return path


def test_verdicts_recovers_exponential_rate_window_and_horizon(tmp_path):
    n = np.arange(1, N + 1)
    lam, d_sat = 0.05, 0.5
    dw_nudge = 1e-6 * np.exp(lam * n)  # noiseless: ln D_w is exactly linear in n
    ds_flat = 0.1 + 0.01 * np.random.default_rng(0).standard_normal(N)
    path = _write(tmp_path, 20, dw_nudge, np.full(N, d_sat), ds_flat)
    out = F.verdicts([path])
    an = out["per_seed"][20]
    assert an["d_sat"] == pytest.approx(d_sat)
    assert an["growth"]["lambda_per_seq"] == pytest.approx(lam, rel=0.01)
    # window: 1e-6 <= D_w <= 0.1 * D_sat = 0.05  <=>  n <= ln(5e4) / 0.05 = 216.4
    assert an["growth"]["window"] == [1, 216]
    assert an["growth"]["n_window"] == 216
    # horizon: D_w >= 0.5 * D_sat = 0.25  <=>  n >= ln(2.5e5) / 0.05 = 248.6
    assert an["n_star"] == math.ceil(math.log(0.5 * d_sat / 1e-6) / lam) == 249
    assert an["verdicts"] == {"H2.1": "supported", "H2.2": "supported", "H2.3": "supported"}
    assert out["aggregate"]["H2.2"] == "supported"


def test_flat_growth_rejects_h21(tmp_path):
    noise = np.random.default_rng(1).standard_normal(N)
    dw_flat = 1e-4 * np.exp(0.01 * noise)  # flat in ln D_w, with 1% scatter
    path = _write(tmp_path, 21, dw_flat, np.full(N, 0.5), np.full(N, 0.1))
    an = F.verdicts([path])["per_seed"][21]
    assert an["growth"]["n_window"] == N  # every sequence is inside the window
    lo, hi = an["growth"]["ci95"]
    assert lo <= 0 <= hi, (lo, hi)
    assert an["verdicts"]["H2.1"] == "rejected"
    assert an["n_star"] is None and an["verdicts"]["H2.2"] == "rejected"


def test_lockstep_base_equals_block_after_three_sequences(tmp_path):
    seed = 20
    t0 = time.time()
    res = F.main(["--seed", str(seed), "--n-seq", "3", "--out", str(tmp_path / "smoke.json")])
    doc = json.loads((tmp_path / "smoke.json").read_text())
    assert doc["n_seq"] == 3 and len(doc["pairs"]["base_nudge"]["D_w"]) == 3

    # reference: pcl_column_hdp.main's phase-1 setup and P.block on the same keys with n_exc = 3
    P = F.P
    P.K.DT = P.DT
    amp, w_cancel = P.K.calibrate()
    s = w_cancel / 5.3
    rng = np.random.default_rng(seed)
    k_p1 = jax.random.split(jax.random.PRNGKey(seed), 4)[0]
    w0 = P.nudge_ulp(P.init_weights(rng, s, (0, 1, 4, 5)), 0)
    w_ref, *_ = P.block(P.make_runner(s, amp, (0, 1, 4, 5)), jnp.asarray(w0, jnp.float32), k_p1, 3)
    w_ref = np.asarray(w_ref)

    assert np.any(w_ref != np.asarray(w0, np.float32)), "fixture must move the weights"
    np.testing.assert_array_equal(res["w_base_final"], w_ref)
    print(f"test_lockstep wall {time.time() - t0:.1f} s")
