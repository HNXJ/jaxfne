"""0.5.5 ATLAS 6: AT-10 on N_20 (phases + stabilization assay)."""

import json

import numpy as np
import pytest

from artifacts.atlas import at10_n20_055 as A
from artifacts.atlas import g20_genome as G


def _results():
    return json.loads(A.RESULTS_PATH.read_text(encoding="utf-8"))


def test_committed_results_match_the_runner_spec():
    res = _results()
    assert res["spec"] == json.loads(json.dumps(A.spec()))
    seeds = [res["assay"]["noise_seed"], *(r["noise_seed"] for r in res["replicates"])]
    assert seeds == [A.NOISE["seed"], *A.REPLICATE_NOISE_SEEDS]
    thr = res["null"]["threshold_hz"]
    assert abs(thr - (max(abs(v) for v in res["null"]["evoked_hz"].values()) + 1.0)) < 1e-3  # stored evoked rounded
    for ph in res["phases"].values():
        assert ph["n_significant_vs_null"] == sum(v > thr for v in ph["evoked_hz"].values())
    for a in [res["assay"], *res["replicates"]]:
        verdicts = {t["perturbation"]: t["verdict"] for t in a["declared_tests"]}
        assert all(verdicts[f"kick={k}"] == "STABILIZED" for k in A.KICKS)
        assert all(arm["disabled"]["w_unchanged"] for arm in a["arms"].values())  # W frozen


def test_propagation_reads_a_known_hierarchy_wave():
    """Area k (distance k/19 from H01) answers each pulse 3k ms later and weaker."""
    names = G.area_names()
    area = np.repeat(names, 5)
    n_steps = int(A.PHASE_MS / A.DT_MS)
    sp = np.zeros((n_steps, area.size), dtype=np.float32)
    rng = np.random.default_rng(1)
    sp[rng.random(sp.shape) < 0.002] = 1.0  # background
    for o in A._onsets_ms():
        for k, a in enumerate(names):
            if k == 0 or k > 12:
                continue
            s = int((o + 3.0 * k) / A.DT_MS)  # onsets 3-36 ms, inside EVOKED_MS
            cols = np.nonzero(area == a)[0][: 5 - k // 3]  # fewer responders with distance
            sp[s:s + int(40.0 / A.DT_MS), cols] = 1.0  # 40 ms: ends by 76 ms, before controls
    out = A.propagation(sp, area)
    assert out["spearman_evoked_distance"] < -0.9
    assert out["spearman_latency_distance"] > 0.9
    assert out["n_significant"] == 12
    assert out["latency_ms"]["H02"] <= 5.0 and out["latency_ms"]["H13"] <= 40.0


@pytest.mark.slow
def test_primary_assay_reproduces_committed_verdicts():
    got = A.run_assay()
    want = _results()["assay"]
    assert got["declared_tests"] == want["declared_tests"]
    assert np.allclose(got["reference_window_rates_hz"], want["reference_window_rates_hz"], atol=1e-3)
