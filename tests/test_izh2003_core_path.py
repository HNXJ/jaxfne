"""Izhikevich 2003 Fig. 3 on the library path (packet 9C-E1, pilot 2).

Loads ``artifacts/etudes/papers/izhikevich2003/core/run_izh_core.py`` by path. The
full-size model (1000 cells, 1e6 edges) is built once per module. Realization
assertions read the realized model through its accessors.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = (REPO_ROOT / "artifacts" / "etudes" / "papers" / "izhikevich2003" / "core"
             / "run_izh_core.py")


def _load_core():
    spec = importlib.util.spec_from_file_location("run_izh_core_under_test", CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = _load_core()


@pytest.fixture(scope="module")
def model0():
    return core.build_core_model(0)


@pytest.fixture(scope="module")
def real0(model0):
    return core.realization_checks(model0, 0)


def test_jaxfne_imports_from_this_repo():
    assert Path(core.jtfne.__file__).resolve().is_relative_to(REPO_ROOT.resolve())


def test_cell_params_realized_per_cell(real0):
    assert real0["n_E_cells"] == 800 and real0["n_I_cells"] == 200
    assert real0["labels_E_first_then_I"], "fixture must put the E cells first (kernel order)"
    cell = real0["cell_params_max_abs_diff"]
    for k in ("a", "b", "c", "d", "v0", "u0", "drive"):
        assert cell[k]["realized_vs_kernel"] <= 1e-4, (k, cell[k])
        assert cell[k]["intended_vs_kernel"] <= 1e-4, (k, cell[k])
    assert cell["sign"]["realized_vs_kernel"] == 0.0


def test_noise_coefficient_per_cell(real0):
    e_std, i_std = real0["noise_std_E_I"]
    assert e_std == pytest.approx(5.0 * np.sqrt(2.0), rel=1e-12)
    assert i_std == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-12)
    assert real0["noise_coef_max_abs_diff_vs_kernel"] <= 1e-6


def test_edges_are_all_to_all_with_self_and_signed_weight_blocks(real0):
    assert real0["n_edges"] == 1_000_000
    assert real0["self_edges"] == 1000
    assert real0["rule_compiled_n_edges"] == {"e_all": 800_000, "i_all": 200_000}
    e = real0["weight_blocks_by_pre_type"]["E"]
    i = real0["weight_blocks_by_pre_type"]["I"]
    assert e["n_edges"] == 800_000 and i["n_edges"] == 200_000
    assert e["n_negative"] == 0 and 0.0 <= e["abs_min"] and e["abs_max"] < 1.0
    assert i["n_positive"] == 0 and i["n_negative"] == i["n_edges"], "fixture must hold I edges"
    assert i["abs_max"] < 2.0
    assert abs(e["abs_mean"] - 0.5) < 0.01
    assert abs(i["abs_mean"] - 1.0) < 0.02


def test_receipt_counts_and_no_default_coupling(real0):
    assert real0["connectivity_compilation"] == {
        "connectivity_mode": "explicit",
        "default_edge_count": 0,
        "declared_rule_edge_count": 1_000_000,
        "total_compiled_edge_count": 1_000_000,
    }
    assert real0["emitter_W_has_coupling"] is False


def test_coupling_is_instantaneous(model0, real0):
    # The run couples through the named instantaneous receptor (P-051, ruled 2026-10-28),
    # not a small-tau filter: every rule declares it, every edge carries tau 0, and a
    # single presynaptic spike reaches the postsynaptic current in the next step with
    # weight W and is gone the step after.
    import dataclasses

    import jax
    import jax.numpy as jnp

    from jaxfne.emitters import (
        INSTANTANEOUS_MECHANISM,
        resolve_edge_weight,
        simulate_edge_recurrent_izhikevich,
    )

    assert real0["mechanisms"] == [{"name": INSTANTANEOUS_MECHANISM, "tau_ms": 0.0}]
    assert real0["connection_mechanisms"] == {"e_all": INSTANTANEOUS_MECHANISM,
                                             "i_all": INSTANTANEOUS_MECHANISM}
    assert real0["tau_ms_unique"] == [0.0]

    edges = model0.params["edge_list"]
    params = dataclasses.replace(model0.params["emitter"],
                                 drive=jnp.zeros((core.N,), dtype=jnp.float32))
    n_steps, s = 20, 5
    sched = np.zeros((n_steps, core.N), dtype=np.float32)
    sched[s, 0] = 300.0  # neuron 0 is forced to spike once, at step s
    _, spikes, _, state = simulate_edge_recurrent_izhikevich(
        params, edges, n_steps, core.DT_MS, jax.random.PRNGKey(0),
        drive_schedule=jnp.asarray(sched), noise_scale=0.0, record_edge_current=True)
    spk = np.asarray(spikes)
    assert spk[s, 0] == 1, "fixture: neuron 0 spikes at the forced step"
    cur = np.asarray(state["edge_current_trace"])
    from_0 = np.asarray(edges.pre) == 0
    assert from_0.any(), "fixture must hold edges from neuron 0"
    w = np.asarray(resolve_edge_weight(edges, jnp.float32))
    np.testing.assert_array_equal(cur[s, from_0], 0.0)  # the spike has not arrived yet
    np.testing.assert_array_equal(cur[s + 1, from_0], w[from_0])  # next step, the full weight
    np.testing.assert_array_equal(cur[s + 2:, from_0], 0.0)  # and no carry-over afterwards


def test_seed_changes_spikes(model0):
    rows0, cols0, _, _ = core.run_core_seed(0, 0.2, model=model0)
    rows0b, cols0b, _, _ = core.run_core_seed(0, 0.2, model=model0)
    rows1, cols1, _, _ = core.run_core_seed(1, 0.2)
    assert rows0.size > 0 and rows1.size > 0, "fixture must spike in 0.2 s"
    assert np.array_equal(rows0, rows0b) and np.array_equal(cols0, cols0b), \
        "the same seed must reproduce its spikes"
    assert not (rows0.shape == rows1.shape and np.array_equal(rows0, rows1)
                and np.array_equal(cols0, cols1)), "seed 1 must change the spikes"


def test_scored_out_is_refused_before_compute(tmp_path):
    with pytest.raises(SystemExit, match="refusing"):
        core.check_out(tmp_path / "p3_scored.json")
    with pytest.raises(SystemExit, match="refusing"):
        core.main(["pilot", "--out", str(tmp_path / "SCORED" / "pilot.json")])
    # A literal path: tmp_path carries the test name, which contains "scored".
    core.check_out("pilot/out.json")


def test_score_seed_known_answer_matches_p5_seed1():
    # Kernel route at the P5 settings, seed 1, through score_seed: N1-N4 must equal the signed
    # P5 scores (artifacts/etudes/papers/izhikevich2003/scores/p3_v7_score.json, fig3.seeds "1") exactly; floats to rtol 1e-12, except the
    # peak excess, a least-squares fit whose last digits moved between the P5 SHA and this one
    # on the same spikes (about 5e-8 relative, 2026-10-27): rtol 1e-6.
    oracle = REPO_ROOT / "artifacts" / "etudes" / "papers" / "izhikevich2003" / "scores" / "p3_v7_score.json"
    want = json.loads(oracle.read_text(encoding="utf-8"))["fig3"]["seeds"]["1"]
    rows, cols, n_steps, _ = core.run_kernel_seed(1, core.lane.FIG3_T)
    assert rows.size > 0, "fixture must spike at the P5 settings"
    theta, alpha_band = core.lane.load_calibration()
    got = core.score_seed(rows, cols, n_steps, theta, alpha_band)
    assert set(got) == {"N1", "N2", "N3", "N4"}
    for nid in ("N1", "N2", "N3", "N4"):
        assert got[nid]["verdict"] == want[nid]["verdict"], (nid, got[nid], want[nid])
        assert set(got[nid]["numbers"]) == set(want[nid]["numbers"]), nid
        for key, value in want[nid]["numbers"].items():
            real = got[nid]["numbers"][key]
            if isinstance(value, float):
                rtol = 1e-6 if key == "pe" else 1e-12
                np.testing.assert_allclose(real, value, rtol=rtol, err_msg=f"{nid}.{key}")
            else:
                assert real == value, (nid, key, real, value)
