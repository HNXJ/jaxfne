"""Izhikevich 2003 Fig. 3 on the library path (packet 9C-E1, pilot 2).

Loads ``artifacts/etudes/papers/izhikevich2003/core/run_izh_core.py`` by path. The
full-size model (1000 cells, 1e6 edges) is built once per module. Realization
assertions read the realized model through its accessors.
"""

from __future__ import annotations

import importlib.util
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


def test_synaptic_filter_is_the_instantaneous_limit(real0):
    # The edge_list route filters each edge with its mechanism tau; 1e-3 ms makes
    # exp(-dt/tau) exactly 0.0 in float32 at dt 0.5 ms, so each spike acts for one step.
    assert real0["tau_ms_unique"] == pytest.approx([core.TAU_MS], rel=1e-6)
    assert real0["decay_float32_max"] == 0.0
    assert core.synaptic_decay() == 0.0


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
