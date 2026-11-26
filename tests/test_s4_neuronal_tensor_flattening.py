"""S4 (checklist A2): NeuronalTensor flattening, checked against the declaration.

The checks live in scripts/publication_figures/supplement.py and are shared with the S4
writer. Expectations come from the declared layer boxes, the pose translation and the
in-degree rule, never from another compiled array. Known negatives: a cross-layer position
swap and a dropped edge must each be detected. Not claimed: within-layer relabelling
(``S4_RESIDUAL``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import jaxfne as jtfne

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "scripts/publication_figures/supplement.py"
RUNTIME = jtfne.RuntimeConfiguration(seed=0, duration_ms=2.0, dt_ms=0.5)


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("supplement", GENERATOR)
    assert spec is not None and spec.loader is not None, f"cannot load {GENERATOR}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


supplement = _load_generator()


@pytest.fixture(scope="module")
def model():
    return jtfne.construct(supplement.s4_example_tensor(), RUNTIME)


def test_fixture_has_nonempty_expectations(model):
    rows = model.neuron_table()
    assert len(rows) == 12, "two layers of six neurons"
    assert len(supplement.s4_expected_pairs(rows)) == 9, "fixture must hold 3 E and 3 PV in L4"
    assert len(supplement.s4_actual_pairs(model)) > 0, "fixture must compile edges"


def test_positions_lie_in_their_own_layer_box(model):
    own_outside, in_other = supplement.s4_position_check(
        model.neuron_table(), model.params["positions"], float(model.params["column_height_mm"])
    )
    assert own_outside == [], f"neurons outside their own layer's box: {own_outside}"
    assert in_other == [], f"neurons inside another layer's box: {in_other}"


def test_edges_equal_the_declared_in_degree_rule(model):
    rows = model.neuron_table()
    diff = supplement.s4_pair_difference(
        supplement.s4_actual_pairs(model), supplement.s4_expected_pairs(rows)
    )
    assert diff == {"missing": [], "extra": []}, f"edge set differs from the declared rule: {diff}"


def test_position_check_detects_a_cross_layer_swap(model):
    rows = model.neuron_table()
    positions = np.asarray(model.params["positions"]).copy()
    l23 = next(i for i, r in enumerate(rows) if r["layer"] == "L23")
    l4 = next(i for i, r in enumerate(rows) if r["layer"] == "L4")
    positions[[l23, l4]] = positions[[l4, l23]]
    own_outside, _ = supplement.s4_position_check(
        rows, positions, float(model.params["column_height_mm"])
    )
    assert own_outside, "a cross-layer position swap must be detected"


def test_edge_check_detects_a_dropped_edge(model):
    actual = supplement.s4_actual_pairs(model)
    expected = supplement.s4_expected_pairs(model.neuron_table())
    assert actual, "precondition: the fixture compiles edges"
    dropped = set(actual)
    dropped.discard(sorted(dropped)[0])
    diff = supplement.s4_pair_difference(dropped, expected)
    assert diff["missing"], "a dropped edge must be detected"


def test_s4_writer_passes_and_identity_holds(tmp_path):
    result = supplement.write_s4(tmp_path)
    assert result["verdict"] == "PASS"
    payload = json.loads((tmp_path / "S4_neuronal_tensor.json").read_text(encoding="utf-8"))
    assert payload["identity"] == {"to_dict_equal": True, "positions_equal": True, "edges_equal": True}
    assert payload["residual"] == supplement.S4_RESIDUAL
    assert payload["schema_version"] == "neuronal_tensor_v1"
