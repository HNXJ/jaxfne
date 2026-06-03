"""v0.3.29 SelectorSpec.resolve() and Model.select() tests.

SelectorSpec.resolve() uses strict AND semantics over neuron_table rows, strict
key access (missing required field -> KeyError, never silent), preserves input
order, returns int32 indices, and fails on empty match unless allow_empty=True.
Model.select() is a thin non-mutating wrapper over the model's neuron_table.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import SelectorSpec


def _table():
    """Minimal neuron table with two areas, two layers, mixed cell types."""
    return [
        {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
        {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "PV"},
        {"neuron_id": 2, "area": "V1", "layer": "L5", "cell_type": "E"},
        {"neuron_id": 3, "area": "V4", "layer": "L4", "cell_type": "E"},
    ]


def test_resolve_returns_int32_array():
    idx = SelectorSpec(area="V1").resolve(_table())
    assert idx.dtype == jnp.int32
    assert np.asarray(idx).tolist() == [0, 1, 2]


def test_resolve_and_semantics():
    """All specified fields must match (intersection, not union)."""
    idx = SelectorSpec(area="V1", layer="L4", cell_type="E").resolve(_table())
    assert np.asarray(idx).tolist() == [0]


def test_resolve_preserves_order():
    idx = SelectorSpec(cell_type="E").resolve(_table())
    assert np.asarray(idx).tolist() == [0, 2, 3]


def test_resolve_membership_sequence():
    """A sequence value matches by membership."""
    idx = SelectorSpec(cell_type=["E", "PV"]).resolve(_table())
    assert np.asarray(idx).tolist() == [0, 1, 2, 3]


def test_resolve_by_ids_matches_neuron_id():
    """ids select by neuron_id value, not row position."""
    idx = SelectorSpec(ids=(2, 3)).resolve(_table())
    assert np.asarray(idx).tolist() == [2, 3]


def test_resolve_empty_strict_raises():
    with pytest.raises(ValueError, match="matched no neurons"):
        SelectorSpec(cell_type="NOPE").resolve(_table())


def test_resolve_empty_allowed_returns_empty():
    idx = SelectorSpec(cell_type="NOPE").resolve(_table(), allow_empty=True)
    assert idx.dtype == jnp.int32
    assert int(idx.shape[0]) == 0


def test_resolve_missing_field_raises_keyerror():
    """A requested field absent from a row raises KeyError (never silent)."""
    rows = [{"neuron_id": 0, "area": "V1", "layer": "L4"}]  # no cell_type
    with pytest.raises(KeyError, match="cell_type"):
        SelectorSpec(cell_type="E").resolve(rows)


def test_model_select_wrapper():
    cfg = jtfne.suite2_four_celltype_config(seed=0)
    model = jtfne.construct(cfg)
    idx = model.select(cell_type="E")
    assert idx.dtype == jnp.int32
    table = model.neuron_table()
    for i in np.asarray(idx).tolist():
        assert table[i]["cell_type"] == "E"


def test_model_select_empty_strict_and_allow_empty():
    cfg = jtfne.suite2_four_celltype_config(seed=0)
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError, match="matched no neurons"):
        model.select(cell_type="NOTACELL")
    empty = model.select(cell_type="NOTACELL", allow_empty=True)
    assert int(empty.shape[0]) == 0


def test_model_select_does_not_mutate():
    cfg = jtfne.suite2_four_celltype_config(seed=0)
    model = jtfne.construct(cfg)
    before = model.neuron_table()
    _ = model.select(cell_type="E")
    after = model.neuron_table()
    assert before == after
