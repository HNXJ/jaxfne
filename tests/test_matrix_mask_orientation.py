"""Named matrix masks follow W[post, pre]: rows are targets, columns are sources.

``E_to_I`` selects entries from an E source to an I target, ``I_to_E`` from an I source
to an E target, and ``excitatory_to_all`` every entry whose source is excitatory. Before
the fix ``E_to_I`` selected I-to-E entries and ``excitatory_to_all`` selected rows.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._model_tune import _mask_for_parameter


@pytest.fixture(scope="module")
def model():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _e_label(model) -> np.ndarray:
    labels = model.params["emitter"].labels
    return np.asarray([str(x).startswith("E") for x in labels])


def _mask(model, kind: str) -> np.ndarray:
    return np.asarray(_mask_for_parameter(model, "w", kind), dtype=bool)


def test_fixture_signs_follow_the_source_column(model):
    e = _e_label(model)
    W = np.asarray(model.params["emitter"].W, dtype=float)
    assert e.any() and (~e).any(), "fixture must hold E and I cells"
    assert (W[:, e] >= 0).all() and (W[:, e] > 0).any(), "E sources must give nonnegative columns"
    assert (W[:, ~e] <= 0).all() and (W[:, ~e] < 0).any(), "I sources must give nonpositive columns"


@pytest.mark.parametrize("kind, post_e, pre_e", [
    ("E_to_E", True, True),
    ("E_to_I", False, True),
    ("I_to_E", True, False),
])
def test_population_masks_select_target_rows_and_source_columns(model, kind, post_e, pre_e):
    e = _e_label(model)
    rows = e if post_e else ~e
    cols = e if pre_e else ~e
    np.testing.assert_array_equal(_mask(model, kind), np.outer(rows, cols))


def test_excitatory_to_all_selects_excitatory_source_columns(model):
    e = _e_label(model)
    np.testing.assert_array_equal(_mask(model, "excitatory_to_all"), np.outer(np.ones_like(e), e))


def test_excitatory_to_all_never_selects_an_inhibitory_weight(model):
    W = np.asarray(model.params["emitter"].W, dtype=float)
    mask = _mask(model, "excitatory_to_all")
    assert (W[mask] >= 0).all()
    assert (W[~mask & (W != 0)] < 0).all()


def test_unknown_mask_names_the_supported_set(model):
    with pytest.raises(ValueError, match="I_to_E"):
        _mask(model, "I_to_I")
