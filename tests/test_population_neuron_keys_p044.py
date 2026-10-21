"""P-044: ``Configuration.population()`` refuses neuron-share keys that are not layers.

Before 0.5.7 a ``neurons=`` key that was not a name in ``layers=`` was dropped without a
message: when no key named a layer the counts fell back to layer thickness, and when some
did, the layers left out of ``neurons`` got zero cells. These tests pin the refusal and the
realized per-layer counts through the constructed model's own ``neuron_table()``.
"""

import pandas as pd
import pytest

import jaxfne as jtfne


def _geometry():
    # Layer A is thin (0.2) and B thick (0.8): thickness allocation would give 20/80.
    return (
        jtfne.Configuration()
        .runtime(seed=0, dt_ms=0.1, duration_ms=100.0)
        .geometry(layer_thickness={"A": 0.2, "B": 0.8})
        .cell_types({"E": 0.6, "PV": 0.4})
        .emitter(family="izhikevich")
        .field(kind="laminar_proxy")
        .probe(kind="lfp_proxy")
    )


def _layer_counts(cfg):
    nt = pd.DataFrame(jtfne.construct(cfg).neuron_table())
    return nt.groupby("layer").size().to_dict()


def test_cell_type_keys_outside_layers_are_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _geometry().population(N=100, neurons={"E": 0.8, "I": 0.2}, layers=["A", "B"], name="V1")
    msg = str(exc.value)
    assert "'E'" in msg and "'I'" in msg
    assert "'A'" in msg and "'B'" in msg


def test_unknown_key_beside_a_layer_key_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _geometry().population(N=100, neurons={"A": 9, "I": 1}, layers=["A", "B"], name="V1")
    msg = str(exc.value)
    assert "'I'" in msg
    assert "'A'" in msg and "'B'" in msg


def test_layer_omitted_from_neurons_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        _geometry().population(N=100, neurons={"A": 9}, layers=["A", "B"], name="V1")
    assert "'B'" in str(exc.value)


def test_layer_shares_realize_their_counts():
    counts = _layer_counts(
        _geometry().population(N=100, neurons={"A": 9, "B": 1}, layers=["A", "B"], name="V1")
    )
    assert counts == {"A": 90, "B": 10}


def test_no_population_declared_allocates_by_thickness():
    # Without a population() declaration the column takes the thickness split (20/80).
    counts = _layer_counts(_geometry().column("V1", layers=["A", "B"], n=100))
    assert counts == {"A": 20, "B": 80}
