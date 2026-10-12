"""``poisson_drive["target"]``: only ``"all"`` is implemented, so any other value is refused.

Before this fix ``_make_poisson_drive`` accepted ``target`` and never read it, and the run
metadata recorded the requested target: ``{"target": "E"}`` drove every neuron while the
manifest said E.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._signals import _make_poisson_drive

N = 8
DT = 0.5
POISSON = {"rate_hz": 200.0, "amplitude": 4.0, "seed": 11}


def _model():
    return jtfne.construct(jtfne.suite2_net1_config(seed=11, n=N, duration_ms=20.0, dt_ms=DT))


def _runtime():
    return jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list")


def _simulate(model, **poisson):
    return jtfne.simulate(model, duration_ms=12.0, dt_ms=DT, seed=17, runtime=_runtime(),
                          record_sources=True, record_fields=False,
                          poisson_drive=dict(POISSON, **poisson))


@pytest.mark.parametrize("target", ["E", "PV", "exc", "ALL", ""])
def test_helper_refuses_unimplemented_target(target):
    with pytest.raises(ValueError, match="target"):
        _make_poisson_drive(10, N, rate_hz=200.0, amplitude=4.0, dt_ms=DT, seed=1,
                            target=target)


def test_simulate_refuses_unimplemented_target():
    with pytest.raises(ValueError, match="target"):
        _simulate(_model(), target="E")


def test_all_and_default_target_drive_every_neuron():
    """``"all"`` and an omitted target are the same draw, and that draw reaches every neuron."""
    arr = _make_poisson_drive(400, N, rate_hz=200.0, amplitude=4.0, dt_ms=DT, seed=1,
                              target="all")
    assert float(arr.sum()) > 0, "fixture must draw events"
    assert bool(jnp.all(arr.sum(axis=0) > 0)), "every neuron must receive events"
    default = _make_poisson_drive(400, N, rate_hz=200.0, amplitude=4.0, dt_ms=DT, seed=1)
    assert jnp.array_equal(arr, default)

    model = _model()
    a = _simulate(model, target="all")
    b = _simulate(model)
    assert np.array_equal(np.asarray(a.V_m), np.asarray(b.V_m))
    off = _simulate(model, target="all", amplitude=0.0)
    assert not np.array_equal(np.asarray(a.V_m), np.asarray(off.V_m)), "drive must be consumed"
