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


def test_probability_over_one_raises():
    """rate_hz * dt_ms / 1000 > 1 is refused, not clamped."""
    with pytest.raises(ValueError, match="exceeds 1"):
        _make_poisson_drive(10, N, rate_hz=5000.0, amplitude=4.0, dt_ms=DT, seed=1,
                            target="all")
    with pytest.raises(ValueError, match="exceeds 1"):
        _simulate(_model(), rate_hz=5000.0)


def test_negative_rate_raises():
    """A negative rate_hz is refused, not clamped to 0."""
    with pytest.raises(ValueError, match="rate_hz"):
        _make_poisson_drive(10, N, rate_hz=-1.0, amplitude=4.0, dt_ms=DT, seed=1,
                            target="all")
    with pytest.raises(ValueError, match="rate_hz"):
        _make_poisson_drive(10, N, rate_hz=float("nan"), amplitude=4.0, dt_ms=DT, seed=1,
                            target="all")


def test_probability_exactly_one_is_allowed():
    """rate_hz * dt_ms / 1000 == 1 still draws (boundary, not over)."""
    arr = _make_poisson_drive(10, N, rate_hz=2000.0, amplitude=4.0, dt_ms=DT, seed=1,
                              target="all")
    assert float(arr.sum()) == 10 * N * 4.0


# --- G-POPDRIVE: triple (area, layer, cell_type) target on the LIF route ---

TRIPLE = ("mc", "L23", "E")
TRIPLE_DRIVE = {"rate_hz": 1000.0, "amplitude": 5.0, "counts": 4, "seed": 0}


def _lif_two_pop_model():
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="poisson-triple",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=N,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.5, emitter="lif", drive=0.0),
                nt.NeuronType.make("PV", fraction=0.5, emitter="lif", drive=0.0),
            ])])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, duration_ms=20.0, dt_ms=DT, emitter="lif"))


def _triple_indices(model):
    from jaxfne._signals import resolve_poisson_drive_target

    return resolve_poisson_drive_target(model, TRIPLE)


def test_triple_target_resolves_via_model_select():
    """A triple target resolves through Model.select (area, layer, cell_type)."""
    model = _lif_two_pop_model()
    idx = _triple_indices(model)
    want = [int(i) for i in model.select(area="mc", layer="L23",
                                        cell_type="E").tolist()]
    assert len(idx) > 0 and len(idx) < N, "fixture needs two nonempty pops"
    assert sorted(idx) == sorted(want)


def test_triple_target_masks_drive_array():
    """Masked columns are exactly zero; targeted columns equal the all-draw."""
    from jaxfne._signals import _make_poisson_drive

    model = _lif_two_pop_model()
    idx = _triple_indices(model)
    others = [i for i in range(N) if i not in set(idx)]
    assert others, "fixture needs an untargeted population"
    kw = dict(rate_hz=1000.0, amplitude=5.0, n_steps=40, n_neurons=N,
              dt_ms=DT, seed=0, counts=4)
    full = np.asarray(_make_poisson_drive(**kw, target="all"))
    assert float(full.sum()) > 0, "fixture must draw events"
    masked = np.asarray(_make_poisson_drive(**kw, target=TRIPLE,
                                            target_indices=idx))
    assert bool((masked[:, others] == 0.0).all()), \
        "untouched populations receive zero drive"
    assert np.array_equal(masked[:, idx], full[:, idx]), \
        "per-neuron draw unchanged on the named population"


def test_triple_target_simulates_only_named_population(monkeypatch):
    """End to end on the LIF route: the kernel's drive array carries events
    only on the named population (captured at the simulate call site; the
    fixture network is recurrent, so V_m equality across runs cannot hold)."""
    import jaxfne._model_simulate as ms
    from jaxfne._signals import _make_poisson_drive as _real_drive

    seen: dict = {}

    def _spy(**kw):
        arr = _real_drive(**kw)
        seen["target"] = kw["target"]
        seen["target_indices"] = kw["target_indices"]
        seen["drive"] = np.asarray(arr)
        return arr

    monkeypatch.setattr(ms, "_make_poisson_drive", _spy)
    model = _lif_two_pop_model()
    idx = _triple_indices(model)
    others = [i for i in range(N) if i not in set(idx)]
    base = dict(duration_ms=20.0, dt_ms=DT, seed=0,
                runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    signals = model.simulate(jtfne.Simulation(
        poisson_drive=dict(TRIPLE_DRIVE, target=TRIPLE), **base))
    assert np.asarray(signals.spikes).shape == (40, N), "targeted drive simulates"
    assert tuple(seen["target"]) == TRIPLE
    assert sorted(int(i) for i in seen["target_indices"]) == sorted(idx)
    drive = seen["drive"]
    assert float(drive[:, idx].sum()) > 0, "fixture must drive the target"
    assert bool((drive[:, others] == 0.0).all()), \
        "untouched populations receive zero drive"
    ref = np.asarray(_real_drive(rate_hz=1000.0, amplitude=5.0, n_steps=40,
                                 n_neurons=N, dt_ms=DT, seed=0, counts=4,
                                 target="all"))
    assert np.array_equal(drive[:, idx], ref[:, idx]), \
        "per-neuron draw unchanged on the named population"
    vnone = np.asarray(model.simulate(jtfne.Simulation(**base)).V_m)
    assert not np.array_equal(np.asarray(signals.V_m), vnone), \
        "drive must be consumed"


def test_triple_target_zero_resolution_refused():
    """A triple naming nothing is refused with a message naming the parts."""
    from jaxfne._signals import resolve_poisson_drive_target

    model = _lif_two_pop_model()
    bad = ("mc", "NoSuchLayer", "E")
    with pytest.raises(ValueError, match="resolves to no neurons"):
        resolve_poisson_drive_target(model, bad)
    with pytest.raises(ValueError, match="NoSuchLayer"):
        model.simulate(jtfne.Simulation(
            duration_ms=20.0, dt_ms=DT, seed=0,
            poisson_drive=dict(TRIPLE_DRIVE, target=bad),
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))


def test_malformed_triple_refused():
    """A non-triple sequence target is refused, not run as all-drive."""
    from jaxfne._signals import resolve_poisson_drive_target

    model = _lif_two_pop_model()
    with pytest.raises(ValueError, match="triple"):
        _make_poisson_drive(10, N, rate_hz=200.0, amplitude=4.0, dt_ms=DT,
                            seed=1, target=("mc", "L23"))
    with pytest.raises(ValueError, match="triple"):
        resolve_poisson_drive_target(model, ("mc", "L23", 5))
