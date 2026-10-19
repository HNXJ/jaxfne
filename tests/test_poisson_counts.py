"""Poisson drive ``counts``: C independent trains per step per cell.

``counts=1`` (default) is one Bernoulli trial per step per cell, exactly the
historical drive bit for bit; ``counts=C`` sums C independent trains, drawn
as a single ``binomial(C, p)`` per step per cell. Non-positive-int counts
are refused at every entry point.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._signals import _make_poisson_drive
from jaxfne.inputs import (
    check_current_route_poisson_drive,
    compile_poisson_sources,
)


def test_counts_one_matches_bernoulli_history():
    a = _make_poisson_drive(rate_hz=900.0, amplitude=0.1, n_steps=40, n_neurons=25,
                            dt_ms=0.5, seed=0, counts=1)
    vals = np.unique(np.asarray(a, dtype=np.float64))
    assert len(vals) <= 2 and vals[0] == 0.0, \
        "counts=1 must stay a single Bernoulli trial per step per cell"
    assert vals[-1] == pytest.approx(0.1), \
        "a hit must inject exactly one amplitude"
    assert a.shape == (40, 25)


def test_counts_scales_the_mean():
    n_steps, n_cells, seed = 4000, 50, 0
    one = _make_poisson_drive(rate_hz=20.0, amplitude=0.1, n_steps=n_steps,
                              n_neurons=n_cells, dt_ms=0.5, seed=seed, counts=1)
    many = _make_poisson_drive(rate_hz=20.0, amplitude=0.1, n_steps=n_steps,
                               n_neurons=n_cells, dt_ms=0.5, seed=seed, counts=8)
    p = 20.0 * 0.5 / 1000.0
    assert float(one.mean()) == pytest.approx(p * 0.1, rel=0.15), \
        "fixture must realize the Bernoulli mean"
    assert float(many.mean()) == pytest.approx(8 * p * 0.1, rel=0.15), \
        "counts=C must sum C independent trains"
    assert float(many.max()) > 0.1, "counts>1 must admit several events per step"


def test_counts_bounded_by_trains_times_amplitude():
    drive = _make_poisson_drive(rate_hz=1000.0, amplitude=0.5, n_steps=200,
                                n_neurons=20, dt_ms=0.5, seed=1, counts=6)
    assert bool(np.all(drive >= 0.0)), "drive must stay nonnegative"
    assert bool(np.all(drive <= 6 * 0.5)), "drive must cap at counts * amplitude"
    assert set(np.unique(np.round(drive / 0.5))).issubset({0, 1, 2, 3, 4, 5, 6}), \
        "drive must quantize to whole-train multiples of amplitude"


def test_counts_default_is_one():
    defaulted = _make_poisson_drive(rate_hz=900.0, amplitude=0.1, n_steps=40,
                                    n_neurons=25, dt_ms=0.5, seed=0)
    explicit = _make_poisson_drive(rate_hz=900.0, amplitude=0.1, n_steps=40,
                                   n_neurons=25, dt_ms=0.5, seed=0, counts=1)
    assert np.array_equal(np.asarray(defaulted), np.asarray(explicit)), \
        "omitted counts must read exactly as counts=1"


def test_target_refusal_kept():
    with pytest.raises(ValueError, match="only 'all'"):
        _make_poisson_drive(rate_hz=1.0, amplitude=0.1, n_steps=4, n_neurons=2,
                            dt_ms=0.5, seed=0, target="E")


@pytest.mark.parametrize("bad", [True, False, 0, -3, 2.5, "2", (2,), float("nan")])
def test_drive_refuses_bad_counts(bad):
    with pytest.raises(ValueError, match="counts"):
        _make_poisson_drive(rate_hz=1.0, amplitude=0.1, n_steps=4, n_neurons=2,
                            dt_ms=0.5, seed=0, counts=bad)


@pytest.mark.parametrize("bad", [True, False, 0, -3, 2.5, "2", (2,)])
def test_declaration_check_refuses_bad_counts(bad):
    with pytest.raises(ValueError, match="counts"):
        check_current_route_poisson_drive({"rate_hz": 1.0, "counts": bad})


def test_declaration_check_accepts_counts():
    check_current_route_poisson_drive({"rate_hz": 9.0, "counts": 100})


def test_declaration_check_still_refuses_window_with_counts():
    with pytest.raises(ValueError, match="delay_ms"):
        check_current_route_poisson_drive({"rate_hz": 9.0, "counts": 100,
                                           "delay_ms": 1.5})


def test_compile_records_counts_only_when_declared():
    (plain,) = compile_poisson_sources({"rate_hz": 9.0})
    assert dict(plain.schedule) == {"amplitude": 0.5}, \
        "default schedules must stay unchanged"
    (entry,) = compile_poisson_sources({"rate_hz": 9.0, "counts": 100})
    assert dict(entry.schedule)["counts"] == 100, \
        "declared counts must ride the XSource schedule"
    assert entry.value == 9.0


def _lif_tensor():
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="poisson-counts",
        areas=[nt.Area(name="net", layers=[nt.Layer(
            name="L0", n_neurons=4,
            neuron_types=[nt.NeuronType.make("E", fraction=1.0, emitter="lif",
                                             drive=0.0)])])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, duration_ms=20.0, dt_ms=0.5, emitter="lif"))


def test_simulate_accepts_counts_and_records_receipt():
    model = _lif_tensor()
    sim = jtfne.Simulation(
        duration_ms=20.0, dt_ms=0.5, seed=0,
        poisson_drive={"rate_hz": 1000.0, "amplitude": 5.0, "seed": 0, "counts": 4},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    signals = model.simulate(sim)
    assert signals.metadata["poisson_drive"]["counts"] == 4, \
        "stochastic-drive receipt must record counts"
    assert int(np.asarray(signals.spikes).sum()) > 0, "fixture must spike"


def test_simulate_counts_one_matches_default_drive():
    model = _lif_tensor()
    kw = dict(duration_ms=20.0, dt_ms=0.5, seed=0,
              runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    a = np.asarray(model.simulate(jtfne.Simulation(
        poisson_drive={"rate_hz": 1000.0, "amplitude": 5.0, "seed": 0}, **kw)).spikes)
    b = np.asarray(model.simulate(jtfne.Simulation(
        poisson_drive={"rate_hz": 1000.0, "amplitude": 5.0, "seed": 0,
                       "counts": 1}, **kw)).spikes)
    assert np.array_equal(a, b), "counts=1 must equal the historical Bernoulli drive"


@pytest.mark.parametrize("bad", [True, 0, -2, 1.5, "4"])
def test_simulate_refuses_bad_counts(bad):
    model = _lif_tensor()
    with pytest.raises(ValueError, match="counts"):
        model.simulate(jtfne.Simulation(
            duration_ms=20.0, dt_ms=0.5, seed=0,
            poisson_drive={"rate_hz": 10.0, "seed": 0, "counts": bad},
            runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list")))
