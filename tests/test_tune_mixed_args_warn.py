"""tune() must warn — never silently drop — mixed single/multi arguments."""

import warnings

import pytest

import jaxfne as jtfne
from jaxfne._signals import Objective


def _cfg(n=8):
    return (
        jtfne.configuration()
        .network(
            name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1}
        )
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def _sim():
    return jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=0)


def test_multi_path_warns_on_dropped_single_args():
    model = jtfne.construct(_cfg())
    obj = Objective(name="spectrolaminar_objective")
    opt = jtfne.agsdr(
        parameters={"source_scale": (0.5, 3.0)}, generations=1, population_size=2, seed=1
    )
    with pytest.warns(UserWarning, match="single-parameter arguments ignored"):
        model.tune(objectives=obj, optimizer=opt, simulation=_sim(), steps=50)


def test_single_path_warns_on_dropped_multi_args():
    model = jtfne.construct(_cfg())
    obj = Objective(name="spectrolaminar_objective")
    with pytest.warns(UserWarning, match="multi-parameter arguments ignored"):
        model.tune(
            objective=obj,
            parameter="source_scale",
            bounds=(0.5, 3.0),
            generations=10,
            simulation=_sim(),
        )


def test_no_warning_when_nothing_dropped():
    model = jtfne.construct(_cfg())
    obj = Objective(name="spectrolaminar_objective")
    opt = jtfne.agsdr(
        parameters={"source_scale": (0.5, 3.0)}, generations=1, population_size=2, seed=1
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.tune(objectives=obj, optimizer=opt, simulation=_sim())
    assert [w for w in caught if "arguments ignored" in str(w.message)] == []
