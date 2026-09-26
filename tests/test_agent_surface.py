"""0.5.5 item 5a: agent tool surface (realize, simulate, inspect, compare, observe, verify)."""

import dataclasses

import numpy as np
import pytest

import jaxfne as J
from jaxfne import agent as A

AB = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];"


def _spec(weight=0.5, delay=None):
    d = "" if delay is None else f"; delay = {delay}"
    return (
        f"O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = {weight}{d}];\n"
        f"{AB}\nx : A O[k] B : y\n"
    )


def _run(weight=0.5, delay=4.0):
    return A.simulate(A.realize(_spec(weight, delay), seed=11), duration_ms=30.0, dt_ms=1.0)


def test_tfne_run_is_identical_across_classes_and_observable():
    run = _run()
    info = A.inspect(run)
    assert info["n_neurons"] == 8 and info["n_edges"] == 16 and info["tfne"]
    assert info["n_steps"] == 30 and "V_m" in info["recorded"]
    cmp = A.compare(run)
    assert {k: v["verdict"] for k, v in cmp.items()} == {
        "time": "EQUAL", "weight": "EQUAL", "mechanism": "EQUAL", "delay": "EQUAL"}
    assert all(A.verify(run, p)["verdict"] == "PASS" for p in A.PROPERTIES)
    vm = A.observe(run, "vm")
    assert vm.quantity == "V_m" and vm.level == "NATIVE_UNCALIBRATED" and vm.shape == (30, 8)


def test_substituted_model_fails_identity_per_class():
    """A run whose model was built from a different spec must FAIL, class by class."""
    declared = _run(weight=0.5, delay=4.0)
    other = _run(weight=0.25, delay=2.0)
    swapped = dataclasses.replace(other, realization=declared.realization)
    assert A.verify(swapped, "weight_identity")["verdict"] == "FAIL"
    assert A.verify(swapped, "delay_identity")["verdict"] == "FAIL"
    assert A.verify(swapped, "mechanism_identity")["verdict"] == "PASS"  # both AMPA
    short = dataclasses.replace(declared, duration_ms=40.0)
    assert A.verify(short, "time_identity")["verdict"] == "FAIL"


def test_refusals_never_substitute():
    run = _run()
    with pytest.raises(KeyError, match="unknown property"):
        A.verify(run, "stable")
    with pytest.raises(KeyError, match="unknown observable"):
        A.observe(run, "eeg")
    no_field = dataclasses.replace(run, signals=dataclasses.replace(run.signals, field=None))
    assert "lfp_proxy" not in A.inspect(no_field)["recorded"]
    with pytest.raises(ValueError, match="not recorded"):
        A.observe(no_field, "lfp")
    with pytest.raises(TypeError, match="Realization, Configuration or Model"):
        A.simulate(_spec(), duration_ms=5.0, dt_ms=1.0)
    cfg_run = A.simulate(J.construct(J.suite2_single_neuron_config(seed=0)),
                         duration_ms=5.0, dt_ms=0.5)
    assert A.compare(cfg_run)["weight"]["verdict"] == "NOT_APPLICABLE"
    with pytest.raises(ValueError, match="needs a TFNE run"):
        A.verify(cfg_run, "weight_identity")
    lfp = A.observe(cfg_run, "lfp")  # suite2 preset declares proxy probes
    assert lfp.level == "RELATIVE_PROXY" and np.isfinite(lfp.values).all()
