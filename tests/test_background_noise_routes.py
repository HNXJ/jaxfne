"""P-BALANCED background noise on the simulate route: precedence and refusals.

A caller's ``Simulation.background_noise`` or ``poisson_drive`` replaces the
preset's ``Configuration.background_noise``; the two caller declarations
together, continuation, and SpikeTrainInput are refused.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.neuronal_tensor import BackgroundNoise

_BASE = dict(duration_ms=20.0, dt_ms=0.1, seed=0)
_NOISE = BackgroundNoise(rate_hz=200.0, amplitude=4.0, seed=11)
_POIS = {"rate_hz": 200, "amplitude": 4, "target": "all", "seed": 11}


def _cfg():
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


@pytest.fixture(scope="module")
def plain():
    return jtfne.construct(_cfg())


@pytest.fixture(scope="module")
def preset():
    return jtfne.construct(replace(_cfg(), background_noise=_NOISE))


def test_preset_noise_is_realized_and_recorded(plain, preset):
    a = plain.simulate(jtfne.Simulation(**_BASE))
    b = preset.simulate(jtfne.Simulation(**_BASE))
    assert "realized_noise" not in a.metadata
    assert b.metadata["realized_noise"] == {
        "rate_hz": 200.0, "amplitude": 4.0, "seed": 11, "target": "all",
    }
    assert not np.array_equal(np.asarray(a.V_m), np.asarray(b.V_m))


def test_caller_poisson_drive_replaces_preset_noise(preset):
    s = preset.simulate(jtfne.Simulation(**_BASE, poisson_drive=_POIS))
    assert "realized_noise" not in s.metadata
    assert "poisson_drive" in s.metadata


def test_caller_noise_with_poisson_drive_is_refused(plain):
    with pytest.raises(ValueError, match="background_noise cannot be combined"):
        plain.simulate(jtfne.Simulation(**_BASE, background_noise=_NOISE, poisson_drive=_POIS))


def test_continuation_with_noise_is_refused(plain, preset):
    with pytest.raises(ValueError, match="does not support background_noise"):
        plain.simulate(jtfne.Simulation(**_BASE, background_noise=_NOISE), return_state=True)
    with pytest.raises(ValueError, match="does not support background_noise"):
        preset.simulate(jtfne.Simulation(**_BASE), return_state=True)
