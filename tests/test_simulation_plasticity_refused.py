"""T-33: non-zero ``Simulation.plasticity`` is refused on every construction route.

The gain label was never wired into any kernel (it only reached
``metadata["plasticity_gain"]``), so anything but ``0.0`` raises ``ValueError``
at construction, naming the HDP API that engages real plasticity. Zero stays
valid and the metadata key stays.
"""

from __future__ import annotations

import pytest

import jaxfne as jtfne


def _cfg():
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=6, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def test_direct_construction_with_nonzero_plasticity_is_refused():
    with pytest.raises(ValueError, match="enable_plasticity"):
        jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, plasticity=0.7)
    with pytest.raises(ValueError, match="STDPPlasticityConfig"):
        jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, plasticity=0.5)


def test_factory_with_nonzero_plasticity_is_refused():
    with pytest.raises(ValueError, match="Simulation.plasticity"):
        jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0, plasticity=0.7)


def test_with_plasticity_with_nonzero_gain_is_refused():
    sim = jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0)
    with pytest.raises(ValueError, match="Simulation.plasticity"):
        sim.with_plasticity(0.7)
    assert sim.with_plasticity(0.0).plasticity == 0.0


def test_nan_plasticity_is_refused():
    with pytest.raises(ValueError, match="Simulation.plasticity"):
        jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, plasticity=float("nan"))


def test_zero_plasticity_stays_valid_and_metadata_key_stays():
    model = jtfne.construct(_cfg())
    signals = model.simulate(jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0))
    assert signals.metadata["plasticity_gain"] == 0.0
