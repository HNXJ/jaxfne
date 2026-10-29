"""Model.tune candidate seeds follow the optimizer's SeedSequence scheme (T56C).

The scalar black-box loop seeds each candidate simulation with
``_candidate_sim_seed(seed, 0, idx)`` -- the same function the AGSDR
optimizer uses -- so scans with adjacent run seeds share no simulation
seed (the old ``seed + idx`` scheme shared every seed but the first).
"""

import numpy as np

import jaxfne as jtfne
from jaxfne._model import Model
from jaxfne.optim.core import _candidate_sim_seed


def _model():
    cfg = (
        jtfne.configuration()
        .network(n=20, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim():
    return jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=0)


def _objective():
    return jtfne.rate_targets(groups={"all": list(range(20))}, targets_hz={"all": 15.0})


def _scan(model, sim, seed, n=3):
    """Run a scalar tune, recording each candidate simulation's seed."""
    recorded = []
    orig_simulate = Model.simulate

    def wrapper(self, simulation, *args, **kwargs):
        recorded.append(int(simulation.seed))
        return orig_simulate(self, simulation, *args, **kwargs)

    Model.simulate = wrapper
    try:
        result = model.tune(
            objectives=_objective(),
            optimizer=jtfne.agsdr(),
            simulation=sim,
            parameter="drive_gain",
            bounds=(0.5, 1.5),
            steps=n,
            seed=seed,
        )
    finally:
        Model.simulate = orig_simulate
    assert len(recorded) == n, f"expected {n} candidate simulations, got {len(recorded)}"
    return result, recorded


def test_fixture_spikes():
    model = _model()
    assert float(np.asarray(model.simulate(_sim()).spikes).sum()) > 0, "fixture must spike"


def test_candidate_seeds_match_optimizer_scheme():
    model = _model()
    _, recorded = _scan(model, _sim(), seed=0)
    assert len(set(recorded)) == len(recorded), "seeds within one scan are distinct"
    expected = [_candidate_sim_seed(0, 0, idx) for idx in range(3)]
    assert recorded == expected


def test_adjacent_scans_share_no_simulation_seed():
    model = _model()
    _, seeds0 = _scan(model, _sim(), seed=0)
    _, seeds1 = _scan(model, _sim(), seed=1)
    assert not set(seeds0) & set(seeds1), "adjacent scans share no simulation seed"
