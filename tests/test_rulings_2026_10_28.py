"""Hamm's rulings of 2026-10-28 on jaxfne core code (packet W1B).

Covers: default seed 0 for two builders, half-up rounding of a seeded drive subset,
protocol E G3 non-owner support flag, a declared cell type realizing at least one cell,
and nested PRNG keys plus SeedSequence-derived simulation seeds in the matrix AGSDR path.
"""

from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.geometry import make_ei_cloud_network
from jaxfne.paradigm import coop_omission_oddball_paradigm


def _event_signature(paradigm):
    return [(e.label, e.onset_ms, e.is_omission) for e in paradigm.conditions[0].events]


def test_ei_cloud_default_seed_is_zero():
    default = make_ei_cloud_network()
    explicit = make_ei_cloud_network(seed=0)
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Precondition: the seed does change the geometry, so the equality above has power.
    other = make_ei_cloud_network(seed=42)
    assert not np.array_equal(np.asarray(default[0]), np.asarray(other[0]))


def test_coop_oddball_default_seed_is_zero():
    default = coop_omission_oddball_paradigm()
    explicit = coop_omission_oddball_paradigm(seed=0)
    assert _event_signature(default) == _event_signature(explicit)
    # Precondition: seed 42 gives a different omission pattern, so the check has power.
    assert _event_signature(default) != _event_signature(coop_omission_oddball_paradigm(seed=42))


def test_subset_drive_rounds_exact_half_up(monkeypatch):
    """frac * n = 2.5 exactly; the even floor is 2, half-up keeps 3."""
    import jaxfne._construct_adex as construct_adex
    from artifacts.etudes.papers.destexhe2009 import destexhe_api as D
    from jaxfne.adex_destexhe import subset_drive_kwargs

    targets = np.arange(10, dtype=np.int64)
    frac = 0.25
    assert frac * targets.size == 2.5  # precondition: an exact half
    monkeypatch.setattr(
        construct_adex,
        "adex_external_drive_kwargs",
        lambda model, sim: {"ext_targets": targets, "ext_lambda": 1.0},
    )
    sim = jtfne.Simulation(duration_ms=D.KICK_MS, dt_ms=D.DT_MS, poisson_drive=D.kick_drive(1))
    sub = subset_drive_kwargs(None, sim, frac)
    assert sub["ext_targets"].size == 3


def test_non_owner_support_flag_follows_non_owner_deviation():
    from jaxfne.protocol_e_integration.e3_execution import verify_non_owner_h_k_support

    owners = [0, 1]
    H = np.ones((5, 4))
    H[:, 0] = 1.2  # owner deviates: allowed
    H[3, 2] = 1.1  # non-owner deviates: must fail the support flag
    g3 = verify_non_owner_h_k_support(H, owners, 4)
    assert g3["passed"] is False
    assert g3["support_only_on_O_H"] is False
    H[3, 2] = 1.0
    g3 = verify_non_owner_h_k_support(H, owners, 4)
    assert g3["passed"] is True
    assert g3["support_only_on_O_H"] is True


def test_declared_type_with_positive_fraction_realizes_one_cell():
    from jaxfne.jdna.genome import AreaGenome, LayerGenome, PseudoGenome, develop

    layer = LayerGenome(
        name="L1", n_neurons=10, depth_band=(0.0, 0.5),
        cell_type_fractions={"E": 0.95, "PV": 0.05},
        fraction_tolerance={"E": (0.0, 1.0), "PV": (0.0, 0.1)},
    )
    genome = PseudoGenome(
        name="tiny", areas=(AreaGenome(name="A", layers=(layer,)),),
        development_parameters={"fraction_jitter_sigma": 0.0},
    )
    with pytest.raises(ValueError, match="PV"):
        develop(genome, seed=0)


def test_declared_zero_fraction_type_may_realize_zero():
    from jaxfne.jdna.genome import AreaGenome, LayerGenome, PseudoGenome, develop

    layer = LayerGenome(
        name="L1", n_neurons=10, depth_band=(0.0, 0.5),
        cell_type_fractions={"E": 1.0, "PV": 0.0},
        fraction_tolerance={"PV": (0.0, 0.1)},
    )
    genome = PseudoGenome(
        name="zero", areas=(AreaGenome(name="A", layers=(layer,)),),
        development_parameters={"fraction_jitter_sigma": 0.0},
    )
    develop(genome, seed=0)


def test_fallback_keys_distinct_over_gen_row_step():
    from jaxfne.optim.core import _fallback_key

    base = jax.random.PRNGKey(7)
    keys = {
        np.asarray(_fallback_key(base, gen, row, step)).tobytes()
        for gen, row, step in itertools.product(range(3), range(12), range(102))
    }
    assert len(keys) == 3 * 12 * 102


def test_candidate_sim_seeds_do_not_shift_with_run_seed():
    from jaxfne.optim.core import _candidate_sim_seed

    population = 12
    generations = 3

    def seeds(run_seed):
        return {
            _candidate_sim_seed(run_seed, gen, row)
            for gen in range(generations)
            for row in range(population)
        }

    a = seeds(100)
    b = seeds(100 + population)
    assert len(a) == generations * population
    assert not (a & b)
    for s in a | b:
        jax.random.PRNGKey(s)  # every derived seed is a usable simulation seed


def test_candidate_sim_seed_accepts_negative_run_seed():
    from jaxfne.optim.core import _candidate_sim_seed

    neg = {_candidate_sim_seed(-5, 0, row) for row in range(4)}
    assert len(neg) == 4
    assert neg.isdisjoint({_candidate_sim_seed(5, 0, row) for row in range(4)})
