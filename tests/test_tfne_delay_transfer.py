"""TFNE-PARAM-02: declared delay reaches execution (0.5.2 decision 0b).

Delay is declared in ms on the connection rule and realized as
``delay_steps = round(delay_ms / dt_ms)`` at the construction timestep;
configured ms and realized steps are both recorded. Absent delay is the
pre-0.5.2 execution bit-identically. A positive delay rounding to 0 steps
is refused rather than dropped.

Coverage follows the PARAM-01 equivalence-by-semantic-class pattern: the
delay class keeps its own independent checks (limit identity, arrival
timing, continuation with spikes in flight, composition hook), and
perturbations are asymmetric and non-neutral (4.0 ms vs 0.0 vs absent).
"""

import numpy as np
import pytest

import jaxfne
from jaxfne.tfne import TFNEError, parse, realize, resolve, to_configuration

DT_MS = 1.0
DURATION_MS = 60.0
DELAY_MS = 4.0
DELAY_STEPS = 4

AB = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4];"


def _spec(delay=None, weight=0.5):
    d = "" if delay is None else f"; delay = {delay}"
    return (
        f"O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = {weight}{d}];\n"
        f"{AB}\nx : A O[k] B : y\n"
    )


def _realize(delay=None, weight=0.5, seed=11):
    program = parse(_spec(delay, weight))
    return realize(resolve(program), program, seed=seed)


def _construct(r, dt_ms=DT_MS, duration_ms=DURATION_MS):
    return jaxfne.construct(to_configuration(r, duration_ms=duration_ms, dt_ms=dt_ms))


def _run(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0):
    signals = jaxfne.simulate(model, duration_ms=duration_ms, dt_ms=dt_ms, seed=seed)
    return np.asarray(signals.V_m), np.asarray(signals.spikes)


def _executed_delay_steps(model):
    from jaxfne.emitters import resolve_edge_delay_steps

    return np.asarray(resolve_edge_delay_steps(model.params["edge_list"]))


# --------------------------------------------------------------------------- #
# 0b conversion unit contract
# --------------------------------------------------------------------------- #


def test_delay_steps_from_ms_exact_and_rounding():
    from jaxfne.connectivity import delay_steps_from_ms

    assert delay_steps_from_ms(4.0, 1.0) == 4
    assert delay_steps_from_ms(0.0, 0.1) == 0
    assert delay_steps_from_ms(0.32, 0.1) == 3
    assert delay_steps_from_ms(2.0, 0.1) == 20


def test_delay_steps_from_ms_refusals():
    from jaxfne.connectivity import delay_steps_from_ms

    with pytest.raises(ValueError, match="rounds to 0 steps"):
        delay_steps_from_ms(0.04, 0.1)
    with pytest.raises(ValueError, match=">= 0"):
        delay_steps_from_ms(-1.0, 0.1)
    with pytest.raises(ValueError, match=">= 0"):
        delay_steps_from_ms(float("inf"), 0.1)
    with pytest.raises(ValueError, match="dt_ms"):
        delay_steps_from_ms(1.0, 0.0)


def test_positive_delay_rounding_to_zero_is_refused():
    """0b fail-closed: 0.04 ms at dt 0.1 rounds to 0 steps."""
    r = _realize(delay=0.04)
    with pytest.raises(TFNEError, match="E_DELAY_ROUNDS_TO_ZERO"):
        to_configuration(r, duration_ms=DURATION_MS, dt_ms=0.1)


# --------------------------------------------------------------------------- #
# Zero-delay limit: bit-identical to pre-0.5.2 execution
# --------------------------------------------------------------------------- #


def test_zero_delay_limit_bit_identical():
    """Absent delay and `delay = 0.0` execute bit-identically.

    The kernel already consumes `EdgeList.delay_steps`; this proves the new
    compiler chain adds nothing in the limit: identical trajectories,
    identical edges (weights and all-zero steps), identical storage — while
    the manifest still records what was configured (absent stays absent,
    declared-zero records 0 ms -> 0 steps).
    """
    r_bare = _realize()
    r_zero = _realize(delay=0.0)
    m_bare = _construct(r_bare)
    m_zero = _construct(r_zero)

    v_bare, s_bare = _run(m_bare)
    v_zero, s_zero = _run(m_zero)
    assert np.array_equal(v_bare, v_zero)
    assert np.array_equal(s_bare, s_zero)

    e_bare, e_zero = m_bare.params["edge_list"], m_zero.params["edge_list"]
    assert np.array_equal(np.asarray(e_bare.weight), np.asarray(e_zero.weight))
    assert bool((_executed_delay_steps(m_bare) == 0).all())
    assert np.array_equal(_executed_delay_steps(m_bare), _executed_delay_steps(m_zero))
    assert e_bare.delay_storage == e_zero.delay_storage

    assert np.allclose(np.asarray(r_bare.s["edge_delay_ms"]), 0.0)
    assert np.allclose(np.asarray(r_zero.s["edge_delay_ms"]), 0.0)

    man_bare = m_bare.manifest()
    man_zero = m_zero.manifest()
    assert "tfne_delay" not in man_bare
    assert "executed_delay" not in man_bare
    assert man_zero["tfne_delay"]["declared_ms"] == {
        next(iter(man_zero["tfne_delay"]["declared_ms"])): 0.0
    }
    assert set(man_zero["tfne_delay"]["realized_steps"].values()) == {0}


# --------------------------------------------------------------------------- #
# Arrival timing: the declared delay defers coupling by exactly its steps
# --------------------------------------------------------------------------- #


def test_declared_delay_shifts_arrival_timing():
    """Postsynaptic arrival shifts by exactly the realized steps.

    Three runs share every seed: undelayed, delayed (4 ms -> 4 steps), and
    an uncoupled (weight 0) baseline. Presynaptic spikes are identical in
    all three (one-directional coupling), so the first postsynaptic
    deviation from baseline marks arrival: instantaneous at t*+1, delayed
    at t*+1+4.
    """
    r0, rK, rB = _realize(), _realize(delay=DELAY_MS), _realize(weight=0.0)
    m0, mK, mB = _construct(r0), _construct(rK), _construct(rB)

    assert np.allclose(np.asarray(rK.s["edge_delay_ms"]), DELAY_MS)
    assert bool((_executed_delay_steps(mK) == DELAY_STEPS).all())

    v0, s0 = _run(m0)
    vK, sK = _run(mK)
    vB, _ = _run(mB)
    assert np.array_equal(s0[:, :4], sK[:, :4])
    assert np.array_equal(s0[:, :4], np.asarray(_run(mB)[1])[:, :4])

    t_star = int(np.argwhere(s0[:, :4] > 0.5)[:, 0].min())

    def first_post_diff(vx, vy):
        per_step = np.abs(vx[:, 4:] - vy[:, 4:]).max(axis=1)
        hits = np.argwhere(per_step > 0)
        assert hits.size, "delayed input never reached the postsynaptic population"
        return int(hits[0, 0])

    d0 = first_post_diff(v0, vB)
    dK = first_post_diff(vK, vB)
    assert d0 == t_star + 1
    assert dK == t_star + 1 + DELAY_STEPS
    assert dK - d0 == DELAY_STEPS

    man = mK.manifest()
    assert man["tfne_delay"]["dt_ms"] == DT_MS
    assert set(man["tfne_delay"]["declared_ms"].values()) == {DELAY_MS}
    assert set(man["tfne_delay"]["realized_steps"].values()) == {DELAY_STEPS}
    assert man["executed_delay"]["max_steps"] == DELAY_STEPS
    assert man["executed_delay"]["n_delayed_edges"] == rK.s["n_edges"]


# --------------------------------------------------------------------------- #
# Continuation: spikes in flight survive a chunk boundary bit-exactly
# --------------------------------------------------------------------------- #


def _segmented(model, *, n_steps, schedule=None, continuation=None, return_state=False):
    sim = jaxfne.Simulation(
        duration_ms=float(n_steps),
        dt_ms=DT_MS,
        seed=0,
        runtime=jaxfne.RuntimeConfig(
            dtype="float32",
            recurrent_backend="edge_list",
            enable_hdp=False,
            hdp_params={"noise_scale": 0.0},
        ),
        record_sources=True,
        record_fields=False,
    )
    return model.simulate(
        sim, paradigm=schedule, continuation=continuation, return_state=return_state
    )


def test_delayed_continuation_across_chunk_boundary_with_spikes_in_flight():
    """Split mid-propagation: delayed arrivals pending at the boundary land.

    Delay is 6 steps; the split at step 8 leaves presynaptic spikes whose
    arrivals (spike + 1 + 6) fall in the second segment. Segmented ==
    continuous bit-exactly (V_m, spikes, delay_state), so in-flight spikes
    are carried, not dropped.
    """
    import jax.numpy as jnp

    delay, t1, t2 = 6.0, 8, 22
    model = _construct(_realize(delay=delay))
    full, st_full = _segmented(model, n_steps=t1 + t2, return_state=True)
    first, st1 = _segmented(model, n_steps=t1, return_state=True)
    assert st1.delay_state is not None
    second, st2 = _segmented(model, n_steps=t2, continuation=st1, return_state=True)

    assert float(jnp.max(jnp.abs(full.V_m - jnp.concatenate([first.V_m, second.V_m])))) == 0.0
    assert (
        float(jnp.max(jnp.abs(full.spikes - jnp.concatenate([first.spikes, second.spikes])))) == 0.0
    )
    assert float(jnp.max(jnp.abs(st_full.delay_state - st2.delay_state))) == 0.0

    # Spikes genuinely in flight: fired before the split, arriving after it.
    pre = np.asarray(first.spikes)[:, :4]
    fired = np.argwhere(pre > 0.5)[:, 0]
    in_flight = [int(s) for s in fired if s < t1 <= s + 1 + delay]
    assert in_flight, "split carried no in-flight spikes; the test is vacuous"


# --------------------------------------------------------------------------- #
# Composition hook for 0.5.4: declared + importable only
# --------------------------------------------------------------------------- #


def test_composition_hook_declared_and_importable():
    """`compose_delay_metadata` exists, merges purely, and runs nothing.

    The 0.5.4 composition operator will call it to join per-area delay
    records. Here it is only imported and unit-checked; execution never
    invokes it — a two-rule model keeps one unmerged per-rule record.
    """
    from jaxfne.tfne import compose_delay_metadata

    a = {"declared_ms": {"r0": 2.0}, "realized_steps": {"r0": 2}, "dt_ms": 1.0}
    b = {"declared_ms": {"r1": 5.0}, "realized_steps": {"r1": 5}, "dt_ms": 1.0}
    merged = compose_delay_metadata([a, b])
    assert merged == {
        "declared_ms": {"r0": 2.0, "r1": 5.0},
        "realized_steps": {"r0": 2, "r1": 5},
        "dt_ms": 1.0,
    }
    assert compose_delay_metadata([a, b]) == merged  # pure + deterministic
    assert compose_delay_metadata([]) == {"declared_ms": {}, "realized_steps": {}}
    with pytest.raises(TFNEError, match="E_DELAY_COMPOSITION_AMBIGUOUS"):
        compose_delay_metadata(
            [a, {"declared_ms": {"r0": 9.0}, "realized_steps": {"r0": 9}, "dt_ms": 1.0}]
        )
    with pytest.raises(TFNEError, match="E_DELAY_COMPOSITION_AMBIGUOUS"):
        compose_delay_metadata([a, {"declared_ms": {}, "realized_steps": {}, "dt_ms": 0.5}])

    # Not wired into 0.5.2 execution: two delayed rules coexist unmerged.
    abc = "A := [C = {E}; N = 4]; B := [C = {E}; N = 4]; C := [C = {E}; N = 4];"
    spec = (
        "O[k] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5; delay = 2.0];\n"
        "O[j] := [direction = >; mechanism = AMPA; probability = 1.0; weight = 0.5; delay = 5.0];\n"
        f"{abc}\nx : A O[k] B O[j] C : y\n"
    )
    program = parse(spec)
    r = realize(resolve(program), program, seed=11)
    model = _construct(r)
    block = model.cfg.metadata["tfne_delay"]
    assert set(block["declared_ms"].values()) == {2.0, 5.0}
    assert set(block["realized_steps"].values()) == {2, 5}
    assert set(_executed_delay_steps(model)) == {2, 5}
