"""Tests for paradigm objects and task-flow schemas."""

import inspect
import json
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne


def test_standard_visual_omission_paradigm_conditions():
    """Test that standard_visual_omission returns a Paradigm with 12 conditions."""
    paradigm = jtfne.standard_visual_omission()

    assert paradigm.name == "standard_visual_omission"
    assert len(paradigm.conditions) == 12

    condition_names = paradigm.condition_names()
    expected_names = [
        "AAAB", "AXAB", "AAXB", "AAAX",
        "BBBA", "BXBA", "BBXA", "BBBX",
        "RRRR", "RXRR", "RRXR", "RRRX",
    ]
    assert condition_names == expected_names

    # Verify each condition can be retrieved by name
    for name in expected_names:
        cond = paradigm.condition(name)
        assert cond is not None
        assert cond.name == name


def test_condition_number_mapping():
    """Test that condition numbers are correctly mapped to conditions."""
    paradigm = jtfne.standard_visual_omission()

    expected_mapping = {
        "AAAB": (1, 2),
        "AXAB": (3,),
        "AAXB": (4,),
        "AAAX": (5,),
        "BBBA": (6, 7),
        "BXBA": (8,),
        "BBXA": (9,),
        "BBBX": (10,),
        "RRRR": tuple(range(11, 27)),
        "RXRR": tuple(range(27, 35)),
        "RRXR": (35, 37, 39, 41),
        "RRRX": (36, 38, 40) + tuple(range(42, 51)),
    }

    for cond_name, expected_numbers in expected_mapping.items():
        cond = paradigm.condition(cond_name)
        assert cond is not None
        assert cond.condition_numbers == expected_numbers


def test_event_code_mapping():
    """Test that event codes are correctly set."""
    paradigm = jtfne.standard_visual_omission()

    expected_codes = {
        "fx": 10,
        "p1": 101,
        "p2": 103,
        "p3": 105,
        "p4": 107,
        "rw": 96,
    }

    assert paradigm.event_codes == expected_codes
    assert paradigm.comparison_code == 101  # P1
    assert paradigm.comparison_label == "p1"


def test_omission_position_detection():
    """Test that omission positions are correctly detected."""
    paradigm = jtfne.standard_visual_omission()

    # Conditions with omissions
    omission_conds = paradigm.omission_conditions()
    assert len(omission_conds) == 9  # AXAB, AAXB, AAAX, BXBA, BBXA, BBBX, RXRR, RRXR, RRRX

    # Check specific omission positions
    axab = paradigm.condition("AXAB")
    assert axab is not None
    assert axab.has_omission()
    assert axab.omission_position == "p2"
    assert axab.omitted_event_label() == "p2"

    aaax = paradigm.condition("AAAX")
    assert aaax is not None
    assert aaax.has_omission()
    assert aaax.omission_position == "p4"
    assert aaax.omitted_event_label() == "p4"

    # Condition without omission
    aaab = paradigm.condition("AAAB")
    assert aaab is not None
    assert not aaab.has_omission()
    assert aaab.omitted_event_label() is None


def test_analysis_windows():
    """Test that analysis windows are correctly set."""
    paradigm = jtfne.standard_visual_omission()

    assert "baseline" in paradigm.analysis_windows
    assert "event" in paradigm.analysis_windows
    assert "post_event" in paradigm.analysis_windows

    assert paradigm.analysis_windows["baseline"] == (-500.0, 0.0)
    assert paradigm.analysis_windows["event"] == (0.0, 500.0)
    assert paradigm.analysis_windows["post_event"] == (500.0, 1000.0)

    assert paradigm.pre_stimulus_buffer_ms == 1000.0


def test_paradigm_json_safety():
    """Test that Paradigm objects can be safely serialized to JSON."""
    paradigm = jtfne.standard_visual_omission()

    # Test to_dict() for Paradigm
    paradigm_dict = paradigm.to_dict()
    assert isinstance(paradigm_dict, dict)
    assert "name" in paradigm_dict
    assert "conditions" in paradigm_dict
    assert "event_codes" in paradigm_dict
    assert "analysis_windows" in paradigm_dict

    # Test to_dict() for each condition
    for cond in paradigm.conditions:
        cond_dict = cond.to_dict()
        assert isinstance(cond_dict, dict)
        assert "name" in cond_dict
        assert "sequence" in cond_dict
        assert "events" in cond_dict
        assert isinstance(cond_dict["events"], list)

    # Test that paradigm can be JSON serialized
    json_str = json.dumps(paradigm_dict, allow_nan=False)
    assert isinstance(json_str, str)

    # Test that JSON can be loaded back
    loaded = json.loads(json_str)
    assert loaded["name"] == "standard_visual_omission"
    assert len(loaded["conditions"]) == 12


def test_no_truth_gate_expansion():
    """Test that paradigm objects do not expand or mutate truth gates."""
    paradigm = jtfne.standard_visual_omission()

    # Paradigm should not add or modify truth gates
    assert paradigm.metadata.get("claim_level") is None or paradigm.metadata.get("claim_level") == "computational_scaffold"

    # Check that no condition tries to add biological/mechanism claims
    for cond in paradigm.conditions:
        cond_meta = cond.metadata
        for key in cond_meta.keys():
            # Ensure no mechanism_proven, active_inference_validated, etc.
            assert "mechanism_proven" not in key.lower()
            assert "active_inference" not in key.lower()
            assert "prediction_error" not in key.lower()

    # Check that event metadata doesn't contain over-claims
    for cond in paradigm.conditions:
        for evt in cond.events:
            evt_meta = evt.metadata
            for key in evt_meta.keys():
                assert "mechanism_proven" not in key.lower()
                assert "active_inference" not in key.lower()


def test_paradigm_event_creation():
    """Test that ParadigmEvent objects are created and serialized correctly."""
    evt = jtfne.ParadigmEvent(
        label="p1",
        onset_ms=100.0,
        duration_ms=50.0,
        code=101,
        stimulus="visual_stimulus_A",
        is_omission=False,
    )

    assert evt.label == "p1"
    assert evt.code == 101
    assert evt.stimulus == "visual_stimulus_A"
    assert not evt.is_omission

    evt_dict = evt.to_dict()
    assert evt_dict["label"] == "p1"
    assert evt_dict["code"] == 101


def test_paradigm_condition_creation():
    """Test that ParadigmCondition objects are created correctly."""
    evt1 = jtfne.ParadigmEvent(label="p1", onset_ms=100.0, code=101)
    evt2 = jtfne.ParadigmEvent(label="p2", onset_ms=200.0, code=103, is_omission=True)

    cond = jtfne.ParadigmCondition(
        name="TEST_AXXX",
        sequence=("A", "X", "B", "C"),
        omission_position="p2",
        condition_numbers=(1, 2, 3),
        events=(evt1, evt2),
    )

    assert cond.name == "TEST_AXXX"
    assert cond.has_omission()
    assert cond.omission_position == "p2"
    assert len(cond.condition_numbers) == 3


def test_paradigm_batch_method():
    """Test that Paradigm.batch() returns proper trial batch metadata."""
    paradigm = jtfne.standard_visual_omission()

    batch = paradigm.batch(n_trials=100, seed=42)
    assert batch["name"] == "standard_visual_omission"
    assert batch["n_trials"] == 100
    assert batch["seed"] == 42

    # Test with condition weights
    weights = {"AAAB": 2.0, "AAAX": 1.0}
    batch_weighted = paradigm.batch(n_trials=100, seed=42, condition_weights=weights)
    assert batch_weighted["condition_weights"] == weights


def test_presentation_summary_unexpected_error_propagates(monkeypatch):
    """Only expected serialization failures fall back to repr; the rest propagate."""
    import jaxfne.paradigm as paradigm_mod

    def _boom(spec):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        paradigm_mod._summarize_presentations.__globals__, "json_safe", _boom
    )
    with pytest.raises(RuntimeError, match="boom"):
        paradigm_mod._summarize_presentations({"tok": object()})


def test_presentation_summary_expected_error_falls_back(monkeypatch):
    """A documented serialization failure still yields the repr fallback."""
    import jaxfne.paradigm as paradigm_mod

    def _fail(spec):
        raise TypeError("not serializable")

    monkeypatch.setitem(
        paradigm_mod._summarize_presentations.__globals__, "json_safe", _fail
    )
    out = paradigm_mod._summarize_presentations({"tok": object()})
    assert out["tok"]["kind"] == "repr"


# --- Crazy Sailor environment (M1). Contract: artifacts/programme/design/plans/crazy_sailor/m0_contract.md

_sp = jtfne.paradigm
cs = types.SimpleNamespace(
    Calm=_sp.SailorCalm,
    WhiteNoise=_sp.SailorWhiteNoise,
    SailorConfig=_sp.SailorConfig,
    State=_sp.SailorState,
    Info=_sp.SailorInfo,
    reset=_sp.sailor_reset,
    step=_sp.sailor_step,
    rollout=_sp.sailor_rollout,
    DTYPE=_sp._SAILOR_DTYPE,
)


def _cfg(n=2, dt=0.05, beta=0.0, dist=None, **kw):
    return cs.SailorConfig(n_dim=n, dt=dt, damping=beta, disturbance=dist or cs.Calm(), **kw)


def test_sailor_zero_force_preserves_rest():
    cfg = _cfg(n=2, dt=0.01, beta=0.0)
    s = cs.reset(cfg, jax.random.PRNGKey(0))
    assert float(jnp.abs(s.p).max()) == 0.0, "fixture must start at rest"
    s2, _ = cs.rollout(cfg, s, jnp.zeros((200, 2)))
    assert np.array_equal(np.asarray(s2.p), np.zeros(2)), "position must stay at rest"
    assert np.array_equal(np.asarray(s2.v), np.zeros(2)), "velocity must stay at rest"


def test_sailor_constant_force_matches_analytic_undamped():
    cfg = _cfg(n=2, dt=0.05, beta=0.0)
    p0 = np.array([1.0, -1.0], np.float32)
    v0 = np.array([0.5, 0.25], np.float32)
    a = np.array([0.3, -0.2], np.float32)
    T = 40
    assert np.linalg.norm(a) > 0 and T > 0, "fixture must apply a nonzero force over time"
    s = cs.reset(cfg, jax.random.PRNGKey(0), p0=p0, v0=v0)
    s2, info = cs.rollout(cfg, s, jnp.tile(a, (T, 1)))
    n = np.arange(1, T + 1)[:, None]
    t = n * cfg.dt
    v_ref = v0 + t * a
    p_ref = p0 + t * v0 + 0.5 * t * t * a
    assert np.allclose(np.asarray(info.error), np.linalg.norm(p_ref, axis=1), rtol=1e-5, atol=1e-6), (
        "per-step error must match the analytic trajectory at every step"
    )
    assert np.allclose(np.asarray(s2.v), v_ref[-1], rtol=1e-5, atol=1e-6)
    assert np.allclose(np.asarray(s2.p), p_ref[-1], rtol=1e-5, atol=1e-6)


def test_sailor_damped_recurrence_closed_form():
    cfg = _cfg(n=1, dt=0.01, beta=1.5)
    a = np.array([1.0], np.float32)
    T = 300
    assert float(a[0]) != 0.0 and cfg.damping > 0, "fixture must be damped and forced"
    s = cs.reset(cfg, jax.random.PRNGKey(0), v0=np.array([0.0], np.float32))
    s2, _ = cs.rollout(cfg, s, jnp.tile(a, (T, 1)))
    r = 1.0 - cfg.damping * cfg.dt
    v_ref = (1 - r**T) / (cfg.damping * cfg.dt) * cfg.dt * a[0]
    assert np.allclose(np.asarray(s2.v)[0], v_ref, rtol=1e-4, atol=1e-6)


def test_sailor_fixed_seed_replay_is_bit_identical():
    cfg = _cfg(n=2, dt=0.05, dist=cs.WhiteNoise(0.5))
    acts = jnp.ones((50, 2)) * 0.1
    a1, _ = cs.rollout(cfg, cs.reset(cfg, jax.random.PRNGKey(7)), acts)
    a2, _ = cs.rollout(cfg, cs.reset(cfg, jax.random.PRNGKey(7)), acts)
    assert np.array_equal(np.asarray(a1.p), np.asarray(a2.p))
    assert np.array_equal(np.asarray(a1.v), np.asarray(a2.v))


def test_sailor_distinct_seeds_give_distinct_disturbances():
    cfg = _cfg(n=2, dt=0.05, dist=cs.WhiteNoise(0.5))
    acts = jnp.zeros((50, 2))
    a1, _ = cs.rollout(cfg, cs.reset(cfg, jax.random.PRNGKey(1)), acts)
    a2, _ = cs.rollout(cfg, cs.reset(cfg, jax.random.PRNGKey(2)), acts)
    diff = float(jnp.abs(a1.v - a2.v).max())
    assert diff > 0.0, "fixture must produce a nonzero disturbance"
    assert diff > 1e-3, "different seeds must give distinguishable trajectories"


def test_sailor_zero_intensity_white_noise_equals_calm():
    base = _cfg(n=3, dt=0.02, beta=0.3)
    zero = _cfg(n=3, dt=0.02, beta=0.3, dist=cs.WhiteNoise(0.0))
    acts = jnp.linspace(-1, 1, 60 * 3).reshape(60, 3)
    s0 = cs.reset(base, jax.random.PRNGKey(3), p0=np.ones(3, np.float32))
    a, _ = cs.rollout(base, s0, acts)
    b, _ = cs.rollout(zero, s0, acts)
    assert float(jnp.abs(a.v).max()) > 0, "fixture must move"
    assert np.array_equal(np.asarray(a.p), np.asarray(b.p))
    assert np.array_equal(np.asarray(a.v), np.asarray(b.v))


def test_sailor_white_noise_increment_scaling():
    n_samples = 20000
    sigma = 0.8
    keys = jax.random.split(jax.random.PRNGKey(11), n_samples)
    wn = cs.WhiteNoise(sigma)
    zeros = jnp.zeros((1,), cs.DTYPE)

    def var_at(dt):
        inc = jax.vmap(lambda k: wn.increment(k, 0.0, zeros, dt, 1))(keys)
        return float(jnp.var(inc))

    assert n_samples >= 20000 and sigma > 0, "fixture must have enough samples and noise"
    v1 = var_at(0.01)
    v2 = var_at(0.04)
    assert abs(v1 / (sigma**2 * 0.01) - 1) < 0.03, "variance must equal sigma^2 dt"
    assert abs(v2 / (sigma**2 * 0.04) - 1) < 0.03, "variance must equal sigma^2 dt at a second dt"
    assert abs(v2 / v1 - 4.0) < 0.2, "variance must scale linearly in dt"


def test_sailor_disturbance_independent_of_action():
    cfg = _cfg(n=2, dt=0.05, beta=0.0, dist=cs.WhiteNoise(0.6))
    key = jax.random.PRNGKey(5)
    acts_a = jnp.zeros((8, 2))
    acts_b = jax.random.normal(jax.random.PRNGKey(6), (8, 2))
    assert float(jnp.abs(acts_a - acts_b).max()) > 0, "fixture must use different actions"

    def realized_dv(acts):
        s = cs.reset(cfg, key)
        out = []
        for k in range(acts.shape[0]):
            s_next, _ = cs.step(cfg, s, acts[k])
            out.append(np.asarray(s_next.v - s.v - cfg.dt * (cfg.actuator @ acts[k])))
            s = s_next
        return np.stack(out)

    dv_a = realized_dv(acts_a)
    dv_b = realized_dv(acts_b)
    assert np.abs(dv_a).max() > 0, "fixture must have a nonzero disturbance"
    assert np.allclose(dv_a, dv_b, atol=1e-6), "disturbance must not depend on the action"


@pytest.mark.parametrize("n", [1, 2, 3])
def test_sailor_shapes_and_units_for_dimensions(n):
    cfg = _cfg(n=n, dt=0.02, beta=0.1, dist=cs.WhiteNoise(0.2))
    assert cfg.actuator.shape == (n, n), "default actuator must be N x N identity"
    assert cfg.target.shape == (n,), "default target must have shape (N,)"
    s = cs.reset(cfg, jax.random.PRNGKey(0))
    assert s.p.shape == (n,) and s.v.shape == (n,)
    s2, info = cs.step(cfg, s, jnp.ones((n,)) * 0.5)
    assert s2.p.shape == (n,) and s2.v.shape == (n,)
    assert info.error.shape == () and info.sq_error.shape == () and info.force.shape == (n,)
    assert int(s2.t_step) == 1, "one step call advances the step counter once"


def test_sailor_actuator_rank_deficient_dims_and_error_definition():
    cfg = _cfg(n=3, dt=0.02, n_act=1, actuator=[[1.0], [0.0], [0.0]], target=[1.0, 2.0, 0.0])
    assert cfg.n_act == 1 and cfg.actuator.shape == (3, 1), "D < N must be honoured"
    s = cs.reset(cfg, jax.random.PRNGKey(0), p0=np.array([0.0, 0.0, 0.0], np.float32))
    s2, info = cs.step(cfg, s, jnp.array([0.0]))
    expect = np.linalg.norm(np.asarray(s2.p) - np.array([1.0, 2.0, 0.0]))
    assert np.isclose(float(info.error), expect, rtol=1e-6), "error must be ||p - target||"
    assert np.isclose(float(info.sq_error), float(info.error) ** 2, rtol=1e-6)


def test_sailor_jit_matches_eager():
    cfg = _cfg(n=2, dt=0.05, beta=0.2, dist=cs.WhiteNoise(0.4))
    jstep = jax.jit(lambda s, a: cs.step(cfg, s, a))
    acts = jnp.linspace(-1, 1, 20 * 2).reshape(20, 2)
    se = cs.reset(cfg, jax.random.PRNGKey(9))
    sj = se
    for k in range(20):
        se, ie = cs.step(cfg, se, acts[k])
        sj, ij = jstep(sj, acts[k])
    assert float(jnp.abs(se.v).max()) > 0, "fixture must move"
    assert np.allclose(np.asarray(se.p), np.asarray(sj.p), rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(se.v), np.asarray(sj.v), rtol=1e-6, atol=1e-6)


def test_sailor_config_validation_refuses_bad_values():
    with pytest.raises(ValueError):
        cs.SailorConfig(n_dim=2, dt=0.0)
    with pytest.raises(ValueError):
        cs.SailorConfig(n_dim=2, dt=0.1, damping=-1.0)
    with pytest.raises(ValueError):
        cs.SailorConfig(n_dim=2, dt=0.1, actuator=np.ones((3, 2)))
    with pytest.raises(ValueError):
        cs.SailorConfig(n_dim=2, dt=0.1, target=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        cs.WhiteNoise(-0.1)
    cfg = _cfg(n=2)
    with pytest.raises(ValueError):
        cs.step(cfg, cs.reset(cfg, jax.random.PRNGKey(0)), jnp.zeros((3,)))


def test_sailor_step_has_no_controller_or_observation_input():
    params = list(inspect.signature(_sp.sailor_step).parameters)
    assert params == ["cfg", "state", "action"], "step must take no controller or observation"
    assert cs.Info._fields == ("error", "sq_error", "force"), "Info must carry environment-owned fields only"
    assert cs.State._fields == ("p", "v", "key", "t_step"), "State must carry environment-owned fields only"


def test_sailor_nonidentity_actuator_with_fewer_actuators_than_dims():
    B = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, -2.0]], np.float32)  # N=3, D=2
    cfg = _cfg(n=3, dt=0.02, n_act=2, actuator=B)
    a = np.array([0.7, -0.4], np.float32)
    assert np.linalg.norm(B @ a) > 0, "fixture must produce a nonzero force"
    s = cs.reset(cfg, jax.random.PRNGKey(0))
    s2, info = cs.step(cfg, s, a)
    assert np.allclose(np.asarray(info.force), B @ a, rtol=1e-6, atol=1e-7), "force must be B a"
    assert np.allclose(np.asarray(s2.v), cfg.dt * (B @ a), rtol=1e-6, atol=1e-7), "v must integrate B a"


def test_sailor_noise_differs_between_steps():
    cfg = _cfg(n=2, dt=0.05, dist=cs.WhiteNoise(0.6))
    s = cs.reset(cfg, jax.random.PRNGKey(4))
    incs = []
    for _ in range(6):
        s_next, _ = cs.step(cfg, s, jnp.zeros((2,)))
        incs.append(np.asarray(s_next.v - s.v))
        s = s_next
    incs = np.stack(incs)
    assert np.abs(incs).max() > 0, "fixture must have nonzero noise"
    assert np.abs(incs[1:] - incs[:-1]).max() > 1e-4, "noise must differ from step to step"


def test_sailor_info_force_includes_realized_disturbance():
    cfg = _cfg(n=2, dt=0.05, beta=0.0, dist=cs.WhiteNoise(0.6))
    B = np.asarray(cfg.actuator)
    a = jnp.array([0.2, -0.3], jnp.float32)
    s = cs.reset(cfg, jax.random.PRNGKey(8))
    s2, info = cs.step(cfg, s, a)
    dv_w = np.asarray(s2.v) - np.asarray(s.v) - cfg.dt * (B @ np.asarray(a))
    assert np.abs(dv_w).max() > 0, "fixture must have a nonzero realized disturbance"
    assert np.allclose(np.asarray(info.force), B @ np.asarray(a) + dv_w / cfg.dt, rtol=1e-5, atol=1e-6), (
        "Info.force must equal B a + realized dv^w / dt"
    )


def test_sailor_damped_position_matches_discrete_reference():
    cfg = _cfg(n=2, dt=0.02, beta=0.8)
    p0 = np.array([0.5, -0.5], np.float32)
    v0 = np.array([0.1, 0.2], np.float32)
    a = np.array([0.4, 0.1], np.float32)
    T = 100
    assert cfg.damping > 0 and np.linalg.norm(a) > 0, "fixture must be damped and forced"
    s = cs.reset(cfg, jax.random.PRNGKey(0), p0=p0, v0=v0)
    s2, _ = cs.rollout(cfg, s, jnp.tile(a, (T, 1)))
    # Reference: the contract's discrete recurrence, written independently in float64 numpy.
    p, v = p0.astype(np.float64), v0.astype(np.float64)
    dt, beta = cfg.dt, cfg.damping
    for _ in range(T):
        v_new = v + dt * (-beta * v + a)
        p = p + 0.5 * dt * (v + v_new)
        v = v_new
    assert np.allclose(np.asarray(s2.p), p, rtol=1e-4, atol=1e-5), "damped p must match the recurrence"
    assert np.allclose(np.asarray(s2.v), v, rtol=1e-4, atol=1e-5), "damped v must match the recurrence"
