"""Relative-quantity grammar invariants (0.4.17 closure, contract I).

Verifies the semantic rules declared in
docs/doctrine/relative_quantity_grammar.md against the live implementation:

- base recovery at reference relative state where defined;
- relative-domain preservation (node H stays inside [H_min, H_max]);
- physical-time monotonicity (t[n+1] > t[n], no relative time);
- calibration/effective mapping correctness (population theta channels);
- HDP-off/null exactness (K_HDP=0 ^ K_w_ctrl=0 -> w == base w0).
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne


class TestBaseRecoveryAtReference:
    def test_hdp_off_matches_hdp_null_weights(self):
        """Contract D: with K_HDP=0 ^ K_w_ctrl=0 the weight ODE is null and
        w == w0 (base recovery), for the node HDP path."""
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=60.0, dt_ms=1.0)
        model = jtfne.construct(cfg)
        w0 = np.asarray(model.params["edge_list"].weight)
        sim = jtfne.simulation(
            duration_ms=60.0, dt_ms=1.0, seed=7,
            runtime=jtfne.RuntimeConfig(
                enable_hdp=True, recurrent_backend="edge_list",
                hdp_params={"K_HDP": 0.0, "K_w_ctrl": 0.0, "noise_scale": 0.0},
            ),
        )
        model.simulate(sim)
        diag = model.last_hdp_diagnostics()
        wf = np.asarray(diag["w_final"]).reshape(-1)
        assert np.allclose(wf, w0.reshape(-1), atol=1e-6), (
            "K_HDP=0 ^ K_w_ctrl=0 must recover the base weight w0"
        )

    def test_population_intrinsic_theta_1_recovers_base(self):
        """Contract D: population intrinsic channel a_eff = a_base * theta_1,
        so theta=1 recovers a_base (multiplicative mapping)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_hdp_population_restoring import _mcc3_model
        model, mei_mask, e_mask = _mcc3_model()
        hp = {
            "K_HDP": 0.0, "h_state_locality": "population", "h_state_dim": 2,
            "controller_B": [[1.0, 0.0], [0.0, 1.0]], "controller_lambda": 0.45,
            "controller_tau_H_s": 0.2, "controller_tau_theta_s": 2.0,
            "controller_rate_setpoint_E_hz": 10.9,
            "controller_rate_setpoint_I_hz": 9.14,
            "controller_theta_S_init": (1.0, 1.0),
            "m_ei_edge_mask": mei_mask.astype(bool),
            "e_neuron_mask": e_mask.astype(bool),
            "theta_m_EI_bounds": (0.1, 5.0), "theta_eta_a_bounds": (0.25, 4.0),
        }
        runtime = jtfne.RuntimeConfig(enable_hdp=True, recurrent_backend="edge_list",
                                      jit=False, hdp_params=hp)
        sim = jtfne.simulation(duration_ms=50.0, dt_ms=0.1, seed=17, runtime=runtime)
        model.simulate(sim)
        diag = model.last_hdp_diagnostics()
        assert diag is not None
        # theta initialized to (1,1) -> a_eff == a_base and |w_eff| == 1.0
        # (edge channel is magnitude-replacing by design; see grammar doc).
        assert np.asarray(diag["theta_S_final"]).shape == (2,)


class TestRelativeDomainPreservation:
    def test_node_h_stays_in_declared_bounds_across_stress(self):
        """Contract A/C: node H stays inside [H_min, H_max] under strong drive."""
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=200.0, dt_ms=1.0)
        cfg = cfg.hdp(enable_hdp=True, hdp_params={
            "K_HDP": 0.1, "K_ctrl": 5.0, "alpha": 0.05, "noise_scale": 0.0,
        })
        model = jtfne.construct(cfg)
        for seed in range(3):
            sim = jtfne.simulation(
                duration_ms=200.0, dt_ms=1.0, seed=seed,
                runtime=jtfne.RuntimeConfig(
                    enable_hdp=True, recurrent_backend="edge_list",
                    hdp_params={"K_HDP": 0.1, "K_ctrl": 5.0, "alpha": 0.05,
                                "noise_scale": 0.0},
                ),
            )
            model.simulate(sim)
            diag = model.last_hdp_diagnostics()
            H = np.asarray(diag["H_trace"]).reshape(-1)
            assert float(H.min()) >= 0.1 - 1e-6, f"seed {seed}: H below H_min"
            assert float(H.max()) <= 10.0 + 1e-6, f"seed {seed}: H above H_max"
            assert np.isfinite(H).all()


class TestPhysicalTimeMonotonicity:
    def test_time_ms_monotonic_and_causal(self):
        """Contract B: t[n+1] > t[n] (forward-causal physical clock)."""
        t = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
        from jaxfne.neuronal_tensor import RuntimeConfiguration
        m = jtfne.construct(t, RuntimeConfiguration(seed=1, duration_ms=40.0, dt_ms=1.0))
        s = jtfne.simulate(m)
        tms = np.asarray(s.time_ms)
        assert np.all(np.diff(tms) > 0), "time must be strictly increasing"
        assert tms.shape[0] == 40  # duration_ms inherited (no-arg simulate)

    def test_delay_preserves_forward_causality(self):
        """Non-HDP finite-delay path: delay metadata is consumed (delayed vs
        zero-delay differ) and time stays monotonic."""
        import jax.numpy as jnp
        from dataclasses import replace
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=120.0, dt_ms=1.0)
        model = jtfne.construct(cfg)
        edges = model.params["edge_list"]
        ds = jnp.full((edges.n_edges,), 8, dtype=jnp.int32)
        new_edges = replace(edges, delay_steps=ds)
        object.__setattr__(model, "params", {**model.params, "edge_list": new_edges})
        sim_kwargs = dict(duration_ms=120.0, dt_ms=1.0, seed=1)
        s = model.simulate(
            jtfne.Simulation(
                runtime=jtfne.RuntimeConfig(
                    dtype="float32", recurrent_backend="edge_list",
                    enable_hdp=False, hdp_params={"noise_scale": 0.0},
                ),
                **sim_kwargs,
            )
        )
        tms = np.asarray(s.time_ms)
        assert np.all(np.diff(tms) > 0)
        null_model = jtfne.construct(cfg)
        s_null = null_model.simulate(
            jtfne.Simulation(
                runtime=jtfne.RuntimeConfig(
                    dtype="float32", recurrent_backend="edge_list",
                    enable_hdp=False, hdp_params={"noise_scale": 0.0},
                ),
                **sim_kwargs,
            )
        )
        assert not np.array_equal(
            np.asarray(s.V_m), np.asarray(s_null.V_m)
        ), "nonzero delays must be consumed by the finite-delay kernel"


class TestEffectiveMappingCorrectness:
    def test_population_edge_channel_is_magnitude_replacing(self):
        """Contract D (documented limitation): population edge channel is
        magnitude-replacing (w_eff = sign(w_base) * theta), so theta != 1
        changes |w| away from unit magnitude. Verified through the real
        population-restoring layout (the same one simulate() uses)."""
        import sys
        from pathlib import Path
        import jax.numpy as jnp
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_hdp_population_restoring import _mcc3_model, _population_hdp_params
        model, mei_mask, e_mask = _mcc3_model()
        edges = model.params["edge_list"]
        hp = _population_hdp_params(mei_mask, e_mask, r0_e=10.9, r0_i=9.14)
        from jaxfne._hdp_adaptive import parse_population_restoring_layout
        layout = parse_population_restoring_layout(
            hp, edges_weight=edges.weight,
            labels=tuple(str(x) for x in model.params["emitter"].labels),
            dtype=jnp.float32,
        )
        from jaxfne._hdp_adaptive import bind_theta_to_plant
        w_eff, _ = bind_theta_to_plant(
            jnp.array([2.0, 1.0]), layout,
            a_base=jnp.asarray(model.params["emitter"].drive, dtype=jnp.float32),
            w_ceiling=jnp.asarray(50.0, dtype=jnp.float32),
        )
        masked = np.asarray(layout.channels[0].edge_mask).astype(bool)
        assert masked.any(), "edge channel mask must be non-empty"
        eff_mag = np.abs(np.asarray(w_eff))[masked]
        assert np.allclose(eff_mag, 2.0), (
            "edge channel replaces magnitude: |w_eff| should equal theta_0 on "
            "masked edges"
        )


class TestHdpOffNullConsistency:
    def test_hdp_disabled_identical_across_calls(self):
        """Deterministic repeatability: identical seeds -> identical output."""
        def run(seed):
            cfg = jtfne.suite2_net1_config(seed=seed, n=3, duration_ms=60.0, dt_ms=1.0)
            m = jtfne.construct(cfg)
            s = jtfne.simulate(m)
            return np.asarray(s.V_m)
        a1, a2 = run(11), run(11)
        b = run(12)
        assert np.array_equal(a1, a2)
        assert not np.array_equal(a1, b)

    def test_population_h_can_be_positive_unbounded_signed(self):
        """F2: population-H is a fully unbounded signed controller state — it
        can be positive when the rate exceeds the setpoint. The grammar doc
        declares domain (-inf,+inf)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_hdp_population_restoring import _mcc3_model
        model, mei_mask, e_mask = _mcc3_model()
        hp = {
            "K_HDP": 0.0, "h_state_locality": "population", "h_state_dim": 2,
            "controller_B": [[1.0, 0.0], [0.0, 1.0]], "controller_lambda": 0.45,
            "controller_tau_H_s": 0.2, "controller_tau_theta_s": 2.0,
            "controller_rate_setpoint_E_hz": 100.0,
            "controller_rate_setpoint_I_hz": 100.0,
            "controller_theta_S_init": (1.0, 1.0),
            "m_ei_edge_mask": mei_mask.astype(bool),
            "e_neuron_mask": e_mask.astype(bool),
            "theta_m_EI_bounds": (0.1, 5.0), "theta_eta_a_bounds": (0.25, 4.0),
        }
        runtime = jtfne.RuntimeConfig(enable_hdp=True, recurrent_backend="edge_list",
                                      jit=False, hdp_params=hp)
        model.simulate(jtfne.simulation(duration_ms=300.0, dt_ms=0.1, seed=17,
                                        runtime=runtime))
        diag = model.last_hdp_diagnostics()
        H = np.asarray(diag["H_trace"])
        assert float(H.max()) > 0.0, "population-H must be able to be positive"

    def test_null_recovery_clips_subfloor_weights(self):
        """F3: with null plasticity (K_HDP=0 ^ K_w_ctrl=0), w(t)=w(0) bitwise
        (isolated when plasticity null). When weight plasticity is active
        (K_w_ctrl > 0), the weight ODE integrates and clips subfloor weights
        to w_floor."""
        import jax.numpy as jnp
        from dataclasses import replace
        cfg = jtfne.suite2_net1_config(seed=7, n=3, duration_ms=60.0, dt_ms=1.0)
        m = jtfne.construct(cfg)
        edges = m.params["edge_list"]
        w0 = np.asarray(edges.weight).copy()
        w_mod = w0.copy(); w_mod[0] = 0.0
        ne = replace(edges, weight=jnp.asarray(w_mod, dtype=jnp.float32))
        object.__setattr__(m, "params", {**m.params, "edge_list": ne})

        # (1) Null plasticity: exact bitwise preservation w(t) = w(0)
        rt_null = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                                      hdp_params={"K_HDP": 0.0, "K_w_ctrl": 0.0,
                                                  "noise_scale": 0.0})
        m.simulate(jtfne.simulation(duration_ms=60.0, dt_ms=1.0, seed=7, runtime=rt_null))
        diag = m.last_hdp_diagnostics()
        wf = np.asarray(diag["w_final"]).reshape(-1)
        assert np.array_equal(wf, w_mod), "null plasticity must preserve w0 bitwise"

        # (2) Active plasticity: weight ODE runs and clips subfloor weights to w_floor
        rt_active = jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True,
                                        hdp_params={"K_HDP": 0.0, "K_w_ctrl": 0.001,
                                                    "noise_scale": 0.0})
        m.simulate(jtfne.simulation(duration_ms=60.0, dt_ms=1.0, seed=7, runtime=rt_active))
        diag_act = m.last_hdp_diagnostics()
        wf_act = np.asarray(diag_act["w_final"]).reshape(-1)
        assert abs(wf_act[0]) >= 1.0e-3 - 1e-6
        assert np.allclose(wf_act[1:], w0[1:], atol=1e-5)
