# Crazy Sailor — M0 contract (v0.1)

Status: SPECIFIED. Base `origin/dev` 06f9bf2630f297518e905e53cdf01f1ed96471bb. Implementation: `jaxfne/crazy_sailor.py` (M1).
Scope: environment only. No controller, learning, training or dashboard code (M2–M5).

## 1. Objects and dimensions

| Symbol | Shape | Meaning |
|---|---|---|
| N | int ≥ 1 | spatial dimension (test: 1, 2, 3) |
| D | int ≥ 1 | actuator dimension (default D = N) |
| p | (N,) | position, units `length` |
| v | (N,) | velocity, units `length/time` |
| p* | (N,) | target, constant in v0.1 (default 0) |
| a | (D,) | action supplied by the controller, units `force/mass` |
| B | (N, D) | actuator map (default identity) |
| β | scalar ≥ 0 | linear damping, units `1/time` |
| w | (N,) | disturbance acceleration, units `length/time²` |
| Δt | scalar > 0 | physics step, units `time` |

Units are declared, not calibrated. No absolute calibration is implied.

## 2. Dynamics (discrete, per physics step)

Continuous model: ṗ = v, v̇ = −βv + Ba + w(p, t; θ, ξ).

Discrete update applied at step k → k+1 (fixed order, one key split per step):

1. ξ_k, key' = split(key) — disturbance increment drawn from ξ_k.
2. v_{k+1} = v_k + Δt(−β v_k + B a_k) + Δv^w_k
3. p_{k+1} = p_k + Δt·(v_k + v_{k+1}) / 2   (trapezoidal position update)

Properties (by construction):
- With β = 0, a constant, Δv^w = 0: v_n = v_0 + nΔt·Ba and p_n = p_0 + nΔt·v_0 + (nΔt)²·Ba/2 exactly.
- With Δv^w = 0, constant a: v_n = (1−βΔt)^n v_0 + (1−(1−βΔt)^n)/β·Ba for β > 0; the test uses the recurrence closed form.

Error e_k = ‖p_k − p*‖₂. Cost J_T = Σ_k Δt·e_k² / T (time-weighted mean squared error, as in the handoff §2).

Note: `step` passes `t = t_step·Δt` (float32) to the disturbance interface for future time-dependent generators. No M1 generator reads it, and float32 time precision at very large step counts is untested.

## 3. Disturbance interface

A disturbance is a stateless generator with one method:

    increment(key, t, p, dt, n) -> (N,) velocity increment Δv^w

Rules:
- Takes an explicit key; no global RNG.
- Returns a velocity increment already scaled by dt (the caller does not multiply by dt).
- Does not read the action, the controller state or any observation.

v0.1 generators (M1):
- `Calm`: returns zeros.
- `WhiteNoise(sigma)`: Δv^w = σ √Δt ε, ε ~ N(0, I_N). Per the handoff, this is the increment of dv = … dt + σ dW. It is never a fixed-variance acceleration per step.

Later (M4, not in M1): correlated OU force, spatial waves, nonlinear field, mixed. The interface does not change for them.

Separated quantities (for later reporting): amplitude/variance (σ), temporal correlation (none for white noise), spatial correlation (none), determinism/nonlinearity (none), sensitivity to initial conditions (not claimed; no chaos claim is made for noise).

## 4. Configuration

`SailorConfig` (frozen dataclass), owned by the environment:

- `n_dim: int` (N)
- `n_act: int` (D; default N)
- `dt: float` (Δt)
- `damping: float` (β ≥ 0)
- `actuator: array (N, D)` (B; default identity)
- `target: array (N,)` (p*; default zeros)
- `disturbance: Disturbance` (default `Calm()`)

Parameter ownership: only the environment reads these. A parameter that is set but not consumed is a defect. Validation rejects shape mismatch, Δt ≤ 0, β < 0 and σ < 0.

## 5. State, reset and step (pure functions)

    reset(cfg, key, p0=None, v0=None) -> State
    step(cfg, state, action) -> (State', Info)
    rollout(cfg, state, actions) -> (State', Info stacked over T)   # lax.scan over step; actions (T, D)

- `State` is a NamedTuple: `p`, `v`, `key`, `t_step` (int32). Nothing else.
- `Info`: `error` (scalar e_k computed at the new position), `sq_error` = e_k² and the realized `force` = Ba + Δv^w/Δt.
- `reset` default `p0 = 0`, `v0 = 0`. The key is the episode seed; independent episodes use independent keys.
- No controller, learner or observation object is an argument. The environment does not return observations; observation encoding is M2.

## 6. RNG contract

- Episode seed → `key`. Each step consumes exactly one split of the state key.
- Replay: same cfg, same key, same actions → bit-identical trajectory (eager).
- Action independence: the disturbance sequence is a function of the key and step count only. Two action sequences from the same key see identical Δv^w.

## 7. Timing contract

- One `step` call = one physics interval Δt.
- The action at step k is applied during [t_k, t_{k+1}]. The state after the call is at t_{k+1}.
- Control-interval (20 ms) and neural-interval (1 ms) timing belong to M2. M1 has only Δt.

## 8. Acceptance criteria

M0 is accepted when an independent reviewer can identify, from this file alone: every state variable (p, v, key, t_step), every action (a ∈ R^D), every stochastic source (Δv^w via key), the shape of each object, the update order, and the parameter owner of each config field.

M1 is accepted when all tests in `tests/test_crazy_sailor_env.py` pass, the mutation checks in §10 each fail the test named for them, and the gate receipts are in §10.

## 9. M1 test list (mapped to the handoff)

| Test | Checks |
|---|---|
| zero force preserves rest | p, v stay 0 for a = 0, β any, Calm |
| constant force analytic (β=0) | p_n, v_n match closed form to float32 tolerance |
| damped recurrence closed form | v_n matches (1−βΔt)^n recurrence |
| fixed seed replay | identical trajectories for the same key |
| distinct seeds distinguishable | different keys → different Δv^w |
| zero intensity = calm | WhiteNoise(0) bit-equals Calm |
| noise scaling | Var(Δv^w) ≈ σ²Δt; ratio across two Δt ≈ √ratio, using a fixed seed |
| noise independent of action | same key, different actions → same Δv^w sequence |
| shapes/units N=1,2,3 | shapes of p, v, a, B, Info hold |
| JIT vs eager | jitted step matches eager within float32 tolerance |
| config validation | bad shape, Δt ≤ 0, β < 0, σ < 0 raise |
| no controller leakage | step signature takes no controller/observation argument; Info keys are environment-owned only |
| non-identity B, D<N | force = B a and v integrates B a for N=3, D=2, a non-identity B, nonzero action |
| noise differs between steps | successive velocity increments differ (key advances each step) |
| Info.force realized | Info.force = B a + realized Δv^w/Δt |
| damped position reference | damped p and v match an independent float64 numpy recurrence |

Known limitations (documented, not tested in M1): the module is float32-only (`DTYPE` fixed at import; the test conftest resets x64 per test, so x64 is unsupported here). Typed PRNG keys replay identically but have no dedicated test. Long-run float32 drift: observed 0.45 absolute position error over 200000 steps (dt 0.01, |p| about 200, 0.2% relative). The cause is inferred as systematic round-off of nearly constant increments; not verified. Episodes in M1 are far shorter; verify before any long-horizon claim.

Each positive test asserts its precondition (for example the fixture must produce a nonzero disturbance).

## 10. Gate receipts

Receipts are appended below by the M1 run (commands, exit codes, pass counts, SHA). Mutation checks (deliberately broken `crazy_sailor.py`, restored after each run):

| Mutant | Failing tests |
|---|---|
| explicit Euler position `p + dt·v` | `test_constant_force_matches_analytic_undamped` |
| noise `σ·dt·ε` instead of `σ·√dt·ε` | `test_white_noise_increment_scaling` |
| action added to disturbance | `test_constant_force_matches_analytic_undamped`, `test_damped_recurrence_closed_form`, `test_disturbance_independent_of_action` |
| force ignores B (first D rows set to a) | `test_nonidentity_actuator_with_fewer_actuators_than_dims` |
| key not advanced; `Info.force` drops noise | `test_noise_differs_between_steps`, `test_info_force_includes_realized_disturbance` |

## 11. Open issues carried forward

- M2: observation encoder, decoder and E/I 10-neuron example. Not started.
- M3: learning and evaluation. Not started.
- M4: correlated, wave, nonlinear and mixed disturbances. Not started.
- M5: public paradigm workflow, dashboard entry, docs. Not started. No root export is added in M1, so the public surface is unchanged.
