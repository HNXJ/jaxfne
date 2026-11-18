# Crazy Sailor — M2 handoff (controller integration)

Status: SPECIFIED. Implementation NOT started.

## Starting point

- Checkpoint branch: `_temp_crazy_sailor_m0m1_b` on origin, verified at `cbb8a29e695d76ca8c6a7f17ce5f25e20520fc87`; later commits on the same branch add docs only.
- M0 contract: `artifacts/programme/design/plans/crazy_sailor/m0_contract.md`.
- M1 environment: `jaxfne/crazy_sailor.py`; tests `tests/test_crazy_sailor_env.py` (19 tests).
- M1 is NOT integrated into `dev`. Integration is blocked by the repository zero-failure gate: 13 pre-existing failures at base `06f9bf26` (see contract §10).

## Dependency on M1 integration

M2 must start from the M1 branch, not `dev`, because `dev` lacks `jaxfne/crazy_sailor.py`. Decision for Hamm: (a) branch M2 from the checkpoint, or (b) wait for M1 integration. Recommended: (a), with M2 kept on its own `_temp_crazy_sailor_m2_*` branch and merged only after M1 integration.

## Scope (M2 only)

1. Generic controller adapter: `controller.init(key, observation_spec, action_spec)` and `controller.step(state, observation, key) -> (new_state, action)`. The environment never passes privileged state to a non-oracle controller.
2. Tonic distance encoder: `I_i = b_i + g_i · e / (e + d0)`, scalar distance `e = ‖p − p*‖` only. Identical distances give identical input regardless of direction.
3. Four-channel spike-to-force decoder: `a = f0(n1 − n2, n3 − n4)` on the motor neurons (+x, −x, +y, −y). Opposing counts cancel.
4. 10-neuron E/I example: 8 excitatory (4 motor), 2 inhibitory. Use existing JaxFNE operators where supported. Neural integration 1 ms; control interval 20 ms; physics step 20 ms (so the M1 `dt` is set to the control interval in this example); episode 60 s; sensory current constant within a control interval; spike counts from the preceding interval applied in the following interval.

## Out of scope (M2)

Learning, reward-modulated eligibility, training, evaluation baselines (M3); disturbance presets beyond calm/white noise (M4); dashboard, docs, public export (M5). No root export from `jaxfne/__init__.py`.

## Required tests (per handoff §7 M2)

- Identical distances → identical sensory input, independent of direction.
- Opposing motor spike counts cancel in the decoded action.
- No future-state information enters the controller (controller receives only the observation of the current interval).
- Neural and physics timing stay synchronized (1 ms vs 20 ms boundaries).
- Controller parameters change realized behavior (perturb `g_i`, `d0`, `b_i`; require a changed trajectory or an explicit refusal).
- Positive and negative controls, and a boundary case (`e = 0`, `d0` small).

## Open decisions for Hamm (scientific, not for the agent to choose)

- Values of `b_i`, `g_i`, `d0`, and the E/I weight structure.
- Whether the spiking example is a fixed hand-set network (M2) or any learned weights (M3).

## Known limitations carried from M1

- float32-only. x64 unsupported in the environment.
- Typed PRNG keys: replay identical, no dedicated test.
- Long-run float32 drift: 0.45 absolute position error over 200,000 steps, cause inferred (round-off), not verified. M2 episodes (60 s at 20 ms = 3000 steps) are far shorter; verify before any long-horizon claim.

## Acceptance

M2 is accepted when the required tests pass, each has a mutation receipt, the targeted gates pass on the final SHA, and an independent reviewer accepts the diff. The zero-failure repository gate is a separate blocker and is not waived.
