# Crazy Sailor — M0 contract (v0.1)

Status: SPECIFIED. Base `origin/dev` 06f9bf2630f297518e905e53cdf01f1ed96471bb. Implementation: `jaxfne/paradigm.py`, Crazy Sailor section, names prefixed `Sailor*` and `sailor_*` (M1). Tests: `tests/test_paradigm.py`, Crazy Sailor section. (Relocated from `jaxfne/crazy_sailor.py` and `tests/test_crazy_sailor_env.py` in the consolidation commit; earlier receipts below name the old files and are kept as history.)
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

M1 is accepted when all Crazy Sailor tests in `tests/test_paradigm.py` pass, the mutation checks in §10 each fail the test named for them, and the gate receipts are in §10.

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

### Gate receipts (M1)

Gated code SHA: `84fb53fd6c94df18598e2b967f4a88d7e48488e7`.

| Check | Command | Result |
|---|---|---|
| Targeted | `pytest tests/test_crazy_sailor_env.py tests/test_generate_errors_index.py tests/test_artifacts_tree_degree.py -q` | 28 passed, rc 0 |
| Tree size | `python3 scripts/check_tree_size.py` | files 2800 of 2800, 0 violations, rc 0 |
| Broad gate | `suite_lock.py -- python3 scripts/run_test_gate.py broad` (clean clone, sandbox off) | 13 failed, 8291 passed, 69 skipped, rc 1 |
| Base comparison | same tests at `06f9bf26` | all 13 fail at base; none caused by M1 |
| x64 | not run as a valid check; module is float32-only, conftest resets x64 per test | documented limitation |
| Typed PRNG | ad hoc script: runs finite, replay bit-identical | observed; no dedicated test |
| Long-run drift | ad hoc script, 200000 steps: abs p error 0.449, rel 0.22% | observed; cause inferred, not verified |

Pre-existing failures at base (not M1): the 13 listed above (Jaxley multicompartment parity, ensemble/continuation, LIF coupling, equivalence gate, score known answers, recomputed-pass tolerance). They are unchanged by this branch and remain under the repository's existing gate policy.

Independent review: reviewer verdict ACCEPT on `b0c3f670` (four corrected findings confirmed by mutants). Minor findings from that review are fixed in `84fb53fd`.

### Checkpoint status (`cbb8a29e`)

- Checkpoint branch `_temp_crazy_sailor_m0m1_b` pushed to origin at `cbb8a29e695d76ca8c6a7f17ce5f25e20520fc87` (verified by `git ls-remote`).
- Docs gate on `cbb8a29e` (clean clone, suite lock): `run_test_gate.py docs` → 293 passed, rc 0.
- Targeted on `cbb8a29e`: 28 passed, rc 0. Tree size: 2800 files, 0 violations.
- Status: M0–M1 implemented and targeted-tested. BLOCKED for `dev` integration by the repository zero-failure gate. No gate exception granted.

### Baseline failure receipts (broad gate, commands and comparison)

Commands (broad, clean clone at `84fb53fd`): `python3 ~/.claude/bin/suite_lock.py --label jaxfne-crazy-sailor -- python3 scripts/run_test_gate.py broad`, sandbox off. Result: 13 failed, 8291 passed, 69 skipped, 2 xfailed, 4 xpassed, rc 1.

Failing at candidate and also failing at base `06f9bf26` (baseline):
1. `test_gaussian_drive_realized_on_receptor_exponential_route`
2. `test_continuation_matches_continuous_and_solo`
3. `test_three_area_no_cross_middle_member_identity`
4. `test_ensemble_slice_matches_solo_eager`
5. `test_ensemble_slice_matches_solo_jit`
6. `test_lif_zero_coupling_equals_no_rule`
7. `test_lif_field_is_generated_by_activity`
8. `test_multicomp_oracle_current_step_matches_jaxley`
9. `test_equivalence_gate_reproducible_7_of_7`
10. `test_score_seed_known_answer_matches_p5_seed1`
11. `test_score_one_seed_known_answer_matches_p5_seed2`
12. `test_recomputed_pass_is_error_within_tolerance`
13. `test_recompute_matches_within_rtol_1e_9`

Base comparison commands (clean clone at `06f9bf26`, cwd in clone): `python3 -m pytest -q -p no:cacheprovider` over the defining files for items 1–11 (`tests/test_izh2003_core_path.py tests/test_neuronal_tensor.py tests/test_composition_aspects_054.py tests/test_composition_identity_054.py tests/test_hh_multicompartment.py tests/test_vis_smoke_all.py tests/test_hdp_rules_ephaptic.py tests/test_construct_weight_range_gaussian_drive.py tests/test_equivalence_gate_v20260815.py tests/test_generate_errors_index.py`): 14 failed, 200 passed; and for items 12–13 (`tests/test_score_port_parity_checks.py tests/test_score_numerics_checks.py tests/test_score_granger_checks.py tests/test_score_field_invariant_checks.py`): 2 failed, 34 passed. All 13 fail at base.

Base-only failures that pass at the candidate (recorded, not investigated): `test_example_08_neuronal_tensor_first_runs`, `test_vis_smoke[exporters.export_figures]`, `test_equivalence_gate_tracked_report_matches_fresh_run`. These passed on the candidate; their base failure is unexplained and nondeterminism is not ruled out.

Net: no failure at the candidate that is absent at base. The earlier two regressions and the folder-limit regression are repaired (see `b0c3f670`).

Repository gate policy is unchanged. These 13 failures remain under their own workstream.

M2 handoff: §12 of this file.

## 12. M2 handoff (controller integration)

Status: IMPLEMENTED on the isolated branch (adapter in `jaxfne/paradigm.py`, 21 M2 test cases from 15 functions in `tests/test_paradigm.py`, §12.9 receipts). NOT integrated into `dev` (§11 gate). The fixture values in §12.9 are engineering fixtures, not the scientific parameters of §12.6. M2 scope decisions of 2026-11-18 are in §12.9 "Decisions".

### 12.9 M2 receipts

- Surface: `sailor_tonic_current`, `sailor_decode`, `sailor_neural_rollout`, `SailorTonicEncoder`. Loop: distance -> tonic drive (20 neural steps of 1 ms, `Model.simulate` with `LIFContinuationState`) -> spikes -> `f0 * (n+x - n-x, n+y - n-y)` on four E motor neurons -> `sailor_step` (physics dt 0.02) -> distance. The controller reads only the scalar distance of the current boundary.
- Fixture (not scientific): E neurons `g = 24`, `-x` motor `g = 0`, `b = 0`, `d0 = 1`, `f0 = 5`, `p0 = (5, 0)`, `p* = 0`, `damping = 0.5`, PV neurons no tonic input.
- Mutants (restored after each run; the M2 tests were run against each):
  - encoder saturation removed (`b + g e`): fails 3 tests (closed form, feedback, d0 perturbation).
  - decoder without opposing cancellation (`n+x + n-x`): fails `test_m2_decoder_opposing_counts_cancel`.
  - distance feedback removed (drive held at `e = 0`): fails 6 tests (no-privileged, causal loop, feedback, g/d0/b perturbations).
- Result: 40 Crazy Sailor tests pass (M1 19 + M2 21 test cases, observed 2026-11-18, targeted run `-k "sailor or m2"`). The M2 distance feedback changes the drive and the membrane trajectory after interval 0 (observed). Motor spike counts were equal in every interval of the feedback probe (f0 = 0 and 5, 10 intervals; counts [7, 2, 7, 7], 53 spikes per interval; observed). The cause is not established (sub-threshold drive is a hypothesis, not verified). Spike-level feedback is therefore not demonstrated by M2 and is deferred (§12.9 Decisions).
- Timing (§12.3.4): the counts of interval k are decoded and applied during interval k+1. `actions[0]` is zero. An earlier revision applied them in the same interval; the independent review (REVISE) caught the deviation and it is fixed.
- Review round 1 (REVISE, no blocker) and fixes: timing corrected; continuation covered by `test_m2_chunked_rollout_equals_one_continuous_simulate`; counts checked against the motor-column spike sums; input refusals tested; reference state uses the same seed family as the rollout.
- Continuation mutant (`continuation=cont` dropped): fails the chunked-equals-continuous test and the d0 perturbation test.
- Totals: 40 Crazy Sailor tests pass (M1 19, M2 21 including parametrized cases).
- Review round 2 (of the revised diff): ACCEPT, with minor notes: (i) the y-channel counts cancel in the rollout fixture, so the y decode path is covered only by the unit test; (ii) the chunked-vs-continuous test does not assert that spikes occur; (iii) the §12.3.1 generic adapter is not implemented.
- Not yet done (tracked in `artifacts/todo_stack.md`, section "Crazy Sailor M2", items CS-1 to CS-6): the two test-hardening items (i) and (ii); the generic `controller.init/step` adapter (iii); the 60 s episode (3000 intervals; each interval is one eager simulate call); the dashboard page; GIFs.

#### Decisions (Hamm, 2026-11-18)

M2 acceptance scope. These define what M2 claims. They do not change the §12.8 criteria.
1. Membrane feedback is sufficient for M2. Verified: distance → tonic drive → membrane trajectory. Spike-count feedback is not an M2 claim.
2. Direction-only control. Verified: spikes → decoded force → boat motion. The fixture force is a fixed-direction push set by the motor gains; M2 makes no goal-directed navigation claim.
3. Fixture values are kept. `b_i`, `g_i`, `d0`, `f0`, the motor gains and the E/I weights are engineering fixtures. M2 makes no scientific parameter claim.
4. M2 stays on `_temp_crazy_sailor_m0m1_b`. No integration into `dev` or `main`; no file-budget change.

Deferred requirements. Not part of M2 acceptance; tracked only in `todo_stack.md`.
- Spike-level feedback: a declared regime with a reproducible nonzero change in motor counts. Next milestone (todo CS-7). The regime is Hamm's scientific choice.
- Goal-directed navigation: a later milestone (todo CS-8).
- Empirical calibration of `b_i`, `g_i`, `d0`, the E/I weights and fixed vs learned weights (§12.6): Hamm supplies the values (todo CS-9).
- Integration into `dev`: blocked by the 13 baseline failures (§10; todo CS-10) and the file budget (allowlist 2798; the ledger reads 2700 files / 95M bytes; this integration adds one file, to 2799, which needs a scoped ruling; todo CS-11). Neither is resolved; the integration path stays closed.

### 12.1 Starting point

- Checkpoint branch: `_temp_crazy_sailor_m0m1_b` on origin, verified at `cbb8a29e695d76ca8c6a7f17ce5f25e20520fc87`; later commits on the same branch add docs and the consolidation only.
- M0 contract: this file, §1–§11.
- M1 environment: `jaxfne/paradigm.py`, Crazy Sailor section; tests `tests/test_paradigm.py`, Crazy Sailor section (19 tests).
- M1 is NOT integrated into `dev`. Integration is blocked by the repository zero-failure gate: 13 pre-existing failures at base `06f9bf26` (§10).

### 12.2 Dependency on M1 integration

M2 must start from the M1 branch, not `dev`, because `dev` lacks the Crazy Sailor section of `jaxfne/paradigm.py`. Decision for Hamm: (a) branch M2 from the checkpoint, or (b) wait for M1 integration. Recommended: (a), with M2 kept on its own `_temp_crazy_sailor_m2_*` branch and merged only after M1 integration. Applied: (a), on `_temp_crazy_sailor_m0m1_b` (§12.9 Decisions, item 4).

### 12.3 Scope (M2 only)

1. Generic controller adapter: `controller.init(key, observation_spec, action_spec)` and `controller.step(state, observation, key) -> (new_state, action)`. The environment never passes privileged state to a non-oracle controller.
2. Tonic distance encoder: `I_i = b_i + g_i · e / (e + d0)`, scalar distance `e = ‖p − p*‖` only. Identical distances give identical input regardless of direction.
3. Four-channel spike-to-force decoder: `a = f0(n1 − n2, n3 − n4)` on the motor neurons (+x, −x, +y, −y). Opposing counts cancel.
4. 10-neuron E/I example: 8 excitatory (4 motor), 2 inhibitory. Use existing JaxFNE operators where supported. Neural integration 1 ms; control interval 20 ms; physics step 20 ms (so the M1 `dt` is set to the control interval in this example); episode 60 s; sensory current constant within a control interval; spike counts from the preceding interval applied in the following interval.

### 12.4 Out of scope (M2)

Learning, reward-modulated eligibility, training, evaluation baselines (M3); disturbance presets beyond calm/white noise (M4); dashboard, docs, public export (M5). No root export from `jaxfne/__init__.py`.

### 12.5 Required tests (per the handoff §7 M2)

- Identical distances → identical sensory input, independent of direction.
- Opposing motor spike counts cancel in the decoded action.
- No future-state information enters the controller (controller receives only the observation of the current interval).
- Neural and physics timing stay synchronized (1 ms vs 20 ms boundaries).
- Controller parameters change realized behavior (perturb `g_i`, `d0`, `b_i`; require a changed trajectory or an explicit refusal).
- Positive and negative controls, and a boundary case (`e = 0`, `d0` small).

### 12.6 Open decisions for Hamm (scientific, not for the agent to choose)

- Values of `b_i`, `g_i`, `d0`, and the E/I weight structure.
- Whether the spiking example is a fixed hand-set network (M2) or any learned weights (M3).

Status (2026-11-18): both questions remain open for the scientific parameters. For M2 scope they are settled by §12.9 Decisions (fixture values kept, deferred to todo CS-9). No value in this section is chosen by the agent.

### 12.7 Known limitations carried from M1

- float32-only. x64 unsupported in the environment.
- Typed PRNG keys: replay identical, no dedicated test.
- Long-run float32 drift: 0.45 absolute position error over 200,000 steps, cause inferred (round-off), not verified. M2 episodes (60 s at 20 ms = 3000 steps) are far shorter; verify before any long-horizon claim.

### 12.8 Acceptance

M2 is accepted when the required tests pass, each has a mutation receipt, the targeted gates pass on the final SHA, and an independent reviewer accepts the diff. The zero-failure repository gate is a separate blocker and is not waived.

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

- M2: observation encoder, decoder and E/I 10-neuron example. Implemented on the isolated branch (§12.9); not integrated into `dev`.
- M3: learning and evaluation. Not started.
- M4: correlated, wave, nonlinear and mixed disturbances. Not started.
- M5: public paradigm workflow, dashboard entry, docs. Not started. No root export is added in M1, so the public surface is unchanged.

## 13. Exploration programme (small-model strategy comparison)

Set 2026-11-18 by Hamm: explore options; no stop for decisions; run parallel lanes. Exploratory, not confirmatory: no claim is made from a run in this section until §13.4 is met. "HTTP / HState / REST" in the request is read as H-state (RBS/RBD/HDP, `artifacts/AGENTS.md`); this is an assumption, flagged for Hamm to correct.

### 13.1 Goal

A small spiking circuit (10 neurons, E/I, §12.3.4) that, from its own spike-decoded action and its own distance input only, learns over a fixed budget to drive the point boat to `p*` from held-out starts. The learning is carried by H-state dynamics (RBD for H, HDP for the plastic weights Theta). Forbidden: an external policy, a privileged state or observation, an oracle controller, or a reward channel that is not a declared H coordinate.

Falsifiable acceptance levels:

- L1 (done, fixture): spike-decoded force moves the boat (§12.9).
- L2: distance changes H; H changes Theta; the weight change is nonzero and reproducible.
- L3: post-training mean final distance over held-out starts is lower than the untrained circuit and lower than the shuffled-feedback control, same seeds, CI excluding 0 over at least 20 seeds.
- L4: the learned behavior holds under white-noise disturbance (M4 level).
- Null: if no strategy beats the controls at L3, the null is reported as the result.

### 13.2 Scenarios

- S1: the 10-neuron fixture (§12.9), calm disturbance.
- S2: scaled circuits, 20/50/100 neurons, same 4:1 E/I ratio.
- S3: direction-agnostic sensing (scalar distance only). Any direction information must come from H.
- S4: white-noise disturbance (M4 level).

### 13.3 Feedback strategies (where distance enters the circuit)

| id | Placement | Rule | Role |
|---|---|---|---|
| F0 | none | fixed weights, no distance drive; the fixture's per-neuron baseline drive (`emitter.drive`) is kept | control; F1 minus F0 isolates the distance term. The network is not silent (observed 2026-11-18: 990 spikes over 40 intervals). Zeroing the baseline would change the S1 fixture, which is not done. |
| F1 | input (tonic drive) | `I = b + g e/(e+d0)`, no H | current M2 |
| F2 | H-state | `dH/dt = −H/τ + κe`; H scales motor gains | RBD only |
| F3 | synapse | three-factor: eligibility × H-derived reward (`−Δe`) on E→motor weights | HDP, existing eligibility rule (`jaxfne/hdp_rule.py`) |
| F4 | intrinsic gain | H scales motor neuron gain | RBD only |
| F5 | synapse | F3 with the distance signal shuffled across time | control |
| F6 | inhibition | H scales the PV→E synaptic weights by (1 + H), H as F2 | alternative placement; the fixture gives PV neurons no tonic input, so the PV→E path is the only place H can act |

### 13.4 Protocol

- Primary metric: mean distance over the final 20% of a 60 s episode (3000 intervals). Secondary: time to reach radius r1 = 1.0 (distance units), declared before the run; "none" if never reached.
- Declared H constants for F2, F2b, F4, F6: H(0) = 0, tau = 2.0 s, kappa = 1.0, H clipped to [0, 4]. Chosen 2026-11-18 for exploration; not tuned.
- Equivalence check: F6 with kappa = 0 must equal F1 bit-for-bit (observed in the Lane C run, seed 1, 40 intervals: all 7 arrays equal).
- At least 20 seeds per strategy at S1. Report effect size, CI and n. Controls F0 and F5 run with the same seeds.
- Freeze the protocol (strategies, metric, seeds, episode length) before the first confirmatory run. Exploratory tuning is allowed, but it is labelled and cannot be reported as confirmatory.

### 13.5 Checklist

- [ ] X1 Confirm which HDP rule names are registered and usable for F3 (`list_registered_hdp_rules`).
- [ ] X2 Harness: one script, strategy × seed, writes JSON to `artifacts/_temp/crazy_sailor_explore/` (gitignored).
- [ ] X3 Controls F0 and F5 at S1.
- [ ] X4 F1, F2, F3, F4, F6 at S1.
- [ ] X5 Scale the best one or two strategies to S2.
- [ ] X6 Results table: effect size, CI, n, test, per level L1–L4.
- [ ] X7 Independent review of the harness and of each claim.
- [ ] X8 Goal-stack proposal for Hamm (goal_stack is owner-owned; no edit without Hamm).

### 13.6 Lanes (parallel, at most three; each writer in its own clone)

- Lane A: F2, F4 (H-state to gain).
- Lane B: F3, F5 (H-gated plasticity).
- Lane C: X2 harness, F0, F1, F6, metrics, CI.

Lane outputs are data until the lead agent re-reads them and re-runs the headline numbers.

### 13.7 Protocol amendment A1 (proposed 2026-11-19; for review; no confirmatory run until the gates pass)

Implementation findings (observed 2026-11-19, 40 intervals, seed 0, unless stated):

- F0 reconciliation. The zero-encoder run (`b = g = 0`, baseline kept) is the §13.3 F0: counts (3,3,3,3), zero decoded force, boat stays at 5.0; K = 600 mean 5.000, 20 seeds identical (Lane C). Lane A's "F0" keeps the distance encoder: its counts (7,2,7,7) equal F1, so its F0 row is F1. Lane A's F2/F2b/F4 values stay valid as H-on-F1, but their "vs F0" differences are "vs F1". Lane A F0 = 527.86 and Lane C F1 = 526.32 at K = 600 differ; cause not found.
- Decoder and plant sign: correct. Counts (+x) give force +x (`sailor_decode`); a constant −x force moves p from (5,0) to (3.59,0) in 40 steps, +x to (6.41,0).
- Fixture defect (not repaired; needs Hamm): tonic gains `g = (24, 0, 24, 24, …)`, so the −x motor neuron has no distance drive. At e = 5 the motor tonic is (20, 0, 20, 20). Net force is +x at every distance (`e/(e+d0)` saturates), so the boat is pushed away from p*.
- Structural consequence: with scalar distance input, a symmetric encoder gives zero net force, and an asymmetric one gives a fixed bias. H as a common gain (F2, F4) cannot change the sign. F6 changes only PV→E weights, which are shared across motor channels. Only F3 (channel-specific eligibility with a Δe reward) could in principle learn a corrective sign. Derived, not yet tested.

Protocol (frozen on acceptance; the start distribution and sample size are fixed here, not chosen from outcomes):

1. Start distribution: `p0 = R (cos θ, sin θ)`, `θ ~ Uniform[0, 2π)`, `R = 5.0` (|p0| of the M2 fixture, set before this amendment), `v0 = 0`. Target p* = 0. The distribution is rotationally symmetric about p*, so it adds no geometry beyond the fixture's distance.
2. Splits: development `N_dev = 20`, `θ` from `numpy.random.default_rng(2025)`; held-out `N = 40`, `θ` from `default_rng(2026)`. Held-out starts are used once, after the gates pass. Disjointness is checked before use.
3. Pairing: every strategy runs every start with the same episode seed (0 at S1, because no stochastic input exists there). A future stochastic element pairs by start.
4. Episode: K = 600 control intervals (12 s), not 3000, because 3000 costs about 18 min per job (0.36 s per interval; 280 jobs for 40 starts × 7 strategies ≈ 84 h serial). The deviation is declared here, before any confirmatory run.
5. Primary metric: `D_j(s)` = mean distance over the final 20% of the episode, per start `j`. Paired difference `Δ_j(s) = D_j(s) − D_j(F0)`. Estimand: mean of `Δ_j(s)` over the 40 held-out starts. The unit is the start, not the RNG seed.
6. Statistic: percentile bootstrap, 10000 resamples over starts, `default_rng(0)`, 95% CI. For F3, also the paired difference against F5 (shuffled) and against F3 untrained (η = 0).
7. Effective-variation gate (before any CI is read): at least 0.9 N distinct `D_j(s)` values (rounded to 1e-6) for each strategy; otherwise that strategy's result is reported as degenerate, with no CI.
8. Stopping rule: none. N = 40 is fixed; no optional stopping.
9. Controls, all on the held-out starts:
   - F0 (zero-encoder): `D_j(F0) = R` for every start (boat stays). Gate: exact.
   - Known-direction probe: a constant force along −p̂ for 40 steps. Gate: error decreases for all 40 starts. Verifies the plant and decoder sign; it is not a strategy.
   - Prediction for F1 (fixture bias): `D_j(F1) > R` for every start with `p_x ≥ 0`, and for the starts with `p_x < 0` the +x push drives the boat through p*. Falsifier: if F1 reduces distance from some starts, the fixture-bias account is wrong.
10. Falsification: a strategy `s` is supported only if the CI of `mean Δ(s)` against F0, and against its own untrained/shuffled control, lies entirely below 0. A CI that includes 0, or lies above 0, is a negative result. Negative results are reported unchanged; the frozen protocol is not retuned.
11. Reporting: the original S1 result is reported as a degenerate-start experiment: F0 = 5.000, F1 = 526.320, F6 = 530.543 (K = 600, Lane C), n_eff = 1, no inferential CI.

Gates: implementation gate open (fixture defect needs a decision: keep the fixture and accept that corrective control is impossible under S1/S3 for F1/F2/F4/F6, or change the fixture gains or sensing, which changes the scenario and needs Hamm). Protocol gate open until the fixture decision is made. Confirmatory runs, §13 conclusions and integration are held.

### 13.7a Revisions after critic review (2026-11-19; §13.7 remains proposed)

Critic verdict: REVISE (worker `critic`; its critique file was blocked by a write hook and is recorded here). Findings and resolution:

1. Lane A vs Lane C "F0 vs F1" gap is a window convention, verified: Lane A's runner averages positions 481..600 (post-step), Lane C averages errors 480..599 (pre-step). The gap is exactly (621.169 − errors_C[480]) / 120 = 1.5438 (verified to 1e-12). Same trajectory, different window. Resolved; no controller difference. Pinned in item 5 below.
2. Item 7 gate: the distinct-value gate must not apply to F0 (identically 5.0 by design, exempt). It applies to each paired contrast Δ_j, not to D_j alone. Before any CI, the L2 gate (weight change nonzero and not saturated at a bound) must pass; a saturated arm makes Δ identically 0.
3. K = 600 changes the estimand (final 2.4 s of 12 s, not 12 s of 60 s). Proposed: confirmatory F0 and F1 use K = 3000; secondary strategies at K = 600 are labelled as such. Compute: about 18 min per job; 40 starts × 3 arms ≈ 120 jobs (≈ 36 h serial, several hours with 8 workers). Needs Hamm's approval of this cost (EX-12).
4. Baseline per strategy: F2, F2b, F4, F6 are compared with F1 (H on F1), F3 with its own η = 0 run, F5 is the shuffle contrast with F3. F0 is a separate stationary reference. Lane A's F2 − F1 = 1104.46 (vs F0 zero-encoder) and 1106.00 (vs lane C F1) are the same effect under two baselines.
5. Window convention (pinned): mean over the final 20% of the pre-step error trace, errors[K − K/5 : K]. For K = 600 this is errors[480:600].
6. Item 9 falsifier (F1 fixture bias): a threshold, not a sign. Count of starts with D_j(F1) ≤ R must be 0 of 40; the CI of mean Δ_j(F1) against F0 must lie above 0. The p_x < 0 starts are predicted to cross p* (verified at K = 60 for 4 of 4 starts in envA).
7. Minor: "net force +x at every distance" is true only where the +x tonic exceeds the baseline, e ≳ 0.14 (critic's inferred estimate, not checked). The decoder sweep check is open.
8. Minor: the shuffle (F5) needs a seed per start and a pairing rule (seed = start index).
9. Minor: R-03, R-04 and R-06 show CIs where n_eff = 1; these must read "no CI". Done in §13.8 below.

Open checks: K = 3000 F1 at θ = 0 (launched detached, EX-16); F1 at θ = π and 3π/4 at K = 600 for the item 6 falsifier.

### 13.8 Experiment registry

Code SHA for every row: 584a56ad (paradigm.py unchanged; only this contract and todo are uncommitted). Outputs: `artifacts/_temp/crazy_sailor_explore/`. "Verified" means the lead re-derived the number from the JSON or re-ran it; "agent" means the worker reported it and the lead did not re-run it.

| id | Config | Seeds / starts | K | Outcome (number) | Status | Notes |
|---|---|---|---|---|---|---|
| R-01 | F0 reconcile (zero-enc vs fixture-enc), library | seed 0, p0 (5,0) | 40 | zero-enc counts (3,3,3,3), zero force, final (5,0); fixture-enc counts (7,2,7,7), final (12.18, 0) | verified | Lane A's "F0" is the fixture encoder = F1 |
| R-02 | Decoder and plant sign probe | p0 (5,0), force ±5 | 40 steps | −x force: p = 3.59; +x: 6.41; zero: 5.00 | verified | Sign correct |
| R-03 | Lane A F0, F2, F2b, F4 (fixture-enc H-gain) | seeds 0–19 | 600 | "F0" (= F1, window 481..600) 527.86; F2 1632.32; F2b 1620.39; F4 1632.32 | verified (recomputed from JSON); no CI (n_eff = 1) | row "F0" is F1; window differs from Lane C by one step (R-06 note); H at clip 4.0 for 95%; F4 ≡ F2 |
| R-04 | Lane B F3 (η 1e-3, 1e-2), F5 shuffled | seeds 0–19 | 600 | F3 526.585 / 527.308 (sd 0); F5 526.615 / 527.293 (sd 0.12, 0.04) | verified; paired CIs are not inferential (F3 has n_eff = 1; all 28 weights at the 0 bound) | paired F3−F5 −0.030 and +0.015 (bootstrap CIs from lane B, not re-run); L2 gate fails |
| R-05 | Lane B L2 check | seeds 0–19 | 600 | F3 identical to F5 per seed: 1/20 (η 1e-3), 14/20 (η 1e-2) | verified | agent reported 20/20; not the same as the agent's claim |
| R-06 | Lane C F0 (zero-enc), F1, F6 | seeds 0–19 | 600 | F0 5.000; F1 526.320 (window 480..599, pre-step); F6 530.543 | verified; no CI (n_eff = 1) | κ = 0 equality agent-reported; F6 rerun needed (d[600] not stored) |
| R-07 | M2 targeted tests (`-k "sailor or m2"`) | — | — | 40 passed, 12 deselected | verified | rerun on the source worktree |
| R-09 | Start-angle sweep (envA), F0 and F1 | 8 angles on R = 5 (θ = 2πj/8) | 60 | F0 final distance 5.000 at all 8 (max |d − 5| 1.3e-9); F1 final distance 9.75 to 20.46, all 8 beyond R; F1 end p_x > 0 for all 8 (starts with p_x < 0 cross p* to +x) | verified (16 rows re-read from JSON) | §13.7 item 9: the p_x ≥ 0 prediction holds (4/4 by sign, 2 of those are float residues at θ = π/2, 3π/2); the p_x < 0 starts were not in the prediction and cross the target instead |
| R-10 | F1 at K = 3000, θ = 0 (single start) | p0 (5,0), seed 0 | 3000 | tail mean (pre-step, errors[2400:3000]) 4114.02; final 4613.19; linear runaway from about interval 1000 at ~1.67 units per interval (every-100 trace in JSON) | verified (JSON read) | K = 600 value 526.32 is a transient; the primary metric is K-dependent and measures runaway speed, not control; §13.4 metric needs revision (decision for Hamm) |
| R-11 | F1 at K = 600, p0 = 5(cos θ, sin θ): θ = 0 (control), π, 3π/4 | R = 5, seed 0 | 600 | θ = 0 reproduces Lane C F1 `errors` exactly (max abs diff 0.0 over 600 values). Pinned mean `errors[480:600]`: θ = 0 526.32, π 507.09, 3π/4 511.31. Final: 619.57, 599.54, 602.65. Minimum (index k in `errors`): θ = 0 5.0 at k = 0–1, rising after; π 0.074 at k = 33; 3π/4 3.536 at k = 28 | verified (JSON re-read by script `analyze_falsifier.py`) | Tail criterion D_j ≤ R: 0 of 3. Transient criterion (min strictly below R): 2 of 3. The boat passes near p* from p_x < 0 and then runs away, as R-09 predicts for p_x < 0. The tail metric supports the §13.7 item 9 falsifier; the transient does not show sustained fixation. Development starts only (not held-out); 3 of 20 dev starts run, see R-12 when complete |
| R-12 | F1 development starts, K = 600, θ ~ Uniform[0, 2π) from `default_rng(2025)` (all 20; script `falsifier/dev_probe.py`) | R = 5, seed 0 | 600 | pinned mean `errors[480:600]` 508.1 to 526.3 (median 520.9); final 600.6 to 619.6; tail mean ≤ R: 0 of 20; min strictly below R after k = 1: 8 of 20, all with p_x < 0 (θ = 103° to 243°; minimum 0.27 at θ = 177°); p_x ≥ 0 starts: 0 of 12 dip below R | verified (JSON re-read by `analyze_dev.py`, n = 20) | sign split is post hoc on the 20 development starts; confirm on the held-out set (not run). Development-start probes only |
| R-08 | F3 mechanism (laneB2): per-interval reward, eligibility; F3 vs F3b (running-mean reward baseline, τ 2 s, exploratory) | seed 0 and seed 1 | 60 | reward r_k ≤ 0 in 60/60 intervals; eligibility ≥ 0 (min 0); 18/28 edges at 0 bound in both F3 and F3b; final distance 5.00 → 20.458 | verified (summaries identical for F3 and F3b, seeds 0 and 1) | seed 0 = seed 1: no noise, not replicates; F3b's baseline does not change the outcome; the sign-definite term is the reward (the boat moves away every interval) |

Negative results so far: no H-state or plasticity strategy beats its control (R-03, R-04); L2 fails for F3 at η 1e-3 and holds only partly at η 1e-2. F3 is seed-invariant at S1 (n_eff = 1).

### 13.9 Next (tournament cycle, 2026-11-19)

Lanes: envA (EX-6, start-angle sweep), laneB2 (EX-7, F3 saturation), critic (EX-8, §13.7 review). Results enter this table only after the lead re-derives each number.
