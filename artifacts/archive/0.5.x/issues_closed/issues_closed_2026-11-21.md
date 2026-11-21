# jaxfne closed issues, batch of 2026-11-21 (0.5.10)

Moved verbatim from the open log (`artifacts/assurance/issue_log/ISSUE_LOG.md`) once the file budget allowed
it (S16, Hamm 2026-11-21: `budget_files` 2825). Earlier closed entries: `issues_closed_2026-10-26.md`,
`issues_closed_2026-10-27.md` and `issues_closed_2026-11-03.md` beside this file, and `../issues_closed.md`,
`../issues_closed_2026-10-20.md` and `../issues_closed_2026-10-25.md`, pinned write-once.

Entry P-040 comes from the whole-project scan of 2026-10-18 (six Haiku workers,
read-only; each line was re-opened and confirmed by the dispatcher unless marked inferred).

### P-040

- Date: 2026-10-18. Type: SCIENCE. Area: Destexhe 2009 lane. Severity: high.
- Re-checked 2026-11-12: `destexhe2009/PREREG_P3.md` no longer exists in the lane, so the row's original
  contradiction (a header reading "draft, UNSIGNED" while `CALIBRATION.md` scores E1-E4 against it) cannot be
  re-read at its cited path. `CALIBRATION.md:3-5` still states the verdict is OFF and unscored, against "the
  unsigned draft `PREREG_P3.md`". `docs/start/replications.md` now reads "pending". The lane is closed by the
  0.5.10 scope freeze (Hamm 2026-10-25) and its P3 scored FAIL, so no scored claim rests on the unsigned prereg.
- Resolution 2026-11-12: closed as superseded, not as fixed. The unsigned-prereg scoring it reports was already
  marked unscored by the 2026-10-18 ruling and the lane's own calibration header; the file it names has been
  removed, so no reader can now take it for a signed protocol. Any future Destexhe scoring needs a new signed
  prereg, which is T-32's 0.5.11 lane work.
- Original observation, kept as history:
  `CALIBRATION.md:3-4` scores E1-E4 against a prereg whose header reads "draft, UNSIGNED — no scored run may use
  it" (`:1`), and its final section says E1 passes (:427) and scores the reserved seeds 4-8. `REPLICATIONS.md:12`
  said "no verdict yet" and "28 passed".
- 2026-10-20 (release plan A3): CALIBRATION header and steps 1-6 marked unscored pilots (ruling
  2026-10-18), body verdicts too; REPLICATIONS row agrees. Closes with the lane's next prereg.
- 2026-10-26: the Destexhe prereg moves to 0.5.9 (scope freeze, Hamm 2026-10-25); the entry stays
  open into 0.5.9.
- 2026-11-03: P3 v3 was signed and scored FAIL 0 of 5 seeds (RESULT_P3.md); lane closed by Hamm 2026-11-03. Destexhe status reads pending/closed. Closed.

### P-057

- Date: 2026-11-17. Type: BUG. Area: public `simulate()` on the Izhikevich route with `enable_hdp=True` and a
  registered rule. Severity: medium for any claim that a zero-parameter rule run equals the rule-free run.
- Observation: a registered rule with every gain at zero does not reproduce the rule-free run through `simulate()`.
- Minimal reproduction: 100-cell `NeuronalTensor` (E 0.9, PV 0.1, one L4 layer), `construct(..., seed=0,
  duration_ms=200, dt_ms=0.5)`, then `simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)` with and without
  `runtime=RuntimeConfig(enable_hdp=True, hdp_params={"hdp_rule": "synthetic_presyn_gain"})` (defaults all 0).
  Reproduced twice (author and independent review, 2026-11-17).
- Expected: identical spikes (`N_system` null under matched PRNG and inputs).
- Actual: 211 spikes rule-free, 214 with the zero rule; `ephaptic_field` at `alpha=0` also gives 214. At kernel
  level with `noise_scale=0.0` the identity holds bit-exactly (`tests/test_hdp_reduction_identity.py`).
- Cause (2026-11-18): the reproduction compared two backends. The rule-free run takes the default
  `recurrent_backend="dense"` (`_runtime_config.py:82`) to `simulate_eig_izhikevich` (`_model_simulate.py:1084`);
  an engaged rule always takes the edge-list registered kernel (`_model_simulate.py:641-643`, `:685-703`), with another
  noise stream (`_chain_noise_schedule`, `:703`). The earlier text excluded membrane noise from kernels the public
  route does not take. On the same backend the runs are bit-identical: rule-free with
  `recurrent_backend="edge_list"` gives 214 spikes, equal to the zero rule in spikes and `V_m` (max |dV| 0.0); at
  `noise_scale=0` 208 = 208. Evidence, with sha256 in `SHA256SUMS`: `F:/warehouse/jaxfne/P-057_2026-11-18/`
  (`stepA_compare.json`, `stepC_compare.json`, scripts).
- Closed 2026-11-18 (Hamm): the zero-gain identity shows no rule defect (one model, one seed, the two rules
  below). `tests/test_hdp_zero_rule_public_identity.py` checks the
  edge-list identity through `simulate()` for `synthetic_presyn_gain` and `ephaptic_field` at `alpha=0`; with a
  dense rule-free reference it fails. The backend difference is P-062.

### P-058

- Date: 2026-11-17. Type: BUG (portability). Area: `artifacts/etudes/papers/pcl/port/run_r2_reference.py::_code_sha`.
  Severity: low; it breaks the PCL lane of `run_test_gate.py broad` in any git worktree.
- Observation: the broad gate on `_temp_int-1117` at 3c7e497e, run in a worktree, failed 5 PCL tests
  (`test_run_r2_reference.py`: `e`, `l`, `n`, `p`, `a`) with `FileNotFoundError` on
  `.git/worktrees/<name>/packed-refs`. The same file at the same SHA passes 16/16 in a regular clone.
- Cause: `_code_sha` resolves `HEAD` by hand and reads branch refs and `packed-refs` from the per-worktree git dir;
  in a worktree those live in the common dir (`commondir`).
- Possible change: resolve with `git rev-parse HEAD` (or read `commondir`), with tests for a regular clone, a
  worktree and a loose ref (Hamm 2026-11-17). Kept out of the `_temp_int-1117` integration by Hamm's ruling.
  Closes when those three tests pass and the PCL lane passes in a worktree.
- Fix 2026-11-21 (R62, `_temp_p058`): `_code_sha` reads `commondir` and looks up the branch ref (loose, then
  `packed-refs`) in the common dir, HEAD in the worktree's own dir; still no subprocess.
  `test_run_r2_reference.py::test_r_code_sha_regular_clone_packed_ref`, `::test_s_..._worktree_packed_ref_in_common_dir`,
  `::test_t_..._worktree_loose_ref_in_common_dir`: s and t failed on the old code (`FileNotFoundError`), all three
  pass; the file passes 19 of 19 in a git worktree (vwin-pc). Independent review ACCEPT 2026-11-21 (old code
  raised `FileNotFoundError` on both worktree fixtures; new `_code_sha` equals `git rev-parse HEAD` on a real
  worktree). On dev with broad gate rc 0 (8459 tests, 0 failed). Closed 2026-11-21.

### P-059

- Date: 2026-11-17. Type: BUG. Area: `jaxfne/optim/core.py` `sdr_transform`, `gsdr_transform`, `agsdr_transform`
  (public Optax transforms; no `jaxfne/` caller, `Model.tune` does not reach them). Severity: medium for any use
  of these transforms on a pytree with equal-shape leaves.
- Observation: found by the 2026-11-17 architecture review (packet A, defect D1). One `key_delta` served every leaf
  of the noise `tree_map`, so equal-shape leaves drew identical noise. Direct `update()` check with zero gradients:
  two (4,) leaves starting at 0 and 1 got identical nonzero updates at steps 0 and 1 in all three transforms.
- Fix (Hamm 2026-11-17, scoped): `_leafwise_normal` draws each leaf from its own split key. Fixed-seed outputs
  change for multi-leaf parameter trees; a single-leaf tree keeps its draws (see P-060).
  `tests/test_optim_transform_leaf_noise.py` failed on the old code for all three transforms and passes;
  `scripts/score_objective_optimizer_checks.py --check` still matches its `score.json` (stochastic term off there).
- Left: the transforms' `update()` is not jittable (state dataclasses are not registered pytrees); recorded, not
  changed. Closed by the fix commit for the three transforms named; the sibling is P-060.

### P-060

- Date: 2026-11-17. Type: BUG (port fidelity). Area: `jaxfne/optim/gsdr.py:305-310` `gsdr_mcdp_transform`. Found
  by the outside review of the P-059 fix.
- Observation: the same pattern as P-059: one `key_delta` for every leaf of the noise `tree_map`, so equal-shape
  leaves drew identical noise.
- Source (Hamm 2026-11-17: check the source first): `HNXJ/GSDR` b4a96fae `gsdr/optimizers.py` splits one key per
  parameter leaf in SDR (:69-71), GSDR (:194-196) and AGSDR (:303-305). The shared draw was a port deviation.
- Effect on results: the signed P3 fit (`RESULT_P3_fit.md`, FAIL) passes one parameter vector
  (`fit/fit_gsdr_500.py:206`), a single leaf, so the defect could not correlate anything there; its verdict stands.
  `_leafwise_normal` draws a single-leaf tree from the key itself, so that fit replays its draws bit for bit.
  Tested: float32 only (`tests/test_optim_transform_leaf_noise.py::test_single_leaf_draw_is_unchanged`; outside
  review probe, 6 steps, `a_dynamic` on and off, byte-identical to 24a95963). Other dtypes hold by construction
  (same key, shape, dtype), untested. No caller passes a multi-leaf tree to this transform (`scripts/`,
  `artifacts/`, `jaxfne/` grepped 2026-11-17).
- Fix: `gsdr_mcdp_transform` uses `_leafwise_normal`; `test_gsdr_mcdp_equal_leaves_get_independent_noise` failed on
  the old code and passes; `scripts/score_port_parity_checks.py --check` still matches. Closed by the fix commit.

### P-061

- Date: 2026-11-17. Type: BUG (test robustness, Windows). Area: `artifacts/etudes/papers/caminos/caminos_api.py:356-357`
  (`mtmp.write_text(...)`, then `os.replace(mtmp, out / "manifest.json")`). Severity: low; a spurious gate failure.
- Observation: the first broad gate on 32a81d86 (regular clone, Windows) failed
  `test_caminos_jax.py::test_p6_gate4_converter_known_answer` with `PermissionError: [WinError 5] Access is denied`
  on the rename `manifest.json.tmp` -> `manifest.json` in the pytest temp dir. The test passed 3 of 3 alone on the same
  clone, the full caminos lane passed (288/288) on the second gate run of the same SHA, and no commit since
  24a95963 touches `artifacts/etudes/papers/caminos/`.
- Cause (inferred, not proven): another process (scanner or indexer) held one of the files during `os.replace`.
- Diagnosis 2026-11-17 (open; the rerun pass shows intermittency, not a fix):
  - Positive control: with a second process holding `manifest.json` open, `os.replace(tmp, manifest.json)` raised
    the same `PermissionError: [WinError 5] Access is denied` on the same file pair. A held handle suffices.
  - Stress, the test's pattern (write then replace, twice per dir), 4 processes: 0 errors in 4,000 replaces under
    `D:/pytest_temp`, and 0 in 12,000 (3 runs) under `D:/tmp/pytest-of-nejath`, the failing run's temp root. Not
    reproduced at idle; the failing run was under full-gate xdist load.
  - Stress shaped like `write_run` (outside review finding): per seed a (120000, 3) int32 `.npy` (about 1.4 MB)
    through tmp + `os.replace`, then a read of the existing `manifest.json`, then a larger manifest replace; 4 and
    8 processes: 0 errors in 12,000 `write_run`s (24,000 replaces) under the failing run's temp root. Still
    untested: the `import jax` / `jax.devices()` step between the two writes, and pytest-xdist load.
  - Windows Defender real-time protection is on; its exclusion list is unreadable without admin. The holding
    process is unidentified.
  - Evidence, with sha256 in `SHA256SUMS`: `F:/warehouse/jaxfne/P-061_2026-11-17/` (failing gate log, passing gate
    log, `stress_replace.py`, `run_default_root.sh`, `stress_replace_full.py`, `run_full_default_root.sh`,
    `stress_full_results_2026-11-17.txt`). Gap (review 2026-11-17): the three solo reruns, the positive control
    and the first stress runs have no archived output, so those numbers rest on the session record only.
  - To prove the cause: reproduce under gate load, or capture the holding process (Process Monitor or a handle
    dump during the gate; needs admin).
- Possible change: retry `os.replace` on `PermissionError` a few times with a short back-off, with a test that
  injects one failed replace. That closes the spurious failure (T-96), not the diagnosis.
- Recurrence 2026-11-19: 0 `WinError 5` in the six later broad-gate logs (06f9bf26, 5e1818f9, 0f1f0d22,
  32d24add, 4dee0a6d, 97395d74; caminos lane PASS in every log that reached it).
- Repair (Hamm 2026-11-19, "Retry on PermissionError"): `caminos_api._replace` retries `os.replace` up to 5 times
  (back-off 0.05 s x attempt) and re-raises the last error; both renames in `write_run` use it. Tests
  `test_caminos_jax.py::test_p6_write_run_retries_a_held_replace` (red before),
  `::test_p6_write_run_reraises_a_persistent_permission_error`. Cause stays inferred. Sibling sites in
  `run_caminos.py` are todo T-96. Outside caminos, 21 bare `os.replace(` calls in 15 files (brunel2000,
  destexhe2009, gsdr2026, mejias2016, pcl/port, potjans2014, vogels2011, `e2b_v2_executor.py`, `scripts/lane_kit.py`,
  `scripts/new_lane.py`; independent review 2026-11-19) have no retry; none has failed so far.
- Closed 2026-11-19: on dev at 314f99f2 (broad gate on 19ab3afe rc 0, 8335 passed, caminos lane PASS;
  independent review ACCEPT; CI (Fast) 44546980997 green 10/10). Evidence: `F:/warehouse/jaxfne/P-061_2026-11-17/`.
  T-96 (Hamm 2026-11-19): `run_caminos.py` gets its own `_replace` for both renames (the script imports neither
  jaxfne nor `caminos_api`); test `test_caminos_analysis.py::test_pilot_retries_a_held_replace` (red before).

### P-062

- Date: 2026-11-18. Type: SCIENCE. Area: public `simulate()` on the Izhikevich route, `recurrent_backend`.
  Severity: medium for any comparison of a rule run with a rule-free run, or of the two backends.
- Observation (found closing P-057): (1) an engaged HDP rule runs the edge-list registered kernel whatever
  `recurrent_backend` says (the `use_hdp` branch, `_model_simulate.py:641-703`, never reads it);
  `recurrent_backend="dense"` with a rule gives the same spikes and `V_m` as `"edge_list"`, with no warning or
  refusal (independent review 2026-11-18). (2) The dense and edge-list rule-free runs differ on the P-057 model:
  211 vs 214 spikes with default noise (`V_m` differs from step 0, different noise streams), and 206 vs 208 at
  `noise_scale=0`, where `V_m` first differs at step 1 by 3.8e-6 mV and the gap then grows to about 100 mV.
  Evidence: `F:/warehouse/jaxfne/P-057_2026-11-18/` (`stepA_compare.json`, `stepC_compare.json`,
  `stepC2_zero_noise_onset.json`).
- (2) attributed 2026-11-18, not a defect: the backends are different, documented synaptic models. Dense couples
  `weights @ prev_spikes` with no synaptic time constant (`emitters.py:748`); edge-list filters each edge through
  an exponential receptor state (`emitters.py:1609`, `:1712`); both docstrings say "NOT trajectory-equivalent"
  (`emitters.py:616-621`, `:1540-1544`). The step-1 gap (2^-18 mV, 2 cells) is float32 summation order of the
  synaptic sum (`W @ prev` vs `segment_sum`); float64 removes it but the run stays 206 vs 208, and with exact sums
  dense equals the edge-list `instantaneous` mechanism over 400 steps. The run gap is the receptor filter. The
  earlier "floating-point order" reading was wrong. Evidence: `F:/warehouse/jaxfne/P-062_2026-11-18/`
  (`e3_float32.json`, `e3_float64.json`, `e4_float32.json`, scripts).
- (1) also mislabels provenance (observed): an HDP run recorded `recurrent_backend: "dense"` in its metadata,
  runtime report and manifest while the edge-list kernel ran (`probe_meta.json` in the same folder). Same defect
  (independent review 2026-11-18, observed): `simulate_batch` with HDP (always the edge kernel there), and
  homeostasis in both entry points, and a placeholder-W model (edges only in `params["edge_list"]`) run through
  `simulate()` with an explicit `RuntimeConfig` (without one it recorded `edge_list`). Not affected: the
  continuation path (refuses anything but `edge_list`).
- Repair, Hamm 2026-11-18, two steps toward the invariant "an explicit backend is honoured or refused, never
  replaced". Step 1 (provenance, no API change): in `simulate()` and `simulate_batch()`, metadata
  `recurrent_backend` (and the manifest's `used_recurrent_backend`) records the backend that ran, the runtime
  report keeps the requested value, and an explicit model-level `recurrent_backend="dense"` is refused under
  homeostasis, under an engaged rule in `simulate()`, and under any `enable_hdp` in `simulate_batch()` (which
  always runs the HDP edge kernel) (`_edge_only_controller`, `_executed_recurrent_backend` in
  `_model_simulate.py`; the P-062 tests in `tests/test_hdp_zero_rule_public_identity.py`, 8 of 9 red on
  6b7acd62). Outputs bit-identical
  to 6b7acd62 (ad hoc, not in the repo tests): spikes, `V_m`, HDP and homeostasis state arrays for `simulate()`
  and `simulate_batch()` (independent review, 52 arrays; HDP zero gain, `ephaptic_field` alpha 0.5, legacy HDP,
  homeostasis, rule-free). Step 2 (open, separate
  change): `auto` selects a backend compatible with the engaged controllers; `dense` and `edge_list` run as named
  or are refused; unknown values are rejected at the configuration boundary (none are today: any string but
  `"edge_list"` runs dense). Default HDP numerics and rule-free behaviour stay. Closes when step 2 is done.
- Step 1 on dev 2026-11-18 at 06f9bf26 (broad gate rc 0, 8308 passed; CI (Fast) 44392030376 green).
- Step 2 audit (2026-11-18, read-only at 06f9bf26): about 20 dispatch sites compare only `== "edge_list"`, so
  `auto` is resolved once per entry point before dispatch; the requested value reaches `metadata["runtime"]`,
  the manifest and `receipt_id` (`_model_readout.py:62-71`); the compile-cache key (`_model_simulate.py:3595`)
  takes the resolved value; `jtfne.runtime()` (`_construct_presets.py:48`) and the column builder
  (`builders.py:218`) write an explicit `"dense"` and move to `auto`; no `RuntimeConfig` round trip exists.
  Rulings, Hamm 2026-11-18: default-run receipt IDs change (requested `"auto"` is hashed; changelog note);
  `auto` on lif, rate and adex runs edge_list instead of refusing. `auto` resolves to edge_list under an
  edge-only controller, a placeholder W, the receptor kernel, rule edges, finite delays and `return_state` or
  continuation, otherwise dense; each of those that refuses the default today runs instead. Failing tests at 32f48569: 17 red, 8 green on 06f9bf26.
- Step 2 implementation (`_temp_p062-auto`, Hamm 2026-11-18 "Implement as planned"): `ALLOWED_RECURRENT_BACKENDS`
  checked in `RuntimeConfig.__post_init__` and `Configuration.runtime`; `_resolve_recurrent_backend` in
  `_model_simulate.py` resolves once per `simulate()` / `simulate_batch()` (and for the tune and optim-handle
  checks via `resolved_recurrent_backend`). Finite
  delays do not steer `auto` in `simulate_batch()`: it never refused a dense run with delayed edges, so moving
  those runs would change results (open: whether that dense batch run drops the delays). The adex route keeps
  its `edge_list` default and reads `auto` as it. Default-runtime outputs bit-identical to 06f9bf26 (59 arrays:
  spikes, `V_m`, sources, 22 HDP and homeostasis diagnostics, 17 `simulate_batch` arrays; rule-free, HDP zero
  gain, gain, `ephaptic_field` alpha 0.5, homeostasis); negative control: the cases differ from each other.
  Seen on the way, not caused here and unverified: `simulate_batch` HDP zero gain and gain give equal `V_m`
  on base and candidate (the batch kernel may not read `hdp_rule_params`).
- Step 2 review (independent, fb3e4d6f, REVISE): (F1) an `auto` RuntimeConfig took a model-level `edge_list`
  and changed a run that ran dense on 06f9bf26; fixed, the model-level value now only refuses (`dense` under an
  edge-only controller or placeholder W), as before step 2. (F4) lif, rate and adex recorded no executed
  backend (the manifest fell back to `"dense"`); they now record `edge_list`. Behaviour changes kept and
  stated in the changelog: an explicit `RuntimeConfig(recurrent_backend="dense")` under homeostasis, HDP or a
  placeholder W, which ran edge_list silently, is refused; an explicit `RuntimeConfig("edge_list")` over a
  model-level `"dense"` now runs. Open, minor: an explicit `RuntimeConfig("auto")` cannot be told from the
  default, so a model-level `"dense"` still refuses it under an edge-only controller; `update_metadata`
  writes `recurrent_backend` unchecked (only `Configuration.runtime` validates).
- Broad gate on 5e1818f9 (rc 1, 5 failed): `_has_finite_delays` read `params["edge_list"] = None` (adex
  tune_handles), fixed; four tests asserted that the old default refuses the receptor kernel, delays and
  rule edges (simulate, batch); they now request `dense` explicitly, and a test pins `auto` running rule edges
  on edge_list.
- Closed 2026-11-18 (Hamm): step 2 on dev at 0f1f0d22 (broad gate rc 0, 8324 passed, 0 failed; independent
  re-review ACCEPT on 5e1818f9; CI (Fast) 44424229013 green, 10 of 10 jobs). Evidence:
  `F:/warehouse/jaxfne/P-062_2026-11-18/` (gate logs, `step2/` probes, `SHA256SUMS`). Follow-ups P-063, P-064.
- Follow-up 2026-11-19 (Hamm, "update_metadata check"): `Configuration.__post_init__` now validates
  `metadata["recurrent_backend"]`, so `update_metadata()` and `Configuration(metadata=...)` reject unknown
  values as `runtime()` did; `runtime()`'s own copy of the check is removed. Test
  `test_hdp_zero_rule_public_identity.py::test_unknown_backend_is_rejected_where_the_configuration_is_built`
  (red before). Explicit `RuntimeConfig("auto")` cannot be told from the default: closed as designed
  (Hamm 2026-11-19); `docs/reference/api/core/runtime.md` says explicit `"auto"` defers to a model-level
  `"dense"` and `"edge_list"` overrides it.

### P-063

- Date: 2026-11-18. Type: BUG. Area: `simulate_batch()` on the Izhikevich route, finite edge delays.
  Severity: high for any batch statistic of a model with nonzero `delay_steps` run on dense.
- Observation: `simulate_batch` with `recurrent_backend="dense"` runs a model whose edges carry nonzero
  `delay_steps` and drops the delays; `simulate()` refuses the same run ("no finite-delay path",
  `_model_simulate.py`, `_simulate_arrays`). `auto` in `simulate_batch` keeps dense for delays (P-062 step 2
  left it so that no succeeding run changed).
- Minimal reproduction: `suite2_net1_config(seed=7, n=40, duration_ms=60, dt_ms=1)`, delay_steps 0 vs 5 on
  every edge, `simulate_batch(n_seeds=2)`: dense `V_m` identical for 0 and 5; edge_list `V_m` differs
  (positive control). Evidence: `F:/warehouse/jaxfne/P-063_P-064_2026-11-18/` (`probe_batch_delay.py`, log).
- Expected: dense refuses nonzero delays as in `simulate()`; `auto` selects edge_list.
- Failing tests: `tests/test_delay_boundary_v0417.py::test_dense_batch_rejects_nonzero_delay_steps`,
  `::test_auto_batch_runs_delayed_edges_on_edge_list`. Repair changes the output of every default batch run
  with delayed edges (those outputs ignore the delays today).
- Repair (Hamm 2026-11-18, "Fix both"): `simulate_batch` refuses dense with finite delays as `simulate()` does,
  and `auto` selects edge_list for them (the resolver's `delays` switch is removed).
- Closed 2026-11-19: on dev at 32d24add (broad gate rc 0, 8329 passed; CI (Fast) 44442354065 green 10/10; two
  independent reviews ACCEPT).

### P-064

- Date: 2026-11-18. Type: BUG. Area: `simulate_batch()` with `enable_hdp=True` and a registered rule.
  Severity: high for any batch statistic of a registered HDP rule with non-default `hdp_rule_params`.
- Observation: `simulate_batch` passes `kernel_kwargs.get("hdp_rule_params", {})` where `kernel_kwargs =
  _hdp_kernel_kwargs(_hdp)` never carries that key, so the registered kernel always gets `{}`; `simulate()`
  passes `hp.get("hdp_rule_params", {})`. A rule with gains runs at its zero-gain defaults, and
  `ephaptic_field` is refused ("needs src, dst, weight and v_rest").
- Minimal reproduction: 60-cell tensor, `synthetic_presyn_gain` with `{"gamma": 5.0, "k_h": 2.0, "k_w": 2.0}`
  vs no params, fresh model each run: `simulate()` `V_m` differs, `simulate_batch()` `V_m` identical.
  Evidence: `F:/warehouse/jaxfne/P-063_P-064_2026-11-18/` (`probe_batch_rule.py`, log).
- Failing tests: `tests/test_hdp_zero_rule_public_identity.py::test_simulate_batch_applies_hdp_rule_params`,
  `::test_simulate_batch_runs_ephaptic_field`. Repair changes the output of every batch run of a registered
  rule with non-default parameters (those outputs ignore the parameters today).
- Repair (Hamm 2026-11-18, "Fix both"): the registered batch kernel gets `hdp_params["hdp_rule_params"]`, as in
  `simulate()`. Against 06f9bf26 on the P-062 probe, only the gain-rule batch arrays change (3 of 58) and the
  `ephaptic_field` batch runs; every `simulate()` output and diagnostic is bit-identical. Closes when on dev.
- Sibling (independent review, 2026-11-18): the same call dropped `plasticity_mask`, which `simulate()` passes,
  so masked edges learned in the batch. It now passes `kernel_kwargs.get("plasticity_mask")`; unmasked runs pass
  `None`, the kernel default. Test `::test_simulate_batch_applies_plasticity_mask` (red before the repair);
  `::test_simulate_batch_runs_ephaptic_field` now asserts alpha 0.5 vs 0 changes `V_m`.
- Closed 2026-11-19: on dev at 32d24add (gate and CI as P-063). Unmasked outputs vs 05240cd0: 62 of 62 probe
  arrays bit-identical.

### P-065

- Date: 2026-11-18. Type: BUG. Area: homeostasis kernel, `simulate()` and `simulate_batch()`, finite edge delays.
  Severity: high for any homeostasis run of a model with nonzero `delay_steps`. Predates P-062.
- Observation: with `enable_homeostasis=True` on edge_list, `V_m` for delay_steps 5 equals delay 0 in both
  `simulate()` and `simulate_batch()`; without homeostasis it differs (positive control). The run neither
  refuses nor records the drop.
- Minimal reproduction: `suite2_net1_config(seed=7, n=40, duration_ms=60, dt_ms=1)`, delay_steps 0 vs 5 on every
  edge, `RuntimeConfig(recurrent_backend="edge_list", enable_homeostasis=True)`. Evidence:
  `F:/warehouse/jaxfne/P-063_P-064_2026-11-18/` (`probe_homeo_delay.py`, log).
- Expected: the homeostasis kernel honours delays, or refuses finite delays.
- Repair (Hamm 2026-11-19, "Refuse"): `simulate()` and `simulate_batch()` raise "enable_homeostasis has no
  finite-delay path" when the homeostasis kernel would run on edges with nonzero `delay_steps`. Tests
  `tests/test_delay_boundary_v0417.py::test_homeostasis_rejects_nonzero_delay_steps` (red before the repair),
  `::test_homeostasis_runs_zero_delay_steps`. No successful run changes: the P-062 probe vs 32d24add gives 62 of
  62 arrays bit-identical.
- Closed 2026-11-19: on dev at 4dee0a6d (broad gate rc 0, 8331 passed, 0 failed; independent review ACCEPT;
  CI (Fast) 44517404267 green 10/10). Evidence: `F:/warehouse/jaxfne/P-065_2026-11-19/`. Review notes, minor:
  `ablation="disconnected_null"` with delays is refused although zero weights make delays inert; traced
  `delay_steps` skip the check (inferred).
