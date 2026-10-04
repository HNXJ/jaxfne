# Deep review sweep 2026-10-04 — code lane (thorough)

Task: read-only review of jaxfne/, tests/, scripts/, examples/ against match-code-style (review mode).
Worker: general subagent (thorough). No writes made by worker; verification spot-checks by integrator noted [I].
Branch: dev @ d463e574. Deduplicated against closed: P-025, P-026, units removal, pair-gate, oracle hardening, P-016/P-020, P-022/P-023, C5-C7.

## Verbatim report

VERDICT: REVISE

| Sev | Location | Problem | Evidence | Fix |
|---|---|---|---|---|
| blocker | jaxfne/paradigm.py:629-632,691-692,725-762 | `omission_oddball_paradigm(standard_onset_ms, deviant_onset_ms)` accepted/documented but never consumed — events built at buffer offsets only | Grep `standard_onset_ms\|deviant_onset_ms` hits only signature+docstring+None-fold (lines 630,632,646,651,691,692); zero uses in event construction | Wire onsets into event `onset_ms` or refuse non-default values |
| major | scripts/audit_w3_broad_handlers.py:29-154 | `_OVERRIDES` has duplicate dict keys (silent loss) and rotted line anchors — audit classifies wrong lines | Dup keys: `(_construct_connectivity.py,592)` lines 30+150, `(_model_evaluate.py,297)` lines 70+78 (F601 ×2 in ruff); actual handlers now at `_model.py:648` (key says 593), `_construct_connectivity.py:756,878` (key says 592), `neuronal_tensor.py:1235` (key says 913), `_model_simulate.py:889` (key says 592) | Key overrides by handler content/regex, dedupe keys, re-run tally |
| major | jaxfne/solvers.py:87-92 | Unknown `solver_type` silently falls through to `Tsit5` (fail-open) | `if/elif/else: solver = Tsit5()` — no refuse branch | `raise ValueError` on unrecognized `solver_type` |
| major | jaxfne/solvers.py:18-21 vs 119-121 | `SolverConfig(rtol, atol)` silently ignored on Euler path (stored-but-unconsumed) | `solve_ode` routes `rtol/atol` only to `DiffraxSolver`; `EulerSolver` takes `dt` only | Refuse non-default `rtol/atol` with `method="euler"` or document+test invariance |
| major | jaxfne/solvers.py:38,95 | Silent `round()` grid: non-divisible `(t_end-t_start)/dt` truncated without error | `n_steps = int(round(...))`; `agent.py:90` already refuses such rounding elsewhere | Refuse non-integral step counts like `agent.py` does |
| major | jaxfne/validation.py:28-41 | `_is_finite_value` returns `True` for any `int`/`str` without checking (fail-open validator) | `if isinstance(val,(int,str)): return True` — `"abc"` validates finite | Only auto-pass `bool`/`None`; numeric-check the rest |
| major | jaxfne/neuronal_tensor.py:600-608 | Future `schema_version` loads with warning; diverged fields silently dropped | `warnings.warn(...); Loading anyway; fields may be silently dropped` | Fail closed (or `strict=` flag); never load-and-drop silently |
| major | jaxfne/bridges.py:44-52 | `_install_jax_clip_compat` permanently monkeypatches global `jnp.clip`, no restore | `jnp.clip = _clip_compat` module-global side effect on `require_jaxley()` path | Scope the shim (context manager) or document irreversibility + add teardown |
| major | jaxfne/hdp_network.py:34-41; jaxfne/_construct_population.py:31-56; jaxfne/builders.py:64-80 | Three competing "canonical" compositions with different numbers, no single source of truth | `LAYER_CELL_TYPE_FRAC_DEFAULT` (L2 E 0.65) vs `CANONICAL_LAYER_CELL_TYPE_FRACTIONS` (L2 E 0.50) vs `_SUITE2_LAYER_CELL_TYPES_V1` (L2 E 0.75) | One canonical table; others import/derive or are renamed non-canonical |
| major | jaxfne/_model.py:646-649 | Bare `except Exception` on `z` extraction masks real shape/dtype bugs as `None` | `try: z_value = float(positions[idx,2]) except Exception: z_value = None` | Catch `(IndexError, TypeError, ValueError)` only |
| major | jaxfne/streaming.py:33-45 | Hardcoded Izhikevich constants, magic `5.0` syn tau, no Eq cite; `dt_ms>5` makes decay negative | `dv = 0.04*v*v + 5.0*v + 140.0...`, `spiked = v>=30.0`, `s*(1-dt/5.0)`; no named consts | Named `*_MS`/`*_MV` constants with Izhikevich-2003 source + Eq comment; guard `dt_ms` |
| major | jaxfne/streaming.py:53-72 vs plasticity.py:111-129 | STDP `dW` update duplicated verbatim in two kernels | Same `dW_ltp/dW_ltd/update_mask/clip/no-autapse` block in both files | One shared kernel function; both call sites use it |
| major | jaxfne/fields/solvers.py:146 | `convergence_status` threshold `1e-3` absolute, scale-blind, unjustified | `"converged" if residual_norm < 1e-3` on unscaled residual | Relative residual tolerance with justification, or expose as named param |
| major | jaxfne/util.py:444-896 | ~30 bare `except Exception` in summary/diff helpers risk silent misreport | Grep count: 30 of the repo's ~60 broad handlers live in `util.py` alone | Narrow to expected errors; let structural bugs raise |
| major | jaxfne/_runtime_config.py:326-356 | `SurrogateConfig(beta, applies_to)` declaration-only, never consumed (stored-but-unconsumed) | Docstring: "records the declaration only; does not alter dynamics" — no kernel reads it | Consume it in a gradient path or refuse non-default values |
| major | tests/test_analysis_metrics.py:28,48,62,75-82,94+ | Unseeded `np.random.rand` with threshold asserts (`>0.9`, bands) — nondeterministic, flake-prone | Multiple tests draw unseeded randomness then assert tight bands; only some set `np.random.seed` | Seed every stochastic test (explicit `Generator`) or assert distribution-free properties |
| major | examples/00_minimal_column.py:15-39 | Example uses stale vocabulary (`phi_e`/`J_e`/`source` probe modes, legacy builder chain) — read-only, not executed | Modes `["spikes","V_m","source","phi_e","J_e","CSD","LFP"]` vs current `*_proxy` contract; `model.probe/signals` flow unverified | Update to `*_proxy` modes + current API or mark retired |
| minor | scripts+pkg ruff | 171 findings vs P-001's 176 — count roughly confirmed, no mass change | `ruff check jaxfne scripts` → 171 (F401 56, F541 32, F841 21, E741 20, E702 13, E402 12, E722 7) | Unchanged handling; NEW beyond lint is the `_OVERRIDES` rot above |
| minor | jaxfne/geometry.py:35-41 | Magic `0.7` split, `0.01/0.1` weights; positions unitless, weights units unstated | Literals inline, docstring lacks units for `W` | Named consts with units+source |
| minor | jaxfne/analysis/spectral.py:105-107,174-178 | Magic `1e-12`, `100.0`, `3.0` unnamed | `band_power/(max+1e-12)`; `100*exp(-3*mse)` | Named consts with one-line why |
| minor | jaxfne/presets.py:80-90 | `RECEPTOR_KINETICS` NMDA/GABA_B cite "Standard neuroscience literature" (no locator); `sign` mixes int/float | Lines 73/84/95 sign types differ; vague source strings | Precise sources; uniform float signs |
| minor | jaxfne/__init__.py:396 | `Net = Model` silent alias vs documented deprecation/removal | `public_surface` deprecates `Net`; alias emits no warning | `DeprecationWarning` on use (or document silent intent) |
| minor | jaxfne/paradigm.py:311,404-408 | `event_windows` silently wins over `event_codes`; `comparison_label` silently overwrites a sequence code | No ambiguity refusal; map assignment overwrites | Refuse joint/conflicting specification |
| minor | jaxfne/stimulus.py:27 | `n_steps=int(duration/dt)` silently truncates remainder | Remainder time dropped without error | Refuse non-divisible grids (cf. `agent.py:90`) |
| minor | jaxfne/analysis/metrics.py + tests | NaN→0 imputation by convention (pinned by tests, documented) | `test_nan_handling` asserts silent zeroing | Keep convention but note in docstrings that NaN input is masked, not flagged |
| minor | jaxfne/_pipeline.py:976,990 | Bare-except dtype/itemsize fallbacks silently misreport memory | `except Exception: itemsize=4` / `pass` with noqa | Narrow to `(TypeError, ValueError)` |
| minor | jaxfne/tutorial_utils.py:52,148-154 | Leftover `print` debugging in library path | Grep hits `print(` outside docstrings/examples | Route through logging or return values |
| minor | examples/08_neuronal_tensor_first.py:59-64 | Hardcoded `/tmp` path + deprecated `load_neuronal_tensor/construct_neuronal_tensor` imports | Lines 59,64; compat names per `COMPATIBILITY_DEPRECATIONS` | `tempfile`/cwd-relative path; current `load`/`construct` API |

P-014/P-015 probe (6 params, code-path reading): `h_state_readout/coupling` CONSUMED (`emitters.py:3713-3725`); `canonical_biophysics` CONSUMED (`_construct_core.py:216,254,1380`); `synaptic_kernel` CONSUMED (construction gate + `test_consumption_gate_055`); `precision` REFUSED (`_runtime_config.py:145-149`); `n_contacts` CONSUMED (field readout); `vmap` REFUSED-or-batch-mode (tensor path refuses, config path batch-only). No new unconsumed params among these. New violations found instead: paradigm onsets (blocker), Euler `rtol/atol` + `solver_type` defaulting (majors), `SurrogateConfig.beta` (major).

Tests circularity spot-check: `test_augment.py` (`to_dict()==` no-op asserts) and `test_api_smoke.py` shape asserts are legitimate behavior pins, not circular. Genuinely circular-looking: none confirmed by reading; the real test defect is unseeded randomness (`test_analysis_metrics.py`, major above). `test_consumption_gate_055.py` is the strongest suite in the repo (every field has a case).

Counts: blockers 1 / majors 15 / minors 11.

Top-5: (1) paradigm.py unconsumed onset params; (2) `audit_w3_broad_handlers._OVERRIDES` rot; (3) solvers.py fail-open + ignored rtol/atol; (4) three competing canonical compositions; (5) unseeded stochastic tests in `test_analysis_metrics.py`.

## Integrator verification [I] (2026-10-04, dev)
- Blocker CONFIRMED: `standard_onset_ms|deviant_onset_ms` grep hits only signature+docstring+None-fold (8 hits, all ≤692); zero uses in event construction. → P-027.
- Solvers fail-open CONFIRMED by reading (`else: solver = diffrax.Tsit5()`). → P-028.
- Minted: P-027…P-032 (defects), P-033 (prose batch), P-034 (figures); `_OVERRIDES` rot appended to todo item 6 (already logged); H2 dismissed (fact_stack touches are quiz-authorized rounds); H3 dismissed (Decisions-in-force is standing convention).
