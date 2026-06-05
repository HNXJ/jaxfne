# jaxfne Glossary & Benchmark

**A living document.** This is the canonical 100-factor benchmark for `jaxfne` —
10 categories × 10 factors. It is *not* a one-time audit: each factor defines the
**ideal (100/100)** so the package can be scored, re-scored, and driven toward
the ideal over time. Update it whenever a factor's status changes.

- **Status:** living · **Last reconciled:** 2026-06-04 @ `main == dev == 33f99db` (`jaxfne 0.3.29`)
- **Scope guard:** scores are *evidence-backed where marked, otherwise `TBD`* (needs measurement). Never round an estimate up to a measured score.

## How to use

1. To benchmark: score each factor 0–100 against its **Ideal** column, citing a receipt (command + output, test name, or file:line).
2. `TBD` = not yet measured this cycle — do not treat as passing.
3. Roll up per-category means and an overall mean (see §Rollup). Record the date + SHA each cycle.
4. When a backlog item lands, update the affected factor *and* its evidence in the same change.

## Scoring rubric

| Band | Meaning |
|---|---|
| 90–100 | Ideal met, receipt on file |
| 75–89 | Strong; minor gap or partial coverage |
| 50–74 | Partial; known work remaining |
| 25–49 | Stub/declared only |
| 0–24 | Absent or violated |
| `TBD` | Not measured this cycle |

Legend: ✅ measured this cycle · 🔶 estimated from prior audit · ⛔ gated/planned

---

## A. Provenance & Release Engineering

| ID | Factor | Ideal (100/100) | Score | Evidence / notes |
|---|---|---|---|---|
| A01 | Git provenance | clean tree; `main==dev` aligned to a known SHA | 100 ✅ | `main==dev==33f99db`, clean |
| A02 | Version metadata alignment | `pyproject`/`__init__`/`mkdocs`/`version.md` identical | 100 🔶 | v0.3.29 aligned (hotfix e377829) |
| A03 | Tag/release identity | annotated tag; peeled commit SHA == intended | 100 ✅ | `v0.3.29^{}`==fab4c9c |
| A04 | Dist hygiene | `dist/` holds exactly the current-version artifacts | 90 🔶 | zsh `*.egg-info` glob trap (B02) — fixed manually, script unhardened |
| A05 | Build + twine | `python -m build` + `twine check` pass | 100 ✅ | both PASSED on fab4c9c |
| A06 | CI matrix | green on all supported Pythons (3.10/3.11/3.12) | 100 ✅ | PRs #21–23 green ×3 |
| A07 | Release freeze discipline | one CI receipt at terminal state; no mid-run mutations | 90 🔶 | doctrine followed; not automated |
| A08 | Changelog completeness | every release has a dated entry + SHA | TBD | needs `changelog.md` audit |
| A09 | Install smoke | `pip install` + Colab import both verified | TBD | Colab path unmeasured |
| A10 | Remote-mutation gating | tag/upload/release each separately authorized | 100 ✅ | per release-mutation-guard |

## B. Public API & Surface Hygiene

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| B01 | `__all__` curated | exactly the intended public names; no private leaks | 88 ✅ | `_KNOWN_METRICS` leaks into `__all__` |
| B02 | Canonical import | `import jaxfne as jtfne` everywhere | 100 🔶 | doctrine enforced |
| B03 | No invented/dead names | every public name resolves to a real object | 96 ✅ | 146 names resolve |
| B04 | Loud stubs not silent | unimplemented paths raise, never silently fall back | 100 ✅ | emitter-family guard (PR#21); LIF/GLIF raise |
| B05 | Namespace leaks | no stray module objects in `dir(jtfne)` | 80 ✅ | `sys` leaks (not in `__all__`) — deferred (B03 in loop backlog) |
| B06 | Deprecation/wrapper policy | renames keep wrappers; loud deprecation | 84 🔶 | TuneResult tuple-unpack deprecation pending |
| B07 | Dataclass/spec consistency | specs JSON-safe, typed, validated | 90 🔶 | broad ConfigValidationResult coverage |
| B08 | Selector/identity grammar | `NodeIdentity` quartet + `model.select` stable | 96 ✅ | selectors tested v0.3.29 |
| B09 | Signals query API | `Signals.get`/`get_signal` stable + finite | 96 ✅ | tested v0.3.29 |
| B10 | Module organization | thin root routing layer; one role per module | 80 🔶 | tracked root already thin; `.legacy/` 121 files |

## C. Config / Model / Simulation Contract

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| C01 | Config schema + validators | typed schema, strict validation | 92 🔶 | `Configuration.validate()` |
| C02 | Config→simulate contract | honors runtime defaults or fails loud | 96 ✅ | verified smoke |
| C03 | `construct()` entry point | earliest stable public entry; deterministic | 96 ✅ | smoke + family guard |
| C04 | Emitter-family validation | unsupported family raises, no silent Izhikevich | 100 ✅ | PR#21 + 7 tests |
| C05 | Seed propagation | deterministic seed → identical run | 92 🔶 | explicit PRNG paths |
| C06 | Trial/paradigm contract | trial batch + paradigm JSON-safe | TBD | needs focused check |
| C07 | Readout spec grammar | declarative ReadoutSpec → ReadoutResult | 88 🔶 | present, partial doc |
| C08 | Suite configs reproducible | `suite2_*` deterministic | 96 ✅ | smoke passes |
| C09 | Strict JSON round-trip | `.jcfg.json` save/load lossless | TBD | needs round-trip test |
| C10 | Truth boundary metadata | every config carries truth gates | 96 🔶 | `config_truth_boundary` |

## D. JAX Runtime Discipline

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| D01 | Explicit PRNG keys | no hidden global RNG in traced/reproducible paths | 90 ✅ | 47 explicit-key sites; objectives nulls fixed (B01/PR#22) |
| D02 | `lax.scan` for time | time evolution lowers to one WhileOp | 88 ✅ | 12 scan sites |
| D03 | `vmap` for batch | batch/seed/candidate via vmap | 88 ✅ | 34 vmap sites |
| D04 | JIT opt-in | jit only on numeric hot paths | 88 ✅ | 10 jit sites, RuntimeConfig opt-in |
| D05 | No I/O in JIT | no JSON/plot/print inside jit | 90 🔶 | none observed |
| D06 | Recompilation guard | `N_compile <= 1` | 80 ✅ | 2 opt-in tests trip N_compile=2 (B05) |
| D07 | dtype/x64 policy | float32 default; x64 opt-in before arrays | 90 ✅ | x64 False default, 20 enable_x64 refs |
| D08 | Device discipline | CPU-first; no hard device transfers in hot loop | 88 🔶 | CPU verified |
| D09 | PyTree/FlatNet boundary | no Python objects in traced kernels; FlatNet public | 60 ⛔ | FlatNet not yet surfaced (B08) |
| D10 | Sharding readiness | mesh/candidate/replicated specs available | 78 🔶 | sharding_utils present |

## E. Numerical Safety & Reproducibility

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| E01 | Finite/NaN guards | trap NaN/Inf before serialization | 90 🔶 | `_finite_or_none`, validate_* |
| E02 | Strict JSON | `allow_nan=False` everywhere | 90 🔶 | io.save_json |
| E03 | Host-side RNG seeded | null/surrogate stats reproducible | 85 ✅ | objectives nulls fixed (B01); others seeded |
| E04 | Shape/axis contract | named (B,Z,C,T); no silent axis guess | 90 ✅ | selector slicing verified |
| E05 | Physiological sanity | Vm rest −60…−80 mV, spike +30…+50 mV | TBD | needs range assertion in suite |
| E06 | Golden-value regression | key outputs pinned to golden values | 70 🔶 | partial (dedup test adds one) |
| E07 | Seed reproducibility | same seed → identical full run | 88 🔶 | per-factor, not end-to-end pinned |
| E08 | Conservation proxy diagnostics | proxy conservation metrics finite | 88 🔶 | compute_conservation_proxy_diagnostics |
| E09 | Kernel row-normalization | projection rows sum to 1 (proxy) | 96 ✅ | `_test_kernel_row_normalization` |
| E10 | Trace stability | no recompile drift across calls | 80 ✅ | tied to D06 |

## F. Source / Field / Probe (proxy) Correctness

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| F01 | Proxy not PDE | `field_solver_status="laminar_proxy_no_pde"` | 96 ✅ | enforced; Poisson "future" only |
| F02 | `*_proxy` naming | proxy outputs carry `_proxy` suffix | 92 🔶 | FieldOutput fields |
| F03 | No J_e synthesis | never fabricate current density | 96 🔶 | declared not_applicable |
| F04 | Probe operators (8) | SPK/Vm/source/LFP/CSD/EEG/MEG/EMM finite + tested | 90 🔶 | proxy transforms tested |
| F05 | Leadfield honesty | leadfield status `toy_or_declared_proxy` | 92 🔶 | LinearReadout status |
| F06 | CSD sign convention | declared convention in report | 96 ✅ | `csd_sign_convention` field |
| F07 | Boundary/gauge | declared-metadata-only until solver | 96 ✅ | report fields |
| F08 | Source calibration status | `uncalibrated_*` until calibration | 96 ✅ | report metadata |
| F09 | Field helper single-source | no duplicated validators across modules | 95 ✅ | dedup (PR#23) — diagnostics canonical |
| F10 | Multi-area projection invariants | `validate_projection_invariants` holds multi-area | 88 🔶 | tested partial |

## G. Optimizers & Objectives

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| G01 | SDR-family honesty | AGSDR/GSDR/SDR differentiability declared | 88 🔶 | OptimizerSpec metadata |
| G02 | Optax wrappers | declared surrogate/gradient path | 86 🔶 | optax_adam/sgd specs |
| G03 | Hard-spike non-diff guard | block Optax unless gradient-path-safe | 90 🔶 | `Model.tune()` guard |
| G04 | Objective null reproducibility | nulls accept explicit rng/seed | 95 ✅ | B01/PR#22 (rng + null_seed + factory) |
| G05 | Trainer reports JSON-safe | objective/tune reports strict JSON | 88 🔶 | ObjectiveReport |
| G06 | TuneResult API | `.model`/`.summary`, no tuple-unpack deprecation | 84 🔶 | deprecation still warned |
| G07 | Rate/synchrony targets | declarative multi-group targets | 88 🔶 | rate_targets, rate_synchrony_targets |
| G08 | Candidate/batch eval | vmap over candidates/seeds | 86 🔶 | tied to D03 |
| G09 | Optimizer state pytree | states are valid pytrees | 86 🔶 | AGSDRState etc. |
| G10 | Convergence diagnostics | reports carry convergence evidence | TBD | needs check |

## H. Scientific Truth Gates & Claim Language

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| H01 | truth_safe_unverified | default truth mode on all outputs | 96 ✅ | 229 gate token sites |
| H02 | computational_scaffold | package status declared | 96 ✅ | doctrine + metadata |
| H03 | amplitude claim gate | `physical_amplitude_claim_allowed=False` | 100 ✅ | enforced, never flipped |
| H04 | No EEG/MEG overclaim | proxy wording only | 96 ✅ | scan: no active overclaim |
| H05 | No solver/PDE claims | future-tense only until evidence | 96 ✅ | all Maxwell/Poisson hits future/negated |
| H06 | Claim wording | "simulated/proxy/scaffold/diagnostic" | 96 🔶 | doctrine §13 |
| H07 | Receipts write-once | `save_receipt` refuses overwrite | 96 🔶 | core.py guard |
| H08 | Manuscript claim_level | claim escalation gated by evidence | TBD | hnyxj/rules cross-check |
| H09 | Null tests for mechanism | nulls/ablations before mechanism claims | 85 🔶 | null framework present (reproducible now) |
| H10 | TFNE grammar alignment | source→field→probe→objective→report grammar | 90 🔶 | manuscript_alignment doc |

## I. Notebooks / Tutorials / Docs

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| I01 | Notebooks execute clean | nbconvert run, zero cell errors | 90 ✅ | Etude executes clean |
| I02 | Cell structure | ids present; no adjacent code; ≤ length budget | 80 ✅ | Etude cell #32 = 397 lines (accepted debt) |
| I03 | Tutorial doctrine | `duration_ms>=1000`, `dt=0.1`, float32, seed | 88 🔶 | enforced by audit script |
| I04 | mkdocs --strict | strict build exit 0 | 100 ✅ | exit 0 |
| I05 | Nav completeness | no orphaned pages | 60 ✅ | 22 docs not in nav (docs audit) |
| I06 | No duplicate pages | one canonical page per topic | 55 ✅ | 7 root↔guides dup pairs |
| I07 | API doc coverage | all 146 public names documented | 60 ✅ | index lists all; per-module 48/146 |
| I08 | Public callable docstrings | every public callable has a docstring | 70 ✅ | 11 public callables undocumented |
| I09 | Index summaries | index shows accurate summaries | 90 ✅ | new api/index auto-generated |
| I10 | Asset hashing | tutorial assets SHA-pinned | 86 🔶 | sha256_file/text + audit |

## J. Optional Ecosystem Bridges & Performance

| ID | Factor | Ideal | Score | Evidence / notes |
|---|---|---|---|---|
| J01 | Lazy optional deps | no optional backend imported on core import | 90 ✅ | require_jaxley/require_optax lazy |
| J02 | Jaxley bridge guarded | optional, round-trip metadata safe | 88 ✅ | audit "Jaxley bridge GREEN" |
| J03 | PyNWB export | optional lazy writer, round-trip, units/status | 30 ⛔ | planned (B09) |
| J04 | Equinox pytree backend | filtered transforms for Config→Net→FlatNet | 20 ⛔ | design-only |
| J05 | Diffrax solver backend | future HH/ODE emitters (not Izhikevich default) | 15 ⛔ | design-only |
| J06 | Lineax field solver | future `K Φ = Q` solve experiments | 15 ⛔ | design-only |
| J07 | Benchmark/profiling receipts | committed CPU baselines (count×contacts×seeds) | 30 🔶 | no committed receipts (audit F33) |
| J08 | Sparse > dense at large W | sparse recurrent path wins at scale | 78 🔶 | EdgeList/segment_sum present |
| J09 | Connectivity compiler | deterministic sparse edge arrays + SHA artifact | 40 ⛔ | planned (B07) |
| J10 | Colab/install readiness | one-cell Colab bootstrap verified | TBD | unmeasured |

---

## Rollup (2026-06-04 @ 33f99db)

Per-category means are **indicative** (mix of measured ✅, estimated 🔶, gated ⛔,
and `TBD` excluded from the mean). Recompute each cycle.

| Category | Indicative mean | Notes |
|---|---|---|
| A. Release engineering | ~96 (8 scored) | A08/A09 TBD |
| B. API & surface hygiene | ~88 | `_KNOWN_METRICS`, sys leak, FlatNet |
| C. Config/model/sim | ~93 (8 scored) | C06/C09 TBD |
| D. JAX runtime | ~84 | FlatNet (D09), recompile (D06) drag |
| E. Numerical safety | ~85 (9 scored) | E05 TBD; golden coverage thin |
| F. Source/field/probe | ~94 | strong; dedup landed |
| G. Optimizers/objectives | ~87 (9 scored) | G10 TBD |
| H. Truth gates | ~94 (9 scored) | H08 TBD |
| I. Notebooks/docs | ~74 | nav/dup/coverage gaps (docs audit) |
| J. Bridges/perf | ~39 (9 scored) | mostly planned/gated |
| **Overall (measured+estimated)** | **~83** | J & I are the largest headroom |

## Maintenance protocol

- **Cadence:** re-score on each release candidate and after any factor-affecting PR.
- **Discipline:** a score change requires a fresh receipt in the Evidence column. No receipt → revert to `TBD`.
- **Linkage:** factors map to the loop-context backlog (`internal_docs/loop_context/05_BACKLOG.md`) and the docs audit (`internal_docs/docs_audit_v0330.md`). When a backlog item ships, bump its factor here.
- **Growth:** factors may be added (keep IDs stable; append, don't renumber). Target stays ~100 across 10 categories; split a category only if it exceeds ~14 factors.
- **Truth rule:** never mark a scientific factor (H*, E05, F*) green without a Truth-plane receipt.
