# Authorized facts — jaxfne

Small set of **stable, human-authorized** project facts. **Not evidence.**
Repository code, tests, and receipts remain evidence and may contradict a fact.

**Agents:** read, use, test, and challenge these facts. **Do not add, remove, or
change any fact without explicit human authorization.** If evidence contradicts a
fact, flag the contradiction for human review — do not silently rewrite this file.

Form: two rows per fact. Row 1 is `keywords | scope-validation` (what the fact
covers and what checks it); row 2 is the fact. Authorized by the human in the
2026-09-29 quiz, which replaced the earlier bullet list.

---

## Plasticity and state

| Keywords \| Scope-validation |
|---|
| **HDP, plasticity abstraction, named mechanism** \| plasticity rules; rule-registration tests |
| HDP (Hidden-state Dependent Plasticity) is the general plasticity abstraction, `dΘ/dt = P(X, H, B, Θ)`, each rule using only the state it needs. A named mechanism (STDP, STP, homeostasis) is one choice of state, law and target, and needs no separate engine. |
| **H, RBS, RBD, H ≠ HDP** \| state container and state dynamics; continuation tests |
| `H` is a finite dependency-state container, not intrinsically homeostasis and not one scalar. RBD is `Ḣ = F_H`; plasticity is not required (`Ẇ = 0` is valid). HDP is `Ẇ = F_W(H, …)`. H ≠ HDP. |
| **rule grammar, declared targets, refusal** \| generic finite-state rule registration; refusal tests |
| Generic rules are finite-state over declared state layouts, event/history inputs, RNG semantics and a finite declared target set. Structural change, exogenous ports and undeclared targets are outside the grammar and refused. |
| **continuation, carrier, complete state** \| `X`, `H`, `W`/`Θ`, delay ring `B`, `K`; continuation bit-exactness tests |
| Continuation preserves the complete declared dynamic state bit-exactly (membrane, recovery, spike history, synaptic `X`; `H`; `W`/`Θ`; delay ring `B`; controller coordinates `K` where declared; rule-owned auxiliary state); no hidden state lives outside the carrier. |
| **delay ring, events, dt grid** \| `B`; continuation and delay tests |
| Delays live in a declared delay ring `B` carried with the state. Event times are in ms on the declared `dt` grid; no sub-step timing is implied. |

## Semantics and evidence

| Keywords \| Scope-validation |
|---|
| **semantics layers, configured, realized, executed, effective** \| lifecycle stages; consumption gates |
| configured ≠ realized ≠ executed ≠ effective are distinct lifecycle stages. relative ≠ calibrated: internal coordinates are not physical calibration. proxy ≠ physical measurement: a projection or readout is not a solved field or calibrated instrument output unless explicitly validated. |
| **lifecycle, perturbation, 0/None/boundary** \| every structural config parameter; perturbation test |
| A capability passes only when a realized change, or a refusal, is observed at 0, None and boundary values of each structural parameter; a stored parameter, a count or a neutral value alone is not evidence. |
| **refusal, construct, dropped declaration** \| all routes; construct-time refusal tests |
| Declared fields a route would ignore are refused at construct with an actionable message, never dropped silently. |
| **labels, descriptive, no behavior** \| `connectivity` route labels, `network(kind=)`, probe `modes`; documented as labels |
| A descriptive label nothing reads stays a label and never implies behavior; only structural parameters must be consumed or refused. |
| **unwired code, owner decision** \| `units.py`, `pynwb_compat`; wiring audit |
| Code nothing calls is a defect until an owner decision wires or removes it; it is never accepted silently as a supported feature. |
| **geometry, radius_mm, height_mm, laminar route** \| `build_laminar_column`; geometry-consumption tests |
| A geometry declaration the route does not read is applied or refused, never accepted and ignored. |
| **drive, plain route, P-018** \| `drive()`; drive-realization tests |
| A declared drive is wired and realized on the plain route; a drive the route cannot apply is refused, never dropped. |
| **probe kernels, refusal, neutral values** \| `Configuration.probe`; kernel refusal tests |
| A probe kernel refuses `position`, `reference` or `filter_spec` it does not apply; the neutral values (`"none"`) are accepted. |
| **field regime, solved_poisson, proxy** \| field regimes; refusal and admissibility tests |
| Only declared field regimes run. `solved_poisson` and other future regimes are refused until implemented and validated; the proxy is the shipped path. |
| **readout, proxy, relative units** \| LFP/CSD/EEG probes; readout tests |
| Field readouts are relative-unit proxies from channel currents. No calibrated or solved-field claim is made. |
| **calibration, physical_amplitude_calibrated, absolute units** \| `validation.py` flag; calibration-boundary tests |
| `physical_amplitude_calibrated` is a declared-off future state; absolute units arise only through an explicit calibration transform and are never escalated implicitly. |
| **stability language, Floquet, margin** \| HDP, homeostasis, W3 analyses; docs and docstrings |
| "Finite" or "bounded" is claimed where tested; "stable" is claimed only with a Floquet or margin result for that regime. |
| **evidence, test quality, vacuous PASS** \| all tests and gates |
| A test is evidence when it can fail on the defect it guards (shown pre-fix or by mutation), its fixture builds the case it is named after, and gates are read per check. |
| **provenance, origin, constructed vs requested** \| origin/provenance queries; inspection tests |
| Each declared parameter and realized array traces to its declaration and route; inspection reports what was constructed, not what was requested. |
| **errors, refusals, warnings** \| construct and simulate; refusal tests |
| A refusal names the offending field, the value and the fix. No bare `except` or silent pass on scientific paths; warnings are for recoverable, non-scientific conditions only. |

## Representation, equivalence and structure

| Keywords \| Scope-validation |
|---|
| **compact storage, derivable, equivalence** \| compaction and tensor layout; equivalence tests |
| A compact form replaces the authoritative one only when derivable from authoritative metadata and equivalent under the declared criterion; correlations do not authorize compaction rules (a sign→τ map applies only where explicitly qualified). Scientific semantics outrank storage. |
| **exactness, bounds, predeclared** \| identity, continuation, parity comparisons |
| Exactness is required where declared (identity, continuation). Elsewhere divergence is judged against predeclared, observable-specific bounds, never ad hoc. |
| **REP-03, sparse-direct, dense** \| `_SPARSE_DIRECT_N`; sparse ≡ dense equivalence test |
| `_SPARSE_DIRECT_N` is not lowered without bit-exact sparse-direct ≡ dense evidence. |
| **jit, eager, parity** \| drive-swept bounds (P-019); parity tests |
| jit and eager runs agree within a declared, tested bound per drive regime, not bit for bit. |
| **population, cell-type fractions, rounding** \| plain route; label bit-identity tests |
| Cell-type counts are `n·frac` rounded to integers. Fractions within 1e-9 of unit mass are not renormalized, because dividing by 1 ± ulp can move a count across a .5 boundary and change existing labels. |
| **connectivity, identity, ownership** \| construct and compile; refusal and equivalence tests |
| Compilation and optimization preserve topology, signs, receptor and mechanism identity, geometry, locality and parameter ownership; route drops of declared structure are refused. |
| **sign, receptor, inhibition** \| per-projection declarations; compact-storage derivation tests |
| Sign and receptor identity are declared per projection and preserved through compilation; inhibition is a declared receptor, not inferred from a label. |
| **emitter, family, unsupported method** \| emitter families; refusal tests |
| The emitter is a declared choice; a method not generalized to a family refuses with an actionable error rather than failing obscurely. |
| **grammars, scientific, execution, optional components** \| `Emitter → … → Manifest`; `CircuitSpec → … → Signals`; public-surface tests |
| Two invariant grammars: scientific (`Emitter → Source → Field → Probe → Objective → Optimizer → Manifest`) and execution (`CircuitSpec → construct → Model → simulate → Signals`). Paradigm, Objective, training, visualization and export are optional downstream components, not stages of either. |
| **entrances, single lowering** \| Configuration, NeuronalTensor, JDNA, agent API; equivalence tests across entrances |
| Every entrance lowers to the same construct → simulate path (`CircuitSpec → construct → Model → simulate → Signals`); no entrance has its own semantics. |
| **TFNE, containment, composition** \| doctrine docs; `tfne_containment_architecture.md` |
| TFNE is a containment and composition model for neural models of different resolution, not one prescribed equation. Nested biological semantics are preserved while tensors may be flattened for compute. |

## Runs, numerics and optimization

| Keywords \| Scope-validation |
|---|
| **dt, units, ms** \| all time-stepped code; unit-named parameters (`dt_ms`) |
| Time and `dt` are in milliseconds, named with units, declared in the config and never inferred; a unit change is stated at the change site. |
| **seed, PRNG, determinism** \| all stochastic paths; seed-changes-output tests |
| Explicit keys only. Same seed, config and backend give the same output; cross-backend agreement is judged by declared bounds. |
| **paradigm, stimulus, P-017** \| `Paradigm`, `simulate`; paradigm tests |
| Only events carrying a stimulus inject, for their own duration at the declared amplitude and targets; marker events are silent; a multi-condition `Paradigm` is refused. |
| **optimizer, owned parameters, manifest** \| `jaxfne.optim`; bounds and manifest tests |
| An optimizer changes only declared parameters it owns, within bounds, keeps identity, topology and signs, and records the result in a manifest. |
| **manifest, digest, seeds, versions** \| run manifests; manifest-quoted Methods |
| A run manifest records the config digest, seeds, versions (jaxfne, JAX, libraries), units, calibration level and tolerances, quoted from generated artifacts and not typed. |
| **optimization, profile, diff** \| `artifacts/perf/`; output diff against a frozen copy |
| A change is justified by a measured bottleneck and an output diff against a frozen copy, and leaves semantics, API and numerics unchanged (minimum complexity subject to complete required semantics). |
| **performance, T_compute, M_compute, E_semantic** \| `artifacts/perf/`; measured artifact per AT |
| A performance claim needs a traceable measured artifact (`T_compute`, `M_compute` per AT). Programme deltas against v0.4.25 are measured, with `E_semantic = 0`. |
| **precision, dtype, declared** \| config `dtype`; precision-comparison tests |
| Precision is declared in the config, never inferred. Accumulation and comparisons state their precision; a float32 result is not compared to float64 without a declared bound. |
| **backends, CPU fallback, observable** \| runtime backend selection; manifest records the backend |
| A CPU fallback exists and which backend ran is recorded. A requested backend that is unavailable is refused or reported, never silently substituted. |
| **data I/O, imported vs synthetic, no imputation** \| NWB/SONATA/CSV bridges; missing-input refusal tests |
| Imported and synthetic data are labelled as such. Missing input raises and is never imputed; synthetic fixtures never count as empirical verification; rate, units and frame come from the data. |
| **objectives, fit claims, identifiability** \| `Objective`, `tune`; selection-status and identifiability checks |
| A fit claims only what its declared objective measures, with held-out or selection status stated. Identifiability of fitted parameters is checked before interpretation. |

## Atlas, claims and publication

| Keywords \| Scope-validation |
|---|
| **Atlas, scale ladder, reduction matrix** \| AT-01…AT-10, AT-10-N20; regeneration from manifests |
| The Atlas shows the same model language from 1 neuron to 20 areas and records what survives each simplification; failures stay in the matrix. |
| **reduction, survival matrix, tolerance** \| ATLAS item 7; `run_reduction` |
| A simplification states which observations survive within a predeclared tolerance and which do not; failures stay in the matrix. |
| **AT-10-N20, synthetic hierarchy, caveats** \| `artifacts/atlas/at10_n20_055.py`; Atlas caveat rows |
| AT-10-N20 is a synthetic-hierarchy simulation, not a biological fit. Its caveats stay with it: selection status, control-window dependence, no-stimulus null. |
| **claim ledger, V(claim), evidence** \| Atlas and manuscript; `atlas_coverage.json` mechanical check |
| Each claim has a ledger row with V(claim) and a PASS evidence path; package-capability claims stay apart from scientific-result claims; negative and failed results are reported as such. |
| **frozen receipt, re-freeze, no re-tuning** \| frozen protocols and `results_*`; human sign-off |
| A frozen receipt stays as made. A re-freeze is a new receipt that names the cause and needs human sign-off; nothing is re-tuned after seeing its outcome. |
| **frozen Figures 1–7, Atlas manuscript, write-once** \| `artifacts/publication/`; `frozen_manifest.json` |
| The Figure 1–7 snapshot stays untouched. The Atlas manuscript is new, write-once per figure, and consumes generated data only. |
| **seed, split, out of sample, selection status** \| protocols and results; seed-changes-output check |
| A split is called out of sample only after the seed is shown to change the output; each result records its selection status. |
| **public docs, positive description, traceable numbers** \| `docs/`, README; docs style audit |
| Public pages are compact positive mathematical descriptions: no agent governance, no status headers, no timing or size claim without a traceable artifact. |
| **tutorial track, visuals, gh-pages** \| `docs/tutorials`, `scripts/generate_docs_visuals.py`; docs audits |
| One tutorial track with visuals generated by the same code (still plus interactive). Docs stay in the repo; gh-pages publishing is a separate human decision. |
| **notebooks, declared vs executed, fast gate** \| release notebooks; notebook-execution check |
| A notebook declares and plots only what runs, is executed by a fast gate, and its figures come from the same code path. |
| **figures, generated, themes, proxy label** \| `jaxfne.vis`; theme-invariance tests |
| Figures are generated from data by code with no hand edits, units and axes labelled. Dark and light themes change presentation only, and a proxy readout is labelled as one. |

## Programme, authority and process

| Keywords \| Scope-validation |
|---|
| **API compatibility, aliases, deprecation** \| public surface contract; surface tests |
| Public and runtime compatibility is preserved unless a release explicitly changes it. Aliases are retained, deprecations name their replacement, and a break ships only in a release that says so. |
| **bit-identity, canonical outputs, seal receipt** \| frozen baseline; release-tier baseline check |
| Existing canonical outputs change only through an authorized repair, each listed in the seal receipt with regenerated hashes. |
| **push, dev, main, tags, force-push** \| git routine; branch and upstream check before push |
| A validated state is pushed to `dev`. `main`, tags, releases, force-push and history rewrites need explicit instruction. |
| **green, suite, gates, CI** \| push gate; suite summary line and CI run count |
| Push only after the suite summary shows 0 failed (or each failure re-run alone as a known flake) and no CI run on the branch is in progress. A gate passing says nothing about the suite. |
| **delegation, verify, one writer** \| worker packets; diff review before integration |
| Delegated output is evidence to verify, not a result. One writer per worktree; every diff is reviewed before integration. |
| **authority, human-owned facts, agents** \| `fact_stack.md`, `todo_stack.md`, `AGENTS.md` |
| The human authorizes facts and goals; agents read, test and challenge them and flag contradictions; state and todos are agent-maintained. The fact stack is not evidence. |
| **stacks, todo, goal, fact, problem** \| `artifacts/*_stack.md`; archive per release cycle |
| The todo stack holds remaining work only and finished items are deleted. Goal and fact stacks are human-owned. The problem stack targets empty: solve and delete. Sealed stacks are archived byte for byte. |
| **memory, lessons, not evidence** \| memory files and lessons; scope and evidence per lesson |
| Memory holds verified reusable working lessons `{trigger, cause, repair, evidence, scope}`. It is never current-state evidence and never duplicates facts or code. |
| **repo skills, canonical, mirrors, style pointers** \| `artifacts/skills/jaxfne-*`; `sync_skills.py --check` |
| Repo skills are task-shaped, benchmarked routers. The canonical copy is in `artifacts/skills`; mirrors are generated and hash-checked. Style rules point at the canonical style skills and are not restated. |
| **vocabulary, allowlist, --check** \| `scripts/audit_vocabulary.py --check`; gate |
| Each term carries one meaning; an allowlist bounds words such as "contract". The audit runs with `--check` as a gate; bare mode exits 0 and proves nothing. |
| **naming, package, acronyms, one meaning** \| code and docs; vocabulary audit |
| The package is `jaxfne`. Acronyms are those defined in this stack; a term is defined once and reused with one meaning. |
| **scripts, thin entry points, manifests** \| `scripts/`; results in `artifacts/` |
| Scripts are thin entry points and generators over the package; results land in `artifacts/` with manifests. No science logic lives only in a script. |
| **CI, POSIX checks, clean checkout** \| GitHub Actions by SHA; executable bit and line-ending checks |
| CI covers what a local run does not: POSIX-only checks (executable bits, line endings) and the clean-checkout suite. Local green ≠ CI green: check CI by SHA after every push. |
| **line endings, bytes, scripted edits** \| all tracked text; eol guard, hash-verified restores |
| Tracked text is LF; an edit preserves each file's bytes, a restore is verified by hash, and a scripted edit asserts its anchor count before and after. |
| **agent surface, pinned task set, corrections** \| agent API and catalog; `tests/test_agent_bench_055.py` |
| The agent API has the same semantics as the human API. The frozen agent task set stays pinned; corrections are listed explicitly. |
| **releases, authorities file, seal receipt** \| `current_release_authorities.json`; archived stack per cycle |
| The release authorities file is the source of versions. Versions and SHAs are not stored in persistent rules. A cycle closes with an archived stack and a seal receipt. |
