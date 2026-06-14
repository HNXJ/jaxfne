# jaxfne Multi-Factor Scoreboard Glossary Grammar

Version: `v0.1-draft`
Scope: `jaxfne` release, architecture, JAX runtime, scientific-status, docs, tutorial, and ecosystem readiness scoring.
Status: `truth_safe_unverified` · Claim level: `computational_scaffold` · Physical amplitude allowed: `false`

> **Relationship to `jaxfne_glossary.md`.** This file is the **grammar / ontology**
> layer: it fixes how a score is *defined, evidenced, and aggregated*. The sibling
> `internal_docs/jaxfne_glossary.md` is the **instance** — one concrete 100-factor
> scoreboard. Author new scoreboards against this grammar; keep instances under
> `internal_docs/scoreboards/`. Last reconciled: 2026-06-05 @ `main == dev == 263c180`
> (v0.3.28 circuit declarations merged; v0.3.30 connectivity compiler in PR).
>
> **⚠ Legacy-instance compatibility (READ before computing §18 metrics).** This
> grammar is the **canonical, going-forward standard**. The existing
> `jaxfne_glossary.md` was authored *before* this grammar and is therefore a
> **legacy `v0` instance**: its factor-ID assignment and its rubric differ from
> this document and are **NOT factor-ID-comparable** with grammar-conformant
> scoreboards. Concretely (legacy → grammar, non-exhaustive):
>
> | ID | legacy `jaxfne_glossary.md` | this grammar |
> |---|---|---|
> | A02 | Version metadata comparison | main/dev comparison |
> | A03 | Tag/release identity | agy isolation |
> | B01 | `__all__` curated | Canonical import |
> | D01 | Explicit PRNG keys | JAX array math |
>
> Rules: (1) the **§18 consistency metrics (`AC`/`RC`/`RAC`) and scope catalogue
> temperature compare ONLY scoreboards authored under this grammar (v0.1+)** —
> never the legacy instance factor-by-factor. (2) The legacy instance is
> **frozen as a v0 snapshot**; it is re-mapped to grammar IDs the next time it is
> re-scored (target: the first measured scoreboard after the v0.3.30 compiler
> lands), at which point it becomes the first grammar-conformant baseline. Until
> then, treat legacy and grammar scores as separate series.

The transfer from the multi-LLM ontology technical report is the **ontology discipline**:
factor definitions are fixed before scoring, undefined values stay undefined (never
silently `0`), and disagreement/dispersion is *measured*, not hidden.

---

## 0. Purpose

A standard, ontology-constrained grammar for scoring `jaxfne` across many factors,
so every audit, release review, worker report, and scope catalogue decision is comparable
across time. Each factor has a stable ID, name, definition, ideal 100/100 state,
required evidence type, score rubric, blockers, dependencies, score status, and
truth/status gates. This bans vague scoring ("good", "mostly ready"); every score
maps to evidence.

---

## 1. Score tensor grammar

### 1.1 Score element

```text
J(r, a, c, f) ∈ [0, 100] ∪ {∅}
```

| Symbol | Meaning |
|---|---|
| `J` | jaxfne scoreboard value |
| `r` | repo state, release, branch, SHA, or run |
| `a` | evaluator / agent / auditor / model |
| `c` | context |
| `f` | factor |
| `∅` | undefined / not applicable / not enough evidence |

### 1.2 Context set

```text
c ∈ { release, api, runtime, jax, numerical, source_field_probe,
      objective_optimizer, docs_tutorials, ecosystem, technical report }
```

### 1.3 Factor set

```text
F = { A01..A10, B01..B10, ..., J01..J10 }   # 100 factors, 10 categories
```

### 1.4 Category sets

```text
A = Release / provenance        F = Source / field / probe
B = Public API / surface        G = Objectives / optimizers
C = Configuration / model / sim  H = Truth gates / claim safety
D = JAX runtime                 I = Notebooks / docs / artifacts
E = Numerical safety            J = Ecosystem / performance / future bridges
```

### 1.5 Category score

```text
Score(category_k) = mean_non_null( J(r, a, c, f ∈ category_k) )
```

### 1.6 Overall score

```text
Score(overall) = weighted_mean_non_null( category_scores )
```

Default category weights:

| Category | Weight |
|---|---:|
| A Release / provenance | 1.0 |
| B Public API / surface | 1.0 |
| C Configuration / model / simulation | 1.2 |
| D JAX runtime | 1.2 |
| E Numerical safety | 1.1 |
| F Source / field / probe | 1.2 |
| G Objectives / optimizers | 1.0 |
| H Truth gates / claim safety | 1.3 |
| I Notebooks / docs / artifacts | 1.0 |
| J Ecosystem / performance / future bridges | 0.8 |

---

## 2. Score status grammar

| Status | Meaning |
|---|---|
| `measured` | backed by command output, tests, source line, or generated artifact |
| `estimated` | based on code inspection, not fully measured |
| `gated` | cannot be scored until a prerequisite lands |
| `TBD-not-implemented` | planned but not implemented |
| `blocked` | a known blocker prevents progress |
| `not_applicable` | factor does not apply to current scope |

### 2.1 Score confidence

| Confidence | Meaning |
|---|---|
| `high` | exact command/test/source evidence |
| `medium` | direct code inspection, no command receipt |
| `low` | inferred from architecture only |

---

## 3. Score rubric

The table below gives **10-point anchors**; the score space is the full integer
range `[0, 100]`. Anchors are reference descriptions, **not** the only legal
values — a score *between* anchors (e.g. `75`) is valid and means "between the
two nearest anchors" (here: better than a functional prototype, not yet
fully usable-with-debt). Use this scale unless a factor defines stricter thresholds.

> **Legacy note.** The legacy `jaxfne_glossary.md` instance used *range bands*
> (90–100, 75–89, 50–74, 25–49, 0–24) rather than these anchors. The two are
> close but not identical at the boundaries, so a legacy band score is not
> silently equal to a grammar anchor. Re-grade the legacy instance against this
> anchor scale when it is re-mapped (see the legacy-compatibility note above).

| Score | Meaning |
|---:|---|
| 100 | complete, tested, documented, stable, no known blocker |
| 90 | complete and tested; minor polish or docs gap |
| 80 | usable, but incomplete coverage or some known debt |
| 70 | functional prototype; not release-grade |
| 60 | partial implementation; important gaps |
| 50 | concept present but weakly integrated |
| 40 | skeleton/stub exists |
| 30 | planned but little implementation |
| 20 | design notes only |
| 10 | vague intent |
| 0 | absent or failing |
| ∅ | not applicable or no evidence |

---

## 4. Evidence grammar

Each score must cite at least one evidence token.

```yaml
evidence:
  type: command | test | source | artifact | docs | notebook | release | audit
  ref: string
  result: string
  sha: string | null
```

Examples:

```yaml
evidence:
  type: command
  ref: "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line"
  result: "2008 passed, 66 skipped, 4 xfailed"
  sha: "263c180"
```

```yaml
evidence:
  type: source
  ref: "jaxfne/core.py::Configuration.connections"
  result: "declaration-only connection grammar"
  sha: "263c180"
```

---

## 5. Truth gate grammar

Every scoreboard must carry these package-level gates.

```yaml
truth_gates:
  truth_status: truth_safe_unverified
  claim_level: computational_scaffold
  field_solver_status: laminar_proxy_no_pde
  physical_amplitude_claim_allowed: false
  biological_mechanism_claim_allowed: false
  calibrated_sensor_claim_allowed: false
```

### 5.1 Forbidden escalations

Not scorable as supported unless solver/calibration/evidence exists:

```text
real EEG · real MEG · calibrated amplitude · biological metabolism ·
mechanism proof · solved Maxwell dynamics · solved Poisson dynamics ·
validated physical forward model · empirical biological validation
```

Use instead:

```text
EEG-like proxy · MEG-like proxy · relative proxy readout · simulated scaffold ·
declared source-field assumption · computational target · finite/status-validated artifact
```

---

## 6. Factor record schema

Every factor uses this markdown block:

```markdown
### <ID> — <Factor name>
- Category:
- Context:
- Definition:
- Ideal 100/100:
- Minimum acceptable release state:
- Evidence required:
- Score:
- Score status:
- Confidence:
- Current evidence:
- Blockers:
- Dependencies:
- Stop rules:
- Improvement path:
```

---

## 7. Category A — Release / provenance

- **A01 Repo state frozen** — reports include branch, SHA, dirty state, origin/main, origin/dev, origin/agy. Ideal: every report starts with exact repo state, no mutation before freeze. Evidence: `git status`, branch, SHA, origin refs.
- **A02 main/dev comparison** — `main`/`dev` match unless staged PR flow. Ideal: divergence `0/0`. Evidence: `git rev-list --left-right --count origin/main...origin/dev`.
- **A03 agy isolation** — `agy` untouched unless authorized. Evidence: before/after `origin/agy` SHA.
- **A04 tag/release provenance** — tags point to intended SHA, verified by peeled commit. Evidence: `git rev-parse vX.Y.Z^{commit}`.
- **A05 version metadata comparison** — package/pyproject/docs/tag versions agree. Evidence: import smoke, pyproject, generated version.
- **A06 dist hygiene** — `dist/` has exactly wheel + sdist for current version. Evidence: `python -m build`, `twine check`, dist listing.
- **A07 CI terminal state** — merge only after CI terminal green. Evidence: PR number, run ID, job results.
- **A08 release mutation gating** — tag/upload/merge/publish are separate authorized gates. Evidence: release report.
- **A09 changelog/release-note comparison** — notes reflect merged code, no overclaim. Evidence: changelog diff.
- **A10 rollback/recovery path** — reports identify prior safe SHA/tag. Evidence: release receipt.

## 8. Category B — Public API / surface

- **B01 Canonical import** — all docs/tutorials use `import jaxfne as jtfne`. Evidence: grep.
- **B02 Root public API stability** — `jaxfne.__all__` intentional, documented, tested. Evidence: root API audit.
- **B03 Loud unsupported paths** — unsupported features fail loudly. Evidence: unsupported family/mode tests.
- **B04 Backward-compatible wrappers** — moved functions keep wrappers or documented migration. Evidence: import tests.
- **B05 No namespace leaks** — `dir(jtfne)` only intended names. Evidence: public-surface diff.
- **B06 Stub honesty** — stubs marked and fail loudly. Evidence: `NotImplementedError` tests.
- **B07 API signatures stable** — public signatures documented + tested. Evidence: generated API index.
- **B08 Alias discipline** — `vm`/`spk`/`V_m`/`spikes` stable + tested. Evidence: `Signals.get` tests.
- **B09 Selector API stability** — strict AND semantics, fail on missing metadata. Evidence: selector tests.
- **B10 Public API docs coverage** — every public symbol has a table row or docstring. Evidence: docs audit.

## 9. Category C — Configuration / model / simulation

- **C01 Chainable Configuration grammar** — compose without mutation (frozen dataclass). Evidence: immutability tests.
- **C02 Runtime configuration** — duration/dt/seed/dtype explicit + validated, JSON-safe. Evidence: runtime tests.
- **C03 Circuit declarations** — cell params/mechanisms/connections/lesions/trainables/objective outputs in metadata; JSON-safe; no runtime side effects. Evidence: circuit-ownership tests.
- **C04 Construct compatibility** — `construct(cfg)` works for supported declarations or fails loudly; no silent application. Evidence: construct smoke.
- **C05 Model identity table** — `Model.neuron_table()` with area/layer/cell_type/id, stable + tested. Evidence: selector tests.
- **C06 Model selection wrapper** — `Model.select(...)` strict semantics + empty-match handling. Evidence: selector tests.
- **C07 Signals contract** — `Signals` carries V_m/spikes/sources/field/metadata + `get`. Evidence: signal tests.
- **C08 Simulation finite outputs** — supported paths finite, non-astronomical. Evidence: smoke tests.
- **C09 Unsupported emitter family guard** — explicit unsupported family raises; never silently runs another. Evidence: emitter-family validation tests.
- **C10 No tutorial-local simulator logic** — tutorials call package APIs, not local integrators. Evidence: notebook audit.

## 10. Category D — JAX runtime

- **D01 JAX array math** — numerical kernels use `jax.numpy` in traced/hot paths. Evidence: code audit.
- **D02 Explicit PRNG discipline** — stochastic paths take explicit keys/seeds; no hidden global RNG. Evidence: PRNG tests.
- **D03 `lax.scan` for time** — time stepping uses scan; no Python loop in hot path. Evidence: source audit.
- **D04 `vmap` for batch/candidate axes** — batch/seed axes via vmap; shape + determinism tested. Evidence: batch tests.
- **D05 JIT opt-in discipline** — JIT wraps hot numerical fns only; no plotting/JSON/I/O inside JIT. Evidence: source audit.
- **D06 Recompilation guard** — `N_compile <= 1` for stable shapes. Evidence: recompilation tests.
- **D07 Static argument boundary** — Python spec objects don't enter traced kernels. Evidence: JIT tests.
- **D08 PyTree readiness** — flatten/unflatten + tree tests pass. Evidence: FlatNet tests.
- **D09 Device neutrality** — CPU-first; GPU/TPU guarded. Evidence: runtime report.
- **D10 dtype/x64 policy** — float32 default, x64 opt-in, dtype in metadata. Evidence: dtype tests.

## 11. Category E — Numerical safety

- **E01 finite arrays** — NaN/Inf rejected before export. Evidence: finite tests.
- **E02 JSON strictness** — `json.dumps(..., allow_nan=False)`. Evidence: JSON tests.
- **E03 shape contracts** — explicit shapes, no silent axis guessing. Evidence: shape tests.
- **E04 neuron-axis safety** — selector slicing only on declared neuron axis. Evidence: `Signals.get` tests.
- **E05 edge-array safety** — connection compiler emits finite aligned sparse edge arrays. Evidence: connectivity tests.
- **E06 source conservation checks** — detect double count / conservation defects. Evidence: conservation tests.
- **E07 kernel normalization** — field/proxy kernels report row normalization + finite summaries. Evidence: field diagnostics tests.
- **E08 artifact hash validation** — external arrays require path/array/sha256. Evidence: artifact tests.
- **E09 matrix shape validation** — matrix modes validate shape; wrong shape fails loudly. Evidence: matrix tests.
- **E10 numeric range/status report** — outputs report scale/status, not physical amplitude unless calibrated. Evidence: probe reports.

## 12. Category F — Source / field / probe

- **F01 emitter-source boundary** — emitter outputs → declared source proxies with calibration status; no native current treated as amperes. Evidence: source calibration metadata.
- **F02 source bookkeeping modes** — explicit, no double counting. Evidence: source tests.
- **F03 field solver boundary** — proxy field distinct from solved field; `field_solver_status` explicit. Evidence: field reports.
- **F04 CSD sign convention** — declared in metadata. Evidence: CSD tests/docs.
- **F05 LFP-like readout** — proxy/status-labeled; shapes/status tested. Evidence: probe tests.
- **F06 CSD-like readout** — finite, status-labeled. Evidence: CSD tests.
- **F07 EEG-like readout** — proxy, not sensor-level EEG. Evidence: wording + probe report.
- **F08 MEG-like readout** — proxy, not calibrated MEG. Evidence: wording + probe report.
- **F09 EMM proxy** — signaling/energy-like proxy, not biological metabolism. Evidence: docs grep/report.
- **F10 solver-readiness ladder** — P0..P5 ladder explicit; current level + missing evidence reported. Evidence: solver-readiness docs.

## 13. Category G — Objectives / optimizers

- **G01 Objective grammar** — separate signal/metric/score/null/gates/calibration; JSON-safe. Evidence: objective tests.
- **G02 Null distribution reproducibility** — null generators take explicit seed/rng; same seed reproduces. Evidence: null reproducibility tests.
- **G03 Null-normalized score safety** — null-normalized labels only when a null distribution exists. Evidence: objective report tests.
- **G04 Spectrolaminar objective** — alpha/beta + gamma laminar profiles scored as computational motifs; finite + null tests. Evidence: spectrolaminar tests.
- **G05 Rate/synchrony metrics** — stable + finite over silence/synchrony/random. Evidence: metrics tests.
- **G06 Optimizer report structure** — returns model/result/summary, JSON-safe. Evidence: optimizer tests.
- **G07 AGSDR status honesty** — wording matches algorithm; no adaptive/self-supervised overclaim. Evidence: docs/source tests.
- **G08 Trainable declarations** — declared, not silently optimized (`declared_not_optimized`). Evidence: config tests.
- **G09 Objective output declarations** — schema declared + JSON-safe. Evidence: config tests.
- **G10 Differentiability boundary** — nondifferentiable spike vs surrogate paths separated. Evidence: optimizer docs/tests.

## 14. Category H — Truth gates / claim safety

- **H01 truth status present** — every artifact carries `truth_safe_unverified`. Evidence: manifest tests.
- **H02 computational scaffold status** — outputs labeled scaffold; no empirical-validation overclaim. Evidence: docs grep.
- **H03 physical amplitude gate** — false unless calibration evidence; all proxy readouts false. Evidence: probe reports.
- **H04 biological mechanism gate** — no mechanism-proof wording. Evidence: wording scan.
- **H05 EEG/MEG wording safety** — `-like`/proxy unless forward model exists. Evidence: docs grep.
- **H06 calibration label coverage** — every source/readout has calibration status. Evidence: source/probe reports.
- **H07 field solver status** — every field output states proxy/solver state. Evidence: field tests.
- **H08 technical report-package comparison** — claims map to artifact evidence. Evidence: technical report audit.
- **H09 anti-guess constraints** — agents can't invent APIs/results; verify-before-claim gates. Evidence: worker reports.
- **H10 failure provenance** — failures classified before tests/code change. Evidence: PR reports.

## 15. Category I — Notebooks / docs / artifacts

- **I01 public notebook execution** — all public notebooks execute without cell error. Evidence: nbconvert receipts.
- **I02 notebook structure** — IDs, reasonable cells, separators, no hidden local engines. Evidence: notebook audit.
- **I03 tutorial artifact completeness** — notebook/markdown/manifest/figures/validation present. Evidence: tutorial audit.
- **I04 PNG figure stability** — core figures PNG; Plotly optional. Evidence: file audit.
- **I05 API index completeness** — index lists every public name. Evidence: docs audit.
- **I06 per-module API pages** — role/public table/example/status note each. Evidence: mkdocs.
- **I07 orphan/duplicate docs** — nav has no orphaned duplicates unless internal. Evidence: mkdocs/nav audit.
- **I08 docs strict build** — `mkdocs build --strict` exit 0. Evidence: command.
- **I09 artifact hashes** — generated artifacts have hash manifests. Evidence: hash tests.
- **I10 context bundle freshness** — bundles state audited SHA + stale inputs. Evidence: bundle manifest.

## 16. Category J — Ecosystem / performance / future bridges

- **J01 Jaxley bridge** — lazy, optional, guarded; import without Jaxley works. Evidence: optional import tests.
- **J02 PyNWB bridge** — optional, metadata-rich, round-trip validated with units/status. Evidence: NWB tests.
- **J03 Equinox readiness** — PyTree/static-array boundary planned/prototyped; no hard dep. Evidence: design/test.
- **J04 Diffrax readiness** — optional ODE/SDE backend, not default tutorials. Evidence: optional tests.
- **J05 Lineax solver readiness** — optional linear solver tied to residual/boundary/gauge tests; no solver claim without residual. Evidence: solver tests.
- **J06 Optax wrapper readiness** — honest differentiability status; optional + tested. Evidence: optimizer tests.
- **J07 BrainPy reference policy** — design reference, not hard dep. Evidence: dependency audit.
- **J08 performance benchmarks** — runtime benchmarks with hardware/backend receipts. Evidence: benchmark report.
- **J09 connectivity compiler** — declarative rules compile to sparse finite edge arrays; all weight modes/selectors/mechanisms/artifacts tested. Evidence: connectivity compiler tests.
- **J10 FlatNet / weld / reconstruct path** — weld/clone/flatten/simulate safe + tested. Evidence: architecture gate tests.

---

## 17. Scoreboard markdown template

```markdown
# jaxfne Scoreboard
Release: · Branch: · SHA: · Date: · Evaluator:
Truth status: truth_safe_unverified · Claim level: computational_scaffold
Field solver status: laminar_proxy_no_pde · Physical amplitude allowed: false

## Commands
| Command | Result |
|---|---|
| `git status --short` | |
| `python -m compileall -q jaxfne tests examples scripts` | |
| `pytest tests/ -q --tb=line` | |
| `python scripts/audit_notebooks_and_assets.py --check` | |
| `mkdocs build --strict` | |

## Category scores
| Category | Score | Status | Evidence |
|---|---:|---|---|
| A Release / provenance | | | |
| B Public API / surface | | | |
| C Configuration / model / simulation | | | |
| D JAX runtime | | | |
| E Numerical safety | | | |
| F Source / field / probe | | | |
| G Objectives / optimizers | | | |
| H Truth gates / claim safety | | | |
| I Notebooks / docs / artifacts | | | |
| J Ecosystem / performance / future bridges | | | |

## Factor scores
| ID | Factor | Score | Status | Confidence | Evidence | Next action |
|---|---|---:|---|---|---|---|
| A01 | Repo state frozen | | | | | |
| ... | ... | | | | | |

## Top blockers
| Rank | Factor | Blocker | Minimal patch | Owner |
|---:|---|---|---|---|

## Next safe action
```

---

## 18. Consistency metrics

Mirroring the multi-LLM ontology paper, adapted to repo audits.

> **Comparability precondition.** `mean_f` and `mean_{a'}` are only meaningful
> when both operands share **the same factor-ID assignment**. Compute these
> metrics **only across scoreboards authored under this grammar (v0.1+)**. Do
> not compute them against the legacy `jaxfne_glossary.md` instance until it is
> re-mapped to grammar IDs (see the legacy-compatibility note at the top).

### 18.1 Auditor Consistency
```text
AC(a_i, a_j) = mean_f [ (J(r, a_i, c, f) - J(r, a_j, c, f))^2 ]
```
How similarly two auditors scored the same repo state.

### 18.2 Release Consistency
```text
RC(r_i, r_j) = mean_f [ (J(r_i, a, c, f) - J(r_j, a, c, f))^2 ]
```
How much a release moved in score space.

### 18.3 Release-Agent Consistency
```text
RAC(r, a) = mean_f [ (J(r, a, c, f) - mean_{a'≠a} J(r, a', c, f))^2 ]
```
Whether one auditor is an outlier for a release.

### 18.4 Scope catalogue temperature
```text
T_R = c * V_env / n
```

| Symbol | Meaning |
|---|---|
| `T_R` | scope catalogue temperature |
| `V_env` | volume of the enclosing ellipsoid across category-score vectors |
| `n` | number of releases / branches / runs |
| `c` | scaling constant (default 1.0) |

| Temperature | Meaning |
|---|---|
| low | stable, compact scope catalogue state |
| high | dispersed; architecture goals diverging |
| increasing | new work expanding unresolved surface |
| decreasing | implementation consolidating around stable contracts |

---

## 19. File location

```text
internal_docs/jaxfne_scoreboard_glossary.md        # this grammar
docs/development/scoreboard_glossary.md             # optional public summary
```

Do not publish raw internal scores in public docs unless every score has measured evidence.

---

## 20. Maintenance protocol

1. Re-score at every merged architecture PR.
2. Re-score before every tag.
3. Mark non-measured scores `estimated` or `TBD-not-implemented`.
4. Never let a score imply empirical validation.
5. Link every score to command/source/test/artifact evidence.
6. Keep truth/status gates at the top of every scoreboard.
7. Preserve prior scoreboards under `internal_docs/scoreboards/YYYYMMDD_<sha>_scoreboard.md`.
8. Use score deltas to drive backlog: `priority = score_gain / implementation_risk`.

---

## 21. Minimal backlog grammar

```markdown
| Rank | Factor | Current | Target | Patch | Risk | Stop rule | Expected gain |
|---:|---|---:|---:|---|---|---|---:|
```

Classification: `PATCH` (small fix) · `FOLLOW-UP` (known, not urgent) ·
`REPRODUCE` (needs failing repro first) · `PLAN` (design before code) ·
`REJECT` (do not do) · `ACCEPT` (no action).

Reversibility: `GREEN` (autonomous safe) · `YELLOW` (do then hold for review) ·
`RED` (human design decision before code).

---

## 22. First recommended instantiation

Create the first measured scoreboard after:

```text
v0.3.28 circuit declarations merged   ← DONE (263c180)
v0.3.30 connectivity compiler branch starts   ← next
```

First measured categories:

| Category | First measured evidence |
|---|---|
| A | branch/SHA/tag/CI |
| B | `jaxfne.__all__` audit |
| C | config-ownership tests |
| D | JAX scan/vmap/jit grep + runtime tests |
| E | finite/JSON tests |
| F | source/field/probe reports |
| G | objective/null tests |
| H | truth-gate grep |
| I | notebook audit + mkdocs |
| J | connectivity/bridge state |

---

## 23. One-line standard

A valid `jaxfne` score is not an opinion. It is:

```text
factor + score + status + evidence + blocker + next action
```
