# jaxfne Biophysics Glossary and Publication-Readiness Guide

> **Reference document — not the session entry point.** Read [`AGENT_QUICKREF.md`](AGENT_QUICKREF.md) first; re-freeze git SHA and `evidence_inventory.py` before citing publication state from this file.

**Purpose:** durable deep reference for evidence-ready JAX/JIT/Jaxley-compatible computational neurobiophysics, neuroelectrodynamics, and tensor-field source-to-readout scaffold.

**Audience:** Cursor/Composer agents, repo workers, paper-editing workers, and future release auditors (on demand).

**Canonical import:**

```python
import jaxfne as jtfne
```

**Current posture to preserve unless stronger run evidence exists:**

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

**Core thesis:** `jaxfne` makes emitter-to-source-to-field/probe assumptions explicit, executable, auditable, and hashable. It is a compact JAX-native computational scaffold, not yet a validated EEG/MEG forward solver or calibrated physical-amplitude simulator.

---

## 1. Cursor start protocol

Before any publication, tutorial, package, or technical report work:

```bash
git fetch --all --prune
git branch --show-current
git status --short
git rev-parse HEAD
python3 scripts/evidence_figures_inventory.py
```

For publication-track work:

```bash
git switch cur
git pull --ff-only origin cur
python3 scripts/evidence_figures_inventory.py
```

Expected current publication posture (sync and verify):

```text
main figures: 8/8
extended data: 10/10
completed ED: ED1–ED10 (through ed10_release_archive_receipt)
next ED target: none (evidence artifact stack complete; release/tag/archive pending approval)
live HEAD after ED10 merge: verify with `git rev-parse HEAD` on `cur`
```

Permanent branches: `main`, `dev`, `agy`, `cur`. Publication work lands on `cur`.

Do not mutate `main`, `dev`, or `agy` without explicit approval. Do not force-push, tag, release, or publish packages without explicit approval.

**Branch cleanup rule:** after a feature/publication branch merges into a permanent branch, delete the source branch locally and on `origin` unless explicitly retained.

---

## 2. Core operator grammar

### 2.1 Pipeline identity

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer -> Manifest/Validation
```

### 2.2 Tensor/operator closure

```text
Y = P o F o S o E
L = O(Y, target, gates, manifest)
theta_next = A(theta, L, constraints, key)
M_next = V(M, X, Q, Z, Y, L, theta_next)
```

### 2.3 Worded equation

```text
readout = probe(field(source(emitter(state, drive, params, key))), manifest)
objective report = metrics + criteria + nulls + validation metadata
next parameters = optimizer(current parameters, objective report, constraints, key)
run evidence = numerical arrays + reports + manifests + hashes + validation status
```

### 2.4 Implementation locations

| Operator | Meaning | Package locus | Evidence requirement |
|---|---|---|---|
| `E` emitter | Neural state dynamics | `emitters.py`, `core.py`, Jaxley bridge later | finite state, seed, dtype, params, update rule |
| `S` source | Emitter state to source tensor | `fields.py`, source reports | source mode, calibration status, no double count |
| `F` field | Source to field/proxy | `fields.py` | solver/proxy status, boundary/gauge if physical |
| `P` probe | Field/source to readout | `core.py`, `fields.py`, probe reports | shape, units/status, method, calibration, interpretation |
| `O` objective | Metrics/gates/nulls | `objectives.py` | finite metrics, gates, null status, targets |
| `A` optimizer | Parameter proposal/update | `optim.py` | bounds, seed, budget, rejection reasons |
| `V` validation | Manifest/report closure | `validation.py`, `io.py` | strict JSON, hashes, finite outputs, gates |

---

## 3. Core glossary

| Term | Definition | jaxfne status | Stronger-evidence requirement |
|---|---|---|---|
| **Emitter** | Local neural or biophysical state generator: reduced neuron, conductance model, compartment, population, or local circuit. | implemented scaffold | calibrated state variables and units for physical interpretation |
| **Source** | Declared current/source tensor derived from emitter outputs. | proxy/source bookkeeping | calibration bridge, units, source support, conservation tests |
| **Field** | Potential/current/electromagnetic variable from source through a field operator. | laminar proxy path | PDE/forward solve with boundary, gauge, residual, convergence |
| **Probe** | Readout operator sampling spikes, voltage, source, LFP-like, CSD-like, EEG-like, MEG-like, or EMM-proxy. | implemented proxy readouts | geometry, units, sensor model, validation for physical claims |
| **Objective** | Metric, loss, criterion, null comparison, or rejection report computed from readouts. | computational target | empirical interpretation needs held-out data, nulls, ablations |
| **Optimizer** | Search/update procedure under declared bounds and objective reports. | GSDR/AGSDR/random-search scaffold | uniqueness/mechanism support requires additional evidence |
| **Manifest** | JSON-safe closure object recording runtime, seed, dtype, source/field/probe status, validation, and hashes. | required evidence artifact | final release tag/SHA/wheel/archive for submission |
| **Truth gate** | Machine-readable claim boundary. | required | may only escalate with evidence-specific validator |
| **Physical amplitude** | Claim that numeric values are in calibrated physical units. | false by default | source calibration + solver + units + validation |
| **Mechanism support** | Claim that a biological mechanism is supported. | not from objective alone | nulls, ablations, perturbations, repeated seeds, empirical comparison |

---

## 4. Local-nonlinear / global-linear split

### 4.1 Local nonlinear elements

Treat these as local maps unless explicitly declared otherwise:

- ion-channel gating
- membrane dynamics
- spike threshold/reset
- synaptic saturation/release
- dendritic integration
- receptor traces
- plasticity rules
- local population transfer functions
- metabolic or vascular local state variables
- local adaptive medium state

### 4.2 Global linear elements for fixed metadata

Treat these as linear operators when medium/source/probe metadata are fixed:

- source projection
- laminar mixing
- volume conduction
- field spreading
- leadfield projection
- sensor readout
- fixed-kernel long-range coupling

### 4.3 Failure test

If a proposed global operator changes its kernel as a nonlinear function of distant activity, do **not** call it a fixed linear field operator. Represent it as:

1. a local adaptive medium state,
2. an explicitly nonlinear model component, or
3. a separate state-dependent operator with status metadata and tests.

---

## 5. Mathematical glossary flow

Every tutorial/technical report equation should follow this sequence:

```text
formal equation -> term definitions -> worded equation -> implementation location -> interpretation boundary
```

### 5.1 Emitter dynamics

```text
dx/dt = f(x, u, theta, t)
X[k+1] = Psi_dt(X[k], U[k], theta, A, key[k])
```

**Terms:**

- `X`: emitter state array.
- `U`: drive/input array.
- `theta`: model parameters.
- `A`: coupling/connectivity.
- `key`: explicit PRNG key.
- `Psi_dt`: JAX-executable update kernel.

**Implementation:** pure JAX kernel; `lax.scan` preferred for time.

**Boundary:** emitter scaffold unless state variables and drives are calibrated.

### 5.2 Reduced Izhikevich emitter

```text
dv/dt = 0.04 v^2 + 5v + 140 - u + I_native
du/dt = a(bv - u)
if v >= 30: v <- c, u <- u + d
```

**Bridge term:** `I_native`.

**Boundary:** native drive is not amperes by default. It can participate in proxy/source bookkeeping; physical current requires a calibration bridge.

### 5.3 Conductance and membrane-current balance

```text
C_m dV_m/dt = -sum(I_ion) - I_syn + I_inj
I_ion = gbar * gates * (V_m - E_rev)
```

**Bridge term:** `I_ion + I_syn`.

**Boundary:** physical when morphology, area, units, sign convention, and calibration are declared. Scaffold when toy-scaled.

### 5.4 Source alternatives

```text
q = chi * I_m,total + q_ext
q = q_cap_ion + q_syn + q_ext
```

**Rule:** total-current mode and decomposed-current mode are disjoint alternatives. Use one source mode per run.

**Stop condition:** any source path double-counting synaptic/current components.

### 5.5 Source projection and calibration

```text
q(x,t) = sum_n I_n(t) eta_epsilon_n(x - x_n)
int eta_epsilon_n(x - x_n) dx = 1
I_phys_n(t) = alpha_I * I_native_n(t)
```

**Bridge term:** calibration map `alpha_I` or `A[n,beta]`.

**Boundary:** physical amplitude requires calibrated source map, source support, units, and validation.

### 5.6 Quasi-static field law

```text
E_e = -grad(phi_e)
J_e = sigma_e E_e = -sigma_e grad(phi_e)
div(J_e) = q
div(-sigma_e grad(phi_e)) = q
```

**Boundary:** physical in quasi-static resistive regime only when conductivity, source units, domain, boundary condition, gauge, solver residual, convergence, and finite fields are declared.

### 5.7 Charge continuity

```text
partial_mu J^mu = 0
partial_t rho + div(J) = 0
```

**Boundary:** required for physical source-field interpretation. Proxy runs may export q-like arrays but must not claim physical charge/current unless continuity metadata exists.

### 5.8 Boundary and gauge

```text
J_e dot n = g_N on boundary
int_Omega q dx = int_boundary J_e dot n dS
int_Omega phi_e dx = 0 or phi_e(x0,t) = 0
```

**Boundary:** physical diagnostic when source integration, boundary flux, gauge residual, solver residual, and finite fields are validated.

### 5.9 Conductivity passivity

```text
sigma_e = sigma_e^T
z^T sigma_e z >= 0
sigma_e = L L^T + epsilon I
```

**Boundary:** passive tissue medium constraint. Active stimulation or metabolic energy belongs in source/stimulation terms, not hidden field activity.

### 5.10 Laminar proxy

```text
Phi = S W^T
Phi[k,c] = sum_n S[k,n] W[c,n]
CSD_like = Delta_zz Phi
```

**Bridge term:** projection weights `W[c,n]`.

**Boundary:** simulated/proxy field path. Supports within-run relative comparisons when projection, normalization, sign convention, and metadata are exported.

### 5.11 Probe report

```text
Y[j] = P_j(source, field, metadata)
R[j] = {kind, method, shape, units/status, operator_status, calibration_status, interpretation_level}
```

**Rule:** probe report is as important as array data.

### 5.12 LFP-like / CSD-like

```text
Y_LFP_like = Phi
Y_CSD_like = D_zz Phi
Y_CSD_physical = div(J_e)
```

**Boundary:** physical CSD requires field spacing, sign convention, calibrated current density or validated estimator.

### 5.13 EEG-like / MEG-like

```text
Y_EEG_like = L_EEG Phi
Y_MEG_like = L_MEG O S
B(r,t) = mu_0/(4 pi) int J(x,t) x (r-x)/||r-x||^3 dx
```

**Boundary:** physical EEG/MEG requires source orientation, head/volume model, leadfield, sensor geometry, units, and validation. Current tutorial readouts are proxies.

### 5.14 EMM-proxy

```text
EMM_proxy[k] = w_spk rho_spk[k] + w_src norm(S[k,:]) + w_field norm(D[k,:]) + w_syn a_syn[k]
```

**Boundary:** normalized within-run activity/source/field-cost proxy. Physical power density requires calibrated `J`, `E`, conductivity/admittivity, geometry, boundary conditions, and solver evidence.

---

## 6. Electromagnetic admissibility ladder

| Level | Name | Allowed claims | Required metadata/tests | Forbidden claims |
|---:|---|---|---|---|
| P0 | Proxy projection | simulated/proxy readout, relative within-run comparison | finite arrays, shape, proxy labels, manifest | physical amplitude, solved field |
| P1 | Declared leadfield-like projection | declared geometry proxy | source/probe geometry metadata, fixed linear operator, finite outputs | empirical EEG/MEG equivalence |
| P2 | Boundary-normalized kernels | boundary-aware projection scaffold | normalization, conservation-style checks, sign convention | PDE solve unless system solved |
| P3 | Discrete volume solve | numerical field solve candidate | `K phi = q`, boundary, gauge, residual, convergence, finite phi/J/CSD | calibrated amplitude unless source units calibrated |
| P4 | Differentiable adjoint solve | differentiable solver candidate | VJP/adjoint checks, gradient tests, JIT/VMAP safety | empirical mechanism proof |
| P5 | External validation | externally compared solver/readout | comparison to external tool or empirical reference, held-out tests | universal accuracy/superiority |

**Current v0.3.x default:** P0.

**Solver namespace rule:** future physical solvers should live under experimental namespace until P3/P4/P5 evidence exists, e.g. `jtfne.fields.experimental.solve_volume_conductor_smoke(...)`.

---

## 7. Source calibration ladder

| Level | Source status | Meaning | Evidence needed |
|---:|---|---|---|
| S0 | native reduced drive | unitless or native model drive | finite outputs, emitter params, seed |
| S1 | normalized proxy source | source-like array for readout scaffolds | source mode, normalization metadata |
| S2 | calibrated current bridge | native features mapped to amperes | calibration factor/map, units, tests |
| S3 | morphology/area-aware current | membrane current with geometry support | morphology, area, sign convention |
| S4 | source-density conservation report | source density integrates consistently | source support, integral checks |
| S5 | empirical/external calibration | source amplitude validated externally | observed/external comparison |

---

## 8. Probe/readout ladder

| Readout | Default v0.3 status | Stronger evidence required | Manifest fields required |
|---|---|---|---|
| `spk` | simulated spike/event readout | emitter specification, finite output, seed | kind, shape, threshold/reset, finite |
| `vm` | voltage-like for reduced emitters | state units and calibration | units/status, emitter family, calibration |
| `source` | proxy/declared source tensor | source mode, calibration, conservation | source_mode, source_calibration_status |
| `lfp_like` | laminar potential-like proxy | calibrated potential, electrode geometry | field_solver_status, probe geometry |
| `csd_like` | finite-difference proxy | spacing, sign convention, current density or validated estimator | dz, sign, method/status |
| `eeg_like` | proxy projection | leadfield/head model/sensors/units/validation | leadfield_status, geometry, units/status |
| `meg_like` | proxy projection | current orientation, magnetic forward model, sensors/units | magnetic_forward_status, sensor metadata |
| `emm_proxy` | normalized cost proxy | calibrated fields/currents and energy accounting | proxy_weights, calibration status, field status |

---

## 9. JAX/JIT discipline glossary

| Rule | Required behavior | Stop condition |
|---|---|---|
| JAX arrays | numerical kernels use JAX arrays and `jax.numpy` | NumPy in hot kernel without reason |
| PRNG keys | explicit keys, no global random state | hidden randomness or missing seed |
| Time loops | prefer `lax.scan` for time | Python loop in hot path without reason |
| Batch axes | prefer `vmap` for seeds/candidates/readouts | manual loops for vectorizable hot path |
| JIT | only pure numerical hot paths | plotting/JSON/file I/O under JIT |
| Timing | warmup separately; `block_until_ready()` before timer stop | measuring async dispatch only |
| dtype | default float32; x64 opt-in | silent float64 drift |
| CPU correctness | CPU smoke before accelerator claims | GPU-only path without CPU baseline |

---

## 10. Jaxley and PyNWB compatibility

### Jaxley

Role: optional detailed differentiable cell/compartment emitter bridge.

Rules:

- Keep as optional lazy dependency.
- Use guarded imports in bridge modules.
- Do not make top-level `import jaxfne as jtfne` require Jaxley.
- Do not reimplement Jaxley internals in jaxfne.
- Bridge output must enter TFNE through explicit source maps and metadata.

### PyNWB

Role: optional export/archive target.

Rules:

- Keep as optional lazy dependency.
- Export explicit schema, units/status, provenance, session metadata when available.
- Validate read/write round trips.
- Avoid NaN/Inf metadata.
- Do not claim empirical data validity from NWB export alone.

---

## 11. Publication-readiness scoreboard

| # | Factor | Score now | Target | Evidence needed | Next action | Stop condition |
|---:|---|---:|---:|---|---|---|
| 1 | Branch/release hygiene | 90 | 100 | clean branches, pinned SHA/tag, no force-push | keep `cur` isolated | dirty state before work |
| 2 | Main figure stack | 88 | 100 | 8/8 generated scripts/PNGs/manifests/hashes | final rerun bundle | hand-edited figure without script |
| 3 | Extended Data stack | 98 | 100 | 10/10 ED panels with receipts | release approval only | ED without receipt/manifest |
| 4 | Manifest/hash closure | 72 | 100 | hash table for all artifacts | rerun ED5 after new ED | self-reference drift unreported |
| 5 | Notebook execution evidence | 45 | 95 | smoke/full receipts, paths, cells, errors | expand ED3 | claiming full execution without receipt |
| 6 | JSON/schema validation | 65 | 95 | strict JSON, schemas, expected failures | broaden ED2 | NaN/Inf or raw ndarray in JSON |
| 7 | API stability | 65 | 95 | `__all__` snapshot, root import, wrappers | final ED1 refresh | public API removal |
| 8 | Optional dependency laziness | 70 | 95 | subprocess import receipt | maintain ED4 | root import requires optional dep |
| 9 | JAX numerical discipline | 65 | 95 | scan/vmap/jit receipts and lints | add targeted tests | I/O or plotting inside jit |
| 10 | Source bookkeeping | 55 | 95 | one source mode, calibration status, source reports | ED source-accounting panel | current double-counting |
| 11 | Probe/readout contracts | 78 | 95 | all readouts finite/status-labeled | ED7 done; keep probe tests | physical labels on proxy arrays |
| 12 | Electromagnetic admissibility | 35 | 90 | boundary/gauge/continuity/passivity/residual tests | solver ladder docs/tests | PDE/field claim without residual |
| 13 | Physical amplitude discipline | 80 | 100 | amplitude gate false unless evidence present | preserve gates | calibrated claim without calibration |
| 14 | Mechanism-claim discipline | 82 | 95 | nulls, ablations, repeated seeds, alternatives | ED9 done; keep null tests | objective success called proof |
| 15 | Benchmark evidence | 65 | 85 | hardware/env/timing-phase receipts | ED6 done; keep local receipt only | speedup claim from local smoke |
| 16 | Adjacent-tool positioning | 70 | 95 | capability comparison with citations | final Fig8/ED text | superiority claim |
| 17 | Tutorial-to-package discipline | 45 | 90 | notebook helpers moved to package/utils | tutorial dedupe pass | notebook-local solver/readout/optimizer |
| 18 | Release archive readiness | 72 | 100 | tag, wheel hash, clean install, archive/DOI if used | release/tag/archive approval | package release without approval |
| 19 | Technical report comparison | 50 | 95 | TBDs replaced with exact release receipts | post-ED technical report pass | numeric claim without manifest |
| 20 | Empirical validation readiness | 10 | 70 | datasets, nulls, held-out tests, observed comparison | later scope | empirical claim in scaffold paper |
| 21 | Config-first backbone | 35 | 95 | circuit state in `Configuration` | 0.3.28 after evidence stack | breaking old tutorials |
| 22 | Identity/selectors | 30 | 95 | stable area-layer-type IDs | 0.3.29 | nondeterministic selectors |
| 23 | Connectivity rules | 25 | 95 | typed rules, mechanism/weight specs | 0.3.30 | silent empty rule selection |
| 24 | Weld/reconstruct/flatten | 15 | 90 | weld, to_config, flatten, tracking maps | 0.3.31-0.3.33 | lost identity map |
| 25 | Solver-readiness | 20 | 90 | source schema, field schema, boundary/gauge metadata | post-0.3.34 | solver work before schemas stable |

---

## 12. Extended Data ladder

| ED | Topic | Reviewer question | Status |
|---:|---|---|---|
| ED1 | API stability snapshot | What public API is exposed? | completed |
| ED2 | JSON/schema validation | Are configs/manifests strict and valid? | completed |
| ED3 | Notebook execution receipts | What notebooks are executed or receipt-tracked? | completed |
| ED4 | Optional dependency laziness | Does root import stay lightweight? | completed |
| ED5 | Manifest hashes | Are artifacts internally hash-consistent? | completed (`ed05_manifest_hashes`) |
| ED6 | Benchmark receipt table | What are local runtime costs and hardware conditions? | completed (`ed06_benchmark_scaling_tables`) |
| ED7 | Probe/readout contracts | Are readouts separated and status-labeled? | completed (`ed07_probe_operator_contracts`) |
| ED8 | Tutorial atlas coverage | What tutorials exist and what evidence do they export? | completed (`ed08_tutorial_atlas_coverage`) |
| ED9 | Failure/null controls | What prevents overinterpretation? | completed (`ed09_failure_modes_and_nulls`) |
| ED10 | Release bundle receipt | What exact release generated the paper? | completed (`ed10_release_archive_receipt`; release actions pending approval) |

---

## 13. 0.3.28-0.3.34 package hardening ladder

Do not start this ladder on publication branches until main+ED evidence stack stabilizes.

| Version | Theme | Gate |
|---:|---|---|
| 0.3.28 | config owns circuit declarations | no essential free-floating network state |
| 0.3.29 | area-id-layer-type identity | stable selectors and neuron table |
| 0.3.30 | typed connectivity rules | pre/post selectors, mechanisms, weights |
| 0.3.31 | weld configs/models | deterministic renaming and metadata preservation |
| 0.3.32 | construct/reconstruct/clone | `construct(cfg) -> model -> cfg` stable |
| 0.3.33 | flatten for JAX/JIT | arrays + reversible tracking maps |
| 0.3.34 | full integration gate | 35 conditions pass |

---

## 14. Stop rules

Stop and report if any occur:

- proxy path described as solved field
- EEG/MEG-like proxy described as real EEG/MEG
- uncalibrated native current described as amperes
- physical amplitude claim without calibration, solver, units, geometry, and validation
- PDE solve claim without boundary, gauge, residual, convergence, finite fields
- EMM-proxy described as physical power density without calibrated `J` and `E`
- objective success described as biological mechanism proof
- package API removed without wrapper
- optional dependency imported eagerly by root import
- NaN/Inf or raw ndarray exported to strict JSON
- notebook-local solver/readout/objective/optimizer added when package API exists
- test modified before failure provenance is known

---

## 15. Cursor prompt: ED5 manifest hashes

```text
Resume jaxfne publication work from cur.

Start:
- git fetch --all --prune
- git switch cur
- git pull --ff-only origin cur
- git status --short
- git rev-parse HEAD
- python3 scripts/evidence_figures_inventory.py

Expected:
- 8/8 main figures
- 4/10 Extended Data
- clean tree

If `internal_docs/skills/README.md` still points to `.cursor/rules/jaxfne-super-skills.mdc`, patch it to `.cursor/rules/00-jaxfne-baseline.mdc` first. Validate `git diff --check`, inventory, and mkdocs. Do not touch package code.

Then create:
- branch `pub/ed05-manifest-hashes`
- `scripts/evidence_figures/ed05_manifest_hashes.py`
- `figures/evidence/ed05_manifest_hashes.png`
- `outputs/evidence/ed05_manifest_hashes_manifest.json`
- `outputs/evidence/ed05_manifest_hashes_receipt.json`

ED5 scope:
- local artifact integrity receipt only
- no archival completeness claim
- no release immutability claim
- no package API changes
- no tags/releases/packages
- preserve physical_amplitude_calibrated=false

ED5 must cover:
- fig01-fig08 scripts, PNGs, manifests
- ED1-ED4 scripts, PNGs, manifests, receipts when available
- publication_inventory.json if generated
- publication_checklist.json

For each row record:
- group
- artifact_id
- script path
- png path
- manifest path
- receipt path
- existence booleans
- SHA256 full values in receipt
- SHA prefixes in figure
- manifest-declared PNG SHA if present
- actual PNG SHA
- png_manifest_match
- json_strict_status
- gates_preserved if available

Validation:
- python3 -m compileall -q scripts/evidence_figures jaxfne tests
- python3 scripts/evidence_figures/ed05_manifest_hashes.py
- python3 scripts/evidence_figures_inventory.py
- python3 -m json.tool outputs/evidence/ed05_manifest_hashes_manifest.json >/dev/null
- python3 -m json.tool outputs/evidence/ed05_manifest_hashes_receipt.json >/dev/null
- python3 -m json.tool docs/evidence_artifacts/evidence_checklist.json >/dev/null
- python3 -m mkdocs build --strict
- PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/test_api_smoke.py tests/test_root_import_lightweight.py -q --tb=line

Acceptance:
- inventory becomes 8/8 main + 5/10 ED
- fig01-fig08 PNG/manifest SHA checks pass
- ED1-ED4 rows present
- ED5 row does not create unreported circular self-reference
- strict JSON passes
- no package API changes
- working tree clean except intended ED5 files before commit

Report:
Status, repo state, changed files, commands run, exact results, truth/evidence status, blockers, next safe action.
```

---

## 16. Cursor prompt: publication-readiness scoreboard doc

```text
Create or update `internal_docs/loop_context/PUBLICATION_READINESS_SCOREBOARD.md`.

Do not touch package code.
Do not start a new figure unless asked.
Do not change truth gates.
Do not create tags/releases.

Include:
1. current branch/SHA/inventory
2. core thesis: jaxfne makes emitter-to-source-to-field/probe assumptions explicit, executable, auditable, hashable
3. statement: jaxfne is not currently a validated EEG/MEG solver
4. scoreboard table with at least 20 factors:
   branch hygiene, main figures, ED stack, manifest hashes, notebook receipts, JSON schemas, API stability, optional deps, JAX discipline, source bookkeeping, probe contracts, electromagnetic admissibility, physical amplitude discipline, mechanism discipline, benchmarks, adjacent tools, tutorial discipline, release archive, technical report comparison, empirical validation
5. each factor includes score_now, target_score, evidence_needed, next_action, stop_condition
6. final targets: package paper, computational biophysics scaffold, physical solver later, empirical validation later

Validate:
- git diff --check
- python3 scripts/evidence_figures_inventory.py
- python3 -m mkdocs build --strict

Report with exact commands/results.
```

---

## 17. Cursor prompt: tensor-field biophysics objective map

```text
Create `internal_docs/loop_context/TENSOR_FIELD_BIOPHYSICS_OBJECTIVE_MAP.md`.

This is planning/documentation only.
Do not touch package code.
Do not implement solvers.
Do not change tutorials.
Do not claim physical amplitude.

Define final objective:
`jaxfne should become a compact JAX-native emitter-to-source-to-field/probe-to-objective scaffold for computational biophysics, with a staged path from proxy readouts to calibrated source bridges, electromagnetic admissibility checks, physical field solvers, probe validation, and empirical comparison.`

Sections:
1. operator chain
2. tensor closure equations
3. local nonlinear/global linear split
4. electromagnetic admissibility ladder P0-P5
5. source calibration ladder S0-S5
6. probe/readout ladder
7. Jaxley/PyNWB integration rules
8. stop rules
9. implementation scope catalogue
10. scoreboard with at least 15 factors

Validate:
- git diff --check
- python3 scripts/evidence_figures_inventory.py
- python3 -m mkdocs build --strict

Report exact commands/results.
```

---

## 18. Final publication targets

### 18.1 Package-methods paper target

A complete jaxfne paper demonstrates:

- install/import smoke
- deterministic simulations
- typed configuration
- source bookkeeping
- proxy readout families
- objective reports
- validation manifests
- reproducible figures
- strict JSON
- artifact hashes
- notebook receipts
- optional dependency laziness

### 18.2 Computational biophysics scaffold target

A stronger computational-biophysics jaxfne demonstrates:

- source calibration ladder
- source conservation checks
- probe geometry reports
- CSD sign convention
- passivity/conservation metadata
- solver/proxy separation
- Jaxley emitter bridges
- PyNWB export receipts

### 18.3 Physical solver target

A physical solver jaxfne line requires:

- stable source schema
- stable field schema
- conductivity/admittivity tensors
- domain/geometry
- boundary condition
- gauge condition
- residuals
- convergence checks
- calibrated units
- external or empirical validation

### 18.4 Empirical mechanism target

Mechanism-level claims require:

- fixed datasets
- repeated seeds
- held-out conditions
- null distributions
- ablations
- perturbation survival
- comparison to simpler alternatives
- observed-data metrics
- preregistered or frozen manifests when possible

---

## 19. Safe wording bank

Use:

```text
simulated/proxy readout
local artifact integrity receipt
computational scaffold
source bookkeeping
field-proxy metadata
probe report
validation manifest
operator status
calibration status
interpretation boundary
```

Avoid unless evidence exists:

```text
real EEG/MEG
calibrated amplitude
physical CSD amplitude
solved volume conductor
Maxwell solver
Poisson solver
mechanism proof
biological validation
power density
metabolic mechanism
```

---

## 20. Summary for agents

`jaxfne` is publication-close as an auditable JAX-native computational scaffold. The evidence artifact stack is complete at 8/8 main figures and 10/10 Extended Data (ED1–ED10). Release, tag, wheel publish, and archive/DOI remain approval-gated. Regenerate `outputs/evidence/*` on live checkout before final submission. The deeper biophysical path follows: config-first 0.3.28-0.3.34, then source calibration, solver metadata, experimental field solvers, PyNWB/Jaxley bridges, and empirical validation.

Do not upgrade claims faster than evidence.
