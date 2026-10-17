# Calibrated reproduction: the residual ("dark matter") protocol

Hamm (2026-10-13): a replication must reproduce. When the port misses the paper, find the missing
piece and say what it is. This protocol applies after a signed prereg has been scored, and the
prereg's verdict stays on record unchanged. Everything here is post hoc, and results are labelled
**calibrated reproduction**.

## Terms

- y = f(x): the paper's value of each declared estimand (a rate, a peak frequency, a curve point),
  taken from the source document (`SOURCE_P1.md` / prereg), with its uncertainty where stated.
- z = g(x; θ): the jaxfne model at the paper's parameters θ₀.
- Relative error per estimand: e = |z − y| / |y| (absolute error for estimands near 0, with the
  scale named in the lane's plan).

## Verdict bands

| Band | Rule | Label |
|---|---|---|
| exact | e = 0 within the backend's rounding | reproduced |
| close | every estimand e ≤ 0.05 | reproduced (within 5 %) |
| off | any e > 0.05 | run the steps below |

## Steps when off

1. **Simplify first.** Remove or isolate components until the smallest model that still shows the
   miss is found. Record each step and its errors.
2. **Fidelity before fitting.** Check every known port-versus-paper difference (units, update
   order, missing terms, wrong defaults, unread source settings) and fix real ones as fidelity
   fixes, with a test each. A fidelity fix is not a fit.
3. **Residual term.** Add one declared component X(x) at a named place in the model (for example an
   extra current, a missing synaptic family, an input term) with gain a, so the model is
   g(x; θ) + a·X(x). Fit a (θ fixed at θ₀) with Adam (`jaxfne.optim.optax_adam`) when g is
   differentiable, or AGSDR (`jaxfne.optim.agsdr`) when it is not, on
   L = Σ_estimands e². The fitted a and the drop in L say how much the miss needs X.
4. **Absorb a into θ.** Declare the degrees of freedom D ⊆ θ before fitting: parameters the paper
   leaves unstated, states only approximately, or that the port had to assume, each with a range.
   Never a parameter the paper states exactly, unless it is declared as a deviation with a reason.
   Fit θ ∈ D on L + λ·a², ramping λ up until a reaches 0. Gradient step on θ:
   Δθ = −η ∂L/∂θ, and the drift of a tells which parameters stand in for X.
5. **Remove X and re-run.** With a = 0 and X deleted from the model, run θ* fresh.
   - Fit on seeds 1–3.
   - Score on held-out seeds 4–8.
   - The held-out run sets the band.
6. **When a cannot reach 0** within the bands, report the smallest |a|, the form of X, and the
   alternatives tried. That X is the finding: the paper's model needs something its text does not
   state.

## Records (one file per lane, `CALIBRATION.md` beside the lane's prereg)

- the estimand table: y, z at θ₀, z at θ*, e;
- every simplification, fidelity fix, X form, a trajectory and θ change, with its file and seed;
- D with ranges and the reason for each parameter;
- the held-out result and its band;
- the claim: "reproduced after calibration of [θ ∈ D] (post hoc), held-out seeds 4–8" or
  "needs X: [form], |a| = …".

## Engine: jaxfne core only

Hamm (2026-10-13): every replication runs on jaxfne's core and grammar
(`CircuitSpec`/`Configuration -> construct -> Model -> simulate -> Signals`, registered emitters,
HDP rules, probes, `jaxfne.optim`). z is always the core model's output.

- A mechanism the paper needs and the core lacks (an emitter, a synapse family, a plasticity
  rule, a probe) is added to `jaxfne/` as a core component: its own module, registered through
  the existing registries, with unit tests and refusal of unconsumed parameters (P-014, P-015).
  An étude-local simulation kernel is never the scored engine.
- A transcription of the authors' algorithm (event-exact, numba, C++ order) is an **oracle**:
  it checks the core model (agreement gates), and its numbers are never reported as z.
- A residual X(x) from the steps above is written as a core component too, so a finding of the
  form "needs X" names a jaxfne object.
- Every model is written in the form (h, y) = f(x, h') on a neural tensor A (x : A : y): h the
  state, x and y the inputs and outputs or interactions, A a multi-area, multi-layer
  `NeuronalTensor` (N = 1 included: one self-exciting neuron is one area, one layer, one
  connection, 3-D space and 1-D time). The core must be sufficient for any such model; where it
  is not, the gap is closed in the core.
- Geometry is part of A. Each replication declares its geometry (layer `Geometry3D`, area
  `Pose3D`), and every term that depends on space (delays, connection probability, fields)
  reads it from A, so a transformation A -> A' changes the dynamics exactly through those terms.

## Runs

- WSL Linux, JAX on CUDA: `~/venv_jaxfne` (jax with CudaDevice; optax installed 2026-10-13),
  float32 unless a test needs x64.
- One GPU: wrap every GPU job as `flock /tmp/jaxfne_gpu.lock <command>`. CPU oracles (the PCL numba
  port) run in WSL without the lock.
- Every run's manifest records code SHA, backend, dtype and seeds.
- Never run the authors' code; read it as data.

## Replication checklist

Every replication passes each item before its stage closes. Sources: Hamm's rulings
(2026-10-10 to 2026-10-17), `release_plan_0.5.6-0.5.10.md` §3, the sections above.
A row marked *gate* is checked by a script; the others are checked by the reviewer, who
names the file and line that satisfies each one.

### Engine

| # | Rule | Check |
|---|---|---|
| E1 | The scored run builds and simulates through jaxfne: `Configuration`/`CircuitSpec -> construct -> Model -> simulate -> Signals`, registered emitters, rules, probes and `jaxfne.optim`. | grep the scored entry point for `construct(` and `.simulate(` |
| E2 | A mechanism the paper needs and the core lacks (emitter, synapse family, morphology, plasticity rule, probe, estimator) goes into `jaxfne/` as a core component with unit tests and refusal of unconsumed parameters (P-014, P-015). | new names live under `jaxfne/`; the étude imports them |
| E3 | Estimators (rates, spike detection, kappa, PSD, regimes) come from `jaxfne.analysis`. A paper-specific estimator is added there first. | no local copy of a core operator in the étude |
| E4 | A residual X from the residual protocol is a core component, so "needs X" names a jaxfne object. | X is importable from `jaxfne` |
| E5 | The model is stated as (h, y) = f(x, h') on a `NeuronalTensor` with declared geometry; terms that depend on space read it from the tensor. | PORT_SPEC names A, x, h, y and the geometry |

### Source and authors' code

| # | Rule | Check |
|---|---|---|
| S1 | The authors' code is read as text only: never imported, executed or added to `sys.path`. Packets say so. | *gate*: grep for imports, `exec`, `subprocess`, `sys.path` into downloaded trees |
| S2 | A transcription of the authors' model is an oracle: it is called from tests or agreement gates, and its numbers are never reported as the result. | oracle modules are imported only by `tests/` or a gate script |
| S3 | Every paper used or named is in `docs/reference/references.md` with a Crossref-checked DOI, in the commit that first uses it. | grep the reference list |
| S4 | `SOURCE_P1.md` lists methods, code, data, licence and every gap, each item observed or inferred. | the gap list has no unknowns at P1 close |

### Fidelity

| # | Rule | Check |
|---|---|---|
| F1 | `PORT_SPEC.md` compares the port with the reference row by row (wiring, morphology, units, kinetics, solver, update order, stimulus, recording site), each row with file:line on both sides. | second review of the table |
| F2 | A declared simplification is tested against the reference before any fit: show that it leaves the declared estimands unchanged, or retire it. (GSDR S1, one compartment, was not neutral: CALIBRATION §6.10.) | a test or recorded run per simplification |
| F3 | A fidelity fix carries a test that fails on the code before the fix. | the test fails on the parent commit |
| F4 | A new core component that has a public reference implementation (jaxley, NEURON, Brian2) is checked against it on a small case with a stated tolerance. | an oracle test, skipped only when the reference is not installed |

### Protocol and runs

| # | Rule | Check |
|---|---|---|
| P1 | `PREREG_P3.md` fixes estimands, bands, nulls, seeds and sensitivity runs before scoring, and Hamm signs it in the release batch. | signature line present; record-integrity pins the file |
| P2 | Seed 0 is for pilots. Scored runs use declared seeds, and the seed shown to change the output is named. | two seeds give different outputs |
| P3 | Float32 by default; one float64 run on one seed checks every scored result. GPU reductions vary in order: a quoted number is reproduced once, or taken from CPU. | the manifest records backend and dtype |
| P4 | Every run writes a manifest with code SHA, backend, dtype, seeds and the activity-regime block. | *gate*: manifest keys present |
| P5 | Runs of 5 minutes or more are detached, take the GPU lock (`flock /tmp/jaxfne_gpu.lock`) and are complete only when their receipt exists. | receipt path in the ledger |
| P6 | A frozen protocol is never tuned after its outcome. A miss goes through the residual protocol, labelled post hoc, with held-out seeds 4–8. | CALIBRATION records each step |

### Results and records

| # | Rule | Check |
|---|---|---|
| R1 | A stage closes only when its record exists: P1 `SOURCE_P1.md`, P2 `PORT_SPEC.md`, P3 `PREREG_P3.md`, P5 `LEDGER.md` and `RESULT_P3.md`, calibration `CALIBRATION.md`, P6 equivalence result. | *gate*: files present for the stage on the dashboard |
| R2 | Every number quoted in a record is reopened from its JSON or receipt before commit. | reviewer reopens each number |
| R3 | Verdicts use the status tags (reproduced, reproduced (conditional), in progress, not started, failed). A scored miss keeps the paper in progress with its criteria visible. | dashboard row matches the folder's records |
| R4 | A paper counts as replicated with jaxfne only at P6: the same estimands through the public API, equal to P5 within the declared tolerance. | P6 result file |
| R5 | `dashboard/sources.json`, `REPLICATIONS.md` and `docs/start/replications.md` are regenerated with the stage change (`build_dashboard_data.py --check`, `generate_replications_page.py --check`). | *gate*: both `--check` pass |
| R6 | The paper folder keeps at most 7 files and 7 folders (`run/`, `fit/`, `diag/`, `tests` in `tests/`). | *gate*: tree-degree test |

### Packets and review

| # | Rule | Check |
|---|---|---|
| K1 | Before writing a packet from a todo row, read `git log --grep <id>`, the live code and the lane's CALIBRATION. | packet cites the base commit |
| K2 | Each feature gets an adversarial review by a separate agent; science claims go to a second reviewer. | review verdict in the commit message |
| K3 | Workers run targeted tests; the dispatcher runs the broad gate once, in series, before the push. | gate summary line quoted |
