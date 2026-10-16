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
