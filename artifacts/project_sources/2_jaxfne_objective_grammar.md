# jaxfne Objective and Optimization Grammar

## 1. Purpose

An objective in jaxfne is not merely a scalar loss. It is an evidence-bearing map from declared readouts and targets to finite metrics, gates, null checks, rejection reasons, and an optimization-compatible score.

Let the TFNE readout be

\[
Y_\Theta = (P_\eta \circ F_\gamma \circ S_\psi \circ E_\theta)(X_0,U,K).
\]

An objective is

\[
\mathcal{O}: (Y_\Theta, T, G, N) \mapsto R,
\]

where `T` is the target specification, `G` the truth/validity gates, `N` the null/control specification, and `R` an objective report.

## 2. Objective report

A robust report contains at least:

```text
metrics
score/loss
targets
constraints
gates
null/control status
finite status
rejection reasons
readout provenance
seed/runtime provenance
```

The optimizer consumes the score only after validity gates pass. A finite scalar is necessary but not sufficient evidence.

## 3. Metric grammar

Metrics should be typed by the readout they consume:

```text
spike/event metrics
rate metrics
state/Vm-like metrics
source metrics
field/readout proxy metrics
spectral metrics
spectrolaminar metrics
cross-condition metrics
regularization/constraint metrics
```

A metric must specify:

- expected shape and axis semantics;
- units/status;
- aggregation over time/trials/channels/seeds;
- finite-value policy;
- target convention;
- whether differentiability is required.

Shared metrics belong in package registries/APIs, not duplicated in tutorials.

## 4. Composition

For metrics `m_j` and nonnegative or explicitly signed weights `w_j`, a composite objective may be

\[
L(\Theta)=\sum_j w_j\,\ell_j(m_j(Y_\Theta),t_j)+R(\Theta),
\]

but every term must remain inspectable in the report. Do not expose only a total score when interpretation depends on multiple components.

Normalization must be explicit. Avoid hidden rescaling that makes objective values incomparable across conditions.

## 5. Gates precede interpretation

Recommended order:

```text
simulate
-> finite/shape/unit-status checks
-> readout validity gates
-> metrics
-> null/control checks
-> objective score
-> optimizer update
-> validation report
```

If a gate fails, report the failure; do not silently coerce the run into a valid score.

## 6. Nulls and falsification

Every publication-facing objective should have at least one meaningful null/control appropriate to the claim. Examples:

- shuffled layer labels;
- shuffled cell-type labels;
- HDP weight-update null (`N_W^{HDP}`), RBS-dynamics null (`N_H`), or
  full-system null (`N_{\mathrm{system}}`) as defined in
  `4_tfne_theory_and_neural_tensor.md` §8.4; never infer one from
  `K_HDP=0` alone;
- source-map ablation;
- field/readout operator ablation;
- stimulus-target shuffle;
- repeated-seed null distribution;
- parameter recovery from synthetic ground truth.

Nulls must test the proposed explanatory structure, not merely produce a worse-looking figure.

## 7. Optimizer grammar

An optimizer is a bounded proposal/update operator

\[
A:(\Theta,L,\mathcal C,K)\rightarrow\Theta',
\]

with explicit search space, constraints, budget, seed/key, and stopping criterion.

Optimization evidence must record:

```text
initial parameters
bounds/constraints
optimizer name/version
budget/steps
PRNG seed/key policy
objective components
best parameters
best score
termination reason
failure/rejection count
```

Do not call successful numerical optimization biological learning.

## 8. Gradient-based and search-based paths

Where differentiable:

\[
\Theta_{k+1}=\Pi_\mathcal{C}(\Theta_k-\alpha_k\nabla_\Theta L).
\]

Where gradient-free, the same objective report contract applies. AGSDR or any other search procedure is an optimization strategy, not a scientific validation layer.

Gradient checks should compare autodiff with finite differences on small deterministic problems before relying on gradients for publication claims.

## 9. RBS/RBD/HDP objectives

For kernels with Relative Biophysical State (RBS) and optional Hidden-state
Dependent Plasticity (HDP), distinguish at least:

1. regulation/stability objectives;
2. adaptation/recovery objectives;
3. omission/oddball response objectives;
4. parameter-identification objectives.

Do not optimize only a final mean if the claim concerns temporal stability. Include trajectory-level diagnostics such as boundedness, overshoot, recovery time, steady-state error, and weight/H excursions.

## 10. Acceptance tests

Publication-facing objective code should satisfy:

- deterministic repeatability under identical seed/config;
- finite outputs;
- explicit axes and units/status;
- null/control behavior;
- component-wise report visibility;
- optimizer bounds respected;
- no mutation of scientific inputs hidden outside the manifest;
- JSON-safe report serialization.
