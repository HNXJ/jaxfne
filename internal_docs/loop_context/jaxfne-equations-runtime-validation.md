<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# JAXFNE Equations, Runtime, and Validation

## Mathematical glossary flow

Every technical equation in tutorials, docs, figures, and technical report text follows:

```text
formal equation -> term definitions -> worded equation -> implementation location -> evidence/status boundary
```

## Core tensor equations

```math
X_emitter -> S_source -> F_field_proxy -> Y_probe -> L_objective -> theta_optimizer
Y = P o F o S o E
L = O(Y, target, gates, manifest)
theta_next = A(theta, L, constraints, key)
S_k(t) = sum_n A_kn g_n(t)
Y(t) = S(t) W^T
Phi[k,c] = sum_n S[k,n] W[c,n]
CSD_like = D_zz Phi
```

Quasi-static physical baseline reserved for future solver evidence:

```math
J_e = -sigma_e grad(phi_e)
div(J_e) = q
div(-sigma_e grad(phi_e)) = q
CSD = div(J_e)
```

Current tutorials keep:

```yaml
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

## JAX runtime discipline

| Rule | Required behavior | Stop condition |
|---|---|---|
| Arrays | numerical kernels use JAX arrays and `jax.numpy` | NumPy math in hot kernels without reason |
| PRNG | explicit keys and deterministic seeds | hidden randomness or missing seed |
| Time | `jax.lax.scan` for hot time loops | Python loop in traced hot path |
| Batches | `jax.vmap` for seeds/candidates/readouts when practical | manual loop where vectorization is intended |
| JIT | pure numerical hot paths only | plotting, file I/O, JSON, markdown, or Python object mutation inside JIT |
| Timing | warmup separately and block before timer stop | async-dispatch timing interpreted as runtime |
| dtype | `float32` default; x64 opt-in | silent float64 drift |
| CPU baseline | CPU correctness first | GPU-only claim without CPU baseline |

## Optimizer status

GSDR/AGSDR/random-search are computational search methods over declared objectives. They are not biological learning rules or mechanism proof.

Reports include:

```yaml
optimizer_name: string
search_space: object
seed: int
budget: int
objective_name: string
hard_gates: list
best_score: number_or_null
best_parameters: object_or_null
rejection_reasons: list
finite_outputs: bool
truth_mode: truth_safe_unverified
```

Optax is optional and valid only for differentiable or declared-surrogate paths. Hard spiking reset paths require explicit differentiability status.

## Minimal manifest schema

```yaml
run_id: string
jaxfne_version: string
repo_sha: string_or_null
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
source_calibration_status: string
field_solver_status: laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
runtime_report: object
artifact_paths: object
asset_hashes: object
```

## Minimal validation report schema

```yaml
finite_outputs: true
strict_json_pass: true
png_figures_present: true
notebook_execution: pass_or_not_applicable
truth_gates_preserved: true
optional_dependency_laziness: pass_or_environment_contaminated_or_fail
```

## Validation ladder

```bash
python3 -m json.tool docs/evidence_artifacts/evidence_checklist.json >/dev/null
python3 -m compileall -q jaxfne scripts/evidence_figures tests
python3 scripts/evidence_figures_inventory.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/ -q --tb=line
python3 scripts/audit_notebooks_and_assets.py --check
python3 -m mkdocs build --strict
```

Notebook releases also need smoke and full execution receipts:

```bash
TFNE_SMOKE=1 nbclient <notebook.ipynb>
TFNE_SMOKE=0 nbclient <notebook.ipynb>
```

## Solver ladder

| Level | Name | Evidence |
|---|---|---|
| P0 | Proxy projection | finite arrays, shapes, proxy labels, JSON reports |
| P1 | Declared lead-field-like projection | derivation, assumptions, source/probe geometry metadata, fixed operator |
| P2 | Boundary-normalized kernels | normalization, conservation-style checks, boundary metadata, sign convention |
| P3 | Discrete volume-conductor solve | `K Phi = Q`, boundary, gauge, residual, convergence, finite `Phi/J/CSD` |
| P4 | Differentiable adjoint solve | VJP/adjoint checks, gradient tests, JIT/VMAP checks |
| P5 | External validation | comparison to named tool/reference or empirical reference with limits |

Current v0.3.x default is P0.
