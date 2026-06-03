# JAXFNE Equations, Runtime, and Validation

## Mathematical glossary flow

For technical docs and tutorials, show:

```text
formal equation -> term definitions -> worded mechanism -> implementation location -> evidence/status boundary
```

## Core equations

Pipeline:

```math
X_{emitter} \rightarrow S_{source} \rightarrow F_{field} \rightarrow Y_{probe} \rightarrow L_{objective} \rightarrow \theta_{optimizer}
```

Source projection:

```math
S_k(t)=\sum_n A_{kn}g_n(t)
```

Linear readout:

```math
Y(t)=S(t)W^T
```

Extracellular baseline:

```math
\mathbf{J}_e=-\sigma_e\nabla\phi_e
```

```math
\nabla\cdot\mathbf{J}_e=q
```

```math
\mathrm{CSD}=\nabla\cdot\mathbf{J}_e
```

Proxy tutorials keep `field_solver_status=laminar_proxy_no_pde`.

## JAX runtime rules

- JAX arrays for numerical kernels.
- Explicit PRNG keys.
- `lax.scan` for time stepping.
- `vmap` for batches/candidates when useful.
- JIT numerical hot paths only.
- Plotting, JSON, file I/O, markdown, and Python object mutation stay outside JIT.
- `dtype=float32` is the release-facing default unless a tutorial teaches precision.

## Optimizers

GSDR/AGSDR are computational search methods under declared objectives. Reports include search space, seed, budget, objective name, best score, gates, rejection reasons, finite status, and truth mode.

Optax is valid only for differentiable or declared-surrogate paths. Hard spiking resets require explicit differentiability status.

## Minimal schemas

Manifest minimum:

```yaml
run_id: string
jaxfne_version: string
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
source_calibration_status: string
field_solver_status: laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
runtime_report: object
artifact_paths: object
asset_hashes: object
```

Validation report minimum:

```yaml
finite_outputs: true
strict_json_pass: true
png_figures_present: true
notebook_execution: pass_or_not_applicable
truth_gates_preserved: true
```

## Validation gates

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
```

Notebook releases also need:

```bash
TFNE_SMOKE=1 nbclient <notebook.ipynb>
TFNE_SMOKE=0 nbclient <notebook.ipynb>
```

## Solver ladder

| Level | Name | Evidence |
|---|---|---|
| P0 | Proxy projection | finite arrays, shapes, proxy labels, JSON reports |
| P1 | Lead-field-like projection | derivation, assumptions, smoothing, tests |
| P2 | Boundary-normalized kernels | normalization, conservation, boundary metadata |
| P3 | Discrete volume-conductor solve | `K Phi = Q`, residual, boundary, gauge, convergence |
| P4 | Differentiable adjoint solve | VJP/adjoint, gradient checks, JIT/VMAP checks |
| P5 | External validation | comparison to named tool/reference with limits |