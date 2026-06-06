# JAXFNE Equations, Runtime, and Validation

## Mathematical glossary flow

For technical docs and tutorials, show:

```text
formal equation -> term definitions -> worded mechanism -> implementation location -> evidence/status boundary
```

## Core tensor equations

Pipeline:

```math
X_{emitter} \rightarrow S_{source} \rightarrow F_{field} \rightarrow Y_{probe} \rightarrow L_{objective} \rightarrow \theta_{trainer}
```

Source projection:

```math
S_k(t)=\sum_n A_{kn}g_n(t)
```

Linear readout:

```math
Y(t)=S(t)W^T
```

Extracellular baseline identities:

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
- `vmap` for batches/candidates.
- `pmap`/`pjit` only after FlatNet arrays and static maps are separated.
- JIT numerical hot paths only.
- Plotting, JSON, file I/O, markdown, pandas, and Python object mutation stay outside JIT.
- `dtype=float32` is the release-facing default unless a tutorial teaches precision.

## FlatNet JIT contract

Split runtime inputs into:

```text
runtime_static: Python/static metadata marked static or captured outside JIT
runtime_dynamic: JAX arrays and scalar arrays that can be traced
```

FlatNet arrays:

```text
neuron_params: [N, P]
positions: [N, 3]
area_id/layer_id/cell_type_id: [N]
edge_pre/edge_post/edge_weight/edge_mechanism: [E]
probe_positions or readout weights: [C, ...]
```

Tracking maps stay static:

```text
row_to_quartet
global_id_to_row
area_vocab
layer_vocab
cell_type_vocab
mechanism_vocab
edge_rule_origin
```

## Layout rules

Use explicit layout names:

```text
time_node
node_time
trial_time_node
node_trial_time
trial_area_time_contact
area_trial_contact_time
freq_contact
contact_freq
```

Layout conversion is named-axis transposition. It must fail loudly when the source layout lacks a requested axis.

## Optimizers and trainers

AGSDR and related methods are computational search methods under declared objectives. Reports include:

```text
search space
seed
budget
objective output names
best score
best parameters
gates
rejection reasons
finite status
status metadata
```

Optax is valid only for differentiable or declared-surrogate paths. Hard spiking resets require explicit differentiability status.

## Minimal schemas

Config minimum:

```yaml
schema_version: string
runtime: object
geometry: object
circuit: object
probes: object
paradigm: object
objective: object
optimizer: object
metadata: object
```

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
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
```

Notebook releases also need:

```bash
PYTHONPATH=. TFNE_SMOKE=1 jupyter nbconvert --to notebook --execute <notebook.ipynb> --output /tmp/notebook_smoke.ipynb
PYTHONPATH=. TFNE_SMOKE=0 jupyter nbconvert --to notebook --execute <notebook.ipynb> --output /tmp/notebook_full.ipynb
```
