# jaxfne Source Code Review — notebook Expert Audit (2026-06-02)

## Scores

| Module | Score | Key Finding |
|--------|------:|-------------|
| `core.py` | 75 | God object — split schemas, DRY config chaining |
| `emitters.py` | 85 | Duplicate E/PV/SST/VIP param blocks — centralize lookup |
| `fields/proxy.py` | 78 | Manual CSD stencil — use 1D convolution; move diagnostics out |
| `optim/agsdr.py` + `gsdr.py` | 65 | Near-identical — merge into shared base class |
| `fields/probes.py` | 88 | Duplicate probe wrappers — generalize factory |
| `builders.py` | 85 | Hardcoded defaults drift — centralize constants |
| `objectives.py` | 90 | Null distribution generators — refactor to higher-order function |
| `bridges.py` | 92 | Manual to_dict — use dataclasses.asdict + json_safe |

## Priority Actions

### P0 — Merge agsdr.py + gsdr.py (score 65, highest impact)
- Create `BaseGSDRState` with shared tracking fields
- AGSDR extends with adaptive-specific fields only

### P1 — Centralize Izhikevich cell-type constants (emitters.py, score 85)
- Single dict constant or `get_izhikevich_constants(cell_type)` lookup
- Remove duplicate if/elif blocks in `izhikevich_eig_params` and `izhikevich_params_from_labels`

### P1 — Centralize cell-type/layer defaults (builders.py + core.py, score 85)
- Move hardcoded fractions to `constants.py`
- Both builders.py and core.py reference from there

### P2 — CSD stencil optimization (fields/proxy.py, score 78)
- Replace manual slicing with `jax.scipy.signal.convolve` [1, -2, 1] kernel
- Move `validate_projection_invariants` and `_make_field_solution_report` to `fields/diagnostics.py`

### P2 — Generalize probe factory (fields/probes.py, score 88)
- Single `create_probe(kind, data, method, assumptions)` replaces spk_probe/vm_probe/source_probe

### P3 — core.py split (score 75, largest refactor)
- Extract schemas to `specs.py`
- Extract Configuration DSL to dedicated builder
- Consider persistent data structures for config chaining

### P3 — Null distribution refactor (objectives.py, score 90)
- Higher-order function accepting mutation lambda

### P3 — bridges.py cleanup (score 92)
- Use `dataclasses.asdict()` + `json_safe` in JaxleyTraceSpec

## Status
- Review date: 2026-06-02
- Reviewer: notebook expert
- Applied: pending
- Target version: 0.3.28+
