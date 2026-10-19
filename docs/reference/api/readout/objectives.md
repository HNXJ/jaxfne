# Objectives API

Objectives define numerical targets for `Model.tune`.

## Group firing-rate targets

```python
objectives = jtfne.rate_targets(
    groups={"first_half": range(24), "second_half": range(24, 48)},
    targets_hz={"first_half": 5.0, "second_half": 10.0},
    burn_in_ms=100.0,
)
```

`burn_in_ms` defines the start of the rate-measurement window; the optional
`window_end_ms` argument defines its end, with the simulation endpoint as the
default. The report exposes the achieved rate, target rate, rate loss, and
measurement window.

## AGSDR optimizer spec

AGSDR = **Adaptive Genetic Stochastic Delta Rule** — the canonical black-box
optimizer for `Model.tune` (multi-parameter engine:
`jaxfne.optim._run_agsdr_optimization_loop`; scalar engine:
`propose_blackbox_candidates`).

```python
optimizer = jtfne.agsdr(
    parameters={"drive_scale_a": (0.35, 2.25), "drive_scale_b": (0.35, 2.25)},
    generations=8,
    population_size=6,
    seed=42,
)
```

For an edge-list simulation, grouped synaptic magnitudes use the executable
`EdgeList.weight` storage directly:

```python
parameters = {
    "m_EE": jtfne.edge_parameter(
        pre={"cell_type": "E"}, post={"cell_type": "E"}, bounds=(0.1, 5.0)
    ),
    "m_EI": jtfne.edge_parameter(
        pre={"cell_type": "E"}, post={"cell_type": "PV"}, bounds=(0.1, 5.0)
    ),
}
```

Each selected edge receives the declared positive magnitude while retaining
the sign already carried by that executable edge. `EdgeParameterSpec` also
accepts `layer`, `area`, `ids`, `receptor_indices`, and explicit
`edge_indices` constraints through the shared selector grammar. Use
`MatrixParameterSpec` for dense backends that consume `emitter.W`.

## AGSDR surface classification

Every AGSDR-labelled callable has a declared relationship to the canonical
definition above. The authoritative source is
`jaxfne.optim.AGSDR_SURFACE_CLASSIFICATION`:

| Class | Members | Relationship |
|---|---|---|
| **CANONICAL** | `agsdr()`, `AGSDROptimizerSpec`, `OptimizerSpec`, `_resolve_optimizer`, `_run_agsdr_optimization_loop`, `_agsdr_candidates_from_noise` | The multi-parameter `Model.tune` AGSDR engine (population + delta-rule center update + elitist anchor). |
| **EXPERIMENTAL** | `agsdr_transform` | Optax `GradientTransformation` variant; NOT on the `Model.tune(optimizer="AGSDR")` path; exercised only by tests. |
| **COMPATIBILITY** | `AGSDR`, `AGSDRState`, `step_agsdr_transform`, `propose_blackbox_candidates` | Legacy surfaces retained for old notebooks/tests. `AGSDRState`/`step_agsdr_transform` are documented misnomers: they back a plain gradient-descent step and do not implement canonical AGSDR. `propose_blackbox_candidates` is the legacy single-parameter two-phase shrinking-radius search — related to the canonical engine, rather than equivalent to it (fixed center, no population, no delta-rule update). |
| **HYBRID** | `_tune_matrix_agsdr_optax` | Matrix/two-level path: canonical AGSDR outer proposals with an inner Optax (Adam) refinement surrogate; NOT mathematically identical to canonical AGSDR. |
| **TUTORIAL** | `tune_laminar_agsdr` | Tutorial convenience wrapper; not the canonical engine. |

## GSDR with MCDP

`jaxfne.optim.gsdr` carries module-local MCDP support for the Fig. 6 lane:

- `mcdp_factors(v_traces, pre, post, shift_steps, smooth_k=None)` computes
  per-edge squared Pearson correlations of pre voltages against post voltages
  shifted by `shift_steps`, with optional normalized Gaussian smoothing over
  edges (`smooth_k`) applied before z-scoring. Edges with no variance read
  `1.0`. The no-smoothing form reproduces `run_gsdr.mcdp_factors` at
  `shift_steps=10`.
- `gsdr_mcdp_transform(...)` returns an Optax-compatible
  `GradientTransformation` combining an inner SGD step
  (`learning_rate=0.1`, `momentum=0.4`) with an MCDP-gated stochastic delta
  (`stochastic_scale` sets its magnitude) scaled by `lambda_d`, mixed by a
  weight `a` (`a_init`, uniform random walk of half-width `a_step` when
  `a_dynamic`). The walked `a` is kept only when the step improved on the
  best loss, else the accepted `a` is kept. `w_max` clamps the update so
  `params + update` stays in `[0, w_max]` (`None` means no clamp).
  Selection tracks the best loss and parameters; a C-threshold
  revert (`deselection_threshold`), a stagnation reset (`checkpoint_n`) and a
  non-finite-loss revert each return `best_params - params` deltas. `update`
  requires an explicit `key`; `mcdp=None` voids the epoch with zero updates.

The 350/500-generation Fig. 6 fits ran without MCDP; their manifests record
`mcdp_applied: False` with the authors' settings under
`authors_optimizer_settings_not_used` (P-039; erratum in the lane
`CALIBRATION.md`).

## Tune

```python
result = model.tune(objectives=objectives, optimizer=optimizer)
print(result.best_score)
print(result.best_parameters)
print(result.summary)
```

For a grouped rate objective, `result.summary` retains compact candidate
reports with `rate`, `target_rate`, `rate_loss`, `weight_regularizer`,
`H_regularizer`, `invalid_status`, `total_score`, and hashes/statistics for
initial and terminal adaptive weights. Candidate failures are represented as
rejected scores and do not terminate the search.

## Single objective

```python
objective = (
    jtfne.objective()                     # fresh empty Objective
    .loss(name="rate", metric="spike_rate_hz", target=10.0)
)
result = model.tune(objective=objective, parameter="drive_gain", bounds=(0.5, 2.0))
```

## Rate + synchrony targets

`rate_synchrony_targets` builds an `Objective` with a population-rate term and a
synchrony (kappa) term, defaulted to the canonical balanced operating point;
`jtfne.rate_synchrony_targets()` is the standard starting objective for laminar tuning.

| Parameter | Default | Meaning |
|---|---|---|
| `target_rate_hz` | `10.0` | Target population firing rate (Hz). |
| `target_kappa_synchrony` | `0.0` | Target synchrony. `0.0` = asynchronous-irregular; required for an **unbiased spectrolaminar readout** (a global rhythm masks laminar band structure). |
| `rate_weight` | `1.0` | Weight of the rate term. |
| `synchrony_weight` | `0.25` | Weight of the synchrony term. |

**Returns:** an `Objective` usable with `Model.evaluate` / `Model.tune`.

```python
obj = jtfne.rate_synchrony_targets()                  # 10 Hz, kappa 0 (canonical)
obj = jtfne.rate_synchrony_targets(target_rate_hz=5.0, synchrony_weight=0.5)
result = model.tune(obj, optimizer="AGSDR", steps=50)
```

## Result object

`TuneResult` exposes:

- `best_score`
- `best_parameters`
- `history`
- `summary`
- `model`
- `to_dict()`
