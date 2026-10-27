# Tensor Operator Registry

Operators behind the [TFNE Operator Doctrine](../../../theory/operator_doctrine.md),
grouped by pipeline stage and grounded in the runtime registry
`jaxfne.operator_status()`.

## The live registry

```python
import jaxfne as jtfne
jtfne.operator_status()
```

returns:

```python
{
    "E_theta": "prototype_api",
    "S_WDR": "prototype_api",
    "C_mu_nu": "not_implemented",
    "Q_eta_alpha": "prototype_api",
    "F_field": "prototype_api",
    "P_probe": "prototype_api",
    "A_objective": "prototype_api",
    "O_optimizer": "prototype_api",
    "C_constraints": "prototype_api",
}
```

Every value is one of two statuses today: `"prototype_api"` (implemented,
preliminary interface) or `"not_implemented"` (declared, no implementation). See
[Validation: Operator Status](../core/validation.md#operator-status) for the full
status-value reference.

### Symbol mapping

Four symbols map directly onto pipeline stages and are demonstrated in
[Validation: Operator Status](../core/validation.md#operator-status):

| Symbol | Stage |
|---|---|
| `E_theta` | Emitter |
| `S_WDR` | Source |
| `F_field` | Field |
| `P_probe` | Probe |

Three more are name-evident. The English word in the symbol names the stage:

| Symbol | Stage |
|---|---|
| `A_objective` | Objective |
| `O_optimizer` | Optimizer |
| `C_constraints` | Constraints (cross-cutting; not a pipeline stage) |

Two symbols have no confirmed stage mapping in any docstring, rule, or
existing documentation page as of this writing; they are listed here rather
than force-mapped:

- **`C_mu_nu`**: `not_implemented`. Tensor-index naming (`mu`, `nu`) is
  consistent with a rank-2 coefficient (e.g. a conductivity-tensor term in an
  elliptic field solve), but no source file states that correspondence. Do not
  treat this as confirmed.
- **`Q_eta_alpha`**: `prototype_api`. No confirmed stage mapping.

---

## Operators by pipeline stage

This table lists the concrete functions behind each stage. The abstract
registry symbols above are a separate layer; the rows here are the actual callables `jaxfne.__all__` exports.

### Source operators

- `construct_source_tensor(*, mode, ...)`: builds the declared source tensor from state/drive.
- `synaptic_current_tensor(...)`, `synaptic_tau_from_mechanism(...)`, `synaptic_tensor_report(...)`: mechanism-resolved synaptic current and its per-edge tau.

### Field operators

- `project_laminar_sources(sources, positions, *, n_contacts, width)`: source-to-field projection (`FieldOutput`); the declared, unsolved field equation.
- `csd_tensor(...)`: the named current-source-density spatial-derivative operator.

### Probe operators

The eight readout operators are documented in full in [Probe Operators](../../../guides/readout/probe_operators.md):
`SPK`, `Vm`, `Source`, `LFP-proxy`, `CSD-proxy`, `EEG-proxy`, `MEG-proxy`, `EMM-proxy`.
Lead-field transforms: `eeg_proxy_transform(...)`, `meg_proxy_transform(...)`, `emm_proxy_transform(...)`.

### Connectivity operators

- `compile_connection_rules(...)`: compiles `.connections()` declarations into edges; see [Configuration Grammar](../../../guides/grammar/configuration_grammar.md#connectivity) for the sign-only vs mechanism-aware compiler split. Each rule's `sampling` selects how `probability` is realized: `"density"` (default) keeps each compiler's existing draw (mechanism-aware: exact count `round(p*n_pre*n_post)`), while `"bernoulli"` includes each (pre, post) pair independently with probability `p` (tiled past 4M candidate pairs, so no dense mask is materialized; refused with `max_in_degree`).

### Paradigm operators

- `general_sequential_oddball_paradigm(...)`: the backbone for omission/global/local/explicit-event-list paradigms.
- `general_delayed_match_to_sample_paradigm(...)`.

### Objective operators

- `rate_targets(groups, targets_hz, weights)`, `rate_synchrony_targets(target_rate_hz, target_kappa_synchrony, ...)`.

### Optimizer operators

- `agsdr(...)`, `gsdr(...)`, `gsgd(...)`, `optax_adam(...)`, `random_search(...)`.

### Validation / bookkeeping operators

- `validate_source_field_status(...)`, `compute_conservation_proxy_diagnostics(...)`, `is_valid_signal(signals)`.
- `manifest(cfg, signals=None, ...)`, `run_receipt(model, signals, *, tags=...)`.
- `config_truth_boundary(cfg)`: **REMOVED (2026-06-30).** Deleted along with
  the rest of the `JaxFNEConfig` config-path API; see [Validation](../core/validation.md)
  for the replacement pointer (status-gate metadata now lives on the fluent
  pipeline's own objects, e.g. `FieldOutput`/`RunReceipt`; no standalone call is involved).

---

## Report shapes

Every report-producing function above returns a plain JSON-safe `dict`; no
typed dataclass is returned. This is a deliberate, repo-wide convention. Reports must
round-trip through `json.dumps(..., allow_nan=False)` for write-once receipts,
rather than a gap to fill. `jaxfne/experimental_hpc/contracts.py` incubates a
future typed-tensor architecture; every dataclass there raises
`NotImplementedError(">TBI-not-ready")` and is explicitly fenced as experimental,
not a parallel report format in current use.

A representative report shape, from [Limitations and future plans](../../project/limitations_and_future_plans.md)
and [Validation](../core/validation.md):

```python
{
    "claim_level": "computational_scaffold",
    "field_solver_status": "linear_solver",
    "physical_amplitude_calibrated": False,
    ...
}
```

---

## Composability

`tests/test_operator_stage_coverage_v04.py` already exercises the full chain:
config build, construct+simulate, source tensor, field projection, probe
readout, objective evaluation, optimizer tuning, manifest/receipt export,
validation report, and schema migration: checking finiteness and rule
fields at every stage. That end-to-end coverage is the composability evidence
for this registry; this page does not duplicate it.

---

## See Also

- [TFNE Operator Doctrine](../../../theory/operator_doctrine.md): the per-stage contract table this registry grounds
- [Validation](../core/validation.md): `operator_status()`, `is_valid_signal()` (`config_truth_boundary()` REMOVED, see note above)
- [Probe Operators](../../../guides/readout/probe_operators.md): the eight readout operators in full
- [Configuration Grammar](../../../guides/grammar/configuration_grammar.md): the fluent/operator grammar reconciliation
- [Objective Grammar](../../../guides/grammar/objective_grammar.md): the user-facing run sequence, stage by stage
- [Operator Inventory (generated)](../../../_generated/operator_inventory.md): every export, regenerated from `jaxfne.__all__`
