# Tree-budget amendment, 2026-10-23 (Hamm-approved)

`budget_bytes` raised from 90,177,536 to 90,205,000 (+27,464 bytes; +0.0305 % of
the old ceiling). `budget_files` unchanged (2559; 2557 tracked).

## Measured justification

The B3 seed/candidate batching feature (T-32c, release plan 0.5.8 package B3)
and its mandated falsification test set (Hamm ruling 2026-10-23, "APPROVE
option (a), conditional on verification"):

| File | Bytes |
|---|---|
| `jaxfne/_model_simulate.py` (AdEx + LIF seed-batch entries, shared validation) | +12,883 |
| `jaxfne/_model_tune.py` (the by-name candidate-batching refusal) | +591 |
| `jaxfne/emitters_lif.py` (H0 finiteness check, trace-safe) | +296 |
| `pyproject.toml` (the `bitwise_diagnostic` test marker) | +120 |
| `tests/test_adex_route.py` (contract + falsification) | +10,330 |
| `tests/test_lif_tensor_route.py` (contract + falsification) | +14,373 |
| `tests/test_optim_tune.py` (tune refusal + history completeness) | +1,479 |
| **Total B3 delta** | **+40,072** |

Slack before the B3 commits was 19,958 bytes; the interim checklist commits
(`664afef9`, `3122ba85`) consumed 1,389 bytes, giving the reported 21,503-byte
overage — reconciled exactly: 19,958 − 1,389 − 40,072 = −21,503.

Composition: irreducible implementation plus the ruling's required
falsification set; no removable duplication, generated files, or misplaced
documentation contributed. No test was deleted, no tolerance weakened, no
scientific semantics changed to fit.

## Authorization

Hamm, 2026-10-23: "RULING: APPROVED — B3 budget amendment ... limited to the
B3 implementation, required falsification tests, and already-staged
`event_order` documentation. The budget increase is not authorization for
additional features or unrelated growth."

This is an engineering constraint record, not a scientific fact; it belongs to
the release-evidence set, not the fact stack.
