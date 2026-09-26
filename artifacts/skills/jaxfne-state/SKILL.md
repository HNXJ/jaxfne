---
name: jaxfne-state
description: Work with the hidden state H (RBS) and its dynamics (RBD) separately from plasticity.
metadata:
  audience: agents
---
# jaxfne state

## WHEN
Setting, perturbing or reading the dependency state `H`, or running RBD with fixed `W`.

## AUTHORITIES
1. `docs/doctrine/rbs_rbd_hdp.md`; `docs/doctrine/tfne_containment_architecture.md`.
2. Live code: `Model.with_hdp_initial_state` (`jaxfne/_model.py`),
   `Model.last_hdp_diagnostics`.

## RULES
- `H` is a finite-dimensional dependency-state container, not homeostasis by definition;
  `d_H = 1` is a valid realization.
- Influence of `H` on E, S, F, P needs a typed coupling map; do not infer one.
- Bounded `H` is not evidence of stabilization without a perturbation/control assay.
- Diagnostics describe the latest run only; capture them per run when a model is reused.

## STEPS
1. Set the initial state: `model.with_hdp_initial_state(H0=..., w0=...)`.
2. Simulate (`jaxfne-simulate`); read `model.last_hdp_diagnostics()` immediately after.
3. For a perturbation, run baseline and perturbed from identical `W0` and stimulation.

## STOP
- Diagnostics read after another run on the same model; an unperturbed claim of stability.

## VERIFY
- Each arm has its own `H_trace` from `last_hdp_diagnostics()`; baseline and perturbed
  arms share `W0`. `with_hdp_initial_state` is inert unless `Configuration.hdp(...)` is set.

## DONE
- H trajectory per arm, captured right after that arm ran, with the assay design stated.
