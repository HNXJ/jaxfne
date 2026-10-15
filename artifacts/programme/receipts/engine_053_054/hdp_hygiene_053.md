# 0.5.3 ENGINE item 7b receipt — HDP parameter hygiene (W2)

Branch `w2-53`, base `5d679a19373648624b69c1f94a5e5c02e3f29a2d`.
Authority: H7 (dead-param classification), H5 (adversarial tests),
`docs/doctrine/rbs_rbd_hdp.md`, skills jaxfne-core / jaxfne-repo / jaxfne-science.

## H7 classification (observed, then fixed or explicitly classified)

| # | Location | Finding (observed) | Classification | Action |
|---|---|---|---|---|
| 1 | `validate_hdp_params_semantics`, non-dict input, non-strict | returned `[]` (silent pass) | defective | FIXED: returns `[msg]` in both modes; strict raises |
| 2 | `compile_step_fn **hdp_kwargs`, unknown keys | `TypeError` from inside the first step (late, keyless message) on pass-through paths | defective fail-closed UX | FIXED: `reject_unknown_hdp_kwargs` at entry, `ValueError` listing keys |
| 3 | registrable `hdp_rule_params` merge (`{**defaults, **given}`) | unknown rule-param keys silently inert (`.get()` reads) | defective | FIXED: `check_hdp_rule_params` per-descriptor; wired in `compile_step_fn` |
| 4 | `hdp_rule_params` vs `KNOWN_HDP_PARAM_KEYS` | execution consumes it; validator flagged it unrecognized (false positive) | defective validator | FIXED: added to `H_DYNAMICS` group; contract artifact regenerated |
| 5 | validator node-rule check vs registered rules | `synthetic_presyn_gain` etc. flagged "must be one of ..." (false positive) | defective validator | FIXED: registered names accepted via registry |
| 6 | legacy HDP gain keys + registered rule in `compile_step_fn` | inert (registered branch takes only rule/params/trace) | ignored-by-design | KEPT, documented in `reject_unknown_hdp_kwargs` docstring |
| 7 | `_hdp_kernel_kwargs` allowlist projection (Model path) | unknown keys dropped before `compile_step_fn` sees them | compatibility-only projection | KEPT for 7b; item-5 controls validate before relying (see below) |
| 8 | direct kernel calls (`simulate_edge_recurrent_izhikevich_hdp*`) | explicit signatures: unknown keys `TypeError` (fail closed, generic) | fail-closed, terse | KEPT; validated entry points are `compile_step_fn` + Model path via item-5 helper reuse |
| 9 | `hdp_rule_params` + legacy rule | `TypeError` inside step | misplaced | FIXED: refused at entry with direction |

Call-site-owned internals (`drive_schedule`, `dtype`, `init_state`,
`silence_mask`, `noise_schedule`, `step_indices`) are rejected at the
`compile_step_fn` boundary: they would be silently overridden per step.

## Changes

- `jaxfne/hdp_rule.py`: `check_hdp_rule_params`, `reject_unknown_hdp_kwargs`.
- `jaxfne/public_surface.py`: non-dict fail-closed both modes;
  `hdp_rule_params` known; registered rule names accepted.
- `jaxfne/_pipeline.py`: `compile_step_fn` entry validation.
- `artifacts/public_surface_contract_v0413.json`: regenerated via
  `scripts/generate_public_surface_contract.py` (repo-mandated).
- `tests/test_hdp_hygiene_053.py`: 28 adversarial tests.

## Verification

- New: `tests/test_hdp_hygiene_053.py` — 28 passed.
- Affected existing: contract + hdp01 + rec01 + continuation_contract +
  pipeline_pure_functions + util_config_tensor — 91 passed;
  hdp_dispatch + hdp_delayed_registrable + law01 + equiv01 — 31 passed.
- Invariants: `d_H=1` default untouched (no rule requires `d_H>1`);
  K_HDP=0 null path untouched; H≠HDP separation untouched.

## Residual (for item 5 / dispatcher)

- R1: Model-path `_hdp_kernel_kwargs` projection still drops unknown keys
  (row 7). Item-5 controls call `reject_unknown_hdp_kwargs` before relying
  on params; closing the projection itself is dispatcher follow-up.
- R2: `step_indices` user-pass-through is refused; delay callers use the
  `compile_step_fn` delay path (verified by rec01/delayed suites above).
