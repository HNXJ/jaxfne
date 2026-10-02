# 0.5.3 ENGINE item 6 receipt — causal intervention grammar (W2)

Branch `w2-53` (after item-5 commit `51f8c9b`). Authority: H4
(serialization boundary), H5 (adversarial), H8 (`d_H=1`; RBS !=
homeostasis — interventions target plasticity, never relabel H).

## Object

`jaxfne/intervene.py` (~330 lines): `Intervention{name, network, control,
observation}` — one declarative object holding the identical realized
network (closed builder set `suite2_net1`/`hdp_column` + build/run spec +
`hdp_params`, digest-pinned), exactly one mechanism (`plasticity` via the
item-5 controls; anything else refused), and the declared observation
difference (signal `w_final`/`H_final`/`spike_count` × subset
`all`/`clamped`/`unclamped` × expect
`equal`/`different`/`within`/`exceeds` + tolerance).

`run_intervention`: validates baseline params (7b), builds both arms from
the same spec, verifies identical realization (spec digest + realized
edge-weight equality, else raises), runs baseline vs controlled arm
(disable via gain-zeroing; clamp via mask + `with_hdp_initial_state`
pinned `w0`), observes both, verdicts PASS/FAIL. Subset
clamped/unclamped needs signal `w_final` and a control mask (fail closed).

## Manifests (H4)

`to_manifest` is JSON-safe; `intervention_from_manifest` roundtrips
exactly (strict keys at every level; tested: object == restored,
manifest == re-emitted). In-memory verdict and serialized record tested
separately: the result carries the live verdict; both arm manifests carry
`manifest["intervention"]` (arm + full object + digests). Surface change:
one optional `intervention` kwarg on `Model.manifest` (the `trials`
idiom); manifests without interventions are byte-unchanged (tested:
`"intervention" not in model.manifest(sig)`).

## Verification

- New `tests/test_intervention_grammar_053.py`: 16 passed — disable
  differs PASS, all-plastic clamp equal PASS + misdeclared FAIL (verdicts
  can say no), clamp-subset differs PASS with pinned-exactness,
  hdp_column builder runs, arm-manifest additive content, plain-manifest
  absence, JSON roundtrip, result-record preservation, 8 refusal cases
  (second mechanism, extra keys builder/manifest-side, unknown builder,
  subset-without-mask, mask-length mismatch pre-run, baseline mask,
  clamp-without-mask/NaN).
- Existing manifest suites: manifest_v005 + readout_compat + trial_runner
  + api_smoke, 45 passed.
- Invariants: `d_H=1` default (no `h_state_dim` anywhere); K_HDP=0 null
  and H≠HDP untouched; no `docs/` or signature breaks.

## Carried (not added)

- Multi-mechanism interventions, H-perturbation targets (ATLAS item 9
  uses this grammar's shape), population-locality masks (refused, item 5
  boundary), non-mean statistics — all refused loudly, none synthesized.
