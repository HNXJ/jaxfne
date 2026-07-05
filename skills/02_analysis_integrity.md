# Analysis Integrity Skill

## Purpose
Prevent fabricated diagnostics from masking upstream failures.

## Rules
- Never synthesize a fallback spectrum, trace, summary, or diagnostic that can be mistaken for a real result.
- If a computation fails in strict mode, raise immediately.
- If synthetic output is intentionally requested, label it synthetic in the API name, docstring, and rendered output.
- Separate data preparation from rendering so plotting code cannot silently repair broken analysis.
- Do not use random data as a hidden fallback for failed analysis.

## Acceptance checks
- A failed upstream analysis cannot produce a visually valid but false diagnostic.
- Strict-mode tests fail loudly when spectral or trace computation fails.
- Every synthetic fallback is explicit and opt-in.

## Verified case in this repo (2026-07-05)

`jaxfne/sanity_runtime.py::_make_plasticity_metrics`'s disabled-plasticity
branch returned hardcoded stats (`pre_weight_max: 0.08`, etc.) formatted
identically to genuinely-computed values in `plasticity_report.json`, with no
field distinguishing "not measured" from "measured" — a real violation of
this rule (a failed/skipped analysis produced a report indistinguishable from
a real one). Fixed by adding an explicit `placeholder_values: bool` field
(`True` on the not-measured path, `False` when computed from real weight
arrays) rather than nulling the numeric fields, since no existing test or
consumer expected `None` there. Verified it genuinely discriminates: disabled
path -> `placeholder_values=True`; enabled path with real weight arrays ->
`placeholder_values=False`.

Separately, `jaxfne/sanity_delta.py::TaskEpisode.validate()`'s `"strict_json"`
check was `results[check] = True` unconditionally -- a rubber-stamp that
never actually checked anything, the same failure family (a check that
always reports success regardless of whether the underlying condition
holds). Fixed by attempting a real `json.dumps(..., allow_nan=False)` on the
task schedule plus a sampled vm/spikes payload, verified to genuinely
discriminate (NaN in vm -> `False`, all-finite -> `True`).
