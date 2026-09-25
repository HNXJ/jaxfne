# Receipt — 0.5.4 ENGINE items 1–5 + ATLAS items 6–8

Doctrine: Atlas source 8 (`artifacts/project_sources/8_atlas.md` S8/S9);
0.5.3 items 1–7b (ownership, budgets, continuation, replay, controls,
interventions, diagnostics). Hierarchy is a scoping device, not a
numerical one; members couple ONLY through explicit cross edges.

## Item 1 — composition operator (`cd7dbca`, `6792bc7`, `dcec360`)

`connect(*models)` (in `jaxfne/_construct_connectivity.py`) was already
structural (block-diagonal edges, concat emitter/positions, cross rules,
namespace, offset_x/keep). 0.5.4 adds what it lacked:

- **Member RNG domains** (`ensemble_member_seed`, public): ensemble run
  with `seed=S` draws member m's unit-noise stream from
  `PRNGKey(ensemble_member_seed(S, m, n))` (sha256 domain tag, no JAX
  dependency). Solo run with the derived seed reproduces the member
  slice bit-exactly (eager + jit). Kernels provably ignore `key` when a
  schedule is passed (plain/delayed/homeostatic/legacy-HDP audited).
  Scope: plain edge_list path, no paradigm, no poisson_drive (global
  streams by design); batch draws global per-replicate streams
  (statistics utility, deterministic, no solo claim).
- **Continuation member streams** (`_pipeline._ensemble_segment_schedule`,
  optional 4th scan `xs`, `run_continuation(..., noise_rows)`): segments
  draw the continuous member rows, so chunked == continuous on ensembles
  AND equals the members' solo trajectories. Non-ensemble paths pass
  None: bit-identical to prior behavior (continuation suites green).
- **Cross-edge dt**: member `dt_ms` reconciled into cross-rule compile,
  so `delay_ms` realizes to steps under the 0.5.2 rule; recorded in
  `metadata["ensemble"]["dt_ms"]`. Previously raised at connect time.
- **S12 per-statement delay** (`tfne.py`): `[delay=MS]` alone or with
  `[mech=...]` in either order; validated at parse (`E_DELAY_OUT_OF_RANGE`
  on negative/non-numeric/duplicate/unknown); overrides the rule default
  per relation; rides the tensor bridge; realizes to steps. Statement
  options remain a closed set (mech, delay). Per-statement geometry has
  no carrier (connections carry no geometry field); area pose geometry
  is untouched — a geometry override would be a new grammar item.
- Deferred with reason: **1b** (S20.1 cell-type ordering changes realized
  order for existing specs — needs individual human authorization);
  **item 0** (X[k]-rule frontier override) was not needed for AT-08/09
  and stays after 0.5.5 per the stack.

Tests: `test_composition_identity_054.py` (3), `test_composition_aspects_054.py`
(10: cross-delay, internal delays, geometry, fields, continuation+solo,
HDP, batch, 3-area chain, middle-member identity, serialization roundtrip).

## Item 2 — hierarchical-flat identity (`e4bd459`, test-only)

Nested program vs flat program (same leaf multiset in the same realized
order, same rules/frontiers) through realize→to_configuration→construct:
identical edges/weights and bit-identical runs. No engine change; the
invariant is locked against compiler drift. (Two COMPILERS,
realize-route vs tensor-route, wire differently — 16 vs 36 edges on the
probe — and are NOT claimed identical; changing that is out of scope.)

## Item 3 — observations (`4bab2f1`)

`population_rate` (mean spikes → Hz, exact, validated) and
`cross_area_coherence` (magnitude-squared coherence + cross-spectrum
phase, Welch with periodogram fallback, silent bins declared
`valid=False` with exact 0.0, never NaN) in `jaxfne.fields` (no root
export, no contract churn). Verified: identical sines cohere ~1, 5 ms
delay reads as −2πf·d phase, independent noise < 0.7, ensemble per-area
rates + multi-area readout + coherence finite.

## Item 4 — cross-area plasticity (`04bbd26`)

`metadata["ensemble"]["member_edge_counts"]` + `ensemble_edge_ownership`
(public): member ranges and per-rule cross ranges (W_12, W_21 separately)
over the merged edge list; mismatch fails closed. Masks from these ranges
scope HDP: cross-only freezes members bit-exactly while all cross edges
move; member-only mirrors; zero mask disables (all frozen exact); replay
bit-identical. Clamp/disable kernel semantics reused from 0.5.3 untouched.

## Item 5 — scaling (`c15cd07`)

`scripts/benchmark_054_scaling.py` → `artifacts/perf/scaling_054.json`:
k=1..4 chained columns, all finite (12/24/36/48 neurons, 132–655 edges).
Frozen 0.5.1 matrix untouched. First-call compile dominates toy timings;
recorded as measurement, not a gate.

## ATLAS items 6–8 (`991a00a`)

`artifacts/atlas/at08_at09_054.py` (firewall-clean: root `jaxfne` only),
`tests/test_atlas_at0809_054.py` (6 tests), `atlas_gap_054.md`, coverage
AT-08/AT-09 → VALIDATED.

- AT-08 (fixed baseline / full HDP / cross-frozen HDP, same network,
  repeated A1 pulses): A2 spikes fixed 67, local 67, full 73 —
  propagated spike delta 0 with fixed coupling, +6 from coupling
  plasticity; propagated field delta 0.0095 with no spike-count change
  (the R4 dissociation). 3/3 declares PASS, ~10 s.
- AT-09 (cross-only / member-only / frozen masks): cross moves (max
  1.48) with members bit-frozen and vice versa; frozen arm dW = 0 while
  H deviates 0.0094 (R2 separation); C_12 0.50/0.45/0.45 (R3). 2/2 PASS.
- Schema v3: 13 v1/v2 names stable and in order + 10 appended cells
  (SPK/H/Phi per area, W_12/W_21, C_12/dphi_12). Phi_B REFUSED;
  E_reduction is 0.5.5.

## Acceptance mapping

- Canonical configurations bit-identical: broad gate (seal commit).
- Area-alone ≡ area-in-composition (zero cross): tested (2- and 3-area).
- Hierarchical ≡ flattened: tested (programs).
- Serialization roundtrip + chunked continuation: tested (edge-list JSON
  bit-exact; reconnect determinism; ensemble chunked == continuous).
- Firewall AT-07…AT-09: `test_atlas_firewall.py` green.
