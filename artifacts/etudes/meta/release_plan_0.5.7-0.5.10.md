# Release plan 0.5.7–0.5.10: correct tools, then replications on them

Requested by Hamm, 2026-10-19: a strict plan for the next releases, aimed at perfecting the
existing tools and closing the open items they carry, executed with Muse (opencode) and Haiku 5.5
workers. It supersedes the release table of `release_plan_0.5.6-0.5.10.md` (§4); that file keeps
the pipeline P0–P7, the fidelity tiers and the decisions of 2026-10-11 to 2026-10-13, which stay
in force. Tags, main and PyPI still need Hamm's word per release.

Baseline: dev 4969d01b, 0.5.6 released 2026-10-15, 178 commits and 26 changelog fragments since.

## 1. Order and rationale

| Release | Theme | Closes | Opens the way for |
|---|---|---|---|
| 0.5.7 | Truthful core: every public parameter consumed or refused, every test asserts what it names, every record says what was run | P-036 to P-042, T-18, T-19 (tested items), T-20, T-31, T-10 | replications whose verdicts can be trusted |
| 0.5.8 | Core-compliant replications: every lane runs on `construct` and `jaxfne.analysis` | T-30, T-32c, T-32d, T-32e, T-33, T-22, T-25 | scale |
| 0.5.9 | Scale: full-size networks on the GPU, laminar and multi-area | AT-10-R6, the LIF GPU kernel, T-32f core gaps | fields |
| 0.5.10 | Fields and readouts: LFP, EEG, laminar spectra on validated networks | deferred items 3, 4, 5 of the todo stack, if measured | 0.6.0 |

A release ships when its exit gate (§6) holds. Work that misses its release moves to the next
one with its reason; it never blocks the release it missed.

## 2. 0.5.7 work packages

Each package has one owner tier, one acceptance command and one commit (or one per class).

| Id | Package | Items | Tier | Acceptance |
|---|---|---|---|---|
| A1 | Test gates assert their names | P-037: x64 per test instead of global off; skipped float64 asserts run; `except: pass` around asserts removed; cwd-independent paths; `assert_allclose` for optimizer pins; lane tests under `artifacts/etudes/papers/**` and `scripts/harness/tests/` enter a gate; `networkx` declared | Muse code, Muse review | broad gate 0 failed; skips in a fresh clone listed and each justified; every changed test fails on a mutant |
| A2 | Stale pointers and ignored flags | P-038: evidence checklist path in 3 scripts; `evidence_manifest.source_files` checked; mcc3 manifest entries; 4 parsed-unread flags read or removed; `simulate` return documented for `return_state=True` | Haiku worker | `run_test_gate.py report` OK; a test per flag that fails when the flag is ignored |
| A3 | Records match what ran | P-039 manifest wording (`mcdp` as run); P-040 Destexhe headers and REPLICATIONS row; P-041 PCL page claims and the v5 T4 qualifier; P-042 CAMINOS release row and the B2 omissions | Haiku worker, dispatcher checks every number | `check_replications.py --check` with fewer known violations; each changed number opened at its receipt |
| A4 | One sign and preset table | T-18: `presets.py` and `tutorial_utils.py` read `emitters.CELL_TYPE_SIGN` or state their scope; SOM outside the AdEx table (P-043 closed by Hamm 2026-10-19: presets kept, rheobase documented) | Muse code | tests that a catalog sign differing from the table fails |
| A5 | Core design items with a test | T-19: Izhikevich state clamp, one home for drive defaults, the process-wide `jnp.clip` patch in `bridges.py` removed, `object.__setattr__` on frozen dataclasses replaced, `_s`/`_ms` names in the HDP controller; each fixed with a failing-first test or recorded with its reason | Muse code, Muse review | outputs bit-identical on the canonical column and the Atlas smoke unless a fix is declared |
| A6 | Lint and newline policy | T-31(e) E741, E402; T-20 `cell_labels`; T-10 docs newline policy (`changelog.md`, atlas manifest) | Haiku worker, medium | ruff E741/E402 0 in scripts, tests, examples, then gated; 0 LF-only lines in the named files |
| A7 | Issue log hygiene | P-035 and P-036 to the closed archive (re-pin it); each closed issue with its resolution line | Haiku worker | record integrity 0 violations |
| A8 | Replications at their current verdict | Brunel P3 v4 scored and judged; CAMINOS zero-delay scored as section 9 and R = 4 against R = 2 (T-22); Destexhe and GSDR B8 pilots and preregs for D3 | dispatcher (science), Haiku runs | each verdict on the dashboard and in REPLICATIONS with receipts |
| A9 | Release mechanics | assemble the fragments; `release_body.py --tag v0.5.7` claim check (G-1, T-10); rc gate; CI green on the SHA; second review; receipt; RTD `stable` builds | dispatcher, Haiku gate runs | `release_body.py` passes; receipt written |

Order: A1 first (later packages rely on gates that assert), then A2–A7 in parallel lanes (at most
3), A8 alongside, A9 last.

## 3. 0.5.8 work packages

| Id | Package | Items | Tier |
|---|---|---|---|
| B1 | Estimators in `jaxfne.analysis` | T-32d: ISI ratio and CV, pairwise correlation, aperiodic fit, tuning fit, phase delay; lane copies replaced by imports, outputs equal on saved rasters | Muse code |
| B2 | Every route through `construct` | T-30: HH in `construct()` dispatch (GSDR); the `net.m` update order as a core option (Izhikevich); PCL full-network mapping on the LIF tensor route (T-33, O-24, T-25 A') | Muse code, dispatcher reviews semantics |
| B3 | Batched runs | T-32c: seed and candidate batching for the AdEx and LIF routes and `Model.tune` | Muse code |
| B4 | Checklist in code | T-32e: rules E1, S2, F3, P4, R4 in `check_replications.py`; `replication_known_violations.json` shrinks to Hamm's exceptions | Haiku worker |
| B5 | Lane gaps on the core | Brunel S1 option (fixed in-degree at any N); Vogels G-SC; Potjans G-MULT, G-TAUSYN, G-POPDRIVE, G-WDIST, G-DDIST | Muse code |
| B6 | Lane closure | caminos, izhikevich2003, pcl and brunel2000 folders split by kind once closed (tree degree) | Haiku worker |

Exit adds: every paper passes `check_replications.py --check` with no listed violation except
Hamm's exceptions; Potjans and Vogels preregs ready for signature (B8).

## 4. 0.5.9 and 0.5.10 (fixed at the 0.5.8 seal)

- 0.5.9: AT-10-R6 RSS flat over chunks with the XLA cache cleared; the LIF edge kernel within the
  GPU budget at 25000 steps × 12500 cells (chunked rasters); Potjans 2014 and Mejias 2016 scored
  (Mejias G6 constants and the φ(0) normalization first); Lee, Pennartz and Mejias 2025.
- 0.5.10: Mazzoni 2015 LFP proxies and EEG validity on the 0.5.9 networks; deferred items 3–5 of
  the todo stack only where a measurement shows the need.

Packages for both are written at the 0.5.8 seal from what 0.5.8 measured.

## 5. Execution protocol

- Lanes: at most 3 writers at once, each in its own clone off dev; one writer per clone.
- Routing: Muse `code` for builds and fixes, Muse `review` for the adversarial review after each
  feature; Haiku `worker` for sweeps, records, gate runs and chunked runs; the dispatcher keeps
  science, semantics, verdicts and every push.
- Packet: goal, baseline SHA, files to read, paths to write, acceptance command with its expected
  summary line, stop conditions, 200-word report cap.
- Verify before integrating: read the diff, re-run the acceptance command, open each cited number,
  run each new test against the old code and require a failure.
- Push to dev after the gate a change needs (code: broad; records: report plus the tree tests),
  with no CI run in progress; CI green on the exact SHA before the next push of code.
- State: each landed package deletes its todo rows and adds one `state.md` line; this plan's
  tables lose rows as they close.

## 6. Exit gate per release

1. Every package closed or moved with its reason; no open issue in the release's scope without a
   resolution line or Hamm's ruling.
2. Broad gate 0 failed; CI green on the release SHA; `check_replications.py --check` and
   `build_dashboard_data.py --check` pass.
3. `release_body.py --tag <version>` passes its claim check.
4. Second review PASS on the release diff; receipt under `artifacts/release/`.
5. Tag, main and PyPI on Hamm's word.

## 7. Decisions for Hamm (batched, recommendation first)

| Id | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | P-042 16 unreferenced scripts and 8 ungated audit scripts | (a) archive with an index; (b) keep | (a); Hamm chose (a) 2026-10-20 |
| D2 | Destexhe and GSDR preregs after their B8 pilots | sign or revise | per prereg, after review |
| D3 | This plan's order and scope | approve or reorder | approve; Hamm approved 2026-10-20 |
