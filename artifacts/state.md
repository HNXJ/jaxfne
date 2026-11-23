# state

Stamp: jaxfne@_temp_crazy_sailor_m0m1_b@4b9e58e2 · 2026-11-19. Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

Branch `_temp_crazy_sailor_m0m1_b`, worktree `/Users/hamednejat/workspace/dev/jaxfne-crazy-sailor` on the Mac that ran the work. Upstream `origin/_temp_crazy_sailor_m0m1_b`. `dev` is an ancestor of this branch (checked 2026-11-19).
Status: push authorized 2026-11-19 on Hamm's explicit instruction ("push please"), given after the integrity FAIL and the 13 baseline failures were reported. Scope: this branch only; not dev, not main; no merge. The override does not waive the gates: the integrity gate stays FAIL (O-10) and the broad gate stays failing (CS-10). This overrides the earlier push block (2026-11-19 ruling) for this push only.

## Works (observed)

- Plasticity Boat dev-100 files moved out of this branch to jomission `_temp_plasticity_boat` (pushed at `280ebd0`). Commit `4b9e58e2` removes the 13 migrated files here; history is kept.
- V0 regression under JaxFNE source 0.5.9: exact, max abs diff 0.0 (jomission `receipts/dev100_v2/regression/V0_source_0.5.9.log`).
- V0 regression under pinned `jaxfne==0.5.8`: fails with `AttributeError: module 'jaxfne.paradigm' has no attribute 'SailorConfig'`. The pin lacks the Sailor API that the dev-100 code uses (jomission `V0_pinned_0.5.8.log`).
- `python3 scripts/check_tree_size.py --check`: exit 0; files 2798, 88,520,409 of 96,400,000 bytes.
- Owner decision (2026-11-19): the 100-neuron directional encoder is Option A (signed-position population). Recorded in jomission `manifests/todo.json`, `PB-DEV100-V2-VALIDATION`.

## Broken or in flight

| Item | Evidence (observed 2026-11-19) | Row |
|---|---|---|
| Record integrity FAIL | `python3 scripts/check_record_integrity.py --check`: violations 1. `artifacts/release/budget_amendment_2026-10-23.md` pinned `c52e830b…`, live `cdc17e41…` | O-10, EX-30 |
| Broad gate: 13 baseline failures | contract §10; not re-run this session | CS-10 |
| §13.7 critic verdict REVISE, 9 findings | no fix commit yet | O-2, EX-22 |
| Plasticity Boat dev-100: R4 fails at all seven criteria-v2 `g_s` tested; R2 passes only at `g_s` ≤ 4.5 | jomission `dev100_gate_record.md` | O-8, EX-25 |
| Plasticity Boat decoder: sum or mean over each motor group | jomission `O8_decision_packet.md` §8 | O-9, EX-25 |
| Fixation Boat wording not frozen | contract §13 | O-1, EX-21 |
| Eight-strategy list and primary metric not frozen | jomission `protocol_v0.md` §3–5 | O-6, EX-26 |

## Frozen Stage 2 abstract (owner freeze 2026-11-20)

Title: A Spiking Circuit Driven by Scalar Distance: Specification and Implemented Status

We specify a 10-neuron spiking circuit (8 excitatory, 2 inhibitory) receiving scalar target distance (e = ||p - p*||) and decoding two-dimensional force from opponent motor-neuron spike counts. Distance-dependent tonic input provides no directional information. The implemented fixture instantiates all 10 neurons, integrates at 1 ms, and applies force one 20 ms control interval after decoding. The specified reward-gated mechanism is not implemented in the registered rule: eligibility traces exist, but reward coupling is absent. An exploratory B2 run separately computed interval rewards. The registered H-gated weight drive has default gain k_w = 0, preventing weight updates under that default. The fixture produces a fixed +x force, demonstrating direction-only output rather than target-directed control or learning. L2 is not demonstrated: the F3 weights saturate at their bound. L3 lacks a confidence-bearing run; L4 remains unexecuted.

Status: verified 2026-11-20 (three lenses, no contradicted claims). Hamm accepted five unchecked items 2026-11-20; still to resolve before Methods:
1. L2 bound: the registry says "0 bound" (contract R-04, R-08), but the registered rule's default weight floor is 1e-3 (`jaxfne/hdp_rule.py:69`). Name the bound.
2. The k_w used by the lane B F3 runs is not stated; the k_w = 0 sentence covers only the registered default.
3. The lane B2 code is not in this checkout; the B2 claim rests on registry row R-08.
4. Contract line 408 ("holds only partly") conflicts with R-04 ("L2 gate fails"); line 408 is stale.
5. The contract cites paradigm.py at SHA 584a56ad; the check read HEAD fdb19c9a.

## Retired authorization record (kept here)

Removed from `artifacts/release/budget_amendment_2026-10-23.md` by `5e5fa70` (restores the file to the `c52e830b…` pin at `21d5da7b`). Kept verbatim so the 2797 -> 2798 authorization stays recorded. Not pinned; no gate reads it.

## Crazy Sailor contract, 2026-11-19 (Hamm: option B)

`budget_files` 2797 -> 2798 on branch `_temp_crazy_sailor_m0m1_b` (base `06f9bf26`), +1 file. Relative to the
parent of the consolidation commit (`f9c3e696`, `budget_files` 2801, raised by the branch's earlier commits
`e1b47bf2` and `b0c3f670` era), this commit lowers the value 2801 -> 2798 and reverts those unauthorized raises:
`artifacts/programme/design/plans/crazy_sailor/m0_contract.md`, the Crazy Sailor independent
specification (M0 contract, M1 receipts, M2 handoff in §12). Authorized by Hamm in chat 2026-11-19:
"Authorize Option B ... file budget 2797 -> 2798, restricted to the Crazy Sailor contract". No other file
is authorized by this entry. The same branch consolidates its code into existing files
(`jaxfne/paradigm.py`, `tests/test_paradigm.py`), so the branch adds no other file.

Discrepancies recorded, not resolved here:
- This file's ledger (2563 -> 2700 files; 95,000,000 bytes) does not match the enforced allowlist at
  `06f9bf26` (2797 files; `budget_bytes` 96,400,000). The allowlist values were set by commits outside this
  record (`af076b44`, which is on `origin/dev`). The enforced value is the allowlist.
- `origin/dev` at `0f1f0d22` is already at 2798 tracked files with `budget_files` 2798 (P-062 step 2 added one
  file, `fb3e4d6f`). Integrating this contract into `dev` would give 2799 files, one above this amendment. That
  needs a separate ruling; this entry does not authorize it.
- `e124fe94` raised `budget_bytes` to 96,800,000 on a branch that is not `origin/dev`; not applied here.

## Not run or not examined

- Broad gate on this tip: not run this session.
- Plasticity Boat tier-0 suite (33 files) and the `release_artifacts` gate: not run and not examined (EX-31, jomission).

## Environment

- Scratch logs in the session scratchpad are not durable.
- Gitignored evidence is not in git: jomission `artifacts/_temp/plasticity_boat/` (9.6 MB, traces and gate runs) exists only on the Mac that ran the work.

## Next action

1. Hamm: decide O-10 (repin to `cdc17e41…` on a SHA-specific go, or restore `c52e830b…`).
2. Done: push of this branch (Hamm's override, above). Pending: CS-10 (broad gate) and the integrity repair; the branch is not mergeable until both pass.
3. Then CS-10 (broad gate to 0 failures, without weakening the gate), then EX-22 (critic findings).
4. Plasticity Boat start steps: jomission `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/HANDOFF.md`.
