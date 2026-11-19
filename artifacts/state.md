# state

Stamp: jaxfne@_temp_crazy_sailor_m0m1_b@4b9e58e2 · 2026-11-19. Overwritten each session. Where this file and git or a gate disagree, git and the gate win. History: git.

Branch `_temp_crazy_sailor_m0m1_b`, worktree `/Users/hamednejat/workspace/dev/jaxfne-crazy-sailor` on the Mac that ran the work. No upstream. Not pushed. `dev` is an ancestor of this branch (checked 2026-11-19).
Status: local only. The owner's ruling (2026-11-19) blocks the push until record integrity and the required gates pass (O-10, O-11).

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

## Not run or not examined

- Broad gate on this tip: not run this session.
- Plasticity Boat tier-0 suite (33 files) and the `release_artifacts` gate: not run and not examined (EX-31, jomission).

## Environment

- Scratch logs in the session scratchpad are not durable.
- Gitignored evidence is not in git: jomission `artifacts/_temp/plasticity_boat/` (9.6 MB, traces and gate runs) exists only on the Mac that ran the work.

## Next action

1. Hamm: decide O-10 (repin to `cdc17e41…` on a SHA-specific go, or restore `c52e830b…`).
2. Hamm: confirm whether "push all" covers this branch. Under the ruling it does not until O-10 and CS-10 close (O-11).
3. Then CS-10 (broad gate to 0 failures, without weakening the gate), then EX-22 (critic findings).
4. Plasticity Boat start steps: jomission `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/HANDOFF.md`.
