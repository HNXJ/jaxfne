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

## Second increment, 2026-10-24 (PCL-R1 receipt sealing)

+10,000 bytes (90,205,000 -> 90,215,000), authorized by the PCL-R1 sealing
ruling ("Track the completed R1 matrix and its reproducing commands"): the
tracked receipt `artifacts/etudes/papers/pcl/r1_fidelity_audit_2026-10-24.md`
(+9,218 bytes, hash-pinned in `scripts/record_hashes.json`, dated class 20).
Measured tree after sealing: 90,212,449 bytes; headroom 2,551. The probe
scripts (25 files, 25,194 bytes) are preserved OUTSIDE the repository at
`C:/workspace/jaxfne_pclr1_audit/` with sha256s recorded in the receipt.

## Third increment, 2026-10-25 (issue-log batch, changelog fragment, restored README still)

+10,000 bytes (90,215,000 -> 90,225,000) and +2 files (2559 -> 2561), decided by the keeper under
Hamm's standing rule (2026-10-24: the repo stays below 100,000,000 bytes, amendments fine): the
closed-issue batch `artifacts/archive/0.5.x/issues_closed_2026-10-25.md` (+8,428 bytes, one file,
hash-pinned), the changelog fragment `artifacts/changelog.d/refuse_unrealized_sanity_delta_params.
changed.md`, the open-stack rebuild, and `docs/assets/readme/hdp.png` (59,862 bytes, required by
`tests/test_atlas_suite.py::test_readme_still_exists_for_every_fixed_panel`; an orphan scan by basename had
listed it as unreferenced, and the broad gate caught the removal). Measured tree: 90,217,078 bytes.

## Fourth increment, 2026-10-25 (gate false-passes)

+4,000 bytes (90,225,000 -> 90,229,000), no files, authorized by Hamm in chat 2026-10-25 ("record a small
budget increment if the bytes require it"): failing-case tests and fixes for four gates that passed without
examining what they name (`harness_integrity`, `release_artifacts`, `notebooks_assets`, `memory_hygiene`;
the last now scans `artifacts/memory.md` and flags short SHAs), in existing files.

## Fifth increment, 2026-10-25 (per-cell `cell_params`)

+4,000 bytes (90,229,000 -> 90,233,000), no files, under Hamm's standing rule (2026-10-24, amendments
fine) and his 0.5.8 scope decision to fix izhikevich2003 E1 in the core: `cell_params` takes one value per
matched neuron (`_construct_population.py`, three tests in `test_cell_params_compilation_v0401.py`, one
changelog bullet), which unblocks the P3b arm's heterogeneous a, b, c, d.
Measured tree at fd8933d6: 90,230,927 bytes, 2561 files.

## Consolidated review and sixth increment, 2026-10-25 (O-66, paper plan)

O-66 (Hamm 2026-10-25): the three `artifacts/publication/atlas/AT-10-N20/*/lfp.html` stay (3,530,080 +
3,531,433 + 3,532,786 = 10,594,299 bytes; referenced by each bundle's `manifest.json` and `index.html` and
by `atlas_run.json`). They are already inside the measured tree. A raise by their full cost would set the
budget to 100,827,299 bytes, above the standing 100,000,000 cap, so it is not applied; Hamm decides.
Sixth increment: +8,000 bytes and +1 file (90,233,000 -> 90,241,000; 2561 -> 2562) for the paper plan
`artifacts/programme/design/plans/arxiv_paper_plan.md` (G-0, Hamm 2026-10-25: "6kb is ok"), its goal and
todo lines, and this entry; +1 file (2563) for the cherry-picked Vogels fix fragment
`changelog.d/istdp_vogels2011_traces.fixed.md`, reclaimed at release when `assemble_changelog.py` deletes
the fragments. Second to sixth increments: +36,000 bytes (90,205,000 -> 90,241,000) and +4 files
(2559 -> 2563). Further raises wait
for a consolidated review.

## Consolidated raise, 2026-10-25 (Hamm: option a)

Budget 90,241,000 -> 95,000,000 bytes and 2563 -> 2700 files, signed by Hamm in chat 2026-10-25 (option
a of three: one raise below the 100,000,000 cap; a raise by the `lfp.html` cost to 100,835,299 with a
lifted cap and per-change increments were declined). Measured tree before the raise: 90,240,360 bytes,
2563 files. Headroom 4,759,640 bytes and 137 files covers the paper programme G-0 (0.5.9 to 0.6.0:
figures, receipts, tests). O-66 closes: the three AT-10-N20 `lfp.html` stay.

## Docs still twins, 2026-10-27 (no raise)

The light and dark twins of the six `docs/assets/visuals/` stills (Hamm 2026-10-24: a budget amendment for
the twins is fine): +6 files and +850,688 bytes (six `.dark.png` 902,734 bytes; the six light stills
replace the dark ones, 916,374 -> 864,328). Measured tree after the lane: 91,646,911 bytes, 2609 files,
inside the consolidated budget.
