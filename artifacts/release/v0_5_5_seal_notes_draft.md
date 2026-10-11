Draft, not a receipt. Gates, tag and release fields are filled by the release owner.

# 0.5.5 seal notes (draft)

Source: `artifacts/todo_stack.md`, section "## 0.5.5 stack" (SEAL NOTES block,
ATLAS item 8, ACCEPTANCE block). Wording below is quoted from the todo text; no
new claims are added. Status of every entry is "listed" (nothing re-run here).

## H1 (e)

- Commit: `9958a72510ce3c50538a5b2bf960f68125d6e8bc`
- Output change (quoted): "`default_complete_configuration` is laminar; its
  outputs and config hash changed (0 -> 173 inter-area edges at defaults).
  Other callers kept their outputs."
- Evidence paths: none named in the todo text for this item.
- Status: listed.

## P-017

- Commit: `f8960a09548f78187f3e9b1e68c4fe7b16bf5020`
- Output change (quoted): "list the runs whose drive changed (commit f8960a0:
  stimulus-less marker events now silent; stimulus events use their own
  `duration_ms`)."
- Evidence paths: none named in the todo text for this item.
- Status: listed.

## Geometry (apply or refuse)

- Commit: `deca88643662381b2ffe0ceaab536925838d30d8`
- Output change (quoted): "`build_laminar_column(geometry='laminar',
  radius_mm=/height_mm=)` now writes the given values to
  `column_radius_mm`/`column_height_mm`; before, it ignored them. Defaults
  (`None`) declare nothing, so outputs without the arguments are unchanged."
- Evidence paths: none named in the todo text for this item.
- Status: listed.

## AT-02..AT-06 digests

- Commit: none named in the todo text for this item.
- Output change (quoted): "AT-02…AT-06 run records
  (`artifacts/publication/atlas/`) carry the pre-correction spec digest
  (AT-02…AT-05 recorded `n_contacts` 4, executed 16; AT-06 recorded a
  common-average reference and an 8–25 Hz band-pass that were never applied);
  regenerate them with the other Atlas records. The frozen agent task set keeps
  the old digests (`tests/test_agent_bench_055.py` allows exactly these
  corrections)."
- Evidence paths (all exist): `artifacts/publication/atlas/`,
  `tests/test_agent_bench_055.py`.
- Status: listed.

## P-010 chunk_ref

- Commit: `865e74bb410ee4f4c431cd248dea923174e1b0f3`
- Output change (quoted): "`chunk_ref` differs from 051 (174 -> 173 spikes,
  V_sum -6654899.0 -> -6652058.0): deterministic (3 reruns identical);
  bisected to 865e74bb (0.5.3 item 3, P-010 chain-noise fix), an authorized
  correctness repair: list it in the seal receipt and re-baseline `chunk_ref`
  for 055 as a declared repair, not E_semantic != 0 (the reference is the
  edge_list single-call path; `chunk_k4` unchanged)".
- Evidence paths (all exist): `artifacts/perf/matrix_051.json`,
  `artifacts/perf/matrix_055.json`, `artifacts/perf/matrix_055_spec.json`,
  `scripts/benchmark_055_matrix.py`.
- Status: listed.

## P-023 (at10_20area)

- Commit: `83848805b70b493a34ab13cc80dea4eb21db446a`
- Output change (quoted): "`at10_20area` was UNSUPPORTED because P-023
  (83848805) refuses an undeclared synapse time constant; the harness now
  declares the pre-P-023 default `dT_ms=0.1` on each ring AreaConnection
  (in-process check: coupled, 4,747,700 edges, equal to 051; the cell has no
  output checksum, so no bit-compare exists); rerun it in the final matrix and
  list P-023 in the seal receipt".
- Evidence paths (exists): `artifacts/perf/matrix_055.json` (the
  `at10_20area` cell).
- Status: listed.

## P-014 (suite2 presets, experiment_a, protocol_e declared drives)

- Commit: `9cc456597aede76f5b097f985ab8ac07a154acbe` ("fix(0.5.5): refuse
  inert Configuration.cell_type_drives; migrate callers (P-014)"). Review
  follow-up: `5959ace9fd4558707632e025b5fb0e2da056d84e` ("address adversarial
  review of AT-10-N20 and P-014/P-015"). The release owner confirms which runs
  each commit changed.
- Output change (quoted): "Existing canonical configurations bit-identical to
  the frozen baseline, except outputs changed by authorized repairs (P-014:
  suite2 presets, experiment_a, protocol_e now run their declared drives),
  each listed in the seal receipt."
- Evidence paths: none named in the todo text for this item.
- Status: listed.

## Open before the seal

Copied from the todo text; no new facts.

- Quiet-machine matrix: "Warm times were 0.86-3.7x the 051 values on a
  saturated machine, so they are not a regression claim: rerun on a quiet
  machine before the seal, and add `T_test` (the broad gate took 539-811 s
  loaded) to the receipt." Accept when "every null-drive cell's
  `output_checksum` equals `matrix_051.json` (only `mech_hdp` may differ:
  E_semantic = 0), each cell records its drive provenance, and `T_test` is
  measured." Open on the 2026-10-06 file: "the file has no `T_test` and no
  per-cell drive provenance".
- AT-02..AT-06 record regeneration: "regenerate them with the other Atlas
  records" (see AT-02..AT-06 digests above).
- Version bump: `pyproject.toml`, `jaxfne/_model.py` and
  `artifacts/release/current_release_authorities.json` read 0.5.0 (result
  `_meta.jaxfne_version` reads 0.5.0: "version string not yet bumped").
  Regenerate `artifacts/programme/methods_055.md` (via
  `scripts/gen_methods_055.py`).
- Gates: "Run the release-tier baseline check before the seal
  (`python scripts/run_test_gate.py release`, then `rc`; receipt template
  `artifacts/release/v0_5_0_release_receipt.json`; authorities
  `artifacts/release/current_release_authorities.json`)."
- Human acts: "Tag, main merge and release are separate human acts."; "PR #101
  (dev -> main) opened by Hamm"; "no tag or release" until "the 100/100 seal
  is met".
