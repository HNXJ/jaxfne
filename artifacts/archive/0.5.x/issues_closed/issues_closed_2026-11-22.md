# jaxfne closed issues, batch of 2026-11-22 (0.5.10)

Moved verbatim from the open log (`artifacts/assurance/issue_log/ISSUE_LOG.md`) after its resolution line.
Earlier closed entries: `issues_closed_2026-10-26.md`, `issues_closed_2026-10-27.md`,
`issues_closed_2026-11-03.md` and `issues_closed_2026-11-21.md` beside this file, and `../issues_closed.md`,
`../issues_closed_2026-10-20.md` and `../issues_closed_2026-10-25.md`, pinned write-once.

### P-066

- Date: 2026-11-19. Type: BUG. Area: population construction, neuron positions, distance-based connectivity.
  Found by task-units-csd. Severity: medium for any run whose rules use Euclidean distance (localized sampling,
  `spatial_sigma`, radius gates) on the laminar construct path; affected runs not enumerated.
- Observation: x and y are sampled in mm (`[-column_radius_mm, column_radius_mm]` plus the area offset), z in
  the layer's `layer_fractions` range inside [0,1], never scaled by `column_height_mm`
  (`jaxfne/_construct_population.py:77-89,505-529`). `column_height_mm` is used only for layer volume
  (`jaxfne/builders.py:716-731`). `compile_connection_rules` takes Euclidean distance on x, y, z in "the same
  units as x/y/z" (`jaxfne/connectivity.py:886,1110-1115`), so depth counts 1/1.6 of its mm length at the default
  height. `NeuronalTensor` documents positions as mm (`neuronal_tensor.py:96-103`).
- Reproduction (`F:/warehouse/jaxfne/P-066_2026-11-19/probe_zextent.py`, `SHA256SUMS`; `build_laminar_column("V1", 60, geometry="laminar")`):
  known positive radius 0.25 -> 0.5 mm moves x from ±0.247 to ±0.495; height 1.6, 2.0, 8.0 mm leaves z at
  [0.0045, 0.9787] in all three. `tests/test_laminar_geometry_consumed.py` checks that height reaches metadata,
  not positions.
- Expected: one length unit for positions (mm, z scaled by `column_height_mm`), or z declared relative and
  distances refused or computed per axis. The field path reads relative depth (contacts in the same [0,1] axis,
  `fields/proxy.py:266,283-285`) and must move with the choice.
- Ruling (Hamm 2026-11-19): z in mm, scaled by `column_height_mm`; field proxy divides depth by the height.
- Repair (implemented 2026-11-19, checkpoint `31ca6c39` on `_temp_tfne-manuscript`; not merged):
  - `_construct_population.py`: layer z = fraction x `column_height_mm` (default 1.60, as in the uniform path).
  - `fields/proxy.py`: `project_laminar_sources(..., column_height_mm=None)` divides depth by the height when given;
    `project_model_sources` passes `model.params["column_height_mm"]`.
  - `_construct_core.py`: population models carry `params["column_height_mm"]` (not with a geometry override).
  - `_construct_connectivity.py connect()`: carries the height; refuses mixed laminar/non-laminar members and
    mismatched heights.
  - Tests: `tests/test_laminar_z_mm.py` (3), golden key set in `tests/test_construct_golden_snapshot.py`.
- Measured (before = `0a0d5ab5`, after = `31ca6c39`; script `_temp_p066_compare.py`, diff `_temp_p066_diff.py`,
  scratch only, seed-fixed sources, `project_model_sources` on both trees):
  | scenario | positions | field phi | note |
  |---|---|---|---|
  | laminar, height 1.0 | identical | identical | identity check |
  | laminar, height 1.6 (default) | z x1.6 (max 0.979 -> 1.566) | max rel. diff 1.8e-7 | float32 rounding |
  | laminar, height 2.0 / 8.0 | z x2 / x8 | identical | depth = fraction again |
  | laminar, radius 0.5, height 2.0 | z x2; x,y as before | identical | |
  | merge of two height-2.0 laminar models | z x2 | identical | |
  | merge of height 2.0 with height 8.0 | baseline merged silently | now refused | intended |
  | uniform3d, default 1.6 and 2.0 | identical (already mm) | **changed**, max rel. 0.71 and 1.14 | see below |
  - Unexpected: uniform3d positions were already mm before the change, so the baseline projection read them as
    fraction. The change divides by the height, which alters the uniform3d field output. Intended by the same
    logic; not listed in the original observation; needs review.
  - Not measured: realized connectivity of distance rules (`spatial_sigma`, `spatial_radius`) changes, since z
    now spans the column height. Probability-only rules do not read positions.
- Full suite on checkpoint `31ca6c39` (clone): 14 failed, 8391 passed, 49 skipped.
  Baseline `0a0d5ab5` on the same 8 files: 2 failed (`test_scaling_benchmark`, subprocess cannot import
  jaxfne; environment, pre-existing). So 12 failures are from this change, classified:
  - Two real regressions, found and fixed (commit on top of `31ca6c39`):
    1. `_model_simulate.py` (`simulate(record_fields=True)`, two sites) projected mm positions with no height
       (canary `field_lfp_csd`, csd_rms 53.9 vs 201.7). Now passes `column_height_mm`.
    2. `_construct_core.py` canonical PV<->E gate (`exp(-(dz/0.15)^2)`) read raw z; it is a relative-depth
       width. Now divides by the height for layered models; uniform_3d keeps its pre-change units.
       Canary `hdp_plastic_w` raster hash matches the register again.
  - Generated index: `docs/reference/errors.md` stale; regenerated with `scripts/generate_errors_index.py --write`
    (`test_generate_errors_index` 6 passed).
  - Intended unit pins, updated (approved by Hamm 2026-11-19), each to the new z units:
    `test_fields_positions_g8` x2 (delegate reference now passes the column height),
    `test_neuronal_tensor` x1 (z in [0.2h, 0.6h]), `test_neuronal_tensor_connectivity` x1 (z in 3 + [0.5h, 0.6h]),
    `test_tfne_parameter_transfer` x2 (executed z divided by h against the fractional domain),
    `canary_register.json` jdna `positions_mean/std` (only these changed; checked before and after),
    `test_neuronal_tensor_lane6b` (digest split: edges digest unchanged `0bc6b793...`, measured equal on both trees;
    positions digest now `94cb0202...`).
  - Touched files: 8 test/register pins above, plus no other test edits.
- Affected subset after the pin updates: the 8 pinned tests pass (34 in the 5 files; canary 7/7).
- Call-site review (4 Muse lenses, 2026-11-20, read-only; claims checked by hand). Fixed:
  `project_sources_to_laminar_field` forwards `column_height_mm`; `CanonicalDataset` and `FrozenE4Trajectory`
  carry the height (default None = legacy fraction; new builds record it; npz round-trips it when present);
  `materialize_field`, `zero_field`, the receipt leadfield and the E4 field calls use it. Tests:
  `tests/test_height_callsites.py` (4). Not a defect: geometry override (`LaminarSourceGeometry` validates
  depth to [0,1], so positions are already relative).
- Remaining: full suite rerun after all changes (not yet run); connectivity before/after; uniform3d ruling;
  independent review; merge decision. Pre-existing, separate: `Pose3D` plane xz/yz remaps the depth axis
  (`neuronal_tensor.py:144-148`) while the proxy reads `positions[:, 2]`.
- Resolution (2026-11-22): merged into dev as 04f6845b (R17). Repins: five approved by Hamm 2026-11-22 (S23,
  plan §10.1). Connectivity before/after: a distance rule (`spatial_sigma` 0.1, in-degree 10, 200 neurons)
  changes 51.7% of 2000 edges; probability-only rules and the canary chains keep edges bitwise (plan §10.1).
  uniform_3d field proxy change (`phi_e_proxy` max |Δ| 1.56x the old max): integrated as is (Hamm 2026-11-22,
  O-U3D, under S07). Broad gate on 04f6845b: 8409 passed, 43 skipped, 0 failed, all lanes pass. Pose3D plane
  remap split to P-067. Non-writer review of 5adddc8a..04f6845b: ACCEPT, no blocker or major; new tests fail
  7 of 7 on the pre-merge tree; default-column edges and x, y bitwise equal, z x1.6. Minors, carried in todo
  T-P066-FOLLOW: `height None` keeps the fraction reading (documented); experiment_a and e4 now freeze mm
  positions plus height, so their receipts need re-recording on the next regeneration; the PV<->E gate rescale
  (`_construct_core.py:296`) is untested and divides uniform_3d z in a mixed model.
