# Geometry mutation receipt 0510 (A3)

Baseline `c7f49870`, 2026-11-06. One mutant per property test, one file at a
time; each restored by copying back a pre-mutant copy kept outside the repo,
then `sha256sum` compared with the pre-mutant value and `git diff --stat`.

| test | mutated | planted break | failing assertion | restore sha256 |
|---|---|---|---|---|
| `test_round_trip_keeps_count_labels_signs_and_positions` | `jaxfne/neuronal_tensor.py:1425` (`to_dict`, after `d = asdict(self)`) | truncate every area pose translation to 2 components on save | `tests/test_geometry_invariants.py:122` (`back = _build(loaded)`; broadcast `(22, 3)` vs `(2,)` — the dict-equality at :118 passes because both sides go through the same corrupted `to_dict`) | f6e06ebc…9bde473 |
| `test_rigid_motion_keeps_pairwise_distances_and_maps_positions` | `jaxfne/neuronal_tensor.py:1918-1919` (`_apply_pose`) | drop the rotation term (`a_rot = a`, `b_rot = b`) | `tests/test_geometry_invariants.py:155` (`assert_allclose(moved_pos, expected, …)`) | f6e06ebc…9bde473 |
| `test_quarter_turn_about_centroid_keeps_localized_edges_on_a_square_grid` | `jaxfne/connectivity.py:946` (`_candidate_pairs_localized`) | weight by absolute x (`(cand_x - post_x)^2`) instead of pairwise `dist_sq` | `tests/test_geometry_invariants.py:217` (`assert e1 == e0`) | 91da09c9…37b80e61 |
| `test_flattened_neurons_keep_area_layer_label_and_sign` | `jaxfne/_construct_population.py:576` (before `return`) | swap `cell_type` of `neuron_rows[0]` (V1 L2 E) and `neuron_rows[9]` (V1 L2 PV), rows only | `tests/test_geometry_invariants.py:238` (`assert emitter.labels[i] == row["cell_type"]`, `'E' == 'PV'`; same-group swap keeps the Counter) | a233bb14…700af32f528 |
| `test_neuron_order_permutation_keeps_keyed_edges_and_spikes` (new) | `jaxfne/_construct_population.py:576`, same swap as above | same label swap | none: test still passes | a233bb14…700af32f528 |
| `test_rotation_about_origin_keeps_radius_edge_set` | `jaxfne/connectivity.py:946` (same line as the quarter-turn mutant, separate run) | same absolute-x weight break | `tests/test_geometry_radius_rotation.py:90` (`… turned, spatial_radius=RADIUS) == base`) | 91da09c9…37b80e61 |

Survived attempts (no kill, recorded, files restored): grid origin at zeros
instead of the pre-population minimum corner (quarter-turn test passes: the
27-neighbourhood covers the whole 16-point grid, so the anchor never
discriminates); x-slab instead of the Euclidean ball and distance-from-origin
instead of distance-from-post in the radius branch (both rotation-symmetric
here: the rotation is about the origin and the slab keeps the same draws).

Finding. The new permutation test is a post-construction relabelling: one
realized model, attributes carried together, keyed multisets compared. No
implementation mutant breaks it (the label swap above leaves it green), so it
pins the readout convention, not a mechanism; the keeper dropped it (a check that cannot fail proves
nothing), so it is not in `tests/`. A two-run variant — permute the
spec, rebuild, compare — is not expressible through the public API: rebuilding
an area-swapped 30-neuron tensor resamples positions in encounter order
(`_construct_population.py:504,527-537`), draws random v0 in construction
order (`_construct_core.py:1563-1564`), and advances one rng stream across
posts (`connectivity.py:918-919`), so positions, edges and spike counts all
change (probe: 27 vs 26 spikes, edge multisets differ). OPEN: a spec-level
neuron-order permutation property needs order-independent construction.
