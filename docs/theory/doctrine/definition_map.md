# Definition map

Each formal definition in the TFNE theory pages maps to the public jaxfne symbol that realizes it and to the test that exercises it. A `none` entry marks a definition with no realizing symbol or no test; each such row is listed in `KNOWN_GAPS` in `tests/test_definition_map.py`. A symbol with a leading underscore is an internal realization with no public name, listed in `INTERNAL_ONLY`.

| Id | Definition | Source | Symbol | Test |
|---|---|---|---|---|
| D1 | Core form x:A:y, one bounded system between typed boundaries | tfne_algebra.md, Core form | `jaxfne.tfne.parse` | `tests/test_tfne_algebra.py::test_one_cell` |
| D2 | Normalized specification NF(A), deterministic | 7_tfne_algebra.md, 1. Core form | `jaxfne.tfne.normalize` | `tests/test_tfne_algebra.py::test_deterministic_normalization_and_replay` |
| D3 | Specification digest, identity hash of the normalized form | 7_tfne_algebra.md, 24. Identity and hashing | `jaxfne.tfne.spec_hash` | `tests/test_tfne_algebra.py::test_different_orders_give_different_digests` |
| D4 | Realization of NF(A) to (s, h0, I) | tfne_algebra.md, Pipeline | `jaxfne.tfne.realize` | `tests/test_tfne_algebra.py::test_flatten_inspect_round_trip` |
| D5 | Flat execution state from one call | tfne_algebra.md, Pipeline | `jaxfne.tfne.flatten` | `tests/test_tfne_algebra.py::test_flatten_one_call_json_safe_summary` |
| D6 | Realization index map I, path, slice, rule and edge | 7_tfne_algebra.md, 22. Realization index map | `jaxfne.tfne.Realization` | `tests/test_tfne_execution.py::test_index_map_addresses_the_simulated_neuron_axis` |
| D7 | Execution normal form (h_t, x_t; s) to (h_t+1, y_t) | tfne_algebra.md, Core form | `jaxfne.simulate` | `tests/test_tfne_execution.py::test_tfne_source_reaches_kernel_execution` |
| D8 | Executable configuration built from the resolved specs | tfne_algebra.md, Pipeline | `jaxfne.tfne.to_configuration` | `tests/test_tfne_execution.py::test_executed_topology_matches_the_tfne_realization` |
| D9 | Cardinality N[A] and allocation R with sum constraint | 7_tfne_algebra.md, 5. Cardinality and composition | `jaxfne.tfne.resolve` | `tests/test_definition_map_realizations.py::test_d9_counts_come_from_the_mapped_expansion` |
| D10 | Replication A^n, indexed instances with no connectivity | 7_tfne_algebra.md, 6. Replication | `jaxfne.tfne.Replicate` | `tests/test_tfne_algebra.py::test_replication_creates_indexed_instances_without_edges` |
| D11 | Ordered composition A O B | 7_tfne_algebra.md, 10. Ordered composition O | `jaxfne.tfne.Ordered` | `tests/test_tfne_algebra.py::test_ordered_chain_binds_only_its_own_adjacency` |
| D12 | Cross composition A X B | 7_tfne_algebra.md, 11. Cross/lateral composition X | `jaxfne.tfne.Cross` | `tests/test_tfne_algebra.py::test_ungrouped_same_rule_cross_is_refused` |
| D13 | Explicit projections, left-to-right, right-to-left and bidirectional | 7_tfne_algebra.md, 13. Explicit projections | `jaxfne.tfne.Project` | `tests/test_tfne_algebra.py::test_explicit_projections_and_bidirectional` |
| D14 | Projection exclusions, subtracted from the rule expansion | 7_tfne_algebra.md, 14. Projection exclusions | `jaxfne.tfne.Exclude` | `tests/test_tfne_algebra.py::test_exclusion_subtracts_the_projection_it_names` |
| D15 | Declared in and out frontiers | 7_tfne_algebra.md, 9. Interfaces and frontiers | `jaxfne.tfne.resolve` | `tests/test_definition_map_realizations.py::test_d15_frontier_members_are_validated_by_the_mapped_expansion` |
| D16 | Canonical natural ordering of paths | 7_tfne_algebra.md, 20. Canonical ordering | `jaxfne.tfne.resolve` | `tests/test_definition_map_realizations.py::test_d16_paths_come_out_in_natural_order` |
| D17 | Typed boundaries x and y | 7_tfne_algebra.md, 19. Typed boundaries | `none` | `tests/test_tfne_algebra.py::test_typed_boundaries_preserved` |
| D18 | Biological identity, separate from the dynamical model | 7_tfne_algebra.md, 4. Biological identity and dynamical realization | `jaxfne.emitters.Emitter` | `tests/test_tfne_algebra.py::test_shared_type_distinct_models` |
| D19 | Geometry G as a declared object | 7_tfne_algebra.md, 15. Geometry | `jaxfne.Geometry3D` | `tests/test_tfne_algebra.py::test_geometry_compiles_to_static_state` |
| D20 | Declared geometry as fractions of a sampled block within [0,1] | tfne_algebra.md, Pipeline | `jaxfne.tfne.to_configuration` | `tests/test_definition_map_realizations.py::test_d20_declared_geometry_is_a_fraction_of_the_block` |
| D21 | Declared weight reaches the kernel | tfne_algebra.md, Pipeline | `jaxfne.emitters.resolve_edge_weight` | `tests/test_definition_map_realizations.py::test_d21_declared_weight_reaches_the_kernel_resolver` |
| D22 | Declared delay as integer steps, round(delay_ms / dt_ms) | tfne_algebra.md, Pipeline | `jaxfne.tfne.to_configuration` | `tests/test_definition_map_realizations.py::test_d22_declared_delay_realizes_to_round_delay_over_dt` |
| D23 | Mechanism resolution to a synaptic time constant | tfne_algebra.md, Pipeline | `jaxfne.tfne.resolve_mechanism` | `tests/test_tfne_parameter_transfer.py::test_canonical_mechanism_kinetics_reach_execution` |
| D24 | Synaptic kinetics independent of the integration step dt | tfne_algebra.md, Pipeline | `jaxfne.tfne.resolve_mechanism` | `tests/test_tfne_parameter_transfer.py::test_synaptic_tau_does_not_depend_on_the_integration_timestep` |
| D25 | RBS state H in R^(d_H), a finite-dimensional state container | rbs_rbd_hdp.md, 1. Semantic migration | `jaxfne.DynamicState` | `tests/test_relative_grammar_invariants.py::test_node_h_stays_in_declared_bounds_across_stress` |
| D26 | RBD dynamics dH/dt = F_H(H, x, I, W, u) | rbs_rbd_hdp.md, 2. RBD | `none` | `none` |
| D27 | Physical time class T and forward-causal clock t[n+1] > t[n] | relative_quantity_grammar.md, 3. Physical time (class T) | `jaxfne.Signals` | `tests/test_relative_grammar_invariants.py::test_time_ms_monotonic_and_causal` |
| D28 | Quantity kinds B, R, E, T | relative_quantity_grammar.md, 1. Quantity kinds | `none` | `none` |
| D29 | Null HDP leaves the base weights unchanged | relative_quantity_grammar.md, 5. Effective-parameter mappings | `jaxfne.hdp_rule.hdp_params_are_identity` | `tests/test_relative_grammar_invariants.py::test_hdp_off_matches_hdp_null_weights` |
| D30 | Edge weight mapping, magnitude-replacing: w_eff = sign(w_baseline) times theta_0 | relative_quantity_grammar.md, 5. Effective-parameter mappings | `jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp` | `tests/test_definition_map_realizations.py::test_d30_population_edge_weight_is_sign_times_theta0` |
| D31 | Intrinsic parameter mapping: a_eff = a_base times theta_1 | relative_quantity_grammar.md, 5. Effective-parameter mappings | `jaxfne._hdp_adaptive.bind_theta_to_plant` | `tests/test_relative_grammar_invariants.py::test_population_intrinsic_theta_1_recovers_base` |
| D32 | Weight clipping to w_floor when weight plasticity is active | relative_quantity_grammar.md, 5. Effective-parameter mappings | `jaxfne._hdp_family_kernel.apply_registered_rule_update` | `tests/test_relative_grammar_invariants.py::test_null_recovery_clips_subfloor_weights` |
| D33 | Registrable HDP rule F_W, persistent parameter dynamics from H | rbs_rbd_hdp.md, 3. HDP | `jaxfne.hdp_rule.register_hdp_rule` | `tests/test_hdp_family_g3.py::test_identity_rule_matches_baseline_adex` |
| D34 | Typed gain or coupling map g_p(H; kappa) | tfne_containment_architecture.md, 5. Typed gain/coupling maps | `jaxfne.emitters.rbd_recurrent_gain` | `tests/test_definition_map_realizations.py::test_d34_gain_map_is_one_at_neutral_h_and_affine_otherwise` |
| D35 | Calibration boundary C_S(Q; G, Theta) | rbs_rbd_hdp.md, 6. Geometry/calibration boundary | `none` | `none` |
