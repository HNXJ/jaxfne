# jaxfne closed issues, batch of 2026-10-27 (0.5.9)

Moved verbatim from the open log (`artifacts/assurance/issue_log/ISSUE_LOG.md`). Earlier closed
entries: `issues_closed_2026-10-26.md` beside this file, and `../issues_closed.md`,
`../issues_closed_2026-10-20.md` and `../issues_closed_2026-10-25.md`, pinned write-once.

### P-045

- Date: 2026-10-26. Type: BUG. Area: Izhikevich construct routes, `hdp_params["noise_scale"]`.
  Severity: medium (P-014/P-015 class: stored, not consumed).
- `_model_simulate.py:838-839` passes `noise_scale` only to `simulate_edge_recurrent_izhikevich`;
  the `receptor_exponential` kernel hard-codes the coefficient (`emitters.py:4574`,
  `_noise_coef(0.5, ...)`), and the dense kernel receives it only for `update_order="net_m"`
  (`_model_simulate.py:851-856`). On those routes a declared `noise_scale` is dropped silently.
- Found by the 9C-E1 packet (2026-10-26). Expected: realized or refused. Future change: refuse it
  where no kernel reads it, or pass it to the dense default order; the E1 `gaussian_drive` lands
  with the same realize-or-refuse rule.
- Resolved 2026-10-27 (`tests/test_noise_scale_realized.py`): realized on the dense default order,
  receptor_exponential (with `gaussian_drive`), every `simulate_batch` route and the multi-area kernel's
  `control_params`; the plain-route jit cache keys carry it (a reused Model replayed the first value's
  kernel, edge route included); `simulate_batch` refuses `net_m`. Undeclared stays 0.5 bit for bit.

### P-046

- Date: 2026-10-26. Type: BUG. Area: Izhikevich configuration route, `connectivity_mode="explicit"`.
  Severity: high (the receipt misstates the realized network).
- Reproduction: `balanced_ei_population` n=40 (E 0.8), two `ampa` rules E->E and E->I at
  probability 1, `update_metadata(connectivity_mode="explicit")`. Expected: only the 1248 declared
  edges. Actual: 2808 edges, the 1560 default dense-W edges (|w| 0.0791) kept beside the declared
  ones, while `connectivity_compilation` records `default_edge_count: 0`. At n=1000 the explicit
  construct took 826 s against 1.7 s by default (9C-E1 pilot probe).
- Blocks the E1 closure: the Fig. 3 network on the library path needs exactly the declared W.
- Resolved 2026-10-27 (9C-E1 core lane): explicit mode starts from an empty graph on the plain
  branch of `_construct_build_network`; the receipt records the realized default count
  (`tests/test_izh_config_route_gaps.py`). `make_eig_network` still allocates the dropped (n, n) W.

### P-049

- Date: 2026-10-26. Type: FRICTION. Area: `sampling="multinomial"` (`_candidate_pairs_multinomial`
  in `jaxfne/connectivity.py`). Severity: high for the Potjans lane.
- K = round(p * n_post) with p in (0, 1], so the mean in-degree is at most 1; the paper's K gives
  in-degree -ln(1-C) * N_pre (497 for L5i->L5e at full N). `AreaConnection` refuses 54 of the 55
  projections. Potjans PORT_SPEC row 8 calls G-MULT closed; it is not for this lane.
- Proposed change: `in_degree=k` with `sampling="multinomial"` (refused together today) draws
  K = round(k * n_post) synapses with replacement, so `in_degree` is the mean in-degree.
- Resolved 2026-10-27 (9D, a26097f8): as proposed, on `Configuration.connections`, `AreaConnection`
  and `InterConnection`; `in_degree` with `probability` is refused; rows record `realized_k`
  (`tests/test_multinomial_in_degree.py`).

### P-050

- Date: 2026-10-27. Type: PERF. Area: `_reject_duplicate_explicit_edges` in `jaxfne/_construct_core.py`.
  Severity: high (blocks Brunel S1 and any large explicit-mode build).
- One `flatnonzero(inverse == u)` per unique identity makes the check O(E^2): Brunel construct at
  n=500 spends 1.27 s in it, at n=1000 (4x edges) 22.1 s (cProfile, 2026-10-27). The S1 smoke
  (n=6250, about 7.8M edges, extrapolated about 9 h) sat in construct for 52 min and was stopped.
- Resolved 2026-10-27 (9C-E1 core lane): one stable sort with per-group block bounds; 400k
  identities in about 2 s, against 89 s before (`test_explicit_duplicate_check_is_not_quadratic`).
