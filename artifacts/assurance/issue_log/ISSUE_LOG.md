# jaxfne issue log

Open issues only. Closed entries (I-001…I-014, P-001…P-010, P-013..P-015, P-021..P-026) and
their verdicts are archived byte-for-byte at
`artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md` and `artifacts/archive/0.5.x/issues_closed.md`.
A new issue takes the next free P-ID; a solved issue gets a resolution line, then moves to a dated
closed batch beside them (`artifacts/archive/0.5.x/issues_closed_<date>.md`, pinned write-once like
`issues_closed.md`; P-035 and P-036: `issues_closed_2026-10-20.md`; P-038, P-039, P-041 to P-044:
`issues_closed_2026-10-25.md`; from 0.5.8 the batches sit in `artifacts/archive/0.5.x/issues_closed/`,
P-037: `issues_closed_2026-10-26.md`).

Entry fields: date, type (`BUG` `FRICTION` `DOC` `PERF` `SCIENCE` `IDEA`),
area, observation, severity, minimal reproduction, expected, actual, evidence,
possible future change.

## Open

Entry P-040 comes from the whole-project scan of 2026-10-18 (six Haiku workers,
read-only; each line was re-opened and confirmed by the dispatcher unless marked inferred).

### P-040

- Date: 2026-10-18. Type: SCIENCE. Area: Destexhe 2009 lane. Severity: high.
- `destexhe2009/PREREG_P3.md:1` reads "draft, UNSIGNED — no scored run may use it";
  `CALIBRATION.md:3-4` scores E1-E4 against it (header "Verdict: OFF"), and its final section
  says E1 passes (:427) and scores the reserved seeds 4-8. `REPLICATIONS.md:12` says "no verdict
  yet" and "28 passed" (30 pass today).
- 2026-10-20 (release plan A3): CALIBRATION header and steps 1-6 marked unscored pilots (ruling
  2026-10-18), body verdicts too; REPLICATIONS row agrees. Closes with the lane's next prereg.
- 2026-10-26: the Destexhe prereg moves to 0.5.9 (scope freeze, Hamm 2026-10-25); the entry stays
  open into 0.5.9.

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

### P-047

- Date: 2026-10-26. Type: FRICTION. Area: `Configuration.connections` selectors. Severity: low.
- A rule with `target={"cell_type": "all"}` compiles 0 edges on both compilers (`{}` selects every
  cell). It is recorded, not silent: the rule's status reads `compiled_no_matching_edges`, a design
  pinned by `test_connections_no_match_marked_compiled_no_edges`. A refusal would reverse it and break
  that test (9C-E1 core packet, 2026-10-27); left as is. An `"all"` value is not a wildcard.

### P-048

- Date: 2026-10-26. Type: SCIENCE. Area: Izhikevich initial state. Severity: low.
- `u0` keeps the preset value (E -13, I +6.5) when `cell_params` overrides `b` and when `random_v0`
  draws `v0`, so `u0 != b * v0` (the paper's `u = b.*v`); `cell_params` takes no `v0` or `u0`.
  The E1 library path needs a declared per-cell initial state; whether the default should follow
  `b` is Hamm's call (it moves existing realizations).
- 2026-10-27: `cell_params` takes `v0` and `u0` (a declared `v0` needs `runtime(random_v0=False)`);
  the default question stays open.

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

### P-051

- Date: 2026-10-27. Type: FRICTION. Area: Izhikevich `edge_list` route. Severity: low.
- The edge route filters every edge with its mechanism tau (`g <- g exp(-dt/tau) + spikes`), and
  `weight_range` compiles only with a mechanism, so instantaneous coupling (Izhikevich 2003,
  the dense route's `W @ prev_spikes`) has no named form on it. At tau 5 ms the Fig. 3 network
  saturates (2000 Hz); the E1 pilot declares tau 1e-3 ms, where the decay is exactly 0 in float32
  at dt 0.5 ms (`izhikevich2003/core/run_izh_core.py`). Possible change: a named instantaneous
  receptor on the edge route, equal to the tau -> 0 limit.
