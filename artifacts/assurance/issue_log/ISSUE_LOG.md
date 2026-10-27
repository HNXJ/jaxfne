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

### P-046

- Date: 2026-10-26. Type: BUG. Area: Izhikevich configuration route, `connectivity_mode="explicit"`.
  Severity: high (the receipt misstates the realized network).
- Reproduction: `balanced_ei_population` n=40 (E 0.8), two `ampa` rules E->E and E->I at
  probability 1, `update_metadata(connectivity_mode="explicit")`. Expected: only the 1248 declared
  edges. Actual: 2808 edges, the 1560 default dense-W edges (|w| 0.0791) kept beside the declared
  ones, while `connectivity_compilation` records `default_edge_count: 0`. At n=1000 the explicit
  construct took 826 s against 1.7 s by default (9C-E1 pilot probe).
- Blocks the E1 closure: the Fig. 3 network on the library path needs exactly the declared W.

### P-047

- Date: 2026-10-26. Type: BUG. Area: `Configuration.connections` selectors. Severity: medium
  (P-014 class).
- A rule with `target={"cell_type": "all"}` compiles 0 edges on both compilers and nothing refuses it
  (`declared_rule_edge_count: 0`); `{}` selects every cell. Expected: a selector matching no cell
  is refused by name.

### P-048

- Date: 2026-10-26. Type: SCIENCE. Area: Izhikevich initial state. Severity: low.
- `u0` keeps the preset value (E -13, I +6.5) when `cell_params` overrides `b` and when `random_v0`
  draws `v0`, so `u0 != b * v0` (the paper's `u = b.*v`); `cell_params` takes no `v0` or `u0`.
  The E1 library path needs a declared per-cell initial state; whether the default should follow
  `b` is Hamm's call (it moves existing realizations).
