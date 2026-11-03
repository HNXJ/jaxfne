# jaxfne closed issues, batch of 2026-11-03 (0.5.9)

Moved verbatim from the open log (`artifacts/assurance/issue_log/ISSUE_LOG.md`). Earlier closed
entries: `issues_closed_2026-10-26.md` and `issues_closed_2026-10-27.md` beside this file, and
`../issues_closed.md`, `../issues_closed_2026-10-20.md` and `../issues_closed_2026-10-25.md`, pinned
write-once.

### P-048

- Date: 2026-10-26. Type: SCIENCE. Area: Izhikevich initial state. Severity: low.
- `u0` keeps the preset value (E -13, I +6.5) when `cell_params` overrides `b` and when `random_v0`
  draws `v0`, so `u0 != b * v0` (the paper's `u = b.*v`); `cell_params` takes no `v0` or `u0`.
  The E1 library path needs a declared per-cell initial state; whether the default should follow
  `b` is Hamm's call (it moves existing realizations).
- 2026-10-27: `cell_params` takes `v0` and `u0` (a declared `v0` needs `runtime(random_v0=False)`);
  the default question stays open.
- 2026-10-28: ruled and fixed (Hamm: the default is `u0 = b * v0`). `_apply_default_u0`
  (`jaxfne/_construct_population.py`) runs in `construct()` after the canonical block, on the realized
  `b` and `v0`; a declared `u0` still wins. Tests: `tests/test_izh_default_u0.py`. The default route
  moves (`random_v0` is on by default); the three izhi pins in `tests/test_inputs_xsource_t24.py` were
  re-recorded.

### P-051

- Date: 2026-10-27. Type: FRICTION. Area: Izhikevich `edge_list` route. Severity: low.
- The edge route filters every edge with its mechanism tau (`g <- g exp(-dt/tau) + spikes`), and
  `weight_range` compiles only with a mechanism, so instantaneous coupling (Izhikevich 2003,
  the dense route's `W @ prev_spikes`) has no named form on it. At tau 5 ms the Fig. 3 network
  saturates (2000 Hz); the E1 pilot declares tau 1e-3 ms, where the decay is exactly 0 in float32
  at dt 0.5 ms (`izhikevich2003/core/run_izh_core.py`). Possible change: a named instantaneous
  receptor on the edge route, equal to the tau -> 0 limit.
- 2026-10-28: ruled and fixed (Hamm: a named instantaneous receptor). `instantaneous` on the
  `edge_list` route: the current at step t is the weighted sum of presynaptic spikes at t-1, with no
  carry-over. It compiles with `weight_range` and must declare `tau_ms=0.0`
  (`emitters.INSTANTANEOUS_MECHANISM`). Tests: `tests/test_izh_instantaneous_receptor.py`; dense and
  edge rasters are identical and V agrees at rtol 1e-10.

### P-054

- Date: 2026-11-02. Type: SCIENCE. Area: Izhikevich presets `VIP` and `I` (`jaxfne/emitters.py:124,130`).
  Severity: medium; every realization with a VIP or I population. Found while answering jomission MIN-07.
- Observation: the presets carry Izhikevich 2004's class-1 excitable a 0.02, b -0.1, c -55, d 6
  (`izhikevich.org/publications/figure1.m`, panel G), but that panel integrates V' = 0.04V^2 + 4.1V + 108 - u
  + I, while jaxfne integrates 0.04V^2 + 5V + 140 - u + I. Rheobase (5 - b)^2/0.16 - 140 = 22.56 in jaxfne
  against (4.1 - b)^2/0.16 - 108 = 2.25 in the source, so a VIP cell at its preset drive 3.0 is silent
  (documented in `docs/reference/api/model/emitters.md`, "Default drives and rheobase").
- Minimal reproduction: one cell, Euler dt 0.1 ms, 1000 ms, u0 = b*v0, v0 in {-70, -65, -60}: 0 spikes at
  I = 4 and 22.5; 4 spikes in [200, 1000) ms at I = 23.
- Possible change (Hamm's call: it moves every VIP and I realization): keep the preset and record its
  provenance as "class-1 a,b,c,d in the standard equation", or give VIP its own source-faithful v equation
  coefficients, or take a VIP set from another source.
- 2026-11-02: ruled (Hamm: class-1 equation) and fixed. `VIP` and `I` carry `v_lin` 4.1 and `v_const` 108;
  `IzhikevichParams.v_lin`/`v_const` (None when all cells are standard) reach every Izhikevich integrator,
  `connect()`, checkpoints and the sanity runtime's VIP cells. Tests: `tests/test_izh_class1_presets.py`
  (fail on the pre-fix code). Not changed: `w3_stability_analysis` (analytic, standard cells by its
  arguments) and `streaming.py` (RS only).
