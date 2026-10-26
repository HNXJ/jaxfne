# Lane 2 CALIBRATION.md — PCL Fig. 3 on jaxfne core grammar (residual protocol)

Lane: reproduce PCL Fig. 3 (N'dri et al. 2025) with z from jaxfne core only.
Baseline commit: 3dd02ed060390ecf711e04860a1d6d46e854ef83. All changes uncommitted.
Prereg: `prereg/P3_fig3_v5.md` (signed v5); scored result: `prereg/P3_fig3_v5_result.md`
(verdict FAIL — T2 fails both sets; T1, T3, T4 pass; nothing tuned after scoring).

## Estimand table

| Estimand | y (published) | z at θ₀ (oracle, v4 runs) | z at θ* (core) | e (band) |
|---|---|---|---|---|
| T1 R_ori, simple / complex | qualitative (C1) | 3.57 / 5.30, PASS | missing (no core network run) | as v4 |
| T2 SF curve, simple (64 cells) | digitized Fig. 3c | 10/16 points, SSE 0.349 = 1.50 × flat, FAIL | missing | \|ours−fig\| ≤ 0.04 + 2·√(se_fig² + se_ours²), ≥ 80 %, SSE ≤ 0.5 × flat |
| T2 SF curve, complex (128 cells) | digitized Fig. 3d | 12/15 points, SSE 0.204 = 0.52 × flat, FAIL (guard) | missing | same |
| T3 median X, simple / complex | qualitative (C2) | 0.53 / −0.12, PASS | missing | as v4 |
| T4 median L complex − simple | qualitative (C3) | 0.83 vs null 99th 0.42, PASS | missing | complex above simple, above null |

θ₀ = trained cSTDP3 weights (`F:/warehouse/pcl/.../add_simulations_cSTDP/cSTDP3`),
port 5d101eb3, `runs/fig3_cSTDP3` (3168 tasks, camera-0 feed, learning off).

## Core modules added (exact oracle math at edge scale)

- `jaxfne/emitters_lif.py`: current-based LIF with PCL refractory terms, clock-driven
  `lax.scan` translation; reached through `construct` -> `simulate` since Step 9.
- `jaxfne/hdp_rules_pcl.py`: registered core rule `pcl_stdp`, update copied exactly
  from `artifacts/etudes/pcl/pcl_hdp_rule.py` (whose `register()` now skips it).
- Exports: re-export in `jaxfne/emitters.py`, +5 ADVANCED in `jaxfne/public_surface.py`,
  contract artifact regenerated (288 → 293). No existing public function, default or
  numeric touched.
- `--engine {numba,core}` in `port/run_tuning.py` (default `numba`, behaviour unchanged);
  `core` refuses the tuning run with CORE_GAP; manifest records `engine`.

## Simplifications and fidelity fixes (with file)

- Clock translation (`emitters_lif.py` docstring): decay every step; the k arrivals of a
  step share one subtraction of k·(R+A); per-event C++ order inside a step not modelled.
- Simple threshold 10 mV from paper Table 2, not the default config's 30 mV; per-network
  values ride overrides. Simple adaptation increment defaults to 0 (default ETA_SRA).
- LTD test expectation corrected to exp(−0.5/7) (trace enters the arrival step at 1.0);
  rule math unchanged.
- Strict V > thr; floor −20 mV; t_last NEVER=−1e30 (never −1e30).

## Lane 2c — core gaps to the Stage C oracle gate (HEAD ed24cafd)

Doctrine: (h, y) = f(x, h′) on NeuronalTensor A; geometry part of A; input
drive is part of x; same-step interactions are declared intra-step fixed point.

### Gap 1 — LIF input route (CLOSED)

- `simulate(paradigm=StimulusSchedule)` now rides the lif route as native-mV
  input current via `schedule.to_array(...)` (`jaxfne/_model_simulate.py`,
  `_simulate_lif`); emitter constant drive + schedule + poisson all compose.
  `simulate_condition` stays refused for lif (explicit schedule instead).
- Fail-first test `test_lif_paradigm_schedule_drives_targeted_cells_only`
  failed on baseline (`ValueError: ... no paradigm-driven input current`);
  `tests/test_lif_tensor_route.py`: 24 passed; ruff clean.
- Fixture: 4 independent lif cells, `drive=0.0`, no edges (the connected
  fixture spikes without drive from its default emitter drive).

### Gap 2 — per-position local inhibition (CLOSED)

- Kernel: `lif_step(..., cap_groups)` scopes the first-R-in-order cap
  independently per group (`jnp.lexsort` by (group, visit); jit-traced
  clean); `None` keeps the whole-population cap. `simulate_lif_edge`
  takes `local_inhibition_groups` through.
- Construct (`_construct_lif`): groups derived from A's posed `(x, y)`
  (exact equality; `z` distinguishes competitors), recorded as
  `static["lif"]["local_inhibition_group_of_neuron"]`; missing/non-finite
  positions refused naming `local_inhibition`. Simulate passes them to
  the kernel with a population-length check.
- Fail-first tests failed on baseline (`TypeError: lif_step() got an
  unexpected keyword argument 'cap_groups'`; `KeyError:
  'local_inhibition_group_of_neuron'`); both pass after.
- Two existing route tests kept their intent on a co-located fixture
  (degenerate geometry point; random positions leave singleton groups
  and the cap inert): rounds=1 caps 3/step to 1/step (non-vacuous vs
  rounds=99), order/R variants differ.
- `test_emitters_lif + test_pcl_topology + test_lif_tensor_route`: 89
  passed; pins (`test_readme_atlas_pin`, `test_jdna_area_rules_055`)
  pass; `ruff check jaxfne/` shows only the 2 pre-existing HEAD errors
  (`emitters_lif.py` F401/F841, untouched).

### Gap 3a — lif-only labels through the lif bridge (CLOSED)

- `_lif_bridge_alias_tensor` (`jaxfne/_construct_lif.py`) samples lif-only
  types (`"simple"`/`"complex"`) under free Izhikevich names on the bridge
  only (`"simple"` prefers `"E"` when free, so AMPA sources keep P-023
  validity); counts, fractions, drives, geometry and connection selectors
  follow the rename, so RNG draws are preserved. Declared labels, PCL rows
  (simple 18 ms/10 mV, complex 50 ms/3 mV), triple drives and neuron-table
  rows are restored afterwards; the alias map rides `static["lif"]`.
- Fail-first `test_lif_pcl_labels_route_to_pcl_rows` failed on baseline
  (`ValueError: unknown Izhikevich cell type 'complex'`); passes after.
- `test_lif_tensor_route + test_emitters_lif + test_pcl_topology +
  test_readme_atlas_pin + test_jdna_area_rules_055`: 108 passed.

### Gap 3 — complex pooling + cover-table feed (feed + pooling CLOSED; Stage C OPEN)

- `dvs_feed_arrays` (`jaxfne/pcl_topology.py`) bins DVS events to
  `(feed_current, feed_arrivals)` per simple cell: exact oracle `cover_x`/
  `cover_y` cells, exact `w_s` weights per (depth, polarity, RF offset), one
  arrival per event; `np.add.at` so same-bin duplicate (bin, cell) pairs add
  (fancy `+=` is last-wins). Refusals name the feed (shape/finite/
  non-negative/whole-count). Fail-first on baseline
  (`ImportError: cannot import name 'dvs_feed_arrays'`,
  `tests/test_pcl_topology.py:864`); passes after. Mutation check: HEAD bytes
  over the file → the 2 feed tests fail; copy restored, sha256
  `3338ec4b808453bf56ae248e027a159a09c76c5894652598567158eba47ac8db`
  identical after restore.
- Feed kernel channels (`jaxfne/emitters_lif.py`): `simulate_lif_edge` takes
  `feed_current`/`feed_arrivals`; arrivals subtract R+A like edge arrivals,
  sustained `I_ext_pA` stays bare current (Gap-1 numerics unchanged).
  Fail-first on baseline
  (`TypeError: simulate_lif_edge() got an unexpected keyword argument
  'feed_current'`, `tests/test_emitters_lif.py:266`); passes after.
  Mutation check: HEAD bytes → the 2 feed tests fail; restored sha256
  `00a89242efee7d11192c377963d315352f418491eebbfb38e017a79f2146ea22`.
- Pooling: `(pre, post)` set-equality vs `c_cover` is pinned by the Stage-B
  tests above; `test_pooling_executes_broadcast_sum_through_edges` executes
  the fan-out (one driven simple spike arrives one step later with each
  covering cell's own `w_c` weight; characterization, green on baseline too —
  the new mapping is the S1 assembly in `test_pcl_stage_c`). Pooling
  positions are simple-grid indices against complex origins
  (`c_cover` in `pcl_event_ref.py`), not pixels.
- `tests/test_emitters_lif.py + tests/test_pcl_topology.py`: 70 passed.
  `test_lif_tensor_route + test_readme_atlas_pin + test_jdna_area_rules_055`:
  43 passed. Ruff on touched files: only the 2 pre-existing HEAD errors
  (`emitters_lif.py:40` F401, `:355` F841, outside this lane's diff).
- D = {} — no network-scale calibration attempted: there is no core mapping to tune yet.
- θ* = θ₀.
- X (remaining): the per-event depth-sequential cascade — one event's
  candidates visited in ascending index with immediate spike-then-inhibit
  (`_add_event` in `port/pcl_event_ref.py`) — has no clock-step expression; the cap
  resolves once per step, not once per event.
  |a| unquantified — no partial mapping exists to score. This is the protocol's step-6
  outcome: the paper's model needs something the core grammar does not yet state.
- Held-out result: none (no fit, nothing to hold out). Band: the oracle T2 band above stands.

## Claim

Needs X: full-network core cascade mapping (depth-sequential local inhibition, cover
tables, complex pooling as core-expressible dynamics). Core z at Fig. 3 scale is missing;
edge-scale core z (single cell, STDP pairs, delayed drive) is proven by the tests below.
Oracle v5 verdict (FAIL on T2) stands unchanged.

## Files and tests

- `jaxfne/emitters_lif.py`, `jaxfne/hdp_rules_pcl.py` (new); `tests/test_emitters_lif.py`,
  `tests/test_hdp_rules_pcl.py` (new, 13 tests); `port/run_tuning.py` +3 engine tests.
- Baseline demo: new tests fail on baseline without the modules
  (`ModuleNotFoundError: No module named 'jaxfne.hdp_rules_pcl'`, 2 collection errors).
- `tests/test_emitters_lif.py tests/test_hdp_rules_pcl.py`: 13 passed in 25.50s (WSL gate venv, CPU).
- With exports: 32 passed in 49.81s (both new files + contract + snapshot).
- `port/test_run_tuning.py`: 14 passed in 17.63s (Windows, oracle REF counts intact).
- GPU time: 0 — every run in this lane was CPU (JAX_PLATFORMS=cpu / Windows).

## Step 7 — Stage C event-order sweep + 20-clip gate (2026-10-14)

- Pinned oracle is the EVENT oracle (`runs/fig3_cSTDP3_S1`, pinned by
  `test_stage_c_oracle_20clip_record`: simple 418042, complex 365087 over the
  20 clips), i.e. `port/pcl_event_ref.py`, not `pcl_binned_jax.py` (its
  `R_ROUNDS = 4` per 100 us bin is a property of the binned port only).
- Oracle semantics, read from its code: one event's candidates visited in
  ascending index; a spike updates weights, then sends local inhibition to
  same-position depths (`:686-694`, gated on `local_s`), then lateral
  (`:695`), then forwards to complex cells in ascending index (`:697`,
  `:715-719`). No winner cap exists: every suprathreshold candidate in one
  event's pass can spike unless inhibited earlier in the same pass.
- Oracle winner census, clip (0,0), per 500 us bin per position
  (`_temp_oracle_capstat.py`): simple max 3 (305/20329 bins ≥2, 3 ≥3);
  complex max 2 (16/18261 bins ≥2).
- Declared setting, reasoned from the above (not from best r):
  `resolution="event_order"`, dt=0.5 ms, rounds=4. Each position resolves
  `n_rounds` winners per step (`lif_step` in `jaxfne/emitters_lif.py`); rounds=4 is the
  smallest probed value covering the oracle-observed max of 3 with one spare
  pass (rounds=2 caps below the observed max; rounds=8 admits far more than
  observed). dt=0.1 + rounds=4 is near-identical (combined r 0.8480 vs
  0.8541), so dt is not the lever; dt=0.5 stays (binary-exact, 5x cheaper).
- Clip-(0,0) sweep, core vs event oracle (foreground runs):
  | dt | rounds | simple core/oracle r | complex core/oracle r | combined r | total |
  | 0.5 | 1 | 16027/20637 0.7165 | 11150/18277 0.8368 | 0.7865 | 0.698 |
  | 0.5 | 2 | 18306/20637 0.7491 | 13163/18277 0.8261 | 0.8014 | 0.809 |
  | 0.5 | 4 | 21433/20637 0.8115 | 20627/18277 0.8402 | 0.8541 | 1.081 |
  | 0.5 | 8 | 24046/20637 0.8437 | 33049/18277 0.7831 | 0.8382 | 1.467 |
  | 0.1 | 4 | 21561/20637 0.8056 | 20461/18277 0.8330 | 0.8480 | 1.080 |
  (rounds=1 row from the prior probe; index-order-visit diagnostic at
  dt=0.1/r4 is bit-identical to event-time order on this clip.)
- 20-clip gate at dt=0.5/rounds=4/event_order (`_temp_stagec_20clip_eo.py`,
  foreground, wall 476 s): totals S 417011/418042 (−0.247%) and C
  381494/365087 (+4.494%) — both within 5%; mean per-clip per-cell r S
  0.8102, C 0.8453, both 0.8530 — FAIL on r < 0.95. `test_pcl_stage_c`
  core half stays xfail(strict); no flip.
- D = {event_order rounds sweep + 20-clip gate at the declared setting}.
  θ* = θ₀ (nothing tuned; rounds chosen from oracle code + census).
- X (remaining): per-event sequential semantics have no clock-step
  expression. Ceiling evidence: the exact binned port itself (same semantics,
  100 us bins) reaches only r 0.9334/0.9168 vs the event oracle on clip (0,0)
  (lane probe); the whole-step cap1 assembly reached combined 0.9288.
  Clock-stepping loses ~0.07 even before any core approximation.
- Held-out: none (no fit). Band: oracle T2 band above stands.
- Battery (foreground): `test_emitters_lif + test_pcl_topology +
  test_lif_tensor_route + test_pcl_stage_c + test_readme_atlas_pin +
  test_jdna_area_rules_055 + test_pcl_etude`: 125 passed, 1 xfailed
  (the Stage C core gate, failing as recorded). Ruff on the 9 touched
  files: only the 2 pre-existing HEAD errors (`emitters_lif.py:40` F401,
  `:457` F841; both lines byte-identical on HEAD).

## Step 8 — event_scan intra-step fixed point + Stage C 20-clip gate (2026-10-15)

- Declared addition: `resolution="event_scan"` on the local-inhibition
  declaration (`LOCAL_INHIBITION_RESOLUTIONS` gains the third value;
  default `whole_step` and `event_order` byte-identical). Unknown
  resolutions are refused by name, as before.
- Semantics (Hamm 2026-10-14 ruling: same-step interactions are a declared
  intra-step fixed point with rounds and order on A): inside one 500 us
  step the scan replays the event stream in `(timestamp, A-index)` order
  through one `jax.lax.scan`; each event excites its covered simple cells
  ascending with exact inter-event decay (`exp` from integer event times),
  and each spike runs local inhibition, then lateral (when flagged), then
  the forward pass — complex spikes run complex-local inhibition, then
  top-down — exactly as `_add_event` in `pcl_event_ref.py`. No rounds parameter:
  the stream order is the order. Builders: `event_scan_tables`
  (cover/forward/lateral/top-down transcription of `:393-477`) and
  `event_scan_events` (stable time-sorted `(M,4)` rows). Scan-only
  channels are refused by name under other resolutions and vice versa.
- Fail-first: `test_event_scan_matches_oracle_event_for_event`
  (hand-built 3-event clip: scan matches the oracle's two spikes, the
  `event_order` cascade keeps only the ascending winner) failed on
  baseline (`simulate_lif_edge() got an unexpected keyword argument
  'event_scan_events'`) and passes with the scan. Scan tests pass with
  `-W error::UserWarning` (float64 work runs under a local x64 context).
- Incidental defect found by the new test (pre-existing on HEAD,
  `emitters_lif.py` cascade scatter): a later group's pad slot (naming
  cell 0) clobbered cell 0's real results last-write-wins. Fixed with a
  dummy-cell scatter; the mutant (old scatter restored) fails the scan
  test (`[False, False, False] == [True, False, False]`), the fix passes.
- Clip (0,0) probe, dt=0.5 (`_temp_scan_1clip.py 0 0`, foreground):
  S 20637/20637 r=1.0000 exact 5184/5184; C 18277/18277 r=1.0000 exact
  1152/1152; both r=1.0000. First-spike times µs-exact on all 1371+702
  co-spiking cells (`_temp_scan_fs00.py`).
- 20-clip Stage C gate at dt=0.5 (`_temp_scan_launch.py`, 20 parallel
  foreground workers, wall 3523 s, all rc=0): EVERY clip is per-cell
  exact — S 5184/5184 and C 1152/1152 on all 20, every per-clip r=1.0000
  (S, C, both), totals S 418042/418042 and C 365087/365087 (rel 0%).
  Gate (r ≥ 0.95, total within 5%) PASSES on the edge-kernel path.
  Per-clip count matrices: `_temp_scan_1clip_<f>_<s>.npz` (scratch).
- Bin-level train identity, 2000-event slice (`_temp_scan_binexact.py`,
  S1 full-clip times filtered by the slice cutoff as the slice oracle):
  68/68 S and 104/104 C oracle spikes sit in core-spiked bins and the
  `(n_steps, N)` binned matrices are identical — BIN-TRAIN-EXACT.
- Slice cross-check vs live numba oracle (`_temp_scan_slice_valid.py`,
  2000 events): counts exact 5184/5184 + 1152/1152, first-spike exact
  68/68 + 102/102. (Its first draft used `np.minimum.at` for the
  oracle-side argmin and silently kept -1; replaced with
  `np.unique(return_index=True)` on the time-sorted gather.)
- `test_pcl_stage_c` core half stays xfail(strict): it exercises the
  tensor construct/simulate paradigm-drive path, which still refuses by
  name (now naming both `event_order` and `event_scan`); the passing
  numeric gate above runs through `simulate_lif_edge`. No flip: the
  xfailed path did not pass.
- D = {event_scan replay + 20-clip gate at dt=0.5}. θ* = θ₀ (no tuning;
  scan has no free parameter beyond the declared setting).
- Held-out: none (no fit). Cost note: ~2.3 ms/event XLA-CPU
  (~55 min/clip single-core); the 20-clip gate ran 20-way parallel.
- Battery (foreground): same 7 files: 130 passed, 1 xfailed (the tensor
  Stage C gate, unchanged). Ruff on the 9 touched files: only the same 2
  pre-existing HEAD errors (a third, `NX` unused in the new scan loop,
  was mine and is fixed).

## Step 9 — Stage C through the tensor route (2026-10-15)

- Declared mapping: the S1 scan rides the tensor route
  (`build_s1_scan_tensor` declares positions + `resolution="event_scan"`;
  the scan wiring — cover/forward/lateral/top-down tables + LI magnitudes —
  rides the `EventScanDrive` paradigm object, not new tensor fields).
  `construct_lif` compiles scan connections to no edges (anything else
  compiled + scan resolution is refused at simulate as unmapped, by name);
  `model.simulate(..., paradigm=drive)` dispatches to the same scan kernel
  (`_simulate_lif_scan`); `rounds` must be absent for `event_scan` and
  non-ascending order is refused (`validate_local_inhibition`).
- Slice proof first: real-data 300-event slice, clip (0,0)
  (`_temp_scan_slice_diag.py`): tensor route == `simulate_lif_edge`
  spike-for-spike (46/46), V_m all-NaN — plus the synthetic-slice unit
  test `test_lif_event_scan_tensor_route_matches_edge_kernel` (spikes,
  counts, first-spike metadata, NaN V_m) and the by-name refusal test.
  A mid-lane "probe failure" (clip job dead, empty log) diagnosed as a
  harness timeout kill, not a code fault: the log held only the startup
  JAX warning with no traceback, 8 sibling clips completed exact, and the
  slice replays exact.
- 20-clip tensor-route gate (`_temp_scan_tensor_1clip.py` × 20,
  `_temp_scan_tensor_summarize.py`; logs
  `_temp_scan_tensor_1clip_<f>_<s>.log`, matrices
  `_temp_scan_tensor_1clip_<f>_<s>.npz`, scratch): EVERY clip per-cell
  exact — 6336/6336 on all 20, every per-clip r=1.0000 (S, C, both);
  totals S 418042/418042 and C 365087/365087 (rel 0%), matching the
  pinned oracle totals. First-spike µs-exact on all 41794 co-spiking
  cells (`_temp_scan_tensor_fscheck.py` vs oracle
  `simple/complex_times_cell/us`); all 84926 silent cells agree.
  Gate (r ≥ 0.95, total within 5%) PASSES on the tensor route.
- `test_pcl_stage_c` core half stays xfail(strict): its own path is a
  `stimulus_schedule` probe on a generic lif tensor, not the tensor scan
  route — no flip. The suite's numeric evidence is the slice test plus
  this 20-clip receipt.
- D = {tensor-route scan mapping + 20-clip gate at dt=0.5}. θ* = θ₀
  (no tuning; scan has no free parameter). Held-out: none (no fit).
  Cost: construct ~11 s, sim ~55 min/clip single-core; 12-way relaunch
  after the timeout kill finished in ~49 min wall.
- Battery (foreground, `bash scratch/step9_gate.sh`): 69 passed,
  1 xfailed (the schedule-probe gate, unchanged) + topology stub
  10 passed. Ruff `jaxfne/`: only the same 2 pre-existing HEAD errors
  (`emitters_lif.py:40` F401, `:468` F841; both lines HEAD-byte-identical).
- Follow-up (2026-10-15): the xfail is flipped. `test_pcl_stage_c`'s
  schedule probe is replaced by the tensor route on a suite-feasible
  slice — first 300 cam0 events of clip (0,0) through
  `build_s1_scan_tensor` + `EventScanDrive`, trained weights read from
  the warehouse network with numpy only (port `_load_weights` layout,
  no oracle import; `layerInhibitions` pinned). The tensor binned spike
  matrix equals the oracle's on the causally closed steps (bins ending
  at or before the last sliced event; later bins may need post-slice
  events and are excluded both sides — first draft compared them and
  failed with 21 oracle-only spikes at steps 26-28). Committed receipt
  `artifacts/etudes/pcl/stage_c_tensor_route_20clip.json` (observed
  totals, per-clip r, first-spike summary) is checked against the pins
  and the gate by `test_stage_c_tensor_route_20clip_receipt`. Mutation:
  each of `pcl_topology.py` / `_construct_lif.py` / `_model_simulate.py`
  at HEAD bytes fails the slice test; restores sha256-verified. Gate
  now: 71 passed, 0 xfailed + stub 10 passed.
