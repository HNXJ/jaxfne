# P-056 diagnostic specification v1 (DRAFT, 2026-11-15; unsigned, no run authorized)

Separately versioned from the pilot's unrecorded metric. Any change after a candidate result is observed becomes v2; v1 numbers are never retuned. Todo T-90. Parent issue: `artifacts/assurance/issue_log/ISSUE_LOG.md` P-056.

## 1. Estimand (frozen before any run)

Column: Billeh 2020 wiring, Hagen 2016 cells, AdEx conductance route, Vogels 2011 iSTDP settle. Layers L2/3, L4, L5, L6 map to populations e2, e4, e5, e6 plus their interneurons by the digit in the name (the Billeh-to-Hagen map is a design input, recorded with the run).

- Contacts: one per layer, at the layer midpoint of the Hagen source bounds `0, 81.6, 587.1, 922.2, 1170.0, 1491.7` um, scaled to the column.
- `P_l` = band-integrated Welch power of the LFP at contact l: 8 to 25 Hz (package default), Welch Hann, 1 s segments, 50% overlap, `detrend=False`, `fs` = 1000/dt, window = after dropping the first 2.0 s, keep at least 10.0 s (`_ref42_window`). Alpha-beta peak: prominence of the 8 to 25 Hz maximum above a 1/f fit of 2 to 50 Hz excluding 8 to 25 Hz.
- `R_l = P_l / P_L2/3`. Deficit as logged: `R_L5, R_L6 < 0.005`.
- Reference population: L2/3 contact. Normalization: ratio of band powers, never per-channel max (`bandpower_jax` divides by the channel maximum; not used here).
- Numerical tolerance: a repeat run with a different seed must reproduce `R_l` within a spread measured from 5 seeds before any intervention; interventions are judged against that spread, not a fixed number.

## 2. Measurement validity (done and to do)

| control | status |
|---|---|
| positive: known 15 Hz deep drive, tone power = analytic | done, `tests/test_kernel_lfp_band_power.py` (5e-8) |
| negative: zero modulation; time-shuffled drive | done (1e-14; 0.7%) |
| column positive: inject a synthetic 15 Hz count into one deep class on the column geometry; R must equal the analytic ratio of kernel responses | done 2026-11-15 for L5E and L6E, apical and basal placement, all five layer contacts (rtol 3%), plus the whole-cell cancellation case; `tests/test_kernel_lfp_band_power.py` |
| null: spike-time shuffle per population in the real column run; alpha-beta peak prominence must vanish | to do with the baseline |
| absent vs defective: report per-layer rate, spike count and `|H(f)|` at each contact next to `P_l`; deep rate 0 means absent activity, deep rate > 0 with `R` near 0 means kernel or geometry | to do with the baseline |

## 3. Sequence

1. Column positive control (no network simulation).
2. Baseline: settle the AdEx column, record per-layer rates and N, compute `R_l` for 5 seeds. Report the deficit whether or not it reproduces.
3. One mechanism at a time, each its own intervention row, baseline preserved, failures recorded: deep rates (drive or iSTDP target rate), deep kernel amplitude (peak current per class), coherence radius, longer settle.
4. Independent review before interpretation; numerical artefact vs mechanism stated per row.

## 4. Sizing (estimate only; nothing run)

Not retained: pilot N, dt, duration. Planning values: dt 0.1 ms, 12 s (settle 2 s plus 10 s) = 120000 steps; N = 4000 first, 10000 only if the smoke passes.

| quantity | formula | N = 4000 | N = 10000 |
|---|---|---|---|
| dense recurrent weights, f32 | 4 N^2 B | 64 MB | 400 MB |
| per-class spike counts for the kernel (not per-neuron rasters) | 4 T C B | 120000 x 12 classes: 5.8 MB | same |
| per-neuron raster if kept as bool | T N B | 480 MB | 1.2 GB |
| kernel tensor | 4 C P K B | below 10 MB | same |

Keep counts per class, not rasters (the kernel input is class counts). Compile time and step cost are unmeasured: the smoke is 1 s of simulated time at N = 1000, reporting wall time, peak RSS and compile seconds, then scaled by N^2 for the recurrent term. A full run starts only after this spec is reviewed and the smoke is read.

## 5. Geometry-only prediction (derived 2026-11-15; no network run; scripts in a session scratchpad, not committed)

Hagen 2016 reduced cells of L23E, L4E, L5E, L6E (`merge=3`) on the source bounds, contacts at the five layer midpoints (40.8, 334.4, 754.7, 1046.1, 1330.9 um), unit synaptic weights, 15 Hz drive of equal amplitude on one class at a time, 8 to 25 Hz Welch power as in section 1.

| source class (placement) | own-layer contact power / L2/3 class at the L2/3 contact |
|---|---|
| L5E apical-targeted | 0.90 |
| L5E basal-targeted | 0.58 |
| L6E apical-targeted | 0.16 |
| L6E basal-targeted | 0.58 |

At equal drive and unit weights the deep kernels are within a factor 1 to 6 of L2/3, so geometry and kernel amplitude alone cannot give `R < 0.005`. Closing the gap to 0.005 needs a further power attenuation of 0.16/0.005 to 0.90/0.005 = 32 to 180; with power quadratic in drive amplitude that is an amplitude attenuation of 5.7 to 13.4 in the deep 8 to 25 Hz rate modulation or the deep synaptic current relative to L2/3 (inferred under the equal-drive assumption; rates not retained; a hypothesis, not a measured mechanism). That ranks rates and synaptic current above coherence radius for the intervention order.

Design caution (observed): synapses spread over the whole cell give kernels of about 1e-17 mV for L23E, L4E and L5E (an order 1e-12 below the apical or basal placements; consistent with near-complete cancellation of source and return currents, inferred) but 1e-5 to 1e-4 mV for L6E. The L6E exception is explained (observed): 7.5% of the L6E membrane area has its midpoint below the white-matter bound (z up to 1651 um against 1491.7), so a whole-cell weight range drops those compartments and the balance fails; L23E, L4E and L5E have 0% outside. A column must therefore place synapses by explicit depth ranges per class, never by the layer bounds alone. Synapse placement per connection class is therefore a design input.

## 5b. Smoke protocol (declared 2026-11-15 before any smoke run; the receipt is kept whether or not it passes)

- Configuration: N = 1000 split over the eight Hagen populations by a fixed table recorded in the receipt; dt 0.1 ms; 1.0 s simulated (10000 steps); seed 0; float32 unless the route requires x64 (then x64, stated); CPU.
- Budget (declared now): compile at most 180 s; execute at most 120 s; peak RSS at most 4 GB; output at most 50 MB.
- Timing: compile (first call minus second call) and execute (second call) reported separately, each with `block_until_ready`.
- Finite and coverage checks: all outputs finite; spike counts integer-valued and non-negative; per-class count array shape `[T, C]` equals the kernel's class axis; every population present.
- Input-engagement control (before any intervention is read): per-population external input actually reaches each population. Required: for every population, mean membrane input current or rate is non-zero and changes by the intended factor when its drive is scaled by 2 (a 2x drive on one population changes that population's rate by more than the seed spread; run this check with recurrent weights zeroed so that the other populations are expected unchanged, and again with recurrence on, reporting the downstream changes without a pass criterion). A deep population that does not respond to its own drive makes an intervention on it vacuous and stops the sequence.
- A failed smoke is reported with its log; nothing is retuned and re-labelled as the smoke.
- Result (2026-11-15, darwin arm64, jax 0.10.1, float32). The tree is at its file budget (2796), so no script or receipt file is added: the script is `git show a9d09921:scripts/p056_resource_smoke.py` (sha256 prefix 64dc5dda8a70523d; removed from the tip in the next commit, history keeps it), run on a clean tree at that commit, and the receipt follows in the next block, verbatim apart from whitespace: within the budget on every line. First call 6.37 s, second call 6.05 s (compile estimate 0.31 s), 6.05 s per 1.0 s simulated at N = 1000 with 100151 edges, peak RSS 0.58 GB; outputs finite, counts `[10000, 3]` non-negative integers, repeat run bit-identical. Engagement with recurrence zeroed: doubling the E drive moves the E rate 8.0 to 22.0 Hz (x2.75) and leaves PV (47.0) and SOM (10.0) unchanged. With recurrence on and iSTDP active: E 2.85, PV 24.8, SOM 6.46 Hz. Sizing, inferred from one point and an edge-linear cost: a 12 s run at N = 1000 is about 73 s; N = 4000 (about 1.6 M edges, 16x) is about 20 min per run; memory above N = 1000 is not measured (the route takes an edge list, so cost follows edges, not N squared), so a smoke at N = 2000 precedes it. Drive, weights and the E/PV/SOM table are arbitrary here and carry no P-056 meaning.

```json
{"schema":"jaxfne.p056_resource_smoke.v1","label":"resource smoke only: generic E/PV/SOM network, arbitrary weights and drive; says nothing about P-056","config":{"n_e":800,"n_pv":100,"n_som":100,"dt_ms":0.1,"n_steps":10000,"seed":0,"p_connect":0.1,"w_e_nS":1.0,"w_i_nS":3.0,"delay_steps":20,"drive_pA":300.0,"receptor_reversal_mV":[0.0,-80.0],"receptor_tau_ms":[2.0,5.0],"precision":"float32"},"budget":{"compile_s":180.0,"execute_s":120.0,"peak_rss_gb":4.0,"output_mb":50.0},"n_edges":100151,"timing_s":{"first_call":6.366013667080551,"second_call":6.054645749740303,"compile_estimate":0.3113679173402488,"execute":6.054645749740303,"zero_recurrence_base":6.078275667037815,"zero_recurrence_E_x2":6.164872499648482},"peak_rss_gb":0.578094482421875,"rates_hz":{"recurrent":{"E":2.8475,"PV":24.81,"SOM":6.46},"zero_recurrence_base":{"E":8.0,"PV":47.0,"SOM":10.0},"zero_recurrence_E_drive_x2":{"E":22.0325,"PV":47.0,"SOM":10.0}},"checks":{"spikes_bool":true,"counts_shape":[10000,3],"counts_nonnegative_integer":true,"w_final_finite":true,"H_final_finite":true,"populations_present":true,"deterministic_repeat":true},"engagement":{"E_rate_ratio_x2_over_base_zero_recurrence":2.7540625,"PV_rate_ratio":1.0,"SOM_rate_ratio":1.0},"within_budget":{"compile":true,"execute":true,"rss":true},"env":{"python":"3.13.7","platform":"macOS-27.0.1-arm64-arm-64bit-Mach-O","jax":"0.10.1","jaxlib":"0.10.1","numpy":"2.4.6","x64":false,"git_sha":"a9d099217f104cb9ab0a2784d8a4b85eb99a5a92","git_dirty":false}}
```

## 5b-review. Independent review of the N = 1000 smoke (2026-11-15; ACCEPT for resource and stability only)

Corrections to the result above, found by the reviewer and confirmed by its probes:
- Compile: `emitters_adex.py` has no `jax.jit`; the scan runs eagerly. First minus second call (0.31 s here, 0.05 s in the probe) is noise of about 0.3 s, not compile cost. A 10-step run takes about 0.12 s, so trace plus compile plus overhead is at most about 0.1 s; execution is 0.6 ms per step, linear in steps (n_steps 100, 1000, 10000: 0.175, 0.73, 6.1 s). Read "compile estimate 0.31 s" in the embedded receipt as "compile at most about 0.1 s; the field is noise". Compile is not separately measurable by call differencing here.
- Memory: `ru_maxrss` is in bytes on darwin (1 GiB allocation probe read 1.027 GiB); 0.578 GB stands.
- Determinism: spikes, `H_final`, `w_final_nS`, `V`, `w`, `g` are bit-identical across two runs (reviewer's check; the script compared spikes only).
- Rerun from a clean clone at a9d09921: every rate, check and count identical; timings 6.45 s and 6.10 s; RSS 0.577 GB.
- Scope limits: the engagement control above covers the E class only (E drive x2 with recurrence off). PV and SOM own-drive engagement and a recurrence-on drive-x2 run were not done, so the section 5b gate is not yet met for them. The smoke ran three classes (800, 100, 100), not the eight Hagen populations. The 50 MB output budget was declared but not measured. The script docstring cites the old spec path (`design/plans/`); the script lives in history, so the path stays.

## 5d. N = 2000 resource smoke, preregistered 2026-11-15 (result below)

- Configuration: as 5b except N = 2000 (E 1600, PV 200, SOM 200), p_connect 0.1 (about 400000 edges, 4x), 1.0 s at dt 0.1 ms, seed 0, float32, CPU. A new script version is committed first and the run is made from the clean tree; it carries `git_sha` and `git_dirty`.
- Prediction, recorded before the run: execute per simulated second 12 s if cost follows N (2x), 24 s if it follows edges (4x); RSS under 2 GB.
- Limits: execute at most 120 s; peak RSS at most 4 GB; output at most 50 MB, measured and reported. Compile is reported as not separately measurable; the receipt records a 10-step run as the trace-plus-compile floor and no first-minus-second difference.
- Checks: all N = 1000 finite and coverage checks; spikes, `H_final`, `w_final_nS` bit-identical across two runs.
- Engagement control (closes the 5b gate): with recurrence zeroed, each of E, PV, SOM in turn at 2x own drive must change its own rate by more than the seed spread, with the other classes unchanged; with recurrence on, E, PV, SOM x2 runs report downstream changes without a pass criterion.
- Failure rule: a failed limit or check is reported with its log; nothing is retuned and relabelled.

### 5d result (run 2026-11-15; script `scripts/p056_resource_smoke.py` at commit 093483a3 on branch `_temp_t90_smoke_20261115`, clean tree, darwin arm64, jax 0.10.1, float32; not independently reviewed yet)

Limits met: execute 20.7 s per simulated second (limit 120 s), peak RSS 0.995 GB (limit 4 GB, prediction under 2 GB), output 0.002 MB; 399489 edges. Cost prediction: 12 s if cost follows N, 24 s if it follows edges; observed 20.7 s, closer to the edges prediction (3.4x for 4x edges and 2x N, versus 6.05 s at N = 1000; inferred from two points, and edges scale as N^2, so N versus edges cannot be separated; an affine fit a*N + b*edges through both points gives about 28% N-linear cost at N = 1000, collinear terms, rate and cache effects uncontrolled, so any larger-N sizing is a range; separating the terms needs a run that varies p_connect at fixed N). The 10-step floor run took 0.385 s; compile is not separately measurable. Spikes, `H_final`, `w_final_nS` bit-identical across two runs; counts `[10000, 3]` non-negative integers; all finite.

Engagement, "recurrence-off" runs (receipt keys `zero_recurrence_*`; label kept as historical evidence, meaning zero initial weights with iSTDP on; see 5e), one seed (a seed spread was not measured, so "more than the seed spread" is untested; the single-seed ratios below are far from 1):
| class at 2x drive | own-rate ratio | other classes (ratio to base) |
|---|---|---|
| E | 2.70 (7.9 to 21.3 Hz) | PV 1.0, SOM 1.0 |
| PV | 1.81 (47 to 85 Hz) | E 0.88, SOM 1.0 |
| SOM | 2.00 (10 to 20 Hz) | E 0.97, PV 1.0 |

Own-drive engagement holds for all three classes. The "other classes unchanged" expectation fails for E when PV or SOM is driven (0.88, 0.97). Cause, inferred and untested: the "recurrence-off" run zeroes the initial weights, but iSTDP is on, and its I-to-E weights grow from zero during the run (`|w|` clip 0 to 100 nS) at a rate set by the presynaptic I rate, so I-to-E coupling is not off. A clean recurrence-off control needs plasticity disabled too (`plasticity_mask` None is "all plastic", so a non-plastic rule or a zero learning rate is needed). Recurrence-on rates (E 1.81, PV 16.1, SOM 5.45 Hz at base; E, PV, SOM x2 in the embedded receipt) are descriptive; the recurrent PV x2 run leaves E near silent (0.0056 Hz). The "more than the seed spread" clause of the engagement gate was not tested in 5d; 5g measures the spread for the eta-0 contrast only.

```json
{"schema":"jaxfne.p056_resource_smoke.v2","label":"resource smoke only: generic E/PV/SOM network, arbitrary weights and drive; says nothing about P-056","config":{"n_e":1600,"n_pv":200,"n_som":200,"dt_ms":0.1,"n_steps":10000,"floor_steps":10,"seed":0,"p_connect":0.1,"w_e_nS":1.0,"w_i_nS":3.0,"delay_steps":20,"drive_pA":300.0,"receptor_reversal_mV":[0.0,-80.0],"receptor_tau_ms":[2.0,5.0],"precision":"float32"},"limits":{"execute_s":120.0,"peak_rss_gb":4.0,"output_mb":50.0},"prediction":{"execute_s_if_cost_follows_N":12.0,"execute_s_if_cost_follows_edges":24.0,"rss_gb_below":2.0},"n_edges":399489,"timing_s":{"floor_10_steps":0.38540895795449615,"call_1":20.753095583058894,"call_2":20.700646749697626,"zero_recurrence_E_x2":20.407415125053376,"recurrent_E_x2":20.746331125032157,"zero_recurrence_PV_x2":20.034208457916975,"recurrent_PV_x2":20.099126709159464,"zero_recurrence_SOM_x2":20.042960042133927,"recurrent_SOM_x2":20.046326417010278},"peak_rss_gb":0.9949188232421875,"rates_hz":{"recurrent_base":{"E":1.80875,"PV":16.1,"SOM":5.45},"zero_recurrence_base":{"E":7.915,"PV":47.0,"SOM":10.0},"zero_recurrence_E_x2":{"E":21.345625,"PV":47.0,"SOM":10.0},"recurrent_E_x2":{"E":17.54375,"PV":30.44,"SOM":13.235},"zero_recurrence_PV_x2":{"E":6.939375,"PV":85.0,"SOM":10.0},"recurrent_PV_x2":{"E":0.005625,"PV":41.315,"SOM":1.165},"zero_recurrence_SOM_x2":{"E":7.676875,"PV":47.0,"SOM":20.0},"recurrent_SOM_x2":{"E":1.136875,"PV":9.75,"SOM":14.41}},"checks":{"spikes_bool":true,"counts_shape":[10000,3],"counts_nonnegative_integer":true,"w_final_finite":true,"H_final_finite":true,"populations_present":true,"state_identical_across_two_runs":{"spikes":true,"H_final":true,"w_final_nS":true}},"engagement":{"E":{"own_rate_ratio":2.6968572331017056,"other_rate_ratios":{"PV":1.0,"SOM":1.0}},"PV":{"own_rate_ratio":1.8085106382978724,"other_rate_ratios":{"E":0.8767372078332281,"SOM":1.0}},"SOM":{"own_rate_ratio":2.0,"other_rate_ratios":{"E":0.9699147188881869,"PV":1.0}}},"env":{"python":"3.13.7","platform":"macOS-27.0.1-arm64-arm-64bit-Mach-O","jax":"0.10.1","jaxlib":"0.10.1","numpy":"2.4.6","x64":false,"git_sha":"093483a330f980b532e8ef25cb4d5a15529aa3e4","git_dirty":false},"output_mb":0.002374,"within_limits":{"execute":true,"rss":true,"output":true}}
```

### 5e. Recurrence-off control: supported route and confound check (run 2026-11-15, N = 2000, seed 0, same configuration as 5d, script in a session scratchpad, not committed; single seed; not independently reviewed yet)

Route: `hdp_rule_params={"eta": 0.0}` on `simulate_adex_hdp_registered` (neutral rule per `resolve_istdp_params`); no kernel change. All runs have zero initial weights on all 399489 edges; 63969 are plastic I-to-E edges.

| rule | drive | E, PV, SOM rate (Hz) | I-to-E `w_final_nS` mean / max | non-plastic edges max |
|---|---|---|---|---|
| default iSTDP | base | 7.915, 47, 10 | 0.058 / 0.093 | 0 |
| default iSTDP | PV x2 | 6.939, 85, 10 | 0.078 / 0.158 | 0 |
| default iSTDP | SOM x2 | 7.677, 47, 20 | 0.076 / 0.093 | 0 |
| `eta` 0 | base | 8.0, 47, 10 | 0 / 0 | 0 |
| `eta` 0 | PV x2 | 8.0, 85, 10 | 0 / 0 | 0 |
| `eta` 0 | SOM x2 | 8.0, 47, 20 | 0 / 0 | 0 |

Observed: with default iSTDP the I-to-E weights grow from zero (weight evolution) and E moves under PV x2 (-12%) and SOM x2 (-3%) relative to the same-rule base (-13% and -4% relative to the eta-0 rate of 8.0 Hz); with `eta` 0 the weights stay exactly 0 in all three runs and E is 8.0 Hz in all three. So the 5d cross-population effect on E comes from plasticity-grown I-to-E weights, and the "zero_recurrence" label in the 5d receipt means zero initial weights only. Limits: one seed; "E rate difference is caused by the grown weights" rests on the eta-0 contrast of one configuration (inferred, not a dose-response); weight evolution (`w_final_nS`) and its effect on E firing are separate quantities and only the second was shown to differ here by the rate contrast. Compile stays UNKNOWN; seed spread unmeasured. No kernel change is needed for a recurrence-off control; a full recurrence-off control for P-056 additionally needs the owner-held baseline inputs (section 6).

### 5f. Seed-spread preregistration for the 5e contrast (declared 2026-11-15 before any multi-seed run; not run)

- Script: `scripts/p056_istdp_off_control.py OUT.json` on branch `_temp_t90_smoke_20261115` (committed immediately after this section, before the run; run from the clean tree; receipt carries `git_sha`, `git_dirty`; refuses to overwrite). Same configuration as 5d except `seed`.
- Seeds 0, 1, 2, 3, 4; each seed sets both the network draw and the simulation seed. Rules: default iSTDP and `hdp_rule_params={"eta": 0.0}`. Drives: base, PV x2, SOM x2. 30 runs, about 10 min.
- C1: with `eta` 0, `w_final_nS` is exactly 0 on every edge, every seed, every drive.
- C2: with `eta` 0, E rate under PV x2 and SOM x2 equals the base E rate exactly, every seed.
- C3 (default iSTDP): r = E rate at x2 over E rate at base, per seed, for PV x2 and for SOM x2. The effect is called larger than the seed spread if |mean(r) - 1| > 2 sd(r) (sample sd, n = 5). Reported separately for PV and SOM; the result is descriptive, not a threshold for P-056.
- C4 (default iSTDP): mean plastic I-to-E weight above 0 for every seed.
- Failure rule: a failed criterion is reported with its log; nothing is retuned. Compile stays UNKNOWN.

### 5g. Seed-spread result (run 2026-11-15; `scripts/p056_istdp_off_control.py` at `da7e895b`, clean tree, jax 0.10.1; all 30 runs completed, none failed; receipt sha256 prefix 71f606815663b3be; not independently reviewed yet)

Criteria from 5f: C1 pass (eta 0: every weight exactly 0, all 15 runs). C2 pass (eta 0: E rate equals base, every seed). C4 pass (default iSTDP: mean plastic weight above 0, all 15 runs).

C3, default iSTDP, ratio of E rate at x2 drive to base, seeds 0 to 4:

| drive | ratios | mean | sd | larger than 2 sd |
|---|---|---|---|---|
| PV x2 | 0.8767, 0.8762, 0.8763, 0.8767, 0.8764 | 0.8765 | 0.00025 | yes |
| SOM x2 | 0.9699, 0.9674, 0.9664, 0.9699, 0.9696 | 0.9687 | 0.0016 | yes |

Seed 0 reproduces the 5e values exactly. Multi-seed conclusion, separate from the single-seed 5e observation: across five network draws and simulation seeds the E-rate reduction under PV x2 (12.4%) and SOM x2 (3.1%) with default iSTDP is far outside the seed spread, and absent with eta 0 (trivially: eta 0 leaves all weights at zero, so this does not separate plasticity from any nonzero I-to-E weight; a fixed-weight control at the grown mean is unrun). Limits: drive is deterministic current and the iSTDP step consumes no random key (source read, reviewer), so the spread is the connectivity draw only, which probably makes it small (inferred); five seeds; one configuration; the receipt's `config.seed` reads 4 because the script mutates it per seed (the `seeds` field lists all five). Compile stays UNKNOWN.

```json
{"schema":"jaxfne.p056_istdp_off_control.v1","seeds":[0,1,2,3,4],"E_rate_hz_by_run":{"default_iSTDP":{"base":[7.915,7.909375,7.90625,7.903125,7.90875],"PV_x2":[6.939375,6.93,6.928125,6.92875,6.93125],"SOM_x2":[7.676875,7.651875,7.640625,7.665625,7.668125]},"eta_0":{"base":[8.0,8.0,8.0,8.0,8.0],"PV_x2":[8.0,8.0,8.0,8.0,8.0],"SOM_x2":[8.0,8.0,8.0,8.0,8.0]}},"w_plastic_mean_default":{"base":[0.058194508,0.058119718,0.058166938,0.058320637,0.058064631],"PV_x2":[0.078053136,0.077836828,0.077938716,0.078209459,0.077786859],"SOM_x2":[0.075980247,0.075497148,0.075407419,0.075703593,0.075799476]},"w_max_eta0":0.0,"criteria":{"C1":true,"C2":true,"C4":true},"git_sha":"da7e895b8b702c76e5f0ab2950689bf17c393e8f","git_dirty":false}
```

The JSON above is a condensed extract (E rates, mean plastic weights, criteria); PV and SOM rates in every run were the declared 47/85 Hz and 10/20 Hz (PV 47 base, 85 at PV x2; SOM 10 base, 20 at SOM x2). The full receipt is kept at `receipts/p056_istdp_off_control_v1.json` on branch `_temp_t90_smoke_20261115` (not in this tree, file budget); the script at `da7e895b` also regenerates it. Independent review 2026-11-15: ACCEPT with six minor corrections, applied here.

## 5c. What the AdEx route supports today (inventory 2026-11-15, read from source at `fdb62a93`; nothing run)

- Labels are `E`, `PV`, `SOM` only (`make_adex_params` refuses others, `emitters_adex.py:97-101`), so the 17 Billeh populations need a declared map to three classes; the module has no per-population N, no synaptic weight table and no positions (`billeh2020.py`).
- The HDP kernel (needed for iSTDP) takes one Poisson drive set (one weight, receptor, delay); per-population Poisson weights exist only in the plain kernel. Per-population current drive (`I_ext_pA`, `[N]` or `[T, N]`) is available in both.
- The kernel returns the spike raster `[T, N]` bool and has no per-step per-population count output. At N = 1000 and 10000 steps the raster is 10 MB, so counts are formed after the run (`raster @ onehot`); a scan-body change is needed only at large N.
- Each call re-traces the scan (no `jax.jit` in `emitters_adex.py`), so compile and execute timing in section 5b needs two calls with identical shapes; largest tested AdEx N is 100.
- Consequence: the N = 1000 smoke below is a resource smoke on a generic E/PV/SOM network (current drive, fixed table of E 800, PV 100, SOM 100, receptor reversals 0 and -80 mV). It measures cost, stability and input engagement. It says nothing about P-056, and its receipt is labelled so. The P-056 baseline additionally needs the population map, per-class N, synaptic weights and drive amplitudes: scientific inputs for Hamm, not defaults for an agent to choose.

## 6. Owner rulings needed before any candidate result is read

- The threshold for "deep alpha-beta present" (the issue gives none; proposal: `R_L5` or `R_L6 >= 0.05` and peak prominence above the shuffle null's 99th percentile). Without it only the baseline and controls are run.
- Billeh-to-Hagen population map, and the Billeh-to-{E, PV, SOM} class map the AdEx route requires.
- Per-class N, synaptic weights (nS) and external drive amplitudes for the baseline column (none is in the repository).
