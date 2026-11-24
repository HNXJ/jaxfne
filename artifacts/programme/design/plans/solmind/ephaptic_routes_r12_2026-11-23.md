# Routes that run a `drive_bias` rule (`ephaptic_field`), Solmind plan R12, 2026-11-23

Read at da42b095 (same `jaxfne/` as 62059ce5) by a Haiku worker; table checked by the lead. O = read in code at
the cited line; I = inferred. `grep -rEn 'hdp_rule=|hdp_rule_params|registered' jaxfne --include=*.py`: 233 hits
in 34 files.

## Routes that accept the rule

The unit contract lives only in the rule docstring (`jaxfne/hdp_rules_ephaptic.py:9-15`); no route checks it.
`HDPRuleContext` (`jaxfne/hdp_rule.py:86-115`) carries no route or family field, so the rule cannot refuse by route.

| # | route | bias enters | `drive_bias` update | alpha unit | lag (steps) | dtype | `v_rest` |
|---|---|---|---|---|---|---|---|
| 1 | `simulate_edge_recurrent_izhikevich_hdp_registered` (`_hdp_registrable_kernel.py:54`) | `drive + sched_t + bias + syn` at :384, :423, :562, :603, into dv/dt (`emitters.py:312`) | `_apply_rule` :276-288, gate :150 | 1/ms (I) | 2 (rule reads pre-update v, :396, :436) | float32 default :61 | required; None refused (`hdp_rules_ephaptic.py:103-104`) |
| 2 | Izh `Model` `_simulate_arrays` (`_model_simulate.py:458`; registered branch 775-807) | as 1 | as 1 | as 1 | 2 | `compute_dtype` :786 | as 1 |
| 3 | Izh continuation `_simulate_continuation_arrays` (:1260) and `compile_step_fn` (`_pipeline.py:468`) | as 1 | as 1 | as 1 | 2 | `v0.dtype` | as 1 |
| 4 | Izh `simulate_batch` (`_model_simulate.py:3456`; registered 3567-3586) | as 1 | as 1 | as 1 | 2 | `compute_dtype` :3578 | as 1 |
| 5 | `simulate_lif_edge` (`emitters_lif.py:2545`; `use_b` :2810) | `I_tot + bias` :3148-3149, before `lif_step` :3157 | jump add :3217-3223 | per-step voltage increment; alpha/dt in 1/ms (I) | 1 (ctx v = V_n :3166) | `tau_m_ms` dtype :2761 | required |
| 6 | LIF `Model` `_simulate_lif` (`_model_simulate.py:2877`; family check :3017) | as 5 | as 5 | as 5 | 1 | as 5 | required |
| 7 | `simulate_lif_seed_batch` (`_model_simulate.py:3857`; checks 3934-3947) | as 5 | as 5 | as 5 | 1 | as 5 | required |

## Routes that refuse or ignore

| # | route | behaviour |
|---|---|---|
| 8 | LIF event scan `_simulate_lif_scan` (:3232) | refuses `enable_hdp` (:3285-3290; kernel `emitters_lif.py:2503-2507`) |
| 9 | AdEx `Model` `_simulate_adex` (:2194) | refuses (`_construct_adex.py:541-563`) |
| 10 | `simulate_adex_hdp_registered` (`emitters_adex.py:1901`) | refuses `drive_bias` (`_hdp_family_kernel.py:106-110`) |
| 11 | `simulate_adex_seed_batch` (`_model_simulate.py:3763`) | ignores `enable_hdp` silently (no read in 3763-3854) |
| 12 | `simulate_adex` (`emitters_adex.py:832`) | no rule parameter (TypeError, I) |
| 13 | HH `Model` `_simulate_hh` (:2417) | refuses (:2440-2445) |
| 14 | `simulate_hh` (`emitters_hh.py:586`) | no rule parameter (TypeError, I) |
| 15 | rate `Model` (`_construct_rate.py:301-305`) | refuses |
| 16 | homeostatic E/I (`_model_simulate.py:1681-1694` -> 1961) | `Configuration.runtime(enable_hdp=True)` refuses at construct; a simulate-time runtime runs and drops the request |
| 17 | Izh with `enable_hdp=False` and HDP params (`_model_simulate.py:721-735`) | ignores the params silently |

Run-time evidence (`tests/test_hdp_rules_ephaptic.py`, R12 refusal half): rows 8, 9, 10, 13 and 15 raise
`ValueError`; rows 11, 16 (simulate-time) and 17 run, and their refusal tests are strict xfails pinned to P-069
(each fails under `--runxfail` with "DID NOT RAISE"). Rows 12, 14, 18 and 19 were not run.
| 18 | legacy `simulate_edge_recurrent_izhikevich_hdp` (`emitters.py:3395`) | registered name raises (:4325-4327) |
| 19 | `hdp_network.run` (`hdp_network.py:337`) | `hdp_rule_params` -> TypeError (I) |

## Findings

- One alpha means two things: 1/ms on the Izhikevich routes (1-4), a per-step voltage increment on the LIF routes
  (5-7). Plan ruling (b) decides whether LIF moves to rate units.
- Routes 11, 16 (simulate-time) and 17 drop an HDP request without an error; R12 asks that an unsupported route
  refuse. Issue P-069.
- Plan R12 cites `emitters_lif.py:2815` and `:3154`; the current lines are 2810 (`use_b`) and 3148-3149 (bias add).
