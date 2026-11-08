# PCL-R1 fidelity audit - CLOSED (receipt)

R1 CLOSED - FIDELITY AUDIT (sol ruling 2026-10-23/24; Hamm-authorized). NOT PCL REPRODUCED.

## Environment and provenance

- jaxfne: dev 2a627caa (this clone; B2c engine=core landed). Oracle/scored side:
  Windows python 3.14 + numba 0.66.0. Core-side runs: the WSL gate venv (jax 0.11.2, CPU) where used.
- Matched inputs: the trained cSTDP3 NETWORK (F:/warehouse/pcl/data/PCL_datasets_and_networks/add_simulations_cSTDP/cSTDP3)
  + one real file test7/0/0/0/0.npz (camera-0, 829748 events). No tuning, no criteria change;
  the signed v5 T2 FAIL (pcl-port 8f75acd0) stands untouched.
- Reproduced (measured runs) vs inspected (code-path + refusal evidence), per row below:
  F1 measured, F2 measured, F3 measured (round-trip), F4 measured, F5 inspected (refusal runs),
  F6 measured (event counts + the same-run delta).

## Attribution refinement (sol ruling)

F6 is an observation/selection fidelity discrepancy, NOT a scientific-model difference - the
underlying model need not change for the measured output to differ. Same-run measurement:
DeltaY_selection = Y_same_run(4,4) - Y_same_run(0,0) = 422 spike counts (0/64 cells fire in (0,0);
422 in-window in (4,4); driven neighbors 339/363/271/386 prove selection). Inferred, not established:
correcting selection alone resolves T2.

## Design dependencies (for the authorized read-only R2/R4 design study)

F1 -> reference-faithful inhibition; F2 -> explicit training-file continuation; F3 -> canonical
serialization; F4 -> complete continuation state (Y = F(W0, C0, I_0:T; theta): matching W0 without
C0 does not preserve Y); F5 -> replay capability and adaptation semantics; F6 -> reference-faithful
observation selection. The design study is READ-ONLY (no runtime changes), compares a
reference-faithful compatibility realization vs a minimal general extension, maps every oracle
state variable to JaxFNE's X/H/W/B/K, and carries a continuation-equivalence test plus a
generalization-beyond-PCL demonstration. Findings below are preserved without reinterpretation.

---

# PCL-R1 fidelity matrix (scratch, untracked)

Matched inputs throughout: trained cSTDP3 NETWORK
(`.../add_simulations_cSTDP/cSTDP3`) + ONE real file
(`.../counterphase_gratingComplex/test7/0/0/0/0.npz`, camera-0 slice,
829748 events, raw times 47975–1866025). Oracle side = `pcl_event_ref`
(pure) / `pcl_event_ref_numba` (scored); `engine="core"` scan is the
core-side executor under test. No tuning, no criteria change; v5 T2 FAIL stands.

| # | Expected semantics | Oracle observed | Core observed | Attribution | Evidence |
|---|---|---|---|---|---|
| F1 local inhibition | C++ wires local inhibition only on `"local"` (`SpikingNetwork.cpp:351`); cSTDP3 config says `"static"` → C++ runs it with NO local inhibition | Port alias `"static"`→`"local"` (`pcl_event_ref.py:381`) turns local ON both layers (`True True True True`). Matched full-file numba: alias S=5530 C=6377 vs C++-wiring S=2262 C=37661; per-cell exact S 4385/5184, C 3/1152; r S 0.842, C 0.560. Pure prefix n=3000: same S=29 both; C 59 vs 671; inhibition deliveries simple 189219 vs 812096, complex 1829 vs 0 | Core scan tables inherit `net.local_s` (`run_tuning.py:225-238`), so the core executor applies local inhibition under the alias. Direct core counts on this file: REFUSED (see F5) | implementation | `_temp_pclr1_inhib.log:1-13`; `pcl_event_ref.py:380-391`; `run_tuning.py:225-238` |
| F2 training persistence | C++ tuning: weights persist across files in one run; `save()` resets neurons, never weights (`NetworkHandle.cpp:192-200`); `STDP_LEARNING=all`, 5 files × nb_run 2 | Pure-path prefixes (N=100000 of real files A,B): persistent A→B vs frozen-per-file weight Δ max(abs),mean: w_s (3.93, 0.128), wi_s (1429.4, 30.19), w_c (0.42, 0.045), wi_c (15.34, 1.81), LAT (760.3, 0.104), TD (1692.8, 0.233). Determinism (A persistent vs A frozen) exact 0.0. Scored full file C (numba): persistent-trained S=9496 C=10005 vs frozen S=5425 C=6252; exact S 4252/5184, C 593/1152; r S 0.465, C 0.851 | Port inference builds a fresh deepcopy per file with `learning=False` (`run_tuning.py:357-384`) → frozen-per-file semantics; scored side inherits whichever weight state it is handed | implementation | `_temp_pclr1_persist2.log:1-16`; `run_tuning.py:357-384` |
| F3 serialization round-trip | Weight maps round-trip through the C++ layout (WeightMap files in libstdcxx `unordered_map` iteration order, `pcl_event_ref.py:510-528`) | With map-order-correct save: reload array identity True (pristine AND persist-trained). Pristine scored identity exact S 5184/5184, C 1152/1152 (S=5514 C=6355 both) | n/a (oracle-side check; core consumes the same arrays) | implementation | probe 2c stdout (pristine/persist-trained identity lines); `libstdcxx_umap.iteration_order`; naive ascending-index save gives identity False (`_temp_pclr1_persist2.log:14`) |
| F4 scored state is not weights-only | n/a (new finding) | Same persist-trained weight arrays, two carriers: continuous trained-object deepcopy scores S=11355 C=11543; weight-copy into a fresh object (≡ reloaded) scores S=9496 C=10005 (matches F2 persistent arm). Non-weight carrier state: adapted `thr`/`rate`/`V`/`A`/`_last_update` (`_update_neurons`, `pcl_event_ref.py:746-762`; `Cell` slots `:187-205`) | Same consequence for any core handoff that moves weights without carrier state | implementation | probe 2c stdout (persist-trained continuous vs reloaded); `pcl_event_ref.py:746-762` |
| F5 core scan refusal on this file | Core replay returns counts with first-spike times only; frozen thresholds | n/a (oracle runs it: F1 numba arms complete in ~15 s) | `engine="core"` REFUSES: N=3000 prefix — "a cell spiked twice" (`jaxfne/_model_simulate.py`); N=10000 — "events reach past the measurement interval, where the oracle adapts rates and thresholds and the frozen-threshold scan has no expression". No core-side counts observable on matched input. Capability gaps for R4 (design only): (i) multi-spike-per-cell expression — the scan replays counts with first-spike times only and raises instead of reporting a cell that spikes twice (`_temp_pclr1_core.log:2`, `run_tuning.py:323-328`); (ii) past-interval adaptation — once events pass the measurement interval the oracle adapts rates/thresholds (`pcl_event_ref.py:746-762`) while the scan holds thresholds frozen and raises instead of expressing that dynamics (`_temp_pclr1_core.log:5`, `run_tuning.py:301-302`) | implementation | `_temp_pclr1_core.log:1-7` |
| F6 column selection vs drive | Tuning-curve analysis targets column (0,0) (neuron-creation order x,y,z; files 0–63 = column (0,0); notebook `tuningCurvesOutputSine` loops z_neur 0–63, `neuron_id=2560` unused) | test7/0/0/0/0.npz cam0: x 164–192, y 114–142 → 36/81 columns driven (i,j 1–6). Events in (0,0) RF pixels: 0; in (4,4): 100730. Notebook sine window [62500,2001000] clips 551 events (0.07%), 0 past end. SAME-RUN output (one numba run, standard scored config, Y = simple spikes/cell in window): (0,0) 0/64 cells fire, sum 0, mean 0.0000 — entirely silent, full run and window alike; (4,4) sum 422, mean 6.5938 (59/64 cells); ΔY_selection = 422 − 0 = 422 (mean 6.5938). Driven controls same run: (4,5) 339, (5,4) 363, (3,4) 271, (4,3) 386 — the (0,0) zero is selection, not a broken run (run totals S=5530 C=6377, = F1 alias arm) | Upstream of both engines (selection, not execution); B2c scan on this file refused (F5), so no core-side count to compare | scientific-model difference | `_temp_pclr1_samerun.py` stdout (see below); `python -u _temp_pclr1_select.py` stdout; `tuningCurves.ipynb:709-826` |

## Scratch probes (all untracked, repo root)

- `_temp_pclr1_inhib.py` → `_temp_pclr1_inhib.log` (F1; counting wrapper patches the shared inhibition update from scratch only, no tracked file touched)
- `_temp_pclr1_persist2.py` → `_temp_pclr1_persist2.log` (F2 + naive-save round-trip negative)
- `_temp_pclr1_rtrip.py` (F3, F4; map-order-correct save, pristine + persist-trained arms)
- `_temp_pclr1_core.py` → `_temp_pclr1_core.log` (F5; B2c `run_core_events` under both flag sets)
- `_temp_pclr1_select.py` (F6; numpy-only, no oracle construction)
- `_temp_pclr1_samerun.py` (F6 same-run ΔY: one numba run, standard scored config; stdout:
  `flags: local_s=True lat_on=True local_c=True td_on=True NX=9 NY=9 ND=64 NS=5184 NC=1152`;
  `single run wall=18.0s`; `(0,0) | 0 0.0000 | 0 0.0000 | 0/64`;
  `(4,4) | 423 6.6094 | 422 6.5938 | 59/64`; `(4,5) 340/339`, `(5,4) 363/363`,
  `(3,4) 271/271`, `(4,3) 387/386`; `dY_selection window [62500,2001000]: sum 422 - 0 = 422`;
  `(0,0) entirely silent: full=True window=True`; `run totals: S=5530 C=6377`)
- `_temp_pclr1_time.py` (timing: pure learning path 5000 events in 1.2 s — motivated the N=100000 rescope after the numba learning path raised `NotImplementedError`, `pcl_event_ref_numba.py:225`)

## Reproducible commands (Windows python with numba 0.66.0; from repo root)

- `python -u _temp_pclr1_inhib.py`
- `python -u _temp_pclr1_persist2.py`
- `python -u _temp_pclr1_rtrip.py`
- `python -u _temp_pclr1_core.py`
- `python -u _temp_pclr1_select.py`
- `python -u _temp_pclr1_samerun.py`
