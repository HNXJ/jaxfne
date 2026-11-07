# todo stack

One ordered path, end to end: the 0.5.10 release first (checklist and score in
`etudes/meta/plans/release_plan_0.5.9-0.6.0.md`), then 0.5.11 and 0.6.0. Each step names its closing check
and where the path goes next; `→` is the next step, `⇒` a branch on the outcome. `[A]` agent, `[H]` Hamm: an `[H]`
node is a decision the path waits on, asked one at a time. Steps marked `∥` run beside the path in their own lane
(at most 3 lanes, one writer per file set, clones at `E:/repos/_wt/jaxfne/<lane>`, launched from a fixed commit).
Done steps are deleted; history is in git (the sectioned form before this rewrite: `git show 00eade8d:artifacts/todo_stack.md`).
Problems: `open_stack.md`; issues: `assurance/issue_log/ISSUE_LOG.md`; rules B1-B8:
`control/harness/evidence_and_workflow_rules.md`; paper goal G-0: `programme/design/plans/arxiv_paper_plan.md`.

## 0.5.10 (checklist and score in the release plan's 0.5.10 section, approved by Hamm 2026-11-05; 16/100 on 2026-11-07)

4. T-3 [A] 10E PCL Fig 3: R5-approximate (the core on all 3168 tasks under signed R4b, driver
   `F:/warehouse/pcl/r5/run_r5.sh`, resumable) scored in `pcl/prereg/`; P-055 (R5-exact FAIL, 4 of 16 files)
   attributed. History: R3 exact on files 0-4 (CALIBRATION R3), R4b signed 2026-11-03, R5-exact scored
   (`P3_fig3_r5_exact_result.md`, S13). Status 2026-11-06: resumed detached 16:54Z (`F:/warehouse/pcl/r5/resume_approx_core.sh`,
   PATH now exports Git), 1085 tasks pending; 777 done at 2026-11-07 22:18Z (2860 of 3168), finish about 2026-11-08 10Z
   (inferred from the rate); the driver's logged `rc=$?` is wrong (`$(stamp)` resets it): fix before the next resume.
   Then T-22, T-25: PCL B3 on the intra-step operator; trained weights load as A' (`with_trained_weights`).
   Closes when: the R5-approximate result file and score JSON are pinned and P-055 names its cause.

## Beside the path (∥)

11a. ∥ T-72 [A then H] NeuroAI 2027 short paper, deadline 2026-11-20 23:59 AoE (paper plan amendment 2026-11-02;
    rules verified on the OpenReview form, non-archival, no dual-submission bar). Status 2026-11-02: draft in the
    private thesis repo, branch `_temp_neuroai2027`, `artifacts/notes/neuroai2027/`;
    3 pages main text, 2 figures, Table 1; three critic rounds, the round-3 fixes unreviewed. The submission build
    (pdfLaTeX, TeX Live 2026 in WSL, the kit's unmodified style; Hamm's choice) passes: 4 pages, 0 overfull boxes, all
    fonts embedded, no Type 3. Waits on [H]: the anonymous.4open.science mirror of the code repo (Hamm's choice;
    the link replaces `[MISSING: anonymized code link]`), then Hamm reads and submits. Closes when Hamm has submitted.
21. ∥ T-C2A3 [A] Left from the A3 lane (the C2 rail question closed 2026-11-07: model regime; the canonical
    hebbian×linear pair is the measured divergent cell B-05, `emitters_homeostatic_ei.py:301`; the rail fraction
    follows the bound, 0.68 at ±10, 0.985 at ±100 and ±1e6, unchanged with drive halved; n 16, seed 0, 2000 ms):
    spec-level neuron permutation needs order-independent construction. Muse inventory 2026-11-07 (960a0fab): RNG
    streams keyed by position at `_construct_population.py:449,504,527-537,602,699,749`, `_construct_core.py:234`,
    `connectivity.py:537,626,918-919,941,953,729-736,1214,1383`; the fix re-keys them on stable ids, which changes
    every seeded realization (O-91). Closes when: O-91 is ruled and, if kept, A3 has its permutation test killed by
    a mutant.
20. ∥ T-B1 [A] Scored CPU vs RTX A4000 benchmark & memory profile (F4, `artifacts/runs/benchmark/PREREG_B1_0510.md`):
    profile whether a GPU upgrade is justified ($0 spend now; purchase gate based on measured bottlenecks).
    Metrics recorded per cell across N (1e2 to 1e5): compilation time (CPU/JAX compilation overhead),
    steady-state simulation time (GPU throughput), peak GPU memory (capacity requirement; distinguish reserved
    from live/peak), peak host RSS (construction and recording overhead), host-device transfer time (PCIe sensitivity),
    numerical agreement (CPU/GPU validity). Controls: negative control with recording disabled and another with
    reduced recording. Purchase gate: peak VRAM approaches 16 GB despite chunking -> 32-48 GB GPUs; VRAM fits
    but GPU execution dominates -> benchmark faster GPU; CPU construction/compilation dominates -> platform/CPU
    changes; no material bottleneck -> retain A4000. Closes when: benchmark receipt records all 6 metrics across
    N for CPU and A4000 with recording controls, evaluated against the purchase gate. Status 2026-11-06: prereg
    signed and pinned; jaxfne harness `artifacts/runs/benchmark/b1/b1_harness.py` (CPU smoke M1, M2 at N=100; M3 at
    N=1, 10, 100 after the `jax.random.split` fix 2846ee9d, 2026-11-07). The M3 class waits on O-90. Left: review the M1 external-drive scaling rule; the baseline scripts (`b1/b1_brian2.py`, `b1_nest.py`,
    `b1_jaxley.py`, on dev 1e7ee1e5; Brian2 2.10.1 and NEST 3.10.0 in WSL `~/venv_b1_baselines`, Jaxley on Windows)
    type their model constants with source citations: read them from the lane files before scored runs; then the
    scored runs.
33. ∥ T-G1 [A] 10G G1 CAMINOS panels (O-88 on the Cell Reports panels). PLoS text read 2026-11-07 (PMC13286281,
    copy `F:/warehouse/caminos/plos_text/`). Fig 5 is mean-field only: 5A sweeps PV→SOM and SOM→SOM on the full mean
    field (frequency, amplitude, bifurcation diagram); 5B is the reduced Eq. 2 (no synaptic delay, no adaptation)
    beside a spiking raster. 5A producer landed (c3719c79, `fig5a_sweep`); its 0.5 Hz peak-to-peak threshold cut
    small sustained cycles, now replaced by a sustained-cycle test (a 16 s probe from the perturbed fixed point held
    E peak-to-peak constant at SOM→SOM 1.1 and 1.3 nS): PV→SOM ends at 0.7-0.8 nS (figure about 0.7), SOM→SOM near 1.33 nS (figure about 1.7). The
    authors' code form of T_v gives 1.5-1.7 nS and 35.00 Hz (probe; O-93). Panel in `plos_suite.ipynb` (a6f4d867);
    left: re-export the dashboard notebook. Fig 4A mean-field lines: no producer. Fig 3
    mean-field rows: the paper defines randomization only for the network (a fraction replaced by Poisson units at
    the reference mean rate, Results ¶ "silencing and randomizing"); the mean-field treatment is unstated, so the
    build declares one before the run (inferred: the fraction enters as a constant input at the baseline rate).
    Cell Reports 2B bottom (producer `cellrep_figures.py:405`, never run)
    and 6B (`:439`, gain unstated) wait on O-88. Closes when: each kept panel is in its exported notebook
    (`suite/plos_suite.ipynb`, `suite/cellrep_suite.ipynb`) from pinned runs.
27. ∥ T-48 [A then H] 10D D2 Jansen-Rit 1995: the emitter is on dev (cf706780, every parameter required); the
    paper's constants and the Fig. 3 C sweep read into `etudes/papers/jansenrit1995/SOURCE_P1.md` (2026-11-07).
    P3 drafted, unsigned (`jansenrit1995/prereg/P3_v1.md`, driver `run_jr_fig3.py`): regime class, alpha
    sharpening, spike-wave slowing and EEG level per C; pilots seeds 0-2 close E1-E4. Blocked by: Hamm's
    signature (thresholds, 2 ms hold, aggregation). Closes when: the P3 is scored on seeds 10-14.
14. ∥ T-O68 [A then H] Local copies (O-68). Status 2026-11-07: every `E:/cl-*` and `E:/repos/_wt/jaxfne/*` clone
    audited (dirty files, `git cherry` against dev, alternates borrowers); dirty diffs, untracked files and the
    `clogsweep`/`pclr2study` bundles rescued to `F:/warehouse/jaxfne_clone_rescue/` (`HEADS.txt` lists each);
    the delete script `artifacts/_temp/delete_stale_clones_2026-11-07.ps1` waits on Hamm (O-89). Keep `E:/cl-jnwb16e`
    until jnwb-orchestrator confirms its pick. Left for agents: linked worktrees with dirty files
    (`base053`, `bisw1`, `frozen-etude`, `w3`, `jaxfne-wt-p022`, `jaxfne-wt-suites`, one agent worktree) and the
    `C:/workspace` leftovers. Closes when: every O-68 path is gone or rescued, and no surviving clone's
    `objects/info/alternates` names a deleted one.
16. ∥ T-46 [A] C9, C2: 8 gaps in `programme/paper/reproducibility_audit.md` section 7 (left: data and code statement,
    PCL natural-image and DVS128/N-MNIST licences, S16 generators). Close test (Hamm 2026-11-02): section 7 closes
    when the gaps touching 0.5.9 items close; the paper-only gaps (licences, data and code statement) move to the
    paper lane with a dated note.
17. ∥ T-30 [A] Conformance for the rest (`scripts/replication_known_violations.json`; the S1 entries are Hamm's
    exceptions): GSDR (HH in `construct()` dispatch, estimators), Destexhe (estimators, LEDGER, RESULT), pcl_h and
    Q-learning (phase records; pcl_h `sys.path` imports), PCL (closes at R4). Closes when: `check_replications.py
    --check` lists no violation without Hamm's exception.
18. ∥ T-8 [A] Standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev; then the lane
    dashboard (skill `jaxfne-dashboard`). Standing: never closes; each score's commit cites the regenerated
    dashboard.
19. ∥ T-10b [A] jnwb cross-learning: a proxy-claim wording gate, or drop it. T-42 [A] CI plan steps observed at the
    first rc. T-49 [A] split closed lane folders by kind as each lane closes (`caminos/`, `izhikevich2003/`,
    `brunel2000/`). Closes when: T-10b has its gate or a dated drop note; T-42 lists the CI steps seen at the rc;
    each closed lane folder passes `tests/test_artifacts_tree_degree.py` with no R6 entry.

## Owner queue (asked one at a time, in this order)

25. [A] Retired names in dated records kept as history (Hamm 2026-11-02, O-74 to O-79 closed): jarness selfcheck needs
    a history allowlist for them (asked in jchat, jynx keeper).
32. 0.5.10: [H] a named target figure for Lee, Pennartz and Mejias 2025 and for Mazzoni 2015 (both "unscheduled" on
    the dashboard roadmap, O-87). Closes when: each has a named figure (then a package row) or a dated drop note.

## 0.5.10 rows, then later

31. 0.5.10: [A] AGSDR log-scale sweep (Hamm 2026-11-03: "dx and dy could be logarithmic in change; free
    parameters would sweep logscale; 0.1 1.0 1.0 10.0 ... are equally swept"; freeze or restrict a parameter by
    leaving it out or narrowing its bounds). Bounds `(lo, hi, "log")` run the AGSDR loop in ln(theta):
    landed 733b3960 with `tests/test_agsdr_log_scale.py`. Left: use it in the GSDR reproduction and the fast sweeps
    of further models: a new GSDR fit pilot needs a new prereg (the signed P3 fit stays as scored). Closes when:
    `tests/test_agsdr_log_scale.py` passes with linear bounds bit-identical to before, and the GSDR lane names
    which parameters sweep on a log scale.
24. 0.5.10: [H] P-052 surrogate fitness for training, deferred by Hamm 2026-11-03
    (`artifacts/runs/perf/profiles/precision_058.md`). Closes when: Hamm rules on the sign and T-growth record.
26. 0.5.10: T-62 Cell Reports Fig 1D, excluded for 0.5.9 (Hamm 2026-11-02): the raw Onorato data is on request
    from the lead contact; Hamm decides at 0.5.10 whether to request it. T-32 seed and candidate
    batching for AdEx, LIF and `Model.tune` (`_model_tune.py:451`), core homes for lane analysis, Mejias G6, the
    LIF edge kernel on GPU; F2, F4, S16 per the release plan.
6. 0.5.10: T-R28 [A] Hamm's rulings 2026-10-28 left: dashboard P1 then P2; P-052 beta by horizon scan against
   the central finite difference (each output change with a changelog fragment); the jitter reader reruns the
   frozen readout on larger sets first, `w_max` only if it stays weak; paper: C4 Euclidean radius cutoff with an
   arbitrary-angle S3 test, C6 `geometry="laminar"` with Brian2 and Jaxley baselines declared. Status 2026-11-03
   (Muse sweep at bb3680d5): dashboard P1/P2 open (ruling text not in the repo,
   `execution_plans_0.5.9-0.5.10.md:72`); P-052 scan recorded (`precision_058.md:32-37`), beta ruling deferred;
   jitter reader open (no code or receipt found); C4 radius test landed (`tests/test_geometry_radius_rotation.py`),
   the S3 panel not built; C6 inventory only, baselines none run. Closes when: each listed piece has a commit with
   its test or receipt, cited here before the row is deleted.
28. After the 0.5.10 release and 7 replicated papers: T-54 [A then H] awesome-jax PR with Hamm's go.
29. 0.5.11 and 0.6.0: the release plan's later sections.

External and gated: T-11 CAMINOS NWB excerpt for jnwb `tests/` (waits on jnwb's path); T-RG komposer `run_guard.py stop` cwd fallback. Deferred, each
with its trigger: AT-10-R6 RSS growth before any run over 100 s; A8 gh-pages policy; architecture split on a
measured bottleneck; C5-C7 with a signed-zero/NaN contract; the UNTESTED-exact refusal tail and O-67 skips; S27
population family; agent transport (MCP); 0.5.4 items 1b and 0; 2026-09-20 owner calls (`compile_step_fn
**hdp_kwargs`, `validate_hdp_params`, P-003); Atlas ladder; the mechanism rule proposal; stored-only fields
(`LaminarColumnConfig.cell_colors`, `cell_signs`, `cy_m`, `HDPRuleContext.exc_mask`, `SanityDeltaConfig` asserts)
when a caller needs one; the tracked `protocol_w/w3b_parameter_domain/w3b_scan_run.log` at the next W3b refreeze.
Directions, open: augmentation runs; `pcl_stdp_h2`; structural HDP; primate primary-connection mode; theory F3-F5
and Lean proofs; Python 3.15 in CI once its wheels exist.
