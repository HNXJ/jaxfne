# Release plan 0.5.6–0.5.10: replications plus narrowing

Requested by Hamm, 2026-10-11. This is a draft, so every release, tag and API addition
still needs his word.

Each release carries:

1. Three or four paper replications, each with its own preregistration that Hamm signs in
   a batch.
2. A narrowing sweep over code, docs and skills. The sweep ends with less tracked
   volume, and it does not lose a test.

Citations marked *(mem)* were written from memory. P1 checks them against the source
before any work starts.

## Baseline volume (dev 69bd4274, measured 2026-10-11)

| Area | Files | Size |
|---|---|---|
| tracked total | 2529 | 126.0 MB |
| `jaxfne/` package | — | 74.9k lines of Python |
| `tests/` | 443 .py | 77.8k lines |
| `docs/` | 502 | 42.2 MB |
| `artifacts/` | 1179 | 35.2 MB |
| skills (`artifacts/skills` and others) | 17 SKILL.md | — |

## Releases

| Release | Theme | Papers | Library change (needs Hamm) |
|---|---|---|---|
| 0.5.6 | Gamma and V1 tuning | 1. N'dri et al. 2025 PCL, Figs. 2–3 on the authors' trained network. 2. Tahvili, Vinck, di Volo, PLoS CB 2026: CAMINOS Fig. 1, plus PV silencing and randomisation (Figs. 2–3). 3. Tahvili et al., Cell Reports 2025, if its STAR Methods can be read; otherwise Izhikevich 2003 *(mem)*, the 1000-neuron network with its rhythms, on the existing emitter | AdEx emitter with conductance synapses (stage B of L2) |
| 0.5.7 | LIF and balance | 4. Brunel 2000 *(mem)*: AI, SR and SI states. 5. Vogels et al. 2011 *(mem)*: inhibitory plasticity restores balance. 6. Calcium-regulated synaptic homeostasis (bioRxiv 2025, candidate #10, lane L4). 7. PCL Fig. 4: surround, orientation-tuned and cross-orientation suppression | a working LIF emitter (a placeholder today) |
| 0.5.8 | Laminar | 8. Potjans and Diesmann 2014 *(mem)*: the cortical microcircuit, per-layer rates, on GPU in float32. 9. Mejias et al. 2016 *(mem)*: laminar multi-area rate model, gamma up and alpha down, feedforward and feedback. 10. Spectrolaminar motif (Nat Neurosci 2023, with the 2025 reply; lane L3) as a target observable on 8 and 9 | laminar readout helpers, if 8–10 need them |
| 0.5.9 | Predictive coding | 11. Multiple interneuron types and oscillations during predictive coding (candidate #4). 12. Lane L5, predictive coding. 13. PCL Figs. 5–6: spikes removed against N-MNIST and DVS Gesture accuracy, and sparseness. 14. PCL trained from scratch on the natural set (L1-G3) | event-input path for PCL in `jaxfne`, if the etude earns it |
| 0.5.10 | Readout and consolidation | 15. On the validity of EEG predictions from population rates (candidate #9). 16. Mazzoni et al. 2015 *(mem)*: LFP proxies from LIF networks. 17. Metastability from local E–I homeostasis (candidate #11), if reducible to one area | none planned; the release closes the cycle |

The order follows what the code can carry. Each release adds at most one emitter. A
paper whose methods cannot be obtained moves to the next release and does not block its
own.

## Narrowing sweep, every release

Target per release: −10 % tracked files and −10 % docs bytes against the previous
release. The test count stays the same or grows. Each class gets its own commit and gate:

1. **Unused code.** Modules, functions and options that nothing imports or tests
   (`match-cleanup`). Public API is excepted; its removal goes to Hamm.
2. **Docs.** Pages superseded by a newer page (`tutorials_v030`, duplicated guides);
   `docs/_static` assets that nothing references.
3. **Artifacts.** Closed programmes and protocol trees (`artifacts/legacy`,
   `protocol_*`, `e2`, `.lab`) move to a tagged archive branch plus one index file. Git
   history keeps the bytes, and the tree loses them.
4. **Skills.** Merge the 17 into the few that are used, and delete the rest.
5. **Tests.** Merge duplicates that pin the same behaviour. Kill a test only when a
   mutant shows another test catches what it caught.

Each sweep runs as a read-only inventory first (opencode or `match-cleaner`), then one
packet per class. A class is gated by: the full suite at 0 failed, the docs build, the
link check, and the vocabulary audit in check mode.

## Release gate (unchanged)

rc gate, CI green on the SHA, second review, receipt. Tag, main and PyPI only on Hamm's
word.

## Split of work (proposed)

- Claude (vwin): the replication lanes, preregistrations, science review.
- Cursor: release engineering and the GPU benchmark.
- opencode: sweep inventories and bounded build packets.
- agy: literature crawl and P1 checks of the *(mem)* citations.
