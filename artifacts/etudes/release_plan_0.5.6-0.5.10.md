# Release plan 0.5.6–0.5.10: replications plus narrowing

Requested by Hamm, 2026-10-11. This is a draft, so every release, tag and API addition
still needs his word. Citations marked *(mem)* were written from memory. Each was then checked against
Crossref and Europe PMC on 2026-10-11 (see the table below).

| Paper | DOI | Open copy |
|---|---|---|
| Brunel 2000, J Comput Neurosci 8:183 | 10.1023/a:1008925309027 | none found |
| Vogels et al. 2011, Science 334:1569 | 10.1126/science.1211095 | none found |
| Potjans and Diesmann 2014, Cereb Cortex 24:785 | 10.1093/cercor/bhs358 | PMC3920768 |
| Mejias et al. 2016, Sci Adv 2:e1601335 | 10.1126/sciadv.1601335 | PMC5262462 |
| Mazzoni et al. 2015, PLoS CB 11:e1004584 | 10.1371/journal.pcbi.1004584 | PMC4682791; ModelDB 152539 |
| Izhikevich 2003, IEEE TNN 14:1569 | 10.1109/tnn.2003.820440 | none found |
| van Vreeswijk and Farkhooi 2025, bioRxiv | 10.1101/2025.09.04.674182 | preprint, CC BY-NC-ND |
| Lee, Pennartz, Mejias 2025, PLoS CB 21:e1013469 | 10.1371/journal.pcbi.1013469 | PMC12443261 |
| Ness et al. 2025, PLoS CB 21:e1012303 | 10.1371/journal.pcbi.1012303 | PMC12052147 |
| Páscoa dos Santos and Verschure 2025, Netw Neurosci 9:938 | 10.1162/netn_a_00460 | PMC12543307 |

Methods completeness and code availability are still to be checked for all but Mazzoni
(P1).

## 1. Purpose

Each release shows that jaxfne reproduces published results and teaches something beyond
them, while the repository shrinks. The replications are not the end point. They are
validated building blocks for modelling the visual omission paradigm: a model joins that
work only after it has been replicated.

## 2. Definitions

**Fidelity tier.** Every replication states one tier:

| Tier | Meaning |
|---|---|
| T-port | the authors' code or trained artefacts, run in our port (PCL Figs. 2–3) |
| T-methods | a reimplementation from the published methods (CAMINOS) |
| T-jaxfne | the same result through the jaxfne public API |

A paper counts as replicated *with jaxfne* only at T-jaxfne. The etudes are the oracle
that the library path must match.

**Claim ledger.** For each paper, one table lists every claim we test. Each row gives the
claim, its observable, our estimand, the preregistered criterion, the result, and a
verdict.

**Verdict scale:**

- reproduced
- reproduced with a declared deviation
- not reproduced
- not testable, with the reason (for example, methods unavailable)

A negative verdict is a result: it is published and is not tuned away.

## 3. Pipeline per paper (stages P0–P7)

| Stage | Output | Gate | Who |
|---|---|---|---|
| P0 select | why this paper, the insight sought, jaxfne features needed | in the release table | Claude, with Hamm |
| P1 source | methods complete; code and data; licence; citation checked | a gap list with no unknowns | agy or a worker; Claude checks |
| P2 spec | `PORT_SPEC.md`: model, parameters with sources, assumptions | second review | Claude |
| P3 prereg | `PREREG_P3.md`: estimands, bands, nulls, seeds, sensitivity | second review, then Hamm's batch signature | Claude drafts |
| P4 build | etude code and tests (known-answer, mutant-checked); unscored pilot on seed 0 | suite green | opencode or a worker; Claude verifies |
| P5 score | scored run with manifest; analysis JSON; claim ledger | second review of the numbers | Claude |
| P6 library | the same estimands through the jaxfne API | equivalence to P5 within the declared tolerance | worker, Claude |
| P7 insight | one perturbation beyond the paper, one static figure, one gallery page | second review | Claude |

Standing rules:

- Seed 0 is for pilots only.
- Scored runs wait for Hamm's signature.
- Float32 is the default. A float64 sensitivity run on one seed checks every scored
  result.
- Runs of 5 minutes or more go detached; a run is not complete until its receipt exists.

## 4. Releases

| Release | Papers (P0) | Library work (needs Hamm) | Depends on |
|---|---|---|---|
| 0.5.6 | 1. N'dri et al. 2025, PCL Figs. 2–3 (T-port). 2. Tahvili, Vinck, di Volo, PLoS CB 2026: CAMINOS Figs. 1–3 (T-methods, then T-jaxfne). 3. Tahvili et al., Cell Reports 2025, if its methods can be read; otherwise Izhikevich 2003 *(mem)* on the existing emitter | AdEx emitter with conductance synapses | P3 signatures; the Cell Reports methods |
| 0.5.7 | 4. Brunel 2000 *(mem)*: AI, SR and SI states. 5. Vogels et al. 2011 *(mem)*: inhibitory plasticity. 6. Calcium-regulated synaptic homeostasis (bioRxiv 2025). 7. PCL Fig. 4: suppression effects | working LIF emitter (a placeholder today) | 0.5.6 conductance synapses |
| 0.5.8 | 8. Potjans and Diesmann 2014 *(mem)*: microcircuit. 9. Mejias et al. 2016 *(mem)*: laminar multi-area. 10. Spectrolaminar motif (Nat Neurosci 2023 and the 2025 reply) on 8 and 9 | laminar readout, if needed | 0.5.7 LIF; GPU time |
| 0.5.9 | 11–12. Lee, Pennartz, Mejias, PLoS CB 2025 (lane L5, candidate #4, which are the same paper): PV, SST and VIP silencing, deviant enhancement, the ~6 Hz rhythm; closest to the omission and oddball paradigm. 13. PCL Figs. 5–6. 14. PCL trained from scratch (L1-G3) | event input for PCL, if earned | the PCL suppression data and N-MNIST/DVS sets |
| 0.5.10 | 15. Validity of EEG predictions from rates (candidate #9). 16. Mazzoni et al. 2015 *(mem)*: LFP proxies. 17. E–I metastability (candidate #11), if it reduces to one area | none | 0.5.8 laminar models |

A paper that cannot pass P1 moves to the next release without blocking its own. Each
release adds at most one emitter.

## 5. Definition of done, per release

1. Every paper has a verdict and is at P6, or has a stated reason why P6 does not apply.
2. A replication index (`artifacts/etudes/REPLICATIONS.md`) lists each paper's tier,
   verdict, ledger, receipts and gallery page.
3. The narrowing target is met (§6) and the size gate is green.
4. The usual release gate: rc gate, CI green on the SHA, second review, receipt.
5. Tag, main and PyPI happen on Hamm's word.

## 6. Narrowing

Baseline (69bd4274): 2525 tracked files, 126.0 MB. Docs take 42 MB and artifacts 35 MB.
There are 17 skills.

The first inventory found that safe removals come to only 1.4 %. Most of the bytes are
generated HTML: about 50 MB across `artifacts/publication`, `docs/assets`, `docs/_static`
and `artifacts/tutorials`.

Plan:

1. **Decide where generated assets live** (Hamm): regenerate them at the docs build
   (recommended), or keep them as release assets. The docs keep their inline embeds
   either way.
2. **Size gate in CI**, so the tree does not grow back:
   - no new tracked file over 1 MB without an allow-list entry;
   - a total-byte budget per release.
3. **Per release, one class per commit**, each with its gate: unused code, docs,
   artifacts to an archive branch plus an index, skills merged, duplicate tests merged.
   A removed test needs a mutant caught elsewhere.
4. **Results stay small:** each replication ships one static figure and one page, with
   no new interactive HTML unless it is generated.

Targets: −10 % files and −10 % bytes per release once decision 1 is made, and no loss of
tests or public names.

## 7. Structure in the repo

```
artifacts/etudes/<paper>/      PORT_SPEC.md, PREREG_P3.md, code, tests, LEDGER.md
artifacts/etudes/REPLICATIONS.md   index across releases
F:\warehouse\<paper>\         source data, downloads, scored runs (outside git)
docs/replications/<paper>.md  gallery page, one static figure (P7)
```

Branches: one per lane, each in its own worktree. Each lane merges to dev after a second
review, then the release branch is cut from dev.

## 8. Roles

- Claude (vwin): the lanes, specs, preregistrations, verdicts, insight.
- Cursor: release engineering, sweep commits, size gate, GPU benchmark.
- opencode: bounded builds and tests.
- agy and workers: P1 checks and inventories.
- Hamm: selection, signatures, API additions, tags.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Methods unavailable (Cell Reports) | companion papers; move the paper to the next release; Hamm opens the page |
| Etude results never reach the library path | P6 is part of the definition of done |
| Pilot results leak into the bands | seed 0 excluded; prior exposure stated in each prereg |
| The GPU is shared with Cursor | one job at a time, claimed in jchat |
| Licences (bioRxiv CC-BY-NC) | no copied figures or text; cite only |
| Assumptions stack up (map order, drive reading) | sensitivity runs named in each prereg |

## 10. Open decisions (Hamm)

1. Approve the release table and the order.
2. Where generated assets live (§6.1).
3. The AdEx (0.5.6) and LIF (0.5.7) API additions.
4. Is the stated purpose right: replications as building blocks for omission modelling?

Decided 2026-10-11 (on meeting 3790d96e): preregistrations are signed in a batch per release,
each after a second-review PASS and a passing calibration gate; manuscript sections 1–7 freeze
on the 0.5.5 evidence and this programme fills section 8; bindings B1–B7 are standing rules
(`artifacts/todo_stack.md`); the jnwb short list folds into the 0.5.6 library lane.

Decided 2026-10-12 (Hamm): AdEx is public as `emitter(family="adex")` plus ADVANCED names in
`jaxfne.emitters` (decision 3, AdEx part); generated assets are regenerated at the docs build
(decision 2, §6.1); the third 0.5.6 paper is Izhikevich 2003 on the existing emitter.

Decided 2026-10-12 (Hamm, on the 0.5.6 review): 0.5.6 is mostly cleaning, organizing, a major
docs update and a dashboard page per paper. §5 item 1 is relaxed for 0.5.6: papers ship at their
P5 verdicts, CAMINOS reaches P6 if F6 lands, and PCL Fig. 3 and Izhikevich 2003 carry a stated
reason. The dashboard holds the per-paper pages; the docs carry one Replications page linking
them, in place of `docs/replications/<paper>.md` (§7).
