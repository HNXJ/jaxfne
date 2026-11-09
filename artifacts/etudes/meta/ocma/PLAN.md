# OCMA plan: map every reference paper's replication state, then expand the dashboard

Goal (Hamm 2026-11-08): list the reference papers jaxfne already reproduces in whole or in part (Kopell-family PING,
Wang-Buzsaki, AdEx, kernel LFP, ...) and give each a dashboard entry. Method: the pipeline of Nejat et al., Sci Rep
2026 (`hnxj/mllm-public`): an expert glossary, a constrained prompt, an independent council, validation, agreement
against a shuffled baseline, adjudication. This audit is case 1 of the paper project `E:/repos/ocma` (Hamm
2026-11-08); the glossary and prompt stay here, frozen per council run, until that project takes them over.

## Stages

| Stage | What | Who | Output | Check |
|---|---|---|---|---|
| A | Mechanical inventory of all 117 rows (DOI and author-year hits; dashboard match; this folder excluded) | script | `data/inventory.json` | `tools/inventory.py` exit 0, 117 rows |
| B | Scope: the study list (59 rows) plus core rows naming a model or observable (Neural dynamics, Sources, reduced and population models, circuits): about 85 rows; simulators, reproducibility and differentiable-computation rows are out (no mechanism) | keeper | `data/scope.json` (88 in, 29 out) | each excluded row has a reason |
| C | Council: 3 independent Haiku raters score every in-scope row on the 24 factors (`glossary.md`, `prompt.md`), 5 rows per packet | Haiku x3 | HNXJ/ocma `cases/jaxfne/council/full/r?.json` (batches merged by `tools/split_council.py`) | schema enforced at output; citations by `tools/validate.py` |
| D | Validation: a non-zero score without an existing `path:line` is set to null and counted (no retry; deviation from the first plan, which sent packets back) | script | counts in `council/agreement.json` | listed |
| E | Feedback round: each rater sees the other two raters' JSON for its batch and keeps or revises each disagreement (spread > 0.4) | Haiku x3 | HNXJ/ocma `cases/jaxfne/council/full/r?.rev.json` | agreement after E reported |
| F | Agreement: Krippendorff alpha (interval) per group, against a shuffled baseline (paper labels permuted per rater, 1000 draws) | script | `council/agreement.json`, `council/consensus.json` | alpha above the 95th shuffled percentile, or reported as not |
| G | Adjudication of residual H2 disagreements (the replication claim) | keeper | `RESULT.md` | each verdict cites the file |
| H | Dashboard: H2 >= 0.6 and H3 >= 0.5 become core-check entries (scorer script, `score.json`, `sources.json` entry, model page); H1 >= 0.6 with H2 < 0.6 become roadmap rows | Haiku workers in clones, keeper reviews | entries, regenerated dashboard | `build_dashboard_data.py --check`, `generate_replications_page.py --check`, docs gate, an outside review |

Rules: papers stay "pending", never "failed", through 0.6; the Mulvey/LANSA lane stays local; no number enters a
criterion unless the paper states it (page and figure cited); a scorer reads values from jaxfne runs, never from
the paper. Raters read the repository only; they edit nothing.

## Cost (inferred)

About 17 packets of 5 rows: 51 rater runs in C, 51 in E, plus about one worker per core-check entry in H.
