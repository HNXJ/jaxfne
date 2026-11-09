# OCMA-J24.1 council result (2026-11-08)

Glossary, prompt, plan, pilot, data, tools and ratings live in HNXJ/ocma `cases/jaxfne/` (config
`configs/ocma-jaxfne.json`); `data/`, `tools/` and `glossary.md` below are relative to that folder. This folder
keeps what the dashboard reads.

88 reference rows (`data/scope.json`), 3 Haiku raters per 5-row batch, then one feedback round (workflow
wf_04d37f0e-2ae, 108 agent runs, 0 errors). Ratings: `council/full/` there (byte copies of jaxfne bd37c52a);
here `council/agreement.json` and `council/consensus.json`
(`tools/agreement.py council/full --seed 0`, reproduced byte-identical from ocma ededb9f).

## Agreement (Krippendorff alpha, interval; baseline: paper labels permuted per rater, 1000 draws)

| Group | Round 1 alpha | Shuffle p95 | Feedback alpha | Shuffle p95 |
|---|---|---|---|---|
| H1 mechanism realized | 0.844 | 0.283 | 0.939 | 0.300 |
| H2 reproduction evidence | 0.880 | 0.106 | 0.971 | 0.114 |
| H3 dashboard readiness | 0.815 | 0.284 | 0.941 | 0.428 |

Round 1 is the independent measure; after feedback the raters have seen each other, so the higher alpha mixes
evidence-driven revision with conformity. The baseline keeps factor identity, so its mean (0.04-0.38) reflects
shared factor means. Uncited non-zero scores set null: 46 in round 1, 16 after feedback, of 6336 per round.

## Consensus actions (median of valid feedback-round scores; rules in `glossary.md`)

| Action | Rows |
|---|---|
| exists (already on the dashboard) | 16 |
| core-check from existing evidence | 1: Jansen and Rit 1995 (H2 0.60, H3 0.72) |
| core-check (build) | 1: Geweke 1982 (H1 1.00, H3 0.74) |
| roadmap | 2: Herreras 2016 (H1 0.60), Mendoza-Halliday 2024 (H1 0.60) |
| study list | 68 (47 of the 59 study-list rows and 21 core rows) |

Both new entries landed after the audit, in the same push: Jansen-Rit 1995 as a paper lane (reproduced,
conditional) and Geweke 1982 as a core check; `consensus.json` keeps the pre-entry actions.

The premise that many mechanism papers are already reproduced does not hold for the 68 rows with action
`study-list`: their median H1 is 0.14 (median over all 88 rows: 0.17), a generic capability with no
paper-specific use. Spot checks agree: no core code
for Montbrio-Pazo-Roxin, FitzHugh-Nagumo, Morris-Lecar or Tsodyks-Markram synapses (grep of `jaxfne/`); PING
appears only in a docstring (`jaxfne/_construct_core.py:268`); Wilson-Cowan has an emitter
(`jaxfne/emitters_rate.py`) and no reproduction (H1 0.50, H2 0.00).

## Adjudication (stage G)

Six rows keep an H2 factor spread above 0.4 after feedback (L24, L33, L45, L133, L137, L140); none changes an
action, since every H2 median sits far from the 0.6 threshold or the row already exists.

- L133 Caminos (Tahvili 2025), Quantitative match {+0.6, +0.6, -0.6}: +0.6 holds.
  `artifacts/etudes/papers/caminos/RESULT_FIGURES.md:25-38` records passes on most panels and fails on Fig 3A
  "abolished" and Fig 4B amplitude, with tolerances chosen after the data (`:3-4`): partial.
- Keeper rulings on the roadmap rows: Herreras 2016 is a review with no target figure (H3 0.33), so no roadmap
  row; Mendoza-Halliday 2024 is the spectrolaminar motif already on the roadmap as unscheduled
  (`dashboard/sources.json`, roadmap).

## Findings for the method

- Self-reference: the inventory and the council could read this folder. 17 mentions in the full-council
  files point to `inventory.json` or `glossary.md`, all in notes but one score (L169 Named target, which may cite the
  reference row anyway). The inventory now excludes this folder (`tools/inventory.py`, `SELF`); later councils
  should run with the tooling outside the rated tree.
- Feedback round cut uncited scores 46 -> 16 and factors with spread > 0.4 in the pilot from 13 to 2.
