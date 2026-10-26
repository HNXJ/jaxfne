# potjans2014 preregistration P3 (draft, UNSIGNED — no scored run may use it)

Lane: potjans2014. Paper: Potjans TC, Diesmann M (2014), The cell-type specific cortical microcircuit: relating structure and activity in a full-scale spiking network model, Cereb Cortex 24(3):785-806, doi:10.1093/cercor/bhs358. No scored run starts from this draft;
Hamm signs the frozen protocol before scoring (checklist P1).

## Estimands and criteria (checklist P1)

| Id | y (paper) | z (jaxfne) | e | criterion |
|---|---|---|---|---|
| (fill) | | | | close: every e <= 0.05; off: any e > 0.05 |

## Seeds (checklist P2)

Seed 0 is for pilots only. Fit (if needed): seeds 1-3. Score: held-out
seeds 4-8. The seed shown to change the output is named.

## Scored / unscored split

- Unscored (pilots, G0 checks): seed 0, timing runs; outputs never under a
  `scored/` path.
- Scored (needs `--signed`): seeds 1-8 per above; one code SHA per output
  directory (enforced by `lane_kit.refuse_existing`).

## Sensitivity set (declared before scoring)

- (fill: S1..Sn, one declared variation each; sensitivity runs never enter
  score, checklist P6)

## Pilot (B8)

- The full pipeline runs on unscored pilot seeds and on a known positive
  before this prereg is signed (project rule B8, Hamm 2026-10-18).
- The sensitivity set is declared in this prereg (see above); after
  signature at most one amendment, any further change is a new protocol
  version scored on fresh seeds.
