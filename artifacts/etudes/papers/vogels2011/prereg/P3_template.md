# vogels2011 preregistration P3 (draft, UNSIGNED — no scored run may use it)

Lane: vogels2011. Paper: Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W (2011), Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks, Science 334(6062):1569-1573, doi:10.1126/science.1211095. No scored run starts from this draft;
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
