# potjans2014 preregistration P3 (draft, UNSIGNED — no scored run may use it)

Lane: potjans2014. Paper: Potjans TC, Diesmann M (2014), The cell-type specific cortical microcircuit: relating structure and activity in a full-scale spiking network model, Cereb Cortex 24(3):785-806, doi:10.1093/cercor/bhs358. No scored run starts from this draft;
Hamm signs the frozen protocol before scoring (checklist P1). Items marked (proposal) are the
orchestrator's draft (2026-10-26) and need his ruling.

## Route (prerequisite, proposal)

Scored runs use the closed-gap core route, adopted by the lane in a P2 refresh before the pilots
(PORT_SPEC rows 8-14 move with it): `sampling="multinomial"` with the paper's K per projection,
Gaussian weights (`w_std`) and delays (`delay_std_ms`) per rule, `tau_syn_ms=0.5`, per-population
Poisson drive at the reference k_ext (8 Hz per train), dt = 0.1 ms (paper h). Weights map pA to mV
through C_m = 250 pF with the exact charge per arrival: per-step increment
`(w / C_m) * tau_syn * (1 - exp(-dt / tau_syn))`, so one PSC deposits `w * tau_syn / C_m` (0.1756 mV
at w = 87.8 pA) at any dt.

Scale D-NSCALE (proposal): 10% of every population (7 717 cortical cells), in-degree per projection
held at its full-scale value, weights and k_ext unchanged. Reason: held in-degree and weights keep the
mean and variance of each cell's input; pairwise correlations are not held (inferred), which is why
synchrony is secondary below. Full scale (about 0.3 billion synapses) is out of reach for 10 s runs
on the local GPU and on CPU (inferred from the N=800 pilot timing in `CALIBRATION.md` section 0).

## Estimands and criteria (checklist P1)

z is the mean over scored seeds; rates and CV from the kept window (first 0.5 s dropped), estimators
from `jaxfne.analysis` only.

| Id | y (paper) | z (jaxfne) | e | criterion |
|---|---|---|---|---|
| R-L23E | 0.86 Hz (Table 6, reference) | L2/3e mean rate | \|z-y\|/y | close: e <= 0.20 (proposal) |
| R-L4E | 4.45 Hz | L4e mean rate | \|z-y\|/y | close: e <= 0.20 |
| R-L5E | 7.59 Hz | L5e mean rate | \|z-y\|/y | close: e <= 0.20 |
| R-L6E | 1.09 Hz | L6e mean rate | \|z-y\|/y | close: e <= 0.20 |
| O-IE | inhibitory rate above excitatory in every layer (Results) | 4 comparisons | pass/fail | all 4 hold |
| O-ORD | L5e highest E rate; L2/3e and L6e below L4e (Results) | E-rate order | pass/fail | holds |
| CV | mean single-unit CV of ISIs above 0.8 in every population (Fig. 6C) | `cv_isi` per population | pass/fail | all 8 hold |
| SYN (secondary) | synchrony highest in L5e, lowest in L6 (Fig. 6D) | `fano_factor` of the summed count of 100 cells per population, 3 ms bins, 5 s | pass/fail | reported, never scored |

The lane closes when every primary row holds; any primary row off makes the result off, recorded
with its e. C2 (100 random parameterizations) and C3 (robustness percentages) are out of scope at
this scale.

## Seeds (checklist P2)

Seed 0 is for pilots only. Fit: none (no free parameter; D is empty). Score: held-out seeds 4-8.
The seed shown to change the output is named in the pilot record.

## Scored / unscored split

- Unscored (pilots, G0 checks): seed 0, timing runs; outputs never under a
  `scored/` path.
- Scored (needs `--signed`): seeds 4-8, T = 10 s at dt = 0.1 ms (proposal); one code SHA per output
  directory (enforced by `lane_kit.refuse_existing`).

## Sensitivity set (declared before scoring)

- S1: scale 20% with in-degree held (seeds 4-5).
- S2: dt = 0.05 ms (seeds 4-5).
- S3: layer-independent k_ext (Table 5 control) on the scored scale (seeds 4-5).
Sensitivity runs never enter the score (checklist P6).

## Pilot (B8)

- The full pipeline runs on unscored pilot seed 0 and on a known positive before this prereg is
  signed (project rule B8, Hamm 2026-10-18).
- Known positive (proposal): a lesion the criteria must catch. Seed 0 with the L5i -> L5e
  projection removed must put R-L5E outside its band (L5e loses its strongest inhibitory input,
  p = 0.373 in Table 5). A lesion run that stays inside the band means the band cannot fail, and the
  prereg goes back for a narrower one.
- The sensitivity set is declared in this prereg (see above); after
  signature at most one amendment, any further change is a new protocol
  version scored on fresh seeds.
