# OCMA-J24 pilot (2026-11-08)

Rows 24 (Brette 2005), 26 (Wang-Buzsaki 1996), 34 (Hagen 2016), 139 (Vogels 2011), 142 (Kopell 2000); 3 Haiku
raters, then one feedback round (workflow wf_e8122ca0-e45). Files: HNXJ/ocma `cases/jaxfne/council/pilot/r{A,B,C}[.rev].json`.

| Check | Result |
|---|---|
| Validation (`validate.py`, 30 records) | 3 problems: two bare file names without directory (rA L26), one uncited -0.6 (rC.rev L34) |
| Factors with spread > 0.4, round 1 -> feedback | 13 -> 2 |
| Known negative (Kopell 2000: no prereg, no score) | H2 = 0.00 in all 6 files: held |
| Known positive (Vogels 2011: signed P3 v4, scored, scorer) | Preregistered +1, Scorer exists +1, R-E off cited to `run/p3_v4_score.json`; H2 0.40 |
| Keeper anchor error found by raters | Brette 2005: AdEx parameters in code cite Tahvili 2026 Table 1 (`jaxfne/emitters_adex.py:35-37`), not Brette; anchor assumed paper parameters |

Glossary defects found, fixed in J24.1 before the full council (see `glossary.md`, Versions): factor 14 conflated
existence and outcome (one rater gave Vogels -1 for a pending criterion); factor 8 at 0 pulled complete mechanisms
down; factor 24 undefined for paper lanes; the action rule gave no dashboard action to any pilot row, the scored
Vogels lane included. Inventory defect: the dashboard match missed entries whose citation omits the year (Vogels);
fixed by matching id, title and citation (16 rows now match).

## Anchors written before any rater output

| Row | Paper | Anchor (expected) |
|---|---|---|
| L24 | Brette and Gerstner 2005 (AdEx) | H1 high: emitter exists (1 = +1); no prereg (13 = 0); known positive for core-check fit |
| L26 | Wang and Buzsaki 1996 | kappa is a core readout with a documented departure (6 = -0.6 candidate); WB emitter absent; H2 low |
| L34 | Hagen et al. 2016 | H1 high: `jaxfne/fields/kernel_lfp.py`, `hagen2016_cells.py`; H2 partial with a documented departure |
| L139 | Vogels et al. 2011 | known positive for H2: signed prereg, scored, dashboard entry |
| L142 | Kopell et al. 2000 | known negative for H2: not started, no prereg, no score; H1 at most +0.2 |
