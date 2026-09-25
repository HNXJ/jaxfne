# Atlas gap matrix v4 — 0.5.5 ENGINE item 1 (frozen measurement vector)

Authority: `artifacts/project_sources/8_atlas.md` (Y vector; OMITTED /
REFUSED never synthesized). Schema and extractors: `artifacts/atlas/y_schema.py`
(`SCHEMA_VERSION = "v4"`, `Y_KEYS_V4` = the v3 names in order, `measure_v4`).
Pins: `tests/test_atlas_y_schema.py` (contract, CI Fast Atlas step) and
`tests/test_atlas_v4_records.py` (`EXPECTED_IMPLEMENTED` per scenario from real
runs). The code and those pins are the source; this table explains them.

## What v4 changes

- One schema module replaces four per-script copies (v0-v3 constants and
  sentinels stay in their scripts as frozen evidence of earlier passes).
- A cell is IMPLEMENTED only when the runner wrote a value, and the cell
  carries that value. v1-v3 set IMPLEMENTED from static per-scenario tables
  and carried only `{"scenario", "status"}`.
- `record` refuses non-finite or non-JSON values, a value on a REFUSED name,
  and names outside the schema; `validate` checks order, fields and the
  value/state rule.
- AT-08/AT-09 arms now write `w_12_max_abs_change` / `w_21_max_abs_change`
  (one per owned cross range, `connect()` rule order A1->A2, A2->A1). v3
  claimed W_12 and W_21 IMPLEMENTED while only the combined cross change was
  recorded.

## Cells v1 claimed that no runner value backs (now OMITTED)

| Scenario | Cells |
|---|---|
| AT-02 | X, Q, M_compute |
| AT-03 | Q, M_compute |
| AT-04 | SPK, Q, M_compute |
| AT-05 | X, SPK, Q, M_compute |
| AT-06 | X, SPK, Q, Phi_E, M_compute |

M_compute is measured inside these runners (`_run_configuration` /
`_tfne_pair_run` return `mem_peak_b`) and dropped from the arm dicts; Q
(sources) is computed and not retained. REDUCTION gains X and Q, which its
runner writes and v1 left OMITTED.

## Y x AT (v4, observed 2026-09-25 on the dev tree)

I = IMPLEMENTED, O = OMITTED, R = REFUSED. AT-04 is the 0.5.2 geometry pass,
AT-04R2 the 0.5.3 H-perturbation arm; AT-10 is the toy 3-area pass.

| Y \ AT | AT-01 | AT-02 | AT-03 | AT-04 | AT-05 | AT-06 | REDUCTION | AT-07 | AT-04R2 | AT-08 | AT-09 | AT-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X | I | O | I | I | O | O | I | I | I | I | I | O |
| H | O | O | O | O | O | O | O | I | I | I | I | O |
| W | O | O | O | O | O | O | O | I | I | I | I | O |
| Q | O | O | O | O | O | O | I | I | I | I | I | O |
| Phi_E | O | I | I | I | I | O | O | I | I | I | I | O |
| Phi_B | R | R | R | R | R | R | R | R | R | R | R | R |
| SPK | I | I | I | O | O | O | O | I | I | I | I | I |
| PSD | O | O | I | O | O | O | O | O | O | O | O | O |
| C | O | O | I | I | I | I | O | I | I | I | I | O |
| phi | O | O | I | O | O | O | O | O | O | O | O | O |
| E_reduction | I | O | O | O | O | O | I | O | O | O | O | O |
| T_compute | I | I | I | I | I | I | I | I | I | I | I | I |
| M_compute | I | O | O | O | O | O | O | I | I | I | I | O |
| SPK_A1 | O | O | O | O | O | O | O | O | O | I | I | O |
| SPK_A2 | O | O | O | O | O | O | O | O | O | I | I | O |
| H_A1 | O | O | O | O | O | O | O | O | O | I | I | O |
| H_A2 | O | O | O | O | O | O | O | O | O | I | I | O |
| Phi_A1 | O | O | O | O | O | O | O | O | O | I | I | O |
| Phi_A2 | O | O | O | O | O | O | O | O | O | I | I | O |
| W_12 | O | O | O | O | O | O | O | O | O | I | I | O |
| W_21 | O | O | O | O | O | O | O | O | O | I | I | O |
| C_12 | O | O | O | O | O | O | O | O | O | I | I | O |
| dphi_12 | O | O | O | O | O | O | O | O | O | I | I | O |

Counts: 86 IMPLEMENTED, 178 OMITTED, 12 REFUSED over 276 cells.

## Owned next (todo stack, 0.5.5)

- Retain the values AT-02..AT-06 already compute (mem_peak_b, source means)
  so M_compute and Q stop being OMITTED there.
- E_reduction scale matrix (item 7) and the AT-10 column (item 6).
- Phi_B stays REFUSED until the AT-07-R4 human decision.
