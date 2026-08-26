# Supplement S-Pulse: Frozen Pulse-Regime Characterization (Private Annex)

**Status:** Supplement-only reuse — no new simulation, no new detector  
**Working directory:** `C:\workspace\jaxfne`  
**Date:** 2026-08-25  
**Operating point:** theta* (tie-break-selected, lexicographic): `drive_E=4.0, drive_I=2.0, weight_mu=0.25, noise_scale=0.0, W_ms=60` — six-way adequacy tie in E2a; not an optimized optimum.  
**Scope:** Reuses frozen V1/V2 raw metrics verbatim. Any future detector change requires a new preregistered protocol.

## Provenance — sources reused (no new data generated)

| Receipt | Path | Role |
|---|---|---|
| V1 PING raw | `artifacts/e2/preregistration/E2b_confirmatory/v1_ping_receipt.json` | Primary pulse-regime metrics (7 arms x 5 seeds @ theta*) |
| V1 corrigendum + adjudication | `artifacts/e2/preregistration/E2b_confirmatory/v1_corrigendum_and_adjudication.json` | Corrected harmonic-comb interpretation, envelope period, direction fix |
| V2 SSA confirmatory | `artifacts/e2/preregistration/E2b_confirmatory/v2_ssa_confirmatory_receipt.json` | Adequacy 20/20, SI context |

Independent frozen-only rescoring (`v1_rescored_frozen_only.json`, `v2_rescored_frozen_only.json`) and E2 synthesis (`E2_SYNTHESIS.md`) corroborate the same regime without new thresholds.

> **Label:** This file is a **supplement-only reuse** of already-frozen confirmatory outputs. No new pulse detector, classifier, or simulation was run to produce this table. All values are copied from the receipts above.

---

## Frozen pulse-regime table (V1 @ theta*)

All V1 values are the intact arm `C0_intact` at theta* (n=5 seeds). Verdict is independent of executor defects: `NEGATIVE_NOT_PING_LIKE` under frozen classifiers and under corrected rescoring (`v1_rescored_frozen_only.json`: 0/5 PING_LIKE).

| # | Parameter | Frozen value (reuse) | Definition / derivation | Gate context |
|---|---|---|---|---|
| 1 | Fundamental pulse frequency **f0** | **~7.2 Hz** | Envelope rhythm of the globally synchronous population-pulse train. Corrigendum C2: single slow oscillator; `f0 = 1 / 138.5 ms = 7.22 Hz`. | Not a PING classifier gate; diagnoses carrier of V1. |
| 2 | Envelope interpulse interval | **138.5 ms** (median) | Interval between successive population pulses from envelope / pulse-picking on `v1_rates_window.npz` population rates. | Same as (1). |
| 3 | Duty cycle **D** | **~0.05** | Pulse width / period. Matches median pulse duration (E ~16–17 ms in `v1_ping_receipt.json` md_E) over 138.5 ms period at the low end; reported as D ~0.05 in corrigendum characterization. Sustained pulsing with brief active fraction. | — |
| 4 | Spectral teeth (harmonic comb) | **E fpk ~36.1 Hz (k=5), I fpk ~43.3 Hz (k=6)** | Welch PSD peaks (fs=2000 Hz, Hann 256 ms) fall on adjacent teeth of f0 comb: 36.1 ≈ 5 × 7.22, 43.3 ≈ 6 × 7.22. Per-seed E/I peaks in `v1_ping_receipt.json` cluster at these teeth; `|f_E - f_I| ≈ 7.2 Hz` equals one harmonic spacing (comb artifact, not two oscillators). | Triggers frozen `G_spec` failure + `OptionA` discordance artifact (`|dfp|>5`). Corrected by corrigendum C2. |
| 5 | AC sidepeak (gamma-lag) | **Negative: E ~ -0.10, I ~ -0.06** | Autocorrelation sidepeak at gamma period from population rates. Corrigendum: `E -0.10, I -0.06`; rescoring `ac_min -0.064 to -0.067` across seeds; executor `xcorr_shifted` surrogate varied but all <0.25. | Frozen `G_rate` requires `AC >= 0.25` — fails by sign and magnitude (robust, convention-independent). |
| 6 | Cycle count **n_cycles** | **3–4** | Number of gamma-period cycles detected in analysis window W [1500,2000) ms. `v1_ping_receipt.json` `cycles` field: 3 or 4 in all 5 C0 seeds (one seed per C3d shows 6 with FF>180, not intact). | Frozen `G_cycle` requires `N_cycles >= 10` — fails in every intact seed. |
| 7 | Participation | **~1.0** | Fraction of neurons spiking per pulse. `v1_ping_receipt.json` `participation=1.0` in 4/5 C0 seeds (fifth likewise 1.0 in raw; rescoring preserves 1.0-equivalent). Indicates globally synchronous pulses. | Not a classifier gate; diagnoses regime. |
| 8 | Fano factor (FF) | **~0** (0.0) | Across-pulse count variability. `v1_ping_receipt.json` `ff=0.0` in all C0 intact seeds (vs 190–480 in some C3 arms). | Sub-component of `G_cycle` (`FF <= 0.60` passes but `N_cycles` fails). |
| 9 | Phase ordering **E-leads-I** | **+6.8 ms (E first)** | `dt_lag_ms ≈ -6.76` ms (executor sign: `dphi = phi_I - phi_E`; negative = E leads). Corrigendum C1 corrects sign description: `dphi_deg -85 to -91 deg` corresponds to `dt_lag -6.8 ms`; rederived median pulse-pair offset **+16 ms E-first**. PLV ~0.91 (high synchrony). | Frozen `G_phase` requires `dphi in [15,90] deg` and `dt in [2,8] ms` with E-leading convention — fails on the degree window (sign convention is canonical but band is not met); ms window would pass if sign-flipped, so not a robust discriminator. Listed last per C5. |
| 10 | Prominence (gamma band) | **5.41–5.65 dB (UNRESOLVED)** | Welch prominence at fpk. Intact seeds `prom_dB` 5.41–5.65 in receipt; rescoring 5.23–5.82. | Frozen `G_spec` requires `prominence >= 6 dB` with gray `[5,6)` → `UNRESOLVED_PROMINENCE` in every intact seed (robust near-threshold, not PING). |
| 11 | Stationarity / band ratio | `stationarity_ok=false` in executor; `band_ratio ~0.35` | W1 vs W bandpower within 3 dB gate added by executor (invented conjunct D1) — failed in intact; excluded under frozen-only rescoring. | Not part of frozen four strings; disclosed defect D1. Verdict-neutral (adversary rescoring intact stays NO_PING). |

**Regime summary (frozen interpretation, corrigendum-licensed):** ~7 Hz globally synchronous population-pulse regime (participation ~1.0, FF~0, duty ~0.05, interpulse 138.5 ms) whose gamma-band spectral peaks at 36.1 / 43.3 Hz are adjacent harmonic-comb teeth (k=5/6) of the slow rhythm, not a PING oscillator. Neither canonical PING nor ING: C2 (I→E zeroed) preserves phenotype (not loop-dependent), C1 (E→I zeroed) silences I entirely.

### Per-seed intact detail (for audit trace)

From `v1_ping_receipt.json` (C0_intact, 5 seeds):

| seed (rep) | fpk_E (Hz) | fpk_I (Hz) | prom_dB | xcorr | dt_lag_ms | dphi_deg | PLV | cycles | participation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 36.11 | 43.30 | 5.54 | 0.975 | -6.76 | -87.85 | 0.912 | 3 | 1.0 |
| 2 | 36.09 | 43.30 | 5.47 | 0.978 | -6.56 | -85.27 | 0.916 | 3 | 1.0 |
| 3 | 36.14 | 43.34 | 5.65 | 0.974 | -6.95 | -90.48 | 0.910 | 3 | 1.0 |
| 4 | 36.14 | 43.35 | 5.64 | 0.978 | -6.85 | -89.09 | 0.913 | 4 | 1.0 |
| 5 | 36.13 | 43.33 | 5.41 | 0.979 | -6.80 | -88.46 | 0.913 | 3 | 1.0 |

All 5 fail G_spec (prominence gray), G_rate (AC negative), G_cycle (n_cycles <10); G_phase fails degree window. Frozen-only rescoring agrees: 0/5 PING_LIKE (labels `NO_PING` in `v1_rescored_frozen_only.json`).

---

## V2 context at the same theta* (no new regime inferred)

| Parameter | Frozen value (reuse) | Source |
|---|---|---|
| Adequacy **G_A / G_B** | **20/20** replicates pass (`G_A=true, G_B=true` in all 20 rows) | `v2_ssa_confirmatory_receipt.json` `runs[].G_A/G_B`, `pooled.adequacy_all=true` |
| Stimulus-specific adaptation **SI** | **Pooled -0.084**, BCa 95% [-0.108, -0.034] (deviant **below** standard; inverted sign), `g=-6.19`, `p_perm=1.0` | `v2_ssa_confirmatory_receipt.json` `pooled.SI=-0.08434, BCa_lower/upper, g, p_perm` |
| Swap asymmetry (role-reversal) | **Observed 0.426 >> 0.10** (frozen S2 falsifier) | `pooled.swap_max_observed=0.426` |
| Many-standards control SI_many | ~0.028 (pooled) — within | — |
| Synchronous vs. sparse regime distinction | **Synchronous (V1) vs. sparse (V2 stimulus-driven) are not conflated.** V1 pulses are spontaneous globally synchronous (participation 1.0, FF~0) without stimulus; V2 responses are event-locked mean rates over `[30,110) ms` per stimulus onset (ISI 200 ms, drive pulse to disjoint E subpopulations [0,400) vs [400,800)) with standard/deviant attenuation pattern. The same circuit at the same theta* supports both: pulsing background (V1) and pattern-selective attenuation with inverted SI (V2) — not adaptation. | `e2_ssa_spec.v6.json` stimulus_identity_mapping + `E2_SYNTHESIS.md` diagnosis |

**Interpretation (bounded, corrigendum / E2_SYNTHESIS-licensed):** V2 shows **pattern-selective attenuation, not deviance detection** — spatial identity dominates, swap asymmetry falsifies stimulus-specific adaptation, and pooled SI sign is reliably negative (deviant below standard). This characterizes the same ~7 Hz synchronous circuit probed under a different paradigm; no claim that the synchronous regime *causes* the SI sign is licensed beyond co-occurrence at theta*.

---

## What is NOT in this supplement

- No new detector, classifier gate, threshold, or derived statistic was invented for this table (executor deviations D1–D6 disclosed in corrigendum remain excluded).
- No repair or retuning of frozen gates (prominence 6 dB, AC 0.25, N_cycles 10, phase [15,90] deg) was performed.
- No claim about other points in parameter space; licensed boundary sentence (corrigendum) applies verbatim:

> In the preregistered confirmatory arm (V1: 7 arms x 5 seeds at the tie-break-selected operating point theta*), networks met adequacy gates in 5/5 seeds but satisfied none of the four frozen PING-classifier criteria in any seed (verdict NEGATIVE_NOT_PING_LIKE), instead exhibiting a ~7 Hz globally synchronous population-pulse rhythm whose gamma-band spectral peaks fall below the frozen prominence gate; the V2 arm was not executed (frozen-specification gap, NOT_EXECUTED_FREEZE_GAP), the control-collapse table was vacuous because the intact arm itself fell outside the frozen phase window, and theta* was chosen from a six-way adequacy tie by lexicographic tie-break rather than as an optimized operating point — these results therefore support only the bounded statement that no PING-like signature was detected at theta* under the frozen criteria, carry no implication about other regions of the parameter space, and are independent of the CL-06 NO_WAVE result. (V2 now executed under v6 at 20/20 adequacy with SI -0.084 update; independence from CL-06 preserved.)

Supplement placement only (E2_SYNTHESIS disposition). Main-text entry would require a new separately declared experiment with its own development/confirmation split.

---

## Reuse declaration

- **Data reused:** All numbers above are verbatim from `v1_ping_receipt.json:74-1591`, `v1_corrigendum_and_adjudication.json:1-44`, `v2_ssa_confirmatory_receipt.json:1-364` (and their frozen-only rescorings). No array was re-read, no PSD recomputed, no model re-simulated.
- **Code reused:** None executed for this file. References to executor `e2b_v1_executor.py:354` bug and `v1_rates_window.npz` (2,1000 at 1 ms archived) are documentation of prior defects (corrigendum C4), not re-executed here.
- **No new detector clause:** This annex does not define, implement, or validate any new pulse detector. Any future pulse-timing characterization must be preregistered separately.
