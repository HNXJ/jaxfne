# E2d rarity-penalty battery -- confirmatory table (frozen 5303efd6)

theta* {"W_ms": 60, "drive_E": 4.0, "drive_I": 2.0, "e_cell_params_native": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, "drive": 5.0, "role": "excitatory_pyramidal_like", "source": "suite2_celltype_presets E + e2b_v1_executor.py E defaults"}, "i_cell_params": {"a": 0.1, "b": 0.2, "c": -65, "d": 2.0, "model_status": "PV-like model parameterization"}, "id": "theta0", "noise_scale": 0.0, "note": "verbatim from E2a_search_receipt.json result.theta*; six-way adequacy tie lexicographic selection, NOT optimum", "weight_mu": 0.25, "weight_scale_reading": "weight_mu scales all projection mus proportionally to nominal mu_EE=0.35: mu_X = mu_X_nom * weight_mu/0.35; sigmas unchanged (frozen)"} | n_outer=20 | paradigm events 80 p_dev 0.15 washout 2 | code_head 2f5adb6e

| rep | rA_std | rA_dev | rB_std | rB_dev | dR_A | dR_B | si_A | si_B | swap | SImany | S |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  0 | 8.17 | 6.99 | 8.51 | 8.05 | -1.18 | -0.45 | -0.078 | -0.027 | +0.091 | 0.043 | -0.105 |
|  1 | 8.23 | 6.58 | 8.22 | 6.34 | -1.65 | -1.89 | -0.111 | -0.130 | -0.019 | 0.031 | -0.241 |
|  2 | 7.74 | 7.23 | 8.25 | 7.05 | -0.51 | -1.20 | -0.034 | -0.078 | +0.019 | 0.026 | -0.112 |
|  3 | 7.95 | 6.17 | 8.70 | 9.18 | -1.78 | 0.48 | -0.126 | 0.027 | +0.242 | 0.033 | -0.100 |
|  4 | 8.66 | 10.17 | 7.77 | 6.02 | 1.51 | -1.75 | 0.080 | -0.127 | -0.313 | 0.044 | -0.047 |
|  5 | 8.52 | 6.97 | 8.21 | 5.93 | -1.55 | -2.28 | -0.100 | -0.161 | -0.098 | 0.027 | -0.262 |
|  6 | 7.73 | 5.17 | 8.88 | 8.41 | -2.57 | -0.47 | -0.199 | -0.027 | +0.306 | 0.015 | -0.226 |
|  7 | 8.16 | 8.04 | 8.08 | 7.19 | -0.12 | -0.89 | -0.007 | -0.058 | -0.061 | 0.002 | -0.065 |
|  8 | 7.94 | 7.25 | 8.46 | 8.09 | -0.70 | -0.37 | -0.046 | -0.022 | +0.086 | 0.029 | -0.068 |
|  9 | 8.69 | 8.35 | 7.97 | 6.14 | -0.35 | -1.84 | -0.020 | -0.130 | -0.195 | 0.039 | -0.151 |
| 10 | 8.21 | 5.09 | 8.89 | 6.98 | -3.13 | -1.91 | -0.235 | -0.120 | +0.191 | 0.030 | -0.355 |
| 11 | 8.14 | 6.31 | 8.45 | 9.07 | -1.83 | 0.62 | -0.127 | 0.036 | +0.199 | 0.026 | -0.091 |
| 12 | 9.02 | 7.23 | 8.51 | 4.16 | -1.78 | -4.35 | -0.110 | -0.343 | -0.287 | 0.035 | -0.453 |
| 13 | 8.23 | 6.94 | 8.14 | 7.06 | -1.29 | -1.08 | -0.085 | -0.071 | +0.003 | 0.026 | -0.156 |
| 14 | 8.26 | 7.36 | 8.22 | 8.32 | -0.91 | 0.10 | -0.058 | 0.006 | +0.059 | 0.033 | -0.052 |
| 15 | 8.32 | 6.10 | 8.41 | 7.27 | -2.22 | -1.14 | -0.154 | -0.073 | +0.091 | 0.017 | -0.227 |
| 16 | 8.70 | 8.07 | 8.05 | 5.94 | -0.62 | -2.10 | -0.037 | -0.150 | -0.190 | 0.027 | -0.187 |
| 17 | 8.10 | 8.10 | 8.09 | 9.23 | -0.00 | 1.13 | -0.000 | 0.066 | +0.064 | 0.018 | +0.065 |
| 18 | 8.69 | 6.97 | 8.51 | 5.65 | -1.73 | -2.86 | -0.110 | -0.202 | -0.113 | 0.034 | -0.312 |
| 19 | 7.95 | 4.43 | 8.88 | 9.56 | -3.51 | 0.68 | -0.284 | 0.037 | +0.426 | 0.031 | -0.247 |

| pooled | mean | BCa 95% | raw 95% | gate |
|---|---:|---|---:|---|
| Delta_R_A (Hz) | -1.296 | [-1.776,-0.802] | [-1.785,-0.814] | falsified upper<0 = True |
| Delta_R_B (Hz) | -1.079 | [-1.658,-0.512] | [-1.659,-0.515] | falsified upper<0 = True |
| si_A | -0.092 | [-0.127,-0.056] | [-0.127,-0.056] | upper<0=True |
| si_B | -0.078 | [-0.121,-0.035] | [-0.121,-0.036] | upper<0=True |
| swap_pooled (mean SI_swap) | +0.025 | [-0.057,+0.108] | abs=0.025 | PASS |swap|<=0.10 & CI inc 0 = True |
| SI_many (abs) | 0.028 | [0.024,0.033] | |clean|<0.10 = True |
| Ulanovsky S=si_A+si_B | -0.170 | [-0.224,-0.118] | falsified S<=0 = True |

**Verdict (frozen ordering INVALID>UNRESOLVED>FALSIFIED>NEGATIVE>SUPPORTED): FALSIFIED** -- channel_SSA_contradicted+ulanovsky

Active falsifiers: FALSIFIER_Delta_R_A_BCa_upper_lt_0, FALSIFIER_Delta_R_B_BCa_upper_lt_0, FALSIFIER_si_A_BCa_upper_lt_0, FALSIFIER_si_B_BCa_upper_lt_0, FALSIFIER_ulanovsky_diagonal

Rarity-penalty H2 supported (penalty direction + SI_many clean): True

Phase-matched rarity control: declared_prospective_control (not yet executed; required for any future rarity-penalty claim) -- declaration satisfies prereg; execution required before mechanistic claim beyond discrimination.

Blinding: E2d_blinded_adequacy.json contains only G_finite/G_A/G_B/G_adequate + mean_rate; grep forbidden =0. Phenotype metrics in this receipt only after unblinding.

Evidence preserved: every run hashed (seq_hash per block) in e2d_confirmatory_receipt.json per_replicate; V2_RUNS_MANIFEST_copy.json retains 20 rep files; washout exactly first 2 events; max forbidden max_over_reps not used as gate (diagnostic only in V2).
