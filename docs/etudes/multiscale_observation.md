# Étude: Multiscale observation (frozen protocol)

**Status:** prospectively frozen before the decisive run  
**Package baseline:** `d5cf9a6` (O1/O3 observation provenance; no numerical F/P split)  
**Bundle:** `artifacts/etudes/multiscale_observation/`  
**Package code:** not modified for this étude

This document is the experimental protocol. Results live in the bundle
(`metrics.json`, `manifest.json`, `gap_review.md`, figures). Do not retune
the protocol after seeing validation outcomes; a scientific change requires a
new protocol identity.

---

## 1. Question

Holding neural state and source history fixed, what does the observation
operator do?

\[
(X_{0:T},Q_{0:T},\mathcal G_{\rm source})
\text{ are immutable experimental causes;}
\quad
O_k\text{ is the intervention;}
\quad
Y_k=O_k(Q_{0:T}).
\]

Hypothesis: measured spatial/spectral structure is

\[
\text{measured spectrum}
=
\text{neural dynamics}
\times
\text{spatial source structure}
\times
\text{observation operator}.
\]

All field/probe outputs remain **relative proxy** quantities
(`amplitude_semantics=relative`, `validation_status=computational`,
`physical_claim=proxy_readout`) unless an explicit calibration transform is
applied (none is applied here).

---

## 2. Frozen neural system (group A)

| item | frozen value |
|------|----------------|
| seed | `7` |
| \(N\) | `40` |
| layers | `L2/3`, `L4`, `L5` with relative thicknesses `0.33, 0.34, 0.33` |
| cell types | `E: 0.7`, `PV: 0.3` |
| emitter | Izhikevich `cortical_eig` |
| HDP / AGSDR | off |
| duration | `2000` ms |
| \(dt\) | `0.5` ms |
| spectral burn-in | first `200` ms discarded for PSDs only |
| dtype | `float32` |
| JIT on simulate | `False` (eager evidence path) |
| drives | `E=8`, `PV=8` (native current) |

Record: \(V_m\), spikes, \(Q\), positions, source provenance, SHA256 of
\((V_m, \text{spikes}, Q, \mathcal G_{\rm source})\).

Simulate **once**. Every \(Y_k\) is post-hoc on that frozen \(Q\).

---

## 3. Observation interventions

### B — Locality (vary only operator geometry)

Package operator: `project_laminar_sources(Q, positions; n_contacts, width)`.

| id | \(n_{\rm contacts}\) | width |
|----|----------------------|-------|
| `lfp_ref` | 16 | 0.10 |
| `lfp_narrow` | 16 | 0.05 |
| `lfp_wide` | 16 | 0.25 |
| `lfp_sparse` | 8 | 0.10 |
| `lfp_dense` | 24 | 0.10 |

Contact-depth operators (declared Gaussian rows, `LinearReadout`):

| id | contact \(z\) | width |
|----|----------------|-------|
| `contact_shallow` | 0.20 | 0.10 |
| `contact_deep` | 0.80 | 0.10 |

Locality metric (1-D laminar, not a new package primitive):

\[
R_{90}(p)=\min\left\{R:\sum_{i:|z_i-c_p|\le R}|K_{p i}|
\ge 0.9\sum_i |K_{p i}|\right\}.
\]

Frequency-dependent effective contributing population: replace \(|K_{pi}|\)
by \(|K_{pi}|\,P_{Q,i}(f_{\rm band})\) and recompute \(R_{90}\). Static \(K\)
is not a frequency-dependent medium; any band dependence is source-structure
× operator, not impedance.

Authority target:

\[
K_a\neq K_b \implies Y_a\neq Y_b,
\qquad X_a=X_b,\; Q_a=Q_b.
\]

### C — Field versus derived observation

From `lfp_ref` compare \(KQ\) (`lfp_proxy` / `phi_e_proxy`) with
\(D_{zz}KQ\) (`csd_proxy`). Same \(Q\); different operators.

### D — Macroscopic observation

Declared EEG leadfields on the same \(Q\) via `LinearReadout` /
`eeg_proxy_transform`:

| id | construction |
|----|----------------|
| `eeg_superficial` | 3 sensors; Gaussian depth weights centered at \(z=0.25\), widths \(0.18,0.20,0.22\) |
| `eeg_deep` | 3 sensors; same widths centered at \(z=0.75\) |

`leadfield_status = toy_or_declared_proxy`.

Optional boundary (not required for success): `meg_relative` is a
per-source random-sign map of `eeg_superficial` weights on scalar \(Q\)
(not a physical MEG operator). Report `orientation_claim: none`. Do not
rename `meg_proxy_transform(..., source_oriented=...)`.

### E — Spectra

For burn-in-trimmed traces, package `spectrolaminar_psd_jax` at
\(f\in[1,150]\) Hz (96 bins):

\[
P_Q(f),\; P_{\rm LFP}(f),\; P_{\rm CSD}(f),\; P_{\rm EEG}(f).
\]

Ask which structures are preserved, attenuated, amplified, or mixed — not
merely whether PSDs differ.

### Negative control

Two `LinearReadout` objects with **identical** \(W=W_{\rm eeg,sup}\) and
different names / `leadfield_status` must yield identical \(Y\).

Compilation identity: `LinearReadout(W=K_{\rm ref})` must match
`lfp_ref.lfp_proxy` within float32 tolerance.

---

## 4. Success levels (predeclared)

**Level A — computational closure.** One \((X,Q)\); many \(O_k\); provenance
present on the eager path; \(Q\) hashes invariant across observations.

**Level B — observation authority.** Distinct numerical \(O\) ⇒ distinct \(Y\);
identical numerical \(O\) ⇒ identical \(Y\) (negative control).

**Level C — neurophysiological consequence.** Operator changes alter
interpretable spatial/spectral measurements (e.g. \(R_{90}\) vs width;
LFP vs CSD spectral centroid; superficial vs deep EEG band ratios), not
merely bitwise inequality.

---

## 5. Gap review (after the run)

Each perceived deficiency is classified as exactly one of:

```text
ETUDE_PRESENTATION_ONLY
ANALYSIS_GAP
GENERAL_OPERATOR_GAP
PHYSICAL_MODEL_GAP
NO_GAP
```

Only `GENERAL_OPERATOR_GAP` is presumptively eligible for further 0.4.15
package work. Physical forward models, conductivity, realistic EEG/MEG, and
calibration are `PHYSICAL_MODEL_GAP` unless required for the 0.4.15 claim.

---

## 6. Reproduce

```bash
python scripts/run_multiscale_observation_etude.py
```

Emitter variation is a **later** étude on the same observation stack.
