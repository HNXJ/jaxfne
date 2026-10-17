# Destexhe (2009): P1 source record

Target: Alain Destexhe, "Self-sustained asynchronous irregular states and Up–Down
states in thalamic, cortical and thalamocortical networks of nonlinear
integrate-and-fire neurons", J Comput Neurosci 27:493–506 (2009),
doi:10.1007/s10827-009-0164-4. Read 2026-10-14 from the author's preprint
arXiv:0809.0654v4 ([q-bio.NC], 10 May 2009, "in press"); local text extraction in
`scratch/destexhe2009/p00.txt`–`p12.txt` (13 pages). EuropePMC confirms the
bibliography (J Comput Neurosci 27(3), Dec 2009, pp. 493–506, PMID 19499317) and
lists the same arXiv PDF as the free full text (Unpaywall). The paper is
Springer copyright; we use its equations and parameters, with no figure
reproduced. ModelDB code was not read in this pass (see gap G8).

"pN" below = preprint text page pN (`§X.Y` = paper section).

## 1. Single-cell aeIF model (§2.1–2.2, p1–p2)

Equations (paper Eqs. 1–2):

```
Cm dV/dt = −gL (V − EL) + gL Δ exp[(V − VT)/Δ] − w/S − Σj gji (V − Ej)
dw/dt    = (1/τw) [a (V − EL) − w]
```

- Cm = 1 μF/cm² (specific), gL = 0.05 mS/cm² (specific), EL = −60 mV (also the
  reset value after spike), Δ = 2.5 mV, VT = −50 mV, S = 20 000 μm² membrane
  area, refractory clamp 2.5 ms at reset after spike, τw = 600 ms. When V reaches
  threshold a spike is emitted and V is reset and clamped for the refractory
  period. At each spike w is incremented by b.
- Absolute-current reading used by this port (divide Eq. 1 by S; S = 2e-4 cm²):
  C = Cm·S = 200 pF, gL = 10 nS, a[nS] = a[μS] × 1000, b[pA] = b[nA] × 1000,
  w in pA. Check: w = 1 nA gives (w/S)/Cm = 5 mV/ms, and 1000 pA / 200 pF =
  5 mV/ms. Consistent.

Cell types (all from §2.2; Fig. 1 shows ±0.25 nA pulses):

| Type | a | b | Absolute (port) | Claim |
|---|---|---|---|---|
| RS, strong adaptation (Fig. 1A) | 0.001 μS | 0.04 nA | a = 1 nS, b = 40 pA | adapting regular spiking (cortical PY) |
| RS, weak adaptation (Fig. 1B) | 0.001 μS (assumed, only b varied) | 0.005 nA | a = 1 nS, b = 5 pA | weakly adapting |
| FS (Fig. 1C, cortical IN) | unstated (gap G1) | 0 | a = 1 nS (assumed), b = 0 | negligible adaptation |
| LTS (Fig. 1D) | 0.02 μS | 0 | a = 20 nS, b = 0 | adapting + rebound burst on hyperpolarizing pulse |
| TC (Fig. 1E) | 0.04 μS | 0 | a = 40 nS, b = 0 | moderate adaptation + strong rebound burst |
| RE (Fig. 1F) | 0.08 μS (corrected, gap G2) | 0.03 nA (corrected, gap G2) | a = 80 nS, b = 30 pA | bursts on both depolarizing and hyperpolarizing stimuli |

Shared by all types unless stated: C = 200 pF, gL = 10 nS, EL = −60 mV,
VT = −50 mV, Δ = 2.5 mV, Vreset = −60 mV, Tref = 2.5 ms, τw = 600 ms. The
network equations index Δi, VTi by cell type (§2.3) but no per-type values are
given; the port takes them uniform (gap G7).

## 2. Synapses (§2.3, p3)

Conductance based, exponential time course: on presynaptic spike a fixed
increment (ge / gi) is assigned to gji, then exponential decay with τE = 5 ms
(excitation), τI = 10 ms (inhibition). Reversals: 0 mV (exc), −80 mV (inh).
No synaptic delays. Typical strengths ge = 6 nS, gi = 67 nS (citing Vogels and
Abbott 2005; postsynaptic sizes 11 mV / 8.5 mV at −70 mV at rest, ~0.5 mV in
the active network). The Fig. 2 minimal circuit uses ge = gi = 30 nS instead.

## 3. Networks (§2.4, p3–p4)

- Thalamus (§2.4.1): TC relay layer + RE layer, no TC–TC connections, no
  interneurons. TC→RE 2%, RE→TC ~8%, RE→RE 8%. 10× fewer neurons than cortex.
  Baseline N = 100; when size varies, p is rescaled inversely to N so synapses
  per neuron are invariant.
- Cortex (§2.4.2): 80% PY (RS), 20% IN (FS). PY→PY, PY→IN, IN→PY, IN→IN all
  2%. Baseline N = 2000 (1600 PY + 400 IN). Same inverse-p size scaling.
- Thalamocortical (§2.4.3, Table 1, p4): 1600 PY + 400 IN + 100 TC + 100 RE.
  Outgoing-p table with afferents-per-neuron for that composition:
  PY→PY 2% (32), PY→IN 2% (32), IN→PY 2% (8), IN→IN 2% (8), PY→TC 2% (32),
  PY→RE 2% (32), TC→RE 2% (2), RE→TC 8% (8), RE→RE 8% (8). Text adds
  TC→PY and TC→IN at 2% each (absent from the table, gap G3). Checks:
  cortical synapses are 80% of TC afferents (32/40); thalamic synapses are ~6%
  of PY excitatory afferents (2/34).
- Two-layer cortex (§2.4.4): Layer A (N = 2000 RS+FS, as Fig. 6A) + Layer B
  (N = 500 with LTS, as Fig. 7B but 10% LTS); reciprocal excitatory-only
  interlayer connectivity at 1% (half the 2% intralayer density).
- Initiation (§2.3, p3): 2–10% of neurons receive random excitatory inputs for
  the first 50 ms at 200–400 Hz mean frequency. Afterwards no input and no
  added noise; the only noise source is the random connectivity ("quenched
  noise"). All equations solved in NEURON (Hines and Carnevale 1997).

## 4. Quantification (§2.5, p4–p5)

- Irregularity: CV_ISI = ⟨σ_ISI,i / ISĪ_i⟩ over all cells (Eq. 3); irregular
  iff CV_ISI > 1 (CV = 1 for Poisson).
- Synchrony: CC = ⟨Cov(Si,Sj) / σ(Si)σ(Sj)⟩ over 500 disjoint pairs (Eq. 4),
  spike counts in 5 ms bins (also checked with 2 ms); asynchronous iff CC < 0.1.

## 5. Figure claims

| Fig | Network | Paper values |
|---|---|---|
| 2 (p5) | 2 TC + 2 RE circuit, ge = gi = 30 nS | self-sustained oscillation ~10 Hz (caption: ~12 Hz); RE fires every cycle, TC subharmonically every 2nd cycle |
| 3 (p5–p6) | thalamic N = 20 (10 TC + 10 RE), ge = 6, gi = 67, Table-1 p | aperiodic oscillation after 50 s transient: CV_ISI = 1.36, CC = 0.025 |
| 4 (p6) | thalamic N = 40 / 60 / 100, same weights | N = 40: 1.45 / 0.027; N = 60: 1.35 / 0.02; N = 100: 1.47 / 0.016 |
| 5 (p6) | thalamic N = 20 / 100, ge × gi sweep | self-sustained irregular states for ge > ~4 nS and gi > ~40 nS; <10 s transients excluded |
| 6A (p6–p7) | cortical N = 2000 RS+FS, strong adaptation | transient AI, dies within a few seconds |
| 6B (p7) | same, weak adaptation | self-sustained AI: CV_ISI = 2.47, CC = 0.005 |
| 7A (p7) | cortical N = 400, 5% LTS (of PY) | intermittent SR/AI: CV_ISI = 2.84, CC = 0.05 |
| 7B (p7) | cortical N = 500, 5% LTS | AI: CV_ISI = 2.07, CC = 0.05 |
| 8 (p7) | N = 500 + LTS, 50 s | LTS–FS cross-correlation positive peak at ~70 ms lag (post-inhibitory rebound) |
| 9 (p7–p8) | cortical N = 80–2000, LTS 0–20% | minimal N for AI: 1800 (0%), 800 (2.5%), 300 (5%), 200 (10%), 100 (20%) |
| 10 (p8) | thalamocortical 2200 cells, b 0.04 → 0.005 nA | A: Up/Down states; D: continuous AI, cortical mean rate 44 Hz, CV_ISI = 2.45, CC = 0.004 |
| 11 (p8–p9) | same, Fig. 10B vs 10D cells | Up/Down vs sustained cellular activity; TC rebound burst inset |
| 12B (p9) | thalamocortical, b 0.02 → 0.005 nA | transition Up/Down → activated state (cf. PPT stimulation in vivo, panel A) |
| 13 (p9) | two-layer cortex, stim at t = 250 ms | self-sustained Up/Down coexisting with resting state: CV_ISI = 2.49, CC = 0.069 |
| text (p8) | thalamocortical AI/Up states | mean rate ~40 Hz on average (above experiment; small-network effect) |
| text (p10) | thalamic N = 20 | robust to 100 s integration |

## 6. Gaps

- G1: FS value of a unstated (only b = 0 given). Port assumes a = 0.001 μS.
- G2: RE parameters printed with swapped units ("b = 0.08 μS and a = 0.03 nA",
  p2). Port reads a = 0.08 μS, b = 0.03 nA (both "larger values" vs TC).
- G3: TC→PY / TC→IN probabilities absent from Table 1; text gives 2% (§2.4.3).
- G4: NEURON step dt and spike-detection threshold unstated. Port: dt = 0.1 ms
  explicit Euler, threshold VT + 5Δ (Brette and Gerstner 2005), as in
  `jaxfne/emitters_adex.py:adex_step`.
- G5: initiation stimulus shape and weight unstated (only fraction 2–10%,
  50 ms, 200–400 Hz). Port: Poisson conductance drive, 5% of cells, 300 Hz,
  ge-sized kicks (PORT_SPEC.md).
- G6: Fig. 10B/C adaptation values unstated (only A: b = 0.04, D: b = 0.005;
  Fig. 12B uses b = 0.02 → 0.005).
- G7: Δi, VTi indexed by cell type but no per-type values; port uniform.
- G8: ModelDB code read 2026-10-14 (accession 126466, mirror
  `github.com/ModelDBRepository/126466`, vendored read-only under
  `scratch/destexhe2009/code126466/`, read as data, never run). It is a
  PyNN conversion (Davison/Muller) of Destexhe's hoc, with Fig. 7 and
  Fig. 13 demos only — no Fig. 6 demo. Confirms: FS a = 0.001 µS (G1
  resolved, port assumption correct); DT = 0.1 fixed-step; delays of one
  step; V0 = −60 uniform with w = 0; spike detection ≈ VT
  (`v_thresh = v_spike ≈ VTR`, new mismatch F6); capped Bernoulli wiring
  with inverse-N p rescaling (F7); kick = N/5 cells × 20 synapses ×
  exponential 70 ms ≈ 285.7 Hz aggregate for 50 ms (code uses 20% of
  the net, outside the paper's stated 2–10% range). NEURON fixed-step
  method detail (G4) remains unstated.
- G9: Up/Down durations and per-phase rates are never quantified in the paper
  (only rasters, CV/CC of Fig. 13, and the ~40 Hz mean rate); duration
  criteria are the port's own (PREREG_P3.md).
- G10: autapse rule unstated; port excludes autapses.
- G11: thalamic N = 100 composition unstated; port takes 50 TC + 50 RE (as the
  N = 20 case, 10 + 10).
