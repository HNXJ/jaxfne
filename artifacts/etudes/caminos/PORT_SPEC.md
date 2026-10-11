# L2 port spec: the CAMINOS E–PV–SOM spiking network

Target papers:

- Tahvili, Vinck, di Volo, *Cell Reports* 2025, DOI 10.1016/j.celrep.2025.116131. This
  paper introduces the model. Its STAR Methods are still unread: cell.com sits behind a
  bot check and bioRxiv 10.1101/2025.02.23.639719 returns 429.
- Tahvili, Vinck, di Volo, *PLoS Comput Biol* 2026, DOI 10.1371/journal.pcbi.1014378
  (PMC13286281, CC-BY). Its Methods give the "reference model" spiking network in full.
  Copy: `F:\warehouse\tahvili\Tahvili2026_PLoSCB_PMC13286281.xml`, MD5
  0d98cd7384a63ce6c6f2411b8f168985. Mean-field code: Zenodo 10.5281/zenodo.20391404.

This spec takes every value from the PLoS CB Methods. That they equal the *Cell Reports*
network is an assumption until its STAR Methods are read. The PLoS CB text calls the
model "recently introduced" in the *Cell Reports* paper, which is its ref. 42.

## Model (PLoS CB Methods, Eqs. 15–18, Tables 1–2)

AdEx neurons:

```
C dV/dt = gL (EL − V) + gL Δ exp((V − VT)/Δ) − w + I
τw dw/dt = a (V − EL) − w,   w += b at each spike
```

A spike is registered when V ≥ VT + 5Δ. V is then reset to Vreset and held for Tref.

| Parameter | E | PV | SOM |
|---|---|---|---|
| VT (mV) | −50 | −50 | −50 |
| Δ (mV) | 2 | 0.5 | 2 |
| Tref (ms) | 5 | 5 | 5 |
| τw (ms) | 500 | 500 | 500 |
| a (nS) | 4 | 0 | 4 |
| b (pA) | 50 | 0 | 80 |
| C (pF) | 200 | 200 | 200 |
| gL (nS) | 10 | 10 | 10 |
| EL (mV) | −65 | −65 | −55 |
| Vreset (mV) | −65 | −65 | −65 |

Synapses are conductance based, `I_syn = g (E_rev − V)`, with
`τ_X dg/dt = −g + Q Σ δ(t − t_j − τd)`:

| Parameter | Value |
|---|---|
| τd (delay) | 2 ms, all synapses |
| τE / τI | 2 / 8 ms |
| E_rev E / I | 0 / −80 mV |
| Q PV,SOM → E,PV | 5 nS |
| Q E → E | 2 nS |
| Q E → PV,SOM | 3 nS |
| Q PV → SOM, SOM → SOM | 0 (absent in the reference model) |
| Q Ext → E,PV | 2.5 nS; SOM gets no external drive |

Network:

- NE = 8000, NPV = 750, NSOM = 750.
- Every pair connects independently with p = 0.07; no autapse rule is stated.
- External drive: p·NE = 560 independent Poisson sources at 4 Hz per target neuron.
- Brian2, Euler, dt = 0.1 ms.
- Population rate in Δt = 0.1 ms bins, smoothed with a 2 ms Gaussian for display.

## Targets (PLoS CB Results, Fig. 1; the reference model)

| Id | Observable | Paper value |
|---|---|---|
| R1 | Gamma oscillation frequency | ≈ 35 Hz |
| R2 | PV→SOM firing delay within a gamma cycle | ≈ 8.5 ms (method of their ref. 42) |
| R3 | Rate order | PV > SOM > E; E at 1–3 Hz |
| R4 | Oscillation onset with p (Fig. 1E) | oscillations above a critical p; asynchronous below it |

The *Cell Reports* observables (drive-dependent frequency, PV and SOM silencing,
SOM/PV density against frequency) follow once its STAR Methods are read. PLoS CB
Figs. 2–3 (40 % PV silencing or randomization) can be scored from this spec now.

## Port design

- Stage A, an etude here: a standalone JAX simulator on WSL2 CUDA in float32 (Hamm). It
  uses sparse fixed-in-degree or Bernoulli connectivity, stated per run, a 20-step
  (2 ms) delay ring buffer, and exponential conductances. It is integrated with Euler at
  dt = 0.1 ms, as Brian2 does.
- Stage B: a jaxfne AdEx emitter with conductance synapses. This adds to the public API,
  so it needs Hamm's word before it lands in `src/`.

## Gates

| Gate | Test | Pass |
|---|---|---|
| L2-G0 | Single AdEx cells under step currents (E, PV, SOM) against an exact-arithmetic or float64 reference of Eq. 15–16 | spike times match within one dt |
| L2-G1 | Network, 5 seeds, 3 s simulated (first 0.5 s discarded) | R1–R3 within the preregistered bands (P3) |
| L2-G2 | p sweep 0.02–0.10 | R4: a transition, with its p reported |

Bands for G1 are set in the preregistration and signed in Hamm's batch before any scored
run.
