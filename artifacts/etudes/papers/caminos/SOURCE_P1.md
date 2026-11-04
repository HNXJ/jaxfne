# CAMINOS reference model (Tahvili, Vinck, di Volo): P1 source record

Status: draft from a Muse read of PORT_SPEC.md, CALIBRATION.md, LEDGER.md, PREREG_P3.md, PREREG_P3_v7_R4.md, PREREG_P6.md and the PLoS Comput Biol 2026 full text (PMC13286281), 2026-10-18; unverified by the dispatcher.

All items labeled **observed** (with source) or **inferred**. Values neither the lane records nor the passages read give are **unknown**. No passage quoted over 15 words. Port outcomes cited only as pointers (CALIBRATION.md sections, RESULT_P6.md).

Target papers: Tahvili, Vinck, di Volo, "PV and SOM cells play distinct causal roles in controlling network oscillations and stability", Cell Reports 44(8):116131 (2025), doi:10.1016/j.celrep.2025.116131, which introduces the CAMINOS model (observed PORT_SPEC.md:5-6); and Tahvili, Vinck, di Volo, "A mean-field model of neural networks with PV and SOM interneurons reveals connectivity-based mechanisms of gamma oscillations", PLoS Comput Biol 22(6):e1014378 (2026), doi:10.1371/journal.pcbi.1014378 (PMC13286281, CC-BY), whose Methods give the ported "reference model" spiking network in full (observed PORT_SPEC.md:8-11). The Cell Reports STAR Methods are unread (bot check; preprint 10.1101/2025.02.23.639719 returns 429), so the spec takes every value from the PLoS CB Methods (observed PORT_SPEC.md:13-15).

## 1. The model (PLoS CB Methods, Eqs. 15-18, Tables 1-2, via PORT_SPEC.md:17-64)

- Cells: AdEx neurons, one type each for E, PV and SOM:
  `C dV/dt = gL (EL - V) + gL Δ exp((V - VT)/Δ) - w + I`;
  `τw dw/dt = a (V - EL) - w`, with `w += b` at each spike (observed PORT_SPEC.md:19-24).
  A spike registers at V >= VT + 5Δ; V resets to Vreset and holds for Tref (observed PORT_SPEC.md:26).
- Parameters (observed PORT_SPEC.md:28-39): VT -50 mV all; Δ 2 (E, SOM) / 0.5 (PV) mV; Tref 5 ms all; τw 500 ms all; a 4 / 0 / 4 nS (E/PV/SOM); b 50 / 0 / 80 pA; C 200 pF all; gL 10 nS all; EL -65 (E, PV) / -55 (SOM) mV; Vreset -65 mV all.
- Synapses: conductance based, `I_syn = g (E_rev - V)`, with `τ_X dg/dt = -g + Q Σ δ(t - t_j - τd)` (observed PORT_SPEC.md:41-42). Delay τd 2 ms on all synapses; τE / τI 2 / 8 ms; E_rev 0 (exc) / -80 (inh) mV (observed PORT_SPEC.md:44-48). Strengths Q: PV,SOM -> E,PV 5 nS; E -> E 2 nS; E -> PV,SOM 3 nS; PV -> SOM and SOM -> SOM 0 (absent in the reference model); external -> E,PV 2.5 nS; SOM gets no external drive (observed PORT_SPEC.md:49-53).
- Connectivity: NE 8000, NPV 750, NSOM 750; every pair connects independently with p = 0.07 (observed PORT_SPEC.md:55-58). The source states no autapse rule; the port removes autapses as its own choice (observed PORT_SPEC.md:58-59).
- Inputs: p·NE Poisson sources at 4 Hz per target E and PV cell; SOM undriven (observed PORT_SPEC.md:60-61, PREREG_P3.md:34). Reading "p·NE Poisson neurons" as private sources per cell is an assumption; the text also allows shared sources (observed PORT_SPEC.md:61, PREREG_P3.md:34).
- dt/duration/integrator: Brian2, Euler, dt = 0.1 ms (observed PORT_SPEC.md:62). Scored runs simulate 12 s and discard the first 2 s so adaptation settles (observed PREREG_P3.md:39-40). Population rate in 0.1 ms bins, smoothed with a 2 ms Gaussian for display (observed PORT_SPEC.md:63-64).

## 2. Training rule

None. All parameters are fixed; the network is simulated, not fitted. The paper separately derives a mean-field model whose transfer functions are computed for the spiking parameters (observed article Results). The port scores only the spiking network (inferred from PORT_SPEC.md and PREREG_P3.md, which fix every value).

## 3. Claims per figure the lane targets

The lane targets Fig. 1 of the PLoS CB paper (observed PREREG_P3.md:11-18):

- R1. The network oscillates at about 35 Hz (low gamma).
- R2. SOM fires after PV within each gamma cycle, delay about 8.5 ms, by the method of the paper's ref. 42 (the Cell Reports paper). The port's P3 estimator differs and is its own (observed PREREG_P3.md:14-16).
- R3. Rates order PV > SOM > E, with E at 1-3 Hz.
- R4. Oscillations appear above a critical connection probability p (Fig. 1E).

Further figures exist and are outside the lane: Figs. 2-3 (PV/SOM silencing and spike-time randomization at 40% and 90%), Fig. 4 (SOM/PV density-ratio gradients and stability under transient E-E increase), Fig. 5 (adding PV -> SOM and SOM -> SOM couplings; a reduced model without synaptic timescales or adaptation), and the linear rate model with its phase-lead result (observed article Results). PLoS CB Figs. 2-3 (40% PV silencing or randomization) can be scored from the current spec (observed PORT_SPEC.md:75-76); the Cell Reports observables follow once its STAR Methods are read (observed PORT_SPEC.md:75).

## 4. Data and code availability with licence

- Paper licence: the PLoS CB article is open access under CC-BY (observed PORT_SPEC.md:8-9 and the article licence statement).
- Mean-field code: Zenodo 10.5281/zenodo.20391404 (observed PORT_SPEC.md:11). Licence CC-BY 4.0 (Zenodo API, latest version 10.5281/zenodo.20391405, read 2026-10-26).
- Spiking-network code location: **unknown** (the lane records read name only the mean-field archive).
- Experimental data: the model is compared against visual-cortex recordings during stimulation (article Results, citing its refs. 19, 33, 45); reuse details are **unknown** here.
- Cell Reports paper: Europe PMC lists it as not open access with no PMCID (observed CALIBRATION.md:89-94); its licence is **unknown**.

## 5. Gap list

- G1: Cell Reports STAR Methods unread; all values rest on the PLoS CB Methods.
- G2: That the Cell Reports network equals the PLoS CB reference model is an assumption (observed PORT_SPEC.md:13-15); the PLoS CB text calls the model "recently introduced" in the Cell Reports paper, its ref. 42 (observed PORT_SPEC.md:15).
- G3: Private vs shared external drive (see §1); the shared reading stays untested (observed dashboard sources.json, caminos sensitivity S4).
- G4: Autapse rule unstated; port removes autapses (observed PORT_SPEC.md:58-59).
- G5: Fig. 1E numeric values; the local copy carries no Fig. 1E image (observed CALIBRATION.md:33, which scores onset qualitatively).
- G6: The ref. 42 lag method reached the lane only as a ruling paraphrase; the port's band-pass band and peak rule are its own choices (observed CALIBRATION.md:42-53, 87-94).
- G7: Mean-field transfer-function fitting details (fitting procedure, T = 15 ms choice) are not used by the port (observed article Results; inferred from PORT_SPEC.md, which ports the spiking network only).
- G8 (open fidelity item F1): the port's 2 ms synaptic delay differs from the preprint model reading, which has no synaptic delay; recorded, not changed (observed CALIBRATION.md:96-100).
- Port-result pointers only: estimand table and the ref. 42 lag finding (CALIBRATION.md §1-2, §10); P5/P6 verdicts (LEDGER.md; RESULT_P6.md).
