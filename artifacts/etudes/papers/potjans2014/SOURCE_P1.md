# potjans2014: P1 source record

Target: Potjans TC, Diesmann M (2014), The cell-type specific cortical microcircuit: relating structure and activity in a full-scale spiking network model, Cereb Cortex 24(3):785-806, doi:10.1093/cercor/bhs358. Read from the open-access PMC record (PMCID PMC3920768; full text read 2026-10-19). Section/table pointers below name the paper's own sections, tables and figures. Every number carries its pointer; status observed or inferred; unknowns stay unknown, never filled.

## 1. Model equations (paper Table 4, "Model description after Nordlie et al. 2009"; Materials and Methods)

Network: 8 cortical populations (excitatory e and inhibitory i in each of L2/3, L4, L5, L6) plus 1 thalamic population Th of fixed-rate Poisson units. Connectivity is random: each synapse draws presynaptic and postsynaptic neurons independently, giving binomial in/out-degree (Materials and Methods, wiring paragraph). Connection probability of a pair follows the integrated map (Table 5 values in §2).

Neuron: current-based leaky integrate-and-fire with exponential synaptic currents, fixed voltage threshold, fixed absolute refractory period with voltage clamp (Table 4, "Neuron and synapse model"). In symbols (paraphrased, not quoted):

- Subthreshold: tau_m dV/dt = -(V - E_L) + R_m * I_syn; dI_syn/dt = -I_syn / tau_syn + sum_j w_j delta(t - t_j - d_j)
- Spike: V(t) >= theta implies t* = t with spike emission at t*
- Reset/clamp: V(t) = V_reset while t - t* < tau_ref
- Thalamus: independent Poisson trains at rate nu (background nu_bg; transient nu_th during input)

Paper-stated model facts: inhibitory postsynaptic amplitudes are negative and larger than excitatory by factor g (Table 5); synaptic strengths and delays are drawn from Gaussian distributions with a fixed sign and positive delays that are multiples of the computation step size (Materials and Methods, simulation paragraph); the L4e to L2/3e excitatory weight is doubled relative to the common excitatory weight (Table 5 caption note; Results, L4e to L2/3e paragraph). E_L and R_m values are not stated in the paper (unknown; the NEST default model the authors used would settle them, see G1).

Simulation frame (not a model equation; context for the target figures): NEST simulator, grid-constrained solver, step h = 0.1 ms; about 80 000 neurons and 0.3 billion synapses (Materials and Methods, simulation paragraph).

## 2. Parameters (paper Table 5, "Parameter specification"; external-input derivation in Materials and Methods "Layer-Specific External Input" with Table 3)

Population sizes N (Table 5, populations block). Status: observed.

| Population | L2/3e | L2/3i | L4e | L4i | L5e | L5i | L6e | L6i | Th |
|---|---|---|---|---|---|---|---|---|---|
| N | 20 683 | 5834 | 21 915 | 5479 | 4850 | 1065 | 14 395 | 2948 | 902 |

Sums are inferred arithmetic on observed inputs: 77 169 cortical neurons plus 902 thalamic units (78 071 total), consistent with the stated "about 80 000".

External Poisson inputs per neuron, reference parameterization k_ext (Table 5, populations block). Status: observed.

| k_ext (reference) | L2/3e | L2/3i | L4e | L4i | L5e | L5i | L6e | L6i |
|---|---|---|---|---|---|---|---|---|
| inputs | 1600 | 1500 | 2100 | 1900 | 2000 | 1900 | 2900 | 2100 |

Layer-independent control k_ext (Table 5). Status: observed: L2/3e 2000, L2/3i 1850, L4e 2000, L4i 1850, L5e 2000, L5i 1850, L6e 2000, L6i 1850. Background rate per synapse nu_bg = 8 Hz (Table 5). Status: observed.

Estimated input-type split per excitatory neuron (Table 3; derivation in Materials and Methods). Status: observed as estimates.

| External inputs | L2/3 | L4 | L5 | L6 |
|---|---|---|---|---|
| Thalamic | 0 | 93 | 0 | 47 |
| Gray matter | 534 | 353 | 389 | 79 |
| Other white matter | 1072 | 1665 | 1609 | 2790 |
| Total | 1606 | 2111 | 1997 | 2915 |

Integrated connectivity map: connection probabilities, presynaptic populations across, postsynaptic down, last column thalamic (Table 5, connectivity block). Status: observed.

| to \ from | L2/3e | L2/3i | L4e | L4i | L5e | L5i | L6e | L6i | Th |
|---|---|---|---|---|---|---|---|---|---|
| L2/3e | 0.101 | 0.169 | 0.044 | 0.082 | 0.032 | 0.0 | 0.008 | 0.0 | 0.0 |
| L2/3i | 0.135 | 0.137 | 0.032 | 0.052 | 0.075 | 0.0 | 0.004 | 0.0 | 0.0 |
| L4e | 0.008 | 0.006 | 0.050 | 0.135 | 0.007 | 0.0003 | 0.045 | 0.0 | 0.0983 |
| L4i | 0.069 | 0.003 | 0.079 | 0.160 | 0.003 | 0.0 | 0.106 | 0.0 | 0.0619 |
| L5e | 0.100 | 0.062 | 0.051 | 0.006 | 0.083 | 0.373 | 0.020 | 0.0 | 0.0 |
| L5i | 0.055 | 0.027 | 0.026 | 0.002 | 0.060 | 0.316 | 0.009 | 0.0 | 0.0 |
| L6e | 0.016 | 0.007 | 0.021 | 0.017 | 0.057 | 0.020 | 0.040 | 0.225 | 0.0512 |
| L6i | 0.036 | 0.001 | 0.003 | 0.001 | 0.028 | 0.008 | 0.066 | 0.144 | 0.0196 |

Target-specificity amendments applied while compiling the map (Table 2 with Materials and Methods "Consistent Modifications of Target Specificity"). Status: observed.

| Projection | T after amendment | Basis stated |
|---|---|---|
| L2/3e to L4 | -0.8 | paired recordings selecting against excitatory targets |
| L5e to L2/3 | -0.4 | photostimulation, preferred interneuron targeting |
| L2/3e to L6 | -0.4 | photostimulation, nonzero but low connectivity |
| L6e to L4 | -0.4 | electron microscopy |
| L2/3e to L5 | 0.29 | anatomical map fill of unmeasured connection |
| L4e to L5 | 0.32 | anatomical map fill of unmeasured connection |
| L2/3i to L5 | 0.4 | anatomical map fill of unmeasured connection |
| L4i to L2/3 | 0.23 | anatomical map fill of unmeasured connection |

Synaptic strengths and delays (Table 5, synapse block). Status: observed: excitatory weight w +- delta_w = 87.8 +- 8.8 pA; relative inhibitory strength g = -4; excitatory delays d_e +- delta_d_e = 1.5 +- 0.75 ms; inhibitory delays d_i +- delta_d_i = 0.8 +- 0.4 ms.

Neuron parameters (Table 5, neuron block). Status: observed: membrane time constant tau_m = 10 ms; absolute refractory period tau_ref = 2 ms; postsynaptic current time constant tau_syn = 0.5 ms; membrane capacity C_m = 250 pF; reset potential V_reset = -65 mV; fixed firing threshold theta = -50 mV; thalamic rate during input nu_th = 15 Hz.

Context for the non-target stimulus figure (Results, response section; not scored at P1): transient thalamic increase of 15 Hz for 10 ms; evoked spike counts use 0.5 ms bins over 100 network/input instantiations with recorded counts L3e 500, L3i 141, L4e 529, L4i 132, L5e 117, L5i 25, L6e 347, L6i 71.

## 3. Claims per figure the lane targets (paper Results; numbers as stated)

Target figure A: spontaneous cell-type specific activity (Fig. 6 with Table 6).

- C1 mean rates, reference parameterization (Fig. 6B; Table 6 model reference row). Status: observed: L2/3e 0.86 Hz; L4e 4.45 Hz; L5e 7.59 Hz; L6e 1.09 Hz. In every layer the inhibitory rate exceeds the excitatory rate (Results, spontaneous-activity paragraph).
- C2 mean rates over 100 randomly drawn external-input parameterizations (Fig. 7C; Table 6 "100 trials" row, mean +- SD of population rates). Status: observed: L2/3e 1.11 +- 0.8 Hz; L4e 4.8 +- 1.1 Hz; L5e 11 +- 6.1 Hz; L6e 0.56 +- 0.9 Hz.
- C3 robustness orderings across the 100 trials (Results, input-robustness paragraph). Status: observed: in 85% of simulations L2/3e and L6e fire below L4e while L5e is highest; in 83% the inhibitory rate exceeds the excitatory rate within a layer.
- C4 irregularity (Fig. 6C; statistics over 1000 spike trains, 60 s each). Status: observed: mean single-unit CV of interspike intervals above 0.8 in every population (exact per-population means are read from the boxplot only, so values stay qualitative).
- C5 synchrony (Fig. 6D; variance-over-mean of the 3 ms spike-count histogram, 5 s per population). Status: observed: highest in L5e, lowest in L6; membrane-potential synchrony confirms the asynchronous state (Supplementary Fig. 8, cited in text).
- C6 experimental context (Table 6 awake-state rows; not a model claim). Status: observed: mouse S1 L2/3e 0.61; rat V1 L2/3e 0.44; rat M1 L2/3e 0.36; rat S1 0.3 / 1.4 / 2-3 / 0.5 (L2/3e / L4e / L5e / L6e); rat A1 <1 / not stated / 3-4 / not stated; rabbit four areas <1 / not stated / 4-7 / <1.

Target figure B: the integrated connection probability table (Table 5, connectivity block, reproduced in §2).

- C7: the 72 recurrent probabilities plus 9 thalamocortical probabilities are the compiled map (Table 5). Status: observed as the numbers in §2; their provenance is the algorithmic merge of the anatomical map (Binzegger et al. 2004, Suppl. Table 1) and the physiological map (Thomson, West et al. 2002 with the Table 1 extensions), reconciled by the Gaussian lateral model and amended by the Table 2 target specificities (Materials and Methods).
- C8: structural checks on the maps (Fig. 2C-D; Results, map-comparison section). Status: observed: recurrence strength and loop strength are statistically indistinguishable between the two source maps (z-test, P > 0.1 and P > 0.05); half of the interlayer discrepancy indices are large.

## 4. Data, code and licence (checklist S4)

- Paper licence: Creative Commons Attribution Non-Commercial 3.0 (Observed; PMC3920768 copyright notice). Commercial reuse needs the publisher's permission.
- Authors' code location and licence: the paper states a reference wiring implementation (RandomPopulationConnect) and the network simulation will be made available with the next release of the NEST simulation tool, http://www.nest-initiative.org (Observed statement in Materials and Methods). The current location and its licence were not verified in this run (unknown); read as text only, never imported, when found (checklist S1).
- Data location and licence: the connectivity inputs are the published anatomical counts (Binzegger et al. 2004, their Fig. 12) and the paired-recording hit rates (Thomson, West et al. 2002, their Table 1, extended by the paper's Table 1) (Observed citations). Machine-readable supplementary tables (Suppl. Tables 1-3) are referenced at the publisher site; their current reachability was not verified in this run (unknown).
- Experimental rate context: the awake-state studies listed in Table 6 (Observed citations in §3 C6).

## 5. Gaps (checklist S4: each gap names what is missing and where the authors' code would settle it)

Model-reading map to the jaxfne core (observed this run): the core has a current-based LIF edge-list kernel (`jaxfne/emitters_lif.py` module docstring; `LIFParams` in `jaxfne/emitters_lif.py`; `LIFNetworkParams` in `jaxfne/emitters_lif.py`), per-edge delays in steps with delays >= 1 enforced (`_check_edges` in `jaxfne/emitters_lif.py`), Poisson external drive through `Simulation(poisson_drive=...)` (`compile_poisson_sources` in `jaxfne/inputs.py` compiles the mapping), and spike-train estimators including CV of ISIs (`cv_isi` in `jaxfne/analysis/metrics.py`) and synchrony measures (Fano factor `fano_factor` in `jaxfne/analysis/metrics.py`, kappa `fleiss_kappa_binary` in `jaxfne/analysis/metrics.py` and `wang_buzsaki_kappa` in `jaxfne/analysis/metrics.py`, `activity_regime` in `jaxfne/analysis/metrics.py`).

- G1: exact single-cell equivalence to the NEST `iaf_psc_exp` dynamics (exponential current kernel with tau_syn = 0.5 ms, absolute refractory clamp at V_reset for tau_ref = 2 ms, fixed threshold theta = -50 mV) is untested in the core at P1. The authors' NEST implementation (or the NEST release carrying it) would settle the update order and the unstated leak reversal.
- G2: the Gaussian weight/delay sampling rule (w +- delta_w, d_e/i +- delta_d_e/i, sign-preserving, delays snapped to multiples of h) has no port row at P1. The authors' wiring code would settle the truncation and snapping rule.
- G3: the L4e to L2/3e weight doubling is motivated in Supplementary Fig. 7 (convergence mismatch with Feldmeyer et al. 2006); its exact placement in the compiled map needs Suppl. Table 3 values. The authors' code and supplement would settle it.
- G4: machine-readable source maps (Suppl. Tables 1-3: raw/modified anatomical and physiological probabilities, amended connection values) were not fetched; the publisher supplementary links were not verified reachable. The authors' archive would settle the per-connection inputs to the merge.
- G5: the thalamic transient protocol (+15 Hz for 10 ms onto L4/L6 via the Th column) and the Fig. 10/11 propagation claims (latencies, gain, i-specific feedback effects) are deferred: not P1 target figures, no port rows. The authors' stimulation script would settle the exact Th drive construction (background overlap handling).
- G6: scale. The reference model is 77 169 cortical plus 902 thalamic neurons with about 0.3 billion synapses; no core run at any size exists at P1. Rule B8 requires a small pilot, then the signature, then fresh seeds before any scored run.
- G7: inhibitory subtypes (fast-spiking, low-threshold, somatostatin-labelled in Table 1 notes) are merged into one inhibitory population per layer; the merge weights (uniform split of pooled pair counts) are stated in the Table 1 notes and carry no uncertainty. Whether that merge preserves the target specificities is untested.
