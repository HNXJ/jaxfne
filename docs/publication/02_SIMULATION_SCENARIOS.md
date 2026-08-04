# Progressive Simulation Scenarios

All scenarios run through jaxfne operator grammar:
  Configuration → Construct → Simulate → Source → Field → Probe → Objective → Optimizer → Manifest

H(t) equation active in all stages unless explicitly ablated:
    τ_H · dH/dt = r_target - r(t)

---

## Stage 0
Single neuron equilibrium.

Outputs: Vm, spikes, adaptation state H(t).
Verify H(t) converges to steady-state at r_target.

## Stage 1
Repeated pulse adaptation.

Measure: firing-rate decay, recovery time, adaptation index.
Reference: Turrigiano & Nelson 2004 (qualitative match expected).

## Stage 2
Population adaptation.

Measure: synchrony, firing rate, H distribution across population.

## Stage 3
Frequency sweep.

Stimulus frequencies: 1, 2, 5, 10, 20, 40 Hz.
τ_H calibration: sweep should show timescale-dependent adaptation depth.

## Stage 4
Amplitude sweep.

## Stage 5
Duration sweep.

## Stage 6
Random stimulus trains.

## Stage 7
Classical oddball.

Standard probability: 80%. Deviant: 20%.
Measure: SSA index.
Compare to Abbott/Varela synaptic-depression null model.

## Stage 8  ← PRIMARY RESULT
Omission paradigm.

Exact stimulus schedule matched to in-vivo recording protocol.

Measure:
  - Omission response presence/absence in LFP proxy field signal
  - Laminar profile (superficial vs. deep layer power)
  - Spectrolaminar motif (γ superficial, α/β deep)
  - Response polarity (rebound vs. suppression)         — empirical result, not assumed
  - Sparse vs. dense spiking at omission time           — empirical result, not assumed
  - Feedforward vs. feedback laminar signature          — empirical result, not assumed
  - H(t) trace at omission onset

Null control: same schedule with H(t) removed (ablation).
Expected from theory: H(t) produces rebound (release from suppression) at omission onset
because sustained stimulation drives H below set-point, and absence of drive allows rebound.
Actual result takes precedence over expectation.

## Stage 9
Global-local oddball.

Hierarchical sequence violations across cortical areas.

## Stage 10
Long-term adaptation.

Thousands of trials. Verify H(t) stable at long timescales.

## Stage 11
Multiple cortical areas.

Test inter-area propagation of H(t) effects.

## Stage 12
Jaxley emitter validation.

Replace Izhikevich with HH/Jaxley emitter — generalization check.

## Stage 13
Optimization with AGSDR.

## Stage 14
Parameter recovery.

Verify τ_H and r_target are identifiable from simulated data.

## Stage 15
Cross-model comparison.

Izhikevich vs. LIF vs. HH/Jaxley.
Claim: omission response is model-class-independent given H(t).
