## Question

Does the jaxfne core, at the authors' parameters, reproduce the Vogels et al. (2011)
inhibitory-plasticity balance: the Fig. 4 recurrent network driven to its target rate
and the Figs. 1-2 single-cell detailed balance of excitation and inhibition?

## Model summary

A conductance-based leaky integrate-and-fire network, 8,000 excitatory and 2,000 inhibitory
cells at 2% sparseness, with spike-timing-dependent plasticity on the inhibitory-to-excitatory
synapses only: near-coincident pre- and postsynaptic spikes potentiate, every lone presynaptic
spike depresses. The coded target rate is rho0 = 3 Hz with a 20 ms trace window; strong
synapses that reliably prevent postsynaptic spikes weaken while weak ones strengthen, so
excitatory and inhibitory weights balance per input channel in feedforward and recurrent
networks alike.

## Source record

`artifacts/etudes/papers/vogels2011/SOURCE_P1.md` (P1): equations from the authors' MATLAB
and Brian code (the MATLAB script was included in the Science SOM), the parameter tables
with code pointers, the Fig. 4 and Figs. 1-2 target claims, and the gaps (paper full text
unreachable so paper-stated numbers stay unknown; conductance synapses vs the current-based
core LIF; no plastic-I-to-E rule in the core; near-zero inhibitory init; 10,000-cell scale;
code licence unknown).

## Status

P1: source record only, no scored run. Next per rule B8: pilot, then signature, then fresh seeds.
