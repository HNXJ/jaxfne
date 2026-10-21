## Question

Does the jaxfne core reproduce the Potjans–Diesmann cortical microcircuit's spontaneous
asynchronous-irregular state (Potjans and Diesmann 2014, Cereb Cortex 24(3):785-806, Fig. 6):
layer- and type-specific mean rates (L2/3e about 0.86 Hz, L4e about 4.45 Hz, L5e about
7.59 Hz, L6e about 1.09 Hz, inhibitory above excitatory per layer), single-unit CV above
0.8, and the lowest synchrony in L6 with the highest in L5e?

## Formulation

Eight current-based leaky integrate-and-fire populations (excitatory and inhibitory in
L2/3, L4, L5, L6; 77 169 neurons) plus a 902-unit Poisson thalamic population, randomly
connected by the integrated 8-by-9 connection probability map (paper Table 5) with
Gaussian weights (87.8 ± 8.8 pA excitatory, inhibitory at g = −4, L4e→L2/3e doubled) and
Gaussian delays (1.5 ± 0.75 ms excitatory, 0.8 ± 0.4 ms inhibitory). Cell parameters:
tau_m 10 ms, tau_ref 2 ms, tau_syn 0.5 ms, C_m 250 pF, V_reset −65 mV, threshold −50 mV.
Reference drive: cell-type-specific Poisson inputs (1600–2900 per cell) at 8 Hz per
synapse. Full P1 source record: `artifacts/etudes/papers/potjans2014/SOURCE_P1.md`.
