# Stage 1: title and abstract proposal (draft 1, 2026-11-19)

Status: proposal. Excluded from the mock-review loop (stage 10) until the owner's methodological lock. Claims in brackets are pending verification and must not be stated as results.

## Title

Plasticity Boat: a closed-loop task for comparing plasticity rules in spiking cortical circuits

## Abstract (draft)

Spiking cortical models are usually evaluated on passive stimuli, which cannot test whether a plasticity rule supports goal-directed control. We introduce Plasticity Boat, a closed-loop task in which a leaky integrate-and-fire (LIF) circuit steers a point mass toward a target using only its sensory input and the spikes of four motor populations. In a 10-neuron circuit, a distance-only sensor cannot express direction: the controller drives the mass away from the target from every development start, and starts on the side of the target where the force points toward it approach it (closest approach 0.27 in one of 20 starts) before running away. Directional observability is therefore a precondition for measuring any plasticity rule. [Directional positive control, 10 neurons: pending, stage 3.] We then specify a 100-neuron circuit and a competition of eight plasticity rules under paired starts, shared circuit initialisation and frozen metrics. [Competition result: pending, not claimed.]

## Claim ids used in the abstract

- A1: a distance-only sensor cannot express direction (evidence C2, C3, C4, R-09, R-11, R-12).
- A2: directional observability is a precondition (depends on the positive control, EX-24; not yet verified).
- A3: paired-start, frozen-metric competition design (protocol_v0.md; no result).

## Terms

LIF; point mass (the boat); development starts; held-out starts; S1 to S8 strategies as defined in protocol_v0.md §3.
