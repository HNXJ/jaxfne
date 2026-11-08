# Jitter reader on a carrier: model spec (dispatcher, 2026-10-22)

Question (Hamm's group, next week): how does a reader neuron preferentially decode ms/sub-ms timing
("jitter") in its inputs while the population oscillation (carrier) is loud? Hypothesis under test
(Mythos, refined): the carrier is a clock that the reader recovers locally through feedforward
inhibition (FFI) and removes as a common mode; the decoded variable is early-vs-late timing within a
cycle. STDP (as HDP) learns the jitter channel.

## Circuit (TFNE grammar: Emitter -> Source -> Field -> Probe; CircuitSpec -> construct -> simulate)

- A: N_A = 100 afferents (input emitters, spike times given).
  - Carrier: every afferent fires as an inhomogeneous Poisson process,
    rate r(t) = r0 * (1 + m * cos(2 pi f t + phi0)), r0 = 10 Hz, f = 40 Hz (gamma), m in {0, 0.5, 0.9}.
  - Payload: K = 20 "pattern" afferents add one spike per carrier cycle at a fixed phase offset
    d_k (drawn once, uniform in [0, 5] ms after the cycle's rate peak) plus Gaussian jitter sigma_j.
    Two stimulus classes differ only in these offsets: class P (offsets d_k) and class Q (offsets
    shuffled across the K afferents). Same per-afferent rates, same carrier: rate-identical by build.
  - Clock objection: carrier frequency drift, f(t) = f0 + df * slow noise (df in {0, 5} Hz), the
    pattern stays locked to the instantaneous carrier phase.
- I: N_I = 20 PV-like fast inhibitory emitters, driven by all A (pooled, weak, dense).
- R: 1 to 10 reader emitters (AdEx or LIF, fluctuation-driven regime), receiving A -> R excitation
  (AMPA, tau ~ 2 ms) and I -> R inhibition (GABA_A, tau ~ 5 ms) with delay 2 ms (Pouille & Scanziani).
  Condition FFI-off: I -> R weights 0 (same total excitation).

## Plasticity as HDP (dW/dt = F_W(H)); H = superposition of independent components

- H components per A -> R synapse, each its own RBD (dH/dt = F_H):
  - H_pre:  dx/dt = -x / tau_+ + sum_pre delta(t - t_pre),   tau_+ = 17 ms
  - H_post: dy/dt = -y / tau_- + sum_post delta(t - t_post), tau_- = 34 ms
- F_W = A_+ * x * S_post(t) - A_- * y * S_pre(t), with soft bounds w in [0, w_max]
  (each term is driven by one independent H component; the rule is their superposition).
- Typed coupling map: H -> W only (H never feeds X directly). No other plasticity.

## Measurements (Probe -> Objective)

1. Readout information: discriminability of P vs Q from the reader's output (spike count per cycle
   and first-spike latency within cycle), as d' or decoding accuracy, versus m, sigma_j in
   {0.25, 0.5, 1, 2, 4} ms, FFI on/off, df.
2. Carrier leakage: reader output power at f relative to its input (how much carrier the reader
   re-reports), FFI on/off.
3. Learning: with HDP on, starting from uniform weights, do pattern afferents' weights separate from
   carrier-only afferents (Masquelier-style), and does P-vs-Q readout rise over training.
Predictions: FFI raises P/Q discriminability and lowers carrier leakage; discriminability falls as
sigma_j approaches the FFI window (~2 ms); drift df hurts little when FFI recovers the clock locally;
HDP selects pattern afferents.

## Rules
Everything through the public jaxfne API and grammar. A mechanism the core lacks (explicit input
spike trains, per-synapse delay, trace-based HDP superposition) goes into jaxfne/ with its own tests,
never a script-side shortcut. Seeds: report two seeds and show they change the output.
