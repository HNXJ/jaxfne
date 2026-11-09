## Question

Does the jaxfne Jansen-Rit emitter reproduce the Fig. 3 regime sequence of the single-column model across the C sweep (noise, noisy alpha, alpha, spike-wave-like, noise), the sharpening of alpha between C = 128 and C = 135, the slowing of spike-wave output between C = 270 and C = 675, and the EEG level at C = 135?

## Model summary

A single cortical column of Jansen and Rit (1995): a pyramidal population driven by an excitatory and an inhibitory interneuron population, each with a second-order post-synaptic kernel (A = 3.25 mV, B = 22 mV, a = 100 s^-1, b = 50 s^-1) and a sigmoid (e0 = 2.5 s^-1, v0 = 6 mV, r = 0.56 mV^-1). The connectivity constant C scales the loop: C1 = C, C2 = 0.8 C, C3 = C4 = 0.25 C. The input is uniform random pulses at 120 to 320 s^-1. The EEG proxy is y1 - y2. The emitter is `simulate_jansen_rit` in `jaxfne/emitters_jansen_rit.py`.

Declared choices, because the paper does not state them: one input draw held per 2 ms; fixed-step RK4 at dt 0.1 ms in place of the paper's RKF45; 10 s from the zero state with the first 1 s discarded; spectra from Hann-windowed 4 s segments at 50 % overlap, peak searched in 1 to 30 Hz.

Classes at the scored spectrum: noise if the oscillation index `osc` is below 0.4; alpha if `osc` is at least 0.4 and the peak lies in 8 to 13 Hz; spike-wave if `osc` is at least 0.4, the peak is below 8 Hz and the EEG SD is at least 5 mV.

## Source record

`artifacts/etudes/papers/jansenrit1995/SOURCE_P1.md` (P1): the equations, the standard parameter set (paper Eq. 17), the Fig. 3 target and the gaps. The paper gives no step size, noise hold or run length.

## Status

Class: reproduced (conditional).

Condition (owner decision 2 of the signed preregistration, ruled as drafted): the 2 ms noise hold and RK4 at dt 0.1 ms replace the paper's unstated hold and RKF45. The pass holds under that declared hold and integrator only.

P3 v1 scored PASS, 5 of 5 seeds (10 to 14; rule: at least 4 of 5), against the signed preregistration `prereg/P3_v1_signed.md`. Result: `artifacts/etudes/papers/jansenrit1995/RESULT_P3.md`; score `artifacts/etudes/papers/jansenrit1995/score/p3_v1_score.json` (sha256 ca0c9d75...). The scorer was written after the driver had printed the scored rows. Smallest margins: noise `osc` up to 0.26 against 0.4; spike-wave EEG SD 11.8 mV against 5 mV.

Scope: regime class, dominant frequency and EEG level of the standard set under the declared hold and integrator. The paper's waveform, hold and integrator are not scored.
