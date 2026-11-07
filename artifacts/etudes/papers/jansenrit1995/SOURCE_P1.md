# Jansen and Rit 1995: P1 source record

Target: Jansen BH, Rit VG (1995), Electroencephalogram and visual evoked potential generation in a mathematical model
of coupled cortical columns, Biol Cybern 73(4):357-366, doi:10.1007/BF00199471. Read 2026-11-07 from the publisher
PDF (institutional access; local copy `F:/warehouse/jansen_rit1995/jansen_rit_1995.pdf`, not in the repo).

Pointers are journal pages and equation numbers. Status: observed (the paper states it), inferred (computed from
stated values), unknown (the paper does not give it). No passage is quoted.

## 1. Single-column model (§2.1, pp. 358-359)

- Sigmoid, Eq. 3 (observed, p. 358): `Sigm(v) = 2 e0 / (1 + exp(r (v0 - v)))`. Equal to
  `jaxfne/emitters_jansen_rit.py` `S(v)` (line 222).
- PSP block, Eq. 4-5 (observed, p. 358): `y'' = A a x - 2 a y' - a^2 y` (excitatory; inhibitory with B, b).
- Six ODEs, Eq. 6 (observed, p. 358): pyramidal `y0` driven by `Sigm(y1 - y2)`; excitatory interneuron branch `y1`
  driven by `p(t) + C2 Sigm(C1 y0)`; inhibitory branch `y2` driven by `C4 Sigm(C3 y0)`. EEG proxy `y1 - y2`.
- Integration (observed, p. 358): Runge-Kutta-Fehlberg 4(5). Step size and tolerances: unknown.

## 2. Parameters (observed, §2.3, p. 360; standard set Eq. 17, p. 360)

| Symbol | Value | Unit | Pointer |
|---|---|---|---|
| A | 3.25 | mV | p. 360, Eq. 17 |
| B | 22 | mV | p. 360, Eq. 17 |
| a | 100 | s^-1 | p. 360 (from Jansen et al. 1993) |
| b | 50 | s^-1 | p. 360 |
| a_d (intercolumn delay block, double column only) | 30 | s^-1 | p. 360 |
| C | 135 | dimensionless | p. 360, Eq. 17 |
| C1, C2, C3, C4 | C, 0.8C, 0.25C, 0.25C | dimensionless | p. 360, Eq. 16 (assumes C1 = 4 C3) |
| v0 | 6 | mV | p. 360, Eq. 17 |
| e0 | 2.5 | s^-1 | p. 360 |
| r | 0.56 | mV^-1 | p. 360 |
| p(t) | uniform random, 120 to 320 | pulses/s | p. 360; §3.1 p. 360 |

Inferred: at C = 135, C1 = 135, C2 = 108, C3 = C4 = 33.75.

## 3. P1 target (observed, §3.1, pp. 360-361)

Fig. 3: two seconds of `y1 - y2` for C = 68, 128, 135, 270, 675, 1350 under the uniform 120-320 input. Stated
outcome: noise, noisy alpha, well-defined alpha at C = 135, then spike-wave-like quasi-periodic output whose
frequency falls to zero, then noise-like. Also stated (p. 361): the same sequence when A, B or v0 alone rises from
low to high; sweep ranges A 2.6-9.75 mV, B 17.6-110 mV, C 108-675, v0 in {3.12, 5.52, 6.0} mV.

Fig. 3's time scale bar reads 500 units = 1 second (observed, p. 361), so the plotted output is sampled every
2 ms (inferred); whether the input was drawn at that interval is unknown.

Unknown before a P3 prereg: the noise sample interval (white noise held per step), the integration step, run
length beyond the 2 s shown, and any transient discarded. A P3 criterion must therefore score regime class and
dominant frequency, not waveform.

## 4. Double-column model (§2.2, p. 359; §3.2-3.3)

Coupling constants K1, K2 through delay blocks (A, a_d); Fig. 4-5 coupling sweeps. Out of P1 scope until the single
column is scored.
