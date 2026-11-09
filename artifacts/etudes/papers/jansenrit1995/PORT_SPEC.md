# Jansen-Rit 1995 port spec (jaxfne core)

Source values: `SOURCE_P1.md`. Engine: the core emitter `jaxfne/emitters_jansen_rit.py`
(`simulate_jansen_rit`, line 291), driven by `run_jr_fig3.py`. No parameter is fitted; the standard set of
`SOURCE_P1.md` §2 is passed as stated. Written after P3 v1 scored, as the P2 record of the mapping the P3 run used
(`prereg/P3_v1_signed.md`).

## Stage order

1. **S1, single column, Fig. 3** (P3 v1, scored): regime class, dominant frequency and EEG level over
   C = 68, 128, 135, 270, 675, 1350.
2. **S2, parameter sweeps** (A, B, v0; `SOURCE_P1.md` §3), after S1.
3. **S3, double column, Figs. 4-5** (K1, K2, delay block a_d), after S2.

## Mapping to jaxfne objects

| Paper object | jaxfne object |
|---|---|
| Sigmoid, Eq. 3 | `jansen_rit_sigmoid` (`emitters_jansen_rit.py:216`) |
| PSP blocks and six ODEs, Eqs. 4-6 | `jansen_rit_rhs` (`emitters_jansen_rit.py:226`), states y0..y5 |
| C1..C4 = C, 0.8C, 0.25C, 0.25C (Eq. 16) | `run_c` in `run_jr_fig3.py:78` passes the four products |
| p(t) uniform 120-320 pulses/s | `drive` in `run_jr_fig3.py:37`: one seeded draw held per `hold_s` |
| EEG proxy y1 - y2 | the emitter's output difference, read by `spectrum_metrics` (`run_jr_fig3.py:45`) |
| RKF45 integration | fixed-step classical RK4, input held per step (`_rk4_matrix_step`, `emitters_jansen_rit.py:261`) |

## Fidelity notes (port-vs-paper, not fits)

- F1: noise hold 2 ms (assumed from Fig. 3's sample interval; the paper does not state it). Owner decision 2 of
  the P3 v1 prereg; a 0.1 ms hold pilot gave the same six classes.
- F2: RK4 at dt = 0.1 ms in place of RKF45 with unstated tolerances (owner decision 2).
- F3: run length 10 s with 1 s discarded; the paper shows 2 s and states no transient handling.
- F4: the waveform is not scored; P3 scores regime class, dominant frequency and EEG level only.
