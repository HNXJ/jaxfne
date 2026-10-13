# Izhikevich 2003 P3: diagnostics of the failed criteria (unscored)

Status: unscored, after the P3 v7 verdict. Hamm (2026-10-13) asked for modifications and other
ideas to find the small thing behind each failure. Nothing here changes the P3 v7 FAIL. A criterion
that changes becomes a new prereg version, P3 v8, with this note as its prior exposure. Scripts:
float64 forward Euler at dt 0.1 ms in the kernel's order. The reviewer of P3 found this reproduces
the kernel's spike rows.

## C8 (RZ): the assumed pulse amplitude

Paper: RZ cells "have damped or sustained subthreshold oscillations" (p. 1570). The pulse amplitude
is undrawn in the figure; P3 assumed 5 for 1 ms.

| Pulse | Spikes | Maxima after the pulse | Frequency |
|---|---|---|---|
| 5 × 1 ms (P3) | 1 | 8, first 15.9 mV (the spike) | 29.8 Hz |
| 3 × 1 ms | 1 | 8 | 29.3 Hz |
| 2 × 1 ms | 0 | 8, decreasing (−60.1, −62.2, −62.5 mV) | 25.2 Hz |

The resonator crosses threshold above a pulse of about 2–3 for 1 ms. Proposal for v8: 2 × 1 ms.

## C4 (FS): the ISI-ratio estimand

At I = 10 the FS ISIs are 4.2, 5.8, 7.2, then 7.6 ms to the end (133 Hz). The cell settles within
three spikes and stays periodic. Last/first is 1.81, so C4 fails at ≤ 1.2. The paper's FS fires
"practically without any adaptation (slowing down)" (p. 1570).

| Cell | ISIs (ms) | Last/first | Time to steady firing |
|---|---|---|---|
| RS | 17.8, 45.2, then 45.1 | 2.53 | 63 ms |
| FS | 4.2, 5.8, 7.2, 7.6, then 7.6 | 1.81 | 14 ms |
| LTS | 3.1, 3.8, 4.8, 6.8, 10.5, … 13.6 | 4.39 | none in 300 ms |

Time to steady firing is the time of the first spike from which every later ISI lies within 10 % of
the last ISI. Last/third separates nothing (RS 1.00, FS 1.06). Proposal for v8: C4 passes on rate
≥ 50 Hz and time to steady firing ≤ 10 % of the window (30 ms). It discriminates FS from RS and
LTS.

## N2: Poisson-like trains

The paper states that "neurons fire Poisson spike trains with mean firing rates around 8 Hz"
(p. 1571). On seed 1, the kernel and the float64 transcription of the paper's own `net.m` agree:

| Arm | ISI CV E / I | CV2 E / I | Fano (500 ms) E / I |
|---|---|---|---|
| kernel | 0.396 / 0.556 | 0.428 / 0.552 | 0.185 / 0.305 |
| `net.m` reference | 0.390 / 0.464 | 0.419 / 0.456 | 0.188 / 0.236 |

Single-cell trains under the published program are more regular than Poisson by every measure
tried. No port change can make N2 pass without changing the model. Paper rates corrected in S1
(7.05–7.23 Hz on the reference) do match "around 8 Hz". Reading the claim as population-level
asynchrony is a change of estimand. That choice is Hamm's: keep N2 as a failed paper claim, or
declare the population reading in v8.

## S1 scoring error found on the way

`cmd_s1` scored the 1 ms reference rows at 0.5 ms (`RESULT_P3.md`, corrected table). The paper's
own program gives 7.1 Hz, and its gamma peak passes N4 in every seed.
