# Izhikevich 2003: P2 port spec (draft for second review)

Source facts: `SOURCE_P1.md`. Tier: T-jaxfne from the start, on the existing dense Izhikevich
kernel `jaxfne.emitters.simulate_eig_izhikevich` (Hamm 2026-10-12: the existing emitter). A
NumPy transcription of the paper's program is the reference arm (sensitivity S1), never the
scored arm.

## Fig. 2 (single cells)

One cell per type, parameters from the `SOURCE_P1.md` table, `IzhikevichParams` built
directly (per-cell a, b, c, d arrays). Kernel `dt_ms = 0.1` (the figure's resolution), forward
Euler, noise off (`noise_scale = 0`), `W = 0`. Initial $v = -65$ (−70 for TC and RZ, assumed),
$u = bv$. Input: `drive_schedule` holds $I = 0$ for 10 ms, then a step to 10 (RS, IB, CH, FS,
LTS). TC: two runs, a positive step from rest and a negative step that holds $v$ near −90 mV
before release (amplitudes assumed, declared in P3). RZ: a brief pulse (assumed).

## Fig. 3 (network)

| Paper (`net.m`) | Kernel mapping | Status |
|---|---|---|
| 800 E, 200 I, per-cell $r \sim U[0,1]$ parameters | `IzhikevichParams` arrays, $r$ from `jax.random` with the run seed | exact form, our generator |
| $S$: $0.5\,U[0,1]$ from E, $-U[0,1]$ from I, all to all, self included | dense `W[post, pre]` | exact form |
| spike adds $S$ to $I$ for the next 1 ms (both half-steps) | kernel adds `W @ prev_spikes` once per step | at `dt_ms = 0.5` the kernel's weight must be doubled to deliver the same $\Delta v$ per spike: declared |
| thalamic $I$: $5N(0,1)$ E, $2N(0,1)$ I, one draw per ms | `noise_scale` per cell, one draw per kernel step | at `dt_ms = 0.5`: scale by $\sqrt{2}$ so that the $v$ increment per ms has the paper's variance; the draws are independent across the two half-steps (paper: shared): declared deviation |
| $v$ two 0.5 ms Euler half-steps, $u$ one 1 ms step | $v$ and $u$ both 0.5 ms Euler steps | declared deviation; S1 measures it |
| spike test at the start of the step, $v \ge 30$ | test after the update, $v \ge 30$ | one-step shift, declared |
| initial $v = -65$, $u = bv$ | `v0`, `u0` | exact |
| 1000 ms | longer runs for spectra (P3 decides), first 200 ms discarded | ours |

The weight and noise scalings make the per-ms input statistics match; they are the only
parameter changes, and P3 states them before any run.

## Reference arm (S1)

`artifacts/etudes/izhikevich2003/net_reference.py`: a float64 NumPy transcription of the
paper's update order (spike test, reset, input, two $v$ half-steps, one $u$ step at 1 ms),
seeded per run. Scored statistics are computed on both arms; S1 reports the differences.

## Open for P3 (criteria are ours; the paper is qualitative)

- Fig. 2: an operational test per type (e.g. RS adaptation as last/first ISI ratio; IB a
  first burst then single spikes; CH bursts with inter-burst frequency; FS ISI coefficient of
  variation and rate; LTS adaptation with higher rate than RS; TC rebound burst; RZ damped
  oscillation frequency).
- Fig. 3: mean rate "around 8 Hz" (a band), asynchronous firing (ISI CV near 1), and
  population-rate spectral power excess near 10 Hz and 40 Hz against a shuffled-spike null;
  seeds 1–5, float32, a float64 run on one seed.
- Library route: whether the scored run goes through `Configuration`/`construct` (B1) or the
  ADVANCED kernel call; the route has no per-cell heterogeneous a, b, c, d today (to verify).

## Finding outside the étude

`jaxfne/presets.py` cites "Izhikevich 2003 Table 1" for its four presets. The paper has no
table; RS, FS and LTS match Fig. 2, and `VIP_IS` does not appear in the paper. The citation
needs correcting (todo).
