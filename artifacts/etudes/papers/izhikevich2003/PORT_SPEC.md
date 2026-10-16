# Izhikevich 2003: P2 port spec (second review PASS 2026-10-12)

Source facts: `SOURCE_P1.md`. Tier: T-jaxfne from the start, on the existing dense Izhikevich
kernel `jaxfne.emitters.simulate_eig_izhikevich` (Hamm 2026-10-12: the existing emitter). The kernel
is the scored arm; a NumPy transcription of the paper's program is the reference arm
(sensitivity S1).

## Fig. 2 (single cells)

One cell per type, parameters from the `SOURCE_P1.md` table, `IzhikevichParams` built
directly (per-cell a, b, c, d arrays). Kernel `dt_ms = 0.1` (the figure's resolution), forward
Euler, noise off (`noise_scale = 0`), `W = 0`. Initial state: each cell's resting fixed point at
$I = 0$ (P3 draft v5; earlier drafts assumed $v = -65$, −70 for TC and RZ). Input: `drive_schedule` holds $I = 0$ for 10 ms (assumed: the figure shows a lead-in
without a value), then a step to 10 (RS, IB, CH, FS, LTS). TC: two runs, a positive step from rest and a negative step that holds $v$ near −90 mV
before release (amplitudes assumed, declared in P3). RZ: a brief pulse (assumed).

## Fig. 3 (network)

| Paper (`net.m`) | Kernel mapping | Status |
|---|---|---|
| 800 E, 200 I, per-cell $r \sim U[0,1]$ parameters | `IzhikevichParams` arrays, $r$ from `jax.random` with the run seed | exact form, our generator |
| $S$: $0.5\,U[0,1]$ from E, $-U[0,1]$ from I, all to all, self included | dense `W[post, pre]` | exact form |
| spike adds $S$ to $I$ for the next 1 ms (both half-steps) | kernel adds `W @ prev_spikes` once per step | at `dt_ms = 0.5` the kernel's weight must be doubled to deliver the same $\Delta v$ per spike: declared |
| thalamic $I$: $5N(0,1)$ E, $2N(0,1)$ I, one draw per ms | `noise_scale` per cell, one draw per kernel step | at `dt_ms = 0.5`: scale by $\sqrt{2}$ so that the $v$ increment per ms has the paper's variance; the draws are independent across the two half-steps (paper: shared): declared deviation |
| $v$ two 0.5 ms Euler half-steps, $u$ one 1 ms step on the $v$ after both half-steps | $v$ and $u$ both 0.5 ms Euler steps, $u$ on the $v$ before the step | declared deviation; S1 measures it |
| when the first half-step crosses 30, the second overshoots (by up to hundreds of mV) and that $v$ enters the $u$ step: a spike-dependent extra $u$ increment before the $+d$ reset | no overshoot: the threshold is tested after each 0.5 ms step and $v$ resets at once | declared deviation (derived); S1 reports the extra $u$ increment per spike and its effect on rate and ISI CV; if it moves a scored statistic outside its band, P3 adds a kernel variant arm |
| spike test at the start of step $t$ on the $v$ left by step $t-1$; reset, then $S$ delivered during $t$ | test after step $k$; reset at once; `W` delivered during step $k+1$ | same order (derived); the test runs every 0.5 ms instead of every 1 ms, so a crossing is caught half a step earlier |
| no baseline current, no other inputs | `drive = 0`; signs carried by `W` (the kernel ignores `sign`); `silence_mask` unset; `source_scale` affects only the `sources` output, which is not scored | exact |
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
  seeds 1–5, float32, a float64 run on one seed. `dtype="float64"` falls back to float32 unless
  `jax_enable_x64` is on (`jaxfne/emitters.py` `_dtype_from_policy`): the float64 run asserts the
  output dtype.
- Library route: whether the scored run goes through `Configuration`/`construct` (B1) or the
  ADVANCED kernel call; the route has no per-cell heterogeneous a, b, c, d today (to verify).

## Finding outside the étude

`jaxfne/presets.py` cited "Izhikevich 2003 Table 1" for its four presets. The paper has no
table; RS, FS and LTS match Fig. 2, and `VIP_IS` does not appear in the paper. The `source`
strings now cite Fig. 2 and mark `VIP_IS` as outside the paper (`tests/test_presets_citation.py`).
