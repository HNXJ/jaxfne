# Balance étude (stage 1): homeostatic inner loop + modulated three-factor learning

Exploratory probe. Standalone JAX (`balance.py`); it does not run the jaxfne
kernels, so nothing here is a claim about them.

## Question

Can a 100-neuron spiking network learn to keep an unstable object upright when
an inner homeostatic loop keeps its activity near fixed rates, and the only
learning signal is a scalar modulator "the error just got smaller"?

## Model

| part | definition |
|---|---|
| task | $d\theta = (a\theta + b u)\,dt + \sigma\,dW_t$, $a=1\,\mathrm{s^{-1}}$ (unstable), $\lvert\theta\rvert>1$ = failure, reset to $U(-0.3,0.3)$ |
| network | 20 sensory (Poisson, Gaussian tuning over $\theta$), 48 E, 12 I, 10 up-motor, 10 down-motor; LIF, $dt=1$ ms |
| action | $u = u_{max}\tanh((\bar r_{up}-\bar r_{down})/10\,\mathrm{Hz})$ |
| inner loop (RBD) | threshold $H_i$: $\tau_H\dot H_i = \rho_i/r^*_i - 1$; each neuron's plastic excitatory input sum held at its initial value |
| outer signal | $M = -\tfrac{d}{dt}\lvert\theta\rvert_{filt}$, failure $=-1$ pulse, centred by a 20 s running mean |
| learning | $\dot w_{ij} = \eta\,M\,e_{ij}$, eligibility $\tau_e\dot e_{ij} = -e_{ij} + \tau_e\,s_i\,x_j$ (post spike × pre trace) |

Plastic synapses: excitatory (sensory, E) onto E and motor neurons.

## Conditions (paired by network seed)

| condition | learning | modulator |
|---|---|---|
| full | on | own |
| frozen | off | — |
| shuffled | on | another agent's (independent of own actions) |
| wired | off | — (hand-wired correct sensory→motor map; positive control) |

Train 300 s, then test 60 s with learning off (inner loop stays on).

## Protocol (fixed before the confirmatory run)

- η is chosen on pilot seeds 0–7 only; confirmatory seeds 1000–1015 are run once.
- Primary metric: test fraction of time with $\lvert\theta\rvert<0.25$ ("in band").
- **P1** full > frozen in ≥13/16 paired seeds and mean difference ≥ 0.10.
- **P2** full > shuffled in ≥13/16 paired seeds and mean difference ≥ 0.10.
- **P3** activity held: every learning agent's test E rate within 0.5–2× its 5 Hz target.
- **P4** positive control: wired in-band ≥ 0.6.

**Frozen before the confirmatory run: η = 1.0.** Pilot (seeds 0–7, 300 s train,
model otherwise unchanged; test in-band, paired wins out of 8):

| η | full | frozen | shuffled | full−frozen | full−shuffled | test E Hz (full / shuffled) |
|---|---|---|---|---|---|---|
| 0.01 | 0.415 | 0.363 | 0.355 | 6/8, +0.052 | 5/8, +0.060 | 5.0 / 5.0 |
| 0.05 | 0.459 | 0.363 | 0.366 | 5/8, +0.096 | 4/8, +0.093 | 5.0 / 5.1 |
| 0.2 | 0.433 | 0.363 | 0.317 | 4/8, +0.070 | 5/8, +0.116 | 5.2 / 6.6 |
| 1.0 | 0.494 | 0.363 | 0.342 | 7/8, +0.130 | 8/8, +0.152 | 6.9 / 9.0 |

Wired control: 0.962 in band, 0 failures/min.

Secondary (reported, not gated): failures per minute, motor-map index
(> 0 when θ>0 sensors drive the down pool and θ<0 sensors the up pool).

## Result (confirmatory, seeds 1000–1015, η = 1.0): FAIL

| criterion | observed | verdict |
|---|---|---|
| P1 full > frozen | 11/16 wins, +0.114 | FAIL (wins) |
| P2 full > shuffled | 13/16 wins, +0.073 | FAIL (difference) |
| P3 E rate 2.5–10 Hz | 27/32 learning agents (range 4.8–16.1 Hz) | FAIL |
| P4 wired ≥ 0.6 | 0.961 | PASS |

Test in-band means: full 0.428, frozen 0.313, shuffled 0.355, wired 0.961.
Motor-map index: full +0.19, frozen +0.00, shuffled −0.26, wired +6.0.

![confirmatory figure](confirm_figure.png)

Reading (inferred, not tested):
- No sensory→motor map is learned: the map index stays near 0 against 6 for
  the wired control.
- The full agents' advantage is present in the first 10 s and fades during
  training. It behaves like fast online weight adaptation, not slow structure
  learning.
- With learning on, E rates swing 2–14 Hz: the outer rule and the inner
  homeostatic loop fight, and the inner loop does not hold activity (P3).

Candidate causes for a stage-1b protocol (each a new, separately frozen run):
modulator noise (−d|θ|/dt at 20 ms filtering), credit diluted over every
co-active synapse by the plain Hebbian eligibility, and timescale overlap
between τ_H = 5 s and the learning rate.

## Reproduce

```bash
python artifacts/etudes/hdp_balance/balance.py --eta <frozen eta> --seed0 1000 --n-seeds 16 --out artifacts/etudes/hdp_balance/confirm.json
```
