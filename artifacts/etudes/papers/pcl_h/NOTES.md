# PCL in the H state (`pcl_stdp_h`)

Same Fig. 2 network, calibration, training and test as K0/K0b; the PCL rule
(`pcl_h_rule.py`, `pcl_stdp_h`) keeps the per-neuron traces in the H state
instead of per-edge aux. Izhikevich neurons only, through
`simulate_edge_recurrent_izhikevich_hdp_registered`. No kernel change: the
H carry already supports vector coordinates (`h_shape=(2,)`).

## Equations (per step, decay = exp(−dt/tau), tau uniform — refused otherwise)

- H[n] = (x[n], Y[n]): x <- x·decay + spikes[n] (presynaptic trace, never
  reset); Y <- Y·decay, set to 1 where n spikes (post trace).
- Per-edge aux: S[e] <- S[e]·decay, then set to x1[pre[e]] (post-update)
  where post[e] spikes; B[e] += pre_sp[e]·Y[post[e]]·decay (old Y).
- LTP of edge e: A[e] = x1[pre[e]] − S_decayed[e] (its own pre events since
  post[e]'s last spike, this step's included). At a post spike:
  dw = (w_max−w)·f·eta·A − w·f·eta·B, w ≥ 0, then group L1-normalization to
  lam where norm = 1; S/B reset.

## H/aux layout and why

| quantity | where | why there |
|---|---|---|
| presynaptic trace x | H[:,0] | per-neuron: neuron n's own spike history |
| postsynaptic trace Y | H[:,1] | per-neuron by definition (was copied per edge) |
| snapshot S | per-edge aux | per (pre,post) pair: x[pre] at post's last spike; reset times differ per post neuron |
| deferred LTD sum B | per-edge aux | per-edge: pre_sp[e] × post trace; no per-neuron sum reproduces every edge |

First version summed incoming pre spikes into LTP[post] (wrong with several
plastic inputs); the snapshot construction fixes it exactly. Non-uniform
per-edge tau raises instead of averaging.

## Results (K0b protocol, `results/fig2_h_avg.json`, seeds 10–12)

| seed | time-averaged w (mV) | suppression |
|---|---|---|
| 10 | 22.88 / 20.15 / 16.04 | 0.381 / 0.195 / 0.106 |
| 11 | 21.53 / 18.03 / 16.16 | 0.379 / 0.177 / 0.117 |
| 12 | 21.67 / 18.06 / 15.64 | 0.413 / 0.177 / 0.076 |
| mean | 22.03 / 18.75 / 15.95 | 0.391 / 0.183 / 0.100 |

Averaged weights ordered w1>w2>w3 in 3/3; suppression 0.39/0.18/0.10 as in
K0b. Nothing tuned: same defaults, same calibration (amp 29.51,
w_cancel 25.72).

## Equivalence

`tests/test_pcl_h.py`: Fig. 2 trajectory (allclose rtol=1e-5, atol=1e-5,
spikes equal); a 3-inputs-onto-one-neuron network (preconditions: post
spikes, input trains differ); a same-step pre+post tie unit check against
`pcl_stdp`; a non-uniform-tau refusal check. The 3-input test fails on the
summed-LTP version (weights 10.04/9.83/10.13 vs 7.36/7.99/14.65, 3/3
mismatch). On K0b the fixed rule gives bit-identical averaged weights and
suppression (`pcl/fig2_hdp_avg.json`); the end snapshot differs at ~1e-7.

## Deviations

None from the paper beyond the K0/K0b ones (see `../pcl/README.md`): same
update, only the state layout moved.
