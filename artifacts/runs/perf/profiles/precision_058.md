# Precision profile: Izhikevich E/PV circuit, float64, float32, bfloat16 (2026-10-27)

Script `scripts/perf/precision_profile.py`; numbers in `precision_058.json` (same folder). Circuit: dense
Izhikevich E/PV (0.8/0.2, preset `cortical_eig`), dt 0.1 ms, seed 0. Loss: the soft-rate MSE on `V_m` that
`tune_handles` differentiates; parameter: per-neuron `a` (`cells.izhikevich.a[all]`). One process per cell;
timing is the median of 5 calls after one warm call, compile excluded, each call ended by
`block_until_ready`. GPU: RTX A4000 16 GB under WSL2, jax 0.10.1 CUDA 12; CPU: i9-10920X under WSL2.

## Results

| Question | Observation |
|---|---|
| Per-step time, GPU, N 100 to 1600 | forward 38 to 46 µs, gradient 88 to 122 µs; flat in N and dtype; linear in steps (250 to 4000) |
| Per-step time, GPU, N 3200 | forward 114 µs (float32, bfloat16), 226 µs (float64); gradient 143 to 146 µs, 264 µs (float64) |
| Small-N bottleneck | nsys, float32 gradient N 100: 8.5 kernels and 2 device-to-device copies per step, each about 1.3 µs on the GPU; `cuLaunchKernel` 5.0 µs median host time. GPU busy about 13 µs of the 110 µs step: host launch-bound |
| Large-N bottleneck | float64/float32 time ratio 1.98 at N 3200; one dense W read per step at 448 GB/s predicts 183 and 91 µs, measured minus the 40 µs floor 186 and 74 µs (bandwidth-bound, inferred) |
| XLA command buffers (WHILE included) | forward 40 to 41 µs/step unchanged; gradient 99 to 87 µs/step (N 100) |
| CPU against GPU, float32 | forward: CPU faster up to N 400 (13 against 45 ms per 1000 steps), GPU faster from N 800 (43 against 84 ms); gradient crossover near N 800; at N 3200 GPU 34x (forward) and 28x (gradient) |
| Batching (`vmap`, gradient, N 100) | GPU step time 90 to 105 µs for B 1 to 256, so 0.35 µs per trial-step at B 256; CPU best 2.1 µs per trial-step at B 16 |
| Gradient memory | compiled temporaries O(N T), about 6 values per neuron-step (float64 N 3200, T 1000: 150 MB; float32: 76 MB). Process peak (512 MB float32, 864 MB float64 at N 3200) is equal for forward and gradient: set by construction-time N x N arrays |
| bfloat16 | `V_m` is stored as bfloat16, while the recurrent dot runs in float32 (`f32[3200,1] dot` in the compiled module); time and spike counts equal float32's |
| Accuracy against float64 | spike totals within 2 of float64 at every N; loss relative error 5e-4 to 5e-2 (chaotic divergence of spike timing); float32 and bfloat16 gradients have cosine 0.46 to 0.99 with the float64 gradient |

## Surrogate gradient (P-052, sigmoid_beta)

Same grid, gradient cells, with `RuntimeConfig(surrogate=SurrogateConfig(method="sigmoid_beta", beta=10))`
(`precision_profile.py --surrogate-beta 10`); receipts under `gpu_surrogate_sigmoid_beta_10` and
`recurrent_grad_check` in the JSON.

| Question | Observation |
|---|---|
| Does the gradient cross neurons? | `recurrent_grad_check.py` (N 20, 2000 steps, float64, CPU): hard spike, off-diagonal 0 of 380 and no Wᵀ in the backward graph; sigmoid_beta at beta 10, 1 and 0.1 per mV, 323, 361 and 380 nonzero, Wᵀ present |
| Does it match the finite difference? | No. Central differences on `a[0]` agree across h 1e-3 and 1e-4 (corr 0.94; 0.12 against h 1e-2); the surrogate column for the 19 other neurons has corr -0.38, -0.23 and -0.22 with it at beta 10, 1 and 0.1, sign agreement 7, 3 and 3 of 19 |
| Per-step time, GPU, N 100 to 1600 | unchanged within run-to-run noise (ratio 0.69 to 1.35; a same-session rerun at float32 N 100 reads 102 µs hard, 83 µs surrogate) |
| Per-step time, GPU, N 3200 | float32 146 to 232 µs, bfloat16 143 to 232 µs, float64 264 to 511 µs: one more W read per step (Wᵀ), 91 µs predicted at float32 and 183 µs at float64 |
| Gradient memory | compiled temporaries +16% (float32 N 3200: 76 to 89 MB; float64: 150 to 174 MB), about 7 values per neuron-step |
| Gradient norm | equal to the hard spike's at 250 and 500 steps; 10 to 1000 times larger from 1000 steps on (float32 N 100: 8.5e5 to 1.0e7), growing with T: backpropagation through the recurrence amplifies |

The process peak of this grid (32 MB at N 100) is lower than the first grid's (64 MB); a same-session
rerun reads 33.7 MB for both the hard spike and the surrogate, so the difference is between the two
sessions and its cause is unknown.

## Limits

- The surrogate is realized on the dense Izhikevich default-order route only; every other route and
  `straight_through` refuse it. The reset stays detached. Its slope is in 1/mV; beta 10 (the
  `SurrogateConfig` default) was not chosen for this neuron model.
- `set_parameter` casts the parameter to its stored float32, so float64 gradients pass a float32 cast.
- `cost_analysis` omits the dense matvec (FLOPs linear in N), so FLOP counts come from the trace.
- Batched GPU results differ from unbatched ones beyond rtol 1e-5 for B 4 to 64; the size of the
  difference was not recorded.
- WSL2 adds host overhead to each launch (inferred); a native Linux run is not measured.
