# B1 benchmark prereg (0.5.10, F4): SIGNED

Status: signed by Hamm 2026-11-06 ("Sign it"), every choice ruled the same day. Frozen: a change after this
line needs a new dated amendment, and a scored cell names the commit of this file it ran under.

## Question and estimands

Q: how fast does jaxfne simulate spiking networks against N, on CPU and GPU, and against Brian2, NEST and
Jaxley on the model classes each supports?

- E1 throughput: neuron-steps/s = N x n_steps / run_s, run_s = wall time of a compiled run (block_until_ready),
  compile excluded.
- E2 compile_s: first call minus run_s, per shape.
- E3 ensemble throughput: neuron-steps/s of `simulate_batch` (vmap over seeds) at batch B.
- E4 precision cost: E1(float64) / E1(float32).
- Validity check per cell (not an estimand): the run produces the model's declared regime. Mean E rate within
  20% of the reference tool's rate on the same seed set, and spikes > 0. A cell that fails validity is reported
  and excluded from the ratios.

## Models (matched classes)

| Model | jaxfne route | Baselines | Why |
|---|---|---|---|
| M1 Brunel 2000 sparse LIF, regime SI | LIF | Brian2, NEST | native LIF in all three |
| M2 Izhikevich 2003 network | Izhikevich | Brian2, NEST | Brian2 equation-level; NEST's built-in `izhikevich` model (Hamm 2026-11-06) |
| M3 single HH multicompartment cell | HH | Jaxley | Jaxley's model class |

Connectivity statistics, drives and equations are taken from each lane's frozen config
(`brunel2000`, `izhikevich2003` lanes); the baselines are written to the same equations, and each baseline's
script records its equation text.

## Grid

- N (Hamm 2026-11-06): 1e2, 1e3, 1e4, 1e5 for M1 and M2. M3: 1, 10, 100 cells (Hamm 2026-11-06).
- dt 0.1 ms, 1000 ms simulated (10 000 steps) per run (Hamm 2026-11-06).
- float32 and float64 (jaxfne); baselines in their default precision, recorded.
- Ensemble B: 1, 8, 32 (jaxfne only).
- Hardware (Hamm 2026-11-06): local CPU (one process per cell, BLAS threads pinned and recorded) and the local
  NVIDIA RTX A4000 (16 GB) via WSL CUDA; nothing else runs on the GPU during a timed cell (`nvidia-smi` read
  before and after, recorded).

## Procedure

Fresh process per cell. One warm-up run, then 5 timed runs on seeds 1-5 (each seed used once). Report the
median and the interquartile range of run_s. RSS peak sampled. Each cell records the SHA, tool versions,
device, dtype, thread counts and the seed list. Harness pattern: `scripts/benchmark_055_matrix.py`.

### Required metrics & controls (Hamm 2026-11-06)

| Metric | Purpose |
|---|---|
| Compilation time | CPU/JAX compilation overhead |
| Steady-state simulation time | GPU throughput |
| Peak GPU memory | Capacity requirement (distinguish reserved vs live/peak via memory_stats) |
| Peak host RSS | Construction and recording overhead |
| Host-device transfer time | PCIe sensitivity |
| Numerical agreement | CPU/GPU semantic and numerical validity |

Controls for memory: negative control with recording disabled and another with reduced recording
to distinguish simulation-state memory from trajectory storage.

### Hardware purchase gate

- Peak VRAM approaches 16 GB despite appropriate chunking: investigate 32–48 GB GPUs.
- VRAM fits but GPU execution dominates: benchmark a faster NVIDIA GPU before purchasing.
- CPU construction or compilation dominates: investigate CPU/platform changes.
- No material bottleneck: retain existing workstation (A4000 + 192 GB RAM). Recommendation: $0 hardware spend now.

## Outputs and receipts

`artifacts/runs/benchmark/b2_0510/` (receipt JSON per cell, hash-pinned in a manifest); F4 renders from it.

## Decision rules (fixed before any scored run)

- No claim of "faster" without the validity check passing on both sides of the ratio.
- A ratio is reported with its IQR-propagated range; no single best-of-n numbers.
- A baseline that cannot run a model class is recorded as a class mismatch (B3), not as a loss.
- The smoke numbers in `artifacts/programme/benchmarks/README.md` and the 2026-07-18 Brian2 receipt are pilots
  and are not reused.
