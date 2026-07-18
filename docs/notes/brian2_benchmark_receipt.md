# Brian2 vs. jaxfne benchmark receipt (2026-07-18)

**The first real, quantitative cross-tool comparison run this session** —
prior benchmarking work (GPU/TPU/CPU accelerator comparisons) was entirely
jaxfne-vs-itself across backends, never against an incumbent tool. This one
is a small, real, honest smoke comparison — not a comprehensive benchmark
suite. Treat it as a first data point, not a definitive claim.

## Setup

Matched task: N Izhikevich neurons, sparse random connectivity targeting
~100 in-degree per neuron, 200ms duration at dt=0.5ms, CPU only, default
settings on both sides (no hand-tuning either tool).

- **jaxfne**: `scripts/benchmarks/jaxfne_izhikevich_comparison.py`, using
  `scripts/cortical_column_localized_workflow.py`'s `build_config` with
  `max_in_degree=100`. Runs in jaxfne's normal environment.
- **Brian2**: `scripts/benchmarks/brian2_izhikevich_comparison.py`, v2.10.1,
  dimensionless Izhikevich equations matching Brian2's own official example
  style, `p_connect = min(1, 100/n)` Erdos-Renyi connectivity. Run in an
  isolated throwaway venv (`pip install brian2`) — Brian2 is intentionally
  **not** a jaxfne dependency, this is a one-off comparison, not a feature.
  Code-generation backend: `auto`, which resolved to the compiled Cython
  path (confirmed available in the test environment) — the representative
  "out of the box" experience for a typical Brian2 user, not a
  maximally-tuned comparison (Brian2 also offers a `cpp_standalone` mode,
  not used here).

## Results

Two independent runs (fresh Brian2 venv both times), confirming the ratios
are reproducible, not a one-off fluke:

| N | Tool | Run | construct() | simulate() |
|---|---|---|---|---|
| 1,000 | Brian2 | 1 | 8.067s | 14.101s |
| 1,000 | Brian2 | 2 | 7.441s | 15.427s |
| 1,000 | jaxfne | 1 | 1.717s | 0.745s |
| 1,000 | jaxfne | 2 | 1.950s | 0.865s |
| 1,000 | **ratio (Brian2/jaxfne)** | 1 / 2 | **4.70x / 3.82x** | **18.93x / 17.83x** |
| 5,000 | Brian2 | 1 | 3.752s | 13.998s |
| 5,000 | Brian2 | 2 | 3.942s | 15.430s |
| 5,000 | jaxfne | 1 | 3.056s | 1.244s |
| 5,000 | jaxfne | 2 | 3.348s | 1.262s |
| 5,000 | **ratio (Brian2/jaxfne)** | 1 / 2 | **1.23x / 1.18x** | **11.25x / 12.23x** |

(edges/synapses and spike counts are identical run-to-run within each tool,
since both scripts use a fixed seed — 99,738 / 3,000 for Brian2 and 96,183 /
2,607 for jaxfne at N=1,000; 499,576 / 15,000 and 500,000 / 15,956 at N=5,000.)

## Honest caveats

- **Edge/synapse counts and spike counts differ slightly between tools at
  the same N** — expected, not a bug. Both target ~100 in-degree via a
  connection *probability*, so the realized edge count is a random draw
  (binomial), and each tool draws it from its own RNG stream (Brian2:
  `numpy.random.seed`; jaxfne: JAX PRNG) — the task specification matches,
  the specific random realization doesn't need to.
- **Small-scale smoke test, not a benchmark suite.** Only 2 sizes tested,
  only CPU, only this specific task shape (sparse Izhikevich, no plasticity).
  Does not generalize to claims about "jaxfne is faster than Brian2" in
  general — only that it was faster on this specific matched task at these
  two sizes, on this machine, with both tools' default settings.
- **Brian2's `construct()` time includes JIT/compile overhead** that Brian2
  itself would likely amortize across repeated runs at the same N within one
  process (code-generation is cached); each of the 2 runs recorded here used
  a fresh Brian2 venv/process, so both are cold-start measurements, not a
  warm-cache steady-state comparison for either tool.
- Neither tool's absolute wall-clock times here should be treated as
  hardware-independent — this ran on the same single machine for both
  sides, which is the correct way to compare, but the specific numbers
  won't transfer to other hardware.

## Reproducing

```bash
# jaxfne side (normal environment)
python3 scripts/benchmarks/jaxfne_izhikevich_comparison.py

# Brian2 side (isolated venv)
python3 -m venv /tmp/brian2_bench_venv
/tmp/brian2_bench_venv/bin/pip install brian2
/tmp/brian2_bench_venv/bin/python3 scripts/benchmarks/brian2_izhikevich_comparison.py
```
