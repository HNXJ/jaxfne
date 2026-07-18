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

| N | Tool | construct() | simulate() | edges/synapses | spikes |
|---|---|---|---|---|---|
| 1,000 | Brian2 | 8.067s | 14.101s | 99,738 | 3,000 |
| 1,000 | jaxfne | 1.717s | 0.745s | 96,183 | 2,607 |
| 1,000 | **ratio (Brian2/jaxfne)** | **4.70x** | **18.93x** | | |
| 5,000 | Brian2 | 3.752s | 13.998s | 499,576 | 15,000 |
| 5,000 | jaxfne | 3.056s | 1.244s | 500,000 | 15,956 |
| 5,000 | **ratio (Brian2/jaxfne)** | **1.23x** | **11.25x** | | |

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
  itself would likely amortize across repeated runs at the same N (code-
  generation is cached); this receipt is a single cold run per size, not a
  warm-cache steady-state measurement for either tool.
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
