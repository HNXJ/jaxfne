# jaxfne scaling benchmark (smoke)

Harness: `bench_scaling.py`. Smoke scale only; not the scored benchmark, which needs a signed protocol.

## What it measures

- Laminar column via the public path: `build_laminar_column`, `construct`, `simulate`.
- compile_s: first `simulate`, seed 0 (compile plus run). run_s: median of run-only calls on seeds 1..k, none used before.
- Seed check: consecutive timed runs must give different spikes (`seed_changes_output`).
- jit: effective decision per call shape (`RuntimeConfig.resolve_jit`).
- Ensemble: `Model.simulate_batch`; compile on base seed 0, timed on base seed 1000; spikes must differ.

## How to run

    python artifacts/programme/benchmarks/bench_scaling.py --out <path.json>
    python artifacts/programme/benchmarks/bench_scaling.py --inventory

## Smoke result (smoke, CPU, not a result)

Source: `C:/workspace/pytest_temp/l3/bench.json` (cpu:0, jax 0.10.1, float32, 200 ms, dt 0.5, 400 steps).

| N | jit single | compile s | run s | neuron-steps/s | jit batch | ensemble run s | ensemble neuron-steps/s | seed changes (run, ensemble) |
|---|---|---|---|---|---|---|---|---|
| 100 | no | 2.644 | 0.1805 | 2.216e5 | yes | 0.00633 | 2.529e7 | yes, yes |
| 300 | yes | 1.452 | 0.00882 | 1.360e7 | yes | 0.02217 | 2.165e7 | yes, yes |
| 1000 | yes | 1.271 | 0.07616 | 5.252e6 | yes | 0.4818 | 3.321e6 | yes, yes |

Inventory: brian2 and nest do not import; jaxley imports (0.14.0).

## Anomalies

- Explained: N=100 single runs are eager (400 x 100 = 40000 is below the 50000 auto threshold, `jaxfne/_runtime_config.py:158-165`). A probe with forced jit gave run 0.0051 s at N=100, log `C:/workspace/pytest_temp/l3/jit_probe.log`. The 4-seed batch is jitted, so it is faster than one eager run.
- Not a result cache: `_model_simulate.py` caches compiled functions only (lines 428-466, 869-908, 2729-2772), keyed on shape and config, with the PRNG key passed as an argument. `_pipeline.py` has no cache. Seed changes the spikes.
- Recompile-guard warnings appear when sizes change within one process.
- A UserWarning reports float64 requested while x64 is off. Observed; not investigated.

## Open protocol choices for Hamm

| Choice | Smoke value | Options |
|---|---|---|
| Sizes | 100, 300, 1000 | full grid, maximum N |
| Durations, dt | 200 ms, 0.5 ms | scored duration, dt sensitivity |
| Hardware | local CPU | local GPU, Colab GPU or TPU, both |
| jit | auto (threshold) | force on or off per size |
| Baselines | none run | install brian2 and NEST, include jaxley |
| Models | laminar column | Brunel 2000, Izhikevich 2003, or both |
| Repeats | 2 run-only seeds | number of trials and statistics |
| Process | one process for all sizes | one process per size |
