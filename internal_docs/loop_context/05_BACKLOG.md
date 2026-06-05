# 05_BACKLOG

Ranked by value-per-risk. Live audit identified PRNG null reproducibility as the strongest immediate patch (`Pasted markdown.md:L95`), while v0.3.30 architecture remains a plan-gated RED class based on the assessment plan.

## Status (live reconciliation @ fab4c9c)

| id | status | note |
|---|---|---|
| B01 | ✅ MERGED | PR #22 merged into `main` (merge commit `33f99db`); objective null RNG reproducibility live. Realized PRNG 62→85. (Related: fields-helper dedup shipped separately as PR #23, merged `e29c604`.) |
| B02 | ▶ READY (next) | Leading GREEN item. Live root cause confirmed: zsh `nomatch` on bare `*.egg-info` aborts the cleanup line, leaving stale 0.3.27 wheels. The find-based cleanup is already encoded in `06_VALIDATION_LADDER.md`. |
| B03 | ⏸ DEFERRED | `sys` leak sits adjacent to the fragile runtime-wrapper (`__init__.py:359,396,400`); deliberately deferred — cosmetic `dir()` leak only, not in `__all__`. |
| B04, B05, B06, B10 | ▶ READY (YELLOW) | Do-then-hold after local verification. |
| B07, B08, B09 | ⛔ GATED | Need a human design decision (plan-before-phase). |

|id|title|classification|tier|exact files|minimal change concept|tests to add|validation commands|stop rules|expected score impact|dependencies|
|---|---|---|---|---|---|---|---|---|---|---|
|B01|Objective null RNG reproducibility|PATCH|GREEN autonomous|jaxfne/objectives.py; tests/test_objective_null_reproducibility_v0330.py|Add keyword-only rng/seed; thread generator through dispatcher/factory; preserve default behavior|same seed=same null, different seed differs, factory forwards null_seed, legacy calls pass|compileall; focused objective tests; full pytest; notebook audit|Stop on shape/key/range change or broken legacy calls|PRNG 62->85; overall +2|none|
|B02|Release clean script hardening|PATCH|GREEN autonomous|release scripts / docs release command block|Replace bare *.egg-info glob with find/nullglob-safe cleanup|dist contains exactly current version artifacts after build|build; twine check; dist sanity|Stop if deletes non-build files|dist hygiene +2|none|
|B03|sys namespace leak cleanup|PATCH|YELLOW do-then-hold|jaxfne/__init__.py|Use local import or del sys after wrapper swap if verified safe|import jaxfne; sys not in dir(jtfne); runtime wrapper still works|api smoke; import smoke; full tests if wrapper touched|Stop on wrapper/import regression|API hygiene +1|none|
|B04|LIF/GLIF export honesty|FOLLOW-UP|YELLOW do-then-hold|jaxfne/emitters.py; jaxfne/__init__.py; docs/API|Keep names but improve loud error/status docs, or formally deprecate before removal|tests assert loud NotImplementedError and docs mark planned|api smoke; docs build|Stop on public API break|API hygiene +2|human decision if removal|
|B05|JIT recompilation guard fixes|REPRODUCE|YELLOW do-then-hold|jaxfne/validation.py and runtime call sites|Reproduce N_compile=2 paths, stabilize static args/signatures|N_compile<=1 where expected; numerical traces equal|focused JIT tests; trace diff|Stop on numerical output change|JAX runtime +2|B01 independent|
|B06|Etude long-cell modularization|FOLLOW-UP|YELLOW do-then-hold|tutorials/etudes/*.ipynb; jaxfne/tutorial_utils.py|Move reusable notebook glue into package helpers only if execution-equivalent|notebook executes; figures/hash/artifacts unchanged|notebook audit; nbconvert execution|Stop if scientific output/artifact names change|Notebook +3|none|
|B07|Connectivity compiler contract|PLAN|RED gated|jaxfne/connectivity.py or builders/contracts; tests/test_connectivity_compiler_v0330.py|Define sparse deterministic edge-array compiler with artifact_ref SHA256 validation|deterministic sparse output, no dense O(N^2) except explicit matrix mode|compileall; focused connectivity tests; full tests|Stop on invented API or dense default|Architecture +5-8|human design|
|B08|FlatNet/PyTree transform boundary|PLAN|RED gated|jaxfne/net.py or experimental_hpc/contracts.py; tests/test_flatnet_v0330.py|Freeze Config->Net->FlatNet, TrackingMaps, PyTree roundtrip, simulate_flat smoke|tree flatten/unflatten roundtrip; JIT/vmap smoke|focused FlatNet tests; full tests|Stop if Python objects enter traced kernels|JAX/PyTree +5-8|B07 maybe|
|B09|Optional PyNWB export bridge|PLAN|RED gated|jaxfne/io_nwb.py; tests/test_pynwb_export_v0330.py|Lazy optional PyNWB writer with units/status/provenance and read/write roundtrip|core import without pynwb; proxy signals not calibrated units|optional-dep tests; import smoke|Stop if PyNWB required on core import|Ecosystem +5|human design|
|B10|Benchmark/profiling receipts|FOLLOW-UP|YELLOW do-then-hold|benchmarks/ or scripts/bench_*.py; docs/performance.md|Add CPU baseline for neuron count/contacts/seeds with fixed SHA and hardware|benchmark JSON/markdown receipt|run benchmark script; hash output|Stop on non-reproducible benchmark|performance +5|none|


## Ready now

- GREEN: **B02** (B01 shipped — see status table).
- YELLOW: B04-B06, B10 after local verification. (B03 deferred — runtime-wrapper adjacency.)

## Human design decision required

- RED/PLAN: B07 connectivity compiler, B08 FlatNet/PyTree boundary, B09 PyNWB export. Assessment plan requires deterministic sparse connectivity, FlatNet tracking maps, optional PyNWB lazy import and round-trip.
