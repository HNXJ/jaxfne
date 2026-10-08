# Model contract: F1–F10 against existing surfaces (post-0.5.5)

Hamm 2026-10-08. Scope: make existing models trainable, transformable,
portable and efficient. No new package, no third model object, no second
runtime. "LNBM" is a model-family name only; no public use until scaling
evidence exists. Starts after the 0.5.5 seal.

Contract over existing objects:

    TFNE/JDNA -> NeuronalTensor --construct--> Model
    (h_next, y) = F_Model(h, x; p, s)
    (M', h', R) = T(M, h, pi)

p model-owned parameters, s fixed execution structure, h complete
continuation state, pi transformation policy, R its record.

## Gap table

Basis: grep-level inspection of `dev` at 725ff95d (2026-10-08), not the
full audit; step 1 below replaces it. Status: Existing / Partial / Missing /
Unknown.

| feature | existing surface | status | gap |
|---|---|---|---|
| F1 model object | `NeuronalTensor`, `Model` (`jaxfne/_model.py`), `_model_manifest.py` | Existing | none found |
| F2 builder | `construct` (`jaxfne/_construct_core.py`), JDNA development | Existing | none found |
| F3 run, continue, checkpoint | `simulate`, `ContinuationState` with C_t = (X, H, W, B, K, A) (`artifacts/programme/continuation_053.md`); 5 test files use it | Partial | one documented state schema for saving and restoring h outside a session |
| F4 freeze and train | `trainable` flags (`_model_tune.py`, `optim/core.py`); 1 test file mentions `trainable` | Partial | no test that frozen values stay fixed under optimizer momentum or weight decay (no test mentions either); training and HDP on the same coordinate has no declared policy |
| F5 augment | `augment(tensor, spec) -> (new, AugmentationRecord)` (`jaxfne/augment.py`) | Existing | record pattern not yet shared with other transforms |
| F6 compress, reduce | none in the package (no prune, low-rank, coarse-grain or distill code); Atlas reductions are scientific comparisons | Missing | transformation contract and record for reductions |
| F7 portable artifact | `save_tensor`/`load_tensor` (`_pipeline.py`), manifest with schema version | Partial | runtime-version pinning and a load-time compatibility check; Safetensors only if measured useful |
| F8 signal exchange | `_signals.py`, `jnwb_view` (export to jnwb) | Partial | cross-model packet contract (units, clock, identity, calibration level, delay rule) |
| F9 performance | `scripts/benchmark_*`, Atlas scaling programme | Partial | controlled-load measurements; evidence-limited |
| F10 CLI, UI | no console entry point in `pyproject.toml`, no `jaxfne/__main__.py`; HTML viewers under `artifacts/viewers/` | Missing (CLI), Unknown (UI) | audit before any design |

## Invariants

1. Unique ownership: every mutable coordinate z has exactly one owner,
   parameter or runtime state. Training and plasticity on the same
   coordinate need an explicit update policy (`artifacts/programme/ownership_053.md`
   is the H/W/K precedent).
2. Equivalence is typed:

   | class | acceptance |
   |---|---|
   | bit-exact | identical state and output bits over a declared horizon |
   | noise-floor | discrepancy within a separately measured numerical noise floor |
   | observable | predeclared statistics within tolerances over tested conditions |
   | structural | valid topology, identity map, initialization and continuation policy |

   Spiking dynamics amplify one float32 ulp into large weight differences
   (PCL K1h, `artifacts/etudes/pcl/README.md`); the noise-floor class takes
   its criterion from gate K1h-nf only once that gate has a result.
3. Portability is runtime-relative: artifact = tensors + manifest + runtime
   compatibility. No serialized tensor carries executable dynamics.

A shared record schema and verification protocol, not one forced function:
augmentation, training and scientific reduction keep their own APIs.

## Steps (post-0.5.5)

1. Gap audit: consumer paths, tests and manifests per row; replaces the
   table above. No new abstraction without a shown gap.
2. State and training: continuation ownership schema; restart equivalence;
   frozen values under momentum, weight decay, HDP and optimizer-state
   restore.
3. Transformations: extend the `augment` record pattern to pruning,
   compression, reparameterization and state migration; existing API kept.
4. Artifacts: version pinning and compatibility checks on the existing
   serialization; Safetensors only on measured benefit.
5. Ecosystem: signal packets, CLI, later a UI, over the existing execution
   API. No large-model claim before scaling tests.
