# Effective Dual Meaning Fix — Findings (HEAD 875b1a9 + 350730a → executed vs effective)

**Task:** Fix dual meaning of `effective` — `canonical_compact_summary` used `effective` for post-simulate runtime quantities, while PseudoGenome/JDNA docs use `effective` for intervention-demonstrated effect (`ΔX`). Change to `configured→realized→executed` for construction/runtime, reserve `effective` for causal evidence (`ΔX` under intervention). Update `docs/guides/jdna.md`, `docs/doctrine`, `docs/guides/configuration_grammar.md`, and `jaxfne/util.py` docstring/comments to use `executed` for post-simulate, `effective` only for `ΔX`. Keep `Δscience=0`, no kernel change. Provide artifact-backed findings.

**Authority:** CODE, workspace `jaxfne` @ `875b1a9` (v0.4.18) + `350730a` docs.

## 1. Artifact-backed before state (dual meaning)

| Location | Before | Evidence | Dual meaning |
|---|---|---|---|
| `jaxfne/util.py:309-366` | `counts: configured vs realized vs effective` and `effective = after simulate() with runtime overrides` | `canonical_compact_summary` docstring `configured vs realized vs effective: configured = declared … effective = after simulate()` and `counts["effective"]` with `n_steps_effective`, `dt_ms_effective`, `n_edges_effective` | Runtime post-simulate called `effective` |
| `jaxfne/util.py:465-471` | `effective_EI_collapsed`, `effective collapsed inhibitory` | `populations_inventory["effective_EI_collapsed"]` | Collapsed E/I called `effective` (not causal) |
| `docs/guides/jdna.md:33-37` | `## Configured → Realized → Effective` and `Effective measures change ΔX` | Heading and `Compact: p_EI → E_EI → ΔX` | Docs already correct: `effective = ΔX`, but summary's `effective = runtime` conflicts ⇒ dual meaning |
| `docs/guides/configuration_grammar.md` | No `configured/realized/executed/effective` note | File has no vocabulary legend | Missing disambiguation |
| `docs/doctrine/relative_quantity_grammar.md:127-144` | `effective = mapped output C_p(p0,r_p)` with no `executed` distinction | `§6 Language and claim gates` lists `effective` as `p_eff` only | No `executed` (runtime) vs `effective` (causal) separation |

**Finding:** Runtime `effective` (summary) vs causal `effective` (PseudoGenome `ΔX`) ⇒ same word, two meanings. Severity HIGH (semantic), confidence 0.99.

## 2. Minimal repair (Δscience=0, no kernel)

**Files changed (staged):**

- `jaxfne/util.py` — docstring: `counts: configured vs realized vs executed; effective reserved for causal ΔX`; params: `realized/executed time axes (executed = realized)`; Notes: `configured vs realized vs executed: … executed = after simulate()` + `effective (ΔX) is causal, not runtime`; code: rename `n_steps_effective→n_steps_executed`, `dt_ms_effective→dt_ms_executed`, `duration_ms_effective→duration_ms_executed`, `counts["executed"]` with `n_edges_executed`, keep `counts["effective"]` as alias to `counts["executed"]` for one-release compat (inner `n_edges_effective` alias, `populations_inventory["EI_collapsed"]` with `effective_EI_collapsed` alias); `format_canonical_text_bundle` now reads `executed` (fallback `effective`), prints `counts (configured → realized → executed)` and `note: effective=ΔX under intervention`.

- `docs/guides/jdna.md` — heading `Configured → Realized → Executed → Effective`; body adds `Executed is runtime-realized quantity after simulate — Signals time axis …`; compact adds `T_executed` stage: `p_EI → E_EI → T_executed → ΔX`.

- `docs/guides/configuration_grammar.md` — new `## Configured → Realized → Executed → Effective (vocabulary note)` section explaining `configured/realized/executed/effective`.

- `docs/doctrine/relative_quantity_grammar.md` — `§6` now lists `effective` as `p_eff` + `effective reserved for causal ΔX` and adds `executed` as runtime-realized; paragraph defines `configured→realized→executed→effective`.

- `docs/api/jdna.md` — `develop` output semantics now notes `realization and execution do not establish effectiveness; vocabulary configured→realized→executed→effective`.

**No kernel change:** `jaxfne/emitters.py`, `jaxfne/_model_simulate.py`, solvers untouched; summary is pure-Python off hot-path (`import jax.numpy as jnp` local, zero overhead when unused). `overhead` field unchanged.

## 3. Verification receipts (live, this checkout)

**Δscience=0 (no kernel):**
```
cfg=jtfne.default_cortical_column_config(n=10,…)
model=jtfne.construct(cfg)
sigs_before=model.simulate(Simulation(10,0.5,seed=0))
summ=jtfne.canonical_compact_summary(model,sigs_before)
sigs_after=model.simulate(Simulation(10,0.5,seed=0))
assert float(sigs_before.V_m[0,0]) == float(sigs_after.V_m[0,0])  # True
```
Run: `pytest tests/test_canonical_compact_summary.py` → 2 passed (13s). `json.dumps(summ,allow_nan=False)` passes.

**Counts (canonical 1000n live):**
```
tensor=load_canonical_neuronal_tensor('canonical-v1-column-1000n')
model=construct(tensor, RuntimeConfiguration(1000,0.5,seed=0))
sigs=model.simulate(Simulation(1000,0.5,seed=0))
summ=canonical_compact_summary(model,sigs,tensor)
configured n_neurons 1000, n_connection_rules 48
realized n_neurons 1000, n_edges 215785, EI_collapsed 12 (alias effective_EI_collapsed 12)
executed n_steps 2000, dt_ms 0.5, n_edges_executed 215785 (alias n_edges_effective)
effective alias == executed (True) — runtime alias preserved; causal ΔX not computed here (requires intervention comparison)
```
`N_static 6001 = 6*1000+1`, `X per_step 2000 (2N)`, `canonical_4N 4000`.

**Doc audit:**
```
python scripts/audit_public_docs_language.py → {"doc_files_with_violations":0,"pass":true}
python scripts/audit_vocabulary.py → pass (protected_terms includes effective)
```
`effective` now correctly flagged as `p_eff` or causal `ΔX`; `executed` for runtime.

## 4. After state

```
configured (declarative: p_EI, duration_ms) → realized (after construct: EdgeList, positions, neuron_table) → executed (after simulate: Signals.time_ms, n_steps, dt_ms, dtype, H/W sizes) → effective (ΔX under intervention: ablation/shuffle/lesion causal evidence)
```
- `effective` never for runtime; `executed` never for causal.
- `relative_quantity_grammar.md §6` and `jdna.md` now cross-reference `util.canonical_compact_summary`.
- Backward compat: `counts["effective"]` and `effective_EI_collapsed`/`n_edges_effective` remain as aliases one release; new code should use `executed`/`EI_collapsed`/`n_edges_executed`.

## 5. Scope & notes

- Change is terminology/docs only; no new dynamics, no parameter mutation, no optimizer exposure change.
- `docs/doctrine` update is `relative_quantity_grammar.md` only; other doctrine files (`rbs_rbd_hdp.md`, `protocol_h_*`) already use `effective` only as `p_eff` or causal, not runtime — no change needed.
- Artifact: `artifacts/effective_dual_meaning_fix_findings.md` (this file) + live `summ["text_bundle"]` (see `python -c "import jaxfne as jtfne; …; print(summ['text_bundle'])"`).

**Result:** Dual meaning eliminated with minimal justified Δ, Δscience=0, no kernel change, artifact-backed, backward-compatible.

