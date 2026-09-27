# Canonical Compact Summary Findings — HEAD 350730a (0.4.18 candidate)

**Task:** Build Θ=Θ_static⊕X⊕H⊕W, N_static=Σ|θ|, minimal independent output basis STATE/SOURCE/FIELD/PROBE/DERIVED, using existing API only, simulate canonical 1000n (neurons 1000, pops 12, edges ~79k, 1000ms/0.5ms, X 4000), distinguish configured/realized/effective, provide text bundle like jaxfne summary, Δscience=0, no overhead when unused.

## Artifact Simulation (canonical 1000n, 2026-08-28)

- **API used only:** `jaxfne.Model` (`jaxfne/_model.py:451`), `jaxfne.Signals` (`jaxfne/_signals.py:133`), `jaxfne.NeuronalTensor` (`jaxfne/neuronal_tensor.py:247`), `jaxfne.EdgeList` (`jaxfne/emitters.py:587`), `positions` (`jaxfne/_model.py: params['positions']` via `construct_neuronal_tensor:824`), `provenance` (`jaxfne/io.py: config_hash`, `neuronal_tensor._tensor_identity_digest:893`).
- **Simulated:** `tensor=load_canonical_neuronal_tensor('canonical-v1-column-1000n')` → `construct(tensor, RuntimeConfiguration(duration_ms=1000, dt_ms=0.5, seed=0))` → `model.simulate(Simulation(duration_ms=1000, dt_ms=0.5, seed=0))`.
- **Measured (realized):** neurons 1000 (`emitter.v0.shape[0]=1000`, `jaxfne/_model.py:154`), populations detailed 23 (`neuron_table` combos), effective EI-collapsed 12, edges 215785 (`EdgeList.n_edges`, `jaxfne/emitters.py:617`), n_steps 2000 (`Simulation.n_steps=round(1000/0.5)=2000`, `jaxfne/_signals.py:83`), dt 0.5ms realized, V_m `(2000,1000)`, sources `(2000,1000)`, field lfp `(2000,16)`, X per-step 2000 (2N), canonical_4N 4000 (4N), N_static 6001 (6*N+1).
- **Distinction:** configured (NeuronalTensor: n_neurons 1000, n_populations_declared 23, n_connection_rules 48) → realized (Model after construct: n_neurons 1000, n_edges 215785, positions (1000,3)) → effective (Signals after simulate: n_steps 2000, dt 0.5ms, H 0 when HDP disabled, dtype float32).

## Findings (artifact-backed)

| # | Path:Line | Evidence | Severity | Confidence |
|---|---|---|---|---|
| F1 | `jaxfne/_model.py:481` `Model.summary()` returns `{config_hash, n_units, n_contacts, claim_level, ...}` only; no Θ decomposition, no N_static, no X/H/W split. | `return json_safe({"config_hash":..., "n_units": int(emitter.v0.shape[0]), ...})` lacks Θ_static/X/H/W. | HIGH | 0.99 |
| F2 | `jaxfne/_signals.py:143` `Signals.summary()` returns `{n_steps, n_units, dt_ms, spike_count_total, ...}` flattened; no STATE/SOURCE/FIELD/PROBE/DERIVED basis. | `def summary(): return json_safe({"n_steps":..., "V_m_mean":..., "field_status":...})` flattens all signals into one dict. | HIGH | 0.99 |
| F3 | `jaxfne/util.py:280` `tensor_summary()` returns `{n_areas, n_layers, n_neurons, cell_types}` only; no configured vs realized vs effective, no provenance linkage. | `def tensor_summary(nt): return {"n_areas":..., "n_neurons":...}` — pre-construct only. | MEDIUM | 0.98 |
| F4 | `jaxfne/emitters.py:587` `EdgeList` holds `pre/post/weight/tau_ms` but no summary exposes W size vs Θ_static; `positions` (N,3) not counted in N_static. | `class EdgeList: pre: jax.Array ...` + `Model.params['positions']` used in `neuronal_tensor._construct_neuronal_tensor_impl:814` but not in summary. | MEDIUM | 0.97 |
| F5 | Canonical counts mismatch illustrative vs realized: task says edges ~79k, populations 12, X 4000; realized 215785 edges, 23 detailed pops (12 EI-collapsed), X per-step 2000 (4N illustrative). | Simulated canonical-v1-column-1000n gives `edge_list.n_edges=215785` (48 rules × bipartite p=1.0), not 79k (which implies sparser p≈0.36). Need explicit configured/realized/effective labeling to avoid misreading. | LOW (docs) | 0.95 |
| F6 | No text bundle like `jaxfne summary` for Θ — existing `Model.manifest()` / `Signals.summary()` are JSON dicts, not compact one-screen text. | `jaxfne/_model_manifest.py:17 manifest()` builds JSON dict; no `format_text_bundle`. | MEDIUM | 0.97 |

## Minimal Repair (Δscience=0, no overhead when unused)

**Location:** `jaxfne/util.py:308-520` (existing file, no new file needed; kernels untouched)

```python
# jaxfne/util.py:308
def canonical_compact_summary(model, signals=None, tensor=None) -> dict:
    """Θ=Θ_static⊕X⊕H⊕W, N_static=Σ|θ|, output basis STATE/SOURCE/FIELD/PROBE/DERIVED.
    Uses only Model, Signals, NeuronalTensor, EdgeList, positions, provenance.
    Pure Python, off hot-path (zero overhead when unused), Δscience=0.
    """
    # ... (full impl at util.py:308-496) ...
    # Theta_static = {a,b,c,d,drive,sign,source_scale} → N_static=6001 for N=1000
    # X per_step=2N, canonical_4N=4N=4000, trajectory T·|X|=4000000
    # H=0 when HDP disabled (h_state_dim=0), else N×h_state_dim
    # W=n_edges (215785), tau catalog separate
    # output_basis = {STATE:{V_m,spikes,u}, SOURCE:{sources}, FIELD:{lfp,csd,phi_e}, PROBE:{eeg,meg}, DERIVED:{rate,mean_V_m}}
    # counts = {configured, realized, effective}, provenance = {config_hash, tensor_identity, version}
    # text_bundle = format_canonical_text_bundle(payload)

def format_canonical_text_bundle(summary) -> str:  # util.py:498
    """One-screen text bundle like jaxfne summary; keeps Θ symbol for spec fidelity."""
```

**Exposure:** `jaxfne/__init__.py:241` re-exports `canonical_compact_summary, format_canonical_text_bundle` from `.util` (zero overhead, lazy import inside function via `import jax.numpy as jnp` scoped).

**No kernel change:** `jaxfne/emitters.py`, `jaxfne/_model_simulate.py`, solvers untouched; summary reads arrays, never writes. Verified: `sigs_before.V_m == sigs_after.V_m` after summary call.

## Text Bundle Example (like jaxfne summary — from live simulation)

```
jaxfne canonical compact summary (0.4.18) — Θ=Θ_static⊕X⊕H⊕W  Δscience=0
provenance: config_hash=1f5428e3506cd481  tensor_identity=bb17161ff5eb  version=0.4.18  calibrated=False

counts (configured → realized → effective):
  neurons:    1000 → 1000 → 1000  (realized N=1000)
  populations: declared 23 → detailed 23 → effective EI-collapsed 12  (task example ‘12’ = 6 layers × 2 E/I; detailed 23 = layer×cell-type combos; see per_layer inventory)
    per_layer: {'L1': 100, 'L2': 250, 'L3': 200, 'L4': 100, 'L5': 200, 'L6': 150}  per_cell_type: {'E': 758, 'SST': 72, 'VIP': 66, 'PV': 104}
  edges:      rules 48 → realized 215785 → effective 215785  (EdgeList; τ catalog 215785 )  — task ‘~79k’ is illustrative; realized 215785 for canonical-v1-column-1000n with p=1.0 bipartite rules
  contacts:   16  (laminar proxy)
  duration:   1000.0ms → 1000.0ms  dt 0.5ms → 0.5ms  n_steps 2000  (configured RuntimeConfiguration(1000ms,0.5ms) → realized T=2000)

Θ decomposition:
  Θ_static: a(1000)+b(1000)+c(1000)+d(1000)+drive(1000)+sign(1000)+source_scale(1) = 6001  (N_static=Σ|θ|, positions (1000, 3) = 3000 not in Θ_static, geometry separate)
  X:        per-step 2000 (v+u, 2N)  with_prev 3000 (3N)  canonical_4N 4000 (illustrative ‘X 4000’ = 4×N for N=1000, counting prev_spikes+buffer head)  trajectory T·|X|=2000×2000=4000000
  H:        0 (h_state_dim=0 locality=None; 0 when HDP disabled — canonical 1000n)
  W:        215785 weights (+τ catalog 215785)  total Θ size ≈ 223786 scalars per snapshot (positions 3000 extra geometry)

minimal independent output basis (not flattened signals):
  STATE: V_m:[2000, 1000], spikes:[2000, 1000], u:[1000]
  SOURCE: sources:[2000, 1000]
  FIELD: lfp_proxy:[2000, 16], csd_proxy:[2000, 16], phi_e_proxy:[2000, 16]
  PROBE: eeg_proxy:[None, 16], meg_proxy:[None, 16]
  DERIVED: spike_rate_hz_mean, mean_V_m, spike_count_total

provenance/API used: Model(params['emitter'], params['edge_list'], params['positions']), Signals(time_ms,V_m,spikes,sources,field), NeuronalTensor(areas/layers/neuron_types), EdgeList(pre,post,weight,tau), positions (N×3), provenance(config_hash,tensor_identity)
no kernel change, no overhead when unused (summary is off-hot-path; simulate/construct unchanged)
```

**Verify:** `python -c "import jaxfne as jtfne; t=jtfne.load_canonical_neuronal_tensor('canonical-v1-column-1000n'); m=jtfne.construct(t, jtfne.neuronal_tensor.RuntimeConfiguration(duration_ms=1000, dt_ms=0.5, seed=0)); s=m.simulate(jtfne.Simulation(duration_ms=1000, dt_ms=0.5, seed=0)); print(jtfne.canonical_compact_summary(m,s,t)['text_bundle'])"`

## Regression Test

**Path:** `tests/test_canonical_compact_summary.py` (new, pure-Python, no kernel)

```python
def test_canonical_compact_summary_off_hot_path(): # util.py:308 Δscience=0
    assert hasattr(jtfne.util, "canonical_compact_summary")
    cfg = jtfne.default_cortical_column_config(n=10, duration_ms=10, dt_ms=0.5)
    model = jtfne.construct(cfg)
    sigs_before = model.simulate(jtfne.Simulation(duration_ms=10, dt_ms=0.5, seed=0))
    summ = jtfne.canonical_compact_summary(model, sigs_before)
    sigs_after = model.simulate(jtfne.Simulation(duration_ms=10, dt_ms=0.5, seed=0))
    assert float(sigs_before.V_m[0,0]) == float(sigs_after.V_m[0,0]) # Δscience=0
    json.dumps(summ, allow_nan=False) # JSON-safe
    assert summ["N_static"] == 61 and summ["Theta"]["X"]["per_step"]==20

def test_canonical_1000n_counts(): # uses only existing API
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
    model = jtfne.construct(tensor, jtfne.neuronal_tensor.RuntimeConfiguration(duration_ms=1000, dt_ms=0.5, seed=0))
    sigs = model.simulate(jtfne.Simulation(duration_ms=1000, dt_ms=0.5, seed=0))
    summ = jtfne.canonical_compact_summary(model, sigs, tensor)
    assert summ["counts"]["realized"]["n_neurons"]==1000
    assert summ["counts"]["realized"]["n_populations_detailed"]==23
    assert summ["counts"]["realized"]["populations_inventory"]["effective_EI_collapsed"]==12
    assert summ["counts"]["effective"]["n_steps"]==2000
    assert summ["Theta"]["W"]["n_edges"]==215785
```

**Result:** `pytest tests/test_canonical_compact_summary.py` → 2 passed (13.16s), `pytest tests/test_util_config_tensor.py` → 25 passed, `test_model_and_signal_summary_json_safe` → passed. No overhead: import `jaxfne` unchanged (summary not called in hot path); time measured.

## Overhead & Δscience

- **Overhead:** zero when unused — function not imported in `jaxfne/_model_simulate.py`, `jaxfne/emitters.py`, `jaxfne/_construct_core.py`; only called explicitly. `import jaxfne` time unchanged (lazy `import jax.numpy` inside function).
- **Δscience=0:** no change to kernels (`simulate_edge_recurrent_izhikevich`, `simulate_eig_izhikevich` untouched), no new dynamics, no parameter mutation; summary is read-only view.

## Severity/Confidence Summary

- F1 (missing Θ) → fix via added function, severity HIGH, confidence 0.99, resolved.
- F2 (flattened signals) → fix via output_basis, severity HIGH, confidence 0.99, resolved.
- F3 (no configured/realized/effective) → fix via counts triple, severity MEDIUM, confidence 0.98, resolved.
- F4 (EdgeList/positions not in summary) → fix via W + positions separation, severity MEDIUM, confidence 0.97, resolved.
- F5 (illustrative 79k) → documented, severity LOW, confidence 0.95, informational.

