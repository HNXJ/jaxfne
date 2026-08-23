# Operator composition

**Chain tensor operators** from raw spikes to laminar LFP/CSD or scalp EEG/MEG
proxies. This page shows real composed pipelines — shapes and dtypes at every
handoff — not isolated operator definitions.

For single-operator contracts see Operator Doctrine (`docs/operator_doctrine.md` — repository-internal reference, excluded from the built site) and
the Tensor Operator Registry (`docs/api/tensor_operators.md` — repository-internal reference, excluded from the built site). The chains here were
verified in
[`tests/test_tensor_pipeline_custom_cfg.py`](https://github.com/HNXJ/jaxfne/blob/main/tests/test_tensor_pipeline_custom_cfg.py)
on a non-canonical 3-layer configuration, not only the default column.

## Chain 1: source → synaptic filter → cable filter → CSD-proxy

```text
source                    (n_steps, n_neurons)   raw per-neuron source
  → synaptic_current_tensor  (n_steps, n_neurons)   receptor-tau low-pass
  → cable_filter_sources     (n_steps, n_neurons)   depth/cell-type low-pass
  → project_laminar_sources  FieldOutput            lfp_proxy + csd_proxy + phi_e_proxy
  → csd_tensor (standalone)  (n_steps, n_contacts)  parity check on phi_e_proxy
```

Run end to end on a 40-neuron, 3-layer (`L1`/`L4`/`L6`), non-canonical-mix
(`E:0.6, PV:0.25, SST:0.15`) column:

```python
import jax, jax.numpy as jnp, numpy as np
import jaxfne as jtfne

MECHANISM_MAP = {"E": "AMPA", "PV": "GABA_A", "SST": "GABA_B", "VIP": "GABA_B"}

cfg = jtfne.laminar_cortex_config(
    seed=7, duration_ms=100.0, dt_ms=0.5, areas=("V1",),
    layers=("L1", "L4", "L6"), cell_types={"E": 0.6, "PV": 0.25, "SST": 0.15},
    n=40, emitter="izhikevich",
)
model = jtfne.construct(cfg)
sig = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.5, seed=3)

nt = model.neuron_table()
cell_type = np.array([row["cell_type"] for row in nt])
depth_z = np.array([row["z"] for row in nt])
source_native = np.asarray(jtfne.get_signal(sig, "source"))
# source_native.shape == (200, 40)   float32
```

### Stage 1 — Synaptic Tensor

```python
mechanisms = [MECHANISM_MAP[c] for c in cell_type]
tau_syn_ms = jtfne.synaptic_tau_from_mechanism(mechanisms)
# synaptic_tau_from_mechanism(mechanism: Sequence[str], *, dtype="float32") -> jax.Array
# tau_syn_ms.shape == (40,), values from {2.0 (AMPA), 5.0 (GABA_A), 150.0 (GABA_B)}

syn = jtfne.synaptic_current_tensor(jnp.asarray(source_native), tau_syn_ms, dt_ms=0.5)
# synaptic_current_tensor(spikes_pre: jax.Array, tau_ms: jax.Array, dt_ms: float) -> jax.Array
# syn.shape == (200, 40)   same as source_native — per-neuron receptor-tau low-pass
```

Domain: a raw per-neuron source (spike/current proxy). Codomain: the same
shape, filtered by the receptor time constant resolved from each neuron's
declared synaptic mechanism — this is the optional pre-filter stage in the
chain, not the depth-dependent one.

### Stage 2 — Cable-filter Tensor

```python
tau_cable_s = jtfne.cable_filter_tau(cell_type, depth_z)
# cable_filter_tau(cell_type, depth_z, *, tau_e_superficial_ms=1.0,
#                   tau_e_deep_ms=5.0, tau_pv_ms=0.5, tau_sst_ms=2.0,
#                   tau_vip_ms=2.0) -> jax.Array
# tau_cable_s.shape == (40,), in SECONDS (e.g. 0.00114, 0.00138, ...)

cab = jtfne.cable_filter_sources(jnp.asarray(source_native), tau_cable_s, dt_ms=0.5, order=2)
# cable_filter_sources(sources, tau_s, dt_ms, *, order=2) -> jax.Array
# cab.shape == (200, 40)
```

Domain: the per-neuron source (raw, or already synaptic-filtered — both
compose, since `cable_filter_sources` only cares about shape). Codomain: the
same per-neuron shape, now passed through a 2nd-order low-pass whose time
constant is a function of cell type **and** depth — this is the operator that
encodes "deep large pyramidal cells filter slower than superficial
interneurons."

### Stage 3 — Readout tensors (LFP, CSD, EEG, MEG)

```python
positions = np.stack([np.zeros(40), np.zeros(40), depth_z], axis=1)
fo = jtfne.project_laminar_sources(cab, positions, n_contacts=12)
# project_laminar_sources(sources, positions, *, n_contacts=16, width=0.1,
#                          mode="density_preserving", dtype="float32") -> FieldOutput
# fo.lfp_proxy.shape == fo.csd_proxy.shape == fo.phi_e_proxy.shape == (200, 12)
# fo.contact_depths.shape == (12,)

dz = fo.contact_depths[1] - fo.contact_depths[0]
csd_standalone = jtfne.csd_tensor(fo.phi_e_proxy, dz)
# csd_tensor(phi_e_proxy: jax.Array, dz: jax.Array | float) -> jax.Array
# csd_standalone.shape == (200, 12); allclose(csd_standalone, fo.csd_proxy) == True
```

`csd_tensor` is the named operator that `project_laminar_sources` calls
internally to derive `csd_proxy` from `phi_e_proxy` (the finite-difference
second spatial derivative). Calling it standalone on `fo.phi_e_proxy`
reproduces `fo.csd_proxy` exactly — this is the composability proof, not an
illustrative claim: `jnp.allclose(csd_standalone, fo.csd_proxy)` is `True` for
this run.

```python
n_contacts = fo.lfp_proxy.shape[1]
rng = np.random.default_rng(0)
eeg_leadfield = jnp.asarray(rng.normal(size=(8, n_contacts)).astype(np.float32))
meg_leadfield = jnp.asarray(rng.normal(size=(6, n_contacts)).astype(np.float32))

eeg = jtfne.eeg_proxy_transform(fo.lfp_proxy, eeg_leadfield)
meg = jtfne.meg_proxy_transform(fo.lfp_proxy, meg_leadfield)
# eeg_proxy_transform(source: jax.Array, leadfield: jax.Array) -> jax.Array
# meg_proxy_transform(source_oriented: jax.Array, leadfield: jax.Array) -> jax.Array
# eeg.shape == (200, 8)   meg.shape == (200, 6)
```

`eeg_proxy_transform` / `meg_proxy_transform` take the same `lfp_proxy`
codomain as input and apply a declared leadfield (a linear, not learned,
spatial mixing matrix you supply) — they compose directly off the field
stage, not off the raw source.

All five tensors in this chain (`syn`, `cab`, `fo.lfp_proxy`, `eeg`, `meg`)
were finite end to end in this run.

## Chain 2: source → field, without the cable filter

The cable-filter stage is optional. Dropping it is also a valid composition —
`project_laminar_sources` only requires a `(n_steps, n_neurons)` source and
`(n_neurons, 3)` positions, regardless of which upstream operator produced
the source:

```text
source  → project_laminar_sources → FieldOutput → eeg_proxy_transform → EEG-proxy
```

This is the chain you get for free any time you skip Stage 2 — every
downstream call in Chain 1 (`csd_tensor`, `eeg_proxy_transform`,
`meg_proxy_transform`) is shape-compatible with `fo` regardless of whether
`fo` was built from `cab` or from `source_native` directly.

## Why composition is the missing piece

[Configuration Grammar](configuration_grammar.md) documents the ~30
builder methods that *specify* a model. Operator Doctrine (`docs/operator_doctrine.md` — repository-internal reference, excluded from the built site)
documents the 7-stage *rule* each tensor operator satisfies on its own.
Neither one shows that the codomain of one operator is the domain of the
next without a conversion step — that the chain actually typechecks, not
just on paper. `tests/test_tensor_pipeline_custom_cfg.py` is what makes that
claim checkable; this page is what makes it readable.

## See also

- TFNE Operator Doctrine (`docs/operator_doctrine.md` — repository-internal reference, excluded from the built site) — the per-operator domain/codomain contract table.
- Tensor Operator Registry (`docs/api/tensor_operators.md` — repository-internal reference, excluded from the built site) — the full operator inventory.
- Operator Inventory (generated) (`docs/_generated/operator_inventory.md` — repository-internal reference, excluded from the built site) — the live export surface.
- [Objective Grammar](objective_grammar.md) — the user-facing run sequence this composition feeds into via `probe()`.
- `tests/test_tensor_pipeline_custom_cfg.py` — the executable proof this page documents.
