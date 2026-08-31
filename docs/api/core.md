# Core API

Main classes for configuration, model construction, simulation, and signal handling.

All source references below point at the post-split module that owns each
symbol (`jaxfne/core.py` is a compatibility re-export shim; the monolith
anchors were retired when `core.py` was split into `_config.py`, `_model.py`,
`_signals.py`, `_construct_core.py`, `_construct_extras.py`,
`_construct_presets.py`, and `_model_readout.py`).

## Configuration

```python
jaxfne.Configuration()
```

Declarative TFNE model configuration (`_config.py`, frozen dataclass). It is the
anatomical/model declaration, separate from the compiled model. Methods return new
objects (immutable, chainable construction).

### Fields

Real dataclass fields (`_config.py`):

- `networks` (`list[dict]`, default `[]`)
- `emitters` (`list[dict]`, default `[]`)
- `fields` (`list[dict]`, default `[]`)
- `probes` (`_ProbeDeclarations`, a list-like, callable proxy — see below)
- `metadata` (`dict`, default from `_default_metadata()`)

`column`/`cell_type_map`/etc. are metadata written by the chainable methods below,
distinct from the top-level dataclass fields listed above.

### Methods

Only methods verified against `core.py` are documented here; `Configuration` has many
more (e.g. `plasticity`, `homeostasis`, `hdp`, `mechanisms`, `connections`, `lesions`,
`trainables`, `objective_outputs`, `areas`, `layer_fractions`,
`area_layer_cell_types`, `drive`, `objective`, `optimizer`, `validate`) — see
`jaxfne-config` / `jaxfne-modeling-optimization-schema` skills for the full surface.

#### `runtime(**kwargs) -> Configuration`

`_config.py`. Maps directly to `update_metadata(**kwargs)` — this is a plain
metadata write, distinct from a compiled `RuntimeConfig`. Typical keys: `seed`,
`dtype`, `duration_ms`, `dt_ms`.

**Example:**
```python
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
```

#### `column(name: str, layers: Sequence[str], n: int) -> Configuration`

`_config.py`. Declares one cortical column; accumulates into
`metadata["columns"]` and rebuilds a single unified `networks[0]` entry
(`kind="multi_column"`) spanning all declared columns. Raises `ValueError` on an
empty/duplicate name, empty `layers`, or non-positive `n`.

**Example:**
```python
cfg = cfg.column("V1", layers=["L2/3", "L4", "L5"], n=100)
```

#### `cell_types(fractions: Mapping[str, float]) -> Configuration`

`_config.py`. Sets cell-type fractions in `metadata["cell_types"]` and on
`networks[0]["cell_types"]`. Values are stored exactly as given, unnormalized.
Raises `ValueError` on empty input, non-finite/negative fractions, or zero
total mass.

**Example:**
```python
cfg = cfg.cell_types({"E": 0.8, "PV": 0.2})
```

Valid labels for the Izhikevich emitter family are `E`, `PV`, `Inl`, `SST`,
`Ing`, `VIP` (`jaxfne.emitters.IZHIKEVICH_CELL_TYPE_DEFAULTS`) — stored keys for
reduced emitter classes, read as `E-like`, `PV-like`, `SST-like`, `VIP-like`
(no literal biological identity; see `docs/api/emitters.md` wording note) — only these
specific labels are accepted, a generic `"I"` aggregate label is rejected;
`construct()` raises `ValueError: unknown Suite No. 2 cell type label` for
any other string. `cell_types()` itself just stores the dict as given; label
validation happens later, at `construct()` time.

#### `connectivity(**kwargs) -> Configuration`

`_config.py`. Declares connectivity metadata into `metadata["connectivity"]`
(merged with any prior call) and sets `metadata["connectivity_status"] =
"declared_metadata_proxy"`. Declaration only — simulated dynamics stay as
configured elsewhere.

**Example:**
```python
cfg = cfg.connectivity(feedforward_gain=1.0)
```

#### `set_emitter(family: str = "izhikevich", preset: str = "cortical_eig") -> Configuration`

`_config.py`. Thin wrapper: `self.emitter(family=family, preset=preset)`.

**Example:**
```python
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
```

#### `probes(modes, *, name="multimodal_probe", n_contacts=None, ensure_defaults=True, **kwargs) -> Configuration`

`_config.py` (`_ProbeDeclarations.__call__`). `cfg.probes` is itself a
list-like object (read path: `len(cfg.probes)`, `cfg.probes[0]`) that is also
**callable** (write path): calling it returns a new `Configuration` via
`cfg._with_probe_modes(...)`. `modes` stays a declarative label list — no
physical-sensor claim is introduced by calling this.

**Parameters:**
- `modes` (`Sequence[str]`): probe/readout mode labels, e.g. `["MUA-proxy", "LFP-proxy", "CSD-proxy"]`
- `name` (`str`, default `"multimodal_probe"`)
- `n_contacts` (`int | None`)
- `ensure_defaults` (`bool`, default `True`): adds canonical Izhikevich emitter + laminar proxy field declarations when they are still absent
- `**kwargs`: extra probe metadata (e.g. `contact_depths`, `claim_level`)

**Example:**
```python
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])
```

---

## Model

```python
jaxfne.Model
```

`_model.py`. Frozen dataclass — the constructed, immutable, runnable model built
from a validated `Configuration`. Also exported as alias `Net`. A computational
scaffold: its field and probe outputs are proxy readouts rather than calibrated
physical signals.

### Fields

- `cfg` (`Configuration`): source configuration
- `params` (`dict[str, Any]`): dynamic pytree (arrays, may be tuned/traced)
- `static` (`dict[str, Any]`): JIT-static, non-array metadata

`Model`'s real attributes are `cfg`/`params`/`static` (no separate `geometry`/
`basis_spec`) — geometry lives inside `params`/`static` (e.g. `params["positions"]`,
`static["n_contacts"]`) as nested data, rather than as its own top-level field.

### Methods

`Model` has many methods beyond those listed; this covers the ones this doc
documents.

#### `simulate(sim: Simulation, paradigm: Any | None = None) -> Signals`

`_model.py`. Runs the default **Izhikevich / edge-list EIG vertical slice**.
`sim` is a **required, positional `Simulation` object** — not `duration_ms`/`dt_ms`/`seed`
keywords directly on `Model.simulate`. Use the module-level `jtfne.simulate(model,
duration_ms=..., ...)` helper (below) for the kwarg form.

When the constructed emitter is Izhikevich (or another edge-list path that
reaches `_simulate_arrays`), and `paradigm` is a `StimulusSchedule` or
`ParadigmCondition`, its drive array is injected as internal (uncalibrated)
current at each timestep.

**`homeostatic_ei` is a separate emitter family.** `Model.simulate` dispatches
to `_simulate_homeostatic_ei` before paradigm resolution; a `paradigm`
argument is **not** applied on that path. Supported extra drive for HEI is the
kernel argument `drive_schedule` on `simulate_homeostatic_ei(...)`, or the
family's declared baseline drive in params. Do not infer cross-family stimulus
equivalence from the Izhikevich `paradigm` contract.

### Supported method surface per emitter family

Methods that require the Izhikevich emitter raise a clear
`NotImplementedError` on other families (never an `AttributeError`). This is
the intended boundary. Supported methods by emitter family:

| Method | Izhikevich family | `homeostatic_ei` family |
|---|---|---|
| `construct` / `simulate` / `probe` / `evaluate` / `run_receipt` | ✅ | ✅ |
| `tune` (scalar + matrix/multi-parameter) | ✅ | ✅ (matrix-parameter tuning) |
| `summary` / `neuron_table` | ✅ | ❌ `NotImplementedError` |
| `checkpoint` / `simulate_batch` | ✅ | ❌ `NotImplementedError` |
| `with_emitter_parameters` | ✅ | ❌ `NotImplementedError` |

If your planned workbench experiments use `homeostatic_ei` and need
`summary`/`neuron_table`/`checkpoint`/`simulate_batch`, use the Izhikevich
family instead (the canonical edge-list path).

**Example:**
```python
model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.1, seed=7)
signals = model.simulate(sim)
# or, equivalently, via the module-level kwarg-form helper:
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

#### `compute_readout(signals: Signals, specs: Sequence[ReadoutSpec]) -> list[ReadoutResult]`

`_model_readout.py`. Canonical v0.1 workflow method — computes scalar features from
`Signals` per a list of `ReadoutSpec`. Returns a **list** of `ReadoutResult`, one
per spec, in the same order (not a single combined `ReadoutResult`).

**Example:**
```python
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz"),
    jtfne.readout_spec("voltage", "mean_V_m"),
])
```

#### `summary() -> dict`

`_model.py`. Compact JSON-safe metadata: `config_hash`, `n_units`,
`n_contacts`, `claim_level`, `source_calibration_status`, `field_solver_status`,
`field_claim_level`, `physical_amplitude_calibrated` (always `False` here — no
calibrated-amplitude claim).

#### `neuron_table() -> list[dict]`

`_model.py`. Returns declared neuron metadata rows (`neuron_id`, `area`,
`layer`, `cell_type`, `z`) for area/layer/cell-type grouping and selectors.

#### `select(*, area=None, area_id=None, layer=None, cell_type=None, ids=None, allow_empty=False) -> jax.Array`

`_model.py`. Resolves semantic selectors to neuron row indices via
`SelectorSpec` over `neuron_table()`. Returns an int32 array of row positions.
Raises `ValueError` on an empty match unless `allow_empty=True`; raises
`KeyError` if a requested field is absent from the neuron table.

---

## Simulation

```python
jaxfne.Simulation
```

`_signals.py`. Frozen dataclass — immutable specification of one simulation run.

### Fields

- `duration_ms` (`float`, default `1000.0`) — must be positive and finite
- `dt_ms` (`float`, default `0.05`) — must be positive and finite
- `plasticity` (`float`, default `0.0`)
- `seed` (`int`, default `0`)
- `record_sources` (`bool`, default `True`)
- `record_fields` (`bool`, default `True`)
- `poisson_drive` (`dict | None`, default `None`)
- `runtime` (`RuntimeConfig | None`, default `None`)
- `ablation` (`str | None`, default `None`)

`n_steps` is a **property** (`round(duration_ms / dt_ms)`), not a constructor
field, and `__post_init__` raises `ValueError` if it computes to `<= 0`.

**Example:**
```python
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.1, seed=7)
```

---

## Signals

```python
jaxfne.Signals
```

`_signals.py`. Frozen dataclass — simulation output container.

### Fields

Real fields (`_signals.py`):

- `time_ms` (`jax.Array`)
- `V_m` (`jax.Array`) — membrane voltage, shape `(n_steps, n_units)`
- `spikes` (`jax.Array`) — spike raster, shape `(n_steps, n_units)`
- `sources` (`jax.Array | None`) — source density (present when `record_sources=True`)
- `field` (`FieldOutput | None`) — laminar proxy field output (LFP/CSD/EEG/MEG proxies live inside this object, not as separate top-level `Signals` fields; present when `record_fields=True`)
- `metadata` (`dict[str, Any]`)

`sources` (plural) is the source-density array, and the field
proxies (LFP/CSD/etc.) are attributes of `field: FieldOutput`, accessed via
`.get(...)` (below) or `signals.field.<attr>`.

### Methods

#### `summary() -> dict`

`_signals.py`. JSON-safe diagnostics: `n_steps`, `n_units`, `dt_ms`,
`spike_count_total`, `spike_rate_hz_mean`, `V_m_mean`, `field_status`
(`"present"`/`"absent"`), `field_claim_level`.

#### `get(key, *, selector=None, area=None, layer=None, cell_type=None, ids=None, trial=None, as_numpy=False) -> Any`

`_signals.py`. Returns a named signal array, filtered to selected
neurons via `area`/`layer`/`cell_type`/`ids` or an explicit `SelectorSpec`
(mutually exclusive with the individual fields). Key aliases: `vm`/`V_m`/`voltage`
-> `V_m`; `spk`/`spikes`/`raster` -> `spikes`; `src`/`sources` -> `sources`;
`lfp`/`csd`/`phi_e` -> the corresponding laminar proxy readout on `field`;
`field_source` -> field source proxy. Unknown keys raise `KeyError`. Multi-trial
execution is handled via `jtfne.run_trials`/`Model.run_trials`.

Use `summary()` for a JSON-safe view, or `jaxfne.io.json_safe(...)` on the fields you need.

**Example:**
```python
vm = signals.get("V_m")
lfp = signals.get("lfp")  # raises ValueError if record_fields=False was used
```

---

## Objective

```python
jaxfne.Objective
```

`_signals.py`. Frozen dataclass — declarative objective specification (losses,
regularizers, diagnostic gates). All specs are plain dicts (no callables), so the
objective is always JSON-serializable. Gate pass/fail is a computational
diagnostic only, not empirical validation.

### Fields

- `name` (`str`, default `"anonymous"`)
- `kind` (`str`, default `"generic"`)
- `losses` (`list[dict]`, default `[]`)
- `regularizers` (`list[dict]`, default `[]`)
- `gates` (`list[dict]`, default `[]`)

### The module-level `objective()` helper — real signature is zero-arg

```python
jaxfne.objective() -> Objective
```

`_construct_presets.py`. Returns a **fresh, empty `Objective()`** — equivalent to
`Objective()` directly. It does **not** take `name`/`metric`/`target`/`weight`
keyword arguments; those belong to the instance methods below, called on the
`Objective` this returns.

**Example (real usage):**
```python
obj = jtfne.objective()
obj = obj.loss("spike_rate", target=10.0, metric="spike_rate_hz", weight=1.0)
```

### Methods (build the spec by chaining on the empty `Objective`)

#### `loss(name, target=None, weight=1.0, metric=None, metadata=None) -> Objective`

`_signals.py`. Appends a loss spec dict to `losses`.

#### `regularizer(name, target=0.0, weight=1.0, metric=None, metadata=None) -> Objective`

`_signals.py`. Appends a regularizer spec dict to `regularizers`.

#### `gate(name, threshold, criterion="below", metric=None, metadata=None) -> Objective`

`_signals.py`. Appends a diagnostic gate spec dict to `gates`.

#### `compose(*others: Objective) -> Objective`

`_signals.py`. Concatenates losses/regularizers/gates from other `Objective`
instances into a new merged `Objective`.

### `rate_targets(groups, targets_hz, weights=None) -> Objective`

`_construct_presets.py`. Separate module-level factory (not a method) that builds an
`Objective` with `kind="group_rate_targets"` directly, for `Model.tune()`'s
group-wise firing-rate optimization loop.

```python
objectives = jtfne.rate_targets(
    groups={"first_half": range(0, 24), "second_half": range(24, 48)},
    targets_hz={"first_half": 5.0, "second_half": 10.0},
)
```

---

## ReadoutSpec

```python
jaxfne.readout_spec(name: str, metric: str, *, time_window_ms=None, n_contacts_slice=None, metadata=None) -> ReadoutSpec
```

`_construct_extras.py` (factory); dataclass at `_signals.py`. Declarative specification
for extracting a scalar feature from `Signals`.

**Parameters:**
- `name` (`str`): unique label for this readout spec
- `metric` (`str`): one of `_KNOWN_READOUT_METRICS`
- `time_window_ms` (`tuple[float, float] | None`, keyword-only): optional temporal slice
- `n_contacts_slice` (`tuple[int, int] | None`, keyword-only): optional contact-depth slice for field modes
- `metadata` (`dict | None`, keyword-only)

Note: `name` and `metric` are positional-or-keyword; the remaining three are
**keyword-only** (`*` in the signature) — `jtfne.readout_spec("x", "spike_rate_hz", time_window_ms=(0, 100))`, not a positional third argument.

**Available metrics** (`_KNOWN_READOUT_METRICS`, `_construct_extras.py`): `spike_rate_hz`,
`spike_count`, `mean_V_m`, `csd_abs_mean`, `lfp_abs_mean`, `source_abs_mean`.

**Example:**
```python
readout = jtfne.readout_spec("firing_rate", "spike_rate_hz")
```

---

## ReadoutResult

```python
jaxfne.ReadoutResult
```

`_signals.py`. Frozen dataclass — result of applying one `ReadoutSpec` to
`Signals`.

### Fields

- `spec_name` (`str`)
- `metric` (`str`)
- `value` (`float | None`)
- `status` (`str`, default `"computed"`) — one of `"computed"`, `"no_field"`, `"unknown_metric"` (also `"empty_time_window"` when a `time_window_ms` slice is empty)
- `claim_level` (`str`, default `"computational_scaffold"`)
- `physical_amplitude_calibrated` (`bool`, default `False`)
- `metadata` (`dict`, default `{}`)

`Model.compute_readout()` returns a `list[ReadoutResult]` (one result object per spec, each self-describing).

`name` is a compatibility `@property` alias for `spec_name`.

### Methods

#### `to_dict() -> dict`

`_signals.py`. JSON-safe dict of all fields above.

**Example:**
```python
for result in readouts:
    print(result.name, result.metric, result.value, result.status)
result_dict = readouts[0].to_dict()
```

---

## Helper Functions

### `construct(cfg, runtime=None, *, geometry=None) -> Model`

`_construct_core.py`. Two call forms:
- `construct(cfg)` / `construct(cfg, geometry=...)` — the `Configuration`-based path (original signature).
- `construct(tensor, runtime)` — the `NeuronalTensor` path (0.4.7+): `tensor` is a `jaxfne.neuronal_tensor.NeuronalTensor`, `runtime` a `RuntimeConfiguration` (defaults to `RuntimeConfiguration()`).

Passing `runtime=` together with a `Configuration` raises `ValueError` (a
`Configuration` already carries runtime via `.runtime(...)`). Passing `geometry=`
together with a `NeuronalTensor` raises `ValueError`.

**Example:**
```python
model = jtfne.construct(cfg)
```

### `simulate(model, sim=None, paradigm=None, **kwargs) -> Signals`

`_construct_core.py`. Module-level convenience wrapper — allows passing either an
explicit `Simulation` via `sim=`, or simulation parameters (`duration_ms`,
`dt_ms`, `seed`, `record_sources`, `record_fields`, `runtime`, `dtype`, ...)
directly as keyword arguments. Passing both `sim=` and other kwargs raises
`ValueError`. When no explicit `runtime`/`Simulation` is given, the runtime
declared on `model.cfg` via `.runtime(...)` is inherited; a `dtype=` kwarg
overrides the inherited dtype (and cannot be combined with `runtime=`).

`paradigm` injection follows the **Izhikevich vertical-slice** contract
documented on `Model.simulate` above. It does not apply to `homeostatic_ei`
models constructed via `set_emitter("homeostatic_ei")`.

**Example:**
```python
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

### `simulation(**kwargs) -> Simulation`

`_construct_presets.py`. Thin factory: `Simulation(**kwargs)`.

**Example:**
```python
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.1, seed=7)
```

### `configuration() -> Configuration`

`_construct_presets.py`. Returns a fresh, empty `Configuration()` — entry point for the
chainable grammar `configuration().network(...).emitter(...)...`.

**Example:**
```python
cfg = jtfne.configuration()
```

### `compute_fields(model: Model, signals: Signals) -> FieldOutput`

`_construct_core.py`. Thin accessor over `signals.field` (already built inside
`simulate()`); raises `ValueError` if `signals.field is None` (no field-capable
probe modes declared) rather than fabricating a placeholder.

### `objective() -> Objective`

See [Objective](#objective) above — documented there since it's the constructor
entry point for that class, not a standalone helper with its own parameters.
