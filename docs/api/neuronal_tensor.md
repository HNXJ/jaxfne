# Neuronal Tensor API

The `neuronal_tensor` module defines the **canonical network representation** for
jaxfne circuits. Every circuit — regardless of how many areas, layers, or cell
types it has — reduces to one uniform shape:

```
NeuronalTensor = [Areas, AreaConnections]
Area           = [Layers × NeuronTypes, InterConnections]
```

Import:

```python
from jaxfne import (
    NeuronalTensor, Area, AreaConnection, Layer, NeuronType,
    Geometry3D, Pose3D, StaticParams, PlasticParams,
    InterConnection, save_neuronal_tensor, load_neuronal_tensor,
    merge_neuronal_tensors, neuronal_tensor_to_configuration,
    construct_neuronal_tensor, default_relative_size,
    list_canonical_neuronal_tensors, load_canonical_neuronal_tensor,
)
```

> **Additive, not replacing.** The existing `jaxfne.core.Configuration` /
> `jaxfne.builders.laminar_cortex_config` path is untouched.
> `neuronal_tensor_to_configuration` bridges a `NeuronalTensor` into that
> existing `construct` / `simulate` pipeline.

---

## Value tags

Every numeric field carries a `value_tag`:

| Tag | Meaning |
|-----|---------|
| `"relative"` | Default. Proportional/structural — not empirically grounded. |
| `"calibrated_proxy"` | Derived from data via a proxy/approximation. |
| `"calibrated"` | Directly backed by experimental evidence. |

Opt in to a stronger tag only with evidence. Default is `"relative"`.

---

## Data classes

### `Geometry3D`

```python
@dataclass
class Geometry3D:
    distribution: str = "uniform_random"
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    z_range: tuple[float, float] = (0.0, 1.0)
    value_tag: ValueTag = "relative"
```

Always 3D. To represent a 2D or 1D layer, collapse one or two axes by fixing
their range at `(0.0, 0.0)`.

Currently only `distribution="uniform_random"` is implemented;
`construct_neuronal_tensor` will raise `NotImplementedError` for any other value.

---

### `NeuronType`

```python
@dataclass
class NeuronType:
    name: str
    relative_size: float = 1.0
    fraction: float | None = None
    value_tag: ValueTag = "relative"
```

`fraction` is an optional per-type population fraction. If every `NeuronType`
in a `Layer` declares a `fraction`, `neuronal_tensor_to_configuration` uses
those (normalized) fractions instead of splitting the layer's population
evenly across its declared types.

**Class method:**

```python
@classmethod
def make(cls, name: str, relative_size: float | None = None,
         fraction: float | None = None,
         value_tag: ValueTag = "relative") -> NeuronType
```

Convenience constructor that auto-fills `relative_size` from
`default_relative_size(name)` when `relative_size` is `None`.

```python
e  = NeuronType.make("E")    # relative_size=5.0 (default)
pv = NeuronType.make("PV")   # relative_size=1.0
```

---

### `Layer`

```python
@dataclass
class Layer:
    name: str
    neuron_types: Sequence[NeuronType] = ()
    geometry: Geometry3D = field(default_factory=Geometry3D)
    n_neurons: int = 0
```

One cortical layer. Attach `NeuronType` entries to declare which cell
populations live in the layer. `n_neurons` is the total neuron budget for
this layer (all cell types combined).

---

### `StaticParams`

```python
@dataclass
class StaticParams:
    g_mech: dict = field(default_factory=dict)          # mechanism → conductance
    reversal_potentials_mV: dict = field(default_factory=dict)  # mechanism → E_rev (mV)
    dT_ms: float = 0.1
    value_tag: ValueTag = "relative"
```

**Never plastic/trainable/gradientable.** Holds conductances, reversal
potentials, and the integration timestep for a connection.

> **Note:** `reversal_potentials_mV` is stored as metadata only. jaxfne's
> compiled edges are native current-based (Izhikevich-style) and have no
> conductance/reversal-potential term in the dynamics. The value is surfaced
> in `cfg.metadata["circuit"]["mechanisms"]` for provenance/inspection but
> does **not** affect `simulate()` output.

---

### `PlasticParams`

```python
@dataclass
class PlasticParams:
    w_mech: float = 1.0   # per-connection gain
    H: float = 0.0        # homeostatic H-factor
    value_tag: ValueTag = "relative"
```

**Trainable/gradientable.** `w_mech` is the per-connection gain matrix
(scalar here, expandable to a full per-synapse matrix). `H` is the homeostatic
H-factor (passive ionic charge income per neuron).

> `H` seeds the HDP controller's initial per-neuron state **only** when HDP
> is separately enabled (`Configuration.hdp(...)` before `jaxfne.construct`,
> or via a post-hoc `RuntimeConfig` override). It is stored but inert when
> HDP is disabled (jaxfne's default). See `construct_neuronal_tensor` for how
> per-neuron `H` is aggregated from all connections touching each neuron.

---

### `InterConnection`

```python
@dataclass
class InterConnection:
    source_layer: str
    source_neuron_type: str
    target_layer: str
    target_neuron_type: str
    mechanism: str          # required — no default (e.g. "AMPA", "GABA_A")
    static: StaticParams = field(default_factory=StaticParams)
    plastic: PlasticParams = field(default_factory=PlasticParams)
```

**Within-area** connection from `(source_layer, source_neuron_type)` to
`(target_layer, target_neuron_type)`. `mechanism` is always required.

---

### `Pose3D`

```python
@dataclass
class Pose3D:
    plane: Literal["xy", "xz", "yz"] = "xy"
    rotation_deg: float = 0.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    value_tag: ValueTag = "relative"
```

Controls where an `Area`'s layer stack sits in global 3D space.

| Field | Effect |
|-------|--------|
| `plane` | Which two global axes hold local (x, y) spread; the third holds local depth (z). `"xy"` is canonical (depth along z). |
| `rotation_deg` | Twist of the in-plane (x, y) spread around the depth axis. |
| `translation` | Global (x, y, z) offset of the entire area. |

**Plane → axis mapping:**

| `plane` | local x → global | local y → global | local depth → global |
|---------|-------------------|-------------------|----------------------|
| `"xy"` | axis 0 (x) | axis 1 (y) | axis 2 (z) |
| `"xz"` | axis 0 (x) | axis 2 (z) | axis 1 (y) |
| `"yz"` | axis 2 (z) | axis 0 (x) | axis 1 (y) |

---

### `Area`

```python
@dataclass
class Area:
    name: str
    layers: Sequence[Layer] = ()
    inter_connections: Sequence[InterConnection] = ()
    pose: Pose3D = field(default_factory=Pose3D)
```

One cortical area. Contains its `Layer` population structure, all
`InterConnection`s (within-area wiring), and a `Pose3D` for 3D placement.

---

### `AreaConnection`

```python
@dataclass
class AreaConnection:
    source_area: str
    source_layer: str
    source_neuron_type: str
    target_area: str
    target_layer: str
    target_neuron_type: str
    mechanism: str = "monotonic_cable_synapse"   # default
    static: StaticParams = field(default_factory=StaticParams)
    plastic: PlasticParams = field(default_factory=PlasticParams)
```

**Between-area** connection. Specifies the full
`(area, layer, neuron_type)` path for both source and target.
Unlike `InterConnection`, `mechanism` has a default of
`"monotonic_cable_synapse"`.

---

### `NeuronalTensor`

```python
@dataclass
class NeuronalTensor:
    areas: Sequence[Area] = ()
    area_connections: Sequence[AreaConnection] = ()
    name: str = "untitled"
```

The top-level container. Holds all areas and all between-area connections.

**Method:**

```python
def to_dict(self) -> dict
```

Returns a fully serialisable `dict` (via `dataclasses.asdict`). This is
what `save_neuronal_tensor` serialises to JSON.

---

## Functions

### `default_relative_size`

```python
def default_relative_size(neuron_type: str) -> float
```

Returns the default `relative_size` for a neuron type (from
`emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE`, shared with HDP `tau_i` scaling):
- `"E"` → `5.0`
- `"PV"` / `"Inl"` → `1.0`
- `"SST"` / `"VIP"` / `"Ing"` → `1.5`

Used internally by `NeuronType.make`.

---

### `save_neuronal_tensor`

```python
def save_neuronal_tensor(tensor: NeuronalTensor, path: str | Path) -> str
```

Serialise a `NeuronalTensor` to a JSON file. Built on `jaxfne.io.save_json`.

Returns the path written as a string.

> **Convention:** configs are data, never code. Build a `NeuronalTensor` once,
> then keep all variants as JSON files.

```python
save_neuronal_tensor(tensor, "circuits/v1_mt.json")
```

---

### `load_neuronal_tensor`

```python
def load_neuronal_tensor(path: str | Path) -> NeuronalTensor
```

Deserialise a `NeuronalTensor` from a previously saved JSON file.

```python
tensor = load_neuronal_tensor("circuits/v1_mt.json")
```

---

### `list_canonical_neuronal_tensors` / `load_canonical_neuronal_tensor`

```python
def list_canonical_neuronal_tensors() -> list[str]
def load_canonical_neuronal_tensor(name: str) -> NeuronalTensor
```

Package-shipped canonical circuits, stored as JSON under `jaxfne/configs/`
and loaded with the same `load_neuronal_tensor` deserializer above:

```python
names = list_canonical_neuronal_tensors()
# ['canonical-v1-column-1000n', 'canonical-v1-v4-pfc-multiarea', 'default-column',
#  'default_macaque_V1', 'homeostatic-h-override-demo', 'laminar-column-4layer',
#  'two-area-feedforward']
tensor = load_canonical_neuronal_tensor("canonical-v1-column-1000n")
```

`canonical-v1-column-1000n` is the verified ground-truth 1000-neuron V1 column
(6 layers L1-L6, full E/PV/SST/VIP composition, the same depth-graded E:I
gradient documented in the cortical-column-default skill — E peaks deep,
I peaks superficial). `canonical-v1-v4-pfc-multiarea` tiles that same column
composition across 3 areas (V1, V4, PFC; 3000 neurons total) with L4-to-L4
feedforward and L6-to-L1 feedback `AreaConnection`s, matching the
`jtfne.build_multi_area_columns(["V1","V4","PFC"], ei_profile="canonical")`
hierarchy already used elsewhere in the docs. The remaining four are smaller
synthetic demos for specific features (homeostatic H-override, multi-area
feedforward, a minimal 4-layer column) — use `canonical-v1-column-1000n` or
`canonical-v1-v4-pfc-multiarea` when you want a realistic reference composition
rather than a feature-isolation demo.

**On species-scaled variants:** jaxfne has no calibrated species-specific
connectivity or composition data — any "mouse"/"macaque"/"human" canonical
config would necessarily be an arbitrary neuron-count rescaling of the same
generic column template, not a calibrated biological difference. Per the
project's own claim-language rule (no biological overclaims without
evidence), no such configs are shipped; if you need one, build it explicitly
via `NeuronalTensor(...)` with a documented, named scale factor rather than
treating a renamed copy of `canonical-v1-column-1000n` as species-accurate.

---

### `merge_neuronal_tensors`

```python
def merge_neuronal_tensors(
    tensors: Sequence[NeuronalTensor],
    poses: Sequence[Pose3D] | None = None,
    *,
    name: str = "merged",
) -> NeuronalTensor
```

The **"unifier"**: concatenate several `NeuronalTensor` configs into one flat
list of areas. From the simulator's perspective N areas across M input tensors
become one `NeuronalTensor`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensors` | `Sequence[NeuronalTensor]` | Configs to merge, in order. |
| `poses` | `Sequence[Pose3D] \| None` | One `Pose3D` per area in flattened encounter order (area 0 of tensor 0, area 1 of tensor 0, …, area 0 of tensor 1, …). Must have exactly as many entries as the total area count across all tensors. If `None`, each area keeps its own declared `pose`. |
| `name` | `str` | Name for the merged `NeuronalTensor`. |

**Area name collision:** if two input tensors share an area name, the later
one is suffixed (`"V1"` → `"V1_1"`). All `AreaConnection` references within
that tensor are rewritten to match.

> **Cross-tensor `AreaConnection`s are NOT inferred.** Only connections
> declared explicitly on the result link areas from different inputs.

```python
merged = merge_neuronal_tensors(
    [v1_tensor, mt_tensor],
    poses=[Pose3D("xy"), Pose3D("xz", translation=(0.0, 2.0, 0.0))],
    name="v1_mt",
)
```

---

### `neuronal_tensor_to_configuration`

```python
def neuronal_tensor_to_configuration(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
) -> Configuration
```

Bridge a `NeuronalTensor` into the existing `construct` / `simulate` pipeline.
Returns a `Configuration` object suitable for `jaxfne.construct`.

**What is wired:**

- Per-area/per-layer cell-type fractions (`Configuration.area_layer_cell_types`),
  split evenly across declared `NeuronType` entries — unless every `NeuronType`
  in the layer declares a `fraction`, in which case those (normalized)
  fractions are used instead.
- Every `InterConnection` (within-area) and `AreaConnection` (between-area)
  compiled into a real selector-based edge rule via
  `Configuration.connections` + `Configuration.mechanisms`. Edge magnitude is
  `w_mech × g_mech / √total_n`. Sign follows the source neuron type
  (E → excitatory, else inhibitory). Connection probability is `1.0`
  (full bipartite between the declared layer × cell-type pair).

**Known fidelity gaps (not yet wired):**

| Gap | Workaround |
|-----|------------|
| `Layer.geometry` / `Area.pose` — 3D placement is dropped; positions come from jaxfne's default uniform-random column sampler. | Use `construct_neuronal_tensor` instead, which overwrites positions post-construct. |
| `StaticParams.reversal_potentials_mV` — stored as metadata only; no effect on dynamics. | Inspect via `cfg.metadata["circuit"]["mechanisms"]`. |
| `PlasticParams.H` — stored but inert unless HDP is separately enabled. | See `construct_neuronal_tensor` for automatic HDP seeding. |

---

### `construct_neuronal_tensor`

```python
def construct_neuronal_tensor(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
) -> Model
```

**Bridge + construct + apply each area's `Pose3D` placement in one call.**
This is the recommended entry point when you need pose-correct 3D placement.

**What it does beyond `neuronal_tensor_to_configuration`:**

1. Calls `neuronal_tensor_to_configuration` and then `jaxfne.construct`.
2. Samples each layer's local neuron positions from its declared `Geometry3D`
   (currently `"uniform_random"` only).
3. Applies each area's `Pose3D` (plane + rotation + translation) to map
   local positions into global 3D space.
4. Overwrites `model.params["positions"]` and
   `model.static["neuron_metadata"]` so field/LFP/EEG/MEG proxy readouts
   — which read positions from there — see the real layout.
5. Aggregates `PlasticParams.H` from every connection touching each neuron
   (mean across all connections whose target layer/cell_type matches) and
   applies via `Model.with_hdp_initial_state`. Untouched neurons default
   to `H=1.0` (HDP equilibrium). This is stored but inert unless HDP is
   separately enabled.

```python
model = construct_neuronal_tensor(tensor, seed=42, duration_ms=500.0)
signals = jaxfne.simulate(model)
```

---

## End-to-end example

```python
import jaxfne
from jaxfne import (
    NeuronalTensor, Area, AreaConnection, Layer, NeuronType,
    Geometry3D, Pose3D, StaticParams, PlasticParams, InterConnection,
    save_neuronal_tensor, load_neuronal_tensor, construct_neuronal_tensor,
)

# ── Define cell types ────────────────────────────────────────────────────────
E  = NeuronType.make("E")    # relative_size=5.0 (default)
PV = NeuronType.make("PV")   # relative_size=1.0

# ── Define layers ───────────────────────────────────────────────────────────
L4  = Layer("L4",  [E, PV], Geometry3D(z_range=(0.0, 0.3)), n_neurons=80)
L23 = Layer("L2/3",[E, PV], Geometry3D(z_range=(0.3, 0.8)), n_neurons=60)

# ── Within-area wiring ──────────────────────────────────────────────────────
v1_connections = [
    InterConnection("L4", "E", "L2/3", "E",  mechanism="AMPA"),
    InterConnection("L4", "E", "L4",   "PV", mechanism="AMPA"),
    InterConnection("L4", "PV","L4",   "E",  mechanism="GABA_A"),
]

# ── Assemble area ───────────────────────────────────────────────────────────
v1 = Area("V1", layers=[L4, L23], inter_connections=v1_connections,
          pose=Pose3D("xy"))

# ── Build tensor ────────────────────────────────────────────────────────────
tensor = NeuronalTensor(areas=[v1], name="v1_minimal")

# ── Save / reload ───────────────────────────────────────────────────────────
save_neuronal_tensor(tensor, "v1_minimal.json")
tensor = load_neuronal_tensor("v1_minimal.json")

# ── Construct and simulate ──────────────────────────────────────────────────
model   = construct_neuronal_tensor(tensor, seed=0, duration_ms=500.0)
signals = jaxfne.simulate(model)
```

---

## Multi-area example with `merge_neuronal_tensors`

```python
from jaxfne import merge_neuronal_tensors, Pose3D

# v1_tensor and mt_tensor built separately (each a NeuronalTensor)
merged = merge_neuronal_tensors(
    [v1_tensor, mt_tensor],
    poses=[
        Pose3D("xy",                     translation=(0.0, 0.0, 0.0)),
        Pose3D("xy", rotation_deg=45.0,  translation=(3.0, 0.0, 0.0)),
    ],
    name="v1_mt",
)

# Wire areas together
from jaxfne import AreaConnection
merged.area_connections = [
    AreaConnection("V1", "L2/3", "E", "MT", "L4", "E"),  # feedforward
    AreaConnection("MT", "L2/3", "E", "V1", "L2/3", "E"),  # feedback
]

model = construct_neuronal_tensor(merged, seed=1, duration_ms=1000.0)
```
