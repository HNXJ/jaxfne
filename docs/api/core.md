# Core API

Main classes for configuration, model construction, simulation, and signal handling.

## Configuration

```python
jaxfne.Configuration()
```

Declarative TFNE model configuration. Methods return new objects, enabling immutable, chainable construction.

### Methods

#### `runtime(**kwargs) -> Configuration`

Set runtime/simulation metadata in chainable form.

**Parameters:**
- `seed` (int): Random seed for reproducible PRNG
- `dtype` (str): JAX dtype, e.g., `"float32"` or `"float64"`
- `duration_ms` (float): Total simulation duration in milliseconds
- `dt_ms` (float): Timestep in milliseconds

**Returns:** Updated `Configuration`

**Example:**
```python
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
```

#### `column(name: str, layers: list, n: int) -> Configuration`

Declare one cortical column with specified layers and neuron count.

**Parameters:**
- `name` (str): Unique column name
- `layers` (list[str]): Layer labels (e.g., `["L2/3", "L4", "L5"]`)
- `n` (int): Number of neurons in the column

**Returns:** Updated `Configuration`

**Example:**
```python
cfg = cfg.column("V1", layers=["L2/3", "L4", "L5"], n=100)
```

#### `cell_types(cell_type_map: dict) -> Configuration`

Define the proportion or count of excitatory/inhibitory cell types.

**Parameters:**
- `cell_type_map` (dict): E/I distribution, e.g., `{"E": 0.8, "I": 0.2}` or `{"E": 80, "I": 20}`

**Returns:** Updated `Configuration`

**Example:**
```python
cfg = cfg.cell_types({"E": 0.8, "I": 0.2})
```

#### `connectivity(**kwargs) -> Configuration`

Declare synaptic connectivity parameters (recurrence, plasticity, etc.).

**Returns:** Updated `Configuration`

**Example:**
```python
cfg = cfg.connectivity()
```

#### `set_emitter(family: str, preset: str) -> Configuration`

Set the neuron model family and parameter preset.

**Parameters:**
- `family` (str): Emitter family (e.g., `"izhikevich"`)
- `preset` (str): Parameter preset (e.g., `"cortical_eig"`, `"tonic_spiking"`)

**Returns:** Updated `Configuration`

**Example:**
```python
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
```

#### `probes(modes: list, **kwargs) -> Configuration`

Declare multimodal proxy probe modes (SPK, Vm, LFP-proxy, CSD-proxy, etc.).

**Parameters:**
- `modes` (list[str]): Probe modes, e.g., `["MUA-proxy", "LFP-proxy", "CSD-proxy"]`
- `name` (str, optional): Probe name (default: `"multimodal_probe"`)

**Returns:** Updated `Configuration`

**Example:**
```python
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])
```

---

## Model

```python
jaxfne.Model
```

Constructed TFNE workflow ready for simulation. Created via `jaxfne.construct(cfg)`.

### Attributes

- `cfg` (Configuration): Source configuration
- `geometry` (LaminarSourceGeometry): Spatial geometry
- `basis_spec` (BasisSpec): Basis function specification

### Methods

#### `simulate(signals_or_params, ...) -> Signals`

Run the neural simulation.

**Parameters:**
- `duration_ms` (float): Simulation duration
- `dt_ms` (float): Timestep
- `seed` (int, optional): Random seed

**Returns:** `Signals` object containing spikes, voltage, and field readouts

**Example:**
```python
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

#### `compute_readout(signals, readout_specs) -> ReadoutResult`

Compute metrics from signals according to specifications.

**Parameters:**
- `signals` (Signals): Output from simulation
- `readout_specs` (list[ReadoutSpec]): Metric specifications

**Returns:** `ReadoutResult` with computed metrics

**Example:**
```python
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz"),
    jtfne.readout_spec("voltage", "mean_V_m")
])
```

---

## Simulation

```python
jaxfne.Simulation
```

Simulation run parameters.

### Attributes

- `duration_ms` (float): Total duration in milliseconds
- `dt_ms` (float): Timestep in milliseconds
- `seed` (int, optional): Random seed

**Example:**
```python
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.1, seed=7)
```

---

## Signals

```python
jaxfne.Signals
```

Neural signals output from simulation.

### Attributes

- `V_m` (jax.Array): Membrane voltage [time, neurons]
- `spikes` (jax.Array): Spike raster [time, neurons]
- `I_syn` (jax.Array, optional): Synaptic current
- `source` (jax.Array, optional): Source density [time, locations]
- `LFP` (jax.Array, optional): Local field potential [time, locations]
- `CSD` (jax.Array, optional): Current source density [time, locations]
- `EEG` (jax.Array, optional): EEG proxy [time, channels]
- `MEG` (jax.Array, optional): MEG proxy [time, channels]
- `EMM` (jax.Array, optional): Metabolic proxy [time]

### Methods

#### `to_dict() -> dict`

Convert signals to JSON-safe dictionary.

#### `save_json(path: str)`

Save signals to JSON file.

**Example:**
```python
signals.save_json("output.json")
```

---

## ReadoutSpec

```python
jaxfne.readout_spec(name: str, metric: str, **kwargs)
```

Specification for a single readout metric.

**Parameters:**
- `name` (str): Human-readable name
- `metric` (str): Metric key (see [Probe Operators](probes.md))

**Returns:** `ReadoutSpec` object

**Available metrics:**
- `spike_rate_hz`: Mean firing rate in Hz
- `mean_V_m`: Mean membrane voltage (mV)
- `mean_source`: Mean source density
- `mean_LFP`: Mean LFP amplitude
- `mean_CSD`: Mean CSD amplitude
- `max_spike_rate_hz`: Maximum firing rate
- `burst_frequency_hz`: Burst frequency estimate

**Example:**
```python
readout = jtfne.readout_spec("firing_rate", "spike_rate_hz")
```

---

## ReadoutResult

```python
jaxfne.ReadoutResult
```

Container for computed readout metrics.

### Attributes

- `specs` (list[ReadoutSpec]): Original specifications
- `results` (dict[str, float]): Computed metric values

### Methods

#### `to_dict() -> dict`

Convert results to JSON-safe dictionary.

**Example:**
```python
results_dict = readout.to_dict()
```

---

## Objective

```python
jaxfne.objective(name: str, metric: str, target: float, **kwargs)
```

Optimization objective specification.

**Parameters:**
- `name` (str): Objective name
- `metric` (str): Target metric (from ReadoutSpec)
- `target` (float): Target value
- `weight` (float, optional): Loss weight in multi-objective optimization

**Returns:** `Objective` object

**Example:**
```python
obj = jtfne.objective(name="spike_rate", metric="spike_rate_hz", target=10.0)
```

---

## Helper Functions

### `construct(cfg: Configuration) -> Model`

Build a compiled Model from Configuration.

**Parameters:**
- `cfg` (Configuration): Declarative configuration

**Returns:** Compiled `Model` ready for simulation

**Example:**
```python
model = jtfne.construct(cfg)
```

### `simulate(model: Model, duration_ms: float, dt_ms: float, seed: int) -> Signals`

Run simulation on a Model.

**Parameters:**
- `model` (Model): Constructed model
- `duration_ms` (float): Duration in milliseconds
- `dt_ms` (float): Timestep in milliseconds
- `seed` (int): Random seed

**Returns:** `Signals` object

**Example:**
```python
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

### `simulation(**kwargs) -> Simulation`

Create a Simulation object.

**Example:**
```python
sim = jtfne.simulation(duration_ms=1000.0, dt_ms=0.1, seed=7)
```

### `configuration() -> Configuration`

Create a new Configuration object.

**Example:**
```python
cfg = jtfne.configuration()
```
