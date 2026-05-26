# Emitters API

Neuron models and emitter implementations for neural dynamics simulation.

## Izhikevich Model

```python
jaxfne.set_emitter("izhikevich", preset="cortical_eig")
```

The Izhikevich neuron model is a phenomenological spiking neuron model with two state variables (v, u). It provides a good balance between computational efficiency and biological realism for tutorial-scale simulations.

### State Variables

- `v` (mV): Membrane potential (-90 to 30 mV typical range)
- `u` (µA): Membrane recovery variable (adaptation/refractory effects)

### Dynamics

```
dv/dt = 0.04*v² + 5*v + 140 - u + I
du/dt = a*(b*v - u)
```

If v ≥ 30 mV, spike occurs and states reset:
```
v ← c
u ← u + d
```

### Parameters

Parameters are organized in `IzhikevichParams` dataclass:

- `a` (float): Recovery timescale (0.02 typical)
- `b` (float): Coupling strength (0.2 typical)
- `c` (float): Reset voltage (-65 mV typical)
- `d` (float): Recovery reset (6 µA typical)
- `I_injected` (float): Injected current baseline (optional)

### Presets

Available presets define different neuron behaviors:

| Preset | Behavior | Use Case |
|--------|----------|----------|
| `"cortical_eig"` | Regular spiking, moderate firing | Tutorial default |
| `"tonic_spiking"` | Continuous firing | Sustained activity |
| `"phasic_spiking"` | Burst then adapt | Transient responses |
| `"fast_spiking"` | High-frequency spikes | Inhibitory interneurons |
| `"chattering"` | Burst patterns | Bursting neurons |

**Example:**
```python
cfg = jtfne.Configuration()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
```

---

## IzhikevichParams

```python
jaxfne.IzhikevichParams
```

Parameter container for Izhikevich neuron model.

### Attributes

- `a` (float): Recovery timescale
- `b` (float): Coupling strength
- `c` (float): Reset voltage (mV)
- `d` (float): Recovery reset (µA)

### Methods

#### `class_method(class_label: str) -> IzhikevichParams`

Create parameters for a specific neuron class.

**Parameters:**
- `class_label` (str): Neuron class (e.g., "excitatory", "inhibitory", "fast_spiking")

**Returns:** `IzhikevichParams` with preset values

**Example:**
```python
exc_params = jtfne.IzhikevichParams.class_method("excitatory")
```

---

## ReceptorSpec

```python
jaxfne.ReceptorSpec
```

Specification for synaptic receptor types and kinetics.

### Attributes

- `name` (str): Receptor name (e.g., "AMPA", "NMDA", "GABA_A")
- `tau_decay` (float): Time constant (ms)
- `reversal` (float): Reversal potential (mV)
- `conductance_scaling` (float): Weight factor

### Standard Receptors

Access via `jaxfne.standard_receptor_specs()`:

- **AMPA:** Fast excitatory (1–2 ms)
- **NMDA:** Slow excitatory (100 ms)
- **GABA_A:** Fast inhibitory (5–10 ms)
- **GABA_B:** Slow inhibitory (100–200 ms)

**Example:**
```python
receptors = jtfne.standard_receptor_specs()
```

---

## SynapseSpec

```python
jaxfne.SynapseSpec
```

Specification for synaptic connections between neurons.

### Attributes

- `source_idx` (int): Source neuron index
- `target_idx` (int): Target neuron index
- `receptor` (str): Receptor type (e.g., "AMPA", "GABA_A")
- `weight` (float): Synaptic strength (conductance)
- `delay` (float, optional): Transmission delay (ms)

---

## EIGNetwork

```python
jaxfne.EIGNetwork
```

Excitatory-Inhibitory-Gap junction network representation.

### Attributes

- `n_exc` (int): Number of excitatory neurons
- `n_inh` (int): Number of inhibitory neurons
- `synapse_specs` (list[SynapseSpec]): Connection list
- `gap_specs` (list[SynapseSpec], optional): Gap junction connections

### Methods

#### `to_dense() -> dict`

Convert to dense adjacency matrix representation.

**Returns:** Dictionary with connectivity matrices

**Example:**
```python
network = jtfne.EIGNetwork(n_exc=100, n_inh=20)
dense = network.to_dense()
```

---

## EdgeList

```python
jaxfne.EdgeList
```

Sparse edge list representation of network connectivity.

### Attributes

- `edges` (jax.Array): Connection indices [2, num_edges]
- `weights` (jax.Array): Synaptic weights [num_edges]

### Methods

#### `from_dense(adj_matrix: jax.Array) -> EdgeList`

Create EdgeList from dense adjacency matrix.

**Parameters:**
- `adj_matrix` (jax.Array): Dense connectivity matrix

**Returns:** `EdgeList`

**Example:**
```python
dense = jnp.ones((100, 100))
edges = jtfne.EdgeList.from_dense(dense)
```

---

## Emitter Functions

### `simulate_eig_izhikevich(cfg, I_ext, seed) -> (spikes, V_m, u)`

Simulate Excitatory-Inhibitory-Gap junction network with Izhikevich dynamics.

**Parameters:**
- `cfg` (Configuration): Network configuration
- `I_ext` (jax.Array): External input current [time, neurons]
- `seed` (int): Random seed

**Returns:** Tuple of (spikes, membrane voltage, recovery variable)

**Example:**
```python
spikes, V_m, u = jtfne.simulate_eig_izhikevich(cfg, I_ext, seed=7)
```

### `make_eig_network(n_exc, n_inh, ...) -> EIGNetwork`

Create an Excitatory-Inhibitory-Gap junction network structure.

**Parameters:**
- `n_exc` (int): Number of excitatory neurons
- `n_inh` (int): Number of inhibitory neurons
- Additional connectivity parameters

**Returns:** `EIGNetwork` object

**Example:**
```python
network = jtfne.make_eig_network(n_exc=80, n_inh=20)
```

### `standard_receptor_specs() -> dict[str, ReceptorSpec]`

Get standard synaptic receptor specifications.

**Returns:** Dictionary of receptor type → `ReceptorSpec`

**Available types:** AMPA, NMDA, GABA_A, GABA_B

**Example:**
```python
receptors = jtfne.standard_receptor_specs()
print(receptors["AMPA"])
```

---

## Scope Notes

- **Izhikevich model is phenomenological:** Not a detailed Hodgkin-Huxley model; suitable for tutorial and prototyping workflows
- **Spike threshold (v ≥ 30 mV):** Fixed threshold; not voltage-dependent channels
- **Synaptic kinetics:** Exponential decay; suitable for learning-scale simulations
- **Network connectivity:** Declared via configuration; not learned (unless using optimization workflow)
