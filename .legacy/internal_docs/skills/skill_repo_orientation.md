# Skill: Repository Orientation

**When to use**: You're new to jaxfne or need to understand how the code is organized.

**Goal**: Navigate the codebase, find key files, understand the pipeline.

## Package Layout

```
jaxfne/
├── __init__.py              # Public API exports
├── core.py                  # Configuration, Model, Simulation classes
├── runtime.py               # RuntimeConfig (execution backend/device settings)
├── emitters/
│   ├── izhikevich.py       # Spiking neuron model
│   ├── hh.py               # Hodgkin-Huxley model (if present)
│   ├── receptor_kinetics.py # Synaptic receptor models
│   └── ...
├── tfne/
│   ├── sources.py          # Source geometry and projection
│   ├── fields.py           # Field solvers, CSD/LFP, probe operators
│   ├── objectives.py       # Loss functions, optimization targets
│   └── ...
├── optim/
│   ├── gsdr.py             # GSDR/AGSDR custom optimizers
│   ├── optax_adapters.py   # Optax integration (optional)
│   └── ...
├── io.py                   # I/O utilities, JSON safety
├── presets.py              # Cell type, receptor, stimulus presets
└── probes/ (or fields.py)  # Probe operator contract
```

## The Core Pipeline

jaxfne follows this data flow:

```
Configuration → Model → Simulation → Signals → Readouts
                                   ↓
                            Probe Operators
```

### 1. Configuration
**Files**: `core.py`

Create a network description:
```python
import jaxfne as jtfne

cfg = jtfne.configuration()
    .network(n=8)
    .emitter(...)  # Izhikevich, HH, etc.
    .field(...)    # TFNE field solver
```

### 2. Model
**Files**: `core.py`

Build the computational graph:
```python
model = jtfne.construct(cfg)
```

### 3. Simulation
**Files**: `core.py`

Define run parameters:
```python
sim = jtfne.simulation(duration_ms=1000.0)
```

### 4. Signals
**Files**: `tfne/sources.py`, `tfne/fields.py`

Run the simulation and get neural outputs (spikes, voltage, field):
```python
signals = model.simulate(sim)
```

### 5. Readouts
**Files**: `fields.py` (probe operators)

Measure specific quantities (spike rate, voltage, CSD, LFP, etc.):
```python
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("spike_rate_hz", ...)
])
```

## Examples Map

| Example | Focus | Size |
|---------|-------|------|
| `examples/00_minimal_column.py` | Quick start | Small |
| `examples/02_spectrolaminar_oddball_scaffold.py` | Laminar cortex oddball | Medium |
| `examples/03_single_neuron_multimodal_probe.py` | Probes (spikes, voltage) | Small |
| `examples/04_two_neuron_ei_multimodal.py` | E/I balance | Medium |
| `examples/05_network_100_ei_multimodal.py` | 100-neuron network | Large |

## Tests Map

| Test File | Purpose |
|-----------|---------|
| `tests/test_probe_operators_v021.py` | Probe report contract (JSON safety, terminology) |
| `tests/test_probe_report_contract_v0212.py` | v0.2.12 contract hardening |
| `tests/test_*.py` (others) | Core functionality, JAX compatibility |

**To run all tests:**
```bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=short
```

**To run one test file:**
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_probe_operators_v021.py -v
```

## Where to Add Code

### New neuron model?
→ `jaxfne/emitters/` (create new file or add to existing)

### New field solver or probe operator?
→ `jaxfne/fields.py` or `jaxfne/tfne/fields.py`

### New optimization algorithm?
→ `jaxfne/optim/` (create new file)

### New utility function?
→ `jaxfne/io.py` or appropriate submodule

### New test?
→ `tests/test_*.py` (use existing pattern)

### New example?
→ `examples/` (follow naming: `NN_description.py`)

## Branch and Test Hygiene

**Branch policy** (development):
- Main branch: stable releases only
- `dev` branch: integration branch for features
- Feature branches: `feat/v0212-*` for feature work

**Before committing:**
```bash
# Compile check
python -m py_compile jaxfne/fields.py jaxfne/emitters/izhikevich.py

# Run affected tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_probe_* -q

# No secrets check
grep -r 'api_key\|secret\|token\|password' jaxfne/ || echo "clean"
```

**After committing:**
```bash
# Run full suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line

# Ensure examples still work
python examples/00_minimal_column.py
python examples/02_spectrolaminar_oddball_scaffold.py
```

## Common Tasks

| Task | Command |
|------|---------|
| Check version | `python -c "import jaxfne; print(jaxfne.__version__)"` |
| Compile syntax | `python -m py_compile jaxfne/fields.py` |
| Run tests | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q` |
| Build docs | `python -m mkdocs build --strict` |
| Run one example | `python examples/00_minimal_column.py` |
| Check imports | `python -c "import jaxfne; print(dir(jaxfne))"` |

## Next Steps

- Explore a small example: `python examples/00_minimal_column.py`
- Read [Probe Reports Skill](skill_probe_reports.md) for readout validation
- Check `docs/probe_operators.md` for operator reference
- Look at `tests/` for usage patterns
