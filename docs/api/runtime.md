# Runtime API

Runtime configuration and execution helpers for jaxfne.

## Overview

The `jaxfne.runtime` module provides a stable import surface for runtime-related configuration objects used when executing simulations and computing readouts.

```python
from jaxfne.runtime import RuntimeConfig
```

## RuntimeConfig

Execution backend and device settings for simulations.

See [jaxfne.core.RuntimeConfig][] for detailed API.

### Example

```python
import jaxfne as jtfne
from jaxfne.runtime import RuntimeConfig

# Create default runtime config
cfg = RuntimeConfig()

# Use in simulation
sim = jtfne.Simulation(duration_ms=100.0)
model = jtfne.construct(jtfne.configuration().network(n=8).emitter(...).field(...))
signals = model.simulate(sim)
```

## Exported objects

| Object | Purpose |
|--------|---------|
| `RuntimeConfig` | Execution backend and device configuration |

## See also

- [Core API](core.md) — Main jaxfne classes
- [Quickstart](../quickstart.md) — Getting started guide
