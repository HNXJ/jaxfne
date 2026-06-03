# v0.3 Canonical Imports Guide

**Version:** v0.3.0+  
**Last updated:** 2026-05-23  
**truth_mode:** truth_safe_unverified  

---

## The Rule

**All v0.3 tutorial notebooks must use this import alias:**

```python
import jaxfne as jtfne
```

**This is not optional.** Consistency across all tutorials ensures readers encounter a uniform interface.

---

## Required Alias: `jtfne`

### Why `jtfne`?

- **`jtfne`** = **j**ax **TFNE** (Tensor-Field Neural Equations)
- Short, pronounceable, unambiguous
- Avoids confusion with other JAX frameworks
- Visually distinct in code (readers see `jtfne.` prefix consistently)

### The import statement

```python
import jaxfne as jtfne
```

That's it. No alternatives.

---

## Forbidden Aliases (DO NOT USE)

❌ `import jaxfne`  
*(bare import; must use alias)*

❌ `from jaxfne import *`  
*(pollutes namespace; hides source of each function)*

❌ `import jaxfne as tfne`  
*(sounds like "TensorFlow"; confusing)*

❌ `import jaxfne as jtnfe`  
*(typo-like; hard to remember)*

❌ `import jaxfne as jtFNE`  
*(inconsistent case; hurts readability)*

❌ `import jaxfne as fn` or `import jaxfne as j`  
*(too vague; readers won't know what `j` or `fn` refers to)*

---

## Correct Usage Pattern

### Import structure for v0.3 tutorials

```python
# Standard scientific Python stack
import jax
import jax.numpy as jnp
from jax import vmap, jit, random

import numpy as np
import matplotlib.pyplot as plt
import json

# jaxfne: ALWAYS use 'jtfne' alias
import jaxfne as jtfne
from jtfne.emitters import IzhikevichParams, simulate_eig_izhikevich
from jtfne.fields import project_laminar_sources, compute_conservation_proxy_diagnostics
from jtfne.io import json_safe, save_json, manifest

# Optional: Plotly (with guarded import)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
```

### Usage throughout notebook

All references to jaxfne functions and classes use the `jtfne.` prefix:

```python
# ✓ Correct usage with canonical alias
config = jtfne.configuration()
model = jtfne.construct(config, emitters=[...])
signals = model.simulation(duration_ms=100, dt_ms=0.1)
manifest = model.manifest(signals)

# ✓ Direct imports can skip prefix (but do so sparingly)
from jtfne.emitters import IzhikevichParams
params = IzhikevichParams(a=0.02, b=0.2, c=-65, d=8)

# ✓ Specific function imports can drop prefix
from jtfne.io import json_safe
output_dict = json_safe(manifest)
```

---

## Import Checklist for Tutorial Authors

Before submitting a notebook, verify:

- [ ] First jaxfne import uses `import jaxfne as jtfne`
- [ ] No bare `import jaxfne` (without alias)
- [ ] No `from jaxfne import *` (wildcard)
- [ ] All `jtfne.*` references are consistent (no mixed `jaxfne.` or `tfne.*` calls)
- [ ] Optional imports (Plotly, etc.) use try/except guards
- [ ] Import order: stdlib → JAX → jaxfne → optional (as shown above)

---

## Why Canonical Imports Matter

### Readability

Readers can instantly recognize `jtfne.` prefix in code examples. They know:
- `jtfne.configuration()` is from jaxfne
- `jnp.array()` is from JAX
- `plt.plot()` is from matplotlib

Mixed aliases (`jaxfne`, `tfne`, etc.) confuse readers and make tutorials harder to follow.

### Consistency

All v0.3 tutorials look the same at the import level. Readers can copy-paste code patterns across notebooks without surprise failures.

### Search and documentation

Readers searching for "jtfne.configuration" in tutorials will find all relevant examples.
Readers searching for "jaxfne.configuration" might miss results if some notebooks use `tfne.configuration()`.

### Community standards

Canonical import conventions are standard practice in mature Python libraries (e.g., `import numpy as np`, `import pandas as pd`). This establishes jaxfne as a professional, well-designed library.

---

## Common Mistakes and Fixes

### Mistake 1: Bare import without alias

```python
# ❌ Wrong
import jaxfne
config = jaxfne.configuration()

# ✓ Correct
import jaxfne as jtfne
config = jtfne.configuration()
```

**Fix:** Add `as jtfne` to the import statement.

### Mistake 2: Wildcard import

```python
# ❌ Wrong
from jaxfne import *
config = configuration()  # Where does 'configuration' come from? Unclear.

# ✓ Correct
import jaxfne as jtfne
config = jtfne.configuration()  # Clear: comes from jaxfne
```

**Fix:** Replace `from jaxfne import *` with `import jaxfne as jtfne`.

### Mistake 3: Mixed aliases in same notebook

```python
# ❌ Wrong (inconsistent)
import jaxfne as jtfne
from jaxfne import configuration

config1 = jtfne.configuration()  # Using prefix
config2 = configuration()        # Using direct import

# ✓ Correct (consistent)
import jaxfne as jtfne

config1 = jtfne.configuration()
config2 = jtfne.configuration()
```

**Fix:** Choose one pattern (preferably `jtfne.` prefix) and stick with it.

### Mistake 4: Abbreviated or variant aliases

```python
# ❌ Wrong (confusing)
import jaxfne as j
import jaxfne as fn
import jaxfne as tfne

# ✓ Correct
import jaxfne as jtfne
```

**Fix:** Always use `jtfne` exactly.

### Mistake 5: Forgetting to alias optional imports

```python
# ❌ Wrong (no guard; crashes if Plotly absent)
import plotly.graph_objects as go
fig = go.Figure()  # Error if plotly not installed

# ✓ Correct (guarded)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

# Later: check before using
if PLOTLY_AVAILABLE:
    fig = go.Figure()
    fig.write_html('output.html')
```

**Fix:** Always guard optional imports with try/except.

---

## Enforcement in v0.3 CI/CD

The v0.3 CI/CD pipeline will include a linter check:

```bash
# Check for canonical import usage
grep -n "import jaxfne" tutorial.ipynb \
  | grep -v "import jaxfne as jtfne" \
  && echo "ERROR: Non-canonical import found" && exit 1

echo "✓ All imports are canonical (jtfne)"
```

Tutorials that don't use `import jaxfne as jtfne` will **fail CI** and not be merged.

---

## Reference: Module Structure

For clarity, here's what you can import from jaxfne:

```python
# Top-level API (typically accessed via jtfne.*)
import jaxfne as jtfne

jtfne.configuration()      # Create default config
jtfne.construct()          # Build model from config + emitters
jtfne.default_basis_spec() # Get immutable basis spec

# Submodules (can import directly if preferred, but use jtfne.* when possible)
from jtfne.emitters import (
    IzhikevichParams,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
)

from jtfne.fields import (
    project_laminar_sources,
    compute_conservation_proxy_diagnostics,
)

from jtfne.io import (
    json_safe,
    save_json,
    manifest,
    sha256_file,
)
```

---

## See Also

- [Tutorial Template](template.md) — Section 4 (Canonical Import)
- [v0.3 Tutorial-Scenario Plan](../v030_tutorial_scenario_plan.md) — Canonical import doctrine
- [README](README.md) — v0.3 overview
