# Tutorial Environment Setup

**Status:** v0.3.0 tutorial environment guide  
**truth_mode:** truth_safe_unverified

---

## Quick Start

Install the minimal environment for running v0.3 tutorials:

```bash
pip install jaxfne==0.2.30 jax jaxlib numpy matplotlib
```

Or use the dedicated requirements file:

```bash
pip install -r requirements-v030-tutorials.txt
```

---

## Core Dependencies

### Required

| Package | Version | Purpose |
|---------|---------|---------|
| **jaxfne** | 0.2.30 | Core TFNE forward-field framework (stable toolbox) |
| **jax** | ≥0.4.0 | JAX computation engine (jit, vmap, autodiff) |
| **jaxlib** | ≥0.4.0 | JAX backend and primitives |
| **numpy** | ≥1.24.0 | Array operations and numerical tools |
| **scipy** | ≥1.9.0 | Scientific computing (interpolation, linear algebra) |
| **matplotlib** | ≥3.7.0 | Static figure generation (PNG, high-resolution) |

### Optional

| Package | Version | Purpose | When to install |
|---------|---------|---------|-----------------|
| **plotly** | ≥5.0.0 | Interactive HTML figures | When generating interactive Plotly artifacts |
| **optax** | ≥0.2.0 | JAX optimization library | For v0.3.25+ optimization tutorials (future) |
| **jaxley** | ≥0.0.10 | Compartmental neuron simulator | For v0.3.20+ Jaxley bridge tutorials (future) |

---

## Installation Methods

### Method 1: Using requirements file (Recommended)

```bash
pip install -r requirements-v030-tutorials.txt
```

This installs all core dependencies and optional extras (Plotly) by default.

### Method 2: Minimal installation (JAX only)

```bash
pip install jaxfne==0.2.30 jax jaxlib numpy matplotlib
```

PNG figures will work. Plotly interactive artifacts will gracefully degrade to PNG fallback.

### Method 3: Full installation with Plotly and Optax

```bash
pip install jaxfne==0.2.30 jax jaxlib numpy scipy pandas matplotlib plotly optax
```

All interactive features enabled. Suitable for development and advanced tutorials.

### Method 4: With Jaxley bridge (future)

When starting Jaxley integration tutorials (v0.3.20+):

```bash
pip install jaxfne==0.2.30 jaxley matplotlib
```

(Jaxley may pull in its own JAX/NumPy dependencies automatically.)

---

## Optional Dependencies

### Plotly for Interactive Artifacts

Plotly is **optional** and gracefully handled:

- **If Plotly is installed:** Tutorials generate interactive HTML artifacts alongside static PNG figures
- **If Plotly is absent:** Tutorials fall back to static PNG only (no error)
- **PNG is always generated:** Static PNG figures do not require Plotly

To add Plotly support:

```bash
pip install plotly>=5.0.0
```

### Optax for Optimization

Optax is **optional** and only needed for optimization tutorials (planned for v0.3.25+):

```bash
pip install optax>=0.2.0
```

Core jaxfne does not depend on Optax. Custom GSDR/AGSDR optimizers are still available without it.

### Jaxley for Bridge Tutorials

Jaxley is **optional** and only needed for compartmental neuron tutorials (planned for v0.3.20+):

```bash
pip install jaxley>=0.0.10
```

Core jaxfne does not depend on Jaxley. The bridge is an adapter layer only.

---

## Verification

### Check jaxfne imports

```python
import jaxfne as jtfne
print(f"jaxfne version: {jtfne.__version__}")

import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
```

### Check Plotly (optional)

```python
try:
    import plotly
    print(f"Plotly version: {plotly.__version__}")
    print("✓ Plotly available for interactive figures")
except ImportError:
    print("✗ Plotly not installed; PNG figures will work fine")
```

### Run a simple tutorial

```bash
cd notebooks/v030
jupyter notebook 01_single_neuron.ipynb
```

---

## Constraints and Design Decisions

### Core jaxfne is independent

- ✓ `import jaxfne as jtfne` works without Plotly, Optax, or Jaxley
- ✓ Top-level imports never require optional dependencies
- ✓ Guarded imports (try/except) used for all optional packages

### PNG generation does not require Plotly

- ✓ Static PNG figures use only matplotlib (in requirements)
- ✓ Plotly HTML is generated from source data if available
- ✓ No PNG-to-Plotly conversion (data integrity maintained)

### JAX version flexibility

- ✓ JAX ≥0.4.0 supported (v0.4.0 through latest v0.10+)
- ✓ CPU-safe execution (no GPU required)
- ✓ pmap with CPU fallback pattern functional

### Python version

- Requires Python ≥3.10 (inherited from jaxfne core)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'plotly'"

This is expected and not an error. PNG figures will still be generated.

**Solution:** Skip interactive tutorials or install Plotly:
```bash
pip install plotly>=5.0.0
```

### "ImportError: cannot import name 'simulate'"

Check that you're importing from the correct jaxfne API:

```python
# Correct:
import jaxfne as jtfne
signals = jtfne.simulation(...)

# Incorrect:
from jaxfne.core import simulate  # This API changed in v0.2.30
```

### JAX device mismatch

If tutorials slow down unexpectedly:

```python
import jax
print(jax.devices())  # Check available devices
# If only CPU, that's normal and expected
```

### Numerical NaN/Inf in simulation

Check acceptance gates:

```python
import jaxfne as jtfne

# Tutorial must satisfy:
# - Firing rate: 2–25 Hz per population
# - Voltages: finite (not NaN/Inf)
# - All outputs JSON-safe
```

---

## Citation and Truth Status

These tutorials are part of **jaxfne v0.2.30** stable toolbox:

```bibtex
@software{jaxfne_v0230,
  title={jaxfne: JAX-native TFNE workflows for reproducible computational neurophysiology},
  author={[author]},
  year={2026},
  url={https://github.com/HNXJ/jaxfne}
}
```

**Truth status:**
- **claim_level:** computational_scaffold
- **truth_mode:** truth_safe_unverified
- **physical_amplitude_claim_allowed:** False
- **biological_metabolism_claim_allowed:** False
- **field_solver_status:** laminar_proxy_no_pde

These tutorials teach computational concepts, not biological facts.

---

**Last updated:** 2026-05-23  
**Stability:** v0.3.0 scaffold infrastructure
