# jaxfne-evidence-validator

**Triggers:** definition of done, manifest, validation report, finite outputs, evidence.

**Purpose:** Convert "it ran" into honest evidence before reporting success.

**Gates:** explicit seed, `float32` default, finite arrays, plausible rates, JSON-safe manifests, PNG artifacts exist, proxy-safe wording.

**Smoke pattern:**

```python
import jaxfne as jtfne
cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
```

**Full skill:** user-installed `jaxfne-evidence-validator`.
