# jaxfne-api-truth

**Triggers:** tutorial, example, notebook, docs, local reimplementation, invented API.

**Purpose:** Tutorials configure; the package computes. No notebook-local engine loops or readout operators.

**Required public path:**

```python
import jaxfne as jtfne
cfg = jtfne.Configuration()  # chainable
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, ...)
readouts = model.probe(signals, ...)
```

**Flag:** local Izhikevich loops, local LFP/CSD/EEG/MEG projectors, local optimizers, non-existent APIs.

**Full skill:** user-installed `jaxfne-api-truth` (often under `_archived` in user install).
