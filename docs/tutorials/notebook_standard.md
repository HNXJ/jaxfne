# jaxfne Colab Notebook Standard

jaxfne tutorials are Colab-ready Jupyter notebooks that teach the source-to-field/readout workflow. This page defines the standard all tutorial notebooks follow.

## Core Rules

### 1. Installation Cell (First Code Cell)

Every notebook **must** start with:

```python
!pip install jaxfne
```

This ensures the notebook runs in any fresh Colab environment without dependency issues.

### 2. Version Verification (Second Code Cell)

The second code cell **must** verify installation:

```python
import jaxfne
print(f"jaxfne version: {jaxfne.__version__}")
```

This confirms successful installation and documents the version used.

### 3. No Private Paths

Notebooks should avoid:
- Absolute paths like `/Users/your-name/...`
- Local-only data paths
- Private credentials or API keys

Use relative paths or public example data only.

### 4. CPU-Safe

Notebooks should run on CPU without GPU requirement. jaxfne is CPU-safe by default; CPU acceleration is recommended.

### 5. Outputs Cleared Before Commit

Notebooks **must** have empty outputs when committed to the repository. Output is regenerated when users run the notebook in Colab or locally.

Use `Kernel → Clear all outputs` in Jupyter before committing.

### 6. Reproducible Output Bundles

When applicable, notebooks **should** demonstrate and inspect jaxfne's output bundles:

```python
# Run simulation
signals = model.simulate(sim)

# Inspect manifest
manifest = model.manifest(signals)
print(f"Simulation complete. Manifest keys: {list(manifest.keys())[:5]}")
```

This teaches users to understand model metadata and claim-status.

### 7. Public Vocabulary

Notebooks **must** use approved public vocabulary. The following terms are **approved**:

- **proxy readouts** — Computed model outputs without empirical/physical validation
- **calibration workflows** — Explicit specification and reporting of calibration mode
- **source-to-field/readout** — Standard framework terminology
- **computational proxy** — Declared, not-yet-validated readout
- **declared metadata** — Explicit specification of assumptions, geometry, parameters

**Avoid:** internal control terminology, undeclared placeholder fields, claims that computed readouts equal real physical measurements without validation evidence.

Always frame readouts as computational or declared proxy, never as validated physical measurements without supporting validation workflow and evidence.

## Notebook Structure

Every tutorial notebook **should** follow this structure:

1. **Title and overview** (markdown)
2. **Install jaxfne** (code cell: `!pip install jaxfne`)
3. **Verify installation** (code cell: import and print version)
4. **Imports** (markdown heading + code cells)
5. **Configuration** (markdown + code cells to set up model)
6. **Simulation** (markdown + code cells to run example)
7. **Inspection** (markdown + code cells to examine outputs)
8. **Next steps** (markdown with links to other tutorials/docs)

## Template

See `notebooks/00_template_colab.ipynb` for a minimal working template.

## Validation

Before pushing a notebook:

1. **Clear outputs:** `Kernel → Clear all outputs`
2. **Run top-to-bottom:** Press `Ctrl+F9` to run all cells
3. **Check for private paths:** Search notebook for `/Users/`, `/home/`, or absolute paths
4. **Verify first cell:** Confirm it contains `!pip install jaxfne`
5. **Inspect vocabulary:** Check that internal terms are not used

## Colab-Ready Checklist

- [ ] First code cell: `!pip install jaxfne`
- [ ] Second code cell: Version verification
- [ ] No absolute/private paths
- [ ] CPU-safe (no GPU requirement)
- [ ] Outputs cleared
- [ ] Public vocabulary throughout
- [ ] Runs top-to-bottom without errors
- [ ] Includes section headings and markdown explanations
- [ ] Links to relevant docs/guides at the end

## Example Notebooks

The tutorial stack is:

1. **01_single_neuron_multimodal** (v0.2.8) — Single Izhikevich neuron with spikes, voltage, LFP-proxy
2. **02_two_neuron_ei** (v0.2.9) — Excitatory-inhibitory pair, recurrent coupling
3. **03_network_100_ei** (v0.2.10) — Balanced network (100 neurons), population dynamics
4. **04_v1_column** (v0.2.11) — Laminar V1 column (six layers), depth-specific readouts
5. **05_v1_pfc_dual_column** (v0.2.14) — Two-column cross-area model, traveling waves

Each notebook teaches a progressively more complex workflow using the same source-to-field/readout framework.

## Next Steps

- **[Tutorials Overview](index.md)** — All tutorial notebooks
- **[Calibration](../calibration.md)** — Preparing outputs for validation
- **[Guides](../guides/index.md)** — Detailed documentation
