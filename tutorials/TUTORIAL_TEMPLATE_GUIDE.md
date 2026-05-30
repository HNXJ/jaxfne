# jaxfne Notebook Template Guide

## Overview

The **jaxfne Notebook Template** (`tutorials/templates/jaxfne_notebook_template.ipynb`) provides a canonical starting point for all jaxfne tutorials, Suites, and Etudes.

## Why Use the Template?

The template enforces:

- ✅ **Hygiene gates:** All code cells ≤ 8 lines, no consecutive code cells
- ✅ **Installation options:** Both PyPI and dev-branch installs
- ✅ **Canonical imports:** `import jaxfne as jtfne`
- ✅ **Truth gates:** Scope gates explicitly documented
- ✅ **Reproducibility:** Deterministic seeding from the start

## How to Use

### 1. Copy the Template

```bash
cp tutorials/templates/jaxfne_notebook_template.ipynb \
   tutorials/jaxfne_suite_no_5_my_feature.ipynb
```

Or for Etudes:

```bash
cp tutorials/templates/jaxfne_notebook_template.ipynb \
   tutorials/etudes/jaxfne_etude_no_2_my_workflow.ipynb
```

### 2. Edit the Notebook

- **Title:** Change the main heading to your tutorial/suite/etude name
- **Configuration:** Set SEED, DURATION_MS, DT_MS, and other parameters
- **Workflow steps:** Add your simulation, optimization, visualization steps
- **Scope gates:** Update if needed for your specific use case

### 3. Structure Rules

**Mandatory:**
- Keep both install cells (PyPI + dev branch)
- Keep truth gates documented at the top
- Separate every code cell with markdown (no consecutive code cells)
- Keep all code cells ≤ 8 lines

**Recommended:**
- Each major step (simulation, optimization, visualization) gets its own markdown section
- Use descriptive markdown labels (not "## Section 1")
- Include SMOKE mode support (`TFNE_SMOKE` environment variable) for large notebooks
- Export artifacts as JSON with metadata and asset hashes

### 4. Validation

Before finalizing:

```python
# Check structure
import nbformat
with open('your_notebook.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == 'code']
print(f"Code cells: {len(code_cells)}")
print(f"Max lines: {max(len(c.source.split(chr(10))) for c in code_cells)}")

# Count consecutive
consecutive = sum(1 for i in range(len(nb.cells)-1) 
                  if nb.cells[i].cell_type == 'code' and 
                     nb.cells[i+1].cell_type == 'code')
print(f"Consecutive code cells: {consecutive}")
```

## Template Cell Structure

| Cell | Type | Content |
|------|------|---------|
| 1 | Markdown | Title & scope gates |
| 2 | Markdown | Installation header |
| 3 | Code | PyPI install |
| 4 | Markdown | Dev branch header |
| 5 | Code | Dev branch install |
| 6 | Markdown | Imports header |
| 7 | Code | Imports & environment |
| 8 | Markdown | Configuration header |
| 9 | Code | Configuration setup |
| 10 | Markdown | Model construction header |
| 11 | Code | Model construction |
| 12 | Markdown | Workflow steps |
| 13 | Markdown | Next steps |

## Scope Gates Template

```markdown
## Scope Gates
- **truth_mode:** `truth_safe_unverified`
- **claim_level:** `computational_scaffold`
- **field_solver_status:** `laminar_proxy_no_pde`
- **physical_amplitude_claim_allowed:** `False`
```

Always include these unless your tutorial requires different gates. Document any deviations.

## Execution & Artifacts

For tutorials that run models:

### Artifacts to Export

```python
# manifest.json
manifest = {
    "artifact_class": "etude",  # or "suite"
    "artifact_id": "your_id",
    "jaxfne_version": jtfne.__version__,
    "truth_mode": "truth_safe_unverified",
    "claim_level": "computational_scaffold",
    # ... other metadata
}
jtfne.save_json(manifest, output_dir / "manifest.json")

# validation_report.json
validation = {
    "artifact_class": "etude",
    "notebook_execution": "nbclient_pass",
    "finite_outputs": True,
    # ... validation gates
}
jtfne.save_json(validation, output_dir / "validation_report.json")

# asset_hashes.json
hashes = {
    "manifest.json": sha256_hash,
    "validation_report.json": sha256_hash,
    # ... other artifacts
}
jtfne.save_json(hashes, output_dir / "asset_hashes.json")
```

## Example Workflow

1. **Copy template**
   ```bash
   cp tutorials/templates/jaxfne_notebook_template.ipynb \
      tutorials/jaxfne_suite_no_5_cortical_column.ipynb
   ```

2. **Edit notebook**
   - Update title
   - Set parameters (SEED, DURATION_MS, etc.)
   - Add workflow cells with markdown separators
   - Keep code cells ≤ 8 lines

3. **Test locally**
   ```bash
   TFNE_SMOKE=1 jupyter nbconvert --to notebook --execute your_notebook.ipynb
   ```

4. **Verify structure**
   - All code cells ≤ 8 lines: ✅
   - No consecutive code cells: ✅
   - Both install options present: ✅
   - Truth gates documented: ✅

5. **Execute full run**
   ```bash
   jupyter nbconvert --to notebook --execute your_notebook.ipynb
   ```

6. **Create receipt**
   - Document execution environment
   - Record notebook commit SHA
   - List all generated artifacts
   - Include validation gates

## Contributing

When creating tutorials:

1. **Always start from the template**
2. **Keep the hygiene gates** (cells ≤ 8 lines, no consecutive code)
3. **Document scope gates** clearly
4. **Export artifacts** as JSON with metadata
5. **Create a receipt** with execution details

## Questions?

Refer to the jaxfne documentation for:
- Configuration domains: `jaxfne.Configuration`
- Simulation API: `model.simulate()`
- Optimization: `model.tune()`, `jtfne.agsdr()`
- Visualization: `jtfne.vis.*`
