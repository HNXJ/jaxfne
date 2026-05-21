# Tutorial Smoke Runner

**Skill purpose:** Validate tutorial infrastructure — examples, notebooks, and documentation — before releases or major feature additions.

**When to use:** Before cutting release tags, before merging large feature branches, or when adding new tutorial examples or notebooks.

**Baseline:** v0.2.17

## Overview

The tutorial smoke runner performs lightweight validation of all tutorial assets without executing notebooks:

1. **Python examples:** Syntax check only (no execution)
2. **Jupyter notebooks:** Structure check (required cells, no outputs, no private paths)
3. **Documentation:** Broken link detection (examples/ and notebooks/ references)

The runner is designed to be fast (< 1 second) and does not require Jupyter, nbconvert, or Plotly.

## Running the Smoke Test

### Default (all checks)

```bash
python scripts/run_tutorial_smoke.py
```

Output:
```
=== Tutorial Smoke Report ===

Status: PASS

Examples:
  ✓ 00_minimal_column
  ✓ 02_spectrolaminar_oddball_scaffold
  ✓ 03_single_neuron_multimodal_probe
  ✓ 04_two_neuron_ei_multimodal
  ✓ 05_network_100_ei_multimodal

Notebooks:
  ✓ 01_single_neuron_multimodal
  ✓ 02_two_neuron_ei_multimodal
  ✓ 03_network_100_ei_multimodal

Docs:
  ✓ tutorial markdown links
```

### Skip examples only

```bash
python scripts/run_tutorial_smoke.py --skip-examples
```

### Skip notebooks only

```bash
python scripts/run_tutorial_smoke.py --skip-notebooks
```

### Generate JSON report

```bash
python scripts/run_tutorial_smoke.py --report-json outputs/tutorial_smoke_report.json
```

The JSON report has this structure:

```json
{
  "status": "pass",
  "examples": {
    "checked": ["00_minimal_column", ...],
    "skipped": [],
    "errors": []
  },
  "notebooks": {
    "checked": ["01_single_neuron_multimodal", ...],
    "skipped": [],
    "errors": []
  },
  "docs": {
    "checked": ["tutorial markdown links"],
    "skipped": [],
    "errors": []
  }
}
```

## What Each Check Does

### Example Validation

For each Python example (`examples/0N_*.py`):
1. File exists
2. Python syntax is valid (can be compiled)
3. No execution is performed

**Why:** Early detection of syntax errors without startup overhead.

### Notebook Validation

For each Jupyter notebook (`notebooks/0N_*.ipynb`):
1. File exists
2. Valid JSON structure
3. Has at least 2 cells
4. **First code cell** contains `!pip install jaxfne`
5. **Second code cell** contains `jaxfne.__version__` verification
6. **No committed outputs** (cells have empty output arrays)
7. **No private paths** (`/Users/`, `/home/`, `~`, `C:\` not found in notebook text)

**Why:** Colab notebooks must be self-contained and reproducible; stale outputs and private paths break reproducibility.

### Documentation Link Check

For each tutorial markdown file (`docs/tutorials/*.md`):
1. Scan for references to `examples/0N_*.py` and `notebooks/0N_*.ipynb`
2. Verify referenced files exist
3. Report broken links

**Why:** Ensures tutorial guides point to actual artifacts; catches deleted files and renamed examples.

## Common Mistakes

### Mistake: Committed Notebook Outputs

**Symptom:** Smoke test fails with "notebook has committed outputs"

**Cause:** Notebook was saved with outputs.

**Fix:**
```bash
# In Jupyter: Kernel → Clear All Outputs
# Or use nbstripout:
pip install nbstripout
nbstripout notebooks/01_single_neuron_multimodal.ipynb
git add notebooks/01_single_neuron_multimodal.ipynb
git commit -m "clear notebook outputs"
```

### Mistake: Missing !pip install Cell

**Symptom:** Smoke test fails with "first code cell missing '!pip install jaxfne'"

**Cause:** Notebook was created locally without the Colab install cell.

**Fix:**
Add as the first code cell:
```python
!pip install jaxfne
```

Add as the second code cell:
```python
import jaxfne
print(f"jaxfne version: {jaxfne.__version__}")
```

### Mistake: Private Path in Notebook

**Symptom:** Smoke test fails with "notebook contains private paths"

**Cause:** Notebook contains absolute paths like `/Users/yourname/...` or `C:\Users\...`

**Fix:**
Use relative paths or environment variables instead:
```python
# ✗ BAD
data_path = "/Users/alice/projects/jaxfne/data/..."

# ✓ GOOD
data_path = Path("data/...").relative_to(Path.cwd())
# or
data_path = Path.home() / "data" / "..."
```

### Mistake: Broken Link in Tutorial Guide

**Symptom:** Smoke test fails with "broken link to examples/0N_..."

**Cause:** Tutorial markdown references an example that doesn't exist or was renamed.

**Fix:**
1. Check if the file exists: `ls examples/0N_...py`
2. If not, remove or correct the reference in the markdown
3. Or create the example if it's documented but missing

## Reporting Results

### In CI/CD

```bash
python scripts/run_tutorial_smoke.py --report-json artifacts/tutorial_smoke.json
# CI system reads JSON report and fails if status != "pass"
```

### In Local Development

Run before committing:
```bash
python scripts/run_tutorial_smoke.py
# Exit code 0 = all checks passed
# Exit code 1 = failures; read output for details
```

### In Release Checklists

```bash
# Before cutting a tag:
python scripts/run_tutorial_smoke.py
# Must pass with all checks in "checked" (not "skipped" or "errors")
```

## Design Philosophy

- **Fast:** No notebook execution (< 1 second for full suite)
- **Minimal dependencies:** Stdlib only (no nbconvert, Jupyter, Plotly)
- **Conservative defaults:** Check everything; use --skip-* flags to opt out
- **Fail loud:** Exit code 1 if any error detected
- **Report both ways:** Human-readable stdout + optional JSON
- **Do not execute:** Execution infrastructure (nbconvert/papermill) is separate

## Next Steps

- **Automated testing:** Integrate into CI to run on every PR
- **Notebook execution:** Future `--execute-notebooks` flag (requires nbconvert, slow)
- **Coverage:** Track which tutorials/examples are tested vs. untested

## Related Skills

- [Probe reports](skill_probe_reports.md) — validate probe report contracts
- [Field solution metadata](skill_field_solution_metadata.md) — validate field outputs
- [Physical field admissibility](skill_physical_field_admissibility.md) — validate Poisson solver outputs
