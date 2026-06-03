# v0.3 Tutorial Docs Audit Policy

**Status:** v0.3.0 scaffold infrastructure  
**truth_mode:** truth_safe_unverified

---

## Purpose

This document defines required checks and standards for every v0.3 tutorial page. All tutorials must pass these audit gates before acceptance.

---

## Core Audit Rules

### 1. All Internal Links Resolve

Every markdown link to another doc or file must:
- Point to a file that exists in the repository
- Use relative paths (e.g., `[docs](../docs/example.md)`, not `https://github.com/...`)
- Not be broken or circular
- Check using: `scripts/audit_v030_docs_links.py`

### 2. All Image Links Resolve

Every image reference must:
- Point to a file that exists (e.g., in `docs/tutorials_v030/figures/` or similar location)
- Use relative paths only
- Be validated before tutorial acceptance
- Example format (do not use as-is; adapt to your docs structure): images/example_figure.png

### 3. All Plotly HTML Links Resolve (When Present)

If a tutorial includes optional Plotly interactive figures:
- HTML files must be generated and committed
- Links must resolve to valid files
- Fallback to PNG is always available

### 4. Every Tutorial Has an "Open in Colab" Link

**Required link format:**

```markdown
[Open in Colab](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/notebooks/v030/<notebook_name>.ipynb)
```

**Rules:**
- Link text must be exactly `Open in Colab` (no verbose preamble)
- Forbidden patterns (verbose, non-normalized):
  - `[Recommended: Open the full interactive tutorial in Colab](https://...)`
  - `[Click here to run in Colab](https://...)`
  - `[Open this tutorial in Colab (faster execution)](https://...)`
- For planned/future scenarios not yet implemented, mark the notebook as `planned` and skip the link
- Present link must be in every currently implemented tutorial

### 5. Every Theoretical/Equation Section Displays LaTeX Equations

**Required structure:**

Every mathematical section must include:

1. **Equation (displayed)**: Must use LaTeX delimiters (`$...$` or math environment)
2. **Term glossary**: List of symbols and their meanings
3. **Worded equation**: English description of what the equation computes
4. **Implementation location**: Where in jaxfne code this is computed
5. **Claim boundary**: What this equation claims and what it does not claim

**Correct pattern:**

```markdown
### Izhikevich Voltage Dynamics

The voltage update follows the cubic-linear Izhikevich form:

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I(t)$$

**Term Glossary:**
- $v$ = membrane potential (mV)
- $u$ = recovery variable (dimensionless)
- $I(t)$ = input current (pA)

**Worded Equation:**
The voltage changes based on a cubic nonlinearity (fast dynamics) and recovery (slow negative feedback), driven by input.

**Implementation:** `jaxfne/emitters/izhikevich.py:simulate_izhikevich_step()`

**Claim Boundary:**
This equation is a mathematical model, not validated against biological data. The native current $I(t)$ is not calibrated to match real whole-cell recordings.
```

**Incorrect pattern (words only, no equation):**

```markdown
The Izhikevich model uses a cubic-linear update for voltage based on recovery and input.
```

### 6. Every Tutorial Includes Claim Gates from Its Manifest

Claim gates must be stated explicitly in the tutorial:

```markdown
### Claim Gates

This tutorial operates under these immutable constraints:

- **truth_mode:** `truth_safe_unverified`
- **claim_level:** `computational_scaffold`
- **physical_amplitude_claim_allowed:** `False`
- **field_solver_status:** `laminar_proxy_no_pde`
- **biological_metabolism_claim_allowed:** `False`
```

### 7. Every Tutorial Includes PNG Figures and Optional Plotly HTML

**Required PNG figures:**
- At least 2 PNG figures per tutorial
- Generated from real simulation data, not manually created
- File size >1 KB (indicates real content, not flat/trivial)
- SHA256 hash recorded in manifest
- Naming: `v030_<scenario_number>_<description>.png` (e.g., `v030_01_spike_raster.png`)
- Resolution: dpi=150

**Optional Plotly HTML:**
- May be generated if Plotly is available
- Must be generated from source data (spike arrays, voltage traces), not PNG conversion
- Guarded import pattern; graceful fallback if absent
- Self-contained HTML (no external dependencies)
- SHA256 hash recorded in manifest (if present)

---

## Audit Workflow

### For Humans (Tutorial Authors)

1. Write tutorial following 13-section template
2. Include all required LaTeX equations with term glossaries
3. Add "Open in Colab" link (exact format)
4. Generate PNG figures from simulation data
5. Run `scripts/audit_v030_docs_links.py` locally
6. Fix any reported broken links or policy violations
7. Submit for review

### For CI/CD (Automated Validation)

1. Run `scripts/audit_v030_docs_links.py` on every PR
2. Check JSON report for failures:
   - `missing_links`: Non-zero → fail
   - `missing_colab_links`: Non-zero for current (non-planned) scenarios → fail
   - `latex_policy_violations`: Non-zero → fail
3. Run `tests/test_v030_docs_audit.py`
4. All tests must pass before merge

### Audit Script Output

The audit script produces `docs_link_audit.json`:

```json
{
  "schema_version": "v0.3.0",
  "status": "pass",
  "checked_files": 6,
  "missing_links": [],
  "missing_colab_links": [],
  "latex_policy_violations": [],
  "notes": []
}
```

**Status meanings:**
- `"pass"` = all checks passed; audit gates satisfied
- `"fail"` = one or more checks failed; gate violation
- `"warn"` = non-critical issues (e.g., future planned scenarios)

---

## Special Cases

### Planned Scenarios (v0.3.16-v0.3.31)

For audit phases v0.3.16-v0.3.31 (planned but not yet implemented):
- Do not require "Open in Colab" link if notebook does not exist yet
- Mark scenario entry as `"status": "planned"` in scenario_index.md
- Still must include docstring and learning objectives

### Legacy Tutorial Conversions

If converting existing jaxfne tutorials to v0.3 format:
- Preserve original content and equations
- Add missing LaTeX displays if currently text-only
- Verify all links still resolve (may need updates for new repo structure)
- Add claim gates if missing
- Run full audit before merging

---

## Failure Modes and Recovery

### Broken Link Detected

1. Run audit script to identify exact location
2. Check if file/image exists in repo
3. If file missing, add it or update link
4. If link wrong, correct it
5. Re-run audit script to confirm fix

### Missing "Open in Colab" Link

1. Check if tutorial notebook exists (e.g., `notebooks/v030/01_single_neuron.ipynb`)
2. If exists, add link with exact format (no preamble)
3. If not yet created, mark scenario as planned in index
4. Re-run audit to confirm

### Missing LaTeX Equation Display

1. Locate equation section (marked with "###" or "####")
2. Add displayed math: `$$...$$` or markdown math block
3. Add term glossary below equation
4. Add worded-equation (English description)
5. Add implementation location and claim boundary
6. Re-run audit to confirm

### Claim Gates Missing

1. Check tutorial notebook or markdown
2. Add Section 8 (Manifest and Claim Gates) if missing
3. Include exact claim gates from template
4. Do not allow overclaiming (biological validation, real EEG/MEG, etc.)

---

## Policy Enforcement

**Hard gates (no exceptions):**
- Every theoretical section must have displayed LaTeX
- Every tutorial must have "Open in Colab" link (exact format, no preamble)
- Claim gates must be immutable and frozen
- No broken internal links

**Soft checks (warn if violated):**
- Future planned scenarios missing links (expected)
- Minor link inconsistencies (e.g., inconsistent slashes)

**Automated:**
- `scripts/audit_v030_docs_links.py` detects policy violations
- `tests/test_v030_docs_audit.py` enforces presence of policy docs
- CI/CD blocks merge if any hard gate fails

---

## Related Documents

- **docs/tutorials_v030/template.md** — 13-section required tutorial structure
- **docs/tutorials_v030/scenario_index.md** — All 15 scenarios and 16 audit phases
- **docs/tutorials_v030/plotly_policy.md** — PNG/Plotly artifact rules
- **scripts/audit_v030_docs_links.py** — Automated audit script
- **tests/test_v030_docs_audit.py** — Policy compliance tests

---

**Status:** v0.3.0 scaffold  
**truth_mode:** truth_safe_unverified
