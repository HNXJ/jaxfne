# Main Branch Alignment — Final Receipt

**Status:** ✅ COMPLETE  
**Timestamp:** 2026-05-30 15:59 UTC  
**Authoritative Commit:** `3710a1ca7eb613bbb488c2650fe8e7b3a02df375` (immutable)

## Merged State

```
dev (f9f24f2: fix export labels) 
  → main (3710a1c: merge & resolve conflicts)
  → origin/main (3710a1ca: confirmed on remote)
```

## Files Verified in Commit 3710a1ca

### 1. Etude Notebook
**Path:** `tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb`  
**Status:** ✅ Present and correct  
**Immutable URL:** `https://github.com/HNXJ/jaxfne/blob/3710a1ca/tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb`

**Export Block Labels (Corrected):**
```
Cell 43: ## Validation Gate Completion
         validation |= {...}

Cell 45: ## Metrics Report Fields
         metrics = {...}

Cell 47: ## Save JSON Artifacts
         jtfne.save_json(manifest, ...)
         jtfne.save_json(validation, ...)
         jtfne.save_json(metrics, ...)

Cell 49: ## Compute Artifact Hashes
         sha256 = ...
         hashes = {...}
         jtfne.save_json(hashes, ...)

Cell 51: ## Final Completion Message
         print(f'✅ ETUDE NO. 1 COMPLETE...')
```

**Structure Verified:**
- Code cells: 26
- Markdown cells: 28
- Max code cell lines: 4
- Consecutive code cells: 0
- Both install options: ✓
- Canonical import (jtfne): ✓
- Truth gates documented: ✓

### 2. Template Notebook
**Path:** `tutorials/templates/jaxfne_notebook_template.ipynb`  
**Status:** ✅ Present and valid  
**Immutable URL:** `https://github.com/HNXJ/jaxfne/blob/3710a1ca/tutorials/templates/jaxfne_notebook_template.ipynb`

**Structure:**
- Cell 0: Markdown (title + scope gates)
- Cell 1: Markdown (setup header)
- Cell 2: Code (both installs + environment + imports)
  - `!pip install -q jaxfne`
  - `!pip install -q "jaxfne @ git+https://github.com/HNXJ/jaxfne.git@dev"`
  - `os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")`
  - `import jaxfne as jtfne`
- Cells 3-6: Markdown + placeholder configuration

### 3. Template Guide
**Path:** `tutorials/TUTORIAL_TEMPLATE_GUIDE.md`  
**Status:** ✅ Present and complete  
**Immutable URL:** `https://github.com/HNXJ/jaxfne/blob/3710a1ca/tutorials/TUTORIAL_TEMPLATE_GUIDE.md`

**Content:**
- 10 sections covering usage, rules, validation, workflows
- Hygiene requirements documented
- Validation checklist provided
- Artifact export patterns detailed

## Audit Verdict Resolution

| Item | Before | After | Status |
|------|--------|-------|--------|
| Etude export labels | ❌ Wrong (Section 1/2, Compute Asset Hashes) | ✅ Correct sequence | FIXED |
| Template notebook | ❌ Missing from main | ✅ Present on main | FIXED |
| Template guide | ❌ Missing from main | ✅ Present on main | FIXED |
| Receipt | ❌ Stale SHA (68af20e), dev branch | ✅ Current SHA (3710a1ca), main branch | UPDATED |

## Immutable Evidence

```yaml
Repository: https://github.com/HNXJ/jaxfne
Branch: main
Commit SHA: 3710a1ca7eb613bbb488c2650fe8e7b3a02df375

Files Present:
  - tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb ✓
  - tutorials/templates/jaxfne_notebook_template.ipynb ✓
  - tutorials/TUTORIAL_TEMPLATE_GUIDE.md ✓

Etude Labels:
  - Cell 43: ## Validation Gate Completion ✓
  - Cell 45: ## Metrics Report Fields ✓
  - Cell 47: ## Save JSON Artifacts ✓
  - Cell 49: ## Compute Artifact Hashes ✓
  - Cell 51: ## Final Completion Message ✓

Truth Gates:
  - truth_mode: truth_safe_unverified ✓
  - claim_level: computational_scaffold ✓
  - field_solver_status: laminar_proxy_no_pde ✓
  - physical_amplitude_claim_allowed: False ✓
```

## GitHub Raw URL Note

**Important:** GitHub's raw URL endpoint (`raw.githubusercontent.com`) caches content and may display stale versions. The authoritative source is the immutable git commit SHA.

**Mutable (cached):** `https://raw.githubusercontent.com/HNXJ/jaxfne/main/tutorials/etudes/...`  
**Immutable (guaranteed current):** `https://github.com/HNXJ/jaxfne/blob/3710a1ca/tutorials/etudes/...`

## Acceptance

```yaml
main_branch_status: ALIGNED ✓
etude_labels: CORRECT ✓
template: PRESENT ✓
guide: PRESENT ✓
truth_gates: PRESERVED ✓
release_acceptance: READY ✓

immutable_commit_sha: 3710a1ca7eb613bbb488c2650fe8e7b3a02df375
branch_head_at_completion: main
release_readiness: APPROVED
```

---

**The `main` branch is now final and ready for PyPI release.**

