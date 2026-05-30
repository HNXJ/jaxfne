# Etude No. 1 — Final Execution Receipt (Superseded by MAIN_ALIGNMENT_FINAL_RECEIPT.md): Final Execution Receipt

**Status:** ✅ **FULLY EXECUTABLE AND VALIDATED**  
**Date:** 2026-05-30

## Commit Information (Three-Field Schema)

```
validated_notebook_commit:  f8e7d13ed01b63c1c80fde18b2fc72c584e15728
receipt_commit:             94a78fe53f0661178f56644fd527f05289ea72b1
branch_head_at_push:        dev
```

## Environment

```
Repository:        /Users/hamednejat/workspace/main/jaxfne
Branch:            dev
Python:            3.14.4
jaxfne:            0.3.14
Platform:          macOS 25.5.0 (arm64)
```

## Notebook Structure

```
Path:              tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb
Validated Commit:  f8e7d13ed01b63c1c80fde18b2fc72c584e15728
Total Cells:       54 (26 code + 28 markdown)
Code Cell Max:     ≤ 8 lines per cell ✓
Consecutive Code:  0 (all separated by markdown) ✓
Colab Badge:       Fixed to point to /tutorials/etudes/ ✓
```

## Installation Options ✅

Both present at notebook top:

```python
!pip install -q jaxfne
!pip install -q "jaxfne @ git+https://github.com/HNXJ/jaxfne.git@dev"
```

## Configuration Domains ✅

All 9 explicitly set with markdown separators:

1. Runtime - seed, duration_ms, dt_ms, dtype
2. Columns & Areas - V1/V4, n_per_area, layers
3. Cell Types - E/PV/SST/VIP fractions
4. Drive - baseline drive, noise amplitude
5. Inter-Column Connectivity - feedforward/feedback parameters
6. Field/Proxy - laminar_proxy_no_pde status
7. Probes - spikes, Vm, source, LFP, CSD
8. Objective - target rate/kappa, weights
9. Optimizer - AGSDR generations/population/seed

## Code Cell Hygiene ✅

| Requirement | Status | Evidence |
|---|---|---|
| All code cells ≤ 8 lines | ✅ | Max: 4 lines per cell |
| No consecutive code cells | ✅ | All 26 separated by 28 markdown cells |
| Both install options | ✅ | Cells 2 & 4 present |
| Explicit config domains | ✅ | All 9 domains explicitly set |
| Colab badge path correct | ✅ | Points to /tutorials/etudes/ |

## Execution Verification ✅

- **SMOKE Mode (TFNE_SMOKE=1):** 300ms, 40 neurons/area → All cells passed
- **FULL Mode (TFNE_SMOKE=0):** 1000ms, 80 neurons/area → All cells passed

## Artifacts Generated ✅

- manifest.json (24 required fields)
- validation_report.json (12 gate fields)
- metrics.json (8 metric fields)
- asset_hashes.json (SHA256 hashes)
- spectrolaminar.png (287 KB visualization)

## Truth Gates Enforced ✅

```
✅ truth_mode:                         truth_safe_unverified
✅ claim_level:                        computational_scaffold
✅ field_solver_status:                laminar_proxy_no_pde
✅ physical_amplitude_claim_allowed:   false
```

## Final Verification Checklist ✅

| Requirement | Status |
|---|---|
| Both install options present | ✅ |
| All 9 config domains explicit | ✅ |
| All code cells ≤ 8 lines | ✅ |
| No consecutive code cells | ✅ |
| Both execution modes pass | ✅ |
| All artifacts generated | ✅ |
| Truth gates preserved | ✅ |
| Colab badge correct | ✅ |

## Final Verdict

**Status:** ✅ **FULLY COMPLETE — ALL REQUIREMENTS SATISFIED**

The notebook at commit `f8e7d13ed01b63c1c80fde18b2fc72c584e15728` is verified to:

- Have both install options (PyPI + dev @git)
- Have all 9 configuration domains explicitly set
- Have all code cells ≤ 8 lines
- Have NO consecutive code cells (all separated by markdown)
- Have correct Colab badge path
- Execute successfully in both SMOKE and FULL modes
- Generate all required artifacts
- Enforce all truth gates

Ready for final acceptance.

---

**Execution Receipt Signed:**

```
[claude-sonnet-4.6][/Users/hamednejat/workspace/main/jaxfne][20260530-1835]
```

**Validated Notebook Commit:** `f8e7d13ed01b63c1c80fde18b2fc72c584e15728`
