# Evidence audit — critic surfaces (jcritic pass)

**Status:** READ-ONLY filing (no core/package changes)  
**Date:** 2026-09-07  
**Repository tip:** `b08f20f7a73b8e7ca0efc751b859ef86a0f6d669` (`main`)  
**Package version at tip:** `0.4.21` (`pyproject.toml`, `mkdocs.yml` extra `jaxfne_version`, `docs/citation.md`, `CITATION.cff`)  
**Tags present through:** `v0.4.20` (no `v0.4.21` tag at audit time)  
**Method:** GitHub-only static review (file reads, grep, nav leaf count). **Gate 0 not run.** Audit harness scripts **not executed** in this pass.  
**Scope:** Critic surfaces only — `artifacts/issue_log/`, `artifacts/skills/`, `artifacts/science/`, `docs/` agent/contributor paths, harness audit script docstrings. **No changes under `jaxfne/` or frozen publication paths.**

---

## Method

| Step | Action | Executed? |
|------|--------|-----------|
| M1 | Read freeze authorities (`ISSUE_LOG` header, `jaxfne-frozen-use`, `SCIENTIFIC_WORKBENCH_STATE.md`, release/seal skills) | yes (static) |
| M2 | Compare authority version claims to tip identity (`pyproject.toml`, `mkdocs.yml`, `CITATION.cff`, git tags) | yes (static) |
| M3 | Inventory live skills under `artifacts/skills/` vs routes in `docs/for_ai_agents.md` | yes (static) |
| M4 | Count MkDocs nav leaf pages vs harness script hardcoded page count | yes (static; `88` leaves) |
| M5 | Sample public doctrine/status pages for H≠homeostasis and proxy≠calibrated language | yes (sampled; consistent) |
| M6 | Run `scripts/harness/gate0_git_reality.py` | **no** |
| M7 | Run `scripts/harness/audit_public_private_boundary.py` | **no** |
| M8 | Run `scripts/harness/audit_81_docs_pages.py` | **no** |
| M9 | Clone-backed full docs audit | **no** |

**Classification key:** `DOC` = documentation/routing drift; `FACT` = observable repository-state mismatch (versions, counts, paths, dates).

---

## Findings

### F-001 — Freeze authorities vs tip version (block)

| Field | Value |
|-------|-------|
| **Severity** | block |
| **Tags** | DOC, FACT |
| **Claim** | Multiple freeze authorities still assert an active v0.4.17 core freeze and 40-day frozen-use window while tip activates v0.4.21 RC. |
| **Evidence** | `artifacts/issue_log/ISSUE_LOG.md` header (v0.4.17 frozen period ~2026-08-19–2026-09-28); `artifacts/skills/jaxfne-frozen-use/SKILL.md` (ΔC_core=0, observe→reproduce→log); `artifacts/science/SCIENTIFIC_WORKBENCH_STATE.md` (_Last indexed: 2026-08-21_, HEAD `caec1f71`, `jaxfne==0.4.17`, **do not edit**); `artifacts/skills/jaxfne-seal/SKILL.md` (authority `jaxfne_v0_4_17_final_100_goals.md`); tip: `pyproject.toml` `version = "0.4.21"`, `mkdocs.yml` `jaxfne_version: "0.4.21"`, `docs/changelog.md` v0.4.21 entry, tags through `v0.4.20`. |
| **Expected** | Freeze narrative, workbench index, and agent authorities align with the active release candidate / post-freeze policy at tip. |
| **Actual** | Authorities pin 0.4.17-era freeze semantics; tip declares 0.4.21 RC with no matching authority refresh. |
| **Scripts run** | none |
| **Recommended follow-up** | Docs/critic-surface PR: refresh ISSUE_LOG header and frozen-use window narrative; update `SCIENTIFIC_WORKBENCH_STATE.md` index; supersede or annotate release/seal skill authorities for 0.4.21+. **Do not patch `jaxfne/` core to resolve.** |

### F-002 — Missing skill routes in `docs/for_ai_agents.md` (should-fix)

| Field | Value |
|-------|-------|
| **Severity** | should-fix |
| **Tags** | DOC |
| **Claim** | Agent onboarding doc routes to skills that do not exist in this checkout. |
| **Evidence** | `docs/for_ai_agents.md` lines 15–16 reference `artifacts/skills/catalog-glossary-jaxfne/SKILL.md` and `artifacts/skills/jaxfne-worker-context-router/SKILL.md` (HTML comments note optional/absent). Live skills (7): `jaxfne-{audit,core,frozen-use,release,repo,science,seal}`. |
| **Expected** | Start-here routes point to present skills or explicitly defer to live code + `artifacts/AGENTS.md`. |
| **Actual** | Two primary routes target absent skill folders. |
| **Scripts run** | none |
| **Recommended follow-up** | Docs PR: replace routes with live skill set or remove stale paths; sync any mirror references in `.lab` snapshots separately if needed. |

### F-003 — `docs/contributing.md` stale `skills/` path (should-fix)

| Field | Value |
|-------|-------|
| **Severity** | should-fix |
| **Tags** | DOC |
| **Claim** | Contributing guide tells maintainers to update root `skills/`; canonical tree is `artifacts/skills/`. |
| **Evidence** | `docs/contributing.md` line 32: "update `skills/`"; `Glob artifacts/skills/**/SKILL.md` → 7 files; no root `skills/` directory. |
| **Expected** | Contributor path matches canonical `artifacts/skills/`. |
| **Actual** | Path points to absent root `skills/`. |
| **Scripts run** | none |
| **Recommended follow-up** | One-line docs fix in contributing guide. |

### F-004 — Harness audit scripts hardcode 81 MkDocs pages (should-fix)

| Field | Value |
|-------|-------|
| **Severity** | should-fix |
| **Tags** | DOC, FACT |
| **Claim** | Doc audit scripts and docstrings assert 81 public nav pages; current nav has 88 leaves. |
| **Evidence** | `scripts/harness/audit_81_docs_pages.py` title/docstring and PASS banner; `scripts/harness/audit_public_private_boundary.py` docstring line 5 and success print line 130; static nav count from `mkdocs.yml` → **88** leaves. |
| **Expected** | Audit scripts derive page count from nav or match current count. |
| **Actual** | Hardcoded `81` stale by 7 pages. |
| **Scripts run** | none (count verified statically) |
| **Recommended follow-up** | Harness docs PR: parameterize nav leaf count or rename script; optional clone-backed re-run after fix. |

### F-005 — Release/seal skills pin v0.4.17 receipt (should-fix)

| Field | Value |
|-------|-------|
| **Severity** | should-fix |
| **Tags** | DOC, FACT |
| **Claim** | `jaxfne-release` and `jaxfne-seal` AUTHORITIES still pin `artifacts/release/v0_4_17_release_receipt.json` while package is 0.4.21 RC. |
| **Evidence** | `artifacts/skills/jaxfne-release/SKILL.md` line 14; `artifacts/skills/jaxfne-seal/SKILL.md` line 14; tip `pyproject.toml` `0.4.21`; tags through `v0.4.20`; no superseding receipt path referenced in skills. |
| **Expected** | Release/seal authorities reference the receipt matching the active RC or document explicit supersession chain. |
| **Actual** | Both skills cite v0.4.17 receipt only. |
| **Scripts run** | none |
| **Recommended follow-up** | Skills/docs PR: add 0.4.21 receipt authority or explicit "receipt chain" note without mutating frozen receipt JSON. |

### F-006 — `SCIENTIFIC_WORKBENCH_STATE.md` stale index (should-fix)

| Field | Value |
|-------|-------|
| **Severity** | should-fix |
| **Tags** | DOC, FACT |
| **Claim** | Scientific workbench routing index lags tip by ~17 days. |
| **Evidence** | `artifacts/science/SCIENTIFIC_WORKBENCH_STATE.md` line 4: _Last indexed: 2026-08-21 (branch: dev, HEAD: caec1f71)_; tip HEAD `b08f20f7` dated 2026-09-07; frozen instrument block still says `jaxfne==0.4.17`. |
| **Expected** | Index date, HEAD, and instrument version reflect current tip or carry explicit stale banner with successor. |
| **Actual** | Index frozen at 2026-08-21 / 0.4.17 / `caec1f71`. |
| **Scripts run** | none |
| **Recommended follow-up** | Update workbench index in a docs/critic-surface PR (routing only; do not override evidence receipts). |

### F-007 — Public/private auditor nav-only scope (unknown / nit)

| Field | Value |
|-------|-------|
| **Severity** | unknown (nit on scope gap) |
| **Tags** | DOC |
| **Claim** | `audit_public_private_boundary.py` scans only MkDocs nav pages; `docs/for_ai_agents.md` is in `mkdocs.yml` `exclude_docs` but contains process paths (`CURRENT_TASK`, worker-context language). Full PASS/FAIL unknown without execution. |
| **Evidence** | `mkdocs.yml` `exclude_docs` includes `for_ai_agents.md`; `docs/for_ai_agents.md` lines 47–48 reference `scratch/CURRENT_TASK.md`; `audit_public_private_boundary.py` `audit_public_docs()` iterates nav files only; `FORBIDDEN_PROCESS_PATTERNS` includes `CURRENT_TASK\.md` and `worker context`. Sampled public doctrine (`docs/scope_and_status.md`, protocol pages): H≠homeostasis and proxy≠calibrated language consistent. |
| **Expected** | Either excluded agent docs are out of audit scope by design, or a separate agent-surface audit covers them. |
| **Actual** | Nav-only scan; excluded agent doc not audited; script outcome not verified in this pass. |
| **Scripts run** | none |
| **Recommended follow-up** | Optional: document audit scope boundary; optional clone-backed run of boundary + 81-page scripts; consider agent-doc scan if policy requires. |

---

## Residue (not checked)

- Gate 0 (`scripts/harness/gate0_git_reality.py`) — not run  
- Full harness adversarial suite  
- Wheel/sdist leak audit (requires built `dist/`)  
- All 88 nav pages scanned for forbidden process language (script not executed)  
- CI green status at tip  
- PyPI publish state for 0.4.21  
- `scratch/CURRENT_TASK.md` freeze frontmatter (may be gitignored / absent in clone)  
- Complete reconciliation of all `.lab` snapshot references to retired skills  

---

## Issue log cross-reference

Findings filed as **I-008** through **I-014** in `artifacts/issue_log/ISSUE_LOG.md`.

---

## Verdict

Seven critic-surface findings documented. **No package-core remediation required or performed.** Recommended work is documentation, skills authority, and harness-script maintenance on allowed writable surfaces.
