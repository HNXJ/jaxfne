# Claude Code Context — jaxfne

**Last updated:** 2026-05-27  
**Scope:** Agent discipline, API contracts, public wording, generated outputs, validation receipts  
**Audience:** Worker agents executing jaxfne tutorial/feature tasks

---

## 1. Project Identity & Workflow

**jaxfne** is a compact JAX-based computational scaffold for Tensor-Field Neural Equations.

**Public grammar — the ONLY way to build in tutorials:**

```python
import jaxfne as jtfne

cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=42, dtype='float32', duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column(name='my_column', layers=['L2/3', 'L4', 'L5', 'L6'], n=12)
cfg = cfg.cell_types({'E': 0.75, 'PV': 0.15, 'SST': 0.05, 'VIP': 0.05})
cfg = cfg.connectivity(kind='laminar_signed_metadata', recurrent=True)
cfg = cfg.set_emitter('izhikevich', 'cortical_eig')
cfg = cfg.probes(['spikes', 'V_m', 'source', 'LFP-proxy', 'CSD-proxy'], n_contacts=16)

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=42)

# Outputs via signals object
spikes = signals.spikes                 # [T, N]
lfp_proxy = signals.field.lfp_proxy     # [T, C]
csd_proxy = signals.field.csd_proxy     # [T, C]
```

**Version discipline:** Package version stays 0.3.5. Tutorial milestones (v0.3.6, v0.3.7, v0.3.8) are separate labels.

---

## 2. Repository Verification

**Canonical path:** `/Users/hamednejat/workspace/main/jaxfne`

**Before every task:**

```bash
pwd && git branch --show-current && git status --short
python -c "import jaxfne; print(jaxfne.__version__)"
```

**PYTHONPATH:** Default is `.` (source at root). NEVER use `src/jaxfne` layout.

**Never:** Work in `jbiophysic` for jaxfne tasks.

---

## 3. API Contracts

**Test API calls in REPL before writing tutorials.** Known contracts (v0.3.5):

- **Field access:** `signals.field.lfp_proxy`, `signals.field.csd_proxy` (NOT dictionary lookup)
- **Configuration:** Fluent chaining, `.probes(...)` is current method
- **Emitter preset:** `'cortical_eig'` is canonical
- **Connectivity:** `'laminar_signed_metadata'`, `'none'`, etc.

**Common mistakes:**

- ❌ `signals['LFP-proxy']` or `model.probe()` returning dict
- ✓ `signals.field.lfp_proxy`
- ❌ `cfg.set_probes(...)` (old)
- ✓ `cfg.probes(...)`

---

## 4. Public vs Private Language

### Public Surfaces:

- README, docs, tutorials, PyPI metadata
- All user-facing code

### Private Surfaces:

- `internal_docs/` (this directory)
- Tests (internal patterns)
- `.claude/` (never committed)

### Public Prose:

✓ Use: "configured workflow", "JAX-based", "proxy-scale", "scope metadata", "interpretation boundary"

❌ Avoid: "grammar-corrected", "claim gates", "Does NOT", "truth_mode", "compliance", "corrective"

---

## 5. Public Wording Scan

```bash
grep -RInE "grammar-corrected|claim_level|Does NOT|truth_mode" docs tutorials || echo "PASS"
```

Expected: Zero hits in public surfaces.

---

## 6. Tutorial Deliverables

**All 13 sections required:**

1. Learning objectives
2. Biological/computational question
3. Mathematical glossary flow
4. Canonical import
5. Configuration block
6. Simulation block
7. Probe/readout block
8. Manifest and scope metadata
9. Figures (PNG, validated)
10. Interpretation (allowed vs. blocked)
11. Failure modes
12. Exercises
13. Scope boundaries

**Numerical defaults (non-negotiable):**

- `duration_ms >= 1000.0`
- `dt_ms = 0.1`
- `dtype = "float32"`
- Deterministic seed
- All outputs finite
- JSON-safe metadata
- PNG figures

---

## 7. Generated Outputs

**Location:** `tutorial_outputs/<tutorial_id>/`

**Rules:**

1. Untracked by default
2. Validate with `JAXFNE_VALIDATE_TUTORIAL_OUTPUTS=1` env var
3. Default pytest must pass WITHOUT pre-existing artifacts
4. Only durable assets under `docs/assets/`

---

## 8. Validation Reports

**Every report must include (exact counts, not summaries):**

1. Repo path, branch, HEAD SHA
2. Files changed
3. Commands run (exact, with flags)
4. Test results: "37 passed, 0 failed"
5. Execution receipt: rates, shapes, finiteness
6. Artifacts: PNG count, JSON validation
7. Public scan: forbidden pattern count
8. Generated output status: untracked confirmation
9. Blockers: exact description
10. Non-actions: explicit statement
11. Next action: one atomic step

---

## 9. Known Failure Modes

❌ Work in `jbiophysic` instead of `jaxfne`
❌ Use stale artifact outputs in reports
❌ Assume API contracts, don't test them
❌ Delete tests to bypass missing deps
❌ Report validation without exact commands
❌ Commit generated files
❌ Expose internal rule language in public docs
❌ Use low-level kernels in tutorials
❌ Confuse tutorial milestones with package version
❌ Treat quiet-state as active-rate tutorial
❌ Forget to push branches before review
❌ Merge before notebook execution receipts
❌ Place durable instructions in `.claude/` instead of `internal_docs/`

---

## 10. ALWAYS Do

- Verify repo and branch before work
- Test API calls in REPL before tutorials
- Use ONLY public configured workflow
- Keep public docs positive
- Separate public from private language
- Generate JSON-safe manifests
- Gate artifact tests with env var
- Run `mkdocs build --strict` for doc changes
- Push branches before review
- Report exact SHAs and test counts

---

## 11. NEVER Do

- Work in `jbiophysic` for jaxfne
- Commit `.claude/` files
- Commit `tutorial_outputs/`, `figures/`
- Bump version without authorization
- Tag or publish without authorization
- Use low-level calls in public code
- Replace missing public API with local code
- Expose internal language in public docs
- Delete tests to pass CI
- Claim execution without receipts
- Report "CI green" without run ID
- Merge with unclassified public scan hits
- Start next milestone before CI is green

---

**Durable context for all future worker-agent runs. Update when lessons emerge. Never expose in public.**
