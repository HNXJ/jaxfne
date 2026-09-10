# Remaining work — v0.4.22 → v0.4.24

Work loop and rules: `artifacts/AGENTS.md` (TODO stack).
Evidence: git, tests, receipts, tags, PyPI — not this file.

Integration baseline: `2338b00` (RC + CI PASS). Published `v0.4.21` @ `8823520` (immutable).

---

# v0.4.22

- **22-DOCS-02** — fix `sync_docs_version.py` pattern for current `install.md` wording
- 22-GEN-01 — regenerate `gallery_manifest.json` (`generate_release_gallery.py`)
- 22-GEN-02 — regenerate canonical atlas manifest if version-gated tests require
- 22-W45-01 — repair W4/W5 STALE public doc lines (`jdna.md` stamp; tutorial placeholder figures)
- 22-W45-02 — tutorial runner: implement or remove dead `--smoke` / `--out-root`; fix six public doc command lines
- 22-W45-03 — public stub contract: resolve `__init__` reachability vs `__all__` / docs for five names
- 22-W9-01 — declare `LegacyMultiAreaSpectrolaminarObjective` deprecation horizon; quiet canonical examples
- 22-W15-01 — BMTK/SONATA architecture study (written artifact only; no interoperability code)
- 22-W16-01 — scientific/developmental programme definition (planning; no public implementation claims)
- 22-W17-01 — downstream mechanism semantics (Jomission); W17.1–4 surfaces specified
- 22-W14-01 — harness workflow refine (W14): minimal router/skills; no duplicate procedures
- 22-VERIFY-01 — identity-sensitive tests after version bump
- 22-RC-01 — full RC on final 0.4.22 candidate SHA
- 22-CI-01 — `release_ci` + `reconcile_release_target.py` PASS @ final candidate SHA
- 22-SEAL-01 — independent release seal (requires explicit authorization to publish)
- 22-PUB-01 — tag `v0.4.22` @ sealed candidate (requires explicit authorization)
- 22-PUB-02 — GitHub release; retained CI artifact bytes + manifest
- 22-PUB-03 — PyPI publish from retained bytes (requires explicit authorization)
- 22-PUB-04 — RTD / deployed docs SHA matches tag
- 22-POST-01 — post-publication clean-install + identity reconcile; update `PUBLISHED_PYPI_VERSION` test constants
- 22-EXT-01 — correct external durable sources that still name stale engineering targets
- 22-STACK-01 — review and rewrite this file from sealed v0.4.22 before any v0.4.23 work

---

# v0.4.23

Entry: v0.4.22 shipped; W3/W7 classifications hold zero `UNKNOWN`; W8 baseline current.

- 23-ENT-01 — confirm 0.4.23 entry criteria (after 22-STACK-01)
- 23-REP-01 — eliminate avoidable persistent `emitter.W` (measured before/after on W8 matrix; bit-exact observables)
- 23-REP-02 — realized topology authoritative; backend layouts execution-specific only
- 23-REP-03 — sparsity routing: honour `p_connect` or refuse; lower `_SPARSE_DIRECT_N` where justified
- 23-PARAM-01 — parameter sharing for class-shared edge/neuron fields (bit-exact)
- 23-H-01 — H/RBS/RBD representation reduction (observable-specific; written per-coordinate argument)
- 23-HDP-01 — HDP null-plasticity bit-exact; no dense mutable `W` when disabled
- 23-DELAY-01 — delay/history compaction (delay classes, ring buffers; bit-exact continuation)
- 23-STOCH-01 — continuation-owned stochastic inputs where required (per-path verification)
- 23-REC-01 — streaming/decimated recording + memory preflight (W17.2)
- 23-EDGE-01 — edge selectors + immutable transforms pipeline (align W16 structural G)
- 23-LAW-01 — custom HDP/RBD law interface
- 23-FIN-01 — long-run transactional finalization (W17.4) if evidence justifies
- 23-SYSID-01 — operating-point system-ID utility (W17.3) if evidenced
- 23-JDNA-01 — minimal `evolve()` runtime substrate only if semantics + tests complete before public claims
- 23-VERIFY-01 — full 0.4.23 RC + CI + seal + publish chain
- 23-STACK-01 — review and rewrite this file from sealed v0.4.23 before v0.4.24 work

---

# v0.4.24

Entry: v0.4.23 shipped; W8 shows cumulative programme receipts from `8823520` baseline.

- 24-ENT-01 — confirm 0.4.24 entry criteria (after 23-STACK-01)
- 24-W16-6 — neurobiophysical geometry / bounded augmentation (W16.6); named observable, metric, ε
- 24-FIELD-01 — field/LFP proxy computational audit with declared error bounds
- 24-JAX-01 — JAX execution profile programme (measure before JIT changes)
- 24-EQUIV-01 — per-observable equivalence acceptance (exact or `d≤ε` pre-declared)
- 24-AUDIT-01 — independent adversarial core audit (implementer ≠ sole auditor)
- 24-EXIT-01 — verify programme exit criteria (roadmap §5.2)
- 24-VERIFY-01 — full 0.4.24 RC + CI + seal + publish chain
- 24-STACK-01 — seal v0.4.24; archive or clear this stack
