# jaxfne v0.3.4 Development Context

**Status:** v0.3.4 released, PyPI published  
**truth_mode:** truth_safe_unverified  
**claim_level:** computational_scaffold  

---

## Identity

jaxfne is a **JAX-native computational scaffold** for TFNE (Tensor-Field Neural Equations).

- **What it does:** Chainable Configuration grammar to simulate spiking neural networks with proxy readouts (MUA, LFP, CSD, etc.)
- **What it is NOT:** A validated biological simulator or empirical modeling tool
- **Scope:** Tutorial-scale teaching, prototyping, optimization experiments
- **Output:** JSON-safe Signals objects with proxy metrics, deterministic PRNG, reproducible figures

---

## Public Grammar (v0.3.4)

```python
import jaxfne as jtfne

cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
cfg = cfg.cell_types({"E": 1.0})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
```

**Key principles:**
- Chainable (each method returns updated cfg)
- Immutable configuration (no side effects)
- Public API only: `Configuration`, `construct`, `simulate`
- No private model/cfg introspection in public examples

---

## Release Gates (Validation)

Every release must pass:

```bash
# 1. Syntax check
python -m compileall -q jaxfne tests examples

# 2. Full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=short
# Expected: 1062 passed, 37 skipped (or better)

# 3. Grammar smoke test
PYTHONPATH=. python - <<'PY'
import jaxfne as jtfne
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=100.0, dt_ms=0.1)
cfg = cfg.column("test", layers=["L2/3"], n=1)
cfg = cfg.cell_types({"E": 1.0})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["MUA-proxy"])
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.1, seed=7)
print("✓ Grammar PASS")
PY

# 4. PyPI availability (after publish)
pip install -U "jaxfne>=0.3.4"
python -c "import jaxfne; assert jaxfne.__version__ == '0.3.4'"
```

---

## Scope Boundaries (Public Wording)

**DO claim:**
- "Computational scaffold"
- "Proxy readout" (not real measurements)
- "Tutorial-scale simulation"
- "Chainable Configuration grammar"
- "JSON-safe outputs"
- "Reproducible with fixed seed"

**DO NOT claim:**
- "Real EEG/MEG/CSD" (use "proxy" instead)
- "Biological validation" (use "tutorial demonstration")
- "Calibrated amplitude" (use "relative metrics")
- "Solved 3D electric field" (use "proxy projection")
- "Biological metabolism" (EMM-proxy is relative activity cost)

---

## Branch & Merge Policy

**Stable branches:** `main` (source-of-truth), `dev` (integration)

**Workflow:**
1. Create feature branch from `dev`
2. Test locally: `pytest tests/`
3. Merge back: `git merge --no-ff <feature>`
4. Before main release: merge dev → main with all gates passing
5. Tag release: `git tag -a v0.3.X`
6. Push: `git push origin main dev v0.3.X`

**No force-push.** If branch history needs fixing, create new branch and retire old one.

---

## Optional Dependencies

**Core:** JAX, NumPy, SciPy, pandas, PyYAML (required)

**Optional:**
- `[viz]`: matplotlib, plotly (for figures, NOT required by core API)
- `[opt]`: optax (for optimization, NOT required by core API)
- `[dev]`: pytest, black, ruff (development only)

**Rule:** Core tests must not require `matplotlib`. Visualization tests use `pytest.importorskip("matplotlib")`.

---

## Version & PyPI

**Current:** v0.3.4  
**Release pattern:** v0.3.X for tutorial-scenario spine on stable v0.2.30 toolbox

**Before publishing:**
- Update `pyproject.toml` version
- Update `jaxfne/core.py` `_JAXFNE_VERSION`
- Update version in docs, requirements, tests
- Build: `python -m build`
- Check: `python -m twine check dist/*`
- Publish: `python -m twine upload dist/*`
- Verify: Fresh venv install + grammar smoke

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `pyproject.toml` | Version, dependencies, extras |
| `jaxfne/core.py` | Core API (version, Configuration, Signals) |
| `README.md` | Public grammar example, install, scope |
| `docs/scope_and_limitations.md` | Detailed scope boundaries |
| `tests/test_compact_facade_v034.py` | Grammar smoke tests |

---

## When to Update This File

Edit `.claude/CLAUDE.md` when:
- New major version released (update version, grammar example)
- Public API changes (update grammar block)
- New release gates added (update validation section)
- Scope boundaries clarified (update wording section)

DO NOT edit for:
- Bug fixes or internal refactors (update tests instead)
- Documentation-only changes (update docs/, not CLAUDE.md)

---

## Quick Commands

```bash
# Install for development
pip install -e '.[dev,viz,opt]'

# Run tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=short

# Check version
python -c "import jaxfne; print(jaxfne.__version__)"

# Release checklist
python -m compileall -q jaxfne tests examples
pytest tests/
python -m twine check dist/*
pip install dist/jaxfne-*.whl && python -c "import jaxfne as j; cfg=j.Configuration(); print('OK')"
```

---

**Last updated:** 2026-05-26  
**Phase:** v0.3.4 stable, ready for v0.3.5+ expansion
