# 06_VALIDATION_LADDER

Expected receipts (compile PASS, notebook audit PASS, mkdocs strict exit 0 in all cases):

| tree | focused (5-file set) | full suite |
|---|---|---|
| `main` @ fab4c9c (pre-PR#22 merge) | `49 passed, 1 skipped` | `1961 passed, 66 skipped, 4 xfailed` |
| with PR#22 (B01) merged | `54 passed, 1 skipped` | `1986 passed, 66 skipped, 4 xfailed` |

The +5 / +25 delta is the B01 reproducibility test file (`tests/test_objective_null_reproducibility_v0330.py`). Match the row to whichever tree you re-froze this tick (`Pasted markdown.md:L63`).

## Copy-paste ladder

```bash
# freeze
git fetch --all --prune
git status --short
git branch --show-current
git rev-parse HEAD origin/main origin/dev

# import/runtime smoke
python - <<'PY'
import sys, platform, jax, jaxlib
import jaxfne as jtfne
print('python', sys.version)
print('platform', platform.platform())
print('jax', jax.__version__)
print('jaxlib', jaxlib.__version__)
print('devices', jax.devices())
print('x64', jax.config.jax_enable_x64)
print('jaxfne', jtfne.__version__)
PY

# verified selector/signal smoke
python - <<'PY'
import jax.numpy as jnp
import jaxfne as jtfne
cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
model = jtfne.construct(cfg)
idx = model.select(cell_type='E')
assert len(idx) >= 1
signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
n_units = int(signals.V_m.shape[-1])
assert signals.get('vm').shape[-1] == n_units
assert signals.get('spk').shape[-1] == n_units
assert signals.get('vm', cell_type='E').shape[-1] == len(idx)
assert jnp.isfinite(signals.get('vm')).all()
assert jnp.isfinite(signals.get('spk')).all()
print('selector_signal_smoke=PASS')
PY

python -m compileall -q jaxfne tests examples scripts

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest   tests/test_emitter_family_validation_v0330.py   tests/test_api_smoke.py   tests/test_identity_v0329.py   tests/test_selectors_v0329.py   tests/test_signals_get_v0329.py   -q --tb=short

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
rm -rf dist build
find . -maxdepth 2 -name '*.egg-info' -type d -exec rm -rf {} +
python -m build
python -m twine check dist/*
python - <<'PY'
from pathlib import Path
files = sorted(p.name for p in Path('dist').glob('*'))
assert len(files) == 2, files
assert all('0.3.29' in f for f in files), files
print('dist_sanity=PASS', files)
PY
```

## Stop rules

- Stop if branch is `agy`.
- Stop if `main` and `dev` diverge unexpectedly.
- Stop if tag peeled SHA differs from intended release SHA.
- Stop if README/docs public surface reintroduces repeated metadata clutter.
- Stop if unsupported path silently succeeds.
- Stop if any JSON export accepts NaN/Inf.
