# jaxfne-tutorial-executor

**Triggers:** run tutorial, smoke, execute notebook, verify example artifacts.

**Purpose:** Prove tutorials run through the public API and emit claimed PNG/JSON/HTML — not just parse.

**Commands:**

```bash
export PYTHONPATH=.
python scripts/run_tutorial_smoke.py 2>&1 | tail -30
```

Inventory outputs after execution; no hidden local-path success.

**Full skill:** user-installed `jaxfne-tutorial-executor`.
