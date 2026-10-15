# Install

## PyPI

```bash
pip install -U jaxfne
```

The published **PyPI** release is **`jaxfne==0.5.5`** (tag `v0.5.5`),
built from commit `28ccee1`. The previous release is **`0.5.0`**
(tag `v0.5.0`). The
**development** public API on `dev` is documented in
[Public API contract](public_surface_contract.md) (208-symbol surface). The
development tree is at **0.5.6**, a release candidate unpublished on
PyPI; the published release stays at 0.5.5 above. To pin
the published PyPI release explicitly:

```bash
pip install "jaxfne==0.5.5"
```

For a development checkout, use the editable install below and check
`jaxfne.__version__` against the version in `pyproject.toml`.

Optional extras:

```bash
pip install "jaxfne[viz]"      # plotting
pip install "jaxfne[opt]"      # Optax adapters
pip install "jaxfne[dev]"      # tests and lint helpers
```

## Source checkout

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e .[dev,viz,opt]
```

## Verify

```python
import jaxfne as jtfne
print(jtfne.__version__)
```
