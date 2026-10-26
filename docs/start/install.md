# Install

## PyPI

```bash
pip install -U jaxfne
```

The published **PyPI** release is **`jaxfne==0.5.8`** (tag `v0.5.8`),
built from commit `11957af`. The previous release is **`0.5.7`**
(tag `v0.5.7`). The
**development** public API on `dev` is documented in
[Public API contract](../reference/public_surface_contract.md) (208-symbol surface). To pin
the published PyPI release explicitly:

```bash
pip install "jaxfne==0.5.8"
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
