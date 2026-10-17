# Install

## PyPI

```bash
pip install -U jaxfne
```

The published **PyPI** release is **`jaxfne==0.5.6`** (tag `v0.5.6`),
built from commit `b24e09e`. The previous release is **`0.5.5`**
(tag `v0.5.5`). The
**development** public API on `dev` is documented in
[Public API contract](../reference/public_surface_contract.md) (208-symbol surface). To pin
the published PyPI release explicitly:

```bash
pip install "jaxfne==0.5.6"
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
