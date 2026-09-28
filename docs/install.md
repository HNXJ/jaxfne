# Install

## PyPI

```bash
pip install -U jaxfne
```

The published **PyPI** release is **`jaxfne==0.5.0`** (tag `v0.5.0`),
built and validated for commit `499c54f`. The previous release is **`0.4.25`**
(tag `v0.4.25`). The **development** public API on `dev` is documented in
[Public API contract](public_surface_contract.md) (192-symbol surface). To pin
the published PyPI release explicitly:

```bash
pip install "jaxfne==0.5.0"
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
