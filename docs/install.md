# Install

## PyPI

```bash
pip install -U jaxfne
```

The current published **PyPI** release is **`jaxfne==0.4.23`** (tag `v0.4.23`),
published from the artifacts built and validated for commit `7e91da3`. The
previous release is **`0.4.22`** (tag `v0.4.22`). The current **development**
public API on `dev` is documented in
[Public API contract](public_surface_contract.md) (190-symbol surface). To pin
the published PyPI release explicitly:

```bash
pip install "jaxfne==0.4.23"
```

For the current development checkout, use editable install (below) and verify
`jaxfne.__version__` prints the version in `pyproject.toml`.

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
