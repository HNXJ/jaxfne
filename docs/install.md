# Install

## PyPI

```bash
pip install -U jaxfne
```

The latest **PyPI** release is **`jaxfne==0.4.8`** (tag `v0.4.8`). To pin it
explicitly:

```bash
pip install "jaxfne==0.4.8"
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
