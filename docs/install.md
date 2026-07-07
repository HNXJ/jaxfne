# Install

## PyPI

```bash
pip install -U jaxfne
```

The latest **PyPI** release is **`jaxfne==0.4.4`** (tag `v0.4.4`). The
**development tree** in this repository is **`0.4.5`** (`pyproject.toml`,
`jaxfne.__version__`). To pin the last published PyPI release for tutorials
that track PyPI:

```bash
pip install "jaxfne==0.4.4"
```

For the current development checkout, use editable install (below) and verify
`jaxfne.__version__` prints `0.4.5`.

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
