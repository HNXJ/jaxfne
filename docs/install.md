# Install

## PyPI

```bash
pip install -U jaxfne
```

The latest **PyPI** release is **`jaxfne==0.4.5`** (tag `v0.4.5`). The
**development tree** in this repository is **`0.4.6`** (internal git tag;
no PyPI upload for 0.4.6). To pin the last published PyPI release:

```bash
pip install "jaxfne==0.4.5"
```

For the current development checkout, use editable install (below) and verify
`jaxfne.__version__` prints `0.4.6`.

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
