# Install

## PyPI

```bash
pip install -U jaxfne
```

The current release is **`jaxfne==0.3.40`** (tag `v0.3.40`), published to PyPI as
both a wheel (`jaxfne-0.3.40-py3-none-any.whl`) and an sdist
(`jaxfne-0.3.40.tar.gz`). To pin the exact release used by the repository
tutorials:

```bash
pip install "jaxfne==0.3.40"
```

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
