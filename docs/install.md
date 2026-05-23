# Install

## From PyPI

```bash
pip install jaxfne
```

For an exact release:

```bash
pip install jaxfne==0.2.30
```

## From GitHub

Install the GitHub release tag:

```bash
pip install git+https://github.com/HNXJ/jaxfne.git@v0.2.30
```

Or the latest development version from `main`:

```bash
pip install git+https://github.com/HNXJ/jaxfne.git@main
```

## From a local checkout

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e .
```

For development tools:

```bash
pip install -e .[dev]
```

## Verify installation

```python
import jaxfne
print(jaxfne.__version__)
```

Expected output: `0.2.30`
