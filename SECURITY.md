# Security Policy

## Supported versions

jaxfne is under active development; only the latest released version on
[PyPI](https://pypi.org/project/jaxfne/) receives security fixes. There is no
long-term-support branch.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a suspected security
vulnerability. Instead, use GitHub's private reporting:

1. Go to the [Security tab](https://github.com/HNXJ/jaxfne/security) of this
   repository.
2. Click **"Report a vulnerability"** to open a private advisory.

If that's unavailable, open a regular issue asking a maintainer to open a
private channel, without describing the vulnerability itself.

## Scope

jaxfne is a computational-neuroscience simulation library (JAX-based). It has
no network-facing service, no authentication layer, and does not process
untrusted remote input by default. The realistic security surface is:

- Deserializing untrusted config/manifest JSON or NWB files
- Dependency vulnerabilities in `jax`/`jaxlib`/`numpy`/`scipy` and optional
  extras (`matplotlib`, `plotly`, `jaxley`, etc.)
- Arbitrary code execution via notebook/script inputs a user chooses to run

Simulation-correctness bugs (wrong numeric output, non-convergent dynamics)
are **not** security issues — file those as regular
[GitHub issues](https://github.com/HNXJ/jaxfne/issues).
