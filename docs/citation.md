# Citation

If you use jaxfne, cite the software. Machine-readable
metadata lives in [`CITATION.cff`](https://github.com/HNXJ/jaxfne/blob/main/CITATION.cff)
at the repository root — GitHub derives its **Cite this repository** button from that file.

## Software citation (current)

Every published GitHub Release is archived on Zenodo. All versions share one
concept DOI, [10.5281/zenodo.21448155](https://doi.org/10.5281/zenodo.21448155),
which resolves to the latest version. Cite it when referring to jaxfne as a project:

```bibtex
@software{jaxfne2026,
  author = {Nejat, Hamed},
  title = {jaxfne: JAX Field Neural Equations},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.21448155},
  url = {https://github.com/HNXJ/jaxfne},
  version = {0.5.6}
}
```

When a result depends on an exact release, cite that release's version DOI
instead; the Zenodo record's **Versions** panel lists one per release.

## Zenodo DOI (maintainers)

Permanent version DOIs come from the **GitHub–Zenodo** integration when a
**GitHub Release** is published and archived. A versioned snapshot is citable as software; a peer-reviewed methods paper is a separate citation.

**One-time setup** (repository admin):

1. Sign in at [zenodo.org](https://zenodo.org) with your GitHub account.
2. Open [Zenodo GitHub settings](https://zenodo.org/account/settings/github/).
3. Enable the **HNXJ/jaxfne** repository.
4. Ensure root `CITATION.cff` is on the default branch (Zenodo imports metadata from it).

**Per release** (after step 10 publish authorization):

1. Create/publish a GitHub Release for the tag (e.g. `v0.4.7`).
2. Zenodo builds a deposit automatically under the concept DOI; confirm on the Zenodo project page.
3. The concept DOI in `CITATION.cff`, `README.md` and this page stays unchanged.

Detailed checklist: Zenodo release DOI guide (`docs/guides/zenodo_doi.md` — repository-internal reference, excluded from the built site).

## Component citations

For specific models or frameworks:

- **Izhikevich model:** Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks.
- **JAX:** Bradbury et al. (2018). JAX: composable transformations of Python+NumPy programs. https://github.com/google/jax
- **Jaxley (if used):** See https://jaxley.readthedocs.io
