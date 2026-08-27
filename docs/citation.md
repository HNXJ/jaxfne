# Citation

If you use jaxfne in your research, please cite the software. Machine-readable
metadata lives in the repository root [`CITATION.cff`](https://github.com/HNXJ/jaxfne/blob/main/CITATION.cff)
— GitHub surfaces a **Cite this repository** button from that file.

## Software citation (current)

Until a Zenodo DOI is minted (see below), cite the GitHub release or PyPI package:

```bibtex
@software{jaxfne2026,
  author = {Nejat, Hamed},
  title = {jaxfne: JAX Field Neural Equations},
  year = {2026},
  url = {https://github.com/HNXJ/jaxfne},
  version = {0.4.17},
  note = {Computational scaffold / proxy readouts; tag v0.4.17}
}
```

After the first Zenodo archival of a tagged release, add the DOI line from the
Zenodo record, for example:

```bibtex
@software{jaxfne2026,
  author = {Nejat, Hamed},
  title = {jaxfne: JAX Field Neural Equations},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/HNXJ/jaxfne},
  version = {0.4.17}
}
```

Replace `10.5281/zenodo.XXXXXXX` with the real DOI from [Zenodo](https://zenodo.org) —
do not invent a DOI before it exists.

## Zenodo DOI (maintainers)

Permanent version DOIs are minted through the **GitHub–Zenodo** integration when a
**GitHub Release** is published and archived. This improves citability (versioned,
citable snapshots) but is **not** equivalent to a peer-reviewed methods paper.

**One-time setup** (repository admin):

1. Sign in at [zenodo.org](https://zenodo.org) with your GitHub account.
2. Open [Zenodo GitHub settings](https://zenodo.org/account/settings/github/).
3. Enable the **HNXJ/jaxfne** repository.
4. Ensure root `CITATION.cff` is on the default branch (Zenodo imports metadata from it).

**Per release** (after step 10 publish authorization):

1. Create/publish a GitHub Release for the tag (e.g. `v0.4.7`).
2. Zenodo builds a deposit automatically; confirm on the Zenodo project page.
3. Copy the new `10.5281/zenodo.*` DOI into:
   - `CITATION.cff` (`identifiers` entry with `type: doi`, or top-level `doi:`)
   - This page (replace the placeholder BibTeX above)
   - Optional: `README.md` citation line

Detailed checklist: Zenodo release DOI guide (`docs/guides/zenodo_doi.md` — repository-internal reference, excluded from the built site).

## Component citations

If using specific models or frameworks:

- **Izhikevich model:** Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks.
- **JAX:** Bradbury et al. (2018). JAX: composable transformations of Python+NumPy programs. https://github.com/google/jax
- **Jaxley (if used):** See https://jaxley.readthedocs.io
