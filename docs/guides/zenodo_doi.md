# Zenodo release DOI

**Mint a permanent DOI** for each published jaxfne release using Zenodo's GitHub
integration. Zenodo reads metadata from root `CITATION.cff` when creating the deposit.

## Prerequisites

- GitHub admin access to [HNXJ/jaxfne](https://github.com/HNXJ/jaxfne)
- Zenodo account linked to the same GitHub identity
- `CITATION.cff` on `main` (or the branch Zenodo archives)

## Enable integration (once)

1. Visit [https://zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/).
2. Flip **ON** for `jaxfne` under the HNXJ organization/user.
3. Confirm the Zenodo badge appears on the GitHub repo (may take a few minutes).

Official reference: [Zenodo — GitHub integration](https://help.zenodo.org/docs/github/).

## Per tagged release

1. Verify release target SHA (see release-control doctrine in maintainer docs).
2. Publish the GitHub Release (annotated tag + release notes).
3. Wait for Zenodo to finish processing (email + Zenodo "Uploads" page).
4. Open the new Zenodo record; copy the **DOI** (`10.5281/zenodo.*`).
5. Update repository metadata:
   - `CITATION.cff` — add under `identifiers:` with `type: doi`, or set top-level `doi:`
   - [Citation](../citation.md) — replace placeholder BibTeX
   - Optional: mention DOI in `CHANGELOG.md` for that version

## What this does and does not do

| Provides | Does not provide |
|----------|------------------|
| Version-pinned, citable software archive | Peer-reviewed methods validation |
| Machine-readable metadata via `CITATION.cff` | Calibrated biophysical claim status |
| GitHub "Cite this repository" + Zenodo record | A journal article equivalent to jaxley's Nature Methods paper |

## Troubleshooting

- **No Zenodo deposit after release:** Check integration is still enabled; only
  **published** releases trigger archival (draft releases do not).
- **Wrong metadata on Zenodo:** Edit `CITATION.cff` before the next release;
  re-archive older versions manually on Zenodo if needed.
- **`.zenodo.json` present:** Zenodo ignores `CITATION.cff` if `.zenodo.json`
  exists — jaxfne uses `CITATION.cff` only (no `.zenodo.json`).
