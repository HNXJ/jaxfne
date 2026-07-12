# Zenodo release DOI

Zenodo archives GitHub Releases and mints DOIs. For jaxfne we care about two
properties:

1. **Updates with each public release** — keep the GitHub↔Zenodo hook enabled;
   every new **published** GitHub Release is archived automatically.
2. **One stable DOI for the project** — use Zenodo’s **Concept DOI**. It does
   not change when you ship `v0.4.7`, `v0.5.0`, …; it always resolves to the
   **latest** archived version.

Official reference: [Zenodo versioning FAQ](https://zenodo.org/help/versioning).

## Two DOI kinds (do not confuse them)

| Kind | Stays fixed? | Resolves to | Use for |
|------|--------------|-------------|---------|
| **Concept DOI** | Yes (one per project) | Landing page of the **latest** version | README badge, “cite this software” generically, `CITATION.cff` project id |
| **Version DOI** | No (new one each Release) | That exact snapshot | Papers / receipts that must pin a specific release |

Example pattern (numbers illustrative):

```text
v0.4.7 (version DOI):  10.5281/zenodo.1111111
v0.5.0 (version DOI):  10.5281/zenodo.2222222
Concept DOI (all versions): 10.5281/zenodo.0000000   ← put this in CITATION.cff
```

## What triggers an update

Zenodo watches **published GitHub Releases**, not bare git tags.

| Event | Zenodo archives? |
|-------|------------------|
| `git tag` / `git push --tags` only (e.g. internal `v0.4.6`) | No |
| GitHub Release published for a tag | Yes (new version DOI; Concept DOI unchanged) |
| Draft Release | No |

So: keep polishing with internal tags if you want; turn on archival when you
**publish** the real Release (planned for **0.4.7** after confirmation).

## One-time setup (already done if the repo is flipped ON)

1. [zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/)
2. Enable **HNXJ/jaxfne**
3. Confirm the repo shows as enabled

## After the first public Release (0.4.7+)

1. Publish the GitHub Release for the intended tag.
2. Open the Zenodo record → note both DOIs (version + concept).
3. Put the **Concept DOI** in repo metadata (stable project id):

```yaml
# CITATION.cff
doi: 10.5281/zenodo.CONCEPT_ID
# and/or
identifiers:
  - type: doi
    value: 10.5281/zenodo.CONCEPT_ID
    description: Concept DOI (all versions; resolves to latest)
```

4. Optionally also list the current **version DOI** in that release’s changelog
   notes — do not replace the Concept DOI in `CITATION.cff` on every bump.
5. Update [Citation](../citation.md) BibTeX with the Concept DOI for general
   cites; mention version DOI only when a paper needs a pinned snapshot.

## Policy for jaxfne

- **Concept DOI** = the static repo-level id (requirement 2).
- **Each public GitHub Release** = automatic Zenodo update (requirement 1).
- Internal tags without a GitHub Release do not update Zenodo.
- Do not invent a DOI before Zenodo mints one.
- Prefer `CITATION.cff` only (no `.zenodo.json` — Zenodo ignores CFF if both exist).

## Scope

Zenodo gives archival + citability. It is not peer review and does not change
proxy/scaffold truth gates.
