# 00_MANIFEST

> **Session entry:** [`AGENT_QUICKREF.md`](AGENT_QUICKREF.md). Publication track uses branch **`cur`**; this manifest reconciles `main`/`dev` bundle history.

Purpose: self-contained context bundle for a downstream Claude Code agent working on `jaxfne` with high intelligence-per-token: frozen contracts, evidence map, validation ladder, and ranked backlog.

Generated (external expert): 2026-06-05T01:03:28Z
Reconciled against live git: 2026-06-04T20:11:00-05:00 @ `main == dev == fab4c9c`, tag `v0.3.29`.

## Live reconciliation (READ FIRST)

This bundle was authored from the stale `jaxfne-main.zip` (package `0.3.27`, no git SHA). It has since been reconciled against the live working tree at `fab4c9c` (`jaxfne 0.3.29`). Deltas applied:

- **B01 (objective null RNG reproducibility) is MERGED** — PR #22 merged into `main` (merge commit `33f99db`): explicit keyword-only `rng` threaded through all six null generators plus a `null_seed` dispatcher/factory parameter. (Related: fields-helper dedup shipped separately as PR #23, merged `e29c604`.) The next ready GREEN item is now **B02**. See `05_BACKLOG.md`.
- Validation receipts refreshed to the post-PR#22 tree: focused `54 passed`, full suite `1986 passed, 66 skipped, 4 xfailed`. See `06_VALIDATION_LADDER.md`.
- `03_JAX_RUNTIME_CONTRACT.md` grep tables populated with live audit counts (PRNG 47, scan 12, vmap 34, jit 10, pytree 23, np.random 16).

Audited source status:
- External-expert source: uploaded `jaxfne-main.zip`, extracted to `repo/jaxfne-main` (stale, package `0.3.27`).
- Live audit report (`Pasted markdown.md`) states live git `main == dev == tag v0.3.29 == fab4c9c`, clean tree, `agy` untouched at `c689ed0`; see `Pasted markdown.md:L30`.
- The downstream agent MUST run `git fetch --all --prune` and re-freeze `main/dev/tag` before any mutation; never act on the line/SHA references in this bundle without re-verifying against live git.

## Input hashes

|input|sha256|
|---|---|
|jaxfne-main.zip|51783154382a11dffc5f20377f9ca9f31d14ad80878cc89f08d0dfd03f577e04|
|jaxfne_longterm_biophysical_plan_bundle.zip|d3b5c365489020992071e2ce5ad0e47ef5226cd0cbcffd73c0363d6a12872507|
|jaxfne_v0328_v0330_20factor_checklist.csv|3317f7974eb13f800b32754009d5c487e30c0d5a829300a3f8d002b08ecc861b|
|jaxfne_v0328_v0330_assessment_plan.md|2a51c06a221db1f67a1da17d8533fa39c549bcd24fbe25fd8996a7a9e414c8ab|
|2026_jtfne_arxiv.pdf|dab524c7e4a1b92797b49f3b30725eccfbd821d5c7ae805f73fc454584cdac23|
|Pasted markdown.md|9c2baeb897968b000d1894506606786cb58ee13757b1b6cacf5dbef753fcd123|


## Bundle files

- `README.md` - directory index + trust order
- `01_REPO_MAP.md` - 01_REPO_MAP
- `02_PUBLIC_API_CONTRACT.md` - 02_PUBLIC_API_CONTRACT
- `03_JAX_RUNTIME_CONTRACT.md` - 03_JAX_RUNTIME_CONTRACT
- `04_TRUTH_GATES_AND_CLAIMS.md` - 04_TRUTH_GATES_AND_CLAIMS
- `05_BACKLOG.md` - 05_BACKLOG
- `06_VALIDATION_LADDER.md` - 06_VALIDATION_LADDER
- `07_V0330_ARCHITECTURE_NOTES.md` - 07_V0330_ARCHITECTURE_NOTES
- `08_RISKS_AND_FRAGILITIES.md` - 08_RISKS_AND_FRAGILITIES

## Bundle file hashes (reconciled copies @ 2026-06-04; `00_MANIFEST.md` self-excluded)

SHA256 of the reconciled files as committed. The manifest excludes itself (its
hash changes when this table is written). Recompute with
`shasum -a 256 internal_docs/loop_context/*.md`.

|file|sha256|
|---|---|
|README.md|047bed63f69c52fbd1fe15834917ed69ba9ca18310111b21b97212181bf29038|
|01_REPO_MAP.md|4af4a0af063180bbf868583d1a90c7abc5ceb2ceb249348e379702eeb8e78eb1|
|02_PUBLIC_API_CONTRACT.md|e7cd07859814620ff0cb6bb989e0169e90bc2a3a44c4564c8c5a1edb7e3fb7a0|
|03_JAX_RUNTIME_CONTRACT.md|8320ff1c96d8da7a6d5d11c02ed344fc3f720726b6eee795cb5731136ecedfa1|
|04_TRUTH_GATES_AND_CLAIMS.md|64b4a5681b682dcbc6d531d759e1aa4188655c252cf118843927d949df9fd2a6|
|05_BACKLOG.md|16e1d4425d6a9f1e20a1afe3df508e2d448ae6189ef0bb66fdbefb011d7ed133|
|06_VALIDATION_LADDER.md|2e1b3b258cb3db622d3c3c0961956915abf31ce52079fa72d75ca89c2cc1baeb|
|07_V0330_ARCHITECTURE_NOTES.md|ba1cc4006beffe1bfbdc9d10face5562439e5d8113555e151be060ba9a0e949b|
|08_RISKS_AND_FRAGILITIES.md|875555d01b5790185b68e7ea8801a8298a35bdba54fa8dc8fce8bb6171a143b8|


## Freshness note

- `jaxfne-main.zip` is stale relative to live git: the uploaded ZIP reports package `0.3.27` and the assessment plan states the old ZIP had no git SHA and only partial validation. Live is `jaxfne 0.3.29` at `fab4c9c`, full validation, split `optim/` + `vis/` modules.
- **Bundle-internal hashes below describe the ORIGINAL external-expert artifacts.** Files in this directory have since been reconciled against live git (see "Live reconciliation"), so their on-disk SHA256 now differs from the original-source table. Treat the table as provenance of the upstream source, not as a checksum of these reconciled copies; the authoritative source of truth is always live git, re-frozen per tick.
- Trust order: **live git > this reconciled bundle > original ZIP/PDF/checklist/assessment markdown** (context only).
