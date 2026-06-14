# jaxfne loop context bundle

**Read first:** [`AGENT_QUICKREF.md`](AGENT_QUICKREF.md)

Deep reference: [`JAXFNE_BIOPHYSICS_GLOSSARY.md`](JAXFNE_BIOPHYSICS_GLOSSARY.md). Live publication snapshot: [`CURRENT_PUBLICATION_STATE.md`](CURRENT_PUBLICATION_STATE.md).

Contract-first context for autonomous ticks and downstream agents. `00_MANIFEST.md` documents bundle provenance and trust order.

## Trust order (non-negotiable)

**live git on the task branch > `CURRENT_PUBLICATION_STATE.md` > this bundle > original ZIP/PDF/checklist** (context only).

Always `git fetch --all --prune` and re-freeze before mutation. Publication track uses **`cur`**; `01–08` bundle files were reconciled against `main`/`dev` history and may lag `cur`.

## Files

| file | use |
|---|---|
| `AGENT_QUICKREF.md` | **session entry** — freeze, gates, smoke, stop rules |
| `CURRENT_PUBLICATION_STATE.md` | live publication inventory snapshot (refresh SHA) |
| `JAXFNE_BIOPHYSICS_GLOSSARY.md` | deep biophysics, scoreboard, ED ladder (on demand) |
| `00_MANIFEST.md` | bundle provenance, input hashes, freshness |
| `01_REPO_MAP.md` | module map (re-verify paths against live git) |
| `02_PUBLIC_API_CONTRACT.md` | root `__all__`, signatures |
| `03_JAX_RUNTIME_CONTRACT.md` | PRNG/scan/vmap/jit contract |
| `04_TRUTH_GATES_AND_CLAIMS.md` | truth gates + claim contract |
| `05_BACKLOG.md` | ranked backlog (`main`/`dev` reconciliation) |
| `06_VALIDATION_LADDER.md` | validation commands + receipts |
| `07_V0330_ARCHITECTURE_NOTES.md` | gated architecture plans |
| `08_RISKS_AND_FRAGILITIES.md` | fragile spots |
| `PUBLICATION_READINESS_SCOREBOARD.md` | pointer to external review scoreboard |
| `JAXFNE_BACKLOG_AND_WORKER_PROMPTS.md` | post-ED10 follow-ups (ED9/ED10 archived) |
| `REPO_INSPECTION_REPORT.md` | **superseded** zip inspection record |

## Current state (publication track)

- Branch: `cur`, version `0.3.29`
- Inventory: 8/8 main figures + 10/10 Extended Data (verify with `evidence_inventory.py`)
- ED9/ED10: complete; release/tag/publish/archive approval-gated
- Package backlog on `main`/`dev`: B01 merged; B02 next GREEN; B07–B09 gated

## Maintenance

When backlog items ship, update `05_BACKLOG.md` and `08_RISKS_AND_FRAGILITIES.md`. Refresh `CURRENT_PUBLICATION_STATE.md` SHA when publication-facing docs change.
