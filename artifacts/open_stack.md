# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or problem, or is dropped.

## In flight

- O-4 artifacts/ reorg P5 (skills, harness, subagents, vocabulary, .lab into control/; manifest regenerated) | Muse | since 2026-10-15 | closes when on dev, `sync_skills.py --check` OK, dev CI green
- O-7 Gate-report and quick docs-gate commands (`scripts/`) | keeper | since 2026-10-15 | closes when both exist, are documented in `artifacts/AGENTS.md` and used by the broad gate
- O-8 Tree-degree test (at most 7 direct entries per folder under artifacts/, README exceptions) after O-4 | keeper | since 2026-10-15 | closes when the test is in the broad gate
- O-9 Destexhe 2009 lane 5d: S2 scoring; 6 uncommitted files in its clone on 6f85f5c3 (pre-reorg paths) | keeper | since 2026-10-14 | closes when integrated on dev (T-21) or recorded with a reason

## Waiting on others

- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls

- O-14 T-28 `tfne._RESERVED_LOWER` {x, y, h, s} defined, never read: wire into name checks or delete (score about 60 either way) | Hamm | since 2026-10-15 | closes when Hamm picks one
- O-15 T-20 `--check` accepted and ignored in `assert_release_freeze.py`, `sync_release_metadata.py` (score about 65 to change) | Hamm | since 2026-10-15 | closes when kept with a note or removed
- O-16 Branch `_temp_never-sweep` on origin, unmerged, 2026-10-12, owner unknown | Hamm | since 2026-10-15 | closes when its owner merges or deletes it

## Known gaps (recorded, no action yet)

- O-17 Frozen docs cite 6 paths that do not resolve, all predating the reorg: 3 in untracked `artifacts/developer/`, 3 pattern placeholders (`fig0X_...`) | keeper | since 2026-10-15 | closes when each is marked a placeholder or its target restored
- O-18 Over-7 folders kept whole as one series: `programme/receipts/{v0_4_23,v0_4_24,tfne,engine_053_054}`, `runs/perf/matrices_055/rerun/`, `tutorials/{etudes,versioned}`, `etudes/experiment_a` | keeper | since 2026-10-15 | closes when split by kind or listed as exceptions in the O-8 test
- O-20 Code markers: `scripts/run_delta_notebook_01.py:251` TODO (`cell_labels` from the model), `jaxfne/_config.py:1519-1528` UNCALIBRATED defaults ("todo 0c") | keeper | since 2026-10-15 | closes when each is done or moved to the todo stack with a reason
- O-21 `docs/citation.md:31,37` Zenodo DOI is a placeholder | Hamm | since 2026-10-15 | closes when a Zenodo DOI is minted and pasted, or the placeholder removed
- O-19 Scratch clones (58 `cl-*`) and finished Muse sessions | keeper | since 2026-10-15 | closes when deleted after O-3 to O-5 land
