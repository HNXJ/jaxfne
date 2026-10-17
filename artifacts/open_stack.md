# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or problem, or is dropped.

## In flight

- O-4 artifacts/ reorg P5: skills, harness, subagents, vocabulary in `control/` (`.lab/` stays: the Labyrinth tool reads `artifacts/.lab`); on dev, `sync_skills.py --check` OK | keeper | since 2026-10-15 | closes when dev CI is green on the push

## Waiting on others

- O-10 Harness proposals (lane-clone helper, worker background guard, jchat lane registry, lock holders in `who`, known-failures register, release events), jchat conference #29 | jynx keeper | since 2026-10-15 | closes when each is taken or refused

## Hamm's calls

- O-14 T-28 `tfne._RESERVED_LOWER` {x, y, h, s} defined, never read: wire into name checks or delete (score about 60 either way) | Hamm | since 2026-10-15 | closes when Hamm picks one
- O-15 T-20 `--check` accepted and ignored in `assert_release_freeze.py`, `sync_release_metadata.py` (score about 65 to change) | Hamm | since 2026-10-15 | closes when kept with a note or removed
- O-16 Branch `_temp_never-sweep` on origin, unmerged, 2026-10-12, owner unknown | Hamm | since 2026-10-15 | closes when its owner merges or deletes it
- O-23 `fact_stack.md:175-176` names `artifacts/skills`; since P5 it is `artifacts/control/skills` (Hamm edits that file) | Hamm | since 2026-10-16 | closes when the two paths are updated

## Known gaps (recorded, no action yet)

- O-17 Frozen docs cite 6 paths that do not resolve, all predating the reorg: 3 in untracked `artifacts/developer/`, 3 pattern placeholders (`fig0X_...`) | keeper | since 2026-10-15 | closes when each is marked a placeholder or its target restored
- O-22 Over-7 lane folders excepted in `tests/test_artifacts_tree_degree.py` while their lanes are open: `etudes/papers/caminos/` (19 files), `etudes/papers/izhikevich2003/` (13) | keeper | since 2026-10-16 | closes when each is split by kind or its exception gets a path-addressing reason
- O-20 Code markers: `scripts/run_delta_notebook_01.py:251` TODO (`cell_labels` from the model), `jaxfne/_config.py:1519-1528` UNCALIBRATED defaults ("todo 0c") | keeper | since 2026-10-15 | closes when each is done or moved to the todo stack with a reason
- O-19 Scratch clones (58 `cl-*`) and finished Muse sessions | keeper | since 2026-10-15 | closes when deleted after O-3 to O-5 land
