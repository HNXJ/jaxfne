# open stack

Anything open right now. Read first at session start, before `todo_stack.md`. One line per item:
`O-n <what is open> | owner | since YYYY-MM-DD | closes when <check>`. Add when it opens, delete when
it closes; at most 30 lines; a line older than 3 days moves to todo or the issue log, or is dropped.
Each line has its todo row, named in the line.

## Hamm's calls (0.5.10 review, asked one at a time in this order)

- O-95 CI `Dashboard` red on dev since 960a0fab (2026-11-07; 960a0fab, c2b0df84, 619ea11a, d2e67b2d):
  `rerun_etude_figures.py --etude heterogeneous` exits "hei P_Q centroid outside tolerance" (`:326`). Bisected
  locally: passes at 2e1af6f8^, fails at 2e1af6f8 and HEAD. 2e1af6f8 made `construct()` pass the stable-mode x
  bounds (±10; was ±1e6 by the dropped kwargs); the etude runs `bound_mode="stable"`
  (`run_heterogeneous_emitters_etude.py:124`), its bundle `metrics_post_p048.json` was pinned under ±1e6. Options:
  (1, recommended) regenerate the bundle and docs figures under the fixed bounds, recording the rail fraction in
  its README; (2) the etude declares `x_min/x_max=±1e6` explicitly, keeping the old figure as a declared variant;
  (3) the etude switches to `bound_mode="minimal"` (unaffected by 2e1af6f8 per its commit message; the figure
  changes to the minimal regime). Hamm 2026-11-09: option 1. | keeper |
  since 2026-11-09 | closes when Dashboard CI is green on dev. Todo: T-8.
