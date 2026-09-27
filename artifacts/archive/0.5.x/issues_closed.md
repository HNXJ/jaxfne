# Closed issues after 2026-09-27

Earlier closed issues: `artifacts/archive/0.5.x/ISSUE_LOG_2026-09-27.md`.

### P-011
- **date:** 2026-09-25
- **type:** FRICTION
- **area:** Kaleido static export under xdist load
- **observation:** `test_vis_smoke[exporters.export_figures]` failed once in the
  broad gate with `RuntimeError: Couldn't close or kill browser subprocess`
  (choreographer/Kaleido browser teardown), on a tree that changed neither
  `jaxfne/vis/exporters.py` nor the test.
- **severity:** MINOR (teardown of the headless browser; no output semantics)
- **minimal reproduction:** broad gate (`-n auto`); alone it passes 3/3
- **expected behavior:** stable PASS
- **actual behavior:** intermittent FAIL under parallel load
- **evidence:** broad gate on `d0d7360` 2026-09-25 (1 failed / 4430 passed); isolated reruns 3/3 PASS. Recurred 2026-09-26 (broad gate before 99482f7) and 2026-09-27 (before the AT-10-N20 commit), each isolated rerun PASS: a gate fix is now due
- **possible future change:** retry Kaleido teardown once, or serialize Kaleido tests (xdist group); open
- **resolution (2026-09-27, agent):** `jaxfne/vis/exporters.py` retries `write_image` once on exactly this teardown error; other errors propagate (`tests/test_vis_export_teardown_retry.py`).

