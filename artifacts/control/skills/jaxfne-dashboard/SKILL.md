---
name: jaxfne-dashboard
description: Refresh the jaxfne lane dashboard artifact (one card per replication lane, one row per saved output) after a push.
metadata:
  audience: agents
---
# jaxfne lane dashboard refresh

## WHEN
After a push to `dev` that touches `artifacts/etudes/papers/`, `dashboard/`, `artifacts/todo_stack.md`,
`artifacts/open_stack.md` or saved outputs under `artifacts/`; or when Hamm asks for a refresh.

## AUTHORITIES
`scripts/lane_dashboard/jaxfne_dashboard.py` (data), `scripts/lane_dashboard/template/` (page),
`scripts/lane_dashboard/standalone_page.py` (packing). Tests: `tests/test_lane_dashboard*.py`.

## STEPS
1. Read the pushed tree, never a working tree with local edits: a clean clone or worktree of `dev` at the
   pushed SHA (`git rev-parse HEAD` equals the pushed SHA).
2. Generate outside the repository (the script refuses an `--out` inside it):
   `python scripts/lane_dashboard/jaxfne_dashboard.py --out <scratch>/dash`
   Read its three summary lines: row counts with the commit and read time, lanes not in the registry,
   lanes without a render. Each is a finding to report, not to fix here.
3. Pack: `python scripts/lane_dashboard/standalone_page.py --dash <scratch>/dash --out <scratch>/pub`
4. Sizes: every `doc_*.json` at most 256 KB, `lanes.json` and `analyses.json` at most 8 MB (the generator
   exits non-zero when one is over).
5. Publish: Artifact publish with `file_path` `<scratch>/pub/index.html`, `url` the existing dashboard
   artifact (keeps its link), `files` {`app.js`, `lanes.json`, `analyses.json`} from `<scratch>/pub/`.
   A first publish (no url) needs Hamm's ask; never pin unless he asks.
6. Report: the link, lane and analysis row counts, the commit read, and the generator's not-found lines.

## RULES
- Every value is read by the generator from a file; never type a number into the template or the script.
- Page numbers are dataset marks: KPIs from `dash.calc` with data-source, data-field, data-where; tables
  with data-source and data-row-key, each `<th>` a data-field. Data text goes in with textContent.
- A render date is not a review status; a receipt count is not freshness. jaxfne has no per-receipt
  staleness judge; `scripts/build_dashboard_data.py --check` judges the registry document only.
- Template change: edit `template/`, run `node --check scripts/lane_dashboard/template/app.js` and the
  three test files, then republish.
