# Review Navigation Manifest

Location of every artifact an external reviewer needs for the Figure 1-7
publication snapshot, and the exact command to regenerate each layer. All
measurements are stated as Absolute (exact value) or Relative (comparison
between specific values); nothing else.

Snapshot reference: commit listed in `artifacts/publication/equivalence_report.json`
with `schema: jaxfne.harness.seam_equivalence.v1`. Frozen evidence baseline:
`artifacts/publication/frozen_manifest.json` (`frozen_at_head` records the manifest head).

## 1. Source of truth and immutable evidence

| Path | Role |
|---|---|
| `artifacts/publication/frozen_manifest.json` | 28 frozen files (7 figure PNGs, 21 JSON evidence artifacts) with exact SHA-256 per file |
| `artifacts/figures/publication/fig0{1..7}_*.png` | Frozen scientific figure PNGs (immutable bytes, recorded in the manifest) |
| `artifacts/publication/fig0{1..7}_*_{spec,audit,receipt}.json` | Per-figure semantic spec, semantic audit (`status: PASSED`), generation receipt |
| `artifacts/publication/figures_1_7_cross_figure_audit.json` | Cross-figure layout audit over figures 1-7 |
| `artifacts/publication/publication_evidence_index.json` | Evidence index (status FROZEN, `write_once: True`) |

## 2. Generator seam and the two-level invariant

- Generators: `scripts/publication_figures/fig0{1,2_04,5,6,7}_*.py`, each exposing
  `build_figure*()` (Figure-object seam) plus `main()`/`draw_figure*()` wrappers.
- Level 1 (exact): the seam refactor changed no scientific content —
  `scripts/publication_figures/equivalence_gate.py` renders all 7 figures from
  `build_figure*()` and requires W, H, RGBA elementwise identity (zero
  tolerance) against the frozen PNGs, plus SHA-256 equality recorded per case.
  Report: `artifacts/publication/equivalence_report.json` — 7/7 cases
  `decoded_pixel_equal: true`.
- Level 2 (Relative, presentation-only): `artifacts/figures/publication/final/fig0{1..7}_*.png`
  (300 DPI) and `fig0{1..7}_*.pdf` (vector) are deliberately re-rendered from the
  same `build_figure*()` with artist-only polish and therefore differ from the
  frozen PNGs at the pixel level.

## 3. Polish layer (presentation-only, Absolute values)

| Path | Role |
|---|---|
| `scripts/publication_figures/polish/run_polish.py` | Orchestrates polish: clip check, palette check, font floor, inset geometry, frozen write-guard |
| `scripts/publication_figures/polish/_polish_common.py` | Shared clip/color/font utilities |
| `artifacts/publication/polish/fig0{1..7}_polish_{spec,audit,receipt}.json` | Per-figure polish spec, audit (all 7 `PASSED`), receipt |
| `artifacts/publication/polish/figures_1_7_final_layout_audit.json` | Cross-layout audit of the final set (status `PASSED`) |
| `artifacts/figures/publication/final/fig0{1..7}_*.png` | Final 300-DPI raster renders (`dpi_meta: [300.0, 300.0]`) |
| `artifacts/figures/publication/final/fig0{1..7}_*.pdf` | Final vector renders |

## 4. Regeneration instructions (run from repo root)

Environment: install jaxfne from source with the figure stack declared in
`pyproject.toml` (Pillow is used directly by the equivalence gate and figure
generators; `jaxfne[viz]` carries the imaging/plot stack). The frozen figure
set was rendered with `matplotlib>=3.10.9,<3.11` and `scipy==1.17.1`, and both
are pinned in the `viz`/`dev` extras: matplotlib version changes alter text
rendering bytes, and scipy 1.18 changes the fig05 estimator numerics that feed
its drawn annotations — either pin drift makes the equivalence gate fail on
byte-identity (verified by clean-room bisection, 2026-08-15). The canonical
Experiment A dataset (`canonical_source.npz`) is deliberately tracked as a frozen
publication-evidence exception (`.gitignore` keeps all other `*.npz` out; the
exception is the single line
`!artifacts/etudes/experiment_a/canonical_source.npz`, added 2026-08-16 so fresh
checkouts and CI never need to regenerate it). Its SHA-256 is recorded in
`artifacts/etudes/experiment_a/b1_canonical_receipt.json` and its `q_hash` in the
fig02–04 cross-figure audit. Regeneration remains independently available for
verification from a fresh clone with
`python3 scripts/run_experiment_a.py`; figures 2-4 consume that bundle.
That runner is verification-first: canonical arrays regenerate into a local
`.npz`; committed receipts are compared (hashes byte-exact, analytic metrics
within rtol 1e-6, run-stamps ignored) and never rewritten — the reviewer tree
stays clean.

```bash
# Gate 1: seam equivalence (7/7 exact decoded-pixel identity vs frozen)
python3 scripts/publication_figures/equivalence_gate.py

# Gate 2: per-figure semantic audit + receipts (write-once enforced)
python3 scripts/publication_figures/figures_1_7_cross_audit.py

# Gate 3: polish pipeline -> artifacts/figures/publication/final/** + polish artifacts
python3 scripts/publication_figures/polish/run_polish.py

# Tests that exercise the layers
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_equivalence_gate_v20260815.py \
  tests/test_fig01_grammar.py tests/test_fig02_04_experiment_a.py \
  tests/test_fig05_protocol_c.py tests/test_fig06_hwd_evidence.py \
  tests/test_fig07_e_integration.py tests/test_figures_1_7_cross_audit.py \
  tests/test_publication_claim_ledger_v20260815.py \
  -q --tb=short
```

The equivalence gate writes renders to `scratch/equivalence_render/` (gitignored)
and its report to `artifacts/publication/equivalence_report.json` (tracked).

The polish runner refuses any write into a path listed in
`artifacts/publication/frozen_manifest.json` (enforced in `_polish_common.py`) and writes only
under `artifacts/figures/publication/final/` and `artifacts/publication/polish/` (both
designated writable in the checkpoint). The cross-audit script and the other
generators (`fig0{1,2_04,5,6,7}_*.py`) now enforce the same write-once rule via
`_pub_figure_common.write_json_strict`: a regeneration whose content matches the
frozen artifact (modulo the run-stamp fields `repo_head` / `audited_at_utc`) is a
no-op, and any real content drift is a hard error — frozen bytes are never
rewritten in place, and the reviewer reproduction path terminates in a clean
tree.

Provenance semantics of the tracked evidence:

- `equivalence_report.json` — **reproducible derived validation evidence**
  (regenerated by the tracked gate on every run and matched by
  `tests/test_equivalence_gate_v20260815.py`). It is intentionally NOT part of
  the frozen set: it is a validator's report, not primary evidence.
- `pec_consolidation_receipt.json` — historically correct at creation
  (16 panels, `next_checkpoint: figure_1_generation`,
  `figure_rendering_authorized: false`, `artifact_commit_sha: c6d4c893...`).
  Write-once: it records what was true then. The claim-level current state
  supersedes it for status questions — see
  `publication_claim_ledger.json` (schema `jaxfne.publication.claim_ledger.v1`,
  status `ACTIVE`).
- Version axes: the package version (`0.4.16` in `pyproject.toml` /
  `jaxfne._model.py`) and the publication milestone (`0.4.17` in
  `docs/publication/evidence_consolidation.md`) are distinct identifiers with
  no ambiguity; both are current in their own axis.

## 5. Excluded from this snapshot

- `scratch/` and `artifacts/developer/` (local task state, gitignored).
- `artifacts/figures/publication/final/` PDF font sub-setting details: the PDFs embed
  `%PDF` header and render as vector; embedded `/Image` objects are the
  expected raster heatmap panels, recorded in the polish receipts.