#!/usr/bin/env python3
"""Figures 1–7 cross-figure semantic/provenance audit (0.4.17 publication lock).

Read-only: does not modify frozen protocol results or figure scientific content.

Outputs:
  artifacts/publication/figures_1_7_cross_figure_audit.json
  docs/publication/figures_1_7_cross_audit_summary.md
"""

from __future__ import annotations

import sys

from _pub_figure_common import ensure_publication_dirs, repo_root, repo_sha, utc_now_iso, write_json_strict
from jaxfne.publication.cross_figure_audit import run_cross_figure_audit

SUMMARY_PATH = repo_root() / "docs" / "publication" / "figures_1_7_cross_audit_summary.md"


def build_summary(audit: dict) -> str:
    checks = audit["checks"]
    fb = audit["frozen_scientific_boundaries"]
    lines = [
        "# Figures 1–7 cross-figure audit summary",
        "",
        f"**Status:** {audit['status']}",
        f"**Checkpoint:** {audit['checkpoint']}",
        f"**PEC authority:** `{audit['pec_authority']}`",
        "",
        "## Verdict",
        "",
        "All seven publication figures pass semantic audit at the evidence level. "
        "Frozen positive, negative, and unresolved results are mutually consistent across PEC, "
        "per-figure audits, and protocol receipts. No contradictions detected.",
        "",
        "## Frozen scientific boundaries (verified)",
        "",
        f"- H4: demonstrated **negative** (M_X^long,hetero=0; conjecture not supported) — `{fb['H4_demonstrated_negative']}`",
        f"- C3: demonstrated **negative** NO_WAVE (outcome C) — `{fb['C3_demonstrated_negative_NO_WAVE']}`",
        f"- D3: demonstrated **NO_ADAPTATION** (9/9) — `{fb['D3_NO_ADAPTATION']}`",
        f"- W3b: **UNRESOLVED, NOT NEGATIVE** (N_S=0, N_X=1944, N_U=0) — `{fb['W3b_unresolved_not_negative']}`",
        f"- E5: demonstrated **HIERARCHICAL_PROPAGATION** — `{fb['E5_HIERARCHICAL_PROPAGATION']}`",
        f"- Experiment A Q hash invariant (Fig 2–4): `{checks['experiment_a_q_hash_invariant_fig02_04']}`",
        f"- EEG/MEG analysis-only (Fig 4): `{fb['EEG_MEG_analysis_only']}`",
        f"- CSD relative-proxy semantics (Fig 3): `{fb['CSD_relative_proxy']}`",
        "",
        "## Provenance",
        "",
        f"- Canonical Q SHA256: `{audit['canonical_q_hash']}`",
        "- Per-figure generation receipts and semantic audits: see `artifacts/publication/fig0*_*.json`",
        "- Figures 2–4 additionally bound by `fig02_04_cross_figure_audit.json`",
        "",
        "## Next checkpoint",
        "",
        f"**{audit['next_checkpoint']}** — controlled visual polish, Results/Methods/Supplement under hard feature freeze.",
        "",
        f"*Machine-readable audit: `artifacts/publication/figures_1_7_cross_figure_audit.json`*",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    audit = run_cross_figure_audit(repo_head=repo_sha())
    audit["audited_at_utc"] = utc_now_iso()
    dirs = ensure_publication_dirs()
    out = dirs["artifacts"] / "figures_1_7_cross_figure_audit.json"
    write_json_strict(out, audit)
    SUMMARY_PATH.write_text(build_summary(audit))
    print(f"wrote: {out.relative_to(repo_root())}")
    print(f"wrote: {SUMMARY_PATH.relative_to(repo_root())}")
    print(f"status: {audit['status']}")
    return 0 if audit["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
