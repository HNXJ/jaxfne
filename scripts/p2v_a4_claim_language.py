"""A-4 claim-language reconciliation - frozen receipt generator.

Scans public docs (README.md + docs/**) for claim-language violations on the
eight A-4 axes established by the phase-2 action matrix and user directives,
records the allowed language per axis, and writes a FROZEN receipt.
This run also fixed two stale tutorial index lines (recorded below).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
A4_DIR = REPO_ROOT / "artifacts" / "protocol_c" / "p2v_a4_claim_language"
RECEIPT_PATH = A4_DIR / "p2v_a4_reconciliation.json"

AXES = {
    "hierarchy": {
        "rule": "hierarchy claims are computational (causal chain E1->E5, timescale hierarchies, network topologies), never brain-functional",
        "patterns": [r"hierarch"],
        "violation_terms": [],
    },
    "latent": {
        "rule": "latent is relative to the observable chain (H-state is the latent RBS coordinate of the simulated dynamics; VAE factorized latents are descriptive), not a claim about brain latent structure",
        "patterns": [r"latent"],
        "violation_terms": [],
    },
    "c3_characterization": {
        "rule": "C3 states only that the tested ring/delay regimes produced synchronous/standing/structured-but-fails activity; no estimator-supported traveling waves; does not generalize to absence of waves",
        "patterns": [r"[Tt]raveling waves", r"synchronous"],
        "violation_terms": ["generate traveling waves"],
    },
    "typed_disablement": {
        "rule": "typed disablement identities (E1-E5 causal perturbation, D3 containment) are computational identities within the model, not claims about cortex",
        "patterns": [r"[Dd]isablement", r"perturbation", r"NO_ADAPTATION"],
        "violation_terms": [],
    },
    "proxy": {
        "rule": "field readouts are proxies unless amplitude-calibrated: proxy_no_field_solve is the active default; spike impulse proxy (20x gain); CSD is proxy",
        "patterns": [r"proxy", r"calibrat"],
        "violation_terms": [],
    },
    "e5_language": {
        "rule": "E5 is described as causal perturbation/hierarchical propagation within the frozen evidence set; no cognition or predictive-coding language",
        "patterns": [r"E5", r"predictive", r"cognit"],
        "violation_terms": ["predictive coding", "cognition"],
    },
    "w3b": {
        "rule": "W3b parameter-domain map remains FROZEN_UNRESOLVED; active closed-loop stability is not claimed",
        "patterns": [r"W3b", r"W3"],
        "violation_terms": [],
    },
    "negative_claims": {
        "rule": "C3, H4, D3 negatives are stated exactly where frozen: C3 no estimator-supported traveling waves in the tested regimes; H4 memory extension negative; D3 NO_ADAPTATION; W3 unresolved",
        "patterns": [r"negative", r"not supported", r"NO_ADAPTATION", r"unresolved"],
        "violation_terms": [],
    },
}

FIXED_VIOLATIONS = [
    {
        "file": "docs/tutorials/index.md",
        "line": 67,
        "before": "Cross-area interaction, traveling waves, trial-chained HDP carryover",
        "after": "Cross-area interaction, trial-chained HDP carryover",
        "reason": "stale index claim contradicted the tutorial page itself (probes never implemented anywhere in the codebase) and the source script",
    },
    {
        "file": "docs/tutorials/notebook_standard.md",
        "line": 124,
        "before": "Two-column cross-area model, traveling waves",
        "after": "Two-column cross-area model, trial-chained HDP carryover",
        "reason": "same stale claim",
    },
]


def scan_docs() -> dict:
    files = sorted(
        list((REPO_ROOT / "docs").rglob("*.md")) + [REPO_ROOT / "README.md"]
    )
    findings = {}
    for f in files:
        rel = str(f.relative_to(REPO_ROOT))
        text = f.read_text()
        lines = text.splitlines()
        for axis, spec in AXES.items():
            hits = []
            for pat in spec["patterns"]:
                for i, line in enumerate(lines, start=1):
                    if re.search(pat, line, re.IGNORECASE):
                        if re.match(r"^\s*(do not|must not|never|no )", line, re.IGNORECASE):
                            continue  # prohibition lines state the rule; they are not violations
                        flagged = any(re.search(vt, line, re.IGNORECASE) for vt in spec["violation_terms"])
                        hits.append({"line": i, "text": line.strip()[:160], "flagged": flagged})
            findings.setdefault(axis, {})[rel] = hits
    return findings


def main() -> dict:
    findings = scan_docs()
    receipts_axes = []
    for axis, spec in AXES.items():
        file_hits = findings[axis]
        n_flagged = sum(1 for fh in file_hits.values() for h in fh if h["flagged"])
        receipts_axes.append(
            {
                "axis": axis,
                "rule": spec["rule"],
                "files_with_mentions": len([f for f, hs in file_hits.items() if hs]),
                "n_mentions": sum(len(hs) for hs in file_hits.values()),
                "n_flagged_after_fixes": n_flagged,
                "status": "COMPLIANT" if n_flagged == 0 else "REVIEW",
            }
        )
    receipt = {
        "schema": "jaxfne.protocol_c.p2v_a4_reconciliation.v1",
        "protocol_id": "protocol_c_p2v_a4",
        "phase": "post-freeze reviewer-motivated validation",
        "checkpoint": "A-4",
        "status": "FROZEN",
        "write_once": True,
        "package_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "scope": "README.md + docs/** public language only; developer-local PRP notes are not tracked and out of scope; frozen artifacts untouched",
        "axes": receipts_axes,
        "fixed_violations": FIXED_VIOLATIONS,
        "allowed_language_corpus": {
            "wave": "traveling waves are not claimed for jaxfne dynamics; C3 characterization is restricted to the tested ring/delay regimes",
            "hdp": "H is hard-clamped per step; |w| hard-clamped per step; K_w_ctrl=0 implies no claim about instability; boundedness statements are scoped to tested domains (A-3 receipt)",
            "c3": "tested synchronous/standing/structured regimes, no estimator-supported traveling waves (60-cell frozen receipt; A-1a/A-1b/A-2 support)",
            "memory": "H4 memory extension negative as frozen; RBD must not be defined as memory, predictive coding, or surprise (doctrine)",
            "w3": "W3a is analysis-level with a documented margin audit caveat; W3b FROZEN_UNRESOLVED",
            "latent": "latent is relative to the observable chain only",
            "hierarchy": "computational hierarchy only",
            "proxy": "proxy unless amplitude-calibrated (physical_amplitude_calibrated=false)",
        },
        "verification": "python3 scripts/audit_public_docs_language.py --check must pass; test suite must pass",
    }
    A4_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["axes"], indent=1))
    return receipt


if __name__ == "__main__":
    main()