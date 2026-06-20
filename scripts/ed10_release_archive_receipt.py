"""ED10 — release-archive receipt (evidence-chain closure).

ED9 produces the homeostasis null/ablation/seed evidence bundle. ED10 *closes the
chain*: it binds the release identity (git branch + commit SHA + dirty state,
package version, conservative truth gates) to the content hashes of the upstream
evidence artifacts, then self-hashes. One receipt makes the whole release state
content-addressable and verifiable after the fact.

This is a COMPUTATIONAL-METHOD / provenance artifact — NOT a biological-mechanism
validation. Truth gates stay conservative (claim_level=computational_scaffold,
field_solver_status=linear_solver, physical_amplitude_calibrated=False,
biological_learning_claim=False, mechanism_claim_status=not_claimed).

It hashes whatever upstream evidence artifacts EXIST (it never fabricates them):
the ED9 evidence bundle + receipt, and the evidence inventory if present. Missing
artifacts are reported as missing, not synthesized.

Usage:
    python scripts/ed10_release_archive_receipt.py
    python scripts/ed10_release_archive_receipt.py --out-dir outputs/ed10_release_archive
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

import jaxfne as jtfne

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Upstream evidence artifacts to bind into the archive receipt (ED9 leads the chain).
EVIDENCE_ARTIFACTS = {
    "ed9_evidence": "outputs/ed9_homeostasis_evidence/ed9_evidence.json",
    "ed9_receipt": "outputs/ed9_homeostasis_evidence/ed9_receipt.json",
    "evidence_inventory": "outputs/evidence/inventory.json",
}

TRUTH_GATES = {
    "claim_level": "computational_scaffold",
    "field_solver_status": "linear_solver",
    "field_claim_level": "proxy_readout",
    "physical_amplitude_calibrated": False,
    "biological_learning_claim": False,
    "mechanism_claim_status": "not_claimed",
}


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _provenance() -> dict:
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    sha = _git(["rev-parse", "HEAD"]) or "unknown"
    status = _git(["status", "--short"])
    dirty = bool(status) if status is not None else False
    prov = jtfne.provenance_receipt(branch=branch, sha=sha, dirty=dirty)
    return prov


def run(out_dir="outputs/ed10_release_archive"):
    # Hash only artifacts that actually exist — never fabricate the upstream evidence.
    present = {name: str(REPO_ROOT / rel) for name, rel in EVIDENCE_ARTIFACTS.items()
              if (REPO_ROOT / rel).is_file()}
    missing = [rel for rel in EVIDENCE_ARTIFACTS.values()
               if not (REPO_ROOT / rel).is_file()]
    hashes = jtfne.asset_hashes(present) if present else {}

    archive = {
        "schema_version": "jaxfne.ed10_release_archive_receipt.v0.1.0",
        "artifact_kind": "release_archive_receipt_evidence_chain_closure",
        "package_version": getattr(jtfne, "__version__", "unknown"),
        "provenance": _provenance(),
        "truth_gates": TRUTH_GATES,
        "evidence_chain": {
            "archived_artifacts": {name: EVIDENCE_ARTIFACTS[name] for name in present},
            "artifact_sha256": hashes,
            "missing_artifacts": missing,
            "complete": not missing,
        },
        "truth_status": "provenance_and_method_evidence_not_mechanism_validation",
        "note": ("Binds release identity to content hashes of upstream evidence "
                 "(ED9 null/ablation/seed bundle). Provenance/method closure only; "
                 "mechanism support requires empirical comparison beyond this chain."),
    }

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "ed10_release_archive.json"
    jtfne.save_json(jtfne.json_safe(archive), str(archive_path))
    digest = jtfne.sha256_file(str(archive_path))
    receipt = {
        "artifact": "ed10_release_archive.json",
        "sha256": digest,
        "schema_version": archive["schema_version"],
        "package_version": archive["package_version"],
        "commit_sha": archive["provenance"].get("sha"),
        "dirty": archive["provenance"].get("dirty"),
        "evidence_chain_complete": archive["evidence_chain"]["complete"],
        "truth_status": archive["truth_status"],
    }
    jtfne.save_json(receipt, str(out / "ed10_receipt.json"))
    return archive, digest, out


def main():
    ap = argparse.ArgumentParser(description="ED10 release-archive receipt (evidence-chain closure)")
    ap.add_argument("--out-dir", default="outputs/ed10_release_archive")
    args = ap.parse_args()
    archive, digest, out = run(out_dir=args.out_dir)
    print(f"ED10 release-archive receipt written to {out}/ (sha256={digest[:16]}...)")
    prov = archive["provenance"]
    print(f"  package_version : {archive['package_version']}")
    print(f"  commit          : {prov.get('sha', 'unknown')[:12]}  dirty={prov.get('dirty')}  branch={prov.get('branch')}")
    chain = archive["evidence_chain"]
    print(f"  archived        : {len(chain['archived_artifacts'])} artifact(s)  complete={chain['complete']}")
    for name, h in chain["artifact_sha256"].items():
        print(f"    {name:20} {h[:16]}...")
    if chain["missing_artifacts"]:
        print(f"  missing         : {', '.join(chain['missing_artifacts'])}")


if __name__ == "__main__":
    main()
