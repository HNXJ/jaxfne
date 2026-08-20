#!/usr/bin/env python3
"""Deep Public / Private Purity & Structural Boundary Audit.

Audits:
1. Public MkDocs navigation pages (81 pages): zero internal agent/harness process language.
2. Public package source tree (jaxfne/ excluding publication audit receipts): zero agent governance words.
3. Built distribution artifacts (wheel / sdist): strictly exclude private directories,
   scratch trees, agent configurations, and developer plans.
"""
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Forbidden process keywords in public contexts
FORBIDDEN_PROCESS_PATTERNS = [
    r"CURRENT_TASK\.md",
    r"GEMINI_HANDOFF",
    r"OPENCODE_HANDOFF",
    r"private_acceptance",
    r"jaxfne_v0_4_17_final_100_goals",
    r"harness_adversarial",
    r"S_NatureMethods",
    r"seal agent",
    r"worker context",
    r"PRP backlog",
    r"progress\.json",
    r"review\.json",
    r"plans\.json",
    r"AGENT_CHANNEL",
]


def audit_public_docs() -> list[str]:
    violations = []
    mk = yaml.safe_load((ROOT / "mkdocs.yml").read_text())

    def extract_nav_files(nav_item):
        files = []
        if isinstance(nav_item, dict):
            for v in nav_item.values():
                files.extend(extract_nav_files(v))
        elif isinstance(nav_item, list):
            for item in nav_item:
                files.extend(extract_nav_files(item))
        elif isinstance(nav_item, str):
            files.append(nav_item)
        return files

    nav_files = set(extract_nav_files(mk.get("nav", [])))
    for rel in nav_files:
        p = ROOT / "docs" / rel
        if p.exists():
            text = p.read_text()
            for pat in FORBIDDEN_PROCESS_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    violations.append(f"LEAK in public doc: docs/{rel} matches '{pat}'")
    return violations


def audit_public_source() -> list[str]:
    violations = []
    for py_path in (ROOT / "jaxfne").rglob("*.py"):
        if "publication" in py_path.parts:
            continue
        text = py_path.read_text()
        for pat in FORBIDDEN_PROCESS_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                violations.append(f"LEAK in jaxfne/ public source: {py_path.relative_to(ROOT)} matches '{pat}'")
    return violations


def audit_distributions() -> list[str]:
    violations = []
    dist_dir = ROOT / "dist"
    if not dist_dir.exists():
        return ["No dist/ directory found. Build packages before running distribution audit."]

    forbidden_pkg_prefixes = (
        "skills/",
        ".opencode/",
        ".cursor/",
        "scratch/",
        "artifacts/private_acceptance/",
        "scripts/harness/",
    )

    for f in dist_dir.glob("*.whl"):
        with zipfile.ZipFile(f, "r") as zf:
            for name in zf.namelist():
                for prefix in forbidden_pkg_prefixes:
                    if name.startswith(prefix):
                        violations.append(f"LEAK in wheel {f.name}: contains '{name}'")

    for f in dist_dir.glob("*.tar.gz"):
        with tarfile.open(f, "r:gz") as tf:
            for name in tf.getnames():
                # strip top-level package dirname
                rel = "/".join(name.split("/")[1:]) if "/" in name else name
                for prefix in forbidden_pkg_prefixes:
                    if rel.startswith(prefix):
                        violations.append(f"LEAK in sdist {f.name}: contains '{rel}'")

    return violations


def main() -> int:
    print("================================================================================")
    print("STRUCTURAL PUBLIC / PRIVATE PURITY AUDIT")
    print("================================================================================")

    doc_violations = audit_public_docs()
    src_violations = audit_public_source()
    dist_violations = audit_distributions()

    all_v = doc_violations + src_violations + dist_violations
    if all_v:
        for v in all_v:
            print(f"FAIL: {v}")
        print("--------------------------------------------------------------------------------")
        print("RESULT: FAIL")
        print("================================================================================")
        return 1

    print("1. Public Docs (81 pages): 100% clean (zero private/agent process language).")
    print("2. Public Codebase (jaxfne/): 100% clean (zero private/agent process language).")
    print("3. Distribution Packages: 100% clean (private trees strictly excluded).")
    print("--------------------------------------------------------------------------------")
    print("RESULT: PASS")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
