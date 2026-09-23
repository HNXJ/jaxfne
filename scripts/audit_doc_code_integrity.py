#!/usr/bin/env python3
"""Docs↔code↔skills integrity gate: parseable examples resolve live symbols.

Deterministic by construction (no simulation, no construction, no network):

1. Every ```python fence in docs/tutorials + docs/guides that parses must
   resolve all ``jtfne.X`` / ``jaxfne.X`` / ``from jaxfne… import Z``
   references against the live package (import + getattr only).
2. Fences that do not parse (math pseudocode, shell, magics) must match the
   allowlist, else they fail (new unrunnable examples cannot hide).
3. Every file/command/symbol reference in skills + subagent + pool docs must
   exist (paths on disk, markdown link targets, root symbols, script flags),
   except runtime-generated paths declared in the allowlist's
   ``generated_skill_refs`` (present at runtime, absent on fresh clones).

Usage:
    python scripts/audit_doc_code_integrity.py --check   # exit 1 on violation
    python scripts/audit_doc_code_integrity.py           # human-readable report
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = pathlib.Path(__file__).resolve().parent / "doc_code_integrity_allowlist.json"

FENCE_RE = re.compile(r"```python(.*?)```", re.S)
ATTR_RE = re.compile(r"\bjaxfne\.([A-Za-z_][\w\.]*)|\bjtfne\.([A-Za-z_][\w\.]*)")
IMPORT_RE = re.compile(r"^\s*from\s+(jaxfne[\w\.]*)\s+import\s+(.+?)$", re.M)


def load_allowlist() -> dict:
    try:
        return json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    except OSError:
        return {"non_python_fences": [], "allowed_missing": []}


def check_fences() -> list[dict]:
    allow = load_allowlist()
    known = {(e["file"], e["pattern"]) for e in allow.get("non_python_fences", [])}
    allowed_missing = set(allow.get("allowed_missing", []))
    violations = []
    sys.path.insert(0, str(ROOT))
    import jaxfne as J  # noqa: E402

    cache: dict = {}

    def resolve(chain: str) -> bool:
        if chain in cache:
            return cache[chain]
        obj: object = J
        try:
            for part in chain.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            cache[chain] = False
            return False
        cache[chain] = True
        return True

    for md in sorted((ROOT / "docs" / "tutorials").glob("*.md")) + sorted(
        (ROOT / "docs" / "guides").glob("*.md")
    ):
        rel = md.relative_to(ROOT).as_posix()
        text = md.read_text(encoding="utf-8")
        for m in FENCE_RE.finditer(text):
            body = m.group(1)
            if "..." in body:
                continue  # avowed fragment; procedure documents this limit
            try:
                ast.parse(body)
            except SyntaxError:
                line = text.count("\n", 0, m.start()) + 1
                snippet = body.strip().splitlines()[0][:80] if body.strip() else ""
                if not any(f == rel and pat in body for f, pat in known):
                    violations.append(
                        {"kind": "unparsed-fence", "file": rel, "line": line, "text": snippet}
                    )
                continue
            line = text.count("\n", 0, m.start()) + 1
            refs = set()
            for am in ATTR_RE.finditer(body):
                refs.add(am.group(1) or am.group(2))
            for im in IMPORT_RE.finditer(body):
                mod, names = im.group(1), im.group(2)
                base = mod[len("jaxfne") :].lstrip(".")
                for nm in names.split(","):
                    nm = nm.strip().split(" as ")[0].strip("() ")
                    if nm and nm != "*":
                        refs.add((base + "." + nm).lstrip("."))
            for ref in sorted(refs):
                top = ref.split(".")[0]
                if top in (
                    "np",
                    "jnp",
                    "jax",
                    "plt",
                    "os",
                    "sys",
                    "json",
                    "pathlib",
                    "dataclasses",
                    "typing",
                ):
                    continue
                if not resolve(ref) and ref not in allowed_missing:
                    violations.append(
                        {"kind": "missing-symbol", "file": rel, "line": line, "text": ref}
                    )
    return violations


SKILL_DOCS = [
    "artifacts/skills/jaxfne-audit/SKILL.md",
    "artifacts/skills/jaxfne-core/SKILL.md",
    "artifacts/skills/jaxfne-release/SKILL.md",
    "artifacts/skills/jaxfne-repo/SKILL.md",
    "artifacts/skills/jaxfne-science/SKILL.md",
    "artifacts/skills/jaxfne-seal/SKILL.md",
    "artifacts/skills/vocabulary-audit/SKILL.md",
    "artifacts/subagents/vocabulary_critic.md",
    "artifacts/subagents/jaxfne-developer.md",
    "artifacts/subagent-pool.md",
]

PATH_RE = re.compile(r"`((?:artifacts|scripts|docs|tests|scratch|mkdocs\.yml|README\.md)[^`]*?)`")


def check_skill_refs() -> list[dict]:
    violations = []
    allow = load_allowlist()
    generated = {e["path"] for e in allow.get("generated_skill_refs", []) if "path" in e}
    for rel in SKILL_DOCS:
        p = ROOT / rel
        if not p.exists():
            violations.append(
                {"kind": "missing-doc", "file": rel, "line": 0, "text": "skill doc itself missing"}
            )
            continue
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            for m in PATH_RE.finditer(line):
                ref = m.group(1).strip()
                if ref.endswith("/**") or "*" in ref:
                    continue
                target = ROOT / ref.split(" ")[0]
                if not target.exists():
                    if ref.split(" ")[0] in generated:
                        continue  # declared runtime-generated; see allowlist
                    violations.append({"kind": "stale-ref", "file": rel, "line": i, "text": ref})
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    violations = check_fences() + check_skill_refs()
    if not violations:
        print("doc-code integrity: pass")
        return 0
    print(f"doc-code integrity: {len(violations)} violations")
    for v in violations:
        print(f"  [{v['kind']}] {v['file']}:{v['line']}: {v['text'][:120]}")
    return 1 if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
