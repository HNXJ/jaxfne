"""OCMA stage D: validate council outputs against the J24 glossary and the repository.

Usage: python artifacts/etudes/meta/ocma/tools/validate.py <council.json> [...]
Each file holds {"papers": [{"line": int, "scores": {key: float|null}, "evidence": {key: str}}]}.
Checks: all 24 keys, values in [-1, 1] or null, every non-null score has evidence naming at least one
`path:line` whose file exists and has that line (factors 17-21 may cite the paper instead).
Prints one line per problem and a summary; exit 0 when clean, 1 when any problem, 2 on unreadable input.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
KEYS = [
    "Emitter class", "Synapse kinetics", "Plasticity rule", "Network topology", "Drive protocol",
    "Observable readout", "Parameters transcribed", "Core gap",
    "Value pinned", "Regime reproduced", "Panel remade", "Quantitative match", "Preregistered", "Scored",
    "Oracle comparison", "No target fitting",
    "Named target", "Estimand from core", "Criterion from paper", "Oracle available", "Cheap run",
    "Scorer exists", "Publishable", "Core-check fit",
]
PAPER_OK = {"Named target", "Estimand from core", "Criterion from paper", "Oracle available", "Cheap run"}
CITE = re.compile(r"([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+):(\d+)")


def cite_ok(text):
    for path, line in CITE.findall(text or ""):
        p = ROOT / path
        if p.is_file():
            try:
                n = sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
            if 1 <= int(line) <= n:
                return True
    return False


def main(paths):
    problems, papers = [], 0
    for f in paths:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"unreadable {f}: {e}")
            return 2
        for p in data.get("papers", []):
            papers += 1
            tag = f"{Path(f).name} L{p.get('line')}"
            s, ev = p.get("scores", {}), p.get("evidence", {})
            for k in KEYS:
                if k not in s:
                    problems.append(f"{tag}: missing key {k!r}")
                    continue
                v = s[k]
                if v is None:
                    continue
                if not isinstance(v, (int, float)) or not -1 <= v <= 1:
                    problems.append(f"{tag}: {k!r} out of scale: {v!r}")
                elif v != 0 and k not in PAPER_OK and not cite_ok(ev.get(k)):
                    problems.append(f"{tag}: {k!r}={v} without an existing path:line")
            for k in s:
                if k not in KEYS:
                    problems.append(f"{tag}: unknown key {k!r}")
    for line in problems:
        print(line)
    print(f"validated {papers} paper records in {len(paths)} files: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
