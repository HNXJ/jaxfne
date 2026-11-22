"""Shared scaffolding of the scripts/score_*_checks.py scorers.

Each scorer recomputes its checks and writes artifacts/etudes/core_checks/<area>/score.json (one
{error, tolerance, pass, reference} row per check) beside a README. --check recomputes and compares
with the stored file. These functions take the scorer's own state (paths, RTOL, NOISE_FRACTION) as
arguments; each scorer binds them in thin wrappers that read its module globals at call time.
"""

import contextlib
import json
import math
import sys

import jax


@contextlib.contextmanager
def x64():
    """x64 on for one block, restored after (jax.enable_x64 is missing from older jax in the declared range)."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


def compare(old, new, rtol, noise_fraction, skippable=False):
    """Mismatch strings between a stored document and a fresh one; [] means agreement.

    skippable: the scorer's rows may hold "skipped" (an optional dependency absent in this environment).
    """
    if set(old.get("checks", ())) != set(new.get("checks", ())):
        return [f"checks: {sorted(old.get('checks', ()))} != {sorted(new.get('checks', ()))}"]
    bad = []
    for name in old["checks"]:
        o, n = old["checks"][name], new["checks"][name]
        if skippable and "skipped" in n and "skipped" not in o:
            # This environment lacks the check's optional dependency; the stored row stands.
            continue
        if skippable and ("skipped" in o) != ("skipped" in n):
            bad.append(f"{name}: skipped {('skipped' in o)} != {('skipped' in n)}")
            continue
        if skippable and "skipped" in o:
            if o["skipped"] != n["skipped"]:
                bad.append(f"{name}.skipped: {o['skipped']!r} != {n['skipped']!r}")
            continue
        if o["pass"] != n["pass"]:
            bad.append(f"{name}.pass: {o['pass']!r} != {n['pass']!r}")
        for key in ("error", "tolerance"):
            if math.isclose(o[key], n[key], rel_tol=rtol, abs_tol=1e-15):
                continue
            # A rounding-level residual differs across platforms (BLAS, float32); below
            # noise_fraction of the tolerance in both copies it cannot change the verdict.
            floor = noise_fraction * min(o["tolerance"], n["tolerance"])
            if key == "error" and o[key] <= floor and n[key] <= floor:
                continue
            bad.append(f"{name}.{key}: {o[key]!r} != {n[key]!r} past rtol {rtol:g}")
    if old.get("pass") != new.get("pass"):
        bad.append(f"pass: {old.get('pass')!r} != {new.get('pass')!r}")
    return bad


def check(doc, score_path, rtol, noise_fraction, skippable=False):
    """Exit status of --check: 0 when the stored score agrees with the fresh document, else 1."""
    try:
        stored = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {score_path}: {exc}", file=sys.stderr)
        return 1
    bad = compare(stored, doc, rtol, noise_fraction, skippable)
    if bad:
        for line in bad:
            print(f"mismatch: {line}", file=sys.stderr)
        return 1
    print(f"fresh: {score_path} matches a recomputation within rtol {rtol:g} "
          f"(errors under {noise_fraction:g} x tolerance in both copies count as rounding residuals)")
    if skippable:
        unchecked = sorted(name for name, row in doc["checks"].items()
                           if "skipped" in row and "skipped" not in stored["checks"][name])
        if unchecked:
            # The stored rows stand; this environment lacks their optional dependency and did not recompute them.
            print(f"not rechecked here: {', '.join(unchecked)}")
    return 0


def write(doc, out_dir, score_path, readme_path, readme_text):
    """Write score.json, and the README beside it when none exists (an existing README is left as is)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(score_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    if not readme_path.is_file():
        readme_path.write_text(readme_text, encoding="utf-8", newline="\n")
