"""Write the release notes for a version from package metadata and the changelog.

The body is two tables, in order: release metadata (package, version, Python
support, extras, install command) read from ``pyproject.toml``, and the
version's section of ``docs/changelog.md``. ``--check`` verifies the tag names
``pyproject.toml``'s version and the changelog carries a section for it, and
exits nonzero otherwise. The check reads files only; it never calls the
network or ``gh``.

    python scripts/release_body.py --tag v0.5.5
    python scripts/release_body.py --tag v0.5.5 --check
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib
from typing import List, NamedTuple, Optional


class ReleaseMetadata(NamedTuple):
    name: str
    version: str
    python_floor: str
    python_ceiling: str
    python_supported: tuple
    extras: tuple


def release_metadata(root: pathlib.Path) -> ReleaseMetadata:
    """Name, version, Python window and extras, derived from ``pyproject.toml``."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    floor = re.search(r">=\s*(\d+\.\d+)", project["requires-python"])
    if floor is None:
        raise RuntimeError(f"requires-python {project['requires-python']!r} names no floor")
    supported = sorted(
        {m.group(1) for c in project.get("classifiers", [])
         if (m := re.search(r"Programming Language :: Python :: (\d+\.\d+)$", c))},
        key=lambda v: tuple(int(p) for p in v.split(".")),
    )
    if not supported:
        raise RuntimeError("pyproject.toml has no versioned Python classifier")
    extras = tuple(sorted(project.get("optional-dependencies", {})))
    return ReleaseMetadata(
        name=project["name"],
        version=project["version"],
        python_floor=floor.group(1),
        python_ceiling=supported[-1],
        python_supported=tuple(supported),
        extras=extras,
    )


def changelog_section(changelog: str, version: str) -> Optional[str]:
    """The section headed ``## v<version> ...`` up to the next level-2 heading, or None.

    The returned text starts with the heading line itself, so the body shows which
    section it quotes -- including the trailer after the version (a date or
    ``(candidate)``), which is not constrained. None when the heading is absent or
    the section holds no text: an empty section would publish notes that say nothing
    about the release.
    """
    heading = re.compile(r"^## v" + re.escape(version) + r"(?=\s|$)[^\n]*", re.MULTILINE)
    match = heading.search(changelog)
    if match is None:
        return None
    start = match.start()
    following = re.compile(r"^## ", re.MULTILINE).search(changelog, match.end())
    section = changelog[start:following.start() if following else len(changelog)]
    body = section[match.end() - start:].strip("\n")
    return section.strip("\n") if body.strip() else None


def render_body(metadata: ReleaseMetadata, section: str) -> str:
    supported = ", ".join(metadata.python_supported)
    extras = ", ".join(metadata.extras) if metadata.extras else "none"
    return (
        "### Release\n"
        "\n"
        "| field | value |\n"
        "|---|---|\n"
        f"| package | {metadata.name} |\n"
        f"| version | {metadata.version} |\n"
        f"| python | >={metadata.python_floor} ({supported}) |\n"
        f"| extras | {extras} |\n"
        f"| install | `pip install {metadata.name}=={metadata.version}` |\n"
        "\n"
        "### Changelog\n"
        "\n"
        f"{section}\n"
    )


def build_body(root: pathlib.Path, tag: str) -> tuple[Optional[str], List[str]]:
    """``(body, [])`` when the tag matches and the section exists, else ``(None, violations)``."""
    metadata = release_metadata(root)
    if tag != f"v{metadata.version}":
        return None, [f"tag {tag!r} does not match pyproject version "
                      f"{metadata.version!r} (expected 'v{metadata.version}')"]
    raw = (root / "docs" / "changelog.md").read_bytes().decode("utf-8")
    section = changelog_section(raw.replace("\r\n", "\n"), metadata.version)
    if section is None:
        return None, [f"docs/changelog.md has no non-empty section headed "
                      f"'## v{metadata.version} ...'"]
    return render_body(metadata, section), []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v0.5.5")
    parser.add_argument("--check", action="store_true",
                        help="verify only: exit 0 when the body would pass, 1 otherwise")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1],
                        help="the repository whose metadata and changelog are read")
    args = parser.parse_args(argv)

    body, violations = build_body(args.root, args.tag)
    if body is None:
        for violation in violations:
            print(f"release body: {violation}", file=sys.stderr)
        return 1
    if args.check:
        print(f"PASS: {args.tag} matches pyproject and docs/changelog.md carries its section")
        return 0
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
