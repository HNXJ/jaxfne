"""Write the release notes for a version from package metadata and the changelog.

The body is two tables, in order: release metadata (package, version, Python
support, extras, install command) read from ``pyproject.toml``, and the
version's section of ``docs/reference/changelog.md``. ``--check`` verifies the tag names
``pyproject.toml``'s version, the changelog carries a section for it, and the section
is dated (``## vX.Y.Z (YYYY-MM-DD)``): a ``(candidate)`` section fails the check,
while the body without ``--check`` still renders it. The check reads files only; it
never calls the network or ``gh``.

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


def _dated_heading(version: str) -> re.Pattern:
    return re.compile(r"^## v" + re.escape(version) + r" \(\d{4}-\d{2}-\d{2}\)", re.MULTILINE)


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


def build_body(root: pathlib.Path, tag: str, require_dated: bool = False
               ) -> tuple[Optional[str], List[str]]:
    """``(body, [])`` when the tag matches and the section exists, else ``(None, violations)``.

    With ``require_dated``, a section whose heading carries no ``(YYYY-MM-DD)`` date --
    a ``(candidate)`` section -- is a violation: the section must be dated before release.
    Without it the body renders whatever section the version has.
    """
    metadata = release_metadata(root)
    if tag != f"v{metadata.version}":
        return None, [f"tag {tag!r} does not match pyproject version "
                      f"{metadata.version!r} (expected 'v{metadata.version}')"]
    changelog = (root / "docs" / "reference" / "changelog.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    section = changelog_section(changelog, metadata.version)
    if section is None:
        return None, [f"docs/reference/changelog.md has no non-empty section headed "
                      f"'## v{metadata.version} ...'"]
    if require_dated and _dated_heading(metadata.version).search(changelog) is None:
        heading = section.split("\n", 1)[0]
        return None, [f"docs/reference/changelog.md carries {heading!r} for v{metadata.version}: "
                      f"the section must be dated before release"]
    return render_body(metadata, section), []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v0.5.5")
    parser.add_argument("--check", action="store_true",
                        help="verify the tag, the section and its date; "
                             "a candidate section fails the check")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1],
                        help="the repository whose metadata and changelog are read")
    args = parser.parse_args(argv)

    body, violations = build_body(args.root, args.tag, require_dated=args.check)
    if body is None:
        for violation in violations:
            print(f"release body: {violation}", file=sys.stderr)
        return 1
    if args.check:
        print(f"PASS: {args.tag} matches pyproject and docs/reference/changelog.md carries its section")
        return 0
    # UTF-8 bytes whatever the console or redirect encoding (cp1252 cannot encode the changelog's arrows).
    sys.stdout.buffer.write(body.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
