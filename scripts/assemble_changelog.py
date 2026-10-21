"""Assemble the files under ``artifacts/changelog.d/`` into ``docs/reference/changelog.md``.

Each change writes its own file, ``artifacts/changelog.d/<name>.<category>.md``,
holding one or more top-level ``- `` bullets. Two changes made in parallel
therefore touch two different files and merge without conflict, where two edits
under ``## [Unreleased]`` collide on the same lines.

At release, ``python scripts/assemble_changelog.py --version X.Y.Z --date
YYYY-MM-DD`` inserts ``## vX.Y.Z (YYYY-MM-DD)`` directly below
``## [Unreleased]``, one ``### <Heading>`` block per category present in
``CATEGORIES`` order, fragments sorted by file name within a block, and then
deletes the fragments it consumed. ``--dry-run`` prints the section and changes
nothing.

``docs/reference/changelog.md`` mixes CRLF and LF lines, so every byte of it is kept:
the file is read and written as bytes, the new section is an insertion, never
an edit. The script refuses a version that already has a ``## v<version>``
section, and refuses while ``[Unreleased]`` still carries hand-written lines,
because merging those with the fragments would have to choose an order for
them. A fragment containing ``\\r`` is refused for the same reason: it would
add a third line-ending style to the file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIR = REPO_ROOT / "artifacts" / "changelog.d"
CHANGELOG = REPO_ROOT / "docs" / "reference" / "changelog.md"

#: Fragment category -> section heading, in Keep a Changelog order. These are
#: the Keep a Changelog headings ``docs/reference/changelog.md`` uses; older sections
#: carry other headings as history, but fragments use only these.
CATEGORIES: Dict[str, str] = {
    "added": "Added",
    "changed": "Changed",
    "deprecated": "Deprecated",
    "removed": "Removed",
    "fixed": "Fixed",
}

#: Files in the fragment directory that are not fragments.
NOT_FRAGMENTS = frozenset({"README.md"})

UNRELEASED = b"## [Unreleased]"
_VERSION = re.compile(r"^\d+(\.\d+)+((a|b|rc)\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MARKER_LINE = re.compile(rb"^## \[Unreleased\][^\n]*\n", flags=re.MULTILINE)


class FragmentError(ValueError):
    """A fragment, or the changelog it would be assembled into, cannot be used as it is."""


def read_fragments(directory: Path) -> List[Tuple[str, str, Path]]:
    """``(category, text, path)`` for each fragment, sorted by category order then file name.

    A file that does not parse as a fragment raises rather than being skipped: a misnamed
    fragment that is silently ignored is a change that never reaches the release notes.
    """
    if not directory.is_dir():
        return []
    found: List[Tuple[int, str, str, str, Path]] = []
    order = list(CATEGORIES)
    for path in sorted(directory.iterdir()):
        if path.name in NOT_FRAGMENTS or path.name.startswith("."):
            continue
        parts = path.name.split(".")
        if not path.is_file() or len(parts) < 3 or parts[-1] != "md" or not parts[0]:
            raise FragmentError(
                f"{path.name}: a fragment is named <name>.<category>.md, with <category> one of "
                f"{', '.join(order)}"
            )
        category = parts[-2]
        if category not in CATEGORIES:
            raise FragmentError(
                f"{path.name}: unknown category {category!r}; expected one of {', '.join(order)}"
            )
        text = path.read_bytes().decode("utf-8")
        if "\r" in text:
            raise FragmentError(f"{path.name}: use LF line endings, as new changelog entries do")
        body = text.strip("\n")
        if not body:
            raise FragmentError(f"{path.name}: the fragment is empty")
        if not body.startswith("- "):
            raise FragmentError(f"{path.name}: a fragment starts with a top-level '- ' bullet")
        if re.search(r"^#", body, flags=re.MULTILINE):
            raise FragmentError(f"{path.name}: a fragment carries no headings")
        found.append((order.index(category), path.name, category, body + "\n", path))
    found.sort(key=lambda item: (item[0], item[1]))
    return [(category, text, path) for _, _, category, text, path in found]


def render_section(version: str, date: str, fragments: Sequence[Tuple[str, str, Path]]) -> str:
    """The release section, ending with the blank line that separates it from the next one."""
    if not _VERSION.match(version):
        raise FragmentError(f"{version!r} is not a version such as 0.5.5")
    if not _DATE.match(date):
        raise FragmentError(f"{date!r} is not a date in YYYY-MM-DD form")
    if not fragments:
        raise FragmentError("there are no fragments to assemble")
    out = [f"## v{version} ({date})\n\n"]
    current: Optional[str] = None
    for category, text, _ in fragments:
        if category != current:
            if current is not None:
                out.append("\n")
            out.append(f"### {CATEGORIES[category]}\n\n")
            current = category
        out.append(text)
    out.append("\n")
    return "".join(out)


def _version_heading(version: str) -> re.Pattern:
    return re.compile(rb"^## v" + re.escape(version).encode("utf-8") + rb"(?=\s|$)",
                      flags=re.MULTILINE)


def assemble(changelog: bytes, version: str, date: str,
             fragments: Sequence[Tuple[str, str, Path]]) -> bytes:
    """``changelog`` with the release section for ``fragments`` inserted below ``[Unreleased]``.

    Bytes in, bytes out: head, gap and rest are kept verbatim -- the section goes
    after the existing blank gap, so no existing byte is changed and the file's
    mixed CRLF/LF line endings survive the insertion.
    """
    if _version_heading(version).search(changelog):
        raise FragmentError(f"docs/reference/changelog.md already has a section for v{version}")
    marker = _MARKER_LINE.search(changelog)
    if marker is None:
        raise FragmentError("docs/reference/changelog.md has no '## [Unreleased]' line")
    head, tail = changelog[:marker.end()], changelog[marker.end():]
    cut = tail.find(b"\n## ")
    gap, rest = (tail, b"") if cut == -1 else (tail[:cut + 1], tail[cut + 1:])
    if gap.strip():
        raise FragmentError("'## [Unreleased]' carries hand-written lines; move them into fragments first")
    if not gap.endswith(b"\n"):
        gap = gap + b"\n"
    section = render_section(version, date, fragments).encode("utf-8")
    return head + gap + section + rest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", required=True, help="release date, YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="print the section only")
    parser.add_argument("--changelog", type=Path, default=CHANGELOG)
    parser.add_argument("--fragments", type=Path, default=FRAGMENT_DIR)
    args = parser.parse_args(argv)
    try:
        fragments = read_fragments(args.fragments)
        original = args.changelog.read_bytes()
        updated = assemble(original, args.version, args.date, fragments)
    except FragmentError as exc:
        print(f"assemble_changelog: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        sys.stdout.write(render_section(args.version, args.date, fragments))
        return 0
    args.changelog.write_bytes(updated)
    if args.changelog.read_bytes() != updated:
        print("assemble_changelog: docs/reference/changelog.md did not read back as written", file=sys.stderr)
        return 1
    for _, _, path in fragments:
        path.unlink()
    print(f"assembled {len(fragments)} fragments into v{args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
