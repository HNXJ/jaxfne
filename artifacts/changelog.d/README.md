# Changelog fragments

One file per change a reader would notice, named `<name>.<category>.md`, where
`<category>` is one of `added`, `changed`, `deprecated`, `removed` or `fixed`
(the Keep a Changelog headings `docs/reference/changelog.md` uses). The file holds one or
more top-level `- ` bullets written as they will read in `docs/reference/changelog.md`,
with LF line endings and no headings.

At release, `python scripts/assemble_changelog.py --version X.Y.Z --date
YYYY-MM-DD` inserts `## vX.Y.Z (YYYY-MM-DD)` below `## [Unreleased]`, one `###`
block per category in the order above and fragments sorted by file name within
a block, and deletes the fragments it used. This file is not a fragment and
stays.
