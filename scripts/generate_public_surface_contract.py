#!/usr/bin/env python3
"""Regenerate the tracked public surface contract artifact.

``artifacts/public_surface_contract_v0413.json`` is a serialization of
:func:`jaxfne.public_surface.public_surface_summary`, which is the single
authority for the tier classification and for ``jaxfne.__all__``.

The artifact was previously hand-maintained, which let its ``counts`` drift away
from the module (``baseline_all`` recorded 265 while the tiers it summarized
already summed to 266). Generating it removes the class of defect;
``tests/test_public_surface_contract_v0413.py`` then holds every count field to
the live module so drift fails a gate instead of sitting unnoticed.

Usage:
    python scripts/generate_public_surface_contract.py            # write
    python scripts/generate_public_surface_contract.py --check    # verify only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxfne.public_surface import public_surface_summary  # noqa: E402

OUT = ROOT / "artifacts" / "public_surface_contract_v0413.json"


def render() -> str:
    return json.dumps(public_surface_summary(), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the tracked artifact differs from the live contract",
    )
    args = parser.parse_args(argv)

    expected = render()
    if args.check:
        if not OUT.exists():
            print(f"MISSING: {OUT}", file=sys.stderr)
            return 1
        actual = OUT.read_text(encoding="utf-8")
        if actual != expected:
            print(
                f"DRIFT: {OUT.relative_to(ROOT)} differs from the live public surface "
                "contract. Run: python scripts/generate_public_surface_contract.py",
                file=sys.stderr,
            )
            return 1
        print("public surface contract: pass")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(expected)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
