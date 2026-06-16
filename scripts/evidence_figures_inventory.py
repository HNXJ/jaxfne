#!/usr/bin/env python3
"""Backward-compatible wrapper for the evidence inventory command.

Older agent context files used ``scripts/evidence_figures_inventory.py`` while
the live implementation is ``scripts/evidence_inventory.py``. Keep this wrapper
so freeze/validation commands fail less often for low-cost agents.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evidence_inventory import main


if __name__ == "__main__":
    raise SystemExit(main())
