"""Atlas figure column order (AT-00-R5).

Every Atlas figure lays its panels out left to right in this order. A
figure spec is ``{"id": str, "panels": [{"column": <name>, "source": <path>}]}``
where ``source`` is the generated result file the panel consumes. This module
only fixes and checks the order; it draws nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

__all__ = ["COLUMNS", "check_figure_spec"]

COLUMNS: tuple[str, ...] = (
    "structure",
    "dynamics",
    "state_plasticity",
    "source",
    "field",
    "observation",
    "computation",
)


def check_figure_spec(spec: dict[str, Any], *, root: Path | None = None) -> list[str]:
    """Return the defects of one figure spec (empty list = conforming).

    A panel column must be one of ``COLUMNS``, panels must appear in
    ``COLUMNS`` order (repeats of a column adjacent), and, when ``root`` is
    given, every panel ``source`` must exist under it.
    """
    out: list[str] = []
    fid = spec.get("id", "<no id>")
    panels: Iterable[dict[str, Any]] = spec.get("panels") or []
    if not panels:
        return [f"{fid}: no panels"]
    last = -1
    for i, p in enumerate(panels):
        col = p.get("column")
        if col not in COLUMNS:
            out.append(f"{fid}[{i}]: unknown column {col!r}")
            continue
        k = COLUMNS.index(col)
        if k < last:
            out.append(f"{fid}[{i}]: column {col!r} out of order")
        last = max(last, k)
        src = p.get("source")
        if root is not None and (not src or not (root / src).exists()):
            out.append(f"{fid}[{i}]: source {src!r} missing")
    return out
