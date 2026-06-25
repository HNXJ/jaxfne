"""Relative accumulative coordination — the one anti-overlap layout rule.

Fixed/absolute offsets between stacked plot elements (traces, panels,
annotations) overlap whenever one element's extent exceeds the assumed
spacing. The fix used everywhere in :mod:`jaxfne.vis`: each element's
position is the *previous* element's position plus the *previous* element's
own measured extent plus a gap — never an absolute/global coordinate. This
guarantees zero overlap regardless of per-element amplitude variance.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def cumulative_stack_offsets(series_list: Sequence[np.ndarray], *, gap_frac: float = 0.15) -> list:
    """Offsets for stacking 1D series with zero overlap, by accumulated extent.

    ``offset[i]`` is defined relative to ``offset[i-1]`` plus series ``i-1``'s
    own measured (max - min) extent plus ``gap_frac`` of that extent as
    breathing room — not a fixed/global step size. A trace with an unusually
    large range simply pushes every later trace further, instead of
    colliding with it.

    Returns one offset per input series (add it to that series before
    plotting). ``gap_frac=0.15`` default leaves visible separation; ``0.0``
    packs traces edge-to-edge with no overlap.
    """
    offsets = []
    cursor = 0.0
    for s in series_list:
        s = np.asarray(s)
        finite = s[np.isfinite(s)]
        lo = float(np.min(finite)) if finite.size else 0.0
        hi = float(np.max(finite)) if finite.size else 1.0
        extent = hi - lo
        if extent <= 0:
            extent = 1.0
        offsets.append(cursor - lo)
        cursor += extent * (1.0 + gap_frac)
    return offsets


def cumulative_panel_extents(sizes: Sequence[float], *, gap_frac: float = 0.1) -> list:
    """Start positions for a row/column of panels sized by ``sizes``, gapless-overlap-free.

    ``sizes[i]`` is panel ``i``'s own width/height; the returned start
    position for panel ``i`` accumulates every prior panel's actual size plus
    a proportional gap, so panels of unequal size never collide.
    """
    starts = []
    cursor = 0.0
    for size in sizes:
        starts.append(cursor)
        cursor += float(size) * (1.0 + gap_frac)
    return starts
