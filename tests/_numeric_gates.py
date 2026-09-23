"""Cross-implementation numeric gates (Item-10 adjudication, 0.5.0).

Rule (human-authorized correction, immutable original FAIL recorded in
``artifacts/programme/item10_adjudication_receipt.md``): cross-implementation
float comparisons use allclose semantics

    |x - y| <= max(eps_abs, eps_rel * max(|x|, |y|)),

with eps_rel=1e-5, eps_abs=1e-8. The absolute leg governs the near-zero
regime; it is NOT an additional mandatory constraint at O(1), where it would
demand sub-ulp agreement across different reduction implementations.

Do not reinterpret this as exact equality, and do not widen it without a new
recorded adjudication.
"""

from __future__ import annotations

import math

EPS_REL = 1e-5
EPS_ABS = 1e-8


def cross_impl_close(
    a: float, b: float, *, eps_rel: float = EPS_REL, eps_abs: float = EPS_ABS
) -> bool:
    """Magnitude-aware closeness for cross-implementation float comparison."""
    if a == b:
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    bound = max(eps_abs, eps_rel * max(abs(a), abs(b)))
    return abs(a - b) <= bound


def assert_cross_impl_close(
    a: float, b: float, *, label: str = "value", eps_rel: float = EPS_REL, eps_abs: float = EPS_ABS
) -> None:
    """Assert helper with a diagnostic message (no silent reinterpretation)."""
    if not cross_impl_close(a, b, eps_rel=eps_rel, eps_abs=eps_abs):
        bound = max(eps_abs, eps_rel * max(abs(a), abs(b)))
        raise AssertionError(
            f"{label}: |{a!r} - {b!r}| = {abs(a - b)!r} exceeds bound {bound!r} "
            f"(eps_rel={eps_rel}, eps_abs={eps_abs})"
        )
