"""Pre-0.4.0 placeholder contract for a *physical* field solver.

This module declares — but does NOT implement — the interface a future v0.4.0
physical (PDE / volume-conductor) field solver would expose. It exists only to
fix the API boundary and to **fail loudly** so the proxy → physical transition
can never happen silently.

Truth gate (non-negotiable): the stable jaxfne field path is the laminar Gaussian
proxy (``field_solver_status = "laminar_proxy_no_pde"``,
``physical_amplitude_claim_allowed = False``). Nothing here is validated. Until a
real, validated solver lands, every entry point raises
``NotImplementedError(">TBI-not-ready: v0.4 physical field solver")``. Importing
or constructing the spec must not, on its own, imply that physical-amplitude or
PDE-solve claims are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import jax

_TBI_V040 = ">TBI-not-ready: v0.4 physical field solver"


def _tbi_v040() -> None:
    """Raise the v0.4 not-implemented sentinel so callers cannot silently proceed."""
    raise NotImplementedError(_TBI_V040)


@dataclass(frozen=True)
class PhysicalFieldSolverSpec:
    """Declarative spec for a future physical (PDE) field solve — NOT implemented.

    Records the inputs a calibrated volume-conductor solve would require:
    ``conductivity`` (S/m, tissue conductivity model), ``boundary`` (boundary
    condition kind), ``gauge`` (potential gauge), and ``geometry`` (electrode /
    domain geometry metadata). Constructing the spec is inert and asserts no
    physical claim; :meth:`validate` and the solver entry points fail loudly.

    ``field_solver_status`` is fixed to ``"physical_solver_not_implemented_v040"``
    to make explicit that no PDE solve exists yet — it must never be read as a
    completed physical solve.
    """

    conductivity: Optional[Mapping[str, Any]] = None
    boundary: str = "unset"
    gauge: str = "unset"
    geometry: Mapping[str, Any] = field(default_factory=dict)
    field_solver_status: str = "physical_solver_not_implemented_v040"
    physical_amplitude_claim_allowed: bool = False

    def validate(self) -> dict[str, Any]:
        """Fail loudly — physical-solver validation is not implemented in v0.3.x."""
        _tbi_v040()


def solve_physical_field(spec: PhysicalFieldSolverSpec, sources: jax.Array) -> Any:
    """Placeholder for a future physical field solve — fails loudly.

    Until a validated v0.4.0 solver exists this raises
    ``NotImplementedError(">TBI-not-ready: v0.4 physical field solver")``. Use the
    stable laminar proxy (``jaxfne.project_sources_to_laminar_field``) instead; its
    outputs are relative-unit proxies, not physical amplitudes.
    """
    _tbi_v040()
