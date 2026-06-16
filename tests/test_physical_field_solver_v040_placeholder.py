"""The v0.4 physical field solver placeholder must fail loudly, never silently.

Guards the truth-gate boundary: the stable field path is the laminar proxy
(`linear_solver`). The pre-0.4 physical-solver contract exists only to fix
the API shape and must raise until a validated solver lands.
"""

import jax.numpy as jnp
import pytest

from jaxfne.experimental_hpc import PhysicalFieldSolverSpec, solve_physical_field


def test_spec_construction_is_inert_and_makes_no_physical_claim():
    spec = PhysicalFieldSolverSpec()
    # Constructing the spec must not assert any physical claim.
    assert spec.physical_amplitude_calibrated is False
    assert spec.field_solver_status == "physical_solver_not_implemented_v040"


def test_validate_fails_loudly():
    with pytest.raises(NotImplementedError, match=">TBI-not-ready: v0.4 physical field solver"):
        PhysicalFieldSolverSpec().validate()


def test_solve_fails_loudly():
    with pytest.raises(NotImplementedError, match=">TBI-not-ready: v0.4 physical field solver"):
        solve_physical_field(PhysicalFieldSolverSpec(), jnp.zeros((4, 3)))
