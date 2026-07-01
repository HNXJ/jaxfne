"""Optimization sub-package namespace for jaxfne.

Exposes specialized Stochastic Delta Rule and stochastic gradient descent optimization steps.

NAMING COLLISION WARNING: ``SDRState``/``GSDRState``/``AGSDRState`` exported
here (from ``.sdr``/``.gsdr``/``.agsdr``) are UNRELATED to the similarly-named
``_TransformSDRState``/``_TransformGSDRState``/``_TransformAGSDRState``
classes defined privately in ``jaxfne.optim.core`` and used by the real
``sdr_transform``/``gsdr_transform``/``agsdr_transform`` Optax-integrated
factories (also exported here). Same bare concept name, two independent
implementations -- see ``jaxfne/optim/core.py``'s own comment above
``_TransformSDRState`` for the full explanation, and each ``step_*_transform``
function's docstring for which implementation is actually on
``Model.tune()``'s call path.
"""
from __future__ import annotations

from .base import BaseSDRState
from .agsdr import AGSDRState, step_agsdr_transform
from .bounds import apply_parameter_constraints, enforce_parameter_bounds
from .gsdr import GSDRState, step_gsdr_transform
from .gsgd import GSGDState, step_gsgd_transform
from .manifests import serialize_optimization_manifest
from .sdr import SDRState, step_sdr_transform
from .core import (
    AGSDR,
    AGSDROptimizerSpec,
    OptimizerSpec,
    agsdr,
    agsdr_transform,
    gsdr,
    gsdr_transform,
    gsgd,
    optax_adam,
    optax_sgd,
    random_search,
    require_optax,
    sdr_transform,
    _agsdr_candidates_from_noise,
    propose_blackbox_candidates,
    _run_agsdr_optimization_loop,
    _quadratic_target_loss,
    quadratic_target_loss_grad,
    _resolve_optimizer,
    _tune_matrix_agsdr_optax,
)

__all__ = [
    "step_agsdr_transform",
    "step_gsdr_transform",
    "step_sdr_transform",
    "step_gsgd_transform",
    "apply_parameter_constraints",
    "enforce_parameter_bounds",
    "serialize_optimization_manifest",
    "BaseSDRState",
    "AGSDRState",
    "GSDRState",
    "SDRState",
    "GSGDState",
    "AGSDR",
    "AGSDROptimizerSpec",
    "OptimizerSpec",
    "agsdr",
    "agsdr_transform",
    "gsdr",
    "gsdr_transform",
    "gsgd",
    "optax_adam",
    "optax_sgd",
    "random_search",
    "require_optax",
    "sdr_transform",
    "propose_blackbox_candidates",
    "quadratic_target_loss_grad",
]
# NOTE: _agsdr_candidates_from_noise, _run_agsdr_optimization_loop,
# _quadratic_target_loss, _resolve_optimizer, _tune_matrix_agsdr_optax are
# intentionally imported above but NOT in __all__ -- they are internal
# helpers (leading underscore = not part of the public API contract), still
# directly importable (e.g. `from jaxfne.optim import _resolve_optimizer`,
# used by core.py and tests/test_v0317_dtype_invariants.py) but excluded
# from `from jaxfne.optim import *` and from being advertised as stable
# public surface.
