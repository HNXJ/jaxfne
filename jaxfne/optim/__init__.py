"""Optimization sub-package namespace for jaxfne.

Exposes specialized Stochastic Delta Rule and stochastic gradient descent optimization steps.
"""
from __future__ import annotations

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
    "optax_adam",
    "optax_sgd",
    "random_search",
    "require_optax",
    "sdr_transform",
    "_agsdr_candidates_from_noise",
    "propose_blackbox_candidates",
    "_run_agsdr_optimization_loop",
    "_quadratic_target_loss",
    "quadratic_target_loss_grad",
    "_resolve_optimizer",
    "_tune_matrix_agsdr_optax",
]
