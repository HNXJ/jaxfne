"""Registrable per-``NeuronType`` emitter surface for the NeuronalTensor path.

A ``NeuronalTensor`` declares one emitter per :class:`NeuronalType
<jaxfne.neuronal_tensor.NeuronType>` (``NeuronType.emitter`` plus optional
``NeuronType.emitter_params``) so any ``(h, y) = f(x, h')`` cell model with
geometry driving dynamics can be specified at the tensor level. This module
is the declaration registry: it validates names and parameter keys, mirroring
:mod:`jaxfne.hdp_rule` (``register`` / ``get`` / ``is_registered`` / ``list``).
Routing (which families construct today) lives in
:mod:`jaxfne.neuronal_tensor`; only ``izhikevich``, ``lif`` and ``adex``
route, and a uniform non-built-in family is refused at bridge time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .emitters_adex import _PARAM_FIELDS as _ADEX_PARAM_FIELDS
from .emitters_lif import _PARAM_FIELDS as _LIF_PARAM_FIELDS


@dataclass(frozen=True)
class EmitterDescriptor:
    """A routable-by-name emitter declaration.

    ``params`` names the per-``NeuronType`` parameter keys this emitter
    consumes (checked by :func:`check_emitter_params`). An empty tuple means
    the emitter consumes no per-type parameters: any declared
    ``emitter_params`` for it are refused as unconsumed.
    """

    name: str
    params: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, EmitterDescriptor] = {}


def register_emitter(descriptor: EmitterDescriptor) -> None:
    """Register an emitter descriptor by name."""
    if not isinstance(descriptor, EmitterDescriptor):
        raise TypeError(
            f"register_emitter needs an EmitterDescriptor; got {type(descriptor).__name__}"
        )
    if not descriptor.name:
        raise ValueError("register_emitter needs a non-empty descriptor name")
    if descriptor.name in _REGISTRY:
        raise ValueError(f"emitter {descriptor.name!r} is already registered")
    for key in descriptor.params:
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"emitter {descriptor.name!r} declares inadmissible param key {key!r}; "
                "param keys must be non-empty strings"
            )
    _REGISTRY[descriptor.name] = descriptor


def get_emitter(name: str) -> EmitterDescriptor:
    """Return the descriptor for a registered emitter name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown registered emitter {name!r}; known: {sorted(_REGISTRY)}"
        ) from None


def is_registered_emitter(name: str | None) -> bool:
    """Whether ``name`` is a registered emitter name."""
    return bool(name) and name in _REGISTRY


def list_registered_emitters() -> tuple[str, ...]:
    """Sorted names of every registered emitter."""
    return tuple(sorted(_REGISTRY))


def check_emitter_params(name: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate per-``NeuronType`` params against the registered descriptor.

    The emitter must be registered (unknown names are refused here, naming
    the known set); ``params`` must be a mapping (None reads as empty); every
    key must be consumed by the descriptor (anything else is refused as
    unconsumed). Returns a plain dict copy.
    """
    descriptor = get_emitter(name)
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValueError(
            f"emitter {name!r} params must be a mapping of {sorted(descriptor.params)}; "
            f"got {params!r}"
        )
    unknown = [k for k in params if k not in descriptor.params]
    if unknown:
        raise ValueError(
            f"emitter {name!r} does not consume params {unknown!r}; "
            f"consumed: {sorted(descriptor.params)}"
        )
    return dict(params)


def _register_builtin_emitters() -> None:
    register_emitter(EmitterDescriptor(name="izhikevich"))
    register_emitter(EmitterDescriptor(name="lif", params=tuple(_LIF_PARAM_FIELDS)))
    register_emitter(EmitterDescriptor(name="glif"))
    register_emitter(EmitterDescriptor(name="adex", params=tuple(_ADEX_PARAM_FIELDS)))


_register_builtin_emitters()
