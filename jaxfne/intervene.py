"""0.5.3 item 6: causal intervention grammar (one declarative object).

An :class:`Intervention` records three things together:

1. the identical realized network (builder + build/run spec + digests),
2. an intervention on ONE mechanism (v1: ``plasticity`` via the item-5
   enable/disable/clamp controls; anything else is refused),
3. the declared observation difference (signal + subset + expectation).

First-class in manifests: :meth:`Intervention.to_manifest` is JSON-safe,
:func:`intervention_from_manifest` roundtrips it exactly (H4: in-memory
behavior and serialization roundtrip are tested separately), and
:func:`run_intervention` attaches ``manifest["intervention"]`` on both
arm manifests (additive key only; the builder surface is unchanged).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA = "jaxfne.intervention.v1"

_MECHANISMS = ("plasticity",)
_MODES = ("disable", "clamp")
_SIGNALS = ("w_final", "H_final", "spike_count")
_SUBSETS = ("all", "clamped", "unclamped")
_EXPECTATIONS = ("equal", "different", "within", "exceeds")
_BUILDERS = ("suite2_net1", "hdp_column")


def _strict_keys(d: Mapping[str, Any], allowed: frozenset[str], what: str) -> dict:
    if not isinstance(d, Mapping):
        raise ValueError(f"{what} must be a mapping, got {type(d).__name__}")
    unknown = [k for k in d if k not in allowed]
    if unknown:
        raise ValueError(f"{what} has unknown keys {sorted(unknown, key=repr)}")
    return dict(d)


def _finite_float(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a finite float, got {x!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{what} must be a finite float, got {x!r}")
    return v


def _mask_tuple(mask: Any) -> tuple[float, ...] | None:
    if mask is None:
        return None
    import numpy as np

    arr = np.asarray(mask, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"intervention mask must be 1-D, got shape {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("intervention mask must be finite (got NaN/inf)")
    return tuple(float(v) for v in arr.tolist())


@dataclass(frozen=True)
class Intervention:
    """One declarative causal object (grammar v1)."""

    name: str
    network: Mapping[str, Any]
    control: Mapping[str, Any]
    observation: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("intervention name must be a non-empty string")
        net = _strict_keys(
            self.network,
            frozenset({"builder", "build", "run", "hdp_params"}),
            "network",
        )
        if net["builder"] not in _BUILDERS:
            raise ValueError(
                f"unknown network builder {net['builder']!r}; expected one of {list(_BUILDERS)}"
            )
        for section in ("build", "run", "hdp_params"):
            if not isinstance(net[section], Mapping):
                raise ValueError(f"network[{section!r}] must be a mapping")
        run = _strict_keys(net["run"], frozenset({"duration_ms", "dt_ms", "seed"}), "network.run")
        for key in ("duration_ms", "dt_ms"):
            _finite_float(run[key], f"network.run[{key}]")
        if not isinstance(run["seed"], int) or isinstance(run["seed"], bool):
            raise ValueError("network.run[seed] must be an int")
        ctl = _strict_keys(
            self.control, frozenset({"mechanism", "mode", "mask", "value"}), "control"
        )
        if ctl["mechanism"] not in _MECHANISMS:
            raise ValueError(
                f"grammar v1 supports exactly one mechanism {list(_MECHANISMS)}; "
                f"got {ctl['mechanism']!r}"
            )
        if ctl["mode"] not in _MODES:
            raise ValueError(f"control mode must be one of {list(_MODES)}; got {ctl['mode']!r}")
        mask = _mask_tuple(ctl["mask"])
        if ctl["mode"] == "clamp":
            if mask is None:
                raise ValueError("clamp requires a mask (which projection is held)")
            _finite_float(ctl["value"], "control.value")
        elif ctl["value"] is not None:
            raise ValueError("disable takes no value (got one; use clamp)")
        obs = _strict_keys(
            self.observation,
            frozenset({"signal", "subset", "expect", "tolerance"}),
            "observation",
        )
        if obs["signal"] not in _SIGNALS:
            raise ValueError(
                f"observation signal must be one of {list(_SIGNALS)}; got {obs['signal']!r}"
            )
        if obs["subset"] not in _SUBSETS:
            raise ValueError(
                f"observation subset must be one of {list(_SUBSETS)}; got {obs['subset']!r}"
            )
        if obs["expect"] not in _EXPECTATIONS:
            raise ValueError(
                f"observation expect must be one of {list(_EXPECTATIONS)}; got {obs['expect']!r}"
            )
        if obs["expect"] in ("within", "exceeds"):
            tol = _finite_float(obs["tolerance"], "observation.tolerance")
            if tol < 0:
                raise ValueError("observation.tolerance must be >= 0")
        if obs["subset"] != "all" and (obs["signal"] != "w_final" or mask is None):
            raise ValueError(
                "subset 'clamped'/'unclamped' needs signal 'w_final' and a control mask"
            )
        object.__setattr__(
            self,
            "network",
            {
                "builder": net["builder"],
                "build": dict(net["build"]),
                "run": dict(run),
                "hdp_params": dict(net["hdp_params"]),
            },
        )
        object.__setattr__(
            self,
            "control",
            {
                "mechanism": ctl["mechanism"],
                "mode": ctl["mode"],
                "mask": mask,
                "value": None if ctl["value"] is None else float(ctl["value"]),
            },
        )
        object.__setattr__(
            self,
            "observation",
            {
                "signal": obs["signal"],
                "subset": obs["subset"],
                "expect": obs["expect"],
                "tolerance": None if obs["tolerance"] is None else float(obs["tolerance"]),
            },
        )

    def to_manifest(self) -> dict[str, Any]:
        """JSON-safe manifest of this intervention (arrays -> lists)."""
        ctl = dict(self.control)
        ctl["mask"] = None if ctl["mask"] is None else list(ctl["mask"])
        return {
            "schema": SCHEMA,
            "name": self.name,
            "network": {
                "builder": self.network["builder"],
                "build": dict(self.network["build"]),
                "run": dict(self.network["run"]),
                "hdp_params": dict(self.network["hdp_params"]),
                "spec_digest": network_spec_digest(self),
            },
            "control": ctl,
            "observation": dict(self.observation),
        }


def intervention_from_manifest(d: Mapping[str, Any]) -> Intervention:
    """Rebuild an :class:`Intervention` from its manifest (strict keys)."""
    m = _strict_keys(
        d, frozenset({"schema", "name", "network", "control", "observation"}), "manifest"
    )
    if m["schema"] != SCHEMA:
        raise ValueError(f"unsupported intervention schema {m['schema']!r}")
    net = _strict_keys(
        m["network"],
        frozenset({"builder", "build", "run", "hdp_params", "spec_digest"}),
        "manifest.network",
    )
    return Intervention(
        name=m["name"],
        network={
            "builder": net["builder"],
            "build": dict(net["build"]),
            "run": dict(net["run"]),
            "hdp_params": dict(net["hdp_params"]),
        },
        control=dict(m["control"]),
        observation=dict(m["observation"]),
    )


def network_spec_digest(iv: Intervention) -> str:
    """Canonical digest of the realized-network spec (builder+build+run+params)."""
    canonical = json.dumps(
        {
            "builder": iv.network["builder"],
            "build": iv.network["build"],
            "run": iv.network["run"],
            "hdp_params": _jsonable(iv.network["hdp_params"]),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _build_model(iv: Intervention):
    import jaxfne as jtfne

    builder = iv.network["builder"]
    build = dict(iv.network["build"])
    if builder == "suite2_net1":
        params = _strict_keys(build, frozenset({"seed", "n"}), "build.suite2_net1")
        return jtfne.construct(jtfne.suite2_net1_config(seed=params["seed"], n=params["n"]))
    if builder == "hdp_column":
        from jaxfne import hdp_network as hn

        params = _strict_keys(
            build,
            frozenset({"n_neurons", "duration_ms", "dt_ms", "seed"}),
            "build.hdp_column",
        )
        return hn.build_model(hn.HDPColumnConfig(**params))
    raise ValueError(f"unknown network builder {builder!r}")


def _realized_weights(model) -> Any:
    import numpy as np

    return np.asarray(model.params["edge_list"].weight)


def _observe(iv: Intervention, signals, model) -> dict[str, Any]:
    import numpy as np

    obs = iv.observation
    if obs["signal"] == "w_final":
        arr = np.asarray(model.last_hdp_diagnostics()["w_final"], dtype=float)
    elif obs["signal"] == "H_final":
        arr = np.asarray(model.last_hdp_diagnostics()["H_final"], dtype=float)
    else:
        arr = np.atleast_1d(float(np.asarray(signals.spikes).sum()))
    if obs["subset"] != "all":
        mask = np.asarray(iv.control["mask"], dtype=float)
        if mask.shape[0] != arr.shape[0]:
            raise ValueError(
                f"control mask length {mask.shape[0]} does not match signal length {arr.shape[0]}"
            )
        keep = mask <= 0.5 if obs["subset"] == "clamped" else mask > 0.5
        arr = arr[keep]
    return {"values": arr.tolist(), "mean": float(arr.mean())}


def _verdict(iv: Intervention, base: dict[str, Any], intr: dict[str, Any]) -> str:
    import numpy as np

    expect = iv.observation["expect"]
    a = np.asarray(base["values"], dtype=float)
    b = np.asarray(intr["values"], dtype=float)
    if expect == "equal":
        return "PASS" if np.array_equal(a, b) else "FAIL"
    if expect == "different":
        return "PASS" if not np.array_equal(a, b) else "FAIL"
    tol = float(iv.observation["tolerance"])
    gap = abs(float(a.mean()) - float(b.mean()))
    if expect == "within":
        return "PASS" if gap <= tol else "FAIL"
    return "PASS" if gap > tol else "FAIL"


def run_intervention(iv: Intervention) -> dict[str, Any]:
    """Execute both arms on the identical realized network; return the result.

    Builds the network twice from the same spec, verifies identical
    realization (spec digest + realized edge weights), runs baseline
    (plastic) and intervened (item-5 control) arms, computes the declared
    observation on each, and verdicts the declared difference. Both arm
    manifests carry ``manifest["intervention"]`` (additive key only).
    """
    import numpy as np

    import jaxfne as jtfne
    from jaxfne import hdp_network as hn
    from jaxfne.public_surface import validate_hdp_params_semantics

    base_hp = dict(iv.network["hdp_params"])
    issues = validate_hdp_params_semantics(base_hp)
    if issues:
        raise ValueError(f"intervention baseline hdp_params invalid: {issues[0]}")
    if "plasticity_mask" in base_hp:
        raise ValueError("baseline hdp_params must not carry a plasticity_mask")
    digest = network_spec_digest(iv)

    model_a = _build_model(iv)
    model_b = _build_model(iv)
    wa = _realized_weights(model_a)
    wb = _realized_weights(model_b)
    if not np.array_equal(wa, wb):
        raise ValueError("identical-network violation: realized weights differ")

    ctl = iv.control
    if ctl["mask"] is not None and len(ctl["mask"]) != len(wb):
        raise ValueError(
            f"control mask length {len(ctl['mask'])} does not match "
            f"realized edge count {len(wb)}"
        )
    if ctl["mode"] == "disable":
        intr_hp = hn.disable_plasticity(
            base_hp, mask=None if ctl["mask"] is None else list(ctl["mask"])
        )
        model_b_int = model_b
    else:
        mask = hn.projection_mask(len(wb), values=list(ctl["mask"]))
        intr_hp = hn.clamp_plasticity(base_hp, mask=mask, value=float(ctl["value"]))
        pinned = hn.pin_projection_weights(wb, mask, float(ctl["value"]))
        model_b_int = model_b.with_hdp_initial_state(w0=pinned)

    run = iv.network["run"]
    rt_base = jtfne.RuntimeConfig(enable_hdp=True, hdp_params=base_hp)
    rt_intr = jtfne.RuntimeConfig(enable_hdp=True, hdp_params=intr_hp)
    sig_a = jtfne.simulate(
        model_a,
        duration_ms=run["duration_ms"],
        dt_ms=run["dt_ms"],
        seed=run["seed"],
        runtime=rt_base,
    )
    sig_b = jtfne.simulate(
        model_b_int,
        duration_ms=run["duration_ms"],
        dt_ms=run["dt_ms"],
        seed=run["seed"],
        runtime=rt_intr,
    )
    obs_a = _observe(iv, sig_a, model_a)
    obs_b = _observe(iv, sig_b, model_b_int)
    verdict = _verdict(iv, obs_a, obs_b)
    manifest = iv.to_manifest()
    arm_a = model_a.manifest(sig_a, intervention={"arm": "baseline", **manifest})
    arm_b = model_b_int.manifest(sig_b, intervention={"arm": "intervened", **manifest})
    return {
        "schema": SCHEMA,
        "name": iv.name,
        "verdict": verdict,
        "spec_digest": digest,
        "baseline": {"observed": obs_a, "manifest": arm_a},
        "intervened": {"observed": obs_b, "manifest": arm_b},
    }
