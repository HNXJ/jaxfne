"""Core object model for jaxfne.

Design target: object-oriented public API, pure-JAX computational core.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from .emitters import izhikevich_eig_params, simulate_izhikevich_eig
from .fields import FieldOutput, project_sources_to_laminar_field
from .io import config_hash, manifest


@dataclass(frozen=True)
class Configuration:
    """Declarative TFNE model configuration.

    This object is deliberately broad. It is the anatomical/model declaration, not the
    compiled model. Methods return new objects to remain JAX/PyTree friendly.
    """

    networks: List[Dict[str, Any]] = field(default_factory=list)
    emitters: List[Dict[str, Any]] = field(default_factory=list)
    fields: List[Dict[str, Any]] = field(default_factory=list)
    probes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
    })

    def network(self, **kwargs: Any) -> "Configuration":
        return replace(self, networks=[*self.networks, dict(kwargs)])

    def emitter(self, **kwargs: Any) -> "Configuration":
        return replace(self, emitters=[*self.emitters, dict(kwargs)])

    def field(self, **kwargs: Any) -> "Configuration":
        return replace(self, fields=[*self.fields, dict(kwargs)])

    def probe(self, **kwargs: Any) -> "Configuration":
        return replace(self, probes=[*self.probes, dict(kwargs)])

    def update_metadata(self, **kwargs: Any) -> "Configuration":
        metadata = dict(self.metadata)
        metadata.update(kwargs)
        return replace(self, metadata=metadata)

    def validate(self) -> Dict[str, Any]:
        issues: List[str] = []
        if not self.networks:
            issues.append("no_networks_declared")
        if not self.emitters:
            issues.append("no_emitters_declared")
        if not self.fields:
            issues.append("no_field_declared")
        if not self.probes:
            issues.append("no_probes_declared")
        return {
            "valid": not issues,
            "issues": issues,
            "config_hash": config_hash(self),
            "truth_mode": self.metadata.get("truth_mode"),
            "claim_level": self.metadata.get("claim_level"),
        }


@dataclass(frozen=True)
class Simulation:
    duration_ms: float = 1000.0
    dt_ms: float = 0.05
    plasticity: float = 0.0
    seed: int = 0
    record_sources: bool = True
    record_fields: bool = True

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_ms / self.dt_ms))

    def with_plasticity(self, gain: float) -> "Simulation":
        return replace(self, plasticity=float(gain))


@dataclass(frozen=True)
class Probe:
    name: str
    modes: Sequence[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    """Simulation output container."""

    time_ms: jnp.ndarray
    V_m: jnp.ndarray
    spikes: jnp.ndarray
    sources: jnp.ndarray
    field: Optional[FieldOutput]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class Objective:
    losses: List[Dict[str, Any]] = field(default_factory=list)
    regularizers: List[Dict[str, Any]] = field(default_factory=list)
    gates: List[Dict[str, Any]] = field(default_factory=list)

    def loss(self, name: str, fn: Optional[Callable[..., Any]] = None, weight: float = 1.0, **kwargs: Any) -> "Objective":
        return replace(self, losses=[*self.losses, {"name": name, "fn": fn, "weight": weight, **kwargs}])

    def regularizer(self, name: str, target: float = 0.0, weight: float = 1.0, **kwargs: Any) -> "Objective":
        return replace(self, regularizers=[*self.regularizers, {"name": name, "target": target, "weight": weight, **kwargs}])

    def gate(self, name: str, **kwargs: Any) -> "Objective":
        return replace(self, gates=[*self.gates, {"name": name, **kwargs}])


@dataclass(frozen=True)
class Paradigm:
    name: str = "none"
    blocks: List[Dict[str, Any]] = field(default_factory=list)

    def habituation(self, sequence: Sequence[str], n_trials: int) -> "Paradigm":
        return replace(self, blocks=[*self.blocks, {"kind": "habituation", "sequence": list(sequence), "n_trials": n_trials}])

    def main_block(self, **kwargs: Any) -> "Paradigm":
        return replace(self, blocks=[*self.blocks, {"kind": "main_block", **kwargs}])

    def batch(self, n_trials: int, seed: int = 0) -> Dict[str, Any]:
        return {"name": self.name, "n_trials": n_trials, "seed": seed, "blocks": self.blocks}


@dataclass(frozen=True)
class Model:
    cfg: Configuration
    params: Dict[str, Any]
    static: Dict[str, Any]

    def simulate(self, sim: Simulation, paradigm: Optional[Mapping[str, Any]] = None) -> Signal:
        """Run the default EIG/Izhikevich vertical slice."""
        key = jax.random.PRNGKey(sim.seed)
        V_m, spikes, sources = simulate_izhikevich_eig(self.params["emitter"], sim.n_steps, sim.dt_ms, key)
        time_ms = jnp.arange(sim.n_steps) * sim.dt_ms
        field_output = None
        if sim.record_fields:
            field_output = project_sources_to_laminar_field(
                sources=sources,
                positions=self.params["positions"],
                n_contacts=self.static.get("n_contacts", 16),
            )
        metadata = {
            "config_hash": config_hash(self.cfg),
            "source_calibration_status": self.cfg.metadata.get("source_calibration_status"),
            "paradigm": dict(paradigm) if paradigm is not None else None,
            "plasticity_gain": sim.plasticity,
        }
        return Signal(time_ms=time_ms, V_m=V_m, spikes=spikes, sources=sources, field=field_output, metadata=metadata)

    def record(self, signals: Signal, modes: Sequence[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "spikes" in modes:
            out["spikes"] = signals.spikes
        if "V_m" in modes:
            out["V_m"] = signals.V_m
        if "source" in modes or "sources" in modes:
            out["sources"] = signals.sources
        if signals.field is not None:
            if "LFP" in modes:
                out["LFP"] = signals.field.lfp
            if "CSD" in modes:
                out["CSD"] = signals.field.csd
            if "phi_e" in modes:
                out["phi_e"] = signals.field.phi_e
            if "J_e" in modes:
                out["J_e"] = signals.field.J_e
        return out

    def evaluate(self, signals: Signal, objective: Objective | str) -> Dict[str, Any]:
        spike_rate_hz = jnp.mean(signals.spikes) * (1000.0 / (signals.time_ms[1] - signals.time_ms[0]))
        mean_pairwise_corr_proxy = _mean_pairwise_corr_proxy(signals.spikes)
        return {
            "spike_rate_hz_mean": float(spike_rate_hz),
            "mean_pairwise_correlation_proxy": float(mean_pairwise_corr_proxy),
            "objective": objective if isinstance(objective, str) else {
                "losses": [x["name"] for x in objective.losses],
                "regularizers": [x["name"] for x in objective.regularizers],
                "gates": [x["name"] for x in objective.gates],
            },
        }

    def tune(self, objective: Objective, optimizer: Any = None, steps: int = 1) -> "Model":
        """Placeholder: object API for future Optax/GSDR tuning.

        Returns self for now and records no parameter updates. This makes the skeleton
        API-editable before optimization behavior is finalized.
        """
        _ = (objective, optimizer, steps)
        return self

    def manifest(self, signals: Optional[Signal] = None) -> Dict[str, Any]:
        return manifest(self.cfg, signals)


def _mean_pairwise_corr_proxy(spikes: jnp.ndarray) -> jnp.ndarray:
    x = spikes.astype(jnp.float32)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    denom = jnp.std(x, axis=0, keepdims=True) + 1e-6
    z = x / denom
    corr = (z.T @ z) / jnp.maximum(1, z.shape[0] - 1)
    n = corr.shape[0]
    mask = 1.0 - jnp.eye(n)
    return jnp.sum(jnp.abs(corr) * mask) / jnp.maximum(1.0, jnp.sum(mask))


def configuration() -> Configuration:
    return Configuration()


def simulation(**kwargs: Any) -> Simulation:
    return Simulation(**kwargs)


def objective() -> Objective:
    return Objective()


def paradigm(name: str = "none") -> Paradigm:
    return Paradigm(name=name)


def construct(cfg: Configuration) -> Model:
    validation = cfg.validate()
    if not validation["valid"]:
        raise ValueError(f"Invalid jaxfne configuration: {validation['issues']}")

    net = cfg.networks[0]
    n = int(net.get("n", 100))
    cell_types = net.get("cell_types", {"E": 0.8, "PV": 0.1, "SST": 0.1})
    params = izhikevich_eig_params(n=n, cell_type_fractions=cell_types)
    positions = _laminar_positions(n)
    return Model(
        cfg=cfg,
        params={"emitter": params, "positions": positions},
        static={"n_contacts": 16, "operator_status": operator_status()},
    )


def _laminar_positions(n: int) -> jnp.ndarray:
    depth = jnp.linspace(0.0, 1.0, n)
    return jnp.stack([jnp.zeros(n), jnp.zeros(n), depth], axis=1)


def operator_status() -> Dict[str, str]:
    return {
        "E_theta": "prototype_api",
        "S_WDR": "prototype_api",
        "C_mu_nu": "specified_future_module",
        "Q_eta_alpha": "prototype_api",
        "F_field": "prototype_api",
        "P_probe": "prototype_api",
        "A_objective": "prototype_api",
        "O_optimizer": "specified_future_module",
        "C_constraints": "prototype_api",
    }
