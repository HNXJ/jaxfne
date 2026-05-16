"""Core object model for :mod:`jaxfne`.

Design target: object-oriented public API, pure-JAX computational core.  The
current package is an honest TFNE scaffold: reduced emitters plus laminar proxy
source/readout status, not a full PDE field solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from .emitters import EIGNetwork, IzhikevichParams, make_eig_network, simulate_eig_izhikevich
from .fields import FieldOutput, probe_laminar_modes, project_laminar_sources
from .io import config_hash, manifest as build_manifest


def _default_operator_status() -> dict[str, str]:
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


def _default_metadata() -> dict[str, Any]:
    return {
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "source_projection_mode": "proxy_no_field_solve",
        "source_decomposition": "proxy_reduced_emitter",
        "boundary_condition": "mean_zero_neumann",
        "gauge": "mean_zero",
        "csd_sign_convention": "proxy_positive_equals_extracellular_source_like",
        "field_solver_status": "laminar_proxy_no_pde",
        "manifest_schema_version": "0.0.4",
        "operator_status": _default_operator_status(),
    }


@dataclass(frozen=True)
class Configuration:
    """Declarative TFNE model configuration.

    It is the anatomical/model declaration, not the compiled model.  Methods
    return new objects, making the public API friendly while keeping mutation
    out of the construction path.
    """

    networks: list[dict[str, Any]] = field(default_factory=list)
    emitters: list[dict[str, Any]] = field(default_factory=list)
    fields: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=_default_metadata)

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

    def validate(self) -> dict[str, Any]:
        issues: list[str] = []
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
class RuntimeConfig:
    """JAX runtime and dtype policy.

    ``dtype='float64'`` is honored only when JAX x64 is enabled.  The manifest
    always reports both requested and actual dtype policy.
    """

    backend: str = "auto"  # "auto" | "cpu" | "gpu" | "tpu"
    dtype: str = "float32"  # "float32" | "float64"
    jit: bool = False
    vmap: bool = False
    precision: str = "default"  # "default" | "high"
    seed: int = 0
    n_steps: int = 0
    # v0.0.3 compatibility names; if provided by old caller, they are folded in.
    device_type: Optional[str] = None
    dtype_primary: Optional[str] = None
    x64_enabled: Optional[bool] = None

    @property
    def requested_dtype(self) -> str:
        return self.dtype_primary or self.dtype

    @property
    def selected_backend(self) -> str:
        return self.device_type or self.backend

    @property
    def actual_dtype(self) -> str:
        if self.requested_dtype == "float64" and bool(jax.config.read("jax_enable_x64")):
            return "float64"
        return "float32"

    @property
    def jnp_dtype(self) -> jnp.dtype:
        return jnp.float64 if self.actual_dtype == "float64" else jnp.float32

    def runtime_report(self) -> dict[str, Any]:
        devices = [str(device) for device in jax.devices()]
        return {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jaxlib_version": _jaxlib_version(),
            "default_backend": jax.default_backend(),
            "available_devices": devices,
            "selected_backend": self.selected_backend,
            "backend": self.backend,
            "requested_dtype": self.requested_dtype,
            "actual_dtype": self.actual_dtype,
            "dtype": self.dtype,
            "jit": bool(self.jit),
            "vmap": bool(self.vmap),
            "precision": self.precision,
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "seed": int(self.seed),
            "n_steps": int(self.n_steps),
            # Compatibility keys from v0.0.3.
            "device_type": self.selected_backend,
            "dtype_primary": self.actual_dtype,
        }


def _jaxlib_version() -> str:
    try:
        import jaxlib  # type: ignore

        return str(jaxlib.__version__)
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class Simulation:
    duration_ms: float = 1000.0
    dt_ms: float = 0.05
    plasticity: float = 0.0
    seed: int = 0
    record_sources: bool = True
    record_fields: bool = True
    runtime: RuntimeConfig | None = None

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_ms / self.dt_ms))

    @property
    def resolved_runtime(self) -> RuntimeConfig:
        base = self.runtime or RuntimeConfig(seed=self.seed, n_steps=self.n_steps)
        return replace(base, seed=self.seed, n_steps=self.n_steps)

    def with_plasticity(self, gain: float) -> "Simulation":
        return replace(self, plasticity=float(gain))


@dataclass(frozen=True)
class Probe:
    name: str
    modes: Sequence[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Signals:
    """Simulation output container holding multiple arrays."""

    time_ms: jax.Array
    V_m: jax.Array
    spikes: jax.Array
    sources: jax.Array
    field: Optional[FieldOutput]
    metadata: dict[str, Any]


# Backwards-compatible alias.
Signal = Signals


@dataclass(frozen=True)
class Objective:
    losses: list[dict[str, Any]] = field(default_factory=list)
    regularizers: list[dict[str, Any]] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)

    def loss(
        self,
        name: str,
        fn: Optional[Callable[..., Any]] = None,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> "Objective":
        return replace(self, losses=[*self.losses, {"name": name, "fn": fn, "weight": weight, **kwargs}])

    def regularizer(
        self,
        name: str,
        target: float = 0.0,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> "Objective":
        return replace(
            self,
            regularizers=[*self.regularizers, {"name": name, "target": target, "weight": weight, **kwargs}],
        )

    def gate(self, name: str, **kwargs: Any) -> "Objective":
        return replace(self, gates=[*self.gates, {"name": name, **kwargs}])


@dataclass(frozen=True)
class Paradigm:
    name: str = "none"
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def habituation(self, sequence: Sequence[str], n_trials: int) -> "Paradigm":
        return replace(
            self,
            blocks=[*self.blocks, {"kind": "habituation", "sequence": list(sequence), "n_trials": n_trials}],
        )

    def main_block(self, **kwargs: Any) -> "Paradigm":
        return replace(self, blocks=[*self.blocks, {"kind": "main_block", **kwargs}])

    def batch(self, n_trials: int, seed: int = 0) -> dict[str, Any]:
        return {"name": self.name, "n_trials": n_trials, "seed": seed, "blocks": self.blocks}


@dataclass(frozen=True)
class Model:
    cfg: Configuration
    params: dict[str, Any]
    static: dict[str, Any]

    def simulate(self, sim: Simulation, paradigm: Optional[Mapping[str, Any]] = None) -> Signals:
        """Run the default EIG/Izhikevich vertical slice."""

        runtime_cfg = sim.resolved_runtime
        key = jax.random.PRNGKey(sim.seed)
        emitter: IzhikevichParams = self.params["emitter"]
        voltages, spikes, sources = simulate_eig_izhikevich(
            emitter,
            sim.n_steps,
            sim.dt_ms,
            key,
            dtype=runtime_cfg.actual_dtype,
        )
        time_ms = jnp.arange(sim.n_steps, dtype=runtime_cfg.jnp_dtype) * jnp.asarray(
            sim.dt_ms, dtype=runtime_cfg.jnp_dtype
        )
        positions = jnp.asarray(self.params["positions"], dtype=runtime_cfg.jnp_dtype)
        field_output = None
        if sim.record_fields:
            field_output = project_laminar_sources(
                sources=sources,
                positions=positions,
                n_contacts=self.static.get("n_contacts", 16),
                dtype=runtime_cfg.actual_dtype,
            )
        metadata = {
            "config_hash": config_hash(self.cfg),
            "source_calibration_status": self.cfg.metadata.get("source_calibration_status"),
            "paradigm": dict(paradigm) if paradigm is not None else None,
            "plasticity_gain": sim.plasticity,
            "runtime": runtime_cfg.runtime_report(),
        }
        return Signals(
            time_ms=time_ms,
            V_m=voltages.astype(runtime_cfg.jnp_dtype),
            spikes=spikes,
            sources=sources.astype(runtime_cfg.jnp_dtype),
            field=field_output,
            metadata=metadata,
        )

    def probe(self, signals: Signals, modes: Sequence[str] | None = None) -> dict[str, Any]:
        """Canonical TFNE readout method."""

        modes = list(modes or [])
        out: dict[str, Any] = {"requested_modes": modes}
        if "spikes" in modes:
            out["spikes"] = signals.spikes
        if "V_m" in modes:
            out["V_m"] = signals.V_m
        if "source" in modes or "sources" in modes:
            out["sources"] = signals.sources
        if signals.field is not None:
            out.update(probe_laminar_modes(signals.field, modes))
        return out

    def record(self, signals: Signals, modes: Sequence[str]) -> dict[str, Any]:
        """User-friendly alias for :meth:`probe`."""

        return self.probe(signals, modes)

    def evaluate(self, signals: Signals, objective: Objective | str) -> dict[str, Any]:
        """Minimal smoke evaluator; full objective path is reserved for v0.0.5."""

        dt_ms = signals.time_ms[1] - signals.time_ms[0]
        spike_rate_hz = jnp.mean(signals.spikes) * (1000.0 / dt_ms)
        mean_pairwise_corr_proxy = _mean_pairwise_corr_proxy(signals.spikes)
        return {
            "evaluation_status": "smoke_only_objective_path_future",
            "spike_rate_hz_mean": float(spike_rate_hz),
            "mean_pairwise_correlation_proxy": float(mean_pairwise_corr_proxy),
            "objective": objective
            if isinstance(objective, str)
            else {
                "losses": [x["name"] for x in objective.losses],
                "regularizers": [x["name"] for x in objective.regularizers],
                "gates": [x["name"] for x in objective.gates],
            },
        }

    def tune(self, objective: Objective, optimizer: Any = None, steps: int = 1) -> "Model":
        """Placeholder: object API for future Optax/GSDR tuning."""

        _ = (objective, optimizer, steps)
        return self

    def manifest(self, signals: Optional[Signals] = None, readout: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        runtime_cfg = None
        if signals is not None and "runtime" in signals.metadata:
            # io.manifest accepts any object with runtime_report(); keep a tiny adapter.
            runtime_cfg = _RuntimeReportAdapter(signals.metadata["runtime"])
        return build_manifest(self.cfg, signals=signals, readout=readout, runtime_config=runtime_cfg)


@dataclass(frozen=True)
class _RuntimeReportAdapter:
    report: dict[str, Any]

    def runtime_report(self) -> dict[str, Any]:
        return self.report


def _mean_pairwise_corr_proxy(spikes: jax.Array) -> jax.Array:
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


def runtime(
    backend: str = "auto",
    dtype: str = "float32",
    jit: bool = False,
    vmap: bool = False,
    precision: str = "default",
    seed: int = 0,
    n_steps: int = 0,
    # v0.0.3 compatibility names.
    device_type: str | None = None,
    dtype_primary: str | None = None,
    x64_enabled: bool | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        backend=backend,
        dtype=dtype,
        jit=jit,
        vmap=vmap,
        precision=precision,
        seed=seed,
        n_steps=n_steps,
        device_type=device_type,
        dtype_primary=dtype_primary,
        x64_enabled=x64_enabled,
    )


def runtime_report(runtime_config: RuntimeConfig | None = None) -> dict[str, Any]:
    return (runtime_config or RuntimeConfig()).runtime_report()


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
    network: EIGNetwork = make_eig_network(n=n, cell_type_fractions=cell_types)
    return Model(
        cfg=cfg,
        params={"emitter": network.params, "positions": network.positions},
        static={"n_contacts": 16, "operator_status": operator_status()},
    )


def operator_status() -> dict[str, str]:
    return _default_operator_status()
