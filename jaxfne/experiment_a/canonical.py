"""B1 — canonical source-of-truth dataset for Experiment A."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.experiment_a.protocol import load_protocol_spec


def array_sha256(x: Any) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class CanonicalDataset:
    """Frozen neural/source trajectory from one simulate call."""

    seed: int
    time_ms: np.ndarray
    X_V_m: np.ndarray
    X_spikes: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    positions: np.ndarray
    H_semantic: str
    cause_hashes: dict[str, str]
    metadata: dict[str, Any]
    package_head: str


def build_experiment_a_config(spec: dict[str, Any] | None = None) -> Any:
    """Construct the frozen Experiment A circuit from the B0 protocol."""
    spec = spec or load_protocol_spec()
    ns = spec["neural_system"]
    return (
        jtfne.configuration()
        .runtime(
            seed=int(ns["seeds"][0]),
            duration_ms=float(ns["duration_ms"]),
            dt_ms=float(ns["dt_ms"]),
            dtype=str(ns["dtype"]),
            jit=bool(ns["jit_simulate"]),
        )
        .population(
            int(ns["N"]),
            neurons={"E": 0.7, "I": 0.3},
            layers=list(ns["layers"]),
            name="V1",
        )
        .cell_types(dict(ns["cell_types"]))
        .geometry(layer_thickness=dict(ns["layer_thickness"]))
        .cell_type_drives(dict(ns["drives"]))
        .set_emitter(ns["emitter"]["family"], ns["emitter"]["preset"])
        .field(
            domain=ns["field_geometry"]["domain"],
            conductivity=ns["field_geometry"]["conductivity"],
            boundary=ns["field_geometry"]["boundary"],
            gauge=ns["field_geometry"]["gauge"],
        )
        .probe(name="experiment_a_probe", modes=["spikes", "V_m"])
    )


def freeze_canonical_dataset(
    *,
    spec: dict[str, Any] | None = None,
    package_head: str | None = None,
    duration_ms: float | None = None,
) -> CanonicalDataset:
    """Simulate once per protocol and freeze X, H, Q arrays.

    ``duration_ms`` overrides protocol duration for fast tests only; production
    runs must use the frozen protocol value (None).
    """
    import subprocess

    spec = spec or load_protocol_spec()
    ns = spec["neural_system"]
    seed = int(ns["seeds"][0])
    dt_ms = float(ns["dt_ms"])
    dur = float(duration_ms if duration_ms is not None else ns["duration_ms"])

    cfg = build_experiment_a_config(spec)
    if duration_ms is not None:
        cfg = cfg.runtime(duration_ms=dur)

    model = jtfne.construct(cfg)
    sim = jtfne.Simulation(
        duration_ms=dur,
        dt_ms=dt_ms,
        seed=seed,
        record_sources=True,
        record_fields=False,
        runtime=jtfne.RuntimeConfig(
            dtype=str(ns["dtype"]),
            jit=bool(ns["jit_simulate"]),
            seed=seed,
        ),
    )
    signals = model.simulate(sim)

    Q = np.asarray(signals.sources, dtype=np.float32)
    V = np.asarray(signals.V_m, dtype=np.float32)
    spikes = np.asarray(signals.spikes)
    positions = np.asarray(model.params["positions"], dtype=np.float32)
    n_steps, n_units = Q.shape
    H = np.ones((n_steps, n_units), dtype=np.float32)
    time_ms = np.arange(n_steps, dtype=np.float32) * np.float32(dt_ms)

    cause_hashes = {
        "V_m": array_sha256(V),
        "spikes": array_sha256(spikes),
        "Q": array_sha256(Q),
        "H": array_sha256(H),
        "positions": array_sha256(positions),
    }

    if package_head is None:
        root = Path(__file__).resolve().parents[2]
        package_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()

    meta = dict(signals.metadata)
    meta["experiment_a_seed"] = seed
    meta["experiment_a_H_semantic"] = str(ns["H_semantic_when_hdp_off"])

    return CanonicalDataset(
        seed=seed,
        time_ms=time_ms,
        X_V_m=V,
        X_spikes=spikes,
        H=H,
        Q=Q,
        positions=positions,
        H_semantic=str(ns["H_semantic_when_hdp_off"]),
        cause_hashes=cause_hashes,
        metadata=meta,
        package_head=package_head,
    )


def load_frozen_canonical_dataset(
    *,
    npz_path: Path | None = None,
    receipt_path: Path | None = None,
    verify_hashes: bool = True,
) -> CanonicalDataset:
    """Load committed Experiment A canonical arrays from disk (one simulate frozen)."""
    root = Path(__file__).resolve().parents[2]
    npz_path = npz_path or root / "artifacts" / "etudes" / "experiment_a" / "canonical_source.npz"
    receipt_path = receipt_path or root / "artifacts" / "etudes" / "experiment_a" / "b1_canonical_receipt.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing canonical npz: {npz_path}")
    data = np.load(npz_path)
    cause_hashes = {
        "V_m": array_sha256(np.asarray(data["X_V_m"], dtype=np.float32)),
        "spikes": array_sha256(np.asarray(data["X_spikes"])),
        "Q": array_sha256(np.asarray(data["Q"], dtype=np.float32)),
        "H": array_sha256(np.asarray(data["H"], dtype=np.float32)),
        "positions": array_sha256(np.asarray(data["positions"], dtype=np.float32)),
    }
    dataset = CanonicalDataset(
        seed=int(data["seed"]),
        time_ms=np.asarray(data["time_ms"], dtype=np.float32),
        X_V_m=np.asarray(data["X_V_m"], dtype=np.float32),
        X_spikes=np.asarray(data["X_spikes"]),
        H=np.asarray(data["H"], dtype=np.float32),
        Q=np.asarray(data["Q"], dtype=np.float32),
        positions=np.asarray(data["positions"], dtype=np.float32),
        H_semantic=str(np.asarray(data["H_semantic"])),
        cause_hashes=cause_hashes,
        metadata={},
        package_head=str(np.asarray(data["package_head"])),
    )
    if verify_hashes and receipt_path.is_file():
        import json

        receipt = json.loads(receipt_path.read_text())
        expected = receipt.get("cause_hashes", {})
        for key, digest in cause_hashes.items():
            if key in expected and expected[key] != digest:
                raise ValueError(
                    f"canonical {key} hash mismatch: disk={digest} receipt={expected[key]}"
                )
    return dataset


def write_canonical_npz(dataset: CanonicalDataset, path: Path) -> None:
    """Persist canonical arrays (local/gitignored .npz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        seed=np.int32(dataset.seed),
        time_ms=dataset.time_ms,
        X_V_m=dataset.X_V_m,
        X_spikes=dataset.X_spikes,
        H=dataset.H,
        Q=dataset.Q,
        positions=dataset.positions,
        H_semantic=np.array(dataset.H_semantic),
        package_head=np.array(dataset.package_head),
    )


def write_b1_receipt(dataset: CanonicalDataset, path: Path) -> dict[str, Any]:
    """Write committed B1 receipt with hashes (arrays stay in .npz)."""
    receipt = {
        "schema": "jaxfne.experiment_a.b1_canonical_receipt.v1",
        "checkpoint": "B1",
        "status": "FROZEN",
        "protocol_id": "experiment_a_v0417_b",
        "seed": dataset.seed,
        "package_head": dataset.package_head,
        "H_semantic": dataset.H_semantic,
        "shapes": {
            "time_ms": list(dataset.time_ms.shape),
            "X_V_m": list(dataset.X_V_m.shape),
            "X_spikes": list(dataset.X_spikes.shape),
            "H": list(dataset.H.shape),
            "Q": list(dataset.Q.shape),
            "positions": list(dataset.positions.shape),
        },
        "cause_hashes": dataset.cause_hashes,
        "canonical_npz": "artifacts/etudes/experiment_a/canonical_source.npz",
        "next_checkpoint": "B2",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
