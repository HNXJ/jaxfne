"""I/O, hashing, and manifest helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np


def json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.ndarray,)):
        return json_safe(obj.tolist())
    if hasattr(obj, "shape") and hasattr(obj, "tolist"):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def config_hash(cfg: Any) -> str:
    payload = json.dumps(json_safe(cfg), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def manifest(cfg: Any, signals: Optional[Any] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "package": "jaxfne",
        "truth_mode": getattr(cfg, "metadata", {}).get("truth_mode", "truth_safe_unverified"),
        "claim_level": getattr(cfg, "metadata", {}).get("claim_level", "computational_scaffold"),
        "config_hash": config_hash(cfg),
        "source_calibration_status": getattr(cfg, "metadata", {}).get(
            "source_calibration_status", "uncalibrated_izhikevich_native_current"
        ),
    }
    if signals is not None:
        data["signals"] = {
            "n_time": int(signals.time_ms.shape[0]),
            "n_neurons": int(signals.V_m.shape[1]),
            "has_field": signals.field is not None,
            "metadata": signals.metadata,
        }
        if signals.field is not None:
            data["field_diagnostics"] = signals.field.diagnostics
    return json_safe(data)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
