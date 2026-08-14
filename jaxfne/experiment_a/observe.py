"""B2 — independent observation operators at fixed (Q, G, M)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxfne.experiment_a.canonical import CanonicalDataset, array_sha256
from jaxfne.experiment_a.protocol import load_protocol_spec
from jaxfne.fields import csd_proxy_probe, lfp_proxy_probe, project_laminar_sources


@dataclass(frozen=True)
class FactorizedObservation:
    """Materialized F then P on frozen canonical Q."""

    field_id: str
    probe_id: str
    Y: np.ndarray
    phi_e: np.ndarray
    q_hash_before: str
    q_hash_after: str
    probe_report: dict[str, Any]


def _field_params(spec: dict[str, Any], field_id: str) -> dict[str, Any]:
    for entry in spec["field_operators_F"]:
        if entry["id"] == field_id:
            return dict(entry["params"])
    raise KeyError(f"unknown field operator {field_id!r}")


def materialize_field(
    dataset: CanonicalDataset,
    field_id: str = "lfp_ref",
    spec: dict[str, Any] | None = None,
):
    """Apply F_{G,M} to frozen Q without re-simulating neural dynamics."""
    spec = spec or load_protocol_spec()
    params = _field_params(spec, field_id)
    return project_laminar_sources(
        jnp.asarray(dataset.Q),
        jnp.asarray(dataset.positions),
        **params,
    )


def apply_independent_probe(
    dataset: CanonicalDataset,
    field_output,
    probe_id: str,
    spec: dict[str, Any] | None = None,
) -> FactorizedObservation:
    """Apply probe P on materialized Phi; Q must remain invariant."""
    spec = spec or load_protocol_spec()
    q_before = array_sha256(dataset.Q)
    phi_e = np.asarray(field_output.phi_e_proxy)
    field_z = np.asarray(field_output.contact_depths)

    if probe_id == "lfp_contact_shallow":
        depths = jnp.asarray([0.20], dtype=jnp.float32)
        readout = lfp_proxy_probe(
            jnp.asarray(phi_e), contact_depths=depths, field_contact_depths=jnp.asarray(field_z)
        )
        Y = np.asarray(readout.data)
        report = readout.report
    elif probe_id == "lfp_contact_deep":
        depths = jnp.asarray([0.80], dtype=jnp.float32)
        readout = lfp_proxy_probe(
            jnp.asarray(phi_e), contact_depths=depths, field_contact_depths=jnp.asarray(field_z)
        )
        Y = np.asarray(readout.data)
        report = readout.report
    elif probe_id == "csd_from_lfp_ref":
        readout = csd_proxy_probe(jnp.asarray(field_output.csd_proxy))
        Y = np.asarray(readout.data)
        report = readout.report
    else:
        raise KeyError(f"unknown B2 probe {probe_id!r}")

    q_after = array_sha256(dataset.Q)
    return FactorizedObservation(
        field_id="lfp_ref",
        probe_id=probe_id,
        Y=Y,
        phi_e=phi_e,
        q_hash_before=q_before,
        q_hash_after=q_after,
        probe_report=report,
    )


def verify_b2_invariants(
    dataset: CanonicalDataset,
    *,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check Q-invariance and probe-dependent locality at fixed F."""
    spec = spec or load_protocol_spec()
    field = materialize_field(dataset, "lfp_ref", spec=spec)
    shallow = apply_independent_probe(dataset, field, "lfp_contact_shallow", spec)
    deep = apply_independent_probe(dataset, field, "lfp_contact_deep", spec)

    q_hash = array_sha256(dataset.Q)
    distinct = float(
        np.linalg.norm(shallow.Y - deep.Y)
        / max(np.linalg.norm(shallow.Y), np.linalg.norm(deep.Y), 1e-12)
    )
    zero_field = project_laminar_sources(
        jnp.zeros_like(jnp.asarray(dataset.Q)),
        jnp.asarray(dataset.positions),
        **_field_params(spec, "lfp_ref"),
    )
    zero_probe = lfp_proxy_probe(
        zero_field.phi_e_proxy,
        contact_depths=jnp.asarray([0.20], dtype=jnp.float32),
        field_contact_depths=zero_field.contact_depths,
    )

    return {
        "q_hash_invariant": (
            shallow.q_hash_before == shallow.q_hash_after == deep.q_hash_after == q_hash
        ),
        "shallow_vs_deep_rel_distinctness": distinct,
        "probe_distinct": distinct > 1e-3,
        "Q_zero_implies_Y_zero": bool(np.max(np.abs(np.asarray(zero_probe.data))) < 1e-6),
        "phi_e_unchanged_between_probes": shallow.phi_e is deep.phi_e
        or np.shares_memory(shallow.phi_e, deep.phi_e)
        or np.array_equal(shallow.phi_e, deep.phi_e),
    }


def write_b2_receipt(results: dict[str, Any], path) -> dict[str, Any]:
    import json
    from pathlib import Path

    receipt = {
        "schema": "jaxfne.experiment_a.b2_probe_receipt.v1",
        "checkpoint": "B2",
        "status": "FROZEN",
        "protocol_id": "experiment_a_v0417_b",
        "fixed_field": "lfp_ref",
        "invariants": results,
        "lfp_proxy_probe_fix": "contact_depths interpolates phi_e along field_contact_depths",
        "next_checkpoint": "B3",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
