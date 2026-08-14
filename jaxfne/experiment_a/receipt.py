"""B3 — Experiment A immutable receipt bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.analysis.spectral import spectrolaminar_psd_jax
from jaxfne.experiment_a.canonical import (
    CanonicalDataset,
    array_sha256,
    freeze_canonical_dataset,
    write_b1_receipt,
    write_canonical_npz,
)
from jaxfne.experiment_a.metrics import max_rel_diff, mean_r90
from jaxfne.experiment_a.observe import apply_independent_probe, materialize_field, verify_b2_invariants
from jaxfne.experiment_a.protocol import PROTOCOL_ID, PROTOCOL_SPEC_PATH, load_protocol_spec
from jaxfne.fields import LinearReadout, eeg_proxy_transform, meg_proxy_transform, project_laminar_sources
from jaxfne.io import json_safe

BUNDLE_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "etudes" / "experiment_a"
REPO_ROOT = Path(__file__).resolve().parents[2]
BURN_IN_MS = 200.0
FS = 2000.0  # 1 / 0.5 ms dt
Y_ATOL = 1e-5
Y_DISTINCT_RTOL = 1e-3


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def gaussian_leadfield(z: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)
    return np.exp(-0.5 * ((centers[:, None] - z[None, :]) / widths[:, None]) ** 2).astype(np.float32)


def psd_of(y: np.ndarray, freqs: Any, fs: float = FS) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    sig = jnp.asarray(arr[None, ...])
    return np.asarray(spectrolaminar_psd_jax(sig, fs=float(fs), freqs=freqs))


def spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    p = np.asarray(psd, dtype=np.float64)
    if p.ndim == 2:
        p = p.mean(axis=1)
    mass = float(p.sum())
    if mass <= 0.0:
        return float("nan")
    return float(np.dot(np.asarray(freqs, dtype=np.float64), p) / mass)


def run_observation_suite(
    dataset: CanonicalDataset,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Post-hoc observation operators on frozen canonical Q only."""
    spec = spec or load_protocol_spec()
    Q = dataset.Q
    positions = dataset.positions
    z = positions[:, 2]
    dt_ms = float(spec["neural_system"]["dt_ms"])
    burn = int(round(BURN_IN_MS / dt_ms))
    q_hash = array_sha256(Q)

    fields: dict[str, Any] = {}
    for entry in spec["field_operators_F"]:
        fields[entry["id"]] = project_laminar_sources(
            jnp.asarray(Q), jnp.asarray(positions), **entry["params"]
        )
        if array_sha256(Q) != q_hash:
            raise RuntimeError("Q mutated during field operator pass")

    lfp_ref = fields["lfp_ref"]
    b2 = verify_b2_invariants(dataset, spec=spec)
    shallow = apply_independent_probe(dataset, lfp_ref, "lfp_contact_shallow", spec)
    deep = apply_independent_probe(dataset, lfp_ref, "lfp_contact_deep", spec)

    W_sup = gaussian_leadfield(z, np.array([0.25, 0.25, 0.25]), np.array([0.18, 0.20, 0.22]))
    W_deep = gaussian_leadfield(z, np.array([0.75, 0.75, 0.75]), np.array([0.18, 0.20, 0.22]))
    eeg_sup = LinearReadout(name="eeg_superficial", W=jnp.asarray(W_sup), leadfield_status="toy_or_declared_proxy")
    eeg_deep = LinearReadout(name="eeg_deep", W=jnp.asarray(W_deep), leadfield_status="toy_or_declared_proxy")
    Y_eeg_sup = np.asarray(eeg_sup.apply(jnp.asarray(Q)))
    Y_eeg_deep = np.asarray(eeg_deep.apply(jnp.asarray(Q)))

    rng = np.random.default_rng(dataset.seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=W_sup.shape[1])
    W_meg = (W_sup * signs[None, :]).astype(np.float32)
    meg_ro = LinearReadout(name="meg_relative", W=jnp.asarray(W_meg), leadfield_status="toy_or_declared_proxy")
    Y_meg = np.asarray(meg_ro.apply(jnp.asarray(Q)))

    meta_a = LinearReadout(name="eeg_meta_a", W=jnp.asarray(W_sup), leadfield_status="toy_or_declared_proxy")
    meta_b = LinearReadout(name="eeg_meta_b", W=jnp.asarray(W_sup), leadfield_status="declared")
    Y_meta_a = np.asarray(meta_a.apply(jnp.asarray(Q)))
    Y_meta_b = np.asarray(meta_b.apply(jnp.asarray(Q)))

    K_ref = np.asarray(lfp_ref.kernel)
    Y_compiled = np.asarray(LinearReadout(name="lfp_compiled", W=jnp.asarray(K_ref)).apply(jnp.asarray(Q)))
    Y_lfp_ref = np.asarray(lfp_ref.lfp_proxy)

    freqs = jnp.linspace(1.0, 150.0, 96, dtype=jnp.float32)
    freqs_np = np.asarray(freqs)
    Q_spec = Q[burn:]
    psd_q = psd_of(Q_spec, freqs)
    psd_lfp = psd_of(np.asarray(lfp_ref.lfp_proxy)[burn:], freqs)
    psd_csd = psd_of(np.asarray(lfp_ref.csd_proxy)[burn:], freqs)
    psd_eeg_sup = psd_of(Y_eeg_sup[burn:], freqs)
    psd_eeg_deep = psd_of(Y_eeg_deep[burn:], freqs)

    r90 = {
        name: mean_r90(np.asarray(fo.kernel), z, np.asarray(fo.contact_depths))
        for name, fo in fields.items()
    }

    distinct = {
        "narrow_vs_wide": max_rel_diff(
            np.asarray(fields["lfp_narrow"].lfp_proxy),
            np.asarray(fields["lfp_wide"].lfp_proxy),
        ),
        "shallow_vs_deep_probe": max_rel_diff(shallow.Y, deep.Y),
        "lfp_vs_csd": max_rel_diff(
            np.asarray(lfp_ref.lfp_proxy),
            np.asarray(lfp_ref.csd_proxy),
        ),
        "eeg_sup_vs_deep": max_rel_diff(Y_eeg_sup, Y_eeg_deep),
    }
    negative_max = float(np.max(np.abs(Y_meta_a - Y_meta_b)))
    compile_max = float(np.max(np.abs(Y_compiled - Y_lfp_ref)))

    level_a = array_sha256(Q) == q_hash and b2["q_hash_invariant"]
    level_b = (
        distinct["narrow_vs_wide"] > Y_DISTINCT_RTOL
        and distinct["lfp_vs_csd"] > Y_DISTINCT_RTOL
        and distinct["shallow_vs_deep_probe"] > Y_DISTINCT_RTOL
        and negative_max <= Y_ATOL
        and compile_max <= Y_ATOL
    )
    centroid = {
        "Q": spectral_centroid(psd_q, freqs_np),
        "LFP": spectral_centroid(psd_lfp, freqs_np),
        "CSD": spectral_centroid(psd_csd, freqs_np),
        "EEG_sup": spectral_centroid(psd_eeg_sup, freqs_np),
        "EEG_deep": spectral_centroid(psd_eeg_deep, freqs_np),
    }
    level_c = abs(centroid["LFP"] - centroid["CSD"]) > 0.5

    provenance = {
        name: json_safe(fo.diagnostics.get("observation"))
        for name, fo in fields.items()
    }
    provenance["lfp_contact_shallow"] = json_safe(shallow.probe_report)
    provenance["lfp_contact_deep"] = json_safe(deep.probe_report)
    provenance["eeg_superficial"] = json_safe(eeg_sup.report())
    provenance["eeg_deep"] = json_safe(eeg_deep.report())
    provenance["meg_relative"] = json_safe(meg_ro.report())

    operator_status = {
        "lfp_ref": "relative_proxy",
        "lfp_contact_shallow": "relative_proxy",
        "lfp_contact_deep": "relative_proxy",
        "csd_from_lfp_ref": "relative_proxy",
        "eeg_superficial": "analysis_only",
        "eeg_deep": "analysis_only",
        "meg_relative": "analysis_only",
    }

    return {
        "fields": fields,
        "distinct": distinct,
        "r90": r90,
        "spectral_centroid_hz": centroid,
        "levels": {"A": bool(level_a), "B": bool(level_b), "C": bool(level_c)},
        "b2_invariants": b2,
        "negative_control_max_abs": negative_max,
        "compilation_identity_max_abs": compile_max,
        "provenance": provenance,
        "operator_status": operator_status,
        "observation_arrays": {
            "Y_eeg_sup": Y_eeg_sup,
            "Y_eeg_deep": Y_eeg_deep,
            "Y_meg": Y_meg,
            "Y_shallow": shallow.Y,
            "Y_deep": deep.Y,
            "freqs": freqs_np,
            "psd_q": psd_q,
            "psd_lfp": psd_lfp,
            "psd_csd": psd_csd,
        },
    }


def write_b3_bundle(
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    """Freeze Experiment A receipt (B3). No manuscript figures."""
    root = bundle_root or BUNDLE_ROOT
    repo = REPO_ROOT
    spec = load_protocol_spec()
    spec_path = root / "b0_protocol_spec.json"
    if not spec_path.is_file():
        spec_path = PROTOCOL_SPEC_PATH
    head = _git_head(repo)

    dataset = freeze_canonical_dataset(spec=spec, package_head=head)

    write_canonical_npz(dataset, root / "canonical_source.npz")
    write_b1_receipt(dataset, root / "b1_canonical_receipt.json")

    suite = run_observation_suite(dataset, spec)
    cause_hashes = dict(dataset.cause_hashes)

    metrics = {
        "protocol": PROTOCOL_ID,
        "package_head": head,
        "cause_hashes": cause_hashes,
        "canonical_q_hash": cause_hashes["Q"],
        "n_neurons": int(dataset.Q.shape[1]),
        "n_steps": int(dataset.Q.shape[0]),
        "dt_ms": float(spec["neural_system"]["dt_ms"]),
        "r90": suite["r90"],
        "distinctness": suite["distinct"],
        "b2_invariants": suite["b2_invariants"],
        "spectral_centroid_hz": suite["spectral_centroid_hz"],
        "negative_control_max_abs": suite["negative_control_max_abs"],
        "compilation_identity_max_abs": suite["compilation_identity_max_abs"],
        "levels": suite["levels"],
        "operator_status": suite["operator_status"],
        "amplitude_semantics": "relative",
        "validation_status": "computational",
        "physical_claim": "proxy_readout",
        "q_hash_invariant": True,
    }
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True) + "\n")

    prov_path = root / "provenance.json"
    prov_path.write_text(json.dumps(suite["provenance"], indent=2, sort_keys=True) + "\n")

    tables = {
        "r90_laminar": suite["r90"],
        "distinctness": suite["distinct"],
        "spectral_centroid_hz": suite["spectral_centroid_hz"],
        "invariant_checks": suite["b2_invariants"],
        "levels": suite["levels"],
    }
    tables_path = root / "observation_tables.json"
    tables_path.write_text(json.dumps(json_safe(tables), indent=2, sort_keys=True) + "\n")

    obs = suite["observation_arrays"]
    np.savez_compressed(
        root / "observations.npz",
        freqs=obs["freqs"],
        psd_q=obs["psd_q"],
        psd_lfp=obs["psd_lfp"],
        psd_csd=obs["psd_csd"],
        Y_eeg_sup=obs["Y_eeg_sup"],
        Y_eeg_deep=obs["Y_eeg_deep"],
        Y_meg=obs["Y_meg"],
        Y_shallow=obs["Y_shallow"],
        Y_deep=obs["Y_deep"],
    )

    manifest = {
        "etude": PROTOCOL_ID,
        "package_head": head,
        "protocol_spec_sha256": _sha256_file(spec_path),
        "metrics_sha256": _sha256_file(metrics_path),
        "levels": suite["levels"],
        "cause_hashes": cause_hashes,
        "representation": "relative_proxy",
        "checkpoints": {"B0": "frozen", "B1": "frozen", "B2": "frozen", "B3": "frozen"},
        "manuscript_figures": False,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    receipt = {
        "schema": "jaxfne.experiment_a.b3_receipt.v1",
        "checkpoint": "B3",
        "status": "FROZEN",
        "protocol_id": PROTOCOL_ID,
        "package_head": head,
        "manifest": str(manifest_path.relative_to(repo)),
        "metrics": str(metrics_path.relative_to(repo)),
        "canonical_npz": "artifacts/etudes/experiment_a/canonical_source.npz",
        "completion_criterion": (
            "One frozen neural/source dataset passes through independently declared "
            "observation/probe operators with explicit semantic status and invariants."
        ),
    }
    receipt_path = root / "b3_experiment_a_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
