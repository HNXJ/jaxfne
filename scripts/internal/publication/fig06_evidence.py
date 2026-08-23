"""Frozen H/W/D evidence loader for Figure 6 (receipt-driven quantities only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[3]


def _load(rel: str) -> dict[str, Any]:
    return json.loads((_REPO / rel).read_text())


@dataclass(frozen=True)
class Fig06Evidence:
    h4_interp: dict[str, Any]
    h4_matrix: dict[str, Any]
    w0: dict[str, Any]
    w2: dict[str, Any]
    w3_stability: dict[str, Any]
    w3b_interp: dict[str, Any]
    d_closure: dict[str, Any]
    d3_interp: dict[str, Any]
    d1: dict[str, Any]


def load_fig06_evidence() -> Fig06Evidence:
    return Fig06Evidence(
        h4_interp=_load("artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json"),
        h4_matrix=_load("artifacts/protocol_h_rbd/h4_matrix/h4_matrix_receipt.json"),
        w0=_load("artifacts/protocol_w/w0_mathematical_contract.json"),
        w2=_load("artifacts/protocol_w/w2_expression/w2_expression_receipt.json"),
        w3_stability=_load("artifacts/protocol_w/w3_stability/w3_stability_receipt.json"),
        w3b_interp=_load("artifacts/protocol_w/w3b_parameter_domain/w3b_interpretation_receipt.json"),
        d_closure=_load("artifacts/protocol_d_biological_rbs/d_closure_interpretation_receipt.json"),
        d3_interp=_load("artifacts/protocol_d_biological_rbs/d3_interpretation_receipt.json"),
        d1=_load("artifacts/protocol_d_biological_rbs/d1_static_expression_receipt.json"),
    )


def h4_primary_mx(ev: Fig06Evidence) -> dict[str, float]:
    return dict(ev.h4_interp["primary_endpoint_results"])


def h3_memory_curves_beta_comparison(ev: Fig06Evidence) -> dict[str, Any]:
    """M_H vs M_X from frozen H4 h3 config (short ring); beta_H=0 vs 0.5."""
    from jaxfne.emitters import EdgeList, IzhikevichParams
    from jaxfne.h3_decodability import H3ProtocolConfig, run_h3_decodability_study
    import jax.numpy as jnp

    h3cfg = ev.h4_matrix["config"]["h3"]
    n = int(ev.h4_matrix["config"]["n_short"])
    delay = int(ev.h4_matrix["config"]["uniform_delay_steps"])
    jdtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.full((n,), -65.0, dtype=jdtype),
        u0=jnp.zeros((n,), dtype=jdtype),
        a=jnp.full((n,), 0.02, dtype=jdtype),
        b=jnp.full((n,), 0.2, dtype=jdtype),
        c=jnp.full((n,), -65.0, dtype=jdtype),
        d=jnp.full((n,), 8.0, dtype=jdtype),
        drive=jnp.zeros((n,), dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    syn_tau = float(ev.h4_matrix["config"]["syn_tau_ms"])
    ds = jnp.full((n,), delay, dtype=jnp.int32)
    edges = EdgeList(
        pre=jnp.arange(n, dtype=jnp.int32),
        post=jnp.asarray([(i + 1) % n for i in range(n)], dtype=jnp.int32),
        weight=jnp.full((n,), float(ev.h4_matrix["cells"]["short_uniform"]["circuit"]["edge_weight"]), dtype=jdtype),
        receptor_index=jnp.zeros((n,), dtype=jnp.int32),
        tau_ms=jnp.full((n,), syn_tau, dtype=jdtype),
        delay_steps=ds,
    )
    lag_steps = tuple(int(x) for x in h3cfg["lag_steps"])
    train = tuple(int(x) for x in h3cfg["train_seeds"])
    test = tuple(int(x) for x in h3cfg["test_seeds"])
    n_steps = int(h3cfg.get("n_steps", ev.h4_matrix["config"]["n_steps"]))
    out: dict[str, Any] = {}
    for beta in (0.0, float(h3cfg["beta_h"])):
        cfg = H3ProtocolConfig(
            delta_h=float(h3cfg["delta_h"]),
            perturbation_step=int(h3cfg["perturbation_step"]),
            lag_steps=lag_steps,
            rbd_family=str(h3cfg["rbd_family"]),
            beta_h=beta,
            kappa_h=float(h3cfg["kappa_h"]),
            tau_h_ms=float(h3cfg["tau_h_ms"]),
            dt_ms=float(h3cfg["dt_ms"]),
            ridge_lambda=float(h3cfg["ridge_lambda"]),
            n_shuffle=int(h3cfg["n_shuffle"]),
            train_seeds=train,
            test_seeds=test,
        )
        study = run_h3_decodability_study(params, edges, n_steps=n_steps, cfg=cfg)
        out[f"beta_h={beta}"] = {
            "M_H": {str(k): float(v) for k, v in study["curves"]["H"].items()},
            "M_X": {str(k): float(v) for k, v in study["curves"]["X"].items()},
        }
    return {"lag_steps": lag_steps, "curves": out, "source": "h4_matrix_receipt.json#config.h3"}


def w3b_counts(ev: Fig06Evidence) -> dict[str, int]:
    c = ev.w3b_interp["counts"]
    return {"N_S": int(c["N_S"]), "N_X": int(c["N_X"]), "N_D": int(c["D"]), "N_U": int(c["U"])}


def d3_classification(ev: Fig06Evidence) -> dict[str, Any]:
    return ev.d3_interp["questions"]["Q2_adaptation"]["counts"]
