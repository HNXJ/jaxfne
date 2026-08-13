"""Protocol H3 — localized RBS perturbation + nested linear decodability."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.h3_decodability import (
    H3ProtocolConfig,
    PRIMARY_DECODER_KINDS,
    build_trial_dataset,
    chance_accuracy,
    distributed_rbs_dispersion,
    run_h3_decodability_study,
    run_localized_rbs_trial,
    shuffle_baseline_curve,
    trial_drive_schedule,
)

CHANCE_MARGIN = 0.12
ABOVE_CHANCE = 0.08


def _params_edges_ring(*, delay_steps: int = 0) -> tuple[IzhikevichParams, EdgeList]:
    jdtype = jnp.float32
    n = 3
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
        labels=("E", "E", "E"),
        layer_labels=("L4", "L4", "L4"),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    ds = jnp.asarray([delay_steps, delay_steps, delay_steps], dtype=jnp.int32)
    edges = EdgeList(
        pre=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        post=jnp.asarray([1, 2, 0], dtype=jnp.int32),
        weight=jnp.asarray([6.0, 6.0, 6.0], dtype=jdtype),
        receptor_index=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        tau_ms=jnp.asarray([3.0, 3.0, 3.0], dtype=jdtype),
        delay_steps=ds,
    )
    return params, edges


def _cfg(
    *,
    delta_h: float = 0.2,
    beta_h: float = 0.5,
    rbd_family: str = "f1",
    train: tuple[int, ...] = (10, 11, 12),
    test: tuple[int, ...] = (20, 21, 22),
) -> H3ProtocolConfig:
    return H3ProtocolConfig(
        delta_h=delta_h,
        beta_h=beta_h,
        rbd_family=rbd_family,
        lag_steps=(2, 5, 10, 20, 35),
        train_seeds=train,
        test_seeds=test,
        n_shuffle=6,
    )


def test_disjoint_seeds_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        H3ProtocolConfig(train_seeds=(1, 2), test_seeds=(2, 3))


def test_primary_decoders_are_label_independent():
    assert PRIMARY_DECODER_KINDS == ("H", "X", "XH")


def test_delta_h_zero_near_chance():
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg(delta_h=0.0, beta_h=0.5)
    out = run_h3_decodability_study(params, edges, n_steps=60, cfg=cfg)
    chance = chance_accuracy(3)
    for kind in PRIMARY_DECODER_KINDS:
        assert max(out["curves"][kind].values()) <= chance + CHANCE_MARGIN


def test_f0_null_near_chance():
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg(delta_h=0.2, beta_h=0.5, rbd_family="f0")
    out = run_h3_decodability_study(params, edges, n_steps=60, cfg=cfg)
    chance = chance_accuracy(3)
    assert max(out["curves"]["H"].values()) <= chance + CHANCE_MARGIN
    assert max(out["curves"]["X"].values()) <= chance + CHANCE_MARGIN


def test_label_shuffle_baseline_near_chance():
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg()
    train = build_trial_dataset(params, edges, n_steps=60, seeds=cfg.train_seeds, cfg=cfg)
    test = build_trial_dataset(params, edges, n_steps=60, seeds=cfg.test_seeds, cfg=cfg)
    rng = np.random.default_rng(0)
    shuf = shuffle_baseline_curve(
        train, test, cfg=cfg, kind="H", n_classes=3, rng=rng
    )
    chance = chance_accuracy(3)
    assert max(shuf.values()) <= chance + CHANCE_MARGIN


def test_beta_h_zero_mh_above_mx_near_chance():
    """kappa_H=0, beta_H=0: H carries local tag; X should stay near chance."""
    params, edges = _params_edges_ring(delay_steps=4)
    cfg = _cfg(delta_h=0.25, beta_h=0.0)
    out = run_h3_decodability_study(params, edges, n_steps=70, cfg=cfg)
    chance = chance_accuracy(3)
    assert max(out["curves"]["H"].values()) >= chance + ABOVE_CHANCE
    assert max(out["curves"]["X"].values()) <= chance + CHANCE_MARGIN


def test_beta_h_positive_enables_mx_above_chance():
    params, edges = _params_edges_ring(delay_steps=4)
    cfg = _cfg(delta_h=0.25, beta_h=0.6)
    out = run_h3_decodability_study(params, edges, n_steps=70, cfg=cfg)
    chance = chance_accuracy(3)
    assert max(out["curves"]["X"].values()) >= chance + ABOVE_CHANCE


def test_shared_background_drive_per_seed_regression():
    """Per-k drive schedules would leak label into X (permanent regression)."""
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg(delta_h=0.0, beta_h=0.5)
    trials = build_trial_dataset(params, edges, n_steps=50, seeds=[42], cfg=cfg)
    ref = trials[0]["voltages"]
    for tr in trials[1:]:
        assert jnp.allclose(ref, tr["voltages"])


def test_per_k_drive_would_break_shared_background_null():
    """Document the confound class: different drives per k break trajectory identity."""
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg(delta_h=0.0, beta_h=0.5)
    drive_a = trial_drive_schedule(50, 3, seed=1)
    drive_b = trial_drive_schedule(50, 3, seed=2)
    a = run_localized_rbs_trial(
        params, edges, n_steps=50, perturbed_index=0, seed=10, cfg=cfg, drive_schedule=drive_a
    )
    b = run_localized_rbs_trial(
        params, edges, n_steps=50, perturbed_index=1, seed=11, cfg=cfg, drive_schedule=drive_b
    )
    assert not jnp.allclose(a["voltages"], b["voltages"])


def test_segmented_perturbation_continuation_regression():
    """Continuation must apply ``H``/``H_final`` at t0 (permanent regression)."""
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = H3ProtocolConfig(
        delta_h=0.2,
        perturbation_step=10,
        beta_h=0.5,
        lag_steps=(5, 15),
        train_seeds=(1,),
        test_seeds=(2,),
    )
    tr = run_localized_rbs_trial(
        params,
        edges,
        n_steps=40,
        perturbed_index=1,
        seed=99,
        cfg=cfg,
    )
    assert tr["voltages"].shape[0] == 40
    assert float(tr["state"]["H_trace"][10, 1]) > 1.1


def test_d_h_zero_without_perturbation():
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg(delta_h=0.0)
    tr = run_localized_rbs_trial(
        params, edges, n_steps=40, perturbed_index=1, seed=3, cfg=cfg
    )
    assert distributed_rbs_dispersion(tr, lag=5, perturbation_step=cfg.perturbation_step) == 0.0


def test_local_h_diagnostic_above_d_h_under_beta_h_zero():
    """Local tag in H_k with beta_H=0 should exceed off-diagonal D_H early on."""
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg(delta_h=0.25, beta_h=0.0)
    out = run_h3_decodability_study(params, edges, n_steps=60, cfg=cfg)
    lag = min(cfg.lag_steps)
    assert out["diagnostics"]["local_h_k"][lag] > out["diagnostics"]["D_H"][lag]


def test_independent_seeds_not_same_trajectory():
    params, edges = _params_edges_ring(delay_steps=0)
    cfg = _cfg()
    a = run_localized_rbs_trial(params, edges, n_steps=50, perturbed_index=0, seed=5, cfg=cfg)
    b = run_localized_rbs_trial(params, edges, n_steps=50, perturbed_index=0, seed=6, cfg=cfg)
    assert not jnp.allclose(a["voltages"], b["voltages"])
