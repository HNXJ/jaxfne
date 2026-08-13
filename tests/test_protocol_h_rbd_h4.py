"""Protocol H4 — factorial matrix machinery and receipt structure."""

from __future__ import annotations

import numpy as np
import pytest

from jaxfne.h4_matrix import (
    CELL_KEYS,
    H4ProtocolConfig,
    build_cell_circuit,
    build_ring_params_edges,
    detect_curve_peaks,
    factorial_design_matrix,
    fit_factorial_mx,
    loop_peak_alignment_table,
    loop_return_time_ms,
    run_h4_matrix,
)
from jaxfne.h3_decodability import H3ProtocolConfig


def _tiny_h4_cfg() -> H4ProtocolConfig:
    return H4ProtocolConfig(
        n_short=3,
        n_long=6,
        n_steps=50,
        h3=H3ProtocolConfig(
            delta_h=0.2,
            beta_h=0.5,
            lag_steps=(2, 5, 10),
            train_seeds=(1, 2),
            test_seeds=(10, 11),
            n_shuffle=3,
        ),
    )


def test_factorial_design_is_full_rank():
    X, keys = factorial_design_matrix()
    assert keys == list(CELL_KEYS)
    assert np.linalg.matrix_rank(X) == 4


def test_loop_return_time_uniform_ring():
    _, edges = build_ring_params_edges(3, delay_steps=4)
    assert loop_return_time_ms(edges, dt_ms=1.0) == 12.0


def test_heterogeneous_delay_pattern_alternates():
    cfg = H4ProtocolConfig()
    _, _, receipt = build_cell_circuit("short", "heterogeneous", cfg=cfg)
    assert receipt["delay_steps"] == [2, 8, 2]


def test_long_ring_larger_t_loop_than_short_uniform():
    cfg = H4ProtocolConfig()
    _, _, short = build_cell_circuit("short", "uniform", cfg=cfg)
    _, _, long = build_cell_circuit("long", "uniform", cfg=cfg)
    assert long["t_loop_ms"] > short["t_loop_ms"]


def test_fit_factorial_exact_on_synthetic_cells():
    cell_mx = {
        "short_uniform": 1.0,
        "short_heterogeneous": 2.0,
        "long_uniform": 3.0,
        "long_heterogeneous": 6.0,
    }
    fit = fit_factorial_mx(cell_mx)
    assert fit["mu"] == pytest.approx(1.0)
    assert fit["alpha_length"] == pytest.approx(2.0)
    assert fit["alpha_heterogeneity"] == pytest.approx(1.0)
    assert fit["alpha_interaction"] == pytest.approx(2.0)


def test_detect_curve_peaks_finds_local_maximum():
    curve = {1: 0.1, 2: 0.5, 3: 0.2, 4: 0.6, 5: 0.1}
    shuf = {k: 0.1 for k in curve}
    peaks = detect_curve_peaks(curve, shuf, min_excess=0.05)
    assert any(p["lag_steps"] == 4.0 for p in peaks)


def test_loop_peak_alignment_diagnostic_rows():
    peaks = [{"lag_steps": 10.0, "excess_mx": 0.1}]
    rows = loop_peak_alignment_table(peaks, t_loop_ms=12.0, dt_ms=1.0, max_harmonic=2)
    assert len(rows) == 2
    assert rows[0]["predicted_loop_ms"] == 12.0


def test_run_h4_matrix_smoke():
    out = run_h4_matrix(_tiny_h4_cfg(), rng_seed=0)
    assert set(out["cells"].keys()) == {
        "short_uniform",
        "short_heterogeneous",
        "long_uniform",
        "long_heterogeneous",
    }
    assert "M_X" in out["cells"]["short_uniform"]["summary"]
    assert "M_H" in out["cells"]["short_uniform"]["summary"]
    assert "D_H" in out["cells"]["short_uniform"]["summary"]
    assert "alpha_length" in out["factorial"]
    assert out["interpretation_scope"]
