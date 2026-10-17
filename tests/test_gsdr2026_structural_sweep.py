"""GSDR structural sweep tests: slice scaling and CLI sweep names (CPU, NumPy only).

Fail-first: on a tree where ``scale_conductances`` scales the whole vector for
``scope="ampa"``, the AMPA-only test fails (GABAa entries move); where the
parser accepts any ``--sweep`` string, the unknown-sweep test fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

N_AMPA = 1800
N_GABAA = 430


def _load_sweep():
    path = (Path(__file__).resolve().parents[1] / "artifacts" / "etudes"
            / "papers" / "gsdr2026" / "diag" / "structural_sweep.py")
    spec = importlib.util.spec_from_file_location("gsdr2026_structural_sweep",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep_mod = _load_sweep()


def _fixture():
    return np.concatenate([np.full(N_AMPA, 2.0, dtype=np.float32),
                           np.full(N_GABAA, 5.0, dtype=np.float32)])


def test_scale_ampa_only_leaves_gabaa_untouched():
    """scope='ampa' scales the first N_AMPA entries, GABAa stays exact."""
    g = _fixture()
    out = sweep_mod.scale_conductances(g, 0.5, "ampa", n_ampa=N_AMPA)
    assert np.all(out[:N_AMPA] == np.float32(1.0)), "AMPA slice must halve"
    assert np.all(out[N_AMPA:] == np.float32(5.0)), \
        "GABAa slice must be untouched, got %r" % (out[N_AMPA:5],)
    assert np.all(g[N_AMPA:] == np.float32(5.0)), "input must not mutate"


def test_scale_all_moves_both_slices():
    """scope='all' scales AMPA and GABAa together."""
    out = sweep_mod.scale_conductances(_fixture(), 2.0, "all", n_ampa=N_AMPA)
    assert np.all(out[:N_AMPA] == np.float32(4.0)), "AMPA slice must double"
    assert np.all(out[N_AMPA:] == np.float32(10.0)), \
        "GABAa slice must double, got %r" % (out[N_AMPA:5],)


def test_cli_refuses_unknown_sweep():
    """--sweep with an unknown name exits (argparse choices)."""
    with pytest.raises(SystemExit):
        sweep_mod.build_parser().parse_args(["--sweep", "bogus"])


def test_ampa_scales_flag_reaches_the_plan():
    """--ampa-scales replaces the default AMPA factors in the run plan."""
    args = sweep_mod.build_parser().parse_args(
        ["--sweep", "ampa", "--ampa-scales", "1e-3", "1e-4"])
    plan = sweep_mod.plan_rows(sweep=args.sweep, ampa_scales=args.ampa_scales)
    assert sorted({s for _, s, _ in plan}) == [1e-4, 1e-3]
    assert len(plan) == 2 * len(sweep_mod.SEEDS)
