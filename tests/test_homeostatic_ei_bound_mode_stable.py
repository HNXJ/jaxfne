"""homeostatic_ei via the grammar with bound_mode="stable" stays bounded (C2A3).

Found 2026-11-07: the construct path built ``HomeostaticEIParams`` without
``x_min``/``x_max``, so the dataclass defaults ``+-1e6``
(``jaxfne/emitters_homeostatic_ei.py:366-367``) applied and ``_soft_bound``
saturated ``x`` at exactly ``+-1e6`` (observed max|V| = 1000000.0, finite but
non-physiological). Fixed in ``jaxfne/_construct_core.py``: the canonical
defaults now carry ``x_min: -10.0``/``x_max: 10.0`` (the values
``make_minimal_ei_params`` documents at
``jaxfne/emitters_homeostatic_ei.py:442``), threaded into the params, with
explicit ``set_emitter(..., x_min=, x_max=)`` overrides read from the config.

The mode's bound promise (``jaxfne/emitters_homeostatic_ei.py:528-531``):
stable applies ``_soft_bound`` to ``x``, ``G`` and ``H`` every step, "a
bounded codomain that cannot be numerically outrun by a large Euler step".
"""
import numpy as np
import pytest

import jaxfne as jtfne


def _run(bound_mode, duration_ms=100.0, **emitter_kw):
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, duration_ms=duration_ms, dt_ms=0.5)
        .network(name="c2a3_hei_stable", n=16)
        .set_emitter("homeostatic_ei", bound_mode=bound_mode, **emitter_kw)
        .field()
        .probe(modes=["vm"])
    )
    model = jtfne.construct(cfg)
    signals = model.simulate(jtfne.Simulation(duration_ms=duration_ms, dt_ms=0.5, seed=0))
    return model, np.asarray(signals.V_m, dtype=float)


def test_stable_bound_mode_stays_within_canonical_x_range():
    """Default stable run: finite and within the canonical [x_min, x_max]."""
    model, v = _run("stable")
    x_min = float(model.params["emitter"].x_min)
    x_max = float(model.params["emitter"].x_max)
    assert (x_min, x_max) == (-10.0, 10.0)
    assert bool(np.all(np.isfinite(v))), "stable mode must stay finite"
    assert float(np.abs(v).max()) <= x_max, f"max|V|={float(np.abs(v).max())!r} left [{x_min}, {x_max}]"


def test_stable_bound_mode_honours_explicit_x_bounds_and_stays_interior():
    """Explicit x_min/x_max reach the kernel and the trace stays off the rails."""
    model, v = _run("stable", x_min=-5.0, x_max=5.0)
    assert float(model.params["emitter"].x_min) == -5.0
    assert float(model.params["emitter"].x_max) == 5.0
    assert bool(np.all(np.isfinite(v))), "stable mode must stay finite"
    assert float(np.abs(v).max()) <= 5.0
    assert float(np.abs(v).max()) < 0.999 * 5.0, f"max|V|={float(np.abs(v).max())!r} sits at the rail"


@pytest.mark.parametrize("kw", [{"x_min": -5.0}, {"x_max": 5.0}])
def test_x_bounds_refused_under_minimal(kw):
    """x is unbounded under "minimal", so a declared x bound would be dropped: refused."""
    with pytest.raises(ValueError, match="read only under bound_mode='stable'"):
        _run("minimal", duration_ms=1.0, **kw)


def test_x_min_must_be_below_x_max():
    with pytest.raises(ValueError, match="x_min < x_max"):
        _run("stable", duration_ms=1.0, x_min=5.0, x_max=5.0)
