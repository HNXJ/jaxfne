"""Small deterministic gradient checks for the declared differentiable paths.

The public differentiability contract (docs/source_field_equations.md,
project source §10) declares exactly one end-to-end differentiable
composition: the ``source_scale`` rescale of an already-simulated run, a
linear post-processing map that never crosses the spike/reset boundary.
These tests verify autodiff against central finite differences on that
declared path with a fixed seed and a spike-raster stability precondition
(the FD comparison is only valid while no threshold crossing occurs within
the perturbation radius).
"""

import numpy as np
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne._model import _model_with_scalar_parameter


def _model():
    cfg = (
        jtfne.configuration()
        .network(n=8, cell_types={"E": 0.75, "PV": 0.25})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m", "source"])
    )
    return jtfne.construct(cfg)


_SIM = None


def _sim():
    global _SIM
    if _SIM is None:
        _SIM = jtfne.simulation(seed=3, duration_ms=10.0, dt_ms=0.1)
    return _SIM


def _raster_at(model, s):
    return np.asarray(
        _model_with_scalar_parameter(model, "source_scale", s).simulate(_sim()).spikes
    )


def test_source_scale_gradient_matches_finite_differences_end_to_end():
    model = _model()
    s0, h = 1.0, 1e-2

    rasters = [_raster_at(model, s) for s in (s0 - h, s0, s0 + h)]
    assert np.array_equal(rasters[0], rasters[1]), (
        "spike raster changed within FD radius; gradient check domain invalid"
    )
    assert np.array_equal(rasters[1], rasters[2])

    def scalar_loss(s):
        m2 = _model_with_scalar_parameter(model, "source_scale", s)
        return jnp.mean(jnp.abs(jnp.asarray(m2.simulate(_sim()).field.source_proxy)))

    autodiff = float(jax.grad(scalar_loss)(jnp.asarray(s0)))
    lp = float(scalar_loss(s0 + h))
    lm = float(scalar_loss(s0 - h))
    fd = (lp - lm) / (2.0 * h)
    assert abs(autodiff - fd) < 1e-3, f"autodiff {autodiff} vs FD {fd}"


def test_source_scale_linear_rescale_map_matches_finite_differences():
    """The tune-loop semantics: sources are a linear rescale of fixed arrays."""
    model = _model()
    baseline = model.simulate(_sim())
    base = jnp.abs(baseline.field.source_proxy)

    def loss(s):
        return jnp.mean(jnp.abs(s * base))

    autodiff = float(jax.grad(loss)(jnp.asarray(1.0)))
    h = 1e-3
    fd = float((loss(1.0 + h) - loss(1.0 - h)) / (2.0 * h))
    assert abs(autodiff - fd) < 5e-3, f"autodiff {autodiff} vs FD {fd}"
