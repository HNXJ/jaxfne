"""Hybrid kernel LFP (jaxfne.fields.kernel_lfp): cable currents, line sources, kernels, convolution."""

import numpy as np
import pytest

from jaxfne.fields.kernel_lfp import (
    cable_cell,
    kernel_lfp,
    line_source_matrix,
    passive_membrane_currents,
    profile_cell,
    synapse_kernel,
    synapse_weights_in_depth_range,
)

PASSIVE = dict(cm_uF_cm2=1.0, rm_ohm_cm2=20000.0, ra_ohm_cm=150.0)
GEOM = dict(radii_um=np.array([20.0, 60.0, 120.0]), sigma_S_m=0.3, min_distance_um=1.0)


def _pyramid():
    # soma 900-880 um, basal 900-1000, apical trunk 880-300, tuft 300-50 (depths below the pia)
    return cable_cell([
        dict(z0=900.0, z1=880.0, diam=20.0, n=1, parent=-1),
        dict(z0=900.0, z1=1000.0, diam=4.0, n=4, parent=0),
        dict(z0=880.0, z1=300.0, diam=3.0, n=12, parent=0),
        dict(z0=300.0, z1=50.0, diam=1.5, n=5, parent=2),
    ])


def test_membrane_currents_sum_to_zero_over_the_cell():
    cell = _pyramid()
    w = synapse_weights_in_depth_range(cell, 50.0, 300.0)
    v, i_m = passive_membrane_currents(cell, w, tau_syn_ms=2.0, dt_ms=0.1, n_steps=300, **PASSIVE)
    assert np.abs(i_m).max() > 1e-4, "fixture must produce axial current"
    np.testing.assert_allclose(i_m.sum(1), 0.0, atol=1e-12 * np.abs(i_m).max() * cell.n_comp)


def test_cable_matches_fine_euler_integration():
    cell = cable_cell([dict(z0=500.0, z1=480.0, diam=15.0, n=1, parent=-1),
                       dict(z0=480.0, z1=200.0, diam=2.0, n=3, parent=0)])
    w = synapse_weights_in_depth_range(cell, 200.0, 300.0)
    tau, dt, steps = 3.0, 0.2, 100
    v, _ = passive_membrane_currents(cell, w, tau_syn_ms=tau, dt_ms=dt, n_steps=steps, **PASSIVE)
    # independent forward Euler on the same equations, from the documented units
    a = cell.area_cm2
    c = PASSIVE["cm_uF_cm2"] * a * 1e3
    gl = a / PASSIVE["rm_ohm_cm2"] * 1e6
    rh = PASSIVE["ra_ohm_cm"] * (cell.length * 1e-4 / 2) / (np.pi * (cell.diam * 1e-4 / 2) ** 2)
    sub = 2000
    h = dt / sub
    x = np.zeros(cell.n_comp)
    ref = [x.copy()]
    for k in range(steps - 1):
        for j in range(sub):
            t = (k * sub + j) * h
            ax = np.zeros_like(x)
            for i, p in enumerate(cell.parent):
                if p >= 0:
                    g = 1e6 / (rh[i] + rh[p])
                    ax[i] += g * (x[p] - x[i])
                    ax[p] += g * (x[i] - x[p])
            x = x + h * (-gl * x + ax + w * np.exp(-t / tau)) / c
        ref.append(x.copy())
    ref = np.array(ref)
    assert ref.max() > 1e-3, "fixture must depolarize"
    np.testing.assert_allclose(v, ref, rtol=2e-3, atol=1e-3 * ref.max())


def test_line_source_matches_point_source_quadrature():
    cell = _pyramid()
    zc = np.linspace(0.0, 1100.0, 12)
    mat = line_source_matrix(cell, zc, **GEOM)
    lo, hi = np.minimum(cell.z0, cell.z1), np.maximum(cell.z0, cell.z1)
    ref = np.zeros_like(mat)
    for r in GEOM["radii_um"]:
        for k in range(cell.n_comp):
            zs = lo[k] + (np.arange(4000) + 0.5) * (hi[k] - lo[k]) / 4000
            d = np.sqrt(r ** 2 + (zc[:, None] - zs[None, :]) ** 2)
            ref[:, k] += (1.0 / d).mean(1) / (4 * np.pi * GEOM["sigma_S_m"])
    ref /= len(GEOM["radii_um"])
    np.testing.assert_allclose(mat, ref, rtol=1e-5)


def test_apical_excitation_gives_superficial_sink_and_deep_source():
    cell = _pyramid()
    w = synapse_weights_in_depth_range(cell, 50.0, 300.0)
    zc = np.array([150.0, 950.0])
    k = synapse_kernel(cell, w, zc, tau_syn_ms=2.0, dt_ms=0.1, n_steps=100, **PASSIVE, **GEOM)
    peak = np.abs(k).max(1)
    assert peak.min() > 0, "fixture must produce a potential at both contacts"
    t = np.abs(k[0]).argmax()
    assert k[0, t] < 0 < k[1, t]


def test_parameters_change_the_kernel():
    cell = _pyramid()
    w = synapse_weights_in_depth_range(cell, 50.0, 300.0)
    zc = np.array([150.0, 500.0, 950.0])
    base = dict(tau_syn_ms=2.0, dt_ms=0.1, n_steps=200, **PASSIVE, **GEOM)
    k0 = synapse_kernel(cell, w, zc, **base)
    for key, val in [("tau_syn_ms", 5.0), ("cm_uF_cm2", 2.0), ("rm_ohm_cm2", 5000.0), ("ra_ohm_cm", 300.0),
                     ("min_distance_um", 30.0), ("radii_um", np.array([200.0]))]:
        k1 = synapse_kernel(cell, w, zc, **{**base, key: val})
        assert not np.allclose(k1, k0, rtol=1e-3), f"{key} not consumed"
    k_sig = synapse_kernel(cell, w, zc, **{**base, "sigma_S_m": 0.6})
    np.testing.assert_allclose(k_sig, k0 / 2, rtol=1e-12, atol=1e-15)
    k_dt = synapse_kernel(cell, w, zc, **{**base, "dt_ms": 0.05, "n_steps": 400})
    np.testing.assert_allclose(k_dt[:, ::2], k0, rtol=1e-8, atol=1e-12 * np.abs(k0).max())


@pytest.mark.parametrize("key,val", [("tau_syn_ms", 0.0), ("dt_ms", -1.0), ("cm_uF_cm2", np.nan),
                                     ("rm_ohm_cm2", np.inf), ("ra_ohm_cm", 0.0), ("sigma_S_m", 0.0),
                                     ("min_distance_um", -2.0), ("n_steps", 0)])
def test_invalid_parameters_are_refused(key, val):
    cell = _pyramid()
    w = synapse_weights_in_depth_range(cell, 50.0, 300.0)
    base = dict(tau_syn_ms=2.0, dt_ms=0.1, n_steps=10, **PASSIVE, **GEOM)
    with pytest.raises(ValueError):
        synapse_kernel(cell, w, np.array([100.0]), **{**base, key: val})


def test_cell_and_synapse_inputs_are_refused():
    with pytest.raises(ValueError):
        cable_cell([dict(z0=0.0, z1=10.0, diam=1.0, n=1, parent=0)])
    with pytest.raises(ValueError):
        cable_cell([dict(z0=0.0, z1=0.0, diam=1.0, n=1, parent=-1)])
    with pytest.raises(ValueError):
        cable_cell([dict(z0=0.0, z1=10.0, diam=0.0, n=1, parent=-1)])
    with pytest.raises(ValueError):
        cable_cell([dict(z0=0.0, z1=10.0, diam=np.inf, n=1, parent=-1)])
    with pytest.raises(ValueError):
        profile_cell(0.0, 10.0, -20.0, [])
    cell = _pyramid()
    with pytest.raises(ValueError):
        line_source_matrix(cell, np.array([100.0]), **{**GEOM, "radii_um": np.array([-50.0])})
    with pytest.raises(ValueError):
        line_source_matrix(cell, np.array([100.0, np.nan]), **GEOM)
    with pytest.raises(ValueError):
        synapse_weights_in_depth_range(cell, 1100.0, 1200.0)
    with pytest.raises(ValueError):
        passive_membrane_currents(cell, np.full(cell.n_comp, 0.5), tau_syn_ms=1.0, dt_ms=0.1, n_steps=5, **PASSIVE)


def test_sections_attach_to_the_last_compartment_of_their_parent():
    cell = _pyramid()
    # soma 0; basal 1-4; trunk 5-16; tuft 17-21
    assert cell.parent == (-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    np.testing.assert_allclose(cell.z1[16], 300.0, rtol=1e-12)
    np.testing.assert_allclose(cell.z0[17], 300.0, rtol=1e-12)


def test_uniform_area_factor_equals_scaled_membrane_constants():
    cell = _pyramid()
    f = 2.5
    scaled = type(cell)(cell.z0, cell.z1, cell.diam, cell.parent, np.full(cell.n_comp, f))
    w = synapse_weights_in_depth_range(cell, 50.0, 300.0)
    kw = dict(tau_syn_ms=2.0, dt_ms=0.1, n_steps=200)
    v0, _ = passive_membrane_currents(cell, w, cm_uF_cm2=f, rm_ohm_cm2=20000.0 / f, ra_ohm_cm=150.0, **kw)
    v1, _ = passive_membrane_currents(scaled, w, **PASSIVE, **kw)
    assert np.abs(v0).max() > 1e-4, "fixture must depolarize"
    np.testing.assert_allclose(v1, v0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(scaled.area_cm2, f * cell.area_cm2, rtol=1e-12)
    sec = cable_cell([dict(z0=500.0, z1=480.0, diam=10.0, n=1, parent=-1),
                      dict(z0=480.0, z1=380.0, diam=2.0, n=2, parent=0, area_factor=3.0)])
    np.testing.assert_allclose(sec.area_factor, [1.0, 3.0, 3.0], rtol=0)


def test_profile_cell_restores_bin_areas_and_chains_branches():
    up = dict(edges_um=[490.0, 400.0, 300.0], area_um2=[3000.0, 1500.0], diam_um=[4.0, 2.5])
    down = dict(edges_um=[510.0, 560.0], area_um2=[2000.0], diam_um=[3.0])
    cell = profile_cell(500.0, 15.0, 20.0, [up, down])
    assert cell.parent == (-1, 0, 1, 0)
    np.testing.assert_allclose(cell.area_cm2 * 1e8, [np.pi * 15 * 20, 3000.0, 1500.0, 2000.0], rtol=1e-12)
    np.testing.assert_allclose(cell.diam, [15.0, 4.0, 2.5, 3.0], rtol=1e-12)
    np.testing.assert_allclose(cell.mid, [500.0, 445.0, 350.0, 535.0], rtol=1e-12)
    bad = [dict(up, edges_um=[480.0, 400.0, 300.0]), dict(up, edges_um=[490.0, 400.0, 420.0]),
           dict(up, area_um2=[3000.0, 0.0]), dict(up, diam_um=[4.0])]
    for b in bad:
        with pytest.raises(ValueError):
            profile_cell(500.0, 15.0, 20.0, [b])


def test_synapse_weights_follow_area_in_range():
    cell = _pyramid()
    w = synapse_weights_in_depth_range(cell, 250.0, 350.0)
    inside = (cell.mid >= 250.0) & (cell.mid <= 350.0)
    assert inside.sum() == 2 and np.ptp(cell.area_cm2[inside]) > 0, "fixture must span two diameters"
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
    assert np.all(w[~inside] == 0)
    np.testing.assert_allclose(w[inside], cell.area_cm2[inside] / cell.area_cm2[inside].sum(), rtol=1e-12)


def test_kernel_lfp_is_the_summed_causal_convolution():
    rng = np.random.default_rng(3)
    t, c, n, k = 200, 3, 4, 25
    spikes = rng.poisson(0.4, size=(t, c)).astype(np.float64) * np.array([1.0, -2.0, 0.5])
    kernels = rng.normal(size=(c, n, k))
    got = np.asarray(kernel_lfp(spikes, kernels))
    ref = np.zeros((t, n))
    for ci in range(c):
        for ni in range(n):
            ref[:, ni] += np.convolve(spikes[:, ci], kernels[ci, ni])[:t]
    assert np.abs(ref).max() > 0
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5 * np.abs(ref).max())
