"""Column tiling and spherical-head MEG/EEG forward models (jaxfne.fields.head)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import eval_legendre

from jaxfne.fields.head import (
    SphereForward,
    column_dipole,
    eeg_sphere_leadfield,
    folded_sphere_surface,
    meg_sphere_leadfield,
    signals_dipole,
    sphere_sensor_cap,
    tile_column_dipoles,
)

MU0 = 4e-7 * np.pi  # literal, independent of the implementation


def _dipoles(m=7, radius=0.07, seed=3):
    """Dipoles at radii in [0.3, 1.0] x radius; a fixed seed per call keeps tests order-independent."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(m, 3))
    pos = radius * rng.uniform(0.3, 1.0, size=(m, 1)) * pos / np.linalg.norm(pos, axis=1, keepdims=True)
    ori = rng.normal(size=(m, 3))
    return pos, ori / np.linalg.norm(ori, axis=1, keepdims=True)


def test_column_dipole_is_first_moment_about_mean_depth():
    q = np.array([[1.0, 1.0], [2.0, -1.0]])
    pos = np.array([[0, 0, 0.2], [0, 0, 0.8]])
    # z - mean = [-0.3, 0.3]: common mode cancels, difference survives.
    assert_allclose(np.asarray(column_dipole(q, pos)), [0.0, -0.9], atol=1e-6)
    with pytest.raises(ValueError):
        column_dipole(q, pos[:1])
    with pytest.raises(ValueError):
        column_dipole(q, pos, axis=3)


def test_meg_radial_component_equals_primary_current_field():
    # B·r̂ outside a sphere comes from the primary current alone (volume currents add nothing).
    pos, ori = _dipoles()
    sens, nrm = sphere_sensor_cap(40, 0.12)
    lead = np.asarray(meg_sphere_leadfield(sens, nrm, pos, ori))
    s, n = np.asarray(sens, float), np.asarray(nrm, float)
    ref = np.empty_like(lead)
    for m in range(pos.shape[0]):
        d = s - pos[m]
        b0 = MU0 / (4 * np.pi) * np.cross(ori[m], d) / np.linalg.norm(d, axis=1, keepdims=True) ** 3
        ref[:, m] = np.sum(b0 * n, axis=1)
    assert np.abs(ref).max() > 0, "fixture must produce a field"
    assert_allclose(lead, ref, rtol=2e-4, atol=1e-6 * np.abs(ref).max())


def _primary_b(p, r0, q):
    d = p - r0
    return MU0 / (4 * np.pi) * np.cross(q, d) / np.linalg.norm(d) ** 3


def _scalar_potential(p, r0, q):
    """U(r) = -∫_0^∞ ∂_t U(r + t r̂) dt = (1/μ0) ∫_0^∞ B_r(r + t r̂) dt, since B = -μ0 ∇U outside a spherical
    conductor and U vanishes at infinity; B_r there is the primary field alone."""
    e = p / np.linalg.norm(p)
    val, _ = quad(lambda t: e @ _primary_b(p + t * e, r0, q), 0.0, np.inf, epsabs=0.0, epsrel=1e-11, limit=200)
    return val / MU0


def test_meg_full_vector_matches_scalar_potential_quadrature():
    # Independent of the closed form: B = -μ0 ∇U with U from quadrature of the primary radial field.
    pos, ori = _dipoles(3, 0.07)
    sens, _ = sphere_sensor_cap(12, 0.12)
    s = np.asarray(sens, float)
    tilt = np.random.default_rng(5).normal(size=s.shape)
    tilt /= np.linalg.norm(tilt, axis=1, keepdims=True)              # arbitrary, mostly non-radial sensor axes
    lead = np.asarray(meg_sphere_leadfield(s, tilt, pos, ori))
    h = 1e-5
    for m in range(pos.shape[0]):
        ref = np.empty(len(s))
        for k, p in enumerate(s):
            grad = [(_scalar_potential(p + h * ax, pos[m], ori[m]) - _scalar_potential(p - h * ax, pos[m], ori[m])) / (2 * h)
                    for ax in np.eye(3)]
            ref[k] = -MU0 * np.dot(grad, tilt[k])
        radial_share = np.abs(np.sum(tilt * s / np.linalg.norm(s, axis=1, keepdims=True), axis=1))
        assert radial_share.min() < 0.5, "fixture must include non-radial sensor axes"
        assert_allclose(lead[:, m], ref, rtol=1e-3, atol=1e-4 * np.abs(ref).max())


def test_meg_radial_dipole_is_silent():
    pos, _ = _dipoles()
    sens, nrm = sphere_sensor_cap(30, 0.12)
    radial = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    tang = np.cross(radial, [0.0, 0.0, 1.0])
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    silent = np.abs(np.asarray(meg_sphere_leadfield(sens, nrm, pos, radial))).max()
    loud = np.abs(np.asarray(meg_sphere_leadfield(sens, nrm, pos, tang))).max()
    assert loud > 1e-9 and silent < 1e-6 * loud


def _eeg_series(e, r0, q, sigma, n_max=400):
    """Neumann-series reference: q·∇_{r0} of Σ (2n+1)/n (b/R)^n P_n(cos γ) / (4πσR), by central differences."""
    big_r = np.linalg.norm(e[0])

    def phi(p):
        b = np.linalg.norm(p)
        cos_g = (e @ p) / (big_r * b)
        n = np.arange(1, n_max)
        terms = ((2 * n + 1) / n)[None, :] * (b / big_r) ** n[None, :] * eval_legendre(n[None, :], cos_g[:, None])
        return terms.sum(1) / (4 * np.pi * sigma * big_r)

    h = 1e-7
    return sum(q[k] * (phi(r0 + h * np.eye(3)[k]) - phi(r0 - h * np.eye(3)[k])) / (2 * h) for k in range(3))


def test_eeg_closed_form_matches_neumann_series():
    elec, _ = sphere_sensor_cap(25, 0.09, 150.0)
    pos, ori = _dipoles(4, 0.07)
    lead = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.33, average_reference=False))
    e = np.asarray(elec, float)
    for m in range(4):
        ref = _eeg_series(e, pos[m], ori[m], 0.33)
        assert np.abs(ref).max() > 0
        assert_allclose(lead[:, m], ref, rtol=1e-3, atol=1e-4 * np.abs(ref).max())


def test_eeg_matches_mne_single_shell():
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    elec, _ = sphere_sensor_cap(30, 0.09, 150.0)
    e = np.asarray(elec, float)
    pos, ori = _dipoles(5, 0.07)
    names = [f"E{i}" for i in range(len(e))]
    info = mne.create_info(names, 1000.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(names, e)), coord_frame="head"))
    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.09, relative_radii=(0.99, 1.0), sigmas=(0.33, 0.33))  # two equal shells = homogeneous
    dip = mne.Dipole(times=np.arange(5) * 1e-3, pos=pos, amplitude=np.ones(5), ori=ori, gof=np.ones(5))
    fwd, _ = mne.make_forward_dipole(dip, sphere, info)
    ref = fwd["sol"]["data"]
    ref = ref - ref.mean(0, keepdims=True)
    ours = np.asarray(eeg_sphere_leadfield(e, pos, ori, conductivity_s_per_m=0.33))
    assert_allclose(ours, ref, rtol=1e-3, atol=1e-4 * np.abs(ref).max())


def test_eeg_conductivity_scales_and_reference_is_consumed():
    elec, _ = sphere_sensor_cap(20, 0.09, 150.0)
    pos, ori = _dipoles(3, 0.06)
    a = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.33))
    b = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.66))
    assert_allclose(a, 2 * b, rtol=1e-5)
    raw = np.asarray(eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.33, average_reference=False))
    assert np.abs(raw.mean(0)).max() > 1e-3 * np.abs(raw).max()
    assert_allclose(a.mean(0), 0.0, atol=1e-6 * np.abs(a).max())


def test_forward_refusals():
    sens, nrm = sphere_sensor_cap(10, 0.12)
    pos, ori = _dipoles(3)
    with pytest.raises(ValueError):
        meg_sphere_leadfield(sens * 0.15, nrm, pos, ori)         # sensors at 18 mm, dipoles at >= 21 mm
    elec, _ = sphere_sensor_cap(10, 0.09)
    with pytest.raises(ValueError):
        eeg_sphere_leadfield(elec, pos, ori, conductivity_s_per_m=0.0)
    with pytest.raises(ValueError):
        eeg_sphere_leadfield(elec * np.linspace(1, 1.1, 10)[:, None], pos, ori, conductivity_s_per_m=0.33)
    with pytest.raises(ValueError):
        eeg_sphere_leadfield(elec * 0.2, pos, ori, conductivity_s_per_m=0.33)   # electrodes at 18 mm
    with pytest.raises(ValueError):
        SphereForward("ecog", jnp.zeros((2, 3)), "x", "y")


def test_folded_surface_normals_and_parameters():
    pos, nrm = folded_sphere_surface(400)
    pos, nrm = np.asarray(pos), np.asarray(nrm)
    assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, rtol=1e-5)
    assert (np.sum(nrm * pos, axis=1) > 0).all(), "normals point outward"
    cos_r = np.abs(np.sum(nrm * pos / np.linalg.norm(pos, axis=1, keepdims=True), axis=1))
    assert cos_r.min() < 0.5 and cos_r.max() > 0.95, "folding must give tangential and radial columns"
    flat_pos, flat_n = folded_sphere_surface(400, fold_amplitude_m=0.0)
    assert_allclose(np.sum(np.asarray(flat_n) * np.asarray(flat_pos), 1), 0.068, rtol=1e-4)  # unfolded: radial
    for kw in (dict(radius_m=0.07), dict(fold_frequencies=(5, 7)), dict(max_polar_deg=60.0)):
        p2, n2 = folded_sphere_surface(400, **kw)
        assert np.abs(np.asarray(p2) - pos).max() > 1e-4, kw
    with pytest.raises(ValueError):
        folded_sphere_surface(0)
    with pytest.raises(ValueError):
        folded_sphere_surface(10, fold_amplitude_m=0.1)


def test_tiling_copies_realizations_with_shifts_and_in_phase_patch():
    t = np.arange(50, dtype=np.float32)
    real = np.stack([np.sin(0.3 * t), np.cos(0.17 * t), t / 50.0])
    mask = np.zeros(12, bool)
    mask[:4] = True
    dip, assign, shift = tile_column_dipoles(real, 12, jax.random.PRNGKey(0), in_phase=mask, in_phase_realization=1)
    dip, assign, shift = np.asarray(dip), np.asarray(assign), np.asarray(shift)
    assert dip.shape == (50, 12)
    for m in range(12):
        assert_allclose(dip[:, m], np.roll(real[assign[m]], shift[m]), rtol=1e-6)
    assert (assign[:4] == 1).all() and (shift[:4] == 0).all()
    other, _, _ = tile_column_dipoles(real, 12, jax.random.PRNGKey(1), in_phase=mask, in_phase_realization=1)
    assert np.abs(np.asarray(other)[:, 4:] - dip[:, 4:]).max() > 0, "the key must drive the copies"
    with pytest.raises(ValueError):
        tile_column_dipoles(real, 12, jax.random.PRNGKey(0), in_phase_realization=3)
    with pytest.raises(ValueError):
        tile_column_dipoles(real, 12, jax.random.PRNGKey(0), in_phase=mask[:5])
    with pytest.raises(ValueError):
        tile_column_dipoles(real, 12, jax.random.PRNGKey(0), in_phase_realization=1)  # no mask to consume it


def test_sphere_forward_applies_and_reports():
    sens, nrm = sphere_sensor_cap(16, 0.12)
    pos, ori = folded_sphere_surface(30)
    fwd = SphereForward("meg", meg_sphere_leadfield(sens, nrm, pos, ori), "sarvas_1987_sphere", "T/(A m)")
    p = jnp.asarray(np.random.default_rng(6).normal(size=(5, 30)) * 1e-9)
    assert_allclose(np.asarray(fwd.apply(p)), np.asarray(p @ fwd.W.T), rtol=1e-6)
    rep = fwd.report()
    assert rep["modality"] == "meg" and rep["physical_amplitude_calibrated"] is False
    with pytest.raises(ValueError):
        fwd.apply(p[:, :5])


def test_signals_dipole_bridges_a_simulation_to_the_head():
    import jaxfne as jtfne

    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=24, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "LFP"])
    )
    model = jtfne.construct(cfg)
    sig = model.simulate(jtfne.simulation(duration_ms=40.0, dt_ms=0.5, seed=1))
    q, pos = np.asarray(sig.sources, np.float64), np.asarray(model.params["positions"], np.float64)
    assert np.abs(q).max() > 0, "fixture must produce sources"
    z = pos[:, 2]
    p = np.asarray(signals_dipole(sig, model))
    assert p.shape == (q.shape[0],)
    assert_allclose(p, q @ (z - z.mean()), rtol=1e-5, atol=1e-6 * np.abs(q).max())
    p_x = np.asarray(signals_dipole(sig, model, axis=0))
    assert np.abs(p_x - p).max() > 0, "axis must be consumed"
    sens, nrm = sphere_sensor_cap(8, 0.12)
    cpos, cnrm = folded_sphere_surface(1)
    fwd = SphereForward("meg", meg_sphere_leadfield(sens, nrm, cpos, cnrm), "sarvas_1987_sphere", "T/(A m)")
    assert fwd.apply(jnp.asarray(p)[:, None] * 1e-9).shape == (q.shape[0], 8)
    no_src = model.simulate(jtfne.simulation(duration_ms=40.0, dt_ms=0.5, seed=1, record_sources=False))
    with pytest.raises(ValueError, match="record_sources"):
        signals_dipole(no_src, model)
    with pytest.raises(ValueError, match="positions"):
        signals_dipole(sig, object())
