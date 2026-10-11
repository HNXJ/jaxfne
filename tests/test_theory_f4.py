"""Gate F4: the OSI ceiling's lemma (L1) and the idealized unit of artifacts/etudes/theory_f4."""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne  # noqa: F401

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

import artifacts.etudes.theory_f4.f4_ceiling as F  # noqa: E402

C = F.C
CENTRAL = 1 * 4 + 1  # position (py, px) = (1, 1): patch rows and columns 3..9, nearest the grid centre


def test_l1_bound_random_and_single_orientation():
    rng = np.random.default_rng(0)
    curves = rng.random((1000, 8)) * (rng.random((1000, 8)) > 0.3)
    curves[np.arange(1000), rng.integers(0, 8, 1000)] += 0.01  # every curve has a nonzero orientation
    assert (curves.sum(1) > 0).all()
    for r in curves:
        osi, bound = F.lemma_check(r)
        assert osi <= bound + 1e-12, (r, osi, bound)
    for k in range(8):
        r = np.zeros(8)
        r[k] = 2.0
        osi, bound = F.lemma_check(r)
        assert abs(osi - 1.0) <= 1e-12 and abs(osi - bound) <= 1e-12, (k, osi, bound)


def test_uniform_kernel_has_no_selectivity_at_h0():
    idx = F.patch_index(7)
    kern = np.ones((1, 98))  # one feature, all taps 1
    r = np.array([F.drives(k, kern, idx)[..., CENTRAL].sum() for k in range(C.N_ORI)])
    assert r.sum() > 0, "fixture must have nonzero response"
    assert F.osi_curves(r) < 0.05, r


def test_event_images_deterministic_and_nonempty():
    for direction in (-1.0, 1.0):
        for ori in range(C.N_ORI):
            c0 = -direction * 13.0
            a = F.event_images(ori, direction, c0)
            b = F.event_images(ori, direction, c0)
            assert np.array_equal(a, b)
            assert a.sum() > 0, f"no events for orientation {ori}, direction {direction}"


def test_h0_tuning_is_flat_across_orientations_L2():
    idx = F.patch_index(7)
    kern = F.oriented_kernels(7)
    D = F.all_drives(kern, idx)  # (8 stimulus orientations, 2, N_C0, N_STEPS, units)
    r = D.sum(axis=(1, 2, 3))  # (8 stimulus orientations, units) at h = 0
    assert (r.min(0) > 0).all(), "fixture must have nonzero response for every unit"
    spread = r.max(0) - r.min(0)
    assert (spread <= 1e-9 * r.max(0)).all(), (spread.max(), r[:, np.argmax(spread)].tolist())


def half_max_tuning():
    """Tuning of every unit at h = 0.5 of its max drive: (8 stimulus orientations, units)."""
    idx = F.patch_index(7)
    kern = F.oriented_kernels(7)
    D = F.all_drives(kern, idx)  # (8 stimulus orientations, 2, N_C0, N_STEPS, units)
    maxd = D.max(axis=(0, 1, 2, 3))  # per unit, over the whole protocol
    return np.maximum(D - 0.5 * maxd, 0.0).sum(axis=(1, 2, 3))


def test_even_orientation_prefers_its_orientation_at_half_max_drive():
    tuning = half_max_tuning()
    assert tuning[:, CENTRAL * C.FS:(CENTRAL + 1) * C.FS].sum() > 0, "fixture must have nonzero response"
    for f in range(C.FS):
        k = f // 2
        if k % 2:  # odd orientations: see test_odd_orientation_argmax_is_adjacent_even
            continue
        r = tuning[:, CENTRAL * C.FS + f]
        assert int(np.argmax(r)) == k, f"feature {f} (orientation {k}) tuning {r.tolist()}"


def test_odd_orientation_argmax_is_adjacent_even_grid_anisotropy():
    tuning = half_max_tuning()
    checked = 0
    for f in range(C.FS):
        k = f // 2
        if k % 2 == 0:
            continue
        r = tuning[:, CENTRAL * C.FS + f]
        a = int(np.argmax(r))
        assert a % 2 == 0 and a in ((k - 1) % C.N_ORI, (k + 1) % C.N_ORI), (
            f"feature {f} (orientation {k}) tuning {r.tolist()}")
        checked += 1
    assert checked == 8, checked  # odd features 2, 3, 6, 7, 10, 11, 14, 15


def stimulus_inside_sequence(ori, direction, c0):
    """pcl_column.stimulus_step's inside mask at every step of a sequence (sampling output discarded)."""
    key = jax.random.PRNGKey(0)
    prev = jnp.zeros(C.G * C.G, bool)
    ins = []
    for i in range(F.N_STEPS):
        _, inside = C.stimulus_step(key, prev, jnp.float32(i * C.DT), ori, direction, jnp.float32(c0))
        ins.append(np.asarray(inside))
        prev = inside
    return np.stack(ins)


def test_event_images_match_stimulus_step_masks():
    for ori in (0, 3, 6):
        for direction in (-1.0, 1.0):
            c0 = float(np.float32(-direction * 13.0 + np.linspace(-1.0, 1.0, F.N_C0)[5]))
            ins = stimulus_inside_sequence(ori, direction, c0)
            prev = np.vstack([np.zeros((1, ins.shape[1]), bool), ins[:-1]])
            on, off = ins & ~prev, ~ins & prev
            assert on.sum() > 0 and off.sum() > 0, f"fixture must spike: ori {ori}, direction {direction}"
            E = F.event_images(ori, direction, c0)
            mine = (E > 0).reshape(F.N_STEPS, -1, 2)
            assert np.array_equal(mine, np.stack([on, off], -1)), (ori, direction)
