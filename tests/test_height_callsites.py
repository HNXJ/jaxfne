"""P-066 call sites: the height must reach every projection of mm positions.

Each test compares the mm path with the same projection given fraction positions (z / height).
"""

import numpy as np

from jaxfne.experiment_a.canonical import CanonicalDataset
from jaxfne.experiment_a.observe import materialize_field
from jaxfne.fields import project_laminar_sources
from jaxfne.fields.proxy import project_sources_to_laminar_field

H = 8.0
SPEC = {"field_operators_F": [{"id": "lfp_ref", "params": {"n_contacts": 8, "width": 0.1}}]}


def _fixture(n=12, t=20, seed=1):
    rng = np.random.default_rng(seed)
    sources = rng.random((t, n))
    xy = rng.uniform(-0.2, 0.2, size=(n, 2))
    z_mm = rng.uniform(0.0, H, size=n)
    pos_mm = np.column_stack([xy, z_mm])
    pos_frac = pos_mm.copy()
    pos_frac[:, 2] = z_mm / H
    return sources, pos_mm, pos_frac


def test_wrapper_forwards_column_height():
    sources, pos_mm, pos_frac = _fixture()
    via_wrapper = project_sources_to_laminar_field(sources, pos_mm, n_contacts=8, column_height_mm=H)
    via_fraction = project_laminar_sources(sources, pos_frac, n_contacts=8)
    np.testing.assert_allclose(
        np.asarray(via_wrapper.phi_e_proxy), np.asarray(via_fraction.phi_e_proxy), rtol=1e-5
    )


def _dataset(positions, column_height_mm):
    n = positions.shape[0]
    t = 20
    sources = np.random.default_rng(2).random((t, n)).astype(np.float32)
    return CanonicalDataset(
        seed=0,
        time_ms=np.arange(t, dtype=np.float32),
        X_V_m=np.zeros((t, n), dtype=np.float32),
        X_spikes=np.zeros((t, n), dtype=np.int8),
        H=np.ones((t, n), dtype=np.float32),
        Q=sources,
        positions=positions.astype(np.float32),
        H_semantic="test",
        cause_hashes={},
        metadata={},
        package_head="test",
        column_height_mm=column_height_mm,
    )


def test_materialize_field_uses_dataset_height():
    _, pos_mm, pos_frac = _fixture()
    mm = materialize_field(_dataset(pos_mm, H), spec=SPEC)
    frac = materialize_field(_dataset(pos_frac, None), spec=SPEC)
    np.testing.assert_allclose(
        np.asarray(mm.phi_e_proxy), np.asarray(frac.phi_e_proxy), rtol=1e-5
    )


def test_legacy_dataset_without_height_keeps_fraction_reading():
    _, _, pos_frac = _fixture()
    legacy = materialize_field(_dataset(pos_frac, None), spec=SPEC)
    assert legacy.phi_e_proxy.shape == (20, 8)
    assert bool(np.isfinite(np.asarray(legacy.phi_e_proxy)).all())


def test_e4_materialize_uses_trajectory_height():
    from jaxfne.protocol_e_integration.e4_execution import FrozenE4Trajectory, materialize_phi_ref

    spec = {
        "primary_evidence_operators": {
            "field_operators_F": [{"id": "lfp_ref", "params": {"n_contacts": 8, "width": 0.1}}]
        }
    }
    _, pos_mm, pos_frac = _fixture()
    n, t = pos_mm.shape[0], 20

    def traj(positions, height):
        return FrozenE4Trajectory(
            seed=0,
            neural_mode="E3-null",
            V_m=np.zeros((t, n)),
            spikes=np.zeros((t, n)),
            Q=np.random.default_rng(3).random((t, n)),
            H_K=np.zeros((t, n)),
            delay_state=np.zeros((t, n)),
            positions=positions,
            identity_map=[],
            time_ms=np.arange(t, dtype=np.float64),
            cause_hashes={},
            column_height_mm=height,
        )

    mm = materialize_phi_ref(traj(pos_mm, H), spec=spec)
    frac = materialize_phi_ref(traj(pos_frac, None), spec=spec)
    np.testing.assert_allclose(np.asarray(mm.phi_e_proxy), np.asarray(frac.phi_e_proxy), rtol=1e-5)
