"""0.5.2 item 4 (H5 adversarial): epistemic levels + calibration refusal gate.

Policy: every Phi/Y output starts RELATIVE_PROXY. A proxy becomes
REDUCED_PHYSICAL or CALIBRATED only through apply_calibration() with an
explicit CalibrationTransform declaring units+conductivity+distance. Every
other relabel path must refuse.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from jaxfne.fields import (
    EPISTEMIC_CALIBRATED,
    EPISTEMIC_LEVELS,
    EPISTEMIC_REDUCED_PHYSICAL,
    EPISTEMIC_RELATIVE_PROXY,
    AppliedCalibration,
    CalibrationTransform,
    EpistemicRefusal,
    FieldOutput,
    ProbeReadout,
    apply_calibration,
    create_probe,
    lfp_proxy_probe,
    project_laminar_sources,
)


def _field() -> FieldOutput:
    z = jnp.zeros((4, 3), dtype=jnp.float32)
    return FieldOutput(
        source_proxy=z,
        phi_e_proxy=z,
        csd_proxy=z,
        lfp_proxy=z,
        kernel=jnp.zeros((3, 2), dtype=jnp.float32),
        contact_depths=jnp.zeros((3,), dtype=jnp.float32),
        diagnostics={},
    )


def _probe() -> ProbeReadout:
    return lfp_proxy_probe(jnp.zeros((4, 3), dtype=jnp.float32))


def _full_transform(**kw) -> CalibrationTransform:
    base = {"units": "V", "conductivity": 0.3, "distance": "declared_contact_geometry"}
    base.update(kw)
    return CalibrationTransform(**base)


# Defaults -----------------------------------------------------------------


def test_defaults_are_relative_proxy():
    p = _probe()
    assert p.epistemic_level == EPISTEMIC_RELATIVE_PROXY
    assert p.calibration is None
    f = _field()
    assert f.epistemic_level == EPISTEMIC_RELATIVE_PROXY
    assert f.calibration is None
    assert EPISTEMIC_LEVELS == (
        EPISTEMIC_RELATIVE_PROXY,
        EPISTEMIC_REDUCED_PHYSICAL,
        EPISTEMIC_CALIBRATED,
    )


def test_to_dict_carries_epistemic_level_json_safe():
    from jaxfne.io import json_safe

    d = _probe().to_dict()
    assert d["epistemic_level"] == EPISTEMIC_RELATIVE_PROXY
    assert d["calibration"] is None
    json_safe(d)  # must not raise


# Happy paths ---------------------------------------------------------------


def test_calibrate_probe_to_calibrated():
    out = apply_calibration(_probe(), _full_transform())
    assert out.epistemic_level == EPISTEMIC_CALIBRATED
    assert isinstance(out.calibration, AppliedCalibration)
    assert out.calibration.transform.units == "V"
    assert out.calibration.transform.conductivity == 0.3
    d = out.to_dict()
    assert d["epistemic_level"] == EPISTEMIC_CALIBRATED
    assert d["calibration"]["transform"]["units"] == "V"


def test_calibrate_field_to_reduced_physical():
    t = _full_transform(target_level=EPISTEMIC_REDUCED_PHYSICAL)
    out = apply_calibration(_field(), t)
    assert out.epistemic_level == EPISTEMIC_REDUCED_PHYSICAL
    np0 = jnp.zeros((4, 3))
    assert bool(jnp.all(out.phi_e_proxy == np0))


def test_calibration_returns_new_object_data_untouched():
    p = _probe()
    before = p.data
    out = apply_calibration(p, _full_transform())
    assert p.epistemic_level == EPISTEMIC_RELATIVE_PROXY  # original untouched
    assert out.data is before


# Refusal: direct construction relabels ------------------------------------


def test_probe_direct_calibrated_construction_refused():
    with pytest.raises(EpistemicRefusal):
        ProbeReadout(
            name="x",
            kind="x",
            data=jnp.zeros((2, 2)),
            report={},
            epistemic_level=EPISTEMIC_CALIBRATED,
        )


def test_probe_dict_calibration_refused():
    with pytest.raises(EpistemicRefusal):
        ProbeReadout(
            name="x",
            kind="x",
            data=jnp.zeros((2, 2)),
            report={},
            epistemic_level=EPISTEMIC_CALIBRATED,
            calibration={"units": "V", "conductivity": 0.3, "distance": "d"},
        )


def test_field_direct_calibrated_construction_refused():
    with pytest.raises(EpistemicRefusal):
        _replace_level(_field(), EPISTEMIC_CALIBRATED)


def _replace_level(f: FieldOutput, level: str) -> FieldOutput:
    return dataclasses.replace(f, epistemic_level=level)


def test_field_dict_calibration_refused():
    with pytest.raises(EpistemicRefusal):
        dataclasses.replace(
            _field(),
            epistemic_level=EPISTEMIC_REDUCED_PHYSICAL,
            calibration={"units": "V"},
        )


def test_applied_calibration_direct_construction_refused():
    with pytest.raises(EpistemicRefusal):
        AppliedCalibration(transform=_full_transform(), level=EPISTEMIC_CALIBRATED)


def test_proxy_carrying_calibration_record_refused():
    sealed_probe = apply_calibration(_probe(), _full_transform())
    with pytest.raises(EpistemicRefusal):
        dataclasses.replace(
            _probe(), calibration=sealed_probe.calibration
        )  # proxy + record is contradictory


def test_unknown_level_refused():
    with pytest.raises(EpistemicRefusal):
        ProbeReadout(name="x", kind="x", data=jnp.zeros(2), report={}, epistemic_level="PHYSICAL")


def test_factory_relabel_kwarg_refused():
    with pytest.raises(TypeError):
        create_probe(
            "x",
            jnp.zeros((2, 2)),
            method="m",
            epistemic_level=EPISTEMIC_CALIBRATED,  # type: ignore[arg-type]
        )


def test_frozen_structures():
    import dataclasses as _dc

    with pytest.raises(_dc.FrozenInstanceError):
        _probe().epistemic_level = EPISTEMIC_CALIBRATED  # type: ignore[misc]
    with pytest.raises(_dc.FrozenInstanceError):
        _field().epistemic_level = EPISTEMIC_CALIBRATED  # type: ignore[misc]


# Refusal: gate authority ----------------------------------------------------


def test_gate_refuses_non_transform_authority():
    with pytest.raises(EpistemicRefusal):
        apply_calibration(_probe(), {"units": "V", "conductivity": 0.3, "distance": "d"})  # type: ignore[arg-type]
    with pytest.raises(EpistemicRefusal):
        apply_calibration(_probe(), None)  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["units", "conductivity", "distance"])
def test_transform_missing_declaration_refused(missing):
    kw = {"units": "V", "conductivity": 0.3, "distance": "d"}
    kw[missing] = None
    with pytest.raises(EpistemicRefusal):
        CalibrationTransform(**kw)


def test_transform_empty_units_refused():
    with pytest.raises(EpistemicRefusal):
        _full_transform(units="  ")


def test_transform_nonfinite_conductivity_refused():
    with pytest.raises(EpistemicRefusal):
        _full_transform(conductivity=float("inf"))
    with pytest.raises(EpistemicRefusal):
        _full_transform(conductivity=-0.3)


def test_transform_negative_distance_refused():
    with pytest.raises(EpistemicRefusal):
        _full_transform(distance=-1.0)


def test_transform_bad_target_refused():
    with pytest.raises(EpistemicRefusal):
        _full_transform(target_level="PHYSICAL")


def test_gate_refuses_double_calibration():
    once = apply_calibration(_probe(), _full_transform())
    with pytest.raises(EpistemicRefusal):
        apply_calibration(once, _full_transform())


def test_gate_refuses_non_carrier():
    with pytest.raises(EpistemicRefusal):
        apply_calibration({"epistemic_level": EPISTEMIC_RELATIVE_PROXY}, _full_transform())
    with pytest.raises(EpistemicRefusal):
        apply_calibration(jnp.zeros((2, 2)), _full_transform())


# Pytree + jit ---------------------------------------------------------------


def test_field_pytree_roundtrip_preserves_seal():
    out = apply_calibration(_field(), _full_transform())
    leaves, treedef = jax.tree_util.tree_flatten(out)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.epistemic_level == EPISTEMIC_CALIBRATED
    assert isinstance(back.calibration, AppliedCalibration)


def test_jitted_projection_stays_relative_proxy():
    src = jnp.ones((6, 4), dtype=jnp.float32)
    pos = jnp.zeros((4, 3), dtype=jnp.float32)

    def _run(s, p):
        return project_laminar_sources(s, p, n_contacts=3)

    out = jax.jit(_run)(src, pos)
    assert out.epistemic_level == EPISTEMIC_RELATIVE_PROXY
    assert out.calibration is None
