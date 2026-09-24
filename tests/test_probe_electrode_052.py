"""Adversarial (H5) probe/electrode semantics — 0.5.2 ENGINE item 5.

No invented contacts or positions on scientific paths (extends Rc P4):

- undeclared ``position``/``reference``/``filter_spec`` are recorded as
  ``"undeclared"``, never invented; declaration is record-only (proxy probes
  apply no reference arithmetic and no filter — data bit-identical to the
  bare call);
- ``contact_depths`` without ``field_contact_depths`` is refused unless the
  caller explicitly opts into the labeled constructed fallback;
- the opt-in path labels the synthesis in the report
  (``synthesized_field_contacts`` + assumption), never silent.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from jaxfne.fields.probes import (
    canonical_source,
    lfp_proxy_probe,
    source_probe,
    spk_probe,
    vm_probe,
)


def _array():
    return jnp.arange(24, dtype=jnp.float32).reshape(6, 4)


# Undeclared stays undeclared, never invented ----------------------------------


@pytest.mark.parametrize("probe", [spk_probe, vm_probe, source_probe])
def test_undeclared_electrode_semantics_recorded_not_invented(probe):
    out = probe(_array())
    assert out.report["position"] == "undeclared"
    assert out.report["reference"] == "undeclared"
    assert out.report["filter"] == "undeclared"
    assert "synthesized_field_contacts" not in out.report


def test_lfp_undeclared_electrode_semantics_recorded_not_invented():
    out = lfp_proxy_probe(_array())
    assert out.report["position"] == "undeclared"
    assert out.report["reference"] == "undeclared"
    assert out.report["filter"] == "undeclared"
    assert "synthesized_field_contacts" not in out.report


def test_declared_electrode_semantics_are_record_only_no_data_change():
    x = _array()
    bare = spk_probe(x)
    declared = spk_probe(
        x,
        position=[0.0, 0.33, 0.66, 1.0],
        reference="common_average",
        filter_spec={"kind": "bandpass", "low_hz": 8.0, "high_hz": 25.0},
    )
    # Declaration only: proxy applies no reference arithmetic and no filter.
    np.testing.assert_array_equal(np.asarray(declared.data), np.asarray(bare.data))
    assert declared.report["position"] != "undeclared"
    assert declared.report["reference"] == "common_average"
    assert "bandpass" in declared.report["filter"]


def test_electrode_fragment_preserves_canonical_source_provenance():
    q = canonical_source(_array())
    out = vm_probe(q, reference="bipolar")
    assert out.report["reference"] == "bipolar"
    assert out.report["position"] == "undeclared"
    # Item-3 provenance survives the item-5 fragment.
    assert "source_projection_mode" in out.report


# Refusal without opt-in --------------------------------------------------------


def test_contact_depths_without_field_contacts_refused():
    with pytest.raises(ValueError, match="refuses to invent field contacts"):
        lfp_proxy_probe(_array(), contact_depths=jnp.array([0.25, 0.75]))


def test_refusal_message_names_the_opt_in():
    with pytest.raises(ValueError, match="allow_synthesized_field_contacts"):
        lfp_proxy_probe(_array(), contact_depths=jnp.array([0.25, 0.75]))


def test_declared_field_contacts_need_no_opt_in():
    out = lfp_proxy_probe(
        _array(),
        contact_depths=jnp.array([0.25, 0.75]),
        field_contact_depths=jnp.linspace(0.0, 1.0, 4),
    )
    assert out.report["method"] == "depth_interpolation_on_phi_e_proxy"
    assert "synthesized_field_contacts" not in out.report


# Opt-in path is labeled, never silent ------------------------------------------


def test_opt_in_synthesized_contacts_labeled_in_report():
    out = lfp_proxy_probe(
        jnp.ones((5, 4)),
        contact_depths=jnp.array([0.2, 0.8]),
        allow_synthesized_field_contacts=True,
    )
    assert out.report["method"] == "depth_interpolation_on_phi_e_proxy"
    assert out.report["synthesized_field_contacts"] is True
    assert "field_contacts_synthesized_explicit_opt_in_not_declared" in out.report["assumptions"]


def test_opt_in_matches_declared_linspace_synthesis_numerically():
    phi = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
    contacts = jnp.array([0.2, 0.8])
    opt_in = lfp_proxy_probe(phi, contact_depths=contacts, allow_synthesized_field_contacts=True)
    declared = lfp_proxy_probe(
        phi, contact_depths=contacts, field_contact_depths=jnp.linspace(0.0, 1.0, 4)
    )
    # The opt-in fallback IS linspace(0, 1) — and says so in the report.
    np.testing.assert_array_equal(np.asarray(opt_in.data), np.asarray(declared.data))
    assert opt_in.report["synthesized_field_contacts"] is True
    assert "synthesized_field_contacts" not in declared.report
