"""v0.3.29 identity/selector public-API export tests.

NodeIdentity and SelectorSpec are defined once in experimental_hpc.contracts and
re-exported from the stable jaxfne root. These tests verify the exports exist,
remain a single canonical class (no duplication), and that the old import path
still works.
"""

import jaxfne as jtfne
from jaxfne.experimental_hpc import (
    NodeIdentity as ContractNodeIdentity,
    SelectorSpec as ContractSelectorSpec,
)


def test_node_identity_exported_from_root():
    assert hasattr(jtfne, "NodeIdentity")


def test_selector_spec_exported_from_root():
    assert hasattr(jtfne, "SelectorSpec")


def test_get_signal_exported_from_root():
    assert callable(jtfne.get_signal)


def test_node_identity_single_canonical_class():
    """Root export and experimental path must be the SAME object (no duplicate)."""
    assert jtfne.NodeIdentity is ContractNodeIdentity


def test_selector_spec_single_canonical_class():
    assert jtfne.SelectorSpec is ContractSelectorSpec


def test_node_identity_quartet_format():
    node = jtfne.NodeIdentity(
        global_id=7, area="V1", area_id="V1", local_id=7, layer="L4", cell_type="PV"
    )
    assert node.quartet == "V1:000007:L4:PV"


def test_old_import_path_still_works():
    """The experimental import path must remain valid for backward compatibility."""
    from jaxfne.experimental_hpc import NodeIdentity, SelectorSpec  # noqa: F401

    assert NodeIdentity is jtfne.NodeIdentity
    assert SelectorSpec is jtfne.SelectorSpec
