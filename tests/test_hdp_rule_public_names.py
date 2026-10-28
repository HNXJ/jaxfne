"""D31/D32 public names live on jaxfne.hdp_rule (owner ruling 2026-10-28)."""

from jaxfne import _hdp_adaptive, _hdp_family_kernel, hdp_rule


def test_bind_theta_to_plant_is_public_alias():
    assert callable(hdp_rule.bind_theta_to_plant)
    assert hdp_rule.bind_theta_to_plant is _hdp_adaptive.bind_theta_to_plant


def test_apply_registered_rule_update_is_public_alias():
    assert callable(hdp_rule.apply_registered_rule_update)
    assert hdp_rule.apply_registered_rule_update is _hdp_family_kernel.apply_registered_rule_update
