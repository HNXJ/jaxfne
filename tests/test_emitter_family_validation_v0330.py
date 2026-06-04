"""Regression tests: unsupported emitter families must fail loudly.

Bug: ``Configuration.set_emitter(family="lif")`` followed by ``construct()``
silently ran the Izhikevich kernel instead of raising. The construct/simulate
path only implements Izhikevich; any other declared family must raise a
``ValueError`` rather than silently falling back. Silent placeholder success
is a truth-gate stop condition.
"""

import pytest

import jaxfne as jtfne
from jaxfne.core import _SUPPORTED_EMITTER_FAMILIES


def _build(family):
    """Minimal valid config (field + probe ensured) with the given emitter family."""
    cfg = jtfne.Configuration()
    cfg = cfg.runtime(seed=0, dtype="float32", duration_ms=10.0, dt_ms=0.1)
    cfg = cfg.column("c", layers=["L2/3"], n=2)
    cfg = cfg.cell_types({"E": 1.0})
    cfg = cfg.connectivity()
    cfg = cfg.set_emitter(family=family)
    cfg = cfg.probes(["MUA-proxy"])
    return cfg


class TestEmitterFamilyValidation:
    """Unsupported emitter families must raise loudly in construct()."""

    def test_lif_family_fails_loudly(self):
        """A 'lif' emitter family must raise, not silently run Izhikevich."""
        with pytest.raises(ValueError) as exc:
            jtfne.construct(_build("lif"))
        msg = str(exc.value)
        assert "lif" in msg, "Error must name the requested family"
        assert "izhikevich" in msg, "Error must list supported families"
        assert "silently" in msg.lower() or "fall back" in msg.lower(), (
            "Error must state there is no silent fallback"
        )

    def test_glif_family_fails_loudly(self):
        """A 'glif' emitter family must also raise loudly."""
        with pytest.raises(ValueError, match="glif"):
            jtfne.construct(_build("glif"))

    def test_arbitrary_unsupported_family_fails_loudly(self):
        """Any family outside the supported set must raise."""
        with pytest.raises(ValueError, match="not_a_real_emitter"):
            jtfne.construct(_build("not_a_real_emitter"))

    def test_izhikevich_family_still_constructs_and_simulates(self):
        """The supported Izhikevich path must remain fully functional."""
        model = jtfne.construct(_build("izhikevich"))
        signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
        assert signals.V_m.shape[-1] == 2

    def test_canonical_suite_path_unaffected(self):
        """The canonical suite2 config path must not regress."""
        cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.1, seed=0)
        n_units = int(signals.V_m.shape[-1])
        assert signals.get("vm").shape[-1] == n_units
        assert signals.get("spk").shape[-1] == n_units

    def test_supported_families_set_contains_izhikevich(self):
        """Guard against accidental removal of the only implemented family."""
        assert "izhikevich" in _SUPPORTED_EMITTER_FAMILIES

    def test_no_emitter_declared_uses_default_path(self):
        """A config with no explicit emitter family must still construct (default)."""
        cfg = jtfne.suite2_four_celltype_config(seed=0, duration_ms=10.0, dt_ms=0.1)
        # suite2 configs declare izhikevich or no family; construct must succeed.
        model = jtfne.construct(cfg)
        assert model is not None
