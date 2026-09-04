"""Regression protection for the 0.4.17 AGSDR surface classification.

Locks the CANONICAL / EXPERIMENTAL / COMPATIBILITY / TUTORIAL distinctions
established in the 0.4.17 semantic contract (G1) so docs and API surfaces
cannot silently collapse them again:

- `AGSDR_SURFACE_CLASSIFICATION` registry exists and maps every AGSDR-labelled
  surface to exactly one classification.
- The canonical engines are what `Model.tune(optimizer="AGSDR")` dispatches to;
  `agsdr_transform` (EXPERIMENTAL) is NOT reachable from the string/spec
  dispatch and requires an explicit PRNG key.
- `step_agsdr_transform` / `AGSDRState` remain the documented plain-GD
  misnomer (COMPATIBILITY), not canonical AGSDR.
- Current-facing docs carry the corrected acronym and classification labels;
  the mislabeled patterns found in G1 Review do not reappear.

Truth posture: repository-semantics regression test; no biological claim.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jaxfne.optim import AGSDR_SURFACE_CLASSIFICATION, agsdr_transform
from jaxfne.optim.agsdr import step_agsdr_transform
from jaxfne.optim.core import _resolve_optimizer

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestClassificationRegistry:
    """The registry is the single source of truth for AGSDR surface labels."""

    def test_registry_exists_and_covers_all_surfaces(self):
        assert isinstance(AGSDR_SURFACE_CLASSIFICATION, dict)
        expected = {
            "agsdr": "CANONICAL",
            "AGSDROptimizerSpec": "CANONICAL",
            "OptimizerSpec": "CANONICAL",
            "_resolve_optimizer": "CANONICAL",
            "_run_agsdr_optimization_loop": "CANONICAL",
            "_agsdr_candidates_from_noise": "CANONICAL",
            "propose_blackbox_candidates": "COMPATIBILITY",
            "_tune_matrix_agsdr_optax": "HYBRID",
            "suite2_tune_noise_agsdr_adam": "HYBRID",
            "agsdr_rate_tuning_panel_grid": "TUTORIAL",
            "agsdr_transform": "EXPERIMENTAL",
            "AGSDR": "COMPATIBILITY",
            "AGSDRState": "COMPATIBILITY",
            "step_agsdr_transform": "COMPATIBILITY",
            "tune_laminar_agsdr": "TUTORIAL",
        }
        assert AGSDR_SURFACE_CLASSIFICATION == expected

    def test_every_registered_name_is_importable(self):
        import jaxfne as jtfne
        import jaxfne.optim as jopt
        import jaxfne.tutorial_utils as jtut

        namespaces = {
            "agsdr": jtfne,
            "AGSDROptimizerSpec": jopt,
            "OptimizerSpec": jopt,
            "_resolve_optimizer": jopt,
            "_run_agsdr_optimization_loop": jopt,
            "_agsdr_candidates_from_noise": jopt,
            "propose_blackbox_candidates": jopt,
            "_tune_matrix_agsdr_optax": jopt,
            "suite2_tune_noise_agsdr_adam": jtfne,
            "agsdr_rate_tuning_panel_grid": jtfne.vis,
            "agsdr_transform": jopt,
            "AGSDR": jopt,
            "AGSDRState": jopt,
            "step_agsdr_transform": jopt,
            "tune_laminar_agsdr": jtut,
        }
        for name, ns in namespaces.items():
            assert hasattr(ns, name), f"Registry name {name!r} not importable from {ns.__name__}"

    def test_registry_labels_match_docstring_classifications(self):
        """Each surface's own docstring carries its registry classification."""
        import inspect

        import jaxfne.optim as jopt
        import jaxfne.tutorial_utils as jtut

        surfaces = {
            "agsdr": jopt.agsdr,
            "AGSDROptimizerSpec": jopt.AGSDROptimizerSpec,
            "agsdr_transform": jopt.agsdr_transform,
            "AGSDR": jopt.AGSDR,
            "AGSDRState": jopt.AGSDRState,
            "step_agsdr_transform": jopt.step_agsdr_transform,
            "tune_laminar_agsdr": jtut.tune_laminar_agsdr,
        }
        for name, obj in surfaces.items():
            doc = inspect.getdoc(obj) or ""
            classification = AGSDR_SURFACE_CLASSIFICATION[name]
            if classification in {"EXPERIMENTAL", "COMPATIBILITY", "TUTORIAL"}:
                assert classification in doc, (
                    f"{name} docstring missing {classification} classification label"
                )
            elif classification == "CANONICAL":
                assert ("canonical" in doc.lower()) or ("AGSDR" in doc), (
                    f"{name} docstring missing canonical AGSDR reference"
                )


class TestCanonicalDispatch:
    """Canonical AGSDR is what Model.tune dispatches; EXPERIMENTAL is not."""

    def test_agsdr_string_dispatch_resolves_to_canonical_spec(self):
        spec = _resolve_optimizer("AGSDR")
        assert spec.optimizer == "AGSDR"
        assert spec.optimizer_class == "blackbox"
        assert spec.is_blackbox()

    def test_agsdr_transform_requires_explicit_prng_key(self):
        """EXPERIMENTAL transform must not silently run without a key."""
        pytest.importorskip("optax")
        transform = agsdr_transform()
        with pytest.raises(ValueError, match="PRNG key"):
            transform.update(None, transform.init(1.0))

    def test_raw_transform_object_not_dispatchable_by_string(self):
        """A raw GradientTransformation is not Model.tune's AGSDR path."""
        pytest.importorskip("optax")
        transform = agsdr_transform()
        resolved = _resolve_optimizer(transform)
        assert resolved.optimizer == "unknown"


class TestCompatMisnomerLocks:
    """step_agsdr_transform stays the documented plain-GD misnomer."""

    def test_step_agsdr_transform_is_plain_gradient_descent(self):
        u_t = jnp.array([1.0, -2.0])
        grad_l = jnp.array([0.5, 0.25])
        state = object()
        hyper = {"eta": 0.1}
        u_next, state_out = step_agsdr_transform(u_t, grad_l, state, hyper)
        expected = u_t - 0.1 * grad_l
        assert jnp.allclose(u_next, expected)
        assert state_out is state  # state passed through unchanged

    def test_step_agsdr_transform_state_fields_never_read(self):
        """Genetic/adaptive fields exist but are inert (documented misnomer)."""
        u_t = jnp.array([1.0])
        grad_l = jnp.array([1.0])
        hyper = {"eta": 0.1}
        u_next, _ = step_agsdr_transform(u_t, grad_l, None, hyper)
        assert jnp.allclose(u_next, 0.9)


class TestDocsClassification:
    """Current-facing docs carry the corrected acronym and labels."""

    def test_acronym_canonical_in_current_facing_docs(self):
        canonical = "Adaptive Genetic Stochastic Delta Rule"
        for rel in ["docs/api/objectives.md"]:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            assert canonical in text, f"{rel} missing canonical AGSDR expansion"

    def test_mislabeled_grid_search_relabeled(self):
        text = (REPO_ROOT / "docs/BASELINE_DRIVE_REFERENCE.md").read_text(encoding="utf-8")
        assert "grid search" in text.lower()
        assert "not the AGSDR" in text

    def test_objective_grammar_uses_factory_not_legacy_class(self):
        text = (REPO_ROOT / "docs/guides/objective_grammar.md").read_text(encoding="utf-8")
        assert "jtfne.agsdr(" in text
        assert "COMPATIBILITY" in text
        assert "jtfne.AGSDR(" not in text

    def test_objectives_page_annotates_classifications(self):
        text = (REPO_ROOT / "docs/api/objectives.md").read_text(encoding="utf-8")
        for label in ["CANONICAL", "EXPERIMENTAL", "COMPATIBILITY", "TUTORIAL"]:
            assert label in text
        assert "Adaptive GSDR" not in text

    def test_sharding_docs_reference_canonical_loop(self):
        text = (REPO_ROOT / "docs/api/sharding.md").read_text(encoding="utf-8")
        assert "_run_agsdr_optimization_loop" in text

    def test_tutorial_11_labels_tutorial_variant(self):
        text = (REPO_ROOT / "docs/tutorials/11_multi_laminar_cortical_agsdr.md").read_text(
            encoding="utf-8"
        )
        assert "TUTORIAL variant" in text
        assert "not the canonical AGSDR" in text

    def test_source_uses_canonical_acronym_only(self):
        """No hyphenated 'Genetic-Stochastic' variant survives in jaxfne/."""
        hits = []
        for py_path in (REPO_ROOT / "jaxfne").rglob("*.py"):
            text = py_path.read_text(encoding="utf-8", errors="ignore")
            if "Genetic-Stochastic" in text:
                hits.append(str(py_path))
        assert hits == [], f"Found non-canonical acronym variant in: {hits}"