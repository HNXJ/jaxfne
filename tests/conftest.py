"""Session-wide pytest fixtures.

Only one concern lives here: resetting jaxfne's module-level
`compilation_registry` singleton between tests. Confirmed via source
(`jaxfne/validation.py::CompilationRegistry`) that it tracks per-function
shape-signature baselines in `self.baselines`, persisted for the lifetime of
the Python process -- with no per-test isolation, a test earlier in the run
order can set a baseline shape for e.g. `simulate`, and an unrelated later
test calling `simulate` with a different (but individually-correct) shape
trips a false "Re-compilation guard alert" that has nothing to do with any
real recompilation inside that later test. Only
`tests/test_v0320_recompilation_guards.py`'s own tests called
`compilation_registry.reset()` themselves; nothing reset it for the rest of
the suite, so a full-suite run's alert count depended on test order rather
than actual behavior.
"""
import jax
import pytest

from jaxfne import compilation_registry


@pytest.fixture(autouse=True)
def _reset_jax_x64_between_tests(request):
    """Protocol analysis modules may enable x64 at import; reset before each test."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    if "test_protocol_w_" in request.node.fspath.basename:
        jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(autouse=True)
def _reset_compilation_registry():
    compilation_registry.reset()
    yield
    compilation_registry.reset()


def pytest_collection_modifyitems(config, items):
    """Dynamically assign smoke, fast, slow, and notebook markers based on module taxonomies."""
    smoke_modules = {
        "test_api_smoke",
        "test_root_import_lightweight",
        "test_public_surface_contract_v0413",
        "test_continuation_contract",
        "test_public_docs_hygiene",
        "test_agent_context_hygiene",
        "test_core_class_hygiene",
    }

    fast_modules = smoke_modules | {
        "test_signals_get_v0329",
        "test_neuronal_tensor_connectivity",
        "test_neuronal_tensor",
        "test_connection_rule_compile_v0330",
        "test_mcc",
        "test_relative_grammar_invariants",
        "test_closure_hp_reconciliation",
        "test_agsdr_classification",
        "test_jdna_truth_gate",
    }

    marker_smoke = pytest.mark.smoke
    marker_fast = pytest.mark.fast

    for item in items:
        mod_name = item.module.__name__.split(".")[-1]
        if mod_name in smoke_modules:
            item.add_marker(marker_smoke)
        if mod_name in fast_modules:
            item.add_marker(marker_fast)

