"""Mechanism-aware connection-rule compiler switch in construct().

Two compilers exist for declarative .connections() rules:
  - core._compile_connection_rules: sign-only (receptor inferred from weight
    sign, tau hardcoded exc=2ms/inh=5ms). Always the fallback.
  - core._compile_mechanism_aware_connection_rules: wraps
    connectivity.compile_connection_rules + ConnectionCompileResult.to_edge_list,
    giving mechanism-correct tau/receptor_index, with a sign-correction pass
    on top (compile_connection_rules itself never applies rule["sign"] to
    edge_weight -- verified separately; without the correction an inhibitory
    mechanism would keep a positive weight).

construct() selects the mechanism-aware path ONLY when every declared
connection rule has a resolvable .mechanisms() reference
(_all_connection_rules_declare_resolvable_mechanism); otherwise the sign-only
path runs completely unchanged. This file proves: (1) parity when a declared
mechanism's tau mirrors the legacy default, (2) genuine divergence when it
doesn't, (3) the sign-only path is untouched for models that never declare
a mechanism, (4) a mixed rule set (one rule with mechanism, one without)
also falls back to the sign-only path.
"""

import numpy as np
import jaxfne as jtfne

BASE_KW = dict(
    seed=0, duration_ms=50.0, dt_ms=0.5, areas=("V1",), layers=("L4",),
    cell_types={"E": 0.5, "PV": 0.5}, n=12, emitter="izhikevich",
)


def _declared_rule_tau(model):
    """tau_ms values restricted to the declared-rule edges is hard to isolate
    cleanly post-concat with the base recurrent edges, so tests instead check
    set-membership/presence rather than exact equality of the full array."""
    return np.asarray(model.params["edge_list"].tau_ms)


def test_mechanism_aware_path_matches_legacy_when_tau_mirrors_default():
    cfg_mech = (
        jtfne.laminar_cortex_config(**BASE_KW)
        .mechanisms(name="ampa_legacy", kind="AMPA", tau_ms=2.0, sign="excitatory")
        .mechanisms(name="gabaa_legacy", kind="GABA_A", tau_ms=5.0, sign="inhibitory")
        .connections(name="e_to_i", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=1.0, weight=0.5, sign="excitatory", mechanism="ampa_legacy")
        .connections(name="i_to_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=1.0, weight=0.5, sign="inhibitory", mechanism="gabaa_legacy")
    )
    cfg_legacy = (
        jtfne.laminar_cortex_config(**BASE_KW)
        .connections(name="e_to_i", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=1.0, weight=0.5, sign="excitatory")
        .connections(name="i_to_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=1.0, weight=0.5, sign="inhibitory")
    )

    model_mech = jtfne.construct(cfg_mech)
    model_legacy = jtfne.construct(cfg_legacy)

    el_mech = model_mech.params["edge_list"]
    el_legacy = model_legacy.params["edge_list"]

    assert el_mech.n_edges == el_legacy.n_edges

    def sorted_triples(el):
        pre, post, w, tau = (np.asarray(el.pre), np.asarray(el.post),
                              np.asarray(el.weight), np.asarray(el.tau_ms))
        order = np.lexsort((post, pre))
        return pre[order], post[order], w[order], tau[order]

    mp, mpo, mw, mt = sorted_triples(el_mech)
    lp, lpo, lw, lt = sorted_triples(el_legacy)

    np.testing.assert_array_equal(mp, lp)
    np.testing.assert_array_equal(mpo, lpo)
    np.testing.assert_allclose(mw, lw)
    np.testing.assert_allclose(mt, lt)


def test_mechanism_aware_path_diverges_when_tau_genuinely_differs():
    cfg = (
        jtfne.laminar_cortex_config(**BASE_KW)
        .mechanisms(name="nmda_exc", kind="NMDA", tau_ms=100.0, sign="excitatory")
        .mechanisms(name="gabaa_legacy", kind="GABA_A", tau_ms=5.0, sign="inhibitory")
        .connections(name="e_to_i", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=1.0, weight=0.5, sign="excitatory", mechanism="nmda_exc")
        .connections(name="i_to_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=1.0, weight=0.5, sign="inhibitory", mechanism="gabaa_legacy")
    )
    model = jtfne.construct(cfg)
    tau = _declared_rule_tau(model)
    assert 100.0 in np.unique(tau), "NMDA tau=100ms must flow through, not collapse to legacy 2.0"


def test_mechanism_aware_path_applies_sign_correctly_to_inhibitory_edges():
    """Regression for the sign-application gap: compile_connection_rules itself
    never applies rule['sign'], so without the wrapper's post-correction an
    inhibitory mechanism edge would keep a positive weight."""
    cfg = (
        jtfne.laminar_cortex_config(**BASE_KW)
        .mechanisms(name="gabaa_legacy", kind="GABA_A", tau_ms=5.0, sign="inhibitory")
        .connections(name="i_to_e_only", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=1.0, weight=0.5, sign="inhibitory", mechanism="gabaa_legacy")
    )
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    weight = np.asarray(el.weight)
    tau = np.asarray(el.tau_ms)
    # isolate the declared-rule edges by their distinctive tau=5.0 AND require
    # at least one such edge to have a negative weight (inhibitory).
    declared_mask = tau == 5.0
    assert declared_mask.any()
    assert np.any(weight[declared_mask] < 0), "inhibitory mechanism edges must carry negative weight"


def test_no_mechanism_declared_uses_unchanged_sign_only_path():
    cfg = (
        jtfne.laminar_cortex_config(**BASE_KW)
        .connections(name="e_to_i", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=1.0, weight=0.5, sign="excitatory")
    )
    model = jtfne.construct(cfg)
    tau = _declared_rule_tau(model)
    # sign-only compiler can only ever produce 2.0 (exc) or 5.0 (inh) tau.
    assert set(np.unique(tau).tolist()) <= {2.0, 5.0}


def test_mixed_rule_set_with_and_without_mechanism_falls_back_to_sign_only():
    """A rule set where only SOME rules declare a resolvable mechanism must
    not partially switch -- the gate requires ALL rules to opt in."""
    cfg = (
        jtfne.laminar_cortex_config(**BASE_KW)
        .mechanisms(name="nmda_exc", kind="NMDA", tau_ms=100.0, sign="excitatory")
        .connections(name="e_to_i", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=1.0, weight=0.5, sign="excitatory", mechanism="nmda_exc")
        .connections(name="i_to_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=1.0, weight=0.5, sign="inhibitory")  # no mechanism
    )
    model = jtfne.construct(cfg)
    tau = _declared_rule_tau(model)
    assert 100.0 not in np.unique(tau), "mixed rule set must fall back to sign-only (no partial switch)"


def test_gate_helper_directly():
    from jaxfne.core import _all_connection_rules_declare_resolvable_mechanism as gate

    assert not gate([], [])
    assert not gate([{"mechanism": None}], [{"name": "ampa"}])
    assert not gate([{"mechanism": "ampa"}], [])
    assert not gate([{"mechanism": "ampa"}, {"mechanism": None}], [{"name": "ampa"}])
    assert not gate([{"mechanism": "ampa"}, {"mechanism": "unknown"}], [{"name": "ampa"}])
    assert gate([{"mechanism": "ampa"}, {"mechanism": "gabaa"}], [{"name": "ampa"}, {"name": "gabaa"}])
