# TFNE-PARAM-01 — CLOSED. Declared parameters reach execution.

**Branch:** `dev`
**Entry state:** `b9f1f96` (TFNE-IMPORT-01 closed)
**Scope:** the compiler/adapter chain. No simulation kernel was modified.
**Authorization:** P1 semantic defect, escalated by Hamm over the remaining
conformance list.

## Defect

TFNE claimed to be an executable specification language while the executed
model used values the specification never declared.

```
declared weight 0.5    -> executed 0.353553
declared weight 0.25   -> executed 0.353553
declared weight 0.125  -> executed 0.353553
declared probability 0.5 -> realized 8 edges, executed 16
```

The executed weight was not a rescaling of the declared value; it was a
constant, `abs(1.0 * g_scale) / sqrt(n)` at `neuronal_tensor.py:839`, so the
declaration had *zero* influence. The declared probability was replaced by a
hardcoded `probability=1.0` in `_wire_connection`, and the declared sign by
`"excitatory" if conn.source_neuron_type == "E" else "inhibitory"`.

Root cause: `realize()` and `to_neuronal_tensor()` were two independent
compilations of one source. `realize()` resolved parameters into
$(s,h_0,\mathcal{I})$; execution went through the tensor bridge, whose
`InterConnection` and `AreaConnection` structures have no field for a weight,
probability or delay, so `construct` supplied its own conventions.

Topology, cell-type split and mechanism identity all transferred, which is why
this survived TFNE-EXEC-01. **Topology identity is not parameter identity.**
My own earlier "topology transfers exactly" PASS was itself an artifact of
testing at `probability = 1.0`, where thinning is a no-op.

## Repair

The smallest design that gives one authoritative resolved representation
feeding both inspection and execution, rather than a second compiler:

```
TFNE -> resolve -> realize -> I["connection_specs"] -> to_configuration -> construct -> simulate
                           \-> (s, h0, I) for inspection
```

- `realize()` records the resolved connectivity in `index_map["connection_specs"]`.
- New `jaxfne.tfne.to_configuration(realization, *, duration_ms, dt_ms, emitter, dtype)` builds the executable `Configuration`. Structure — areas, layers, cell types, neuron order — still comes from the validated `to_neuronal_tensor` + `neuronal_tensor_to_configuration` path; only `metadata["circuit"]["connections"]` and `["mechanisms"]` are replaced by rules built from the realized specs, passing `weight=abs(weight)` with `sign` from its sign.
- No kernel, emitter or `compile_connection_rules` change. `to_neuronal_tensor` keeps its structural role.

Parameters that no execution path consumes are refused, not dropped:

```
E_PARAM_UNSUPPORTED: rule 'k' declares a delay, which no JaxFNE execution
path consumes. Remove it or extend the compiler; it will not be silently
ignored.
```

## Parameter ownership

Every declared parameter class is classified. Unknown is not PASS.

| Parameter | CONFIGURED | REALIZED | EXECUTED | Evidence |
|---|---|---|---|---|
| `weight` | yes | `s["edge_weight"]` | yes — kernel-resolved | `test_a_weight_only`, `test_kernel_resolved_weights_equal_the_realized_weights` |
| `probability` | yes | thins `n_edges` | yes — same thinned set | `test_b_probability_only` |
| `mechanism` identity | yes | `edge_mechanism` + table | yes — receptor index and kind | `test_d_mechanism_and_weight` |
| `direction` | yes | realized pre/post | yes | `test_e_adjacent_relations_keep_distinct_weights` |
| `plasticity` | yes | provenance only | no — not an edge parameter | `test_c2_plasticity_is_preserved_as_provenance_not_dropped` |
| `delay` | refused | refused | refused | `test_c_delay_is_refused_not_dropped` |
| mechanism kinetics (`tau_ms`) | not expressible | `tau_ms: None`, `declared_not_simulated` | bridge's `static.dT_ms`, **not** the named mechanism's standard value | `test_synaptic_tau_does_not_depend_on_the_integration_timestep` |
| fixed mechanism params (`g_mech`, reversal potentials) | not expressible in `tfne/2` rule bodies | — | bridge defaults; reversal potentials are declared metadata only (edges are current-based) | S12 rule bodies are TFNE2-05 |
| geometry (`G`) | yes | `s["geometry"]` | **no** — declared range is inert | `test_declared_geometry_does_not_reach_the_executed_positions` |
| mutable/plastic targets | declared as rule identity | `h0["w"]`, separate from `s["edge_weight"]` | via the registrable HDP surface | `test_plastic_rule_declared_state_param_separation` |

`plasticity` is deliberately *not* refused. It names a rule for the separate
registrable HDP surface rather than a connection parameter, and it is preserved
as inspectable provenance in `rule_params` / `relation_origin`. Declared and
recorded is not the same as declared and discarded. `delay` is refused because
nothing consumes it anywhere: `Configuration.connections()` has no delay field,
`compile_connection_rules` has no delay support, and `InterConnection` carries
none. `EdgeList` does have `delay_steps`, so the gap is the compiler chain
rather than the kernel — logged as **TFNE-PARAM-02**.

## Mechanism kinetics: identity transfers, kinetics does not

Qualifying the parameter classes surfaced a gap that is not TFNE's and is not
repaired here, but must not be left implicit. An edge declared
`mechanism = AMPA` executes with `tau_ms = 0.1`:

```
executed tau_ms        : [0.1]
standard_receptor_specs: {'AMPA': 2.0, 'GABA_A': 5.0, 'NMDA': 100.0, 'GABA_B': 150.0}
```

0.1 is `StaticParams.dT_ms`, the structural bridge's per-connection default
(`neuronal_tensor.py:868` sets the mechanism's `tau_ms` from
`conn.static.dT_ms`). So the mechanism's *identity* transfers — name, receptor
index, and the excitatory/inhibitory split — while its *kinetics* is a bridge
default rather than the named receptor's time constant. A TFNE `AMPA` synapse
does not currently execute with AMPA kinetics.

This is pre-existing and route-independent: the plain
`neuronal_tensor_to_configuration` path yields the same 0.1. TFNE declares no
tau (`mechanism_table` carries `tau_ms: None` and `declared_not_simulated`), so
nothing is being substituted for a declared value — which is why it is reported
rather than fixed under this item. Changing it would alter existing simulated
dynamics. Logged as **TFNE-PARAM-03**.

Kinetics authority was then determined, because the repair cannot be scoped
without it. The candidates do **not** conflict. `StaticParams.dT_ms` owns the
per-connection synaptic tau and is what the bridge reads; despite the "dT" name
it is a time constant, not a timestep. `standard_receptor_specs()` holds the
canonical per-receptor values, and `sign_only_tau_exc_ms`/`_inh_ms` agree with
it at 2.0/5.0. Every hand-written caller copies the canonical value into
`dT_ms` — `dT_ms=AMPA_TAU_MS` with `AMPA_TAU_MS = 2.0` in
`scripts/hdp_1000_neuronal_tensor_column.py:74,101`, and `dT_ms=2.0`/`5.0` in
`examples/08_neuronal_tensor_first.py` and
`scripts/build_canonical_neuronal_tensor_configs.py`. The 0.1 default is a
placeholder, not a competing definition.

That narrows the finding: TFNE's `to_neuronal_tensor` never constructs
`StaticParams`, so a TFNE spec inherits the placeholder while an equivalent
hand-written tensor gets 2.0/5.0. No test asserts the 0.1 — the new tau test
asserts bridge agreement and `dt`-invariance rather than the value, so it
survives a repair, and `tests/test_synaptic_kernel_v011.py:57-60` asserts the
mechanism tau table equals `standard_receptor_specs()`. The full determination
is in `artifacts/todo_stack.md` under TFNE-PARAM-03. It is deliberately not
acted on here: this item's scope is parameter transfer, and retuning receptor
kinetics would change existing scientific trajectories.

### A defect I introduced and fixed here

Because `to_configuration` replaces the circuit's mechanism declarations, it
must supply a `tau_ms`. My first revision used `float(dt_ms)`. That agreed with
the bridge at the default 0.1 **by coincidence** and diverged elsewhere:

```
dt=0.1   : to_configuration tau=[0.1]    tensor-route tau=[0.1]
dt=0.025 : to_configuration tau=[0.025]  tensor-route tau=[0.1]
```

Coupling synaptic decay to the integration timestep means refining `dt`
silently changes the synapse model instead of integrating the same one more
accurately — it breaks dt-refinement convergence. It also would not have been
caught by the equivalence helper, which compares
`(pre, post, weight, mechanism)` and not `tau_ms`. The repair inherits each
mechanism kind's tau from the bridge's own declarations, with
`StaticParams().dT_ms` as the fallback, and is gated by
`test_synaptic_tau_does_not_depend_on_the_integration_timestep`, which asserts
agreement with the bridge at dt = 0.1, 0.025 and 0.5 and invariance across
them.

## Geometry: realized, not executed

Sweeping the parameter classes by semantic class rather than by field surfaced
a second untransferred class. Declared geometry is recorded faithfully and
ignored at execution. At equal seed:

```
declared (0.0, 1.0)   -> realized {'z0': 0.0,  'z1': 1.0},  exec z [0.001, 0.994]
declared (10.0, 20.0) -> realized {'z0': 10.0, 'z1': 20.0}, exec z [0.001, 0.994]
declared (-5.0, -4.0) -> realized {'z0': -5.0, 'z1': -4.0}, exec z [0.001, 0.994]
   identical positions: True
```

The seed must be pinned for this to mean anything: it otherwise defaults to the
normalization digest, which changes with the source text, so the declared range
and the RNG stream would both vary. My first version of the test omitted that
and the confound made it fail.

TFNE's `to_neuronal_tensor` builds `Geometry3D(value_tag="relative")`. A
relative normalization would still let the declared extent matter — 1.0 against
10.0 — so bit-identical output means the declaration is inert, not rescaled.
Logged as **TFNE-PARAM-04** and pinned as the current state rather than
repaired: the intended semantics, absolute or relative, is an open authority
question, and geometry is what field observables are computed against, so
changing it silently would move any LFP-style result.

## Does the kernel consume the realized values?

`edge_table()` could in principle report declared metadata rather than what the
integrator uses, which would make an equivalence test vacuous. It does not: it
reads `params["edge_list"]` and resolves through
`resolve_edge_weight(el, dtype, presynaptic_sign=emitter.sign)`. Every
`simulate_*` in `jaxfne/emitters.py` resolves weight through
`_resolved_edge_weight(edges, jdtype, params)`, which is that same function
with the same arguments.

Measured on a constructed model rather than read off the source:

```
realized  w (unique): [0.5]
kernel    w (unique): [0.5]
edgetable w (unique): [0.5]
kernel == edge_table : True
kernel == realized   : True

declared 8.0 -> kernel unique [8.0]
V_m differs 0.5 vs 8.0 : True   max|dV| = 101.45 mV
```

Stored is not consumed, so the value must also change the dynamics: two specs
differing only in declared weight integrate to different membrane
trajectories. Pinned by `test_declared_weight_changes_the_integrated_trajectory`.

Compact map, rechecked:

- $A \to \mathrm{NF}(A) \to (s,h_0,\mathcal{I})$ — holds; `I` addresses the simulated neuron axis and the two slices partition it.
- $(h_t,x_t;s)\mapsto(h_{t+1},y_t)$ — the kernel consumes the realized connectivity and the realized weight. It does not literally consume the `s` dict; it consumes the `EdgeList` and emitter built from the same resolution. The doctrine Pipeline section now says this rather than implying $s$ is passed to the kernel.

## Harness repair

An edge-count test passed through this defect for the whole of TFNE-EXEC-01, so
the gate is not a count and not prose.

- `tests/test_tfne_parameter_transfer.py` — 15 tests, organized by semantic class (topology, mechanism identity, mechanism kinetics, weight, delay, probability/realization, mutable rule identity, geometry) rather than by field, so coverage extends by class instead of accumulating arbitrary assertions. Realized and executed edges are compared as `Counter` multisets over `(pre, post, weight, mechanism)`, so a substituted parameter cannot pass. Declared values are asymmetric and non-default (0.375, 0.3125, 0.4375, 0.625, 0.875) so a swap or a fallback default is visible. Cases a–h as authorized, plus the two kernel-level checks above, neuron-order agreement, and a pin on the tensor bridge's inability to carry parameters.
- `scripts/run_test_gate.py`: `tests/test_tfne_algebra.py`, `tests/test_tfne_execution.py` and `tests/test_tfne_parameter_transfer.py` added to `DEV_PYTEST_TARGETS`. The curated dev gate previously contained no TFNE test at all, so the entire adoption effort was invisible to it.

## Evidence

```
python -m pytest tests/test_tfne_algebra.py tests/test_tfne_execution.py \
                 tests/test_tfne_parameter_transfer.py -q
    -> 51 passed in 15.98s

python scripts/run_test_gate.py dev
    -> 189 passed, 1 skipped, 2 deselected in 101.83s
       docs language audit: pass
       vocabulary check: pass

python scripts/run_test_gate.py broad
    -> 3931 passed, 75 skipped, 37 deselected, 4 xfailed in 1626.60s
       compileall, ruff (All checks passed!), docs language audit,
       notebook grammar, vocabulary check, docs orphans: all pass
       exit 0
```

## Status

- TFNE-PARAM-01: **CLOSED**
- Opened: **TFNE-PARAM-02** (declared `delay` fails closed; supporting it needs a delay field on the connection-rule surface).
- Opened: **TFNE-PARAM-03** (a named mechanism executes with the bridge's `dT_ms`, not the receptor's own time constant).
- Opened: **TFNE-PARAM-04** (declared geometry is realized but inert at execution).
- Next: remaining measured conformance gaps TFNE2-03 through TFNE2-07, then `CTX-01`.
