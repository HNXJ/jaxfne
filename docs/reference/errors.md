# Errors

jaxfne raises two exported error classes on purpose. Both subclass `ValueError`, so a handler for `ValueError` also catches them.

## TFNEError

Import: `from jaxfne.tfne import TFNEError`. Defined in `jaxfne/tfne.py` and listed in its `__all__`.

Raised for a deterministic TFNE specification or realization failure: parser and lexer errors, and the semantic failure classes. Each semantic class is a subclass, for example `TFNEAddressUnknown` or `TFNEMechanismNotPermitted`.

Catch it with `except TFNEError:`.

## EpistemicRefusal

Import: `from jaxfne.fields import EpistemicRefusal`. Defined in `jaxfne/fields/probes.py` and listed in the `__all__` of `jaxfne/fields/__init__.py`.

Raised when a proxy value would be relabelled as calibrated without authority. `apply_calibration()` is the only path from a proxy to a calibrated level. It refuses a non-`CalibrationTransform` authority, an object with no epistemic level, an object that is already calibrated, and a transform that is incomplete or holds a non-finite or wrong-sign declaration.

Catch it with `except EpistemicRefusal:`.

## Refusal index

Every `raise ValueError(...)`, `raise TFNEError(...)` and `raise EpistemicRefusal(...)` in `jaxfne/` whose message is a string literal or an f-string. f-string fields show as `{...}`. Messages are cut to 100 characters. `scripts/generate_errors_index.py` writes the block; do not edit it by hand.

<!-- ERRORS-INDEX:START -->

2309 raise sites in 116 modules. Generated from `jaxfne/**/*.py` by `python scripts/generate_errors_index.py --write`; do not edit by hand.

**`jaxfne/_config.py`**

| Class | Message |
|---|---|
| `ValueError` | `non-finite value at {...}: {...}` |
| `ValueError` | `non-JSON-safe value at {...}: {...}` |
| `ValueError` | `total must be non-negative` |
| `ValueError` | `fractions must not be empty` |
| `ValueError` | `fractions must be finite and non-negative` |
| `ValueError` | `fractions must have positive mass` |
| `ValueError` | `{...}: n_contacts must be >= 2 and an int; got {...}` |
| `ValueError` | `{...}({...}={...}) is not realized: the implementation realizes {...} in {...} only` |
| `ValueError` | `connection {...} in_degree must be a positive integer; got {...}` |
| `ValueError` | `connection {...} in_degree must be a positive integer; got {...}` |
| `ValueError` | `field(conductivity={...}): the Poisson diagnostic needs a finite conductivity > 0` |
| `ValueError` | `field(n_bins={...}): the Poisson diagnostic needs an int n_bins >= 2` |
| `ValueError` | `field({...}=...) has no consumer; supported keys: {...}` |
| `ValueError` | `{...}(relative_baseline={...}) is not realized: it takes a finite real number` |
| `ValueError` | `connectivity(edge_seed={...}) is not realized: it takes an int seed or None` |
| `ValueError` | `cell type fractions must not be empty` |
| `ValueError` | `cell type fraction for {...} must be finite and non-negative; got {...}` |
| `ValueError` | `cell type fractions must have positive total mass` |
| `ValueError` | `probe({...}={...}) is not realized: contacts sit at linspace(0, 1, n_contacts={...})` |
| `ValueError` | `probe({...}=...) has no consumer; supported keys: {...}` |
| `ValueError` | `emitter({...}) is not realized: construct builds the first emitter {...} only. probes()/set_probes()` |
| `ValueError` | `Probe kind/mode {...} uses the retired '*_like' vocabulary; use {...} instead. '*_like' names are no` |
| `ValueError` | `network({...}=...) has no consumer; supported keys: {...}` |
| `ValueError` | `network(layers={...}) is not realized: layers come from column()` |
| `ValueError` | `network(kind={...}) is not realized: that kind marks the network column()/cell_types() build, and co` |
| `ValueError` | `network({...}) is not realized: construct builds the first network {...} only; declare network() onc` |
| `ValueError` | `emitter({...}=...) has no consumer; supported keys: {...}` |
| `ValueError` | `emitter({...}=...) is not realized for family={...}: construct reads it for family='homeostatic_ei' ` |
| `ValueError` | `emitter({...}=...) is not realized for family={...}: construct reads it for family='adex' only` |
| `ValueError` | `emitter({...}=...) is not realized for family={...}: construct reads it for family='izhikevich' only` |
| `ValueError` | `runtime({...}=...) has no consumer; supported keys: {...}` |
| `ValueError` | `column name must be a non-empty string` |
| `ValueError` | `column layers must contain at least one layer label` |
| `ValueError` | `column n must be positive; got {...}` |
| `ValueError` | `column({...}) is not realized together with network({...}): column() replaces that network; declare ` |
| `ValueError` | `duplicate column name: {...}` |
| `ValueError` | `layer_thickness must not be empty` |
| `ValueError` | `layer thicknesses must be finite and positive` |
| `ValueError` | `population N must be positive; got {...}` |
| `ValueError` | `neurons map must not be empty` |
| `ValueError` | `neuron share for {...} must be finite and non-negative; got {...}` |
| `ValueError` | `neurons map must have positive total` |
| `ValueError` | `neuron share keys {...} are not layers of this population; valid layers are {...}` |
| `ValueError` | `layers {...} have no share in neurons=; give every layer a share (0 for none)` |
| `ValueError` | `cell_types(...) with non-empty columns summing to 0 ({...}): declare column populations before cell_` |
| `ValueError` | `connectivity({...}=...) is not realized: construct reads {...}; {...} are kept as labels.{...}` |
| `ValueError` | `plasticity({...}=...) has no consumer: simulate() runs no plasticity kernel; use jtfne.run_stdp_stre` |
| `ValueError` | `cell_params: unsupported selector keys {...} / parameter keys {...}; selectors: cell_type, layer; pa` |
| `ValueError` | `mechanism requires a non-empty name` |
| `ValueError` | `connection requires a non-empty name` |
| `ValueError` | `connection {...} requires source and target selector mappings` |
| `ValueError` | `connection {...} probability must be finite in [0, 1]; got {...}` |
| `ValueError` | `connection {...} weight must be finite; got {...}` |
| `ValueError` | `connection {...} sign must be excitatory/inhibitory/signed or None; got {...}` |
| `ValueError` | `connection {...}: max_in_degree and probability are mutually exclusive sampling modes -- probability` |
| `ValueError` | `connection {...} delay_ms must be a number in ms; got {...}` |
| `ValueError` | `connection {...} delay_ms must be finite and >= 0; got {...}` |
| `ValueError` | `connection {...} {...} must be a finite number >= 0; got {...}` |
| `ValueError` | `connection {...} {...} must be a finite number >= 0; got {...}` |
| `ValueError` | `connection {...} {...} must be finite and >= 0; got {...}` |
| `ValueError` | `connection {...} declares delay_std_ms={...} without delay_ms; the spread draws around one scalar me` |
| `ValueError` | `connection {...} sampling must be 'density', 'bernoulli', 'capped_bernoulli', 'fixed_indegree' or 'm` |
| `ValueError` | `connection {...} sampling={...} is not implemented with max_in_degree (localized sampling); omit one` |
| `ValueError` | `connection {...} declares sampling='fixed_indegree' without in_degree; the fixed-indegree sampler dr` |
| `ValueError` | `connection {...} sampling={...} is not implemented with in_degree={...} (exact fixed in-degree); omi` |
| `ValueError` | `connection {...} declares sampling='fixed_indegree' with max_in_degree={...}; exact fixed in-degree ` |
| `ValueError` | `connection {...} declares in_degree={...} with probability={...}; the fixed-indegree sampler draws a` |
| `ValueError` | `connection {...} declares in_degree={...} with max_in_degree={...}; exact fixed in-degree and locali` |
| `ValueError` | `connection {...} declares weight_sharing={...} without footprint; sharing groups are the footprint r` |
| `ValueError` | `connection {...} declares footprint with probability={...}; the footprint rule selects pairs determi` |
| `ValueError` | `connection {...} declares footprint with max_in_degree={...}; the footprint rule selects pairs deter` |
| `ValueError` | `connection {...} declares footprint with in_degree={...}; the footprint rule selects pairs determini` |
| `ValueError` | `connection {...} declares footprint with sampling={...}; the footprint rule selects pairs determinis` |
| `ValueError` | `lesion requires a non-empty name` |
| `ValueError` | `trainable requires a non-empty name` |
| `ValueError` | `trainable {...} bounds must have exactly 2 elements (low, high); got {...}` |
| `ValueError` | `trainable {...} bounds must be finite; got {...}` |
| `ValueError` | `trainable {...} bounds reversed: low {...} > high {...}` |
| `ValueError` | `objective output requires a non-empty name` |
| `ValueError` | `duplicate {...} declaration name: {...}` |
| `ValueError` | `duplicate {...} declaration name: {...}` |
| `ValueError` | `set_emitter: {...} apply only to family='homeostatic_ei'; got family={...}` |
| `ValueError` | `probe modes must contain at least one mode label` |
| `ValueError` | `area_names must not be empty` |
| `ValueError` | `duplicate area names detected` |
| `ValueError` | `layer {...} has invalid fraction range: ({...}, {...})` |
| `ValueError` | `area must be a non-empty string` |
| `ValueError` | `radius_mm and height_mm must be positive` |
| `ValueError` | `Configuration.drive: {...} must be a mapping; got {...}` |
| `ValueError` | `Configuration.drive: {...} have no consumer and would not execute (P-015); use stimulus_schedule(...` |
| `ValueError` | `drive(baseline_drive_by_cell_type={{...}: {...}}) is not realized: it takes a finite real number` |
| `ValueError` | `optimizer_family must be one of {...}; got {...}` |

**`jaxfne/_construct_adex.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} = {...} is set away from its default, and emitter family 'adex' does not consume it; remove it` |
| `ValueError` | `runtime({...}={...}) conflicts with Simulation({...}={...}): set {...} on Simulation only` |
| `ValueError` | `network(n=...) is required for family 'adex': it sets the population size` |
| `ValueError` | `network(cell_types=...) is required for family 'adex': it maps each cell type to its fraction (types` |
| `ValueError` | `network(cell_types={...}) gives counts {...} that do not sum to n={...}` |
| `ValueError` | `mechanism {...} has kind={...}; the adex route realizes kind 'receptor' only` |
| `ValueError` | `mechanism {...} is not realized on the adex route: it needs params tau_ms > 0 and a finite reversal_` |
| `ValueError` | `connections(name={...}) has no mechanism: the adex route needs mechanism=... naming a declared mecha` |
| `ValueError` | `connections(name={...}, mechanism={...}) names an undeclared mechanism; declared: {...}` |
| `ValueError` | `connections(name={...}) needs weight (nS) on the adex route` |
| `ValueError` | `connections(name={...}) needs delay_ms on the adex route` |
| `ValueError` | `tensor_triple_drive {...} is not realized: it takes a list of {area, layer, cell_type, drive}` |
| `ValueError` | `tensor_triple_drive(area={...}, layer={...}, cell_type={...}) is not realized: it matches no neuron ` |
| `ValueError` | `param_overrides {...} is not realized: it takes a mapping of (area, layer, cell_type) triple to {fie` |
| `ValueError` | `param_overrides key {...} is not realized: it takes an (area, layer, cell_type) triple` |
| `ValueError` | `param_overrides[{...}] {...} is not realized: it takes a mapping of AdEx field name to value` |
| `ValueError` | `param_overrides[{...}] names unknown AdEx parameter field {...}; expected one of {...}` |
| `ValueError` | `param_overrides[{...}][{...}]={...} is not realized: it takes a finite real number` |
| `ValueError` | `param_overrides[{...}] is not realized: it matches no neuron of the harvested population` |
| `ValueError` | `param_overrides leaves Vpeak_mV={...} at or below EL_mV={...} on neuron {...}; Vpeak_mV must be abov` |
| `ValueError` | `neuron_rows and positions come as a harvested pair; pass both or neither` |
| `ValueError` | `param_overrides needs the harvested neuron_rows it overrides; pass both or neither` |
| `ValueError` | `emitter(family='adex') needs adex_decay ('exp' or 'euler'): the kernel has no default` |
| `ValueError` | `neuron_rows holds {...} rows but network(n={...}); pass the harvested pair unchanged` |
| `ValueError` | `neuron_rows cell types disagree with network(cell_types=...): {...} vs {...}` |
| `ValueError` | `connections(name={...}) delay_ms={...} at dt_ms={...} rounds to 0 steps; zero-delay edges are refuse` |
| `ValueError` | `connections(name={...}) delay_ms={...} at dt_ms={...} is {...} steps; zero-delay edges are refused o` |
| `ValueError` | `connections(name={...}) delay_ms={...} is not realized: it takes a finite number in ms >= 0` |
| `ValueError` | `connections(name={...}) declares intra_step_rounds={...} but delay_ms={...} at dt_ms={...} realizes ` |
| `ValueError` | `connections declare distinct intra_step_rounds={...}; the adex route's intra-step fixed point realiz` |
| `ValueError` | `connections declare intra_step_rounds={...} but realize no delay-0 edge at dt_ms={...}; a declared R` |

**`jaxfne/_construct_connectivity.py`**

| Class | Message |
|---|---|
| `ValueError` | `connection rule {...} declares delay_ms but no dt_ms is known at construction; a delay cannot be rea` |
| `ValueError` | `connection rule {...} sampling must be 'density' or 'bernoulli'; got {...}` |
| `ValueError` | `connection rule {...} declares a per-edge spread (weight_std={...}, delay_std_ms={...}, weight_range` |
| `ValueError` | `connection rule(s) {...} declare delay_ms but no dt_ms is known at construction; a delay cannot be r` |
| `ValueError` | `connect() requires at least two models` |
| `ValueError` | `connect() models have mismatched dtypes: {...}` |
| `ValueError` | `connect() models have mismatched dt_ms: {...}` |
| `ValueError` | `namespace must have one entry per model ({...}), got {...}` |
| `ValueError` | `recurrent_backend={...} cannot be realized by connect(): the merged ensemble runs on concatenated ed` |
| `ValueError` | `connect() models have mismatched n_contacts {...}; field projection needs equal contacts` |
| `ValueError` | `n_members must be >= 2, got {...}` |
| `ValueError` | `member must satisfy 0 <= member < {...}, got {...}` |
| `ValueError` | `ensemble_edge_ownership requires a connect() ensemble model` |
| `ValueError` | `ensemble metadata lacks member_edge_counts; reconnect the members with this jaxfne version` |
| `ValueError` | `ensemble edge ranges cover {...} edges but the edge_list carries {...}` |

**`jaxfne/_construct_core.py`**

| Class | Message |
|---|---|
| `ValueError` | `Specify dtype via runtime=RuntimeConfig(dtype=...), not both runtime= and dtype=.` |
| `ValueError` | `Cannot specify both a Simulation object and individual simulation parameters as keyword arguments.` |
| `ValueError` | `compute_fields(): signals were produced by a different model (config_hash {...} != {...}).` |
| `ValueError` | `signals.field is None -- the Simulation ran with record_fields=False. compute_fields() is a thin acc` |
| `ValueError` | `Invalid jaxfne configuration: {...}` |
| `ValueError` | `Unsupported emitter family {...} for construct/simulate. Supported families: {...}. An explicitly de` |
| `ValueError` | `network({...}) is not realized: construct builds the first network {...} only` |
| `ValueError` | `runtime(vmap={...}) is not realized: the configuration's runtime serves simulate(model), which runs ` |
| `ValueError` | `areas({...}) is not realized: construct builds the areas declared with column()/population(), {...}` |
| `ValueError` | `{...}(p_connect={...}) is not realized: it takes a probability in [0, 1]` |
| `ValueError` | `network(p_connect={...}) is not realized on this construction route: it reads connectivity(p_connect` |
| `ValueError` | `network(layers={...}) is not realized on this construction route: layers come from column()` |
| `ValueError` | `connectivity(edge_seed={...}) is not realized on this construction route: the configuration set none` |
| `ValueError` | `{...}(p_connect={...}) cannot be honoured by this construction route: the configuration set none of ` |
| `ValueError` | `network(layers={...}) is not realized on this construction route: it builds one unlayered population` |
| `ValueError` | `recurrent_backend={...} cannot be realized: {...}. The dense kernel runs on emitter.W, which does no` |
| `ValueError` | `geometry_n_units_total_mismatch: geometry.n_units_total={...} but cfg network n={...}` |
| `ValueError` | `explicit NeuronalTensor connectivity produced duplicate executable synaptic identities; identity is ` |
| `ValueError` | `unsupported connectivity_mode for construction: {...}` |
| `ValueError` | `connection rule(s) declare max_in_degree (spatially-localized sampling) or footprint (retinotopic pa` |
| `ValueError` | `connection rule(s) {...} declare max_in_degree, which is only honoured by the mechanism-aware compil` |
| `ValueError` | `connection rule(s) {...} declare in_degree, which is only honoured by the mechanism-aware compiler. ` |
| `ValueError` | `connection rule(s) {...} declare sampling='multinomial', which is only honoured by the mechanism-awa` |
| `ValueError` | `connection rule(s) {...} declare {...}, which the sign-only path cannot compile. The mechanism-aware` |
| `ValueError` | `connection rule(s) {...} declare spatial_sigma, which only parametrizes max_in_degree sampling on th` |
| `ValueError` | `probe {...} declares n_contacts={...} but the field readout uses the first probe's n_contacts={...}` |
| `ValueError` | `field({...}) is not realized: simulate runs the first Poisson declaration {...} only (n_bins default` |
| `ValueError` | `geometry= is not supported on the NeuronalTensor path -- geometry comes from each Layer's declared G` |
| `ValueError` | `construct(cfg, runtime=...) is not supported for a Configuration -- a Configuration already carries ` |
| `ValueError` | `{...}(p_connect={...}) is not realized: it takes a probability in [0, 1]` |
| `ValueError` | `{...}(p_connect={...}) is not realized on the homeostatic_ei route: it builds a dense conductance ma` |
| `ValueError` | `network(cell_types={...}) is not realized on the homeostatic_ei route: E/I labels come from the cano` |
| `ValueError` | `cell_types({...}) is not realized on the homeostatic_ei route: E/I labels come from the canonical sp` |
| `ValueError` | `network(layers={...}) is not realized on the homeostatic_ei route: it builds one unlayered populatio` |
| `ValueError` | `column({...}, ...) is not realized on the homeostatic_ei route: it builds one population of networks` |
| `ValueError` | `population(...) is not realized on the homeostatic_ei route: it builds one population of networks[0]` |
| `ValueError` | `areas({...}) is not realized on the homeostatic_ei route: it builds one population, not these areas` |
| `ValueError` | `uniform3d(...) is not realized on the homeostatic_ei route: positions come from the canonical E/I la` |
| `ValueError` | `layer_fractions(...) is not realized on the homeostatic_ei route: it builds one unlayered population` |
| `ValueError` | `area_layer_cell_types(...) is not realized on the homeostatic_ei route: it builds one unlayered popu` |
| `ValueError` | `connectivity({...}={...}) is not realized on the homeostatic_ei route: it builds a fixed conductance` |
| `ValueError` | `connectivity(edge_seed={...}) is not realized on the homeostatic_ei route: it builds fixed canonical` |
| `ValueError` | `connectivity(tcm_v1_6pop=...) is not realized on the homeostatic_ei route: it builds a fixed conduct` |
| `ValueError` | `connectivity(within_area={...}) is not realized on the homeostatic_ei route: it builds a fixed condu` |
| `ValueError` | `connectivity(recurrent={...}) is not realized on the homeostatic_ei route: it builds a fixed conduct` |
| `ValueError` | `connectivity_mode={...} is not realized on the homeostatic_ei route: it compiles no connection rules` |
| `ValueError` | `suite2_interarea(...) is not realized on the homeostatic_ei route: it builds one population, not a V` |
| `ValueError` | `inter_column_connectivity(...) is not realized on the homeostatic_ei route: it builds one population` |
| `ValueError` | `drive(...) is not realized on the homeostatic_ei route: the canonical circuit fixes its drive` |
| `ValueError` | `connections(name={...}) is not realized on the homeostatic_ei route: it compiles no connection rules` |
| `ValueError` | `cell_params(...) is not realized on the homeostatic_ei route: per-neuron overrides are never applied` |
| `ValueError` | `emitter(preset={...}) is not realized on the homeostatic_ei route: only homeostatic_ei_rules/homeost` |
| `ValueError` | `runtime(canonical_biophysics=True) is not realized on the homeostatic_ei route: no canonical-biophys` |
| `ValueError` | `runtime(random_v0=True) is not realized on the homeostatic_ei route: initial states are fixed by the` |
| `ValueError` | `runtime(recurrent_backend={...}) is not realized on the homeostatic_ei route: it runs its own kernel` |
| `ValueError` | `runtime(synaptic_kernel={...}) is not realized on the homeostatic_ei route: it runs its own kernel` |
| `ValueError` | `runtime(enable_homeostasis=True) is not realized on the homeostatic_ei route: homeostasis is intrins` |
| `ValueError` | `runtime(homeostasis_params=...) is not realized on the homeostatic_ei route: no homeostasis kernel r` |
| `ValueError` | `runtime(enable_hdp=True) is not realized on the homeostatic_ei route: plasticity is intrinsic to the` |
| `ValueError` | `runtime(hdp_params=...) is not realized on the homeostatic_ei route: no HDP kernel reads them here` |
| `ValueError` | `field(solver={...}) is not realized on the homeostatic_ei route: simulate runs no Poisson diagnostic` |
| `ValueError` | `unknown bound_mode {...} for homeostatic_ei; expected 'minimal' or 'stable'` |
| `ValueError` | `unknown {...}_rule {...} for homeostatic_ei; expected one of {...}` |
| `ValueError` | `homeostatic_ei requires at least 2 neurons; got n={...}` |
| `ValueError` | `connections(name={...}) declares intra_step_rounds={...}: the intra-step fixed point is realized on ` |
| `ValueError` | `construct(geometry={...}) is not realized on the homeostatic_ei route: positions come from the canon` |
| `ValueError` | `construct(geometry={...}) is not realized on the adex route: its cells sit at zero positions and no ` |
| `ValueError` | `construct(geometry={...}) is not realized on the hh route: its cells sit at zero positions and no ge` |

**`jaxfne/_construct_extras.py`**

| Class | Message |
|---|---|
| `ValueError` | `invalid_seed_policy: {...}; must be one of {'paired_by_replicate', 'unique_per_trial'}` |
| `ValueError` | `laminar_source_geometry requires at least one LaminarPopulation` |
| `ValueError` | `Invalid LaminarPopulation(s): {...}` |

**`jaxfne/_construct_hh.py`**

| Class | Message |
|---|---|
| `ValueError` | `network(n=...) is required for the hh route: it sets the population size` |
| `ValueError` | `network(cell_types=...) is required for the hh route: it maps each cell type to its fraction (types:` |
| `ValueError` | `network(cell_types={...}) gives counts {...} that do not sum to n={...}` |
| `ValueError` | `mechanism {...} has kind={...}; the hh route realizes kind 'receptor' only` |
| `ValueError` | `mechanism {...} is not realized on the hh route: it needs params tau_ms > 0 and a finite reversal_mV` |
| `ValueError` | `mechanism {...} needs tau_r_ms > 0; got {...}` |
| `ValueError` | `mechanism {...} needs slope_mV > 0; got {...}` |
| `ValueError` | `mechanism {...} needs a finite v_th_mV; got {...}` |
| `ValueError` | `mechanism {...} needs a finite s0; got {...}` |
| `ValueError` | `the hh route needs at least one mechanisms(kind='receptor') declaration` |
| `ValueError` | `the hh route needs at least one connections() rule` |
| `ValueError` | `connections(name={...}) has no mechanism: the hh route needs mechanism=... naming a declared mechani` |
| `ValueError` | `connections(name={...}, mechanism={...}) names an undeclared mechanism; declared: {...}` |
| `ValueError` | `connections(name={...}) needs weight (mS/cm2) on the hh route` |
| `ValueError` | `connections(name={...}) weight must be >= 0 (the reversal sets the sign); got {...}` |
| `ValueError` | `connections(name={...}) delay_ms={...} is not realized on the hh route: the graded-synapse kernel ha` |
| `ValueError` | `connections(name={...}) max_in_degree is not realized on the hh route` |
| `ValueError` | `the hh route takes emitter(family='hh') only (no other emitter keys; morphology/weight_unit/gate_ini` |
| `ValueError` | `receptors declare mixed s0={...}; the kernel takes one scalar receptor_s0` |
| `ValueError` | `weight_unit must be 'density' (mS/cm2) or 'uS' (point process); got {...}` |
| `ValueError` | `weight_unit='uS' needs morphology=... (an HHMorphology): point-process conversion needs compartment ` |
| `ValueError` | `gate_init must be a finite number or None; got {...}` |
| `ValueError` | `gate_init must be a finite number or None; got {...}` |
| `ValueError` | `{...} needs morphology=... (an HHMorphology); got {...}` |
| `ValueError` | `dt_ms must be finite and > 0; got {...}` |
| `ValueError` | `receptors declare mixed s0={...}; the kernel takes one scalar receptor_s0` |
| `ValueError` | `unknown stored weight_unit {...}; expected 'density' or 'uS'` |

**`jaxfne/_construct_lif.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} needs posed (x, y) positions to group competitors; neuron row {...} has none` |
| `ValueError` | `{...} needs finite posed (x, y) positions; neuron row {...} has ({...}, {...})` |
| `ValueError` | `param_overrides {...} is not realized: it takes a mapping of (area, layer, cell_type) triple to {fie` |
| `ValueError` | `param_overrides key {...} is not realized: it takes an (area, layer, cell_type) triple` |
| `ValueError` | `param_overrides[{...}] {...} is not realized: it takes a mapping of LIF field name to value` |
| `ValueError` | `param_overrides[{...}] names unknown LIF parameter field {...}; expected one of {...}` |
| `ValueError` | `param_overrides[{...}][{...}]={...} is not realized: it takes a finite real number` |
| `ValueError` | `param_overrides[{...}] is not realized: it matches no neuron of the built population` |
| `ValueError` | `scan connection {...}={...} is unmapped: resolution 'event_scan' compiles no edges, so declaring {..` |
| `ValueError` | `lif route has no free Izhikevich bridge name for {...} (types in use: {...}); the bridge samples the` |
| `ValueError` | `lif route inherits a non-finite drive from the bridge` |
| `ValueError` | `local_inhibition declares distinct specs {...}; the lif route executes one grouped cap per step — de` |
| `ValueError` | `lif sharing groups disagree with the edge list: {...} executed edges but {...} compiled rule edges` |
| `ValueError` | `connection rule {...} declares weight_sharing but its compiled sharing_group_of_edge disagrees with ` |
| `ValueError` | `lif sharing groups disagree with the edge list: {...} groups for {...} edges` |
| `ValueError` | `lif route needs params['edge_list']; the model carries none (the lif tensor route always compiles on` |
| `ValueError` | `lif route edge arrays disagree: {...} vs {...} vs {...} vs {...}` |
| `ValueError` | `lif route sharing groups disagree with the edge list: {...} vs {...}` |
| `ValueError` | `lif route sharing groups must be >= 0` |
| `ValueError` | `Simulation(ablation={...}) is not realized for emitter family 'lif': the route has no silence/discon` |
| `ValueError` | `RuntimeConfig(enable_homeostasis=True) is not realized for emitter family 'lif': the route has no ho` |
| `ValueError` | `RuntimeConfig(synaptic_kernel={...}) is not realized for emitter family 'lif': the route runs the ex` |
| `ValueError` | `RuntimeConfig(recurrent_backend={...}) is not realized for emitter family 'lif': edges live only in ` |
| `ValueError` | `RuntimeConfig(vmap={...}) is not realized for emitter family 'lif': single-trial simulate() runs one` |
| `ValueError` | `hdp_params keys {...} are not consumed on emitter family 'lif' without enable_hdp: the lif kernel ru` |

**`jaxfne/_construct_population.py`**

| Class | Message |
|---|---|
| `ValueError` | `drive(baseline_drive_by_cell_type={{...}: {...}}) is not realized: it takes a finite real number` |
| `ValueError` | `drive(baseline_drive_by_cell_type={{...}: {...}}) is not realized: no neuron of the built population` |
| `ValueError` | `tensor_triple_drive(area={...}, layer={...}, cell_type={...}) is not realized: it matches no neuron ` |
| `ValueError` | `cell_params(selector={...}) declares v0, but random_v0 is on (the default): the random initial poten` |
| `ValueError` | `cell_params(selector={...}) is not realized on this construction route: it builds one unlayered popu` |
| `ValueError` | `cell_params(selector={...}): {...} has {...} values but the selector matches {...} neurons; give one` |
| `ValueError` | `cell_params(selector={...}) is not realized: it matches no neuron of the built population` |
| `ValueError` | `column({...}, layers={...}) is not realized: uniform3d() builds every column as one 'uniform_3d' lay` |
| `ValueError` | `{...}({...}, ...) is not realized: no column is named {...}` |
| `ValueError` | `{...}({...}, ...) is not realized: column {...} is one 'uniform_3d' layer and reads no layer table` |
| `ValueError` | `{...}(...) is not realized: every column is one 'uniform_3d' layer and reads no layer table. Drop it` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `unsupported connectivity_mode for construction: {...}` |

**`jaxfne/_construct_presets.py`**

| Class | Message |
|---|---|
| `ValueError` | `groups and targets_hz must be non-empty` |
| `ValueError` | `Group names must match between groups and targets_hz` |
| `ValueError` | `burn_in_ms must be finite and non-negative` |
| `ValueError` | `window_end_ms must be finite when provided` |
| `ValueError` | `window_end_ms must be greater than burn_in_ms` |
| `ValueError` | `epsilon must be finite and positive` |
| `ValueError` | `Group '{...}' indices must be 1D` |
| `ValueError` | `cell_type must be one of {...}` |
| `ValueError` | `target_rate_hz must be a positive (low, high) tuple` |

**`jaxfne/_construct_rate.py`**

| Class | Message |
|---|---|
| `ValueError` | `param_overrides {...} is not realized: it takes a mapping of (area, layer, cell_type) triple to {fie` |
| `ValueError` | `param_overrides key {...} is not realized: it takes an (area, layer, cell_type) triple` |
| `ValueError` | `param_overrides[{...}] {...} is not realized: it takes a mapping of rate parameter field name to val` |
| `ValueError` | `param_overrides[{...}] is not realized: it matches no neuron of the built population` |
| `ValueError` | `weight_sharing={...} is not realized for emitter family 'rate': shared weights execute on the lif ro` |
| `ValueError` | `local_inhibition is not realized for emitter family 'rate': the grouped intra-step cap executes on t` |
| `ValueError` | `rate route inherits a non-finite drive from the bridge` |
| `ValueError` | `rate route needs params['edge_list']; the model carries none (the rate tensor route always compiles ` |
| `ValueError` | `rate route edge arrays disagree: {...} vs {...} vs {...} vs {...}` |
| `ValueError` | `Simulation(ablation={...}) is not realized for emitter family 'rate': the route has no silence/disco` |
| `ValueError` | `RuntimeConfig(enable_homeostasis=True) is not realized for emitter family 'rate': the route has no h` |
| `ValueError` | `RuntimeConfig(synaptic_kernel={...}) is not realized for emitter family 'rate': the route runs the r` |
| `ValueError` | `RuntimeConfig(recurrent_backend={...}) is not realized for emitter family 'rate': edges live only in` |
| `ValueError` | `RuntimeConfig(vmap={...}) is not realized for emitter family 'rate': single-trial simulate() runs on` |
| `ValueError` | `RuntimeConfig(enable_hdp=True) is not realized for emitter family 'rate': the route runs fixed weigh` |
| `ValueError` | `hdp_params keys {...} are not consumed on emitter family 'rate' without enable_hdp: the rate kernel ` |

**`jaxfne/_edge_class_storage.py`**

| Class | Message |
|---|---|
| `ValueError` | `presynaptic_sign required for compact edge weight audit` |

**`jaxfne/_hdp_adaptive.py`**

| Class | Message |
|---|---|
| `ValueError` | `h_state_locality must be 'node' or 'population', got {...}` |
| `ValueError` | `h_state_dim must be a positive integer` |
| `ValueError` | `unknown locality {...}` |
| `ValueError` | `population restoring HDP requires controller_B in hdp_params` |
| `ValueError` | `population-restoring HDP is a two-coordinate controller (rate error channels E/I, theta channels edg` |
| `ValueError` | `controller_B must be ({...}, {...}), got {...}` |
| `ValueError` | `population restoring HDP requires m_ei_edge_mask` |
| `ValueError` | `controller_theta_S_init must have length 2, got {...}` |
| `ValueError` | `population restoring HDP requires controller_rate_setpoint_E_hz and controller_rate_setpoint_I_hz in` |
| `ValueError` | `edge channel requires edge_mask` |
| `ValueError` | `intrinsic channel requires neuron_mask` |
| `ValueError` | `{...}: population H-state locality is not supported for full-state continuation in this release; use` |

**`jaxfne/_hdp_family_kernel.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown HDP family {...}; expected one of {...}` |
| `ValueError` | `hdp_rule {...} declares theta_targets {...} needing state the {...} family lacks; {...} supports {..` |

**`jaxfne/_hdp_registrable_kernel.py`**

| Class | Message |
|---|---|
| `ValueError` | `plasticity_mask must have shape ({...},), got {...}` |
| `ValueError` | `plasticity_mask must be finite (got NaN/inf)` |
| `ValueError` | `forced_spikes must have shape ({...}, {...}), got {...}` |
| `ValueError` | `forced_spikes must have shape ({...}, {...}), got {...}` |
| `ValueError` | `forced_spikes must be finite (got NaN/inf)` |
| `ValueError` | `noise_schedule must have shape ({...}, {...}), got {...}` |
| `ValueError` | `hdp_rule {...} returned undeclared theta_targets {...}; declared: {...}` |
| `ValueError` | `H_final must have shape {...} for hdp_rule {...}, got {...}` |
| `ValueError` | `aux_final must have shape {...} for hdp_rule {...}, got {...}` |
| `ValueError` | `b_final must have shape {...} for hdp_rule {...}, got {...}` |
| `ValueError` | `step_indices must have shape (n_steps,) when provided; got {...} for n_steps={...}` |
| `ValueError` | `step_indices must have shape (n_steps,) when provided; got {...} for n_steps={...}` |

**`jaxfne/_long_diagnostics.py`**

| Class | Message |
|---|---|
| `ValueError` | `twin H shapes differ: {...} vs {...}` |
| `ValueError` | `degenerate assay: twins identical in the early window` |

**`jaxfne/_model.py`**

| Class | Message |
|---|---|
| `ValueError` | `EdgeParameterSpec.bounds must be finite positive (lower, upper), got {...}` |
| `ValueError` | `EdgeParameterSpec requires a pre/post selector, receptor_indices, or explicit edge_indices` |
| `ValueError` | `ParadigmEvent {...} sets both 'target_layer' and 'target_indices' in metadata; pass only one.` |
| `ValueError` | `ParadigmEvent {...} sets metadata['target_layer']={...} but no layer_labels were provided; pass laye` |
| `ValueError` | `ParadigmEvent {...} sets metadata['target_layer']={...} but no neuron has that layer label.` |
| `ValueError` | `Unknown checkpoint schema: {...} (supported: {...})` |

**`jaxfne/_model_simulate.py`**

| Class | Message |
|---|---|
| `ValueError` | `recurrent_backend={...} cannot be realized: {...}. Drop the explicit recurrent_backend, or set recur` |
| `ValueError` | `synaptic_kernel='receptor_exponential' runs only on recurrent_backend='edge_list'; the dense backend` |
| `ValueError` | `RuntimeConfig(surrogate=...) is realized on the dense Izhikevich default-order route only; this run ` |
| `ValueError` | `{...} edge(s) compiled from connections() rules are not carried by the dense backend (it runs on emi` |
| `ValueError` | `ensemble model_sizes sum to {...} but the emitter carries {...} neurons` |
| `ValueError` | `Simulation.gaussian_drive is realized only on the Izhikevich construct routes; emitter route {...} t` |
| `ValueError` | `Simulation.gaussian_drive is refused together with hdp_params['noise_scale']: both set the stochasti` |
| `ValueError` | `Simulation.gaussian_drive is refused together with homeostasis_params['noise_scale']: both set the s` |
| `ValueError` | `Simulation.gaussian_drive std_by_cell_type misses model cell type(s) {...}; the model's cell types a` |
| `ValueError` | `Simulation.gaussian_drive std_by_cell_type names unknown cell type(s) {...}; the model's cell types ` |
| `ValueError` | `edge delay_steps must be >= 0` |
| `ValueError` | `recurrent_backend='dense' has no finite-delay path; edge delay_steps must be all zero (use recurrent` |
| `ValueError` | `SpikeTrainInput is realized only by the Izhikevich edge-list registered-rule HDP kernel (RuntimeConf` |
| `ValueError` | `enable_homeostasis is not supported with synaptic_kernel='receptor_exponential'; use the default exp` |
| `ValueError` | `enable_hdp is not supported with synaptic_kernel='receptor_exponential'; use the default exponential` |
| `ValueError` | `izhikevich_update_order must be one of {...}; got {...}` |
| `ValueError` | `izhikevich_update_order={...} is realized on the dense backend only (it couples emitter.W); this mod` |
| `ValueError` | `SpikeTrainInput is not a StimulusSchedule and cannot be resolved to a drive array here; pass it as s` |
| `ValueError` | `paradigm {...} has {...} conditions ({...}); pass one ParadigmCondition instead.` |
| `ValueError` | `full-state continuation currently supports recurrent and HDP edge-list kernels, not the homeostasis ` |
| `ValueError` | `full-state continuation temporarily does not support ablation={...}; the continuation kernel does no` |
| `ValueError` | `full-state continuation requires recurrent_backend='edge_list'` |
| `ValueError` | `full-state continuation temporarily supports only synaptic_kernel='exponential'; requested {...}` |
| `ValueError` | `drive schedule shape must be ({...}, {...}), got {...}` |
| `ValueError` | `SpikeTrainInput is realized only by the Izhikevich edge-list registered-rule HDP kernel (RuntimeConf` |
| `ValueError` | `forced_spikes must have shape ({...}, {...}), got {...}` |
| `ValueError` | `simulate(continuation=..., return_state=...) is not realized for emitter family 'hh': the route has ` |
| `ValueError` | `paradigm/poisson_drive/ablation is not realized for emitter family 'hh': drive it via jaxfne.emitter` |
| `ValueError` | `connection rule {...} declares weight_sharing={...}; the shared weights execute on the lif route onl` |
| `ValueError` | `connection rule {...} declares local_inhibition={...}; local inhibition executes on the lif route on` |
| `ValueError` | `full-state continuation is not available for homeostatic_ei` |
| `ValueError` | `homeostatic_ei does not consume paradigm/poisson_drive/ablation; drive it via the emitter's own sche` |
| `ValueError` | `SpikeTrainInput cannot be combined with sim.poisson_drive; pass one declared input` |
| `ValueError` | `SpikeTrainInput cannot be combined with ablation={...}; the clamp rides the registered-rule kernel, ` |
| `ValueError` | `SpikeTrainInput n_neurons={...} does not match the model's {...} neurons` |
| `ValueError` | `ablation='shuffled_timing' shuffles a time-varying drive; this run has none (pass a paradigm or pois` |
| `ValueError` | `simulate(continuation=..., return_state=...) is not realized for emitter family 'adex': the route ha` |
| `ValueError` | `simulate(paradigm=...) is not realized for emitter family 'adex': the route has no input current, so` |
| `ValueError` | `intra-step fixed point reached its cap intra_step_rounds={...} with {...} undelivered same-step spik` |
| `ValueError` | `simulate() is not realized for a multi-compartment hh model: run jaxfne.emitters_hh.simulate_hh with` |
| `ValueError` | `simulate() is not realized for plasticity on the hh family: enable_hdp would be silently ignored; ru` |
| `ValueError` | `simulate() does not consume Simulation.seed on the hh family: the run key derives from the declared ` |
| `ValueError` | `simulate(continuation=..., return_state=...) is not realized for emitter family 'rate': the route ha` |
| `ValueError` | `simulate(paradigm=...) is not realized for emitter family 'rate': the route's declared input is Simu` |
| `ValueError` | `simulate(continuation=..., return_state=...) is not realized for emitter family 'lif': the route has` |
| `ValueError` | `local_inhibition={...} declares several distinct (rounds, order, resolution) specs on one tensor; th` |
| `ValueError` | `simulate_lif_seed_batch refuses a template pinning poisson_drive['seed'] ({...}): per-member keying ` |
| `ValueError` | `Simulation(poisson_drive=[...]) entry {...} has target {...}: a list of {...} entries needs an (area` |
| `ValueError` | `Simulation(poisson_drive=[...]) entries {...} and {...} target overlapping neurons: {...} and {...} ` |
| `ValueError` | `local_inhibition resolution {...} is not realized on the tensor simulate path: the cascade needs the` |
| `ValueError` | `paradigm StimulusSchedule has n_neurons={...} but the lif model has {...} neurons; pass a schedule b` |
| `ValueError` | `RuntimeConfig(enable_hdp=True) on emitter family 'lif' needs a registered hdp_rule; got hdp_rule={..` |
| `ValueError` | `hdp_params keys {...} are not consumed on emitter family 'lif' (consumed: {...}); leave them unset` |
| `ValueError` | `hdp_initial_w is not realized for emitter family 'lif' under a registered rule: edge weights come fr` |
| `ValueError` | `local_inhibition groups ({...}) disagree with the lif population ({...})` |
| `ValueError` | `local_inhibition resolution 'event_scan' replays a declared event stream, but paradigm is {...}; pas` |
| `ValueError` | `Simulation(poisson_drive=...) is not realized with local_inhibition resolution 'event_scan': the rep` |
| `ValueError` | `RuntimeConfig(enable_hdp=True) is not realized with local_inhibition resolution 'event_scan': the re` |
| `ValueError` | `local_inhibition resolution 'event_scan' replays the event stream, but the tensor compiled {...} edg` |
| `ValueError` | `EventScanDrive n_steps={...} disagrees with the Simulation grid ({...} steps); build the drive for t` |
| `ValueError` | `EventScanDrive dt_ms={...} disagrees with the Simulation dt_ms={...}; the replay indexes the stream ` |
| `ValueError` | `EventScanDrive groups cover {...} cells but the lif model has {...} neurons; build the drive for thi` |
| `ValueError` | `simulate_condition(condition=...) is not realized for emitter family 'adex': it drives a paradigm, a` |
| `ValueError` | `simulate_condition(condition=...) is not realized for emitter family 'lif': pass an explicit Stimulu` |
| `ValueError` | `simulate_batch does not consume Simulation.{...}; use simulate() per seed or run_trials()` |
| `ValueError` | `simulate_batch does not consume Simulation.record_sources=False; it always returns sources` |
| `ValueError` | `enable_homeostasis is not supported with synaptic_kernel='receptor_exponential'.` |
| `ValueError` | `enable_hdp is not supported with synaptic_kernel='receptor_exponential'; use the default exponential` |
| `ValueError` | `simulate_batch runs the default Izhikevich update order; izhikevich_update_order={...} is realized b` |
| `ValueError` | `seeds must be a non-empty sequence of int seeds; got {...}` |
| `ValueError` | `seeds must be a non-empty sequence of int seeds; got {...}` |
| `ValueError` | `simulate_adex_seed_batch is only realized for emitter family 'adex'; got {...}` |
| `ValueError` | `simulate_adex_seed_batch does not consume Simulation.seed (got {...}); per-member seeds ride poisson` |
| `ValueError` | `simulate_lif_seed_batch is only realized for emitter family 'lif'; got {...}` |
| `ValueError` | `simulate_lif_seed_batch does not consume Simulation.seed (got {...}); per-member seeds are the seeds` |
| `ValueError` | `simulate_lif_seed_batch is not realized for local_inhibition resolution {...}; run event replay thro` |
| `ValueError` | `simulate_lif_seed_batch needs a registered hdp_rule; got hdp_rule={...}` |
| `ValueError` | `hdp_params keys {...} are not consumed on emitter family 'lif' (consumed: {...}); leave them unset` |
| `ValueError` | `hdp_initial_w is not realized for emitter family 'lif' under a registered rule: edge weights come fr` |

**`jaxfne/_model_tune.py`**

| Class | Message |
|---|---|
| `ValueError` | `compact edge_list weight has no magnitude metadata` |
| `ValueError` | `Either 'objective' (legacy) or 'objectives' (public) must be provided` |
| `ValueError` | `generations must be >= 1, got {...}` |
| `ValueError` | `population_size must be >= 1, got {...}` |
| `ValueError` | `EdgeParameterSpec currently uses the black-box AGSDR path; inner gradient refinement is not wired to` |
| `ValueError` | `Unsupported tunable parameter: {...}. Supported parameters: {...}` |
| `ValueError` | `Unknown mask type for parameter {...}: {...}. Supported: E_to_E, E_to_I, I_to_E, excitatory_to_all, ` |
| `ValueError` | `Edge parameter {...} requires a model with an EdgeList` |
| `ValueError` | `Edge parameter {...} contains an out-of-range edge index` |
| `ValueError` | `Edge parameter {...} matched no executable edges` |
| `ValueError` | `EdgeParameterSpec requires simulation.runtime.recurrent_backend='edge_list'` |
| `ValueError` | `MatrixParameterSpec(target='W') addresses emitter.W, while the selected edge_list backend consumes e` |
| `ValueError` | `Edge parameter {...} overlaps another grouped edge parameter` |
| `ValueError` | `Edge parameter {...} requires finite nonzero signed weights to preserve edge identity` |
| `ValueError` | `Edge parameter {...} value {...} is outside declared bounds {...}` |
| `ValueError` | `Matrix parameter {...} targets emitter.W but the model has placeholder W and no edge_list to apply t` |

**`jaxfne/_param_handles.py`**

| Class | Message |
|---|---|
| `ValueError` | `Unknown parameter path {...}: expected a handle string` |
| `ValueError` | `Unknown parameter path {...}: expected 'cells.<izhikevich\|adex>.<field>[<selector>]'` |
| `ValueError` | `Unknown parameter path {...}: family {...} has no field {...} (expected one of {...})` |
| `ValueError` | `Unknown parameter path {...}: expected 'edges.weight[<selector>]'` |
| `ValueError` | `Unknown parameter path {...}: expected 'drive.<name>.rate'` |
| `ValueError` | `Unknown parameter path {...}: expected 'drive.<name>.rate'` |
| `ValueError` | `Unknown parameter path {...}: expected one of 'edges.weight[...]', 'cells.<family>.<field>[...]', 'd` |
| `ValueError` | `Unknown parameter path {...}: missing '[<selector>]' brackets` |
| `ValueError` | `Unknown parameter path {...}: missing '[<selector>]' brackets` |
| `ValueError` | `Unknown parameter path {...}: selector {...} must be 'all', an index, a start:stop slice, a label, o` |
| `ValueError` | `Unknown parameter path {...}: selector key {...} not in {...}` |
| `ValueError` | `Unknown parameter path {...}: selector {...} not in {...}` |
| `ValueError` | `Unknown parameter path {...}: edges selector {...} must be 'all', an index, a start:stop slice, or k` |
| `ValueError` | `Parameter handles are not realized for emitter type {...}` |
| `ValueError` | `Parameter path {...} indexes neuron {...} of {...}` |
| `ValueError` | `Parameter path {...} slice {...} is out of range for {...} neurons` |
| `ValueError` | `Unknown parameter path {...}: selector keys {...} not in {...}` |
| `ValueError` | `Parameter path {...} matched no neurons` |
| `ValueError` | `Parameter path {...}: dense-W models only support 'edges.weight[all]' (indexed edge selectors need a` |
| `ValueError` | `Parameter path {...}: dense-W models only support 'edges.weight[all]'` |
| `ValueError` | `Parameter path {...} indexes edge {...} of {...}` |
| `ValueError` | `Parameter path {...} slice {...} is out of range for {...} edges` |
| `ValueError` | `Unknown parameter path {...}: selector keys {...} not in {...}` |
| `ValueError` | `Parameter path {...} names unknown receptor {...}` |
| `ValueError` | `Parameter path {...} matched no executable edges` |
| `ValueError` | `Parameter path {...} addresses family {...} but the model is family {...}` |
| `ValueError` | `Parameter path {...}: the adex route carries no per-neuron drive array to fit` |
| `ValueError` | `Parameter path {...} selects into a model-wide scalar ({...}); only '[all]' is supported` |
| `ValueError` | `Parameter path {...} selects into a model-wide scalar ({...}); only '[all]' is supported` |
| `ValueError` | `Parameter path {...} addresses a scalar; value shape {...} is not a scalar` |
| `ValueError` | `Parameter path {...}: value shape {...} fits neither the selection ({...}) nor the full array ({...}` |
| `ValueError` | `Parameter path {...}: dense-W value must be a scalar gain or a full {...} array, got {...}` |
| `ValueError` | `Parameter path {...} requires a finite magnitude` |
| `ValueError` | `Parameter path {...} requires finite nonzero signed weights to preserve edge identity` |
| `ValueError` | `Parameter path {...}: value shape {...} fits neither the selection ({...}) nor the full edge vector ` |

**`jaxfne/_pipeline.py`**

| Class | Message |
|---|---|
| `ValueError` | `Unknown checkpoint schema: {...}` |
| `ValueError` | `h_state_dim must be a positive integer` |
| `ValueError` | `dynamic_state_from_model requires model.params['edge_list'] -- the canonical Phase 2 execution mode ` |
| `ValueError` | `hdp_initial_H must have shape {...} for h_state_dim={...}, got {...}` |
| `ValueError` | `delayed continuation requires delay_state (canonical B_t; legacy alias spike_history)` |
| `ValueError` | `delay_state must have shape ({...}, {...}), got {...}` |
| `ValueError` | `kernel must be 'hdp' or 'baseline'` |
| `ValueError` | `record_dH_components=True records the legacy kernel's five dH terms; the registered hdp_rule={...} h` |
| `ValueError` | `hdp_rule_params requires a registered hdp_rule; got hdp_rule={...} with hdp_rule_params {...}` |
| `ValueError` | `ensemble model_sizes sum to {...} but the emitter carries {...} neurons` |
| `ValueError` | `noise_rows must have shape ({...}, {...}), got {...}` |
| `ValueError` | `drive_schedule must have shape (n_steps, n_neurons)` |
| `ValueError` | `stride must be a positive integer; got {...}` |
| `ValueError` | `drive_schedule must have shape (n_steps, n_neurons)` |
| `ValueError` | `drive_schedule must have n_steps > 0` |
| `ValueError` | `recorder stride must be a positive integer; got {...}` |

**`jaxfne/_runtime_config.py`**

| Class | Message |
|---|---|
| `ValueError` | `synaptic_kernel must be one of {...}; got {...}` |
| `ValueError` | `recompilation_guard must be one of {'warning', 'exception', 'off'}; got {...}` |
| `ValueError` | `jit must be a boolean or 'auto'; got {...}` |
| `ValueError` | `vmap must be a boolean or 'auto'; got {...}` |
| `ValueError` | `enable_homeostasis and enable_hdp are mutually exclusive controllers; enable only one.` |
| `ValueError` | `dtype must be one of {...}; got {...}. Previously any unrecognized value silently fell back to float` |
| `ValueError` | `backend must be one of {'auto', 'cpu', 'gpu', 'tpu'}; got {...}.` |
| `ValueError` | `precision={...} is not consumed by any kernel; only 'default' is accepted` |
| `ValueError` | `surrogate method {...} is not realized by any kernel; the kernels realize 'sigmoid_beta' (pass surro` |
| `ValueError` | `surrogate beta must be > 0 (1/mV); got {...}: 0 gives a zero gradient and a negative beta flips its ` |
| `ValueError` | `x64_enabled=True does not enable x64; call jaxfne.enable_x64() before building arrays and request dt` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `Unknown surrogate method: {...} (P-031). Use "none", "straight_through", or "sigmoid_beta".` |
| `ValueError` | `beta must be finite, got {...} (P-031).` |

**`jaxfne/_signals.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} duration_ms must be grid-aligned to dt_ms (duration_ms/dt_ms not integer within tolerance; dur` |
| `ValueError` | `Simulation.gaussian_drive must be a mapping {'std_by_cell_type': {cell_type: std}}; got {...}` |
| `ValueError` | `Simulation.gaussian_drive key(s) {...} are not recognised; the only key is 'std_by_cell_type'` |
| `ValueError` | `Simulation.gaussian_drive requires the key 'std_by_cell_type'` |
| `ValueError` | `Simulation.gaussian_drive['std_by_cell_type'] must be a non-empty mapping from cell type to std; got` |
| `ValueError` | `Simulation.gaussian_drive std_by_cell_type[{...}] must be a finite number >= 0; got {...}` |
| `ValueError` | `Simulation.poisson_drive must be None, a mapping, or a non-empty list of mappings (one per populatio` |
| `ValueError` | `Simulation.duration_ms must be positive and finite; got {...}` |
| `ValueError` | `Simulation.dt_ms must be positive and finite; got {...}` |
| `ValueError` | `Simulation produces n_steps={...} <= 0 for duration_ms={...}, dt_ms={...}` |
| `ValueError` | `Simulation.ablation must be None or one of {...}; got {...}` |
| `ValueError` | `Simulation.plasticity={...} is refused: the gain label is not read by any kernel (only the admitted ` |
| `ValueError` | `runtime.seed={...} conflicts with Simulation.seed={...}; set the seed on Simulation only` |
| `ValueError` | `runtime.n_steps={...} conflicts with the Simulation's n_steps={...}; set duration_ms/dt_ms on Simula` |
| `ValueError` | `Pass either selector=SelectorSpec(...) or the area/layer/cell_type/ids fields, not both.` |
| `ValueError` | `sources not recorded (run with record_sources=True)` |
| `ValueError` | `field not recorded; {...} unavailable (record_fields=True)` |
| `ValueError` | `field output has no attribute {...}` |
| `ValueError` | `Selector cannot be applied to signal {...}: {...} is a laminar field readout with no declared neuron` |
| `ValueError` | `Cannot apply selector to {...}: expected a neuron axis (ndim >= 2), got shape {...}` |
| `ValueError` | `Cannot apply selector to {...}: trailing axis {...} does not match neuron count {...}; no declared n` |
| `ValueError` | `Selector requested but this run has no neuron identity table (metadata['neuron_metadata'] is None). ` |
| `ValueError` | `poisson_drive target {...} is not implemented; only 'all' (every neuron) is supported` |
| `ValueError` | `poisson_drive target {...} must be 'all' or an (area, layer, cell_type) triple of strings` |
| `ValueError` | `poisson_drive target {...} resolves to no neurons (area={...}, layer={...}, cell_type={...}); refusi` |
| `ValueError` | `poisson_drive target {...} resolves to no neurons (area={...}, layer={...}, cell_type={...}); refusi` |
| `ValueError` | `poisson_drive target {...} is not implemented; only 'all' (every neuron) is supported` |
| `ValueError` | `poisson_drive target {...} must be 'all' or an (area, layer, cell_type) triple of strings` |
| `ValueError` | `poisson_drive target {...} is an (area, layer, cell_type) triple: pass target_indices resolved from ` |
| `ValueError` | `poisson_drive target {...} is not implemented; only 'all' (every neuron) is supported` |
| `ValueError` | `poisson_drive counts={...} must be a positive integer` |
| `ValueError` | `poisson_drive counts={...} must be a positive integer` |
| `ValueError` | `poisson_drive rate_hz={...} must be >= 0` |
| `ValueError` | `poisson_drive rate_hz * dt_ms / 1000 exceeds 1, one event per step at most; lower dt_ms or rate_hz` |
| `ValueError` | `poisson_drive target_indices={...} must be a sequence of integer neuron indices` |
| `ValueError` | `poisson_drive target_indices holds {...}: entries must be integer neuron indices` |
| `ValueError` | `poisson_drive target_indices holds {...}: entries must be integer neuron indices` |
| `ValueError` | `poisson_drive target_indices holds {...}: entries must be integer neuron indices` |
| `ValueError` | `poisson_drive target_indices holds {...}: index out of range for n_neurons={...}` |
| `ValueError` | `poisson_drive target_indices is empty; refusing a targeted drive that would reach no neuron` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `target_indices out of bounds: min={...}, max={...}, n_neurons={...}` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `SpikeTrainInput n_neurons={...} must be a positive integer` |
| `ValueError` | `SpikeTrainInput n_neurons={...} must be a positive integer` |
| `ValueError` | `SpikeTrainInput spike_times must be a mapping of neuron index -> sequence of times in ms; got {...}` |
| `ValueError` | `SpikeTrainInput neuron {...} is not an integer neuron index (n_neurons={...})` |
| `ValueError` | `SpikeTrainInput neuron {...} out of range for n_neurons={...}` |
| `ValueError` | `SpikeTrainInput neuron {...} times must be a sequence of spike times in ms; got {...}` |
| `ValueError` | `SpikeTrainInput neuron {...} has non-numeric spike time {...}; times must be finite ms values` |
| `ValueError` | `SpikeTrainInput neuron {...} has non-finite spike time {...}; times must be finite ms values` |
| `ValueError` | `SpikeTrainInput neuron {...} has negative spike time {...}; times must be >= 0 ms` |
| `ValueError` | `SpikeTrainInput neuron {...} spike time {...} ms maps to step {...}, beyond n_steps={...} (dt_ms={..` |
| `ValueError` | `SpikeTrainInput neuron {...} has two spike times in step bin {...} (dt_ms={...}); one spike per bin` |

**`jaxfne/adex_destexhe.py`**

| Class | Message |
|---|---|
| `ValueError` | `mapping must name at least one carrier cell type` |
| `ValueError` | `mapping must hold str -> str, got {...} -> {...}` |
| `ValueError` | `unknown Destexhe cell type {...}; expected one of {...}` |
| `ValueError` | `apply_destexhe_params needs an adex-family model (params['emitter'] is {...}, not AdExParams)` |
| `ValueError` | `mapping names carrier cell types {...} absent from the model labels {...}` |
| `ValueError` | `fraction must be a number in (0, 1]; got {...}` |
| `ValueError` | `fraction must be a number in (0, 1]; got {...}` |
| `ValueError` | `fraction needs a Poisson drive on sim (sim.poisson_drive={...}); got none` |
| `ValueError` | `fraction is not realized for a grouped Poisson drive (sim.poisson_drive={...} names {...} rates): su` |

**`jaxfne/agent.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} must be positive and finite; got {...}` |
| `ValueError` | `duration_ms={...} is not a whole number of dt_ms={...} steps; the kernel would run {...} steps ({...` |
| `ValueError` | `{...} was not recorded in this run (recorded: {...})` |
| `ValueError` | `{...}_identity needs a TFNE run: {...}` |

**`jaxfne/analysis/granger.py`**

| Class | Message |
|---|---|
| `ValueError` | `granger {...} is not realized: it takes a non-empty 1-D trace; got shape {...}` |
| `ValueError` | `granger {...} must stay finite` |
| `ValueError` | `granger traces disagree: {...} has {...} samples, expected {...}` |
| `ValueError` | `granger dt_ms={...} is not realized: it takes a finite value > 0` |
| `ValueError` | `granger order={...} is not realized: it takes an int >= 1` |
| `ValueError` | `granger order={...} is not realized: it takes an int >= 1` |
| `ValueError` | `granger series (n={...}) too short for order={...}: the least-squares fit needs at least {...} post-` |
| `ValueError` | `granger_spectral: a constant series (zero variance) has no directed spectrum; the spectra are undefi` |
| `ValueError` | `granger freqs_hz takes a non-empty finite array` |
| `ValueError` | `granger freqs_hz={...} is not realized: every entry lies in (0, Nyquist] with Nyquist={...} Hz at dt` |
| `ValueError` | `granger_spectral: a series has zero residual variance under the VAR fit (constant or exactly predict` |
| `ValueError` | `granger VAR({...}) transfer is singular at f={...} Hz: the fit is degenerate there (near-unit-root d` |
| `ValueError` | `directionality shapes {...} and {...} do not broadcast` |
| `ValueError` | `directionality inputs must stay finite` |

**`jaxfne/analysis/metrics.py`**

| Class | Message |
|---|---|
| `ValueError` | `spikes must be 2D array, got {...}D` |
| `ValueError` | `spikes must be 2D array, got {...}D` |
| `ValueError` | `active_fraction_threshold must be in [0,1], got {...}` |
| `ValueError` | `spikes must be 2D array, got {...}D` |
| `ValueError` | `bin_size_ms must be > 0, got {...}` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `spikes must be 2D array, got {...}D` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `{...} must be > 0, got {...}` |
| `ValueError` | `{...}={...} must be a positive multiple of dt_ms={...} (bin / dt = {...}, tolerance {...} relative);` |
| `ValueError` | `spikes must be 2D [n_steps, n_neurons], got {...}D` |
| `ValueError` | `unknown activity_regime threshold(s): {...}; known: {...}` |
| `ValueError` | `spikes must be 2D [n_steps, n_neurons], got {...}D` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `spikes must be 2D [n_steps, n_neurons], got {...}D` |
| `ValueError` | `populations must be a non-empty dict of name -> neuron index array, got {...}` |
| `ValueError` | `pass only one of {...} and {...}` |

**`jaxfne/analysis/spectral.py`**

| Class | Message |
|---|---|
| `ValueError` | `unsupported spectral window {...}; expected 'rectangular' or 'hann'` |

**`jaxfne/augment.py`**

| Class | Message |
|---|---|
| `ValueError` | `ScaleN factor must be a positive number; got {...}` |
| `ValueError` | `ScaleN factor must be finite and > 0; got {...}` |
| `ValueError` | `PoseEdit requires a non-empty area name; got {...}` |
| `ValueError` | `RangeEdit requires a non-empty area name; got {...}` |
| `ValueError` | `RangeEdit requires a non-empty layer name; got {...}` |
| `ValueError` | `GeometryTransform.pose_edits must all be PoseEdit records` |
| `ValueError` | `GeometryTransform.range_edits must all be RangeEdit records` |
| `ValueError` | `ambiguous GeometryTransform: two PoseEdits address area {...}; merging them would be order-dependent` |
| `ValueError` | `ambiguous GeometryTransform: two RangeEdits address layer {...} in area {...}; merging them would be` |
| `ValueError` | `{...} must be a finite number; got {...}` |
| `ValueError` | `{...} must be a 3-tuple of finite numbers; got {...}` |
| `ValueError` | `{...} must be a 3-tuple of finite numbers; got {...}` |
| `ValueError` | `{...} must be an (lo, hi) pair within [0, 1]; got {...}` |
| `ValueError` | `{...} must satisfy 0 <= lo <= hi <= 1; got {...}` |
| `ValueError` | `{...} must be a positive number; got {...}` |
| `ValueError` | `{...} must be finite and > 0; got {...}` |
| `ValueError` | `{...} must be a number in [0, 1); got {...}` |
| `ValueError` | `{...} must satisfy 0 <= jitter < 1; got {...}` |
| `ValueError` | `{...} targets must be a sequence of connection addresses; got {...}` |
| `ValueError` | `{...} targets must be non-empty address strings; got {...}` |
| `ValueError` | `duplicate {...} target address {...}; list each connection once` |
| `ValueError` | `{...} must be a finite number; got {...}` |
| `ValueError` | `{...} must be finite; got {...}` |
| `ValueError` | `{...} must be a number >= 0; got {...}` |
| `ValueError` | `{...} must be finite and >= 0; got {...}` |
| `ValueError` | `unknown augmentation record {...}; one record per axis in {...}` |
| `ValueError` | `axis {...} expects a {...} record; got {...} ({...})` |
| `ValueError` | `duplicate augmentation record for axis {...}` |
| `ValueError` | `AugmentationSpec k_v must be an int seed; got {...}` |
| `ValueError` | `a stochastic transform requires its own explicit seed K_V (R4)` |
| `ValueError` | `ScaleN factor {...} maps n_neurons={...} to {...}, not a positive integer; refusing (no silent round` |
| `ValueError` | `ScaleN factor {...} realizes 0 neurons for cell type {...} with fraction {...} in layer {...} of are` |
| `ValueError` | `unknown area {...} in augmentation address` |
| `ValueError` | `unknown layer {...} in area {...} in augmentation address` |
| `ValueError` | `unknown {...} connection address {...}; known addresses: {...}` |
| `ValueError` | `a stochastic {...} transform requires its own explicit seed K_V (R4)` |
| `ValueError` | `{...}: stored {...} {...} is negative or not finite` |
| `ValueError` | `{...}: stored {...} {...} is negative or not finite` |
| `ValueError` | `W_0 refuses {...}: rescaled w_mech {...} is not positive-finite` |
| `ValueError` | `Theta_X refuses {...}.static.g_mech.{...}: undeclared conductance; declare it before augmenting` |
| `ValueError` | `Theta_X refuses {...}.static.g_mech.{...}: rescaled conductance {...} is not positive-finite` |
| `ValueError` | `Theta_X refuses {...}.static.dT_ms: rescaled dT_ms {...} is not finite and > 0` |
| `ValueError` | `Theta_C refuses {...}.delay_ms: rescaled delay_ms {...} is not finite and >= 0` |
| `ValueError` | `Theta_C refuses {...}.probability: stored probability {...} is not in (0, 1]` |
| `ValueError` | `Theta_C refuses {...}.probability: stored probability {...} is not in (0, 1]` |
| `ValueError` | `Theta_C refuses {...}.probability: rescaled probability {...} is above 1 (never clipped)` |
| `ValueError` | `Theta_C refuses {...}.probability: rescaled probability {...} is not finite and > 0` |
| `ValueError` | `H_0 refuses {...}: stored H {...} is not finite` |
| `ValueError` | `H_0 refuses {...}: stored H {...} is not finite` |
| `ValueError` | `H_0 refuses {...}: rescaled H {...} is not finite` |
| `ValueError` | `augment requires a computable tensor identity digest for provenance; the base tensor's digest could ` |
| `ValueError` | `augmentation axis 'N' expects a ScaleN record; got {...} ({...})` |
| `ValueError` | `augmentation axis 'G' expects a GeometryTransform record; got {...} ({...})` |
| `ValueError` | `augmentation axis 'Theta_C' expects a ThetaC record; got {...} ({...})` |
| `ValueError` | `augmentation axis 'Theta_X' expects a ThetaX record; got {...} ({...})` |
| `ValueError` | `augmentation axis 'W_0' expects a W0 record; got {...} ({...})` |
| `ValueError` | `augmentation axis 'H_0' expects a H0 record; got {...} ({...})` |
| `ValueError` | `augmentation axis {...} has no packet-1 implementation` |

**`jaxfne/bridges.py`**

| Class | Message |
|---|---|
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `layout must be 'time_by_unit', 'unit_by_time', or 'recording_by_time', got {...}` |
| `ValueError` | `physical_amplitude_calibrated must be False (immutable); got {...}` |
| `ValueError` | `claim_level must be 'computational_scaffold', got {...}` |
| `ValueError` | `trace must be 2D array, got shape {...} (rank {...})` |
| `ValueError` | `trace contains non-finite values (NaN/Inf)` |
| `ValueError` | `layout must be 'time_by_unit', 'unit_by_time', or 'recording_by_time', got {...}` |
| `ValueError` | `source time dimension {...} != trace {...}` |
| `ValueError` | `recordings must be 1D or 2D (n_recordings, n_time); got shape {...}` |
| `ValueError` | `JaxleyBridge compartment_axis={...} is not implemented; only the default 'last' is supported` |
| `ValueError` | `JaxleyBridge source_mode={...} is not implemented; only the default 'transmembrane_current' is suppo` |
| `ValueError` | `simulate_homeostatic requires recordings: call module.record('v') on the compartments to control bef` |
| `ValueError` | `no recordings for state {...}` |
| `ValueError` | `simulate_laminar_field requires Jaxley HH compartments: the laminar field is generated from reconstr` |
| `ValueError` | `compartment positions unavailable (NaN xyz in module.nodes); pass positions=(n_comp, 3) explicitly.` |

**`jaxfne/builders.py`**

| Class | Message |
|---|---|
| `ValueError` | `ei_profile must be 'flat' or 'canonical', got {...}` |
| `ValueError` | `ei_profile='canonical' requires the canonical 6-layer {...} or 5-layer {...} sequence; got {...}. Pa` |
| `ValueError` | `default_cortical_column_config(layers=...) is not realized: the column is one 'uniform_3d' layer. Us` |
| `ValueError` | `synaptic_kernel must be 'exponential' or 'receptor_exponential'; got {...}` |
| `ValueError` | `radius_mm and height_mm must be positive` |
| `ValueError` | `geometry='uniform3d' builds one 'uniform_3d' layer and reads no layer structure: drop layers/layer_f` |
| `ValueError` | `cell_types fractions must sum to ~1.0; got {...}. Provided: {...}` |
| `ValueError` | `Unknown emitter: {...}. Choose from: izhikevich, lif, glif` |

**`jaxfne/connectivity.py`**

| Class | Message |
|---|---|
| `ValueError` | `delay_ms must be a number in ms; got {...}` |
| `ValueError` | `dt_ms must be a positive number; got {...}` |
| `ValueError` | `dt_ms must be finite and positive; got {...}` |
| `ValueError` | `delay_ms must be finite and >= 0; got {...}` |
| `ValueError` | `delay_ms={...} at dt_ms={...} rounds to 0 steps; a positive delay that fits inside one timestep is r` |
| `ValueError` | `connection {...} delay_ms must be a number in ms; got {...}` |
| `ValueError` | `connection {...} delay_ms must be finite and >= 0; got {...}` |
| `ValueError` | `connection {...} {...} must be a finite number >= 0; got {...}` |
| `ValueError` | `connection {...} {...} must be a finite number >= 0; got {...}` |
| `ValueError` | `connection {...} {...} must be finite and >= 0; got {...}` |
| `ValueError` | `connection {...} weight_range must be a 2-sequence (lo, hi) of finite reals; got {...}` |
| `ValueError` | `connection {...} weight_range must be a 2-sequence (lo, hi) of finite reals; got {...}` |
| `ValueError` | `connection {...} declares weight_range={...} with weight={...}; the range draws each edge's magnitud` |
| `ValueError` | `connection {...} declares weight_range={...} with weight_std={...}; the range and the Gaussian sprea` |
| `ValueError` | `connection {...} weight_range lo must be >= 0; got {...}` |
| `ValueError` | `connection {...} weight_range must satisfy lo < hi; got {...}` |
| `ValueError` | `connection {...} in_degree must be a positive integer; got {...}` |
| `ValueError` | `connection {...} in_degree must be a positive integer; got {...}` |
| `ValueError` | `{...} in_degree with sampling='multinomial' must be a finite positive number (the mean in-degree); g` |
| `ValueError` | `{...} in_degree with sampling='multinomial' must be a finite positive number (the mean in-degree); g` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `{...} selector must be a mapping; got {...}` |
| `ValueError` | `{...} selector declares both {...} and {...}; declare one` |
| `ValueError` | `selector {...} ({...}) matched zero neurons` |
| `ValueError` | `selector {...} ({...}) matched zero neurons` |
| `ValueError` | `connection {...} has no mechanism; declare one and reference it` |
| `ValueError` | `connection {...} references unknown mechanism {...}; declared mechanisms: {...}` |
| `ValueError` | `fixed_indegree in_degree={...} exceeds the {...} available presynaptic neurons for post neuron {...}` |
| `ValueError` | `multinomial sampling draws K>0 synapses but every (pre, post) pair of the rule is a refused self pai` |
| `ValueError` | `connection {...} scalar weight must be finite` |
| `ValueError` | `connection {...} weight must be a number or WeightInitSpec mapping` |
| `ValueError` | `connection {...} scalar value must be finite` |
| `ValueError` | `connection {...} produced non-finite random weights` |
| `ValueError` | `connection {...} matrix shape {...} != (n_pre, n_post)=({...}, {...})` |
| `ValueError` | `connection {...} matrix contains non-finite weights` |
| `ValueError` | `connection {...} artifact_ref requires {...}` |
| `ValueError` | `connection {...} artifact_ref array {...} not supplied; pass it via artifacts={name: array} (compile` |
| `ValueError` | `connection {...} artifact_ref sha256 mismatch for {...}: declared {...} != supplied-array digest {..` |
| `ValueError` | `connection {...} unknown weight mode {...}` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `mechanism declaration missing 'name'` |
| `ValueError` | `mechanism {...} is the tau -> 0 limit: declare tau_ms=0.0; got {...}` |
| `ValueError` | `mechanism {...} tau_ms must be finite and positive; got {...}` |
| `ValueError` | `connection {...} declares intra_step_rounds={...}: the intra-step fixed point is realized on the ade` |
| `ValueError` | `connection {...} sampling must be 'density', 'bernoulli', 'capped_bernoulli', 'fixed_indegree' or 'm` |
| `ValueError` | `connection {...} declares sampling='fixed_indegree' without in_degree; the fixed-indegree sampler dr` |
| `ValueError` | `connection {...} sampling={...} is not implemented with in_degree={...} (exact fixed in-degree); omi` |
| `ValueError` | `connection {...} declares in_degree={...} with probability={...}; the fixed-indegree sampler draws a` |
| `ValueError` | `connection {...} declares in_degree={...} with max_in_degree={...}; exact fixed in-degree and locali` |
| `ValueError` | `connection {...} sampling={...} is not implemented with max_in_degree (localized sampling); omit one` |
| `ValueError` | `connection {...} declares spatial_sigma={...} without max_in_degree; spatial_sigma needs max_in_degr` |
| `ValueError` | `connection {...} declares weight_sharing={...} without footprint; sharing groups are the footprint r` |
| `ValueError` | `connection {...} declares footprint with probability={...}; the footprint rule selects pairs determi` |
| `ValueError` | `connection {...} declares footprint with max_in_degree={...}; the footprint rule selects pairs deter` |
| `ValueError` | `connection {...} declares footprint with in_degree={...}; the footprint rule selects pairs determini` |
| `ValueError` | `connection {...} declares footprint with sampling={...}; the footprint rule selects pairs determinis` |
| `ValueError` | `connection {...} declares footprint but a neuron row lacks x/y posed positions ({...})` |
| `ValueError` | `connection {...} declares footprint but a neuron row carries a non-finite x/y posed position` |
| `ValueError` | `connection {...} probability={...} must be a finite number in [0, 1]` |
| `ValueError` | `connection {...} declares weight_std={...} with a non-scalar weight={...}; the spread draws around o` |
| `ValueError` | `connection {...} declares delay_std_ms={...} without delay_ms; the spread draws around one scalar me` |
| `ValueError` | `compiled edge weights contain non-finite values` |
| `ValueError` | `allow_self_connections for rule {...} must be a bool; got {...}` |
| `ValueError` | `rule {...} must exist exactly once to opt into allow_self_connections; found {...}` |
| `ValueError` | `rule {...} must exist exactly once to declare local_inhibition; found {...}` |

**`jaxfne/emitter_registry.py`**

| Class | Message |
|---|---|
| `ValueError` | `register_emitter needs a non-empty descriptor name` |
| `ValueError` | `emitter {...} is already registered` |
| `ValueError` | `emitter {...} declares inadmissible param key {...}; param keys must be non-empty strings` |
| `ValueError` | `unknown registered emitter {...}; known: {...}` |
| `ValueError` | `emitter {...} params must be a mapping of {...}; got {...}` |
| `ValueError` | `emitter {...} does not consume params {...}; consumed: {...}` |

**`jaxfne/emitters.py`**

| Class | Message |
|---|---|
| `ValueError` | `update_order='net_m' realizes the paper's noise-free thal drive (I = thal[t] + S @ fired); pass nois` |
| `ValueError` | `update_order='net_m' requires a concrete noise_scale=0.0 (zero cannot be verified under trace)` |
| `ValueError` | `update_order='net_m' realizes the paper's noise-free thal drive; got nonzero noise_scale={...}` |
| `ValueError` | `unknown Izhikevich cell type {...}; expected one of {...}` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `cell_type_fractions[{...}] must be finite and non-negative; got {...}` |
| `ValueError` | `cell_type_fractions must have positive mass; got {...}` |
| `ValueError` | `labels must contain at least one emitter label` |
| `ValueError` | `layer_labels length must match labels length` |
| `ValueError` | `unknown Suite No. 2 cell type label: {...}` |
| `ValueError` | `update_order must be one of {...}; got {...}` |
| `ValueError` | `surrogate_beta is realized in update_order='default' only; got {...}` |
| `ValueError` | `update_order='net_m' hard-codes two 0.5 ms v half-steps per 1 ms step; dt_ms must be 1.0, got {...}` |
| `ValueError` | `dt_ms must be positive, got {...}` |
| `ValueError` | `edge delay_ms must be >= 0` |
| `ValueError` | `edge delay_ms must be grid-aligned to dt_ms (delay_ms/dt_ms not integer within tolerance; dt_ms={...` |
| `ValueError` | `delay_ms length {...} != n_edges {...}` |
| `ValueError` | `unsupported EdgeList payload backend: {...}` |
| `ValueError` | `EdgeList payload lacks full array fields (missing: {...}); only summary payloads predate this serial` |
| `ValueError` | `EdgeList payload length mismatch: {...} has {...} rows, expected {...} for n_edges={...} (tau_storag` |
| `ValueError` | `edge_list.tau_storage='from_mechanism_table' requires mechanism_tau_table to be set` |
| `ValueError` | `edge_list.tau_ms has shape {...}, expected ({...},) or a derivable tau_storage mode` |
| `ValueError` | `edge_list.weight_storage='sign_from_receptor' requires weight_magnitude` |
| `ValueError` | `edge_list.weight_storage='magnitude_times_presynaptic_sign' requires presynaptic_sign` |
| `ValueError` | `edge_list.weight_storage='magnitude_times_presynaptic_sign' requires weight_magnitude or mechanism_w` |
| `ValueError` | `edge_list.weight has shape {...}, expected ({...},) or a derivable weight_storage mode` |
| `ValueError` | `materialize_dense_W_from_edge_list requires presynaptic_sign when edge weights use compact storage` |
| `ValueError` | `cannot resolve max edge delay under JAX trace without delay_storage metadata; set delay_storage to u` |
| `ValueError` | `edge delay_steps must be >= 0` |
| `ValueError` | `noise_schedule must have shape ({...}, {...}), got {...}` |
| `ValueError` | `step_indices must have shape (n_steps,) when provided; got {...} for n_steps={...}` |
| `ValueError` | `noise_schedule must have shape ({...}, {...}), got {...}` |
| `ValueError` | `static H_K recovery kernel requires zero edge delays; use simulate_edge_recurrent_izhikevich for del` |
| `ValueError` | `h_k must have shape ({...},), got {...}` |
| `ValueError` | `static H_K recovery requires H_K > 0` |
| `ValueError` | `tau_k_ms must be > 0` |
| `ValueError` | `dynamic H_K recovery kernel requires zero edge delays` |
| `ValueError` | `h_k0 must have shape ({...},), got {...}` |
| `ValueError` | `dynamic H_K recovery requires H_K > 0` |
| `ValueError` | `dynamic owned H_K requires tau_k_ms > 0` |
| `ValueError` | `h_k0 must have shape ({...},), got {...}` |
| `ValueError` | `owner_mask must have shape ({...},), got {...}` |
| `ValueError` | `step_indices must have shape (n_steps,) when provided; got {...} for n_steps={...}` |
| `ValueError` | `tau_a_ms and tau_k_ms must be > 0` |
| `ValueError` | `activity H_K RBD kernel requires zero edge delays` |
| `ValueError` | `h_k0 must have shape ({...},), got {...}` |
| `ValueError` | `h_a0 must have shape ({...},), got {...}` |
| `ValueError` | `D2b requires H_K > 0` |
| `ValueError` | `D2b requires H_A >= 0` |
| `ValueError` | `rbd_family must be one of {...}, got {...}` |
| `ValueError` | `Protocol H1c requires G_H(H;beta_H)>0; nonpositive recurrent gain invalidates the trajectory` |
| `ValueError` | `Protocol H F2 requires H>0; nonpositive initial H invalidates the trajectory` |
| `ValueError` | `delayed continuation requires init_state keys {...}; missing {...}` |
| `ValueError` | `delayed continuation requires delay_state (or legacy spike_history)` |
| `ValueError` | `delay_state must have shape ({...}, {...}), got {...}` |
| `ValueError` | `syn_state must have shape ({...},), got {...}` |
| `ValueError` | `tau_h_ms must be > 0` |
| `ValueError` | `i_ref must be > 0` |
| `ValueError` | `H0 must have shape ({...},), got {...}` |
| `ValueError` | `noise_schedule must have shape ({...}, {...}), got {...}` |
| `ValueError` | `record_stride must be a positive integer; got {...}` |
| `ValueError` | `record_stride must be a positive integer; got {...}` |
| `ValueError` | `{...} must be a 1D integer index array; got shape {...} dtype {...}` |
| `ValueError` | `{...} must be non-empty (None records all)` |
| `ValueError` | `{...} indices out of range for size {...}` |
| `ValueError` | `record_h_subset selects per-neuron H coordinates, which do not exist under population H locality (H ` |
| `ValueError` | `record_w_subset requires record_weight_trace=True (no W trace is recorded to subset)` |
| `ValueError` | `h_state_dim must be a positive integer` |
| `ValueError` | `h_state_dim must be a positive integer` |
| `ValueError` | `plasticity_mask is a node-local control and is refused with population h_state_locality (per-edge we` |
| `ValueError` | `plasticity_mask must have shape ({...},), got {...}` |
| `ValueError` | `plasticity_mask must be finite (got NaN/inf)` |
| `ValueError` | `{...} must be scalar or have shape ({...},), got {...}` |
| `ValueError` | `h_state_readout must have shape ({...},), got {...}` |
| `ValueError` | `h_state_coupling must have shape ({...}, {...}), got {...}` |
| `ValueError` | `noise_schedule must have shape {...}, got {...}` |
| `ValueError` | `H_final must have shape {...} for population H, got {...}` |
| `ValueError` | `theta_S_final must have shape ({...},), got {...}` |
| `ValueError` | `H_final must have shape {...} for h_state_dim={...}, got {...}` |
| `ValueError` | `H_final must have shape {...} for h_state_dim={...}, got {...}` |
| `ValueError` | `Unknown hdp_rule: {...}. Must be one of: signed_linear, signed_quadratic, hebbian_product` |
| `ValueError` | `unrecognized receptor mechanism {...}; valid names: {...}` |
| `ValueError` | `spikes_pre channel dim {...} != tau_ms length {...}` |
| `ValueError` | `receptor_exponential synaptic_kernel has no finite-delay path; edges.delay_steps must be all zero (u` |
| `ValueError` | `neurons_df is empty` |
| `ValueError` | `n must be positive` |
| `ValueError` | `W must have shape {...}, got {...}` |

**`jaxfne/emitters_adex.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown AdEx cell type {...}; expected one of {...}` |
| `ValueError` | `overrides name unknown AdEx cell type {...}; expected one of {...}` |
| `ValueError` | `overrides name cell type {...}, which is not in labels {...}` |
| `ValueError` | `unknown AdEx parameter field {...} for cell type {...}; expected one of {...}` |
| `ValueError` | `overrides for cell type {...}: Vpeak_mV must be a finite float above EL; got {...}` |
| `ValueError` | `overrides for cell type {...}: Vpeak_mV must be a finite float above EL; got {...}` |
| `ValueError` | `overrides for cell type {...}: Vpeak_mV must be a finite float above EL; got {...}` |
| `ValueError` | `overrides for cell type {...}: Vpeak_mV={...} cannot be checked against a non-real EL_mV={...}` |
| `ValueError` | `overrides for cell type {...}: Vpeak_mV={...} must be above EL_mV={...}` |
| `ValueError` | `step_order must be 'default' or 'brian2', got {...}` |
| `ValueError` | `{...} must be a number; got {...}` |
| `ValueError` | `{...} must be a finite float >= 0; got {...}` |
| `ValueError` | `ext_lambda must be a finite float >= 0, a 1-d [n_steps] array, or a 2-d [n_steps, M] array; got {...` |
| `ValueError` | `ext_lambda must be a finite float >= 0, a 1-d [n_steps] array, or a 2-d [n_steps, M] array; got {...` |
| `ValueError` | `ext_lambda schedule must be non-empty; got shape {...}` |
| `ValueError` | `ext_lambda schedule must hold finite values >= 0` |
| `ValueError` | `ext_target_groups must hold one int group per target (shape ({...},)); got shape {...}` |
| `ValueError` | `ext_target_groups must hold integers >= 0` |
| `ValueError` | `ext_target_groups needs a 2-d [n_steps, G] ext_lambda schedule (one rate per group per step); got {.` |
| `ValueError` | `ext_lambda schedule with per-group rates must have shape ({...}, G); got {...}` |
| `ValueError` | `ext_target_groups names group {...} but the ext_lambda schedule holds {...} groups` |
| `ValueError` | `ext_target_groups leaves a group empty (sizes {...}): every group needs at least one target cell` |
| `ValueError` | `{...} must be an int step index or None (with its pair); got {...}` |
| `ValueError` | `ext_t_start_steps={...} and ext_t_stop_steps={...} must satisfy 0 <= t_start < t_stop <= n_steps ({.` |
| `ValueError` | `external drive arguments (ext_targets, ext_lambda, ext_weight_nS, ext_receptor, ext_delay_steps, key` |
| `ValueError` | `ext_targets must be one-dimensional, got shape {...}` |
| `ValueError` | `ext_targets must hold integers` |
| `ValueError` | `ext_targets must hold at least one cell index` |
| `ValueError` | `ext_targets must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `ext_targets must hold unique cell indices; got {...}` |
| `ValueError` | `{...} must be a number; got {...}` |
| `ValueError` | `{...} must be a finite float >= 0; got {...}` |
| `ValueError` | `ext_receptor must index receptors in [0, {...}); got {...}` |
| `ValueError` | `ext_delay_steps must be an integer >= {...}; got {...}` |
| `ValueError` | `key must be a JAX PRNG key; got {...}` |
| `ValueError` | `edge arrays must be all given or all None; missing {...}` |
| `ValueError` | `{...} must be one-dimensional, got shape {...}` |
| `ValueError` | `edge arrays must have equal lengths; got {...}` |
| `ValueError` | `{...} must hold integers` |
| `ValueError` | `edge_pre must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge_post must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge_delay_steps must be >= 1; got min {...}` |
| `ValueError` | `edge_receptor must index receptors in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `delay-0 edge arrays must be all given or all None; missing {...}` |
| `ValueError` | `{...} must be one-dimensional, got shape {...}` |
| `ValueError` | `delay-0 edge arrays must have equal lengths; got {...}` |
| `ValueError` | `{...} must hold integers` |
| `ValueError` | `edge0_pre must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge0_post must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge0_receptor must index receptors in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge0_weight_nS must hold finite values` |
| `ValueError` | `step_order must be 'default' or 'brian2', got {...}` |
| `ValueError` | `decay must be 'exp' or 'euler', got {...}` |
| `ValueError` | `n_steps must be a positive integer, got {...}` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `N * n_receptors must be < 2**31 (int32 segment ids); N = {...}` |
| `ValueError` | `receptor_reversal_mV and receptor_tau_ms must have the same length; got {...} and {...}` |
| `ValueError` | `receptor_tau_ms must be > 0 for every receptor; got {...}` |
| `ValueError` | `decay='euler' needs receptor_tau_ms > dt_ms ({...}); got {...}` |
| `ValueError` | `ext_target_groups needs the external drive; got {...} with no ext_targets -- pass the six drive argu` |
| `ValueError` | `ext_t_start_steps/ext_t_stop_steps need the external drive; got {...}/{...} with no ext_targets — pa` |
| `ValueError` | `ext_delay_steps=0 requests a same-step drive, which step_order='brian2' does not realize (it reads e` |
| `ValueError` | `intra_step_rounds={...} needs the delay-0 edge arrays (edge0_pre, edge0_post, edge0_weight_nS, edge0` |
| `ValueError` | `delay-0 edge arrays (edge0_pre, edge0_post, edge0_weight_nS, edge0_receptor) need intra_step_rounds;` |
| `ValueError` | `ext_lambda schedule with one shared rate must have shape ({...},); got {...}` |
| `ValueError` | `ext_lambda schedule with per-target rates must have shape ({...}, {...}); got {...}` |
| `ValueError` | `I_ext_pA must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `V0_mV must be None or shape ({...},); got {...}` |
| `ValueError` | `intra-step fixed point reached its cap intra_step_rounds={...} with {...} undelivered same-step spik` |
| `ValueError` | `step_order must be 'default' or 'brian2', got {...}` |
| `ValueError` | `decay must be 'exp' or 'euler', got {...}` |
| `ValueError` | `n_steps must be a positive integer, got {...}` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `N * n_receptors must be < 2**31 (int32 segment ids); N = {...}` |
| `ValueError` | `receptor_reversal_mV and receptor_tau_ms must have the same length; got {...} and {...}` |
| `ValueError` | `receptor_tau_ms must be > 0 for every receptor; got {...}` |
| `ValueError` | `decay='euler' needs receptor_tau_ms > dt_ms ({...}); got {...}` |
| `ValueError` | `ext_t_start_steps/ext_t_stop_steps need the external drive; got {...}/{...} with no ext_targets — pa` |
| `ValueError` | `ext_delay_steps=0 requests a same-step drive, which the registered-HDP kernel does not realize: use ` |
| `ValueError` | `plasticity_mask must have shape ({...},), got {...}` |
| `ValueError` | `plasticity_mask must be finite (got NaN/inf)` |
| `ValueError` | `ext_lambda schedule with one shared rate must have shape ({...},); got {...}` |
| `ValueError` | `ext_lambda schedule with per-target rates must have shape ({...}, {...}); got {...}` |
| `ValueError` | `I_ext_pA must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `V0_mV must be None or shape ({...},); got {...}` |
| `ValueError` | `hdp_key must be a JAX PRNG key; got {...}` |
| `ValueError` | `seed must be an int or None; got {...}` |

**`jaxfne/emitters_hh.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} must be finite and > 0; got {...}` |
| `ValueError` | `{...} must be finite and > 0; got {...}` |
| `ValueError` | `gate_init must be a finite number or None; got {...}` |
| `ValueError` | `gate_init must be a finite number or None; got {...}` |
| `ValueError` | `unknown HH cell type {...}; expected one of {...}` |
| `ValueError` | `overrides name unknown HH cell type {...}; expected one of {...}` |
| `ValueError` | `overrides name cell type {...}, which is not in labels {...}` |
| `ValueError` | `unknown HH parameter field {...} for cell type {...}; expected one of {...}` |
| `ValueError` | `edge arrays must be all given or all None; missing {...}` |
| `ValueError` | `{...} must be one-dimensional, got shape {...}` |
| `ValueError` | `edge arrays must have equal lengths; got {...}` |
| `ValueError` | `{...} must hold integers` |
| `ValueError` | `edge_pre must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge_post must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge_receptor must index receptors in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge_weight must be finite; got {...}` |
| `ValueError` | `edge_weight must be >= 0 (the reversal sets the sign); got min {...}` |
| `ValueError` | `morphology must be an HHMorphology; got {...}` |
| `ValueError` | `morphology ra_ohmcm must be a number; got {...}` |
| `ValueError` | `morphology parents, length_um, radius_um and area_um2 must have equal lengths; got {...}, {...}, {..` |
| `ValueError` | `morphology must hold at least one compartment; got {...}` |
| `ValueError` | `morphology length_um must be finite and > 0; got {...}` |
| `ValueError` | `morphology radius_um must be finite and > 0; got {...}` |
| `ValueError` | `morphology ra_ohmcm must be finite and > 0; got {...}` |
| `ValueError` | `morphology area_um2 must be finite and > 0; got {...}` |
| `ValueError` | `morphology parents must hold integers; got {...}` |
| `ValueError` | `morphology parents must hold exactly one root (-1); got {...}` |
| `ValueError` | `morphology parents must index compartments in [0, {...}) or be -1; got {...}` |
| `ValueError` | `morphology parents must not self-parent; got {...}` |
| `ValueError` | `morphology parents must be acyclic; got {...}` |
| `ValueError` | `morphology yields non-finite axial conductances; got {...}` |
| `ValueError` | `{...} must be a scalar or shape ({...},); got shape {...}` |
| `ValueError` | `{...} must hold integers; got {...}` |
| `ValueError` | `{...} must index compartments in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `n_steps must be a positive integer, got {...}` |
| `ValueError` | `receptor_e_rev, receptor_tau_r, receptor_tau_d, receptor_slope and receptor_vth must have the same l` |
| `ValueError` | `receptor_tau_r must be > 0 for every receptor; got {...}` |
| `ValueError` | `receptor_tau_d must be > 0 for every receptor; got {...}` |
| `ValueError` | `receptor_slope must be > 0 for every receptor; got {...}` |
| `ValueError` | `receptor_s0 must be a number; got {...}` |
| `ValueError` | `receptor_s0 must be finite; got {...}` |
| `ValueError` | `ou_sigma must be None or shape ({...},); got {...}` |
| `ValueError` | `ou_sigma must be finite; got {...}` |
| `ValueError` | `noise_gain must be a number; got {...}` |
| `ValueError` | `noise_gain must be finite and >= 0; got {...}` |
| `ValueError` | `ou_tau is required when OU noise is live (ou_key given and ou_sigma * noise_gain nonzero); got None ` |
| `ValueError` | `ou_tau must be None or shape ({...},); got {...}` |
| `ValueError` | `ou_tau must be finite and > 0; got {...}` |
| `ValueError` | `ou_key is required when ou_sigma * noise_gain is nonzero` |
| `ValueError` | `ou_key must be a JAX PRNG key; got {...}` |
| `ValueError` | `I_ext must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `V0_mV must be None or shape ({...},); got {...}` |
| `ValueError` | `morphology is required when any compartment selection or point-process input is given` |
| `ValueError` | `receptor_e_rev, receptor_tau_r, receptor_tau_d, receptor_slope and receptor_vth must have the same l` |
| `ValueError` | `receptor_tau_r must be > 0 for every receptor; got {...}` |
| `ValueError` | `receptor_tau_d must be > 0 for every receptor; got {...}` |
| `ValueError` | `receptor_slope must be > 0 for every receptor; got {...}` |
| `ValueError` | `receptor_s0 must be a number; got {...}` |
| `ValueError` | `receptor_s0 must be finite; got {...}` |
| `ValueError` | `edge_weight_uS and edge_weight are alternative input forms; got both` |
| `ValueError` | `edge arrays must be all given or all None; missing {...}` |
| `ValueError` | `edge_weight_uS must be one-dimensional, got shape {...}` |
| `ValueError` | `edge_weight_uS must be finite; got {...}` |
| `ValueError` | `edge_weight_uS must be >= 0 (the reversal sets the sign); got min {...}` |
| `ValueError` | `ou_sigma must be None or shape ({...},); got {...}` |
| `ValueError` | `ou_sigma must be finite; got {...}` |
| `ValueError` | `noise_gain must be a number; got {...}` |
| `ValueError` | `noise_gain must be finite and >= 0; got {...}` |
| `ValueError` | `ou_tau is required when OU noise is live (ou_key given and ou_sigma * noise_gain nonzero); got None` |
| `ValueError` | `ou_tau must be None or shape ({...},); got {...}` |
| `ValueError` | `ou_tau must be finite and > 0; got {...}` |
| `ValueError` | `ou_key is required when ou_sigma * noise_gain is nonzero` |
| `ValueError` | `ou_key must be a JAX PRNG key; got {...}` |
| `ValueError` | `I_ext and I_ext_nA are alternative input forms; got both` |
| `ValueError` | `I_ext_nA must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `I_ext must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `V0_mV must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `n_steps must be a positive integer, got {...}` |
| `ValueError` | `batch must be a non-empty mapping of batch key to stacked values; got {...}` |
| `ValueError` | `unknown batch keys {...}; expected a subset of {...}` |
| `ValueError` | `batch keys {...} are also given as shared arguments; pass each input once` |
| `ValueError` | `batch[{...}] must stack B elements along a leading axis; got shape {...}` |
| `ValueError` | `batch must hold at least one element; got B={...}` |
| `ValueError` | `batch keys must share one leading size B; {...} has {...}, expected {...}` |
| `ValueError` | `edge_weight_uS and edge_weight are alternative input forms; got both` |
| `ValueError` | `I_ext and I_ext_nA are alternative input forms; got both` |
| `ValueError` | `morphology is required when any compartment selection or point-process input is given` |
| `ValueError` | `batch['edge_weight_uS'] must have shape (B, E); got {...}` |
| `ValueError` | `edge arrays must be all given or all None; missing {...}` |
| `ValueError` | `edge_weight_uS must be finite; got {...}` |
| `ValueError` | `edge_weight_uS must be >= 0 (the reversal sets the sign); got min {...}` |
| `ValueError` | `edge_weight_uS must be one-dimensional, got shape {...}` |
| `ValueError` | `I_ext_nA must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `I_ext_nA must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `noise_gain must be a number; got {...}` |
| `ValueError` | `edge_weight must be finite; got {...}` |
| `ValueError` | `edge_weight must be >= 0 (the reversal sets the sign); got min {...}` |
| `ValueError` | `ou_key must be a JAX PRNG key; got {...}` |
| `ValueError` | `noise_gain must be a number; got {...}` |
| `ValueError` | `noise_gain must be finite and >= 0; got {...}` |
| `ValueError` | `ou_tau is required when OU noise is live (ou_key given and ou_sigma * noise_gain nonzero); got None ` |
| `ValueError` | `ou_key is required when ou_sigma * noise_gain is nonzero` |

**`jaxfne/emitters_homeostatic_ei.py`**

| Class | Message |
|---|---|
| `ValueError` | `hebbian_pairwise requires is_e (a population mask); simulate_homeostatic_ei always passes it, so thi` |
| `ValueError` | `unmeasured conductance x homeostasis pairing ({...} x {...}): no measured outcome exists for this co` |
| `ValueError` | `unknown {...} rule {...}; expected one of {...} or a callable` |
| `ValueError` | `n must be at least 2 (need both an E and an I neuron), got {...}` |
| `ValueError` | `unknown bound_mode {...}; expected 'minimal' or 'stable'` |

**`jaxfne/emitters_lif.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} t_ref_ms must be a finite number >= 0 in ms; got {...}` |
| `ValueError` | `{...} t_ref_ms must be a finite number >= 0 in ms; got {...}` |
| `ValueError` | `{...} tau_syn_ms must be a finite number > 0 in ms; got {...}` |
| `ValueError` | `{...} tau_syn_ms must be a finite number > 0 in ms; got {...}` |
| `ValueError` | `unknown LIF cell type {...}; expected one of {...}` |
| `ValueError` | `overrides name unknown LIF cell type {...}; expected one of {...}` |
| `ValueError` | `overrides name cell type {...}, which is not in labels {...}` |
| `ValueError` | `unknown LIF parameter field {...} for cell type {...}; expected one of {...}` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `cap_groups is set but cap_rounds is None; groups only scope a declared local-inhibition cap` |
| `ValueError` | `local_inhibition order must be one of {...}; got {...}` |
| `ValueError` | `cap_groups must have shape ({...},); got {...}` |
| `ValueError` | `unknown LIF sign label {...}; expected one of {...}` |
| `ValueError` | `dtype='float64' requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x64", Tru` |
| `ValueError` | `labels must contain at least one label` |
| `ValueError` | `layer_labels length must match labels length` |
| `ValueError` | `drive_overrides[{...}] must be finite; got {...}` |
| `ValueError` | `lif_overrides[{...}] must be a mapping of {...}; got {...}` |
| `ValueError` | `lif_overrides names {...}, which matches no built neuron {...} and no PCL cell type {...}` |
| `ValueError` | `unknown LIF parameter field {...} for {...}; expected one of {...}` |
| `ValueError` | `edge arrays must be all given or all None; missing {...}` |
| `ValueError` | `{...} must be one-dimensional, got shape {...}` |
| `ValueError` | `edge arrays must have equal lengths; got {...}` |
| `ValueError` | `{...} must hold integers` |
| `ValueError` | `edge_pre must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge_post must index cells in [0, {...}); got min {...}, max {...}` |
| `ValueError` | `edge_delay_steps must be >= 1; got min {...}` |
| `ValueError` | `edge_weight must be finite` |
| `ValueError` | `local_inhibition_blocks must carry 'members', 'member_mask' and 'weights'; got {...}` |
| `ValueError` | `local_inhibition_blocks members/mask must share one (G, D) shape; got {...} and {...}` |
| `ValueError` | `local_inhibition_blocks weights must have shape ({...}, {...}, {...}); got {...}` |
| `ValueError` | `local_inhibition_blocks weights must be finite` |
| `ValueError` | `local_inhibition_blocks weights must hold non-negative inhibition magnitudes (the cascade subtracts ` |
| `ValueError` | `local_inhibition_blocks member ids must lie in [0, {...})` |
| `ValueError` | `local_inhibition_blocks position {...} is empty; every position holds at least its cell` |
| `ValueError` | `local_inhibition_blocks position {...} spans several groups; local inhibition stays inside one retin` |
| `ValueError` | `local_inhibition_blocks repeats group {...}; positions and groups agree one to one` |
| `ValueError` | `local_inhibition_blocks covers a cell twice; every cell is visited at exactly one position` |
| `ValueError` | `local_inhibition_blocks must cover every grouped cell exactly once; {...} of {...} are uncovered` |
| `ValueError` | `local_inhibition_resolution='event_order' needs local_inhibition_blocks; without inhibition magnitud` |
| `ValueError` | `local_inhibition_resolution='event_order' needs local_inhibition_visit_order; without the per-step v` |
| `ValueError` | `local_inhibition_visit_order must have shape ({...}, {...}, {...}); got {...}` |
| `ValueError` | `local_inhibition_visit_order must hold integer local ids; got {...}` |
| `ValueError` | `local_inhibition_visit_order pads non-member slots with -1; position {...} holds another id` |
| `ValueError` | `local_inhibition_visit_order must visit every member local id exactly once per step; position {...} ` |
| `ValueError` | `event_scan_tables is missing {...}; the scan needs the full cover/forward/lateral/top-down tables ({` |
| `ValueError` | `event_scan_tables[{...}] must be an integer; got {...}` |
| `ValueError` | `simulate_lif_edge event_scan_w_update: local-inhibition position {...} mixes simple and complex cell` |
| `ValueError` | `simulate_lif_edge event_scan_w_update: local_inhibition_blocks disagree with the declared {...} at p` |
| `ValueError` | `event_scan cover tables name a cell outside the ns simple cells; the declared buffer plasticity need` |
| `ValueError` | `{...} must hold exactly the event-scan state fields; unknown {...}, missing {...}` |
| `ValueError` | `{...} with event_scan_w_update needs event_scan_boundary: the learning buffers index this call's cov` |
| `ValueError` | `{...} disagrees with h_update['last_update_us']: the state holds {...}, h_update says {...}` |
| `ValueError` | `{...} disagrees with h_update['rate0_hz']` |
| `ValueError` | `{...} disagrees with h_update['threshold0_mV']` |
| `ValueError` | `{...} disagrees with w_update['wi_s']` |
| `ValueError` | `{...} disagrees with w_update['wi_c']` |
| `ValueError` | `{...} disagrees with event_scan_tables['w_s']` |
| `ValueError` | `{...} disagrees with event_scan_tables['w_c']` |
| `ValueError` | `{...} disagrees with event_scan_tables['lat_w']` |
| `ValueError` | `{...} disagrees with event_scan_tables['td_w']` |
| `ValueError` | `{...} disagrees with local_inhibition_blocks` |
| `ValueError` | `local_inhibition_groups for resolution 'event_scan' must hold integer ids of shape ({...},); got sha` |
| `ValueError` | `event_scan_tables must be the mapping event_scan_tables builds (cover/forward/lateral/top-down); got` |
| `ValueError` | `event_scan_tables[{...}] must be >= 1; got {...}` |
| `ValueError` | `event_scan_tables geometry (nx, ny, nd, cx, cy, cd, ns, nc) disagrees with the ({...},) population` |
| `ValueError` | `event_scan_tables describes a simple/complex two-layer net; params must carry labels ({...})` |
| `ValueError` | `event_scan_tables describes a simple/complex two-layer net with {...} simple then {...} complex cell` |
| `ValueError` | `params {...} for resolution 'event_scan' must have shape ({...},); got {...}` |
| `ValueError` | `params tau_m_ms for resolution 'event_scan' must be finite and > 0` |
| `ValueError` | `params tau_rp_ms for resolution 'event_scan' must be finite and > 0` |
| `ValueError` | `params tau_sra_ms for resolution 'event_scan' must be finite and > 0 (or +inf)` |
| `ValueError` | `params {...} for resolution 'event_scan' must be finite` |
| `ValueError` | `params delta_sra_mV for resolution 'event_scan' must be 0 on complex cells; the oracle carries no ad` |
| `ValueError` | `event_scan_tables must carry the cover/forward/lateral/top-down arrays event_scan_tables builds ({..` |
| `ValueError` | `event_scan_tables['w_s'] must have shape ({...}, 2, sx, sy); got {...}` |
| `ValueError` | `event_scan_tables['w_s'] must be finite` |
| `ValueError` | `event_scan_tables['w_c'] must have shape ({...}, csx, csy, {...}); got {...}` |
| `ValueError` | `event_scan_tables['w_c'] must be finite` |
| `ValueError` | `event_scan_tables[{...}] must be a bool; got {...}` |
| `ValueError` | `event_scan_tables['pix_pos'] must have one row per pixel ({...}); got {...}` |
| `ValueError` | `event_scan_tables['pix_pos'] must hold integers` |
| `ValueError` | `event_scan_tables['pix_n'] must hold one integer count per pixel; got shape {...} dtype {...}` |
| `ValueError` | `event_scan_tables['pix_n'] must lie in [0, cover width]` |
| `ValueError` | `event_scan_tables['pix_pos'] must name simple positions in [0, {...})` |
| `ValueError` | `event_scan_tables['pix_pos'] rows must list covered positions ascending (the oracle visits candidate` |
| `ValueError` | `event_scan_tables['fwd_idx'] must hold integer targets with one row per simple cell ({...}); got {..` |
| `ValueError` | `event_scan_tables['fwd_w'] must be finite with shape ({...}, {...})` |
| `ValueError` | `event_scan_tables['fwd_n'] must hold one integer count per simple cell` |
| `ValueError` | `event_scan_tables['fwd_n'] must lie in [0, forward width]` |
| `ValueError` | `event_scan_tables['fwd_idx'] must name cells in [0, {...})` |
| `ValueError` | `event_scan_tables['fwd_idx'] must name complex cells (simple cells end at {...}); the forward pass o` |
| `ValueError` | `event_scan_tables['fwd_key'] must hold integers (cz, ex, ey) with shape ({...}, {...}, 3); got {...}` |
| `ValueError` | `event_scan_tables['fwd_key'] must name shared complex weights in [0, {...}) x [0, {...}) x [0, {...}` |
| `ValueError` | `event_scan_tables['fwd_w'] must equal w_c[cz, ex, ey, n % nd] at every used forward slot (fwd_key na` |
| `ValueError` | `event_scan_tables['lat_idx'] must hold integer targets with one row per simple cell ({...}); got {..` |
| `ValueError` | `event_scan_tables['lat_w'] must be None or finite with shape ({...}, {...})` |
| `ValueError` | `event_scan_tables['lat_n'] must hold one integer count per simple cell` |
| `ValueError` | `event_scan_tables['lat_n'] must lie in [0, lateral width]` |
| `ValueError` | `event_scan_tables['lat_idx'] must name simple cells in [0, {...})` |
| `ValueError` | `event_scan_tables['td_idx'] must hold integer targets with one row per complex cell ({...}); got {..` |
| `ValueError` | `event_scan_tables['td_w'] must be None or finite with shape ({...}, {...})` |
| `ValueError` | `event_scan_tables['td_n'] must hold one integer count per complex cell` |
| `ValueError` | `event_scan_tables['td_n'] must lie in [0, top-down width]` |
| `ValueError` | `event_scan_tables['td_idx'] must name simple cells in [0, {...})` |
| `ValueError` | `event_scan_tables['s_ox'/'s_oy'] must hold the simple grid origins (({...},) and ({...},))` |
| `ValueError` | `event_scan_events must have shape (M, 4) with [t_us, x, y, p] rows; got {...}` |
| `ValueError` | `event_scan_events must hold integers; got {...}` |
| `ValueError` | `event_scan_events must be stably sorted by time (event order, then ascending index on A)` |
| `ValueError` | `event_scan_events x must lie on the {...}-pixel plane` |
| `ValueError` | `event_scan_events y must lie on the {...}-pixel plane` |
| `ValueError` | `event_scan_events polarity must lie in {0, 1}` |
| `ValueError` | `event_scan_events times must fall on the [{...}, {...}) microsecond grid` |
| `ValueError` | `V0_mV must be None or shape ({...},); got {...}` |
| `ValueError` | `V0_mV must be finite (got NaN/inf)` |
| `ValueError` | `V0_mV is set with event_scan_state; the state carries the potentials` |
| `ValueError` | `event_scan_boundary is set without event_scan_state; the map resets a state` |
| `ValueError` | `local_inhibition_rounds is set but resolution is 'event_scan'; the replay resolves every event, no c` |
| `ValueError` | `local_inhibition_order={...} with resolution 'event_scan' is refused; the scan visits candidates in ` |
| `ValueError` | `local_inhibition_visit_order is set but resolution is 'event_scan'; the replay walks the event strea` |
| `ValueError` | `local_inhibition_resolution='event_scan' needs local_inhibition_groups; without positions there is n` |
| `ValueError` | `local_inhibition_resolution='event_scan' needs local_inhibition_blocks; without inhibition magnitude` |
| `ValueError` | `local_inhibition_resolution='event_scan' needs event_scan_events; without the timestamped stream the` |
| `ValueError` | `local_inhibition_resolution='event_scan' needs event_scan_tables; without the cover/forward/lateral/` |
| `ValueError` | `feed_current is set but resolution is 'event_scan'; the replay feeds from event_scan_events, not the` |
| `ValueError` | `feed_arrivals is set but resolution is 'event_scan'; the replay feeds from event_scan_events, not th` |
| `ValueError` | `{...} is set but resolution is 'event_scan'; the replay drives simple cells from the event cover, no` |
| `ValueError` | `edge_group is set but resolution is 'event_scan'; the replay runs no edges` |
| `ValueError` | `I_ext_pA is set but resolution is 'event_scan'; the replay carries no sustained drive` |
| `ValueError` | `record_v is set but resolution is 'event_scan'; the replay returns final V/A only` |
| `ValueError` | `hdp_rule/hdp_rule_params is set but resolution is 'event_scan'; the replay runs frozen weights` |
| `ValueError` | `plasticity_mask is set but resolution is 'event_scan'; the replay runs frozen weights` |
| `ValueError` | `H0 is set but resolution is 'event_scan'; the replay carries no rule state` |
| `ValueError` | `tau_syn_ms is set but resolution is 'event_scan'; the replay excites cells event by event with exact` |
| `ValueError` | `dt_ms must resolve to whole microseconds under resolution 'event_scan'; got {...}` |
| `ValueError` | `t0_ms must resolve to whole microseconds under resolution 'event_scan'; got {...}` |
| `ValueError` | `n_steps must be a positive integer, got {...}` |
| `ValueError` | `dt_ms must be > 0, got {...}` |
| `ValueError` | `t0_ms must be a number, got {...}` |
| `ValueError` | `t0_ms must be finite, got {...}` |
| `ValueError` | `params tau_syn_ms must have shape ({...},); got {...}` |
| `ValueError` | `params tau_syn_ms must be finite and >= 0 (0 runs the delta path)` |
| `ValueError` | `hdp_rule_params is set but hdp_rule is None; pass a rule name` |
| `ValueError` | `plasticity_mask must have shape ({...},), got {...}` |
| `ValueError` | `plasticity_mask must be finite (got NaN/inf)` |
| `ValueError` | `edge_group must have shape ({...},); got {...}` |
| `ValueError` | `edge_group ids must be >= 0` |
| `ValueError` | `local_inhibition_order must be one of {...}; got {...}` |
| `ValueError` | `local_inhibition_groups must have shape ({...},); got {...}` |
| `ValueError` | `local_inhibition_resolution must be one of {...}; got {...}` |
| `ValueError` | `event_scan_h_update is set but resolution is {...}; the periodic rate and threshold update only runs` |
| `ValueError` | `event_scan_state/event_scan_boundary/event_scan_return_state are set but resolution is {...}; the co` |
| `ValueError` | `event_scan_w_update is set but resolution is {...}; the buffer plasticity only runs under local_inhi` |
| `ValueError` | `event_scan_events/event_scan_tables are set but resolution is {...}; the event replay only runs unde` |
| `ValueError` | `local_inhibition_resolution is set but local_inhibition_rounds is None; the resolution only resolves` |
| `ValueError` | `local_inhibition_order={...} with resolution 'event_order' is refused; the visit order already encod` |
| `ValueError` | `local_inhibition_resolution='event_order' needs local_inhibition_groups; without positions there is ` |
| `ValueError` | `local_inhibition_blocks is set but resolution is 'whole_step'; the cap resolves without per-position` |
| `ValueError` | `local_inhibition_visit_order is set but resolution is 'whole_step'; the cap resolves in the declared` |
| `ValueError` | `I_ext_pA must be None, shape ({...},) or ({...}, {...}); got {...}` |
| `ValueError` | `V0_mV must be None or shape ({...},); got {...}` |
| `ValueError` | `feed_current must be None or shape ({...}, {...}); got {...}` |
| `ValueError` | `feed_current must be finite (got NaN/inf)` |
| `ValueError` | `feed_arrivals must be None or shape ({...}, {...}); got {...}` |
| `ValueError` | `feed_arrivals must be finite (got NaN/inf)` |
| `ValueError` | `feed_arrivals must hold non-negative event counts` |
| `ValueError` | `feed_arrivals must hold whole event counts` |
| `ValueError` | `H0 is set but hdp_rule is None; H seeds the rule's carry only` |
| `ValueError` | `H0 must be None or shape ({...},); got {...}` |
| `ValueError` | `H0 must be finite (got NaN/inf)` |
| `ValueError` | `key must be a JAX PRNG key; got {...}` |
| `ValueError` | `hdp_rule {...} returned undeclared theta_targets {...}; declared: {...}` |

**`jaxfne/emitters_rate.py`**

| Class | Message |
|---|---|
| `ValueError` | `rate emitter param {...}={...} is not realized: it takes a finite real number (bools refused, never ` |
| `ValueError` | `rate emitter param {...}={...} is not realized: it takes a finite real number` |
| `ValueError` | `rate emitter param {...}={...} is not realized: it takes a value >= 0` |
| `ValueError` | `rate emitter params {...} are not realized: they take a mapping of {...}` |
| `ValueError` | `rate emitter does not consume params {...}; consumed: {...}` |
| `ValueError` | `rate emitter param transfer={...} is not realized: expected one of {...}` |
| `ValueError` | `rate emitter param tau_ms={...} is not realized: it takes a value > 0` |
| `ValueError` | `rate transfer {...} is not realized: expected one of {...}` |
| `ValueError` | `rate gain {...} is not realized: it takes finite values >= 0` |
| `ValueError` | `make_rate_params needs at least one neuron label` |
| `ValueError` | `simulate_rate_edge needs n_steps >= 1; got {...}` |
| `ValueError` | `simulate_rate_edge needs a finite dt_ms > 0; got {...}` |
| `ValueError` | `simulate_rate_edge rate columns disagree with the population: {...}, {...}, {...}, {...} for N={...}` |
| `ValueError` | `simulate_rate_edge needs tau_ms > 0 on every unit` |
| `ValueError` | `simulate_rate_edge drive has shape {...} for N={...}` |
| `ValueError` | `simulate_rate_edge edge arrays disagree: {...} vs {...} vs {...} vs {...}` |
| `ValueError` | `simulate_rate_edge edge endpoints leave [0, N)` |
| `ValueError` | `simulate_rate_edge needs edge_delay_steps >= 1 on every edge; zero-step edges floor to 1 in jaxfne._` |
| `ValueError` | `simulate_rate_edge poisson has shape {...}; expected ({...}, {...})` |
| `ValueError` | `simulate_rate_edge poisson must stay finite` |
| `ValueError` | `simulate_rate_edge needs finite drive and r0` |
| `ValueError` | `simulate_rate_edge needs one transfer per neuron from {...}` |
| `ValueError` | `simulate_rate_edge went non-finite at step {...}: drive, gain and sigma stay finite on the tensor ro` |

**`jaxfne/experiment_a/canonical.py`**

| Class | Message |
|---|---|
| `ValueError` | `canonical {...} hash mismatch: disk={...} receipt={...}` |

**`jaxfne/experimental_hpc/contracts.py`**

| Class | Message |
|---|---|
| `ValueError` | `runtime_static.n_steps must be positive` |
| `ValueError` | `runtime_static.dt_ms must be positive` |
| `ValueError` | `runtime_static.dtype must be 'float32' or 'float64'` |
| `ValueError` | `SelectorSpec matched no neurons: {...}` |

**`jaxfne/fields/diagnostics.py`**

| Class | Message |
|---|---|
| `ValueError` | `execution_form must be 'fused' or 'materialized', got {...}` |

**`jaxfne/fields/head.py`**

| Class | Message |
|---|---|
| `ValueError` | `source [T, N] and positions [N, 3] required, got {...} and {...}` |
| `ValueError` | `axis must be 0, 1 or 2, got {...}` |
| `ValueError` | `signals carry no sources; simulate with record_sources=True` |
| `ValueError` | `model must carry params['positions'] (a constructed jaxfne Model)` |
| `ValueError` | `source [T, N] and positions [N, 3] required, got {...} and {...}` |
| `ValueError` | `axis must be 0, 1 or 2, got {...}` |
| `ValueError` | `conductivity must be > 0, got {...}` |
| `ValueError` | `min_distance must be > 0, got {...}` |
| `ValueError` | `dendrite_length must be a finite scalar or [{...}], got shape {...}` |
| `ValueError` | `n_columns must be >= 1, got {...}` |
| `ValueError` | `need radius_m > 0 and 0 <= fold_amplitude_m < radius_m` |
| `ValueError` | `max_polar_deg must be in (0, 180], got {...}` |
| `ValueError` | `degenerate surface normal; change n_columns or the fold parameters` |
| `ValueError` | `realizations must be [K, T], got {...}` |
| `ValueError` | `n_columns must be >= 1, got {...}` |
| `ValueError` | `in_phase_realization must index one of {...} realizations` |
| `ValueError` | `in_phase_realization needs an in_phase mask` |
| `ValueError` | `in_phase must be [{...}], got {...}` |
| `ValueError` | `need n >= 1 and radius_m > 0` |
| `ValueError` | `dipole_pos and dipole_ori must both be [M, 3], got {...} and {...}` |
| `ValueError` | `sensor_pos and sensor_ori must both be [S, 3], got {...} and {...}` |
| `ValueError` | `every sensor must lie farther from the centre than every dipole` |
| `ValueError` | `conductivity_s_per_m must be > 0, got {...}` |
| `ValueError` | `electrode_pos must be [E, 3], got {...}` |
| `ValueError` | `electrodes must lie on one sphere centred at the origin` |
| `ValueError` | `every dipole must lie inside the electrode sphere` |
| `ValueError` | `modality must be 'meg' or 'eeg', got {...}` |
| `ValueError` | `W must be [n_sensors, n_dipoles]` |
| `ValueError` | `dipoles must be [T, {...}], got {...}` |

**`jaxfne/fields/probes.py`**

| Class | Message |
|---|---|
| `ValueError` | `CanonicalSource Q is relative-only (representation={...}); physical units require an explicit calibr` |
| `ValueError` | `CanonicalSource Q must be 2D [T, N]; got shape {...}` |
| `ValueError` | `CanonicalSource Q must be finite.` |
| `ValueError` | `{...}({...}={...}) is not realized: the probe samples no position and applies no reference arithmeti` |
| `ValueError` | `Probe input Q must carry representation {...}; got {...}.` |
| `EpistemicRefusal` | `CalibrationTransform {...} must be finite; got {...}.` |
| `EpistemicRefusal` | `CalibrationTransform {...} must be positive; got {...}.` |
| `EpistemicRefusal` | `CalibrationTransform {...} must be non-negative; got {...}.` |
| `EpistemicRefusal` | `CalibrationTransform target_level must be CALIBRATED or REDUCED_PHYSICAL; got {...}.` |
| `EpistemicRefusal` | `CalibrationTransform requires declared units.` |
| `EpistemicRefusal` | `CalibrationTransform requires declared {...}.` |
| `EpistemicRefusal` | `AppliedCalibration cannot be constructed directly: relabel a proxy only via apply_calibration() with` |
| `EpistemicRefusal` | `AppliedCalibration requires a CalibrationTransform.` |
| `EpistemicRefusal` | `AppliedCalibration level {...} != transform target {...}.` |
| `EpistemicRefusal` | `{...} epistemic_level must be one of {...}; got {...}.` |
| `EpistemicRefusal` | `{...} at RELATIVE_PROXY must not carry a calibration record.` |
| `EpistemicRefusal` | `{...} may become {...} only via apply_calibration() with an explicit CalibrationTransform declaring ` |
| `EpistemicRefusal` | `Proxy relabel refused: authority must be an explicit CalibrationTransform, got {...}.` |
| `EpistemicRefusal` | `Proxy relabel refused: {...} carries no epistemic level.` |
| `EpistemicRefusal` | `Proxy relabel refused: already {...}; re-calibration is not silent.` |
| `EpistemicRefusal` | `Proxy relabel refused: cannot seal {...}: {...}` |
| `ValueError` | `phi_e must be 2D (T, C); got shape {...}` |
| `ValueError` | `field_contact_depths length {...} != phi_e channels {...}` |
| `ValueError` | `lfp_proxy_probe refuses to invent field contacts: contact_depths was given without field_contact_dep` |
| `ValueError` | `{...} must be 2D [T, K], got shape {...}` |
| `ValueError` | `leadfield must be 2D [C, K], got shape {...}` |
| `ValueError` | `{...} and leadfield K dimension mismatch: {...} vs {...}` |
| `ValueError` | `spike_rate must be 1D or 2D, got shape {...}` |
| `ValueError` | `source must be 2D [T, K], got shape {...}` |
| `ValueError` | `field_potential must be 2D [T, X], got shape {...}` |
| `ValueError` | `Time dimension mismatch: spike_rate={...}, source={...}, field={...}` |

**`jaxfne/fields/proxy.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...}(dtype={...}) is not realized: use 'float32' or 'float64'` |
| `ValueError` | `{...}(dtype='float64') requested but JAX x64 is not enabled -- call jax.config.update("jax_enable_x6` |
| `ValueError` | `Invalid projection mode: {...}. Must be 'row_normalize' or 'density_preserving'.` |
| `ValueError` | `project_model_sources needs model.params['positions']; got {...} without it` |
| `ValueError` | `sources must have shape (T, N); got {...}` |
| `ValueError` | `sources carry N = {...} but model.params['positions'] carries N = {...}; refusing model-unmatched pr` |
| `ValueError` | `explicit positions carry N = {...} but model carries N = {...}; refusing model-unmatched projection` |
| `ValueError` | `positions must have shape (N, 3); got {...}` |
| `ValueError` | `non-finite diagnostic value: {...}` |
| `ValueError` | `compute_conservation_proxy_diagnostics got a non-finite {...} array; non-finite inputs are invalid (` |
| `ValueError` | `filtered_spike_source: cell types {...} have no sign; pass cell_signs= with a sign for each (known: ` |
| `ValueError` | `dt_ms must be finite and > 0; got {...}` |
| `ValueError` | `spikes must be 2D [T, N]; got shape {...}` |
| `ValueError` | `spikes must be non-empty; got shape {...}` |
| `ValueError` | `spikes must be finite` |
| `ValueError` | `dt_ms must be finite and > 0; got {...}` |
| `ValueError` | `x and y must have equal length; got {...} vs {...}` |
| `ValueError` | `need at least 8 samples for spectra; got {...}` |
| `ValueError` | `x and y must be finite` |
| `ValueError` | `cross_area_coherence needs T >= 288 samples for 8 Welch segments of at least 64 samples; got T = {..` |
| `ValueError` | `cross_area_coherence needs the Welch estimate; it failed ({...}: {...}), and a single periodogram ca` |
| `ValueError` | `W must be 2D [C, N], got {...}` |
| `ValueError` | `source length {...} does not match W width {...}` |
| `ValueError` | `source width {...} does not match W width {...}` |
| `ValueError` | `source must be 1D or 2D, got {...}` |
| `ValueError` | `cell_type length {...} does not match depth_z length {...}` |
| `ValueError` | `sources must be 2D [T, N], got shape {...}` |
| `ValueError` | `tau_s length {...} does not match sources width {...}` |
| `ValueError` | `order must be a positive integer, got {...}` |
| `ValueError` | `tau_s must be strictly positive, got range [{...}, {...}]` |
| `ValueError` | `total_membrane_current is required for total_membrane_current_proxy` |
| `ValueError` | `Double-counting detected: total_membrane_current_proxy already includes synaptic_current` |
| `ValueError` | `decomposed_cap_ion_plus_synaptic_proxy requires both decomposed_cap_ion and synaptic_current` |
| `ValueError` | `spike_proxy is required for spike_proxy` |
| `ValueError` | `Double-counting detected: total_membrane_current_proxy + synaptic_current_proxy` |
| `ValueError` | `Unsupported construct_source_tensor mode {...}. Valid modes: 'total_membrane_current_proxy', 'decomp` |

**`jaxfne/fields/solvers.py`**

| Class | Message |
|---|---|
| `ValueError` | `precision must be 'float32' or 'float64', got {...}` |
| `ValueError` | `experimental_poisson_1d(precision='float64') requires x64 to already be enabled -- call jax.config.u` |
| `ValueError` | `conductivity array must have shape (N-1,)={...}, got {...}` |
| `ValueError` | `experimental_poisson_1d_from_neuron_table requires every neuron_table row to have a numeric 'z' -- g` |
| `ValueError` | `sources length {...} must match neuron_table length {...}` |
| `ValueError` | `z_max ({...}) must be greater than z_min ({...})` |

**`jaxfne/h3_decodability.py`**

| Class | Message |
|---|---|
| `ValueError` | `train and test seeds must be disjoint; overlap={...}` |
| `ValueError` | `perturbed_index must be in [0, {...}); got {...}` |
| `ValueError` | `perturbation_step must be in [0, {...}); got {...}` |
| `ValueError` | `unknown decoder observation kind {...}` |

**`jaxfne/h4_matrix.py`**

| Class | Message |
|---|---|
| `ValueError` | `ring requires n_neurons >= 2` |
| `ValueError` | `delay_steps must have shape ({...},)` |
| `ValueError` | `ring requires n_neurons >= 2` |
| `ValueError` | `k_neighbors must be in [1, {...})` |
| `ValueError` | `arc_delay_steps must have shape ({...},)` |

**`jaxfne/hdp_network.py`**

| Class | Message |
|---|---|
| `ValueError` | `apply_drive_correction(mode='absolute') requires cfg.base_drive_by_cell_type to be an explicit dict,` |
| `ValueError` | `Unknown mode: {...}. Choose 'multiplicative' or 'absolute'.` |
| `ValueError` | `projection_mask needs exactly one of include / values` |
| `ValueError` | `projection values must have shape ({...},), got {...}` |
| `ValueError` | `projection values must be finite (got NaN/inf)` |
| `ValueError` | `projection include must be 1-D, got shape {...}` |
| `ValueError` | `projection indices out of range for n_edges={...}` |
| `ValueError` | `plasticity_mask must be 1-D, got shape {...}` |
| `ValueError` | `plasticity_mask must be finite (got NaN/inf)` |
| `ValueError` | `enable_plasticity contradicts a present plasticity_mask; clear the mask or use disable_plasticity wi` |
| `ValueError` | `disable_plasticity: registered rule {...} has no 'k_w' coefficient; pass an explicit all-zero mask` |
| `ValueError` | `clamp value must be a finite float, got {...}` |
| `ValueError` | `clamp value must be a finite float, got {...}` |
| `ValueError` | `clamp value must be a finite float, got {...}` |
| `ValueError` | `clamp value must be a finite float, got {...}` |
| `ValueError` | `weights {...} and mask {...} must both be 1-D with equal length` |

**`jaxfne/hdp_rule.py`**

| Class | Message |
|---|---|
| `ValueError` | `HDP rule {...} declares unknown aux_layout {...}; expected one of {...}` |
| `ValueError` | `hdp_rule {...} returned undeclared theta_targets {...}; declared: {...}` |
| `ValueError` | `HDP rule {...} is already registered` |
| `ValueError` | `HDP rule {...} declares unknown aux_layout {...}; expected one of {...}` |
| `ValueError` | `HDP rule {...} declares unknown theta_targets {...}; expected subset of {...}` |
| `ValueError` | `HDP rule {...} declares aux_coords {...} with aux_layout 'none'` |
| `ValueError` | `HDP rule {...} declares inverted {...} {...}; expected (min, max)` |
| `ValueError` | `unknown registered hdp_rule {...}; known: {...}` |
| `ValueError` | `hdp_rule_params for rule {...} must be a mapping, got {...}` |
| `ValueError` | `hdp_rule_params for rule {...} contains unrecognized keys {...}; declared params: {...}` |
| `ValueError` | `kernel must be 'hdp' or 'baseline', got {...}` |
| `ValueError` | `compile_step_fn(kernel={...}) got unrecognized hdp_kwargs keys {...}; allowed: {...}` |
| `ValueError` | `superpose_hdp_rules({...}, ...): no components given` |
| `ValueError` | `superpose_hdp_rules({...}, ...): duplicate components {...}; superposing a rule with itself is refus` |
| `ValueError` | `superpose_hdp_rules({...}, ...): component {...} carries aux {...} (layout {...}); only H-only rules` |
| `ValueError` | `superpose_hdp_rules({...}, ...): component {...} declares {...} h_coords for h_shape {...} (width {.` |
| `ValueError` | `superpose_hdp_rules({...}, ...): scalar component {...} declares h_coords {...}; expected one coordi` |
| `ValueError` | `superpose_hdp_rules({...}, ...): components disagree on {...} {...}; pass compatible rules` |
| `ValueError` | `superpose_hdp_rules({...}, ...): components disagree on w_bounds {...}; pass w_bounds=... explicitly` |
| `ValueError` | `hdp_rule_params for rule {...}: component {...} params must be a mapping, got {...}` |
| `ValueError` | `hdp_rule_params for rule {...}: component {...} contains unrecognized keys {...}; declared params: {` |
| `ValueError` | `hdp_rule {...}: component returned dH of size {...}, expected {...}` |
| `ValueError` | `hdp_rule {...}: components disagree on d_theta {...} shape {...} vs {...}` |
| `ValueError` | `hdp_rule {...}: components disagree on j_theta {...} shape {...} vs {...}` |

**`jaxfne/inputs.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown XSource kind {...}; expected 'constant' or 'poisson'` |
| `ValueError` | `XSource targets must be a non-empty tuple of strings; got {...}` |
| `ValueError` | `XSource({...}) value takes a finite real number; got {...}` |
| `ValueError` | `poisson XSource rate_hz={...} must be >= 0` |
| `ValueError` | `XSource unit must be a non-empty string; got {...}` |
| `ValueError` | `XSource({...}) n_neurons={...} takes an int >= 1` |
| `ValueError` | `constant XSource unit {...} is not realized: expected one of {...}` |
| `ValueError` | `constant XSource takes no schedule; got {...}` |
| `ValueError` | `constant XSource takes no seed; got {...}` |
| `ValueError` | `poisson XSource unit {...} is not realized: expected {...}` |
| `ValueError` | `poisson XSource seed must be an int; got {...}` |
| `ValueError` | `poisson XSource schedule takes (key, value) pairs; got {...}` |
| `ValueError` | `poisson XSource schedule has unknown keys {...}; expected a subset of {...}` |
| `ValueError` | `poisson XSource schedule {...} is not realized: 'amplitude' carries the current-route mapping, alone` |
| `ValueError` | `poisson XSource counts={...} must be a positive integer` |
| `ValueError` | `poisson XSource amplitude={...} takes a finite real number` |
| `ValueError` | `poisson XSource schedule {...} is not realized: the AdEx mapping carries exactly {...}` |
| `ValueError` | `poisson XSource n_sources={...} must be an int >= 1` |
| `ValueError` | `poisson XSource n_sources={...} must be an int >= 1` |
| `ValueError` | `poisson XSource weight_nS={...} must be >= 0` |
| `ValueError` | `poisson XSource mechanism={...} must be a non-empty string` |
| `ValueError` | `poisson XSource delay_ms={...} takes a finite real number` |
| `ValueError` | `poisson XSource schedule {...} declares one window bound without the other; declare t_start_ms and t` |
| `ValueError` | `poisson XSource window {...} takes finite bounds >= 0 in ms` |
| `ValueError` | `poisson XSource t_start_ms={...} must be below t_stop_ms={...}` |
| `ValueError` | `drive(baseline_drive_by_cell_type={{...}: {...}}) is not realized: it takes a finite real number` |
| `ValueError` | `tensor_triple_drive {...} is not realized: it takes a list of {area, layer, cell_type, drive}` |
| `ValueError` | `tensor_triple_drive entry {...} is not realized: it takes {area, layer, cell_type, drive}` |
| `ValueError` | `tensor_triple_drive(area={...}, layer={...}, cell_type={...}, drive={...}) is not realized: it takes` |
| `ValueError` | `poisson_drive seed={...} must be an int` |
| `ValueError` | `poisson_drive counts={...} must be a positive integer` |
| `ValueError` | `poisson_drive counts={...} must be a positive integer` |
| `ValueError` | `poisson_drive {...} declares keys {...} the current route does not consume: it reads only {...} (no ` |
| `ValueError` | `Simulation(poisson_drive=[...]) (a list of per-population drives) is realized only on the lif route ` |
| `ValueError` | `poisson_drive rate_hz={...} must name at least one rate` |
| `ValueError` | `poisson_drive rate_hz={...} must hold rates >= 0` |
| `ValueError` | `poisson_drive rate_hz={...} must hold distinct rates` |
| `ValueError` | `poisson_drive seed={...} must be an int` |
| `ValueError` | `poisson_drive n_sources={...} must be an int >= 1` |
| `ValueError` | `poisson_drive n_sources={...} must split evenly across {...} rates` |
| `ValueError` | `poisson_drive list form is current-route only; route='adex' takes one mapping` |
| `ValueError` | `poisson_drive list must hold at least one mapping` |
| `ValueError` | `poisson_drive list element {...} is not realized: it takes a mapping` |
| `ValueError` | `poisson_drive list element {...} names target_cell_types: the list form is current-route only; the A` |
| `ValueError` | `poisson_drive {...} is not realized: it takes a mapping` |
| `ValueError` | `unknown poisson route {...}; expected 'current', 'adex' or 'auto'` |
| `ValueError` | `poisson_drive rate_hz={...} must be >= 0` |
| `ValueError` | `poisson_drive amplitude={...} takes a finite real number` |
| `ValueError` | `poisson_drive seed={...} must be an int` |
| `ValueError` | `poisson_drive {...} is missing keys {...}` |
| `ValueError` | `poisson_drive target_cell_types={...} must be a non-empty list of cell types` |
| `ValueError` | `recorded poisson_drive list element {...} is not a receipt mapping` |

**`jaxfne/intervene.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} must be a mapping, got {...}` |
| `ValueError` | `{...} has unknown keys {...}` |
| `ValueError` | `{...} must be a finite float, got {...}` |
| `ValueError` | `{...} must be a finite float, got {...}` |
| `ValueError` | `intervention mask must be 1-D, got shape {...}` |
| `ValueError` | `intervention mask must be finite (got NaN/inf)` |
| `ValueError` | `intervention name must be a non-empty string` |
| `ValueError` | `unknown network builder {...}; expected one of {...}` |
| `ValueError` | `network[{...}] must be a mapping` |
| `ValueError` | `network.run[seed] must be an int` |
| `ValueError` | `grammar v1 supports exactly one mechanism {...}; got {...}` |
| `ValueError` | `control mode must be one of {...}; got {...}` |
| `ValueError` | `clamp requires a mask (which projection is held)` |
| `ValueError` | `disable takes no value (got one; use clamp)` |
| `ValueError` | `observation signal must be one of {...}; got {...}` |
| `ValueError` | `observation subset must be one of {...}; got {...}` |
| `ValueError` | `observation expect must be one of {...}; got {...}` |
| `ValueError` | `observation.tolerance must be >= 0` |
| `ValueError` | `subset 'clamped'/'unclamped' needs signal 'w_final' and a control mask` |
| `ValueError` | `unsupported intervention schema {...}` |
| `ValueError` | `unknown network builder {...}` |
| `ValueError` | `control mask length {...} does not match signal length {...}` |
| `ValueError` | `intervention baseline hdp_params invalid: {...}` |
| `ValueError` | `baseline hdp_params must not carry a plasticity_mask` |
| `ValueError` | `identical-network violation: realized weights differ` |
| `ValueError` | `control mask length {...} does not match realized edge count {...}` |

**`jaxfne/io.py`**

| Class | Message |
|---|---|
| `ValueError` | `receipt_file_exists: {...}` |

**`jaxfne/istdp_vogels2011.py`**

| Class | Message |
|---|---|
| `ValueError` | `istdp_vogels2011 eta must be a finite float >= 0; got {...}` |
| `ValueError` | `istdp_vogels2011 tau_stdp_ms must be a finite float > 0; got {...}` |
| `ValueError` | `istdp_vogels2011 rho0_hz must be a finite float >= 0; got {...}` |
| `ValueError` | `istdp_plasticity_mask needs two 1-D per-edge boolean arrays of equal length; got shapes {...} and {.` |
| `ValueError` | `istdp_plasticity_mask selected no edge ({...} edges, 0 plastic): declare the I-to-E edges or drop th` |

**`jaxfne/jdna/completion.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown completion policy {...}` |
| `ValueError` | `unknown completable quantity {...}; JDNA completes only declared quantities: {...}` |
| `ValueError` | `{...} must be derived from another constraint, but no derive function was given` |
| `ValueError` | `{...} is stochastic: it requires a sample function and an explicit K_D key` |
| `ValueError` | `{...} is required but has no canonical default; refusing rather than inventing a consequential value` |
| `ValueError` | `partial {...} domain declares only one bound ({...}={...}, {...}={...}); a half-domain cannot be hon` |
| `ValueError` | `unknown geometry key(s) {...}; JDNA honours only {...}, and any other key would default its axis sil` |
| `ValueError` | `realize_geometry needs n >= 0; got {...}` |
| `ValueError` | `unsupported geometry distribution {...}; only 'uniform' can be honoured exactly` |
| `ValueError` | `degenerate {...} domain [{...}, {...}]; bounds must order` |
| `ValueError` | `declared geometry for leaf {...} has no realized neurons (realized leaves: {...}); refusing rather t` |

**`jaxfne/jdna/genome.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...}: {...} must be in {...}; got {...}` |
| `ValueError` | `{...}: w_mech must be finite and > 0; got {...}` |
| `ValueError` | `{...}: kind must be one of {...}` |
| `ValueError` | `{...}: missing {...}` |
| `ValueError` | `{...}: positions must name at least two areas` |
| `ValueError` | `{...}: decay must be finite and > 0; got {...}` |
| `ValueError` | `{...}: traversal_ms must be finite and >= 0; got {...}` |
| `ValueError` | `PseudoGenome declares schema_version={...}; this package only supports {...}. Register a migration p` |
| `ValueError` | `PseudoGenome {...}: mechanism_tau_ms must be a mapping of mechanism kind to tau_ms, got {...}` |
| `ValueError` | `PseudoGenome mechanism_tau_ms[{...}] must be a number in ms; got {...}` |
| `ValueError` | `PseudoGenome mechanism_tau_ms[{...}] must be finite and > 0; got {...}` |
| `ValueError` | `PseudoGenome.name must be non-empty` |
| `ValueError` | `PseudoGenome must declare at least one area` |
| `ValueError` | `duplicate area name {...}` |
| `ValueError` | `area {...} must declare at least one layer` |
| `ValueError` | `duplicate layer name {...} in area {...}` |
| `ValueError` | `layer {...}: n_neurons must be positive, got {...}` |
| `ValueError` | `layer {...}: depth_band must satisfy 0 <= lo < hi <= 1, got {...}` |
| `ValueError` | `layer {...}: cell_type_fractions must sum to 1, got {...}` |
| `ValueError` | `layer {...}: cell type {...} fraction must be in [0, 1], got {...}` |
| `ValueError` | `layer {...}: tolerance for undeclared cell type {...}` |
| `ValueError` | `layer {...}: tolerance band {...} must satisfy 0 <= lo <= hi <= 1, got {...}` |
| `ValueError` | `layer {...}: base fraction {...}={...} lies outside its declared tolerance band {...}` |
| `ValueError` | `layer {...}: tolerance bands jointly infeasible (sum lo={...}, sum hi={...}); a feasible probability` |
| `ValueError` | `area {...}: connection rule references unknown {...} {...}` |
| `ValueError` | `area {...}: connection rule references unknown {...} {...} in layer {...}` |
| `ValueError` | `genome mechanism_tau_ms[{...}] must be a number in ms; got {...}` |
| `ValueError` | `genome mechanism_tau_ms[{...}] must be finite and > 0; got {...}` |
| `ValueError` | `area_connections entries must be mappings, got {...}` |
| `ValueError` | `area_connections entry references unknown {...} {...}` |
| `ValueError` | `area_connections entry references unknown {...} {...} in area {...}` |
| `ValueError` | `area_connections entry references unknown {...} {...} in layer {...}` |
| `ValueError` | `box-constrained simplex infeasible: sum(lo)={...}, sum(hi)={...}` |
| `ValueError` | `box-simplex projection did not converge (sum={...})` |
| `ValueError` | `layer {...}: fractional allocation exceeds n_neurons={...}` |
| `ValueError` | `developed tensor missing layer {...}` |
| `ValueError` | `layer {...}: developed {...} neurons, genome declares {...}` |
| `ValueError` | `layer {...} cell type {...} has no realized fraction; its count cannot be checked against the declar` |
| `ValueError` | `layer {...}: counts do not sum to n_neurons` |
| `ValueError` | `layer {...} cell type {...}: count {...} outside declared band [{...}, {...}]` |
| `ValueError` | `P-023: {...} uses mechanism {...} with no entry in genome mechanism_tau_ms; time constants are requi` |
| `ValueError` | `unknown development parameter(s) {...}; develop consumes only {...}, and would record the others as ` |
| `ValueError` | `fraction_jitter_sigma must be >= 0; got {...}` |

**`jaxfne/jnwb_view.py`**

| Class | Message |
|---|---|
| `ValueError` | `to_jnwb: metadata dt_ms must be positive and finite, got {...}` |
| `ValueError` | `to_jnwb: one sample and no metadata dt_ms; the sampling rate is unknown` |
| `ValueError` | `to_jnwb: time_ms is not finite and increasing` |
| `ValueError` | `to_jnwb: time_ms is not uniform at dt_ms={...}: sample {...} is {...} ms, expected {...} ms; the sam` |
| `ValueError` | `to_jnwb: time_ms must be 1D, got shape {...}` |
| `ValueError` | `to_jnwb: {...} must be (n_steps={...}, n_units), got shape {...}; trial batches convert one run at a` |
| `ValueError` | `to_jnwb: spikes {...} and V_m {...} differ in shape` |
| `ValueError` | `to_jnwb: threshold-level spikes must be 0 or 1` |
| `ValueError` | `to_jnwb: spikes must be non-negative integer counts per step; a rate or probability readout has no s` |
| `ValueError` | `to_jnwb_trials: {...} was not recorded in every trial` |
| `ValueError` | `to_jnwb_trials: {...} is not a per-step array` |
| `ValueError` | `to_jnwb_trials: empty batch` |
| `ValueError` | `to_jnwb_trials: trials without signals {...}; filter the batch first` |
| `ValueError` | `to_jnwb_trials: trial {...} has dt_ms={...}, t0_s={...}, V_m {...}; trial {...} has dt_ms={...}, t0_` |

**`jaxfne/neuronal_tensor.py`**

| Class | Message |
|---|---|
| `ValueError` | `NeuronType {...} declares emitter_params={...} with emitter=None; emitter_params are validated again` |
| `ValueError` | `{...} length_unit must be one of {...}; got {...}` |
| `ValueError` | `{...} length_unit must be 'mm' (tensor positions are mm); got {...} — relabelling the geometry unit ` |
| `ValueError` | `{...} conduction_velocity_m_per_s must be a number; got {...}` |
| `ValueError` | `{...} conduction_velocity_m_per_s must be finite and > 0; got {...}` |
| `ValueError` | `{...} declares both delay_ms={...} and conduction_velocity_m_per_s={...}; declare one or neither` |
| `ValueError` | `{...} spatial_sigma must be a finite number > 0 in mm; got {...}` |
| `ValueError` | `{...} spatial_sigma must be a finite number > 0 in mm; got {...}` |
| `ValueError` | `{...} spatial_sigma must be finite and > 0 in mm; got {...}` |
| `ValueError` | `{...} allow_self must be a bool; got {...}` |
| `ValueError` | `{...} {...} must be a finite number >= 0; got {...}` |
| `ValueError` | `{...} {...} must be a finite number >= 0; got {...}` |
| `ValueError` | `{...} {...} must be finite and >= 0; got {...}` |
| `ValueError` | `{...} sampling must be one of {...}; got {...}` |
| `ValueError` | `{...} max_in_degree must be a positive integer; got {...}` |
| `ValueError` | `{...} max_in_degree must be a positive integer; got {...}` |
| `ValueError` | `{...} in_degree must be a positive integer; got {...}` |
| `ValueError` | `{...} in_degree must be a positive integer; got {...}` |
| `ValueError` | `{...} drive must be a finite real number; got {...}` |
| `ValueError` | `{...} drive must be finite; got {...}` |
| `ValueError` | `InterConnection delay_ms must be a number in ms; got {...}` |
| `ValueError` | `InterConnection delay_ms must be finite and >= 0; got {...}` |
| `ValueError` | `InterConnection declares both footprint={...} and max_in_degree={...}; the footprint rule selects pa` |
| `ValueError` | `InterConnection declares both footprint={...} and in_degree={...}; the footprint rule selects pairs ` |
| `ValueError` | `InterConnection declares both in_degree={...} and max_in_degree={...}; exact fixed in-degree and loc` |
| `ValueError` | `InterConnection declares sampling='fixed_indegree' without in_degree; the fixed-indegree sampler dra` |
| `ValueError` | `InterConnection declares in_degree={...} with sampling={...}; the fixed-indegree sampler runs under ` |
| `ValueError` | `InterConnection declares both footprint={...} and sampling={...}; the footprint rule selects pairs d` |
| `ValueError` | `InterConnection declares weight_sharing={...} without footprint; sharing groups are the footprint ru` |
| `ValueError` | `Area.connectivity_mode must be 'unspecified' or 'explicit'` |
| `ValueError` | `Area with a provided inter_connections sequence cannot be marked as unspecified` |
| `ValueError` | `Area.connectivity_mode must be 'unspecified' or 'explicit'` |
| `ValueError` | `Area with a provided inter_connections sequence cannot be marked as unspecified` |
| `ValueError` | `AreaConnection delay_ms must be a number in ms; got {...}` |
| `ValueError` | `AreaConnection delay_ms must be finite and >= 0; got {...}` |
| `ValueError` | `AreaConnection declares delay_std_ms={...} without delay_ms; the spread draws around one scalar mean` |
| `ValueError` | `AreaConnection declares delay_std_ms={...} with conduction_velocity_m_per_s={...}; one constant mean` |
| `ValueError` | `AreaConnection probability must be a number; got {...}` |
| `ValueError` | `AreaConnection probability must be in (0, 1]; got {...}` |
| `ValueError` | `AreaConnection declares both max_in_degree={...} and probability={...}; localized sampling and the d` |
| `ValueError` | `AreaConnection declares both in_degree={...} and probability={...}; the fixed-indegree sampler draws` |
| `ValueError` | `AreaConnection declares both footprint={...} and probability={...}; the footprint rule selects pairs` |
| `ValueError` | `AreaConnection declares both footprint={...} and max_in_degree={...}; the footprint rule selects pai` |
| `ValueError` | `AreaConnection declares both footprint={...} and in_degree={...}; the footprint rule selects pairs d` |
| `ValueError` | `AreaConnection declares both in_degree={...} and max_in_degree={...}; exact fixed in-degree and loca` |
| `ValueError` | `AreaConnection declares sampling='fixed_indegree' without in_degree; the fixed-indegree sampler draw` |
| `ValueError` | `AreaConnection declares in_degree={...} with sampling={...}; the fixed-indegree sampler runs under s` |
| `ValueError` | `AreaConnection declares both footprint={...} and sampling={...}; the footprint rule selects pairs de` |
| `ValueError` | `AreaConnection declares weight_sharing={...} without footprint; sharing groups are the footprint rul` |
| `ValueError` | `TrainedW0 {...} must be a sequence of edge indices; got {...}` |
| `ValueError` | `TrainedW0 {...} must hold only ints; got {...}` |
| `ValueError` | `TrainedW0 w must be a sequence of numbers; got {...}` |
| `ValueError` | `TrainedW0 w must hold only real numbers; got {...}` |
| `ValueError` | `TrainedW0 w must be all finite; got {...}` |
| `ValueError` | `W0Provenance origin must be one of {...}; got {...}` |
| `ValueError` | `W0Provenance origin='trained' requires parent_digest (the identity digest of the tensor the weights ` |
| `ValueError` | `W0Provenance origin='declared' carries no {...}; got {...}` |
| `ValueError` | `W0Provenance n_steps must be an int >= 0; got {...}` |
| `ValueError` | `W0Provenance n_steps must be an int >= 0; got {...}` |
| `ValueError` | `W0Provenance {...} must be a non-empty str; got {...}` |
| `ValueError` | `TrainedW0 dtype must be one of {...}; got {...}` |
| `ValueError` | `TrainedW0 pre/post/w must be non-empty and the same length; got {...}` |
| `ValueError` | `NeuronalTensor.connectivity_mode must be 'unspecified' or 'explicit'` |
| `ValueError` | `NeuronalTensor with declared connectivity cannot be marked as unspecified` |
| `ValueError` | `NeuronalTensor.connectivity_mode must be 'unspecified' or 'explicit'` |
| `ValueError` | `NeuronalTensor with declared connectivity cannot be marked as unspecified` |
| `ValueError` | `NeuronalTensor trained_w0 and w0_provenance.origin disagree: trained weights require origin='trained` |
| `ValueError` | `{...} does not look like a NeuronalTensor JSON config (no top-level 'areas' key).` |
| `ValueError` | `NeuronalTensor JSON at {...} declares schema_version={...}, expected {...}; strict=True refuses forw` |
| `ValueError` | `poses must have one entry per area; got {...} for {...} areas` |
| `ValueError` | `merge_neuronal_tensors refuses tensor {...} carrying trained W0 (origin='trained'): merged edges are` |
| `ValueError` | `{...}/{...}: layer block has zero depth ({...}, {...}); a column-relative z_range cannot be placed i` |
| `ValueError` | `RuntimeConfiguration.{...} is reserved and not wired on the tensor path; got {...}. Leave it None.` |
| `ValueError` | `RuntimeConfiguration.vmap is not consumed: simulate(model) runs one trial, and simulate_batch takes ` |
| `ValueError` | `tensor {...} carries trained W0 but resolves to emitter family {...}; trained W0 is realized only on` |
| `ValueError` | `n_contacts={...} is not consumed on the lif tensor route (no laminar probe readout there); leave the` |
| `ValueError` | `n_contacts={...} is not consumed on the adex tensor route (no laminar probe readout there); leave th` |
| `ValueError` | `n_contacts={...} is not consumed on the rate tensor route (no laminar probe readout there); leave th` |
| `ValueError` | `adex_decay={...}, adex_step_order={...} are not consumed for emitter {...} on the current-based rout` |
| `ValueError` | `with_trained_weights: {...} is not array-like ({...})` |
| `ValueError` | `with_trained_weights: {...} must be a 1-D per-edge vector; got shape {...}` |
| `ValueError` | `with_trained_weights: {...} is empty; need at least one edge` |
| `ValueError` | `with_trained_weights: seed must be an int; got {...}` |
| `ValueError` | `with_trained_weights: a Model carries its own edges — pass pre/post only with weight-vector sources ` |
| `ValueError` | `with_trained_weights: trained W0 is realized only on the izhikevich route ({...})` |
| `ValueError` | `with_trained_weights: trained W0 is realized only on the izhikevich route; tensor resolves to {...}` |
| `ValueError` | `with_trained_weights: model has no params['edge_list'] (the adex route stores edges in static['adex'` |
| `ValueError` | `with_trained_weights: model has no HDP diagnostics — re-run it with RuntimeConfig(enable_hdp=True, h` |
| `ValueError` | `with_trained_weights: HDP diagnostics carry no w_final — re-run with enable_hdp=True and pass the tr` |
| `ValueError` | `with_trained_weights: model was not built from this tensor (tensor_identity mismatch) — trained weig` |
| `ValueError` | `with_trained_weights: model has no emitter — cannot resolve native edge signs` |
| `ValueError` | `with_trained_weights: bare Signals carry HDP summaries only (metadata['hdp']), never per-edge weight` |
| `ValueError` | `with_trained_weights: unlabeled leaf list (e.g. checkpoint_state/restore_state leaves carry no edge ` |
| `ValueError` | `with_trained_weights: a ContinuationState carries w but no edges — pass explicit pre= and post= from` |
| `ValueError` | `with_trained_weights: pass edges once — inside the (pre, post, w) triple or as pre=/post= kwargs, ne` |
| `ValueError` | `with_trained_weights: a (Signals, state) pair carries no edges on its own — pass the state with expl` |
| `ValueError` | `with_trained_weights: tuple source must be a (pre, post, w) triple; got length {...}` |
| `ValueError` | `with_trained_weights: pass edges once — inside the dict or as pre=/post= kwargs, never both` |
| `ValueError` | `with_trained_weights: dict source needs keys pre/post/w; missing {...}` |
| `ValueError` | `with_trained_weights: a bare weight vector needs explicit pre= and post= from the trained model` |
| `ValueError` | `with_trained_weights: trained W0 is realized only on the izhikevich route ({...})` |
| `ValueError` | `with_trained_weights: trained W0 is realized only on the izhikevich route; tensor resolves to {...}` |
| `ValueError` | `with_trained_weights: reference construction of tensor {...} produced no params['edge_list']` |
| `ValueError` | `with_trained_weights: reference construction of tensor {...} produced no emitter` |
| `ValueError` | `with_trained_weights: pre/post/w must have the same length; got {...}` |
| `ValueError` | `with_trained_weights: {...} must be an integer array; got dtype {...} (no silent coercion — pass int` |
| `ValueError` | `with_trained_weights: w must be float32 or float64; got dtype {...}` |
| `ValueError` | `with_trained_weights: w must be all finite; got NaN or inf` |
| `ValueError` | `with_trained_weights: candidate pre does not match the edges compiled from tensor {...} at seed={...` |
| `ValueError` | `with_trained_weights: candidate post does not match the edges compiled from tensor {...} at seed={..` |
| `ValueError` | `with_trained_weights: reference edges ({...}) and candidate weights ({...}) disagree in length` |
| `ValueError` | `with_trained_weights: {...}/{...} trained weights flip the native edge sign — HDP updates are sign-p` |
| `ValueError` | `tensor {...} carries trained W0 but the constructed model has no params['edge_list'] — trained W0 ca` |
| `ValueError` | `tensor {...} carries trained W0 for {...} edges but this construction compiled {...} — rebuild with ` |
| `ValueError` | `tensor {...} carries trained W0 but the constructed model has no emitter — cannot resolve native edg` |
| `ValueError` | `tensor {...}: {...}/{...} stored weights flip the compiled native edge sign — refusing` |
| `ValueError` | `tensor {...} stores trained W0 in {...} but {...}/{...} values change when cast to the model dtype {` |
| `ValueError` | `unknown AdEx cell type {...}; expected one of {...}` |
| `ValueError` | `P-023: {...} uses unknown mechanism {...}; known mechanisms are {AMPA, NMDA} (excitatory) and {GABA_` |
| `ValueError` | `P-023: {...}: E source with inhibitory mechanism {...}; use an excitatory mechanism (AMPA, NMDA) for` |
| `ValueError` | `P-023: {...}: non-E source {...} with excitatory mechanism {...}; use an inhibitory mechanism (GABA_` |
| `ValueError` | `Unknown emitter: {...}. Choose from: {...}` |
| `ValueError` | `emitter {...} is registered but not routed by the NeuronalTensor bridge; routable today: {...}` |
| `ValueError` | `partial per-NeuronType emitter declaration is refused: these types declare no emitter while others d` |
| `ValueError` | `heterogeneous per-NeuronType emitter declaration is refused (one family per tensor): {...}` |
| `ValueError` | `emitter {...} is registered but not routed by the NeuronalTensor bridge; routable today: {...}` |
| `ValueError` | `tensor declares emitter {...} on every NeuronType but the runtime asks for {...}; pass RuntimeConfig` |
| `ValueError` | `P-023: {...} declares no time constant (StaticParams.dT_ms is None); time constants are required, ne` |
| `ValueError` | `P-023: {...} declares inadmissible time constant dT_ms={...}; time constants must be a finite number` |
| `ValueError` | `P-023: {...} declares non-numeric time constant dT_ms={...}; time constants must be a finite number ` |
| `ValueError` | `P-023: {...} declares inadmissible time constant dT_ms={...}; time constants must be finite and > 0 ` |
| `ValueError` | `{...} declares allow_self=True but no self pair can exist ({...}.{...}.{...}->{...}.{...}.{...}): se` |
| `ValueError` | `{...} declares spatial_sigma={...} without max_in_degree; spatial_sigma needs max_in_degree (localiz` |
| `ValueError` | `{...} declares both max_in_degree={...} and in_degree={...}; exact fixed in-degree and localized (di` |
| `ValueError` | `{...} declares w_std={...} with weight_sharing={...}; a per-edge spread has no shared-row consumer —` |
| `ValueError` | `{...} declares delay_std_ms={...} without delay_ms; the spread draws around one scalar mean — declar` |
| `ValueError` | `{...} declares spatial_sigma={...}; the conductance (adex) route has no localized-sampling consumer ` |
| `ValueError` | `{...} declares max_in_degree={...}; the conductance (adex) route has no localized-sampling consumer ` |
| `ValueError` | `{...} declares in_degree={...}; the conductance (adex) route has no fixed-indegree consumer (exact p` |
| `ValueError` | `{...} declares weight_sharing={...}; the conductance (adex) route has no shared-weight consumer (onl` |
| `ValueError` | `{...} declares local_inhibition={...}; the conductance (adex) route has no intra-step fixed-point co` |
| `ValueError` | `{...} declares footprint={...}; the conductance (adex) route has no footprint-rule consumer (determi` |
| `ValueError` | `{...} declares w_std={...}; the conductance (adex) route has no per-edge weight-spread consumer (onl` |
| `ValueError` | `{...} declares delay_std_ms={...}; the conductance (adex) route has no per-edge delay-spread consume` |
| `ValueError` | `{...} declares allow_self=True with conduction_velocity_m_per_s={...}; a zero-distance self edge rea` |
| `ValueError` | `P-023: {...} declares no reversal potential for mechanism {...}; the adex route needs static.reversa` |
| `ValueError` | `P-023: {...} declares non-numeric reversal potential {...} for mechanism {...}` |
| `ValueError` | `P-023: {...} declares non-finite reversal potential {...} for mechanism {...}` |
| `ValueError` | `{...} realizes to an edge weight of 0 nS on the conductance (adex) route (plastic.w_mech={...} scale` |
| `ValueError` | `{...} declares neither delay_ms nor conduction_velocity_m_per_s; the adex route needs one delay per ` |
| `ValueError` | `geometry-delay matcher sees edge endpoints outside the neuron table: {...}` |
| `ValueError` | `{...} declares conduction_velocity_m_per_s={...} but selects no compiled edge; a velocity connection` |
| `ValueError` | `{...} with conduction_velocity_m_per_s selects edge {...}, already selected by {...}; overlapping ve` |
| `ValueError` | `cannot compute identity digest for tensor {...}: {...}` |
| `ValueError` | `tensor mixes declared poses ({...}) with undeclared poses ({...}); rigid motion holds only when ever` |
| `ValueError` | `NeuronalTensor must declare at least one area to bridge` |
| `ValueError` | `tensor {...} carries trained W0 (origin='trained'): neuronal_tensor_to_configuration builds the decl` |
| `ValueError` | `{...} declares weight_sharing={...}; the current-based (izhikevich) route has no shared-weight consu` |
| `ValueError` | `{...} declares local_inhibition={...}; the current-based (izhikevich) route has no intra-step fixed-` |
| `ValueError` | `{...} declares weight_sharing={...}; the current-based (izhikevich) route has no shared-weight consu` |
| `ValueError` | `{...} declares local_inhibition={...}; the current-based (izhikevich) route has no intra-step fixed-` |
| `ValueError` | `tensor resolves to emitter 'lif'; the PCL LIF route is built by construct_neuronal_tensor, not this ` |
| `ValueError` | `tensor resolves to emitter 'adex'; the conductance route is built by neuronal_tensor_to_adex_configu` |
| `ValueError` | `Area {...} Layer {...}: n_neurons must be positive, got {...}; a zero-neuron layer is refused, never` |
| `ValueError` | `Unknown emitter: {...}. Choose from: {...}` |
| `ValueError` | `NeuronalTensor must declare at least one area to bridge` |
| `ValueError` | `tensor {...} carries trained W0 (origin='trained'): neuronal_tensor_to_adex_configuration builds the` |
| `ValueError` | `tensor connectivity_mode is 'unspecified' (no declared connections); the conductance (adex) route wi` |
| `ValueError` | `tensor resolves to emitter {...}, not 'adex'` |
| `ValueError` | `adex_decay must be one of {...}; got {...}` |
| `ValueError` | `adex_step_order must be one of {...}; got {...}` |
| `ValueError` | `positions and rows come as a harvested pair; pass both or neither` |
| `ValueError` | `adex tensor bridge harvested no neurons` |
| `ValueError` | `{...} declares conduction_velocity_m_per_s={...} but selects no compiled pair; a velocity connection` |

**`jaxfne/objectives.py`**

| Class | Message |
|---|---|
| `ValueError` | `Unknown synchrony method: {...}` |
| `ValueError` | `relative_standard_error must lie in (0, 1); got {...}` |
| `ValueError` | `null normalization requires at least {...} valid null samples to estimate sigma; {...} of {...} surv` |
| `ValueError` | `null distribution is degenerate (sigma = {...}) over {...} valid samples, so the z-score is undefine` |
| `ValueError` | `null normalization requires at least {...} valid null samples (caller-declared minimum); {...} of {.` |

**`jaxfne/optim/agsdr.py`**

| Class | Message |
|---|---|
| `ValueError` | `step_agsdr_transform: unknown hyperparameter(s) {...}; this kernel reads only ['eta']` |

**`jaxfne/optim/core.py`**

| Class | Message |
|---|---|
| `ValueError` | `parameter_bounds must be non-empty dict` |
| `ValueError` | `n_generations and n_population must be >= 1` |
| `ValueError` | `non-finite bounds for parameter {...}: {...}` |
| `ValueError` | `degenerate bounds for parameter {...}: {...}` |
| `ValueError` | `sdr_transform: {...}={...} is not consumed by this transform; only the default {...} is accepted` |
| `ValueError` | `SDR transform requires explicit PRNG key (key=jax.random.PRNGKey(...))` |
| `ValueError` | `gsdr_transform: checkpoint_n_steps={...} is not consumed by this transform; only the default 50 is a` |
| `ValueError` | `GSDR transform requires explicit PRNG key` |
| `ValueError` | `agsdr_transform: checkpoint_n_steps={...} is not consumed by this transform; only the default 50 is ` |
| `ValueError` | `AGSDR transform requires explicit PRNG key` |
| `ValueError` | `matrix tune with an inner optimizer refuses a placeholder emitter.W (recurrent_backend='edge_list'):` |
| `ValueError` | `declared matrix parameters overlap on entries whose scale bounds have no common value; the refined W` |
| `ValueError` | `learning_rate must be > 0; got {...}: a zero step returns the initial value as tuned and a negative ` |
| `ValueError` | `source_scale gradient loop reads no seed of its own: seed={...} applies only to the default simulati` |
| `ValueError` | `scalar soft-rate gradient loop reads no seed of its own: seed={...} applies only to the default simu` |
| `ValueError` | `Differentiable tuning of {...} is not supported for edge_list-authoritative models (placeholder emit` |

**`jaxfne/optim/gsdr.py`**

| Class | Message |
|---|---|
| `ValueError` | `step_gsdr_transform: unknown hyperparameter(s) {...}; this kernel reads only ['eta']` |
| `ValueError` | `GSDR-MCDP transform requires explicit PRNG key` |

**`jaxfne/optim/gsgd.py`**

| Class | Message |
|---|---|
| `ValueError` | `step_gsgd_transform: unknown hyperparameter(s) {...}; this kernel reads only ['eta']` |

**`jaxfne/optim/handles.py`**

| Class | Message |
|---|---|
| `ValueError` | `tune_handles() requires a non-empty dict {handle path: (lo, hi)}; got {...}` |
| `ValueError` | `Parameter path {...} needs a scalar (lo, hi) bound pair; got {...}` |
| `ValueError` | `Parameter path {...} has non-finite bounds {...}` |
| `ValueError` | `Parameter path {...} has degenerate bounds {...}` |
| `ValueError` | `Parameter path(s) {...} address dense emitter.W, but the simulation runs recurrent_backend='edge_lis` |
| `ValueError` | `soft-rate handle gradient loop reads no seed of its own: seed={...} applies only to the default simu` |
| `ValueError` | `Gradient tuning of parameter path(s) {...} is not supported: edge-list/adex edge writes are concrete` |
| `ValueError` | `Model.tune() cannot gradient-fit parameter path {...}: it crosses the spiking dynamics and Model.tun` |
| `ValueError` | `Either 'objective' (legacy) or 'objectives' (public) must be provided` |
| `ValueError` | `Gradient tuning of parameter path(s) {...} is not supported: edge-list/adex edge writes are concrete` |
| `ValueError` | `Gradient tuning of parameter path(s) {...} crosses the spiking dynamics and needs an explicit surrog` |
| `ValueError` | `Unknown handles surrogate {...} for parameter path(s) {...}; expected one of {...}` |

**`jaxfne/optim/sdr.py`**

| Class | Message |
|---|---|
| `ValueError` | `step_sdr_transform: unknown hyperparameter(s) {...}; this kernel reads only ['eta']` |

**`jaxfne/paradigm.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} must be a 2-tuple/list of (start_ms, end_ms)` |
| `ValueError` | `condition {...} has sequence length {...}, but {...} sequential event labels were provided` |
| `ValueError` | `general_sequential_oddball_paradigm refuses event_windows and event_codes together (event_windows wo` |
| `ValueError` | `general_sequential_oddball_paradigm requires event_windows or event_codes` |
| `ValueError` | `general_sequential_oddball_paradigm requires at least one sequence_event_label` |
| `ValueError` | `general_sequential_oddball_paradigm requires at least one condition` |
| `ValueError` | `condition {...} must provide either 'sequence' or 'events'` |
| `ValueError` | `comparison_label={...} collides with a sequence event code ({...} != {...}): the comparison assignme` |
| `ValueError` | `pre_stimulus_buffer_ms={...} exceeds l4_onset_ms={...}; the baseline window would start before 0.` |
| `ValueError` | `omission_oddball_paradigm refuses non-default onset values (standard_onset_ms={...}, deviant_onset_m` |
| `ValueError` | `{...} must be finite, got {...} (P-024)` |
| `ValueError` | `omission_oddball_paradigm refuses equal standard and deviant drives (standard_drive_amplitude={...},` |

**`jaxfne/pcl_topology.py`**

| Class | Message |
|---|---|
| `ValueError` | `{...} local_inhibition must be a mapping with 'rounds' (and optional 'order'); got {...}` |
| `ValueError` | `{...} local_inhibition carries unknown keys {...}; expected a subset of {...}` |
| `ValueError` | `{...} local_inhibition resolution 'event_scan' replays every event with no cap; 'rounds' must be abs` |
| `ValueError` | `{...} local_inhibition order must be 'ascending' under resolution 'event_scan' (the scan visits cand` |
| `ValueError` | `{...} local_inhibition needs 'rounds' (winners resolved per position per step); got {...}` |
| `ValueError` | `{...} local_inhibition order must be one of {...}; got {...}` |
| `ValueError` | `{...} local_inhibition resolution must be one of {...}; got {...}` |
| `ValueError` | `{...} footprint {...} must be a pair of numbers; got {...}` |
| `ValueError` | `{...} footprint {...} must be a pair of numbers; got {...}` |
| `ValueError` | `{...} footprint {...} must be a pair of numbers; got {...}` |
| `ValueError` | `{...} footprint {...} must be {...}; got {...}` |
| `ValueError` | `{...} footprint must be a mapping with keys {...}; got {...}` |
| `ValueError` | `{...} footprint carries unknown keys {...}; expected a subset of {...}` |
| `ValueError` | `{...} footprint needs 'size' (the covering rectangle); got {...}` |
| `ValueError` | `{...} weight_sharing must be one of {...}; got {...}` |
| `ValueError` | `{...} rounds must be a positive integer; got {...}` |
| `ValueError` | `{...} rounds must be a positive integer; got {...}` |
| `ValueError` | `{...} footprint needs finite pre_xy of shape ({...}, 2); got shape {...}` |
| `ValueError` | `{...} footprint needs finite post_xy of shape ({...}, 2); got shape {...}` |
| `ValueError` | `shared must be a 2-D (n_groups, n_syn) table; got shape {...}` |
| `ValueError` | `group_of_edge must be a non-empty 1-D sequence` |
| `ValueError` | `group_of_edge holds ids outside [0, {...}); got range ({...}, {...})` |
| `ValueError` | `shared must be a 2-D (n_groups, n_syn) table; got shape {...}` |
| `ValueError` | `group_of_edge {...} and edge_delta {...} disagree` |
| `ValueError` | `{...} order must be one of {...}; got {...}` |
| `ValueError` | `{...} v must hold at least one depth` |
| `ValueError` | `{...} t_last {...} and t_spike {...} must both match v ({...},)` |
| `ValueError` | `{...} adaptive=True needs adapt state` |
| `ValueError` | `{...} adapt {...} must match v ({...},)` |
| `ValueError` | `{...} thr {...} must be scalar or ({...},)` |
| `ValueError` | `{...} inputs {...} must be (M, {...}) in visit order` |
| `ValueError` | `{...} inhib {...} must be ({...}, {...})` |
| `ValueError` | `{...} inhib must carry a zero diagonal (no self-inhibition)` |
| `ValueError` | `{...} {...} must be a real number; got {...}` |
| `ValueError` | `{...} tau_m must be finite and > 0; got {...}` |
| `ValueError` | `{...} tau_rp {...} must be finite and > 0` |
| `ValueError` | `{...} tau_adapt must be finite and > 0; got {...}` |
| `ValueError` | `{...} t_now must be a real timestamp; got {...}` |
| `ValueError` | `local_inhibition t_now={...} precedes t_last[{...}]={...}` |
| `ValueError` | `{...} dt_ms must be a real number; got {...}` |
| `ValueError` | `{...} dt_ms must be finite and > 0; got {...}` |
| `ValueError` | `{...} dt_ms must resolve to whole microseconds; got {...}` |
| `ValueError` | `{...} n_steps must be a positive integer; got {...}` |
| `ValueError` | `{...} n_steps must be a positive integer; got {...}` |
| `ValueError` | `{...} {...} must be a positive integer; got {...}` |
| `ValueError` | `{...} origins_x must hold {...} finite values; got shape {...}` |
| `ValueError` | `{...} origins_y must hold {...} finite values; got shape {...}` |
| `ValueError` | `{...} footprint 'size' must hold whole pixels (weight indices); got {...}` |
| `ValueError` | `{...} weights must be a finite array of shape ({...}, 2, {...}, {...}); got shape {...}` |
| `ValueError` | `{...} event arrays disagree in length: t{...} x{...} y{...} p{...}` |
| `ValueError` | `{...} event_t_us must be finite` |
| `ValueError` | `{...} event {...} is off the pixel plane (x={...}, y={...})` |
| `ValueError` | `{...} event {...} has polarity {...}; expected 0 or 1` |
| `ValueError` | `{...} t0_us must be an integer; got {...}` |
| `ValueError` | `{...} event {...} falls outside the {...}-step grid (t={...}, bin={...})` |
| `ValueError` | `{...} ranks and arrivals must be finite 2-D arrays of one shape; got {...} and {...}` |
| `ValueError` | `{...} ranks must be finite` |
| `ValueError` | `{...} arrivals must be finite` |
| `ValueError` | `{...} arrivals must hold non-negative event counts` |
| `ValueError` | `{...} arrivals must hold whole event counts` |
| `ValueError` | `{...} blocks must carry 'members' and 'member_mask'; got {...}` |
| `ValueError` | `{...} blocks members/mask must share one (G, D) shape; got {...} and {...}` |
| `ValueError` | `{...} ranks must be a finite 2-D array; got shape {...}` |
| `ValueError` | `{...} ranks cover {...} cells but the blocks span {...}` |
| `ValueError` | `{...} blocks member ids must lie in [0, {...}); group {...} holds {...}` |
| `ValueError` | `{...} groups must have shape (N,); got {...}` |
| `ValueError` | `{...} inhibition pre/post/weight disagree in length: {...} {...} {...}` |
| `ValueError` | `{...} inhibition endpoints must be whole ids in [0, {...})` |
| `ValueError` | `{...} inhibition weights must be finite` |
| `ValueError` | `{...} inhibition weights must hold non-negative magnitudes (the cascade subtracts them; negate execu` |
| `ValueError` | `{...} inhibition edge {...} crosses groups ({...}->{...}); local inhibition stays inside one retinot` |
| `ValueError` | `{...} {...} must be a whole number; got {...}` |
| `ValueError` | `{...} {...} must be a whole number; got {...}` |
| `ValueError` | `{...} {...} must be a whole number; got {...}` |
| `ValueError` | `{...} {...} must be >= 1; got {...}` |
| `ValueError` | `{...} origins_x must hold nx={...} origins; got {...}` |
| `ValueError` | `{...} origins_y must hold ny={...} origins; got {...}` |
| `ValueError` | `{...} size must be a pair of whole pixels; got {...}` |
| `ValueError` | `{...} size must hold positive pixels; got {...}` |
| `ValueError` | `{...} weights must have shape ({...}, 2, {...}, {...}); got {...}` |
| `ValueError` | `{...} weights must be finite` |
| `ValueError` | `{...} c_ox must hold cx={...} origins; got {...}` |
| `ValueError` | `{...} c_oy must hold cy={...} origins; got {...}` |
| `ValueError` | `{...} c_size must be a pair of whole pixels; got {...}` |
| `ValueError` | `{...} c_size must hold positive pixels; got {...}` |
| `ValueError` | `{...} c_weights must have shape ({...}, {...}, {...}, {...}); got {...}` |
| `ValueError` | `{...} c_weights must be finite` |
| `ValueError` | `{...} c_cover must map every simple (i, j) to complex cells` |
| `ValueError` | `{...} c_cover carries a bad position {...}` |
| `ValueError` | `{...} c_cover[{...}] must list complex (cx, cy) cells` |
| `ValueError` | `{...} c_cover[{...}] carries a bad complex cell {...}` |
| `ValueError` | `{...} c_cover must name every simple position's covering complex cells` |
| `ValueError` | `{...} inhib_range must be a pair of whole pixels; got {...}` |
| `ValueError` | `{...} inhib_range must hold non-negative pixels; got {...}` |
| `ValueError` | `{...} lateral_on must be a bool; got {...}` |
| `ValueError` | `{...} topdown_on must be a bool; got {...}` |
| `ValueError` | `{...} activation must be a bool; got {...}` |
| `ValueError` | `{...} inhib_range {...} is set but lateral is off; the window only windows lateral neighborhoods` |
| `ValueError` | `{...} lateral_weights is set but lateral is off; without the lateral calls no weight executes` |
| `ValueError` | `{...} topdown_weights is set but topdown is off; without the top-down calls no weight executes` |
| `ValueError` | `{...} lateral_weights must have shape ({...}, {...}); got {...}` |
| `ValueError` | `{...} lateral_weights must be finite` |
| `ValueError` | `{...} lateral_weights must hold non-negative magnitudes` |
| `ValueError` | `{...} topdown_weights must have shape ({...}, {...}) (rows are complex sources); got {...}` |
| `ValueError` | `{...} topdown_weights must be finite` |
| `ValueError` | `{...} topdown_weights must hold non-negative magnitudes` |
| `ValueError` | `{...} {...} must be one-dimensional; got shape {...}` |
| `ValueError` | `{...} {...} must hold integers; got {...}` |
| `ValueError` | `{...} event arrays must share one length; got {...}, {...}, {...}, {...}` |
| `ValueError` | `{...} dt_ms must be a real number; got {...}` |
| `ValueError` | `{...} dt_ms must be > 0; got {...}` |
| `ValueError` | `{...} dt_ms must resolve to whole microseconds; got {...}` |
| `ValueError` | `{...} n_steps/vw/vh must be >= 1` |
| `ValueError` | `{...} t0_us must be >= 0; got {...}` |
| `ValueError` | `{...} event x must lie on the {...}-pixel plane` |
| `ValueError` | `{...} event y must lie on the {...}-pixel plane` |
| `ValueError` | `{...} event polarity must lie in {0, 1}` |
| `ValueError` | `{...} event times must fall on the [{...}, {...}) microsecond grid` |
| `ValueError` | `{...} {...} must be >= 1; got {...}` |
| `ValueError` | `{...} ns={...} disagrees with the simple grid nx * ny * nd={...}; the scan indexes simples by grid p` |
| `ValueError` | `{...} wi_s must have shape ({...}, {...}); got {...}` |
| `ValueError` | `{...} wi_s must be finite` |
| `ValueError` | `{...} wi_c must have shape ({...}, {...}); got {...}` |
| `ValueError` | `{...} wi_c must be finite` |
| `ValueError` | `{...}: h_update must be a mapping with keys {...}; got {...}` |
| `ValueError` | `{...}: h_update holds unknown key(s) {...}; allowed keys are {...}` |
| `ValueError` | `{...}: h_update is missing {...}` |
| `ValueError` | `{...}: h_update[{...}] must be a finite real number; got {...}` |
| `ValueError` | `{...}: h_update[{...}] must be finite reals of shape ({...},); got shape {...} dtype {...}` |
| `ValueError` | `{...}: h_update['measurement_interval_us'] must be > 0; got {...}` |
| `ValueError` | `{...}: h_update['rate_alpha'] must lie in (0, 1]; got {...}` |
| `ValueError` | `{...}: h_update['last_update_us'] must be an integer microsecond time; got {...}` |
| `ValueError` | `{...}: w_update[{...}] must be a mapping with keys {...}; got {...}` |
| `ValueError` | `{...}: w_update[{...}] holds unknown key(s) {...}; allowed keys are {...}` |
| `ValueError` | `{...}: w_update[{...}] is missing {...}` |
| `ValueError` | `{...}: w_update[{...}]['mode'] must be one of {...}; got {...}` |
| `ValueError` | `{...}: w_update[{...}][{...}] must be a finite real number; got {...}` |
| `ValueError` | `{...}: w_update[{...}][{...}] must be > 0; got {...}` |
| `ValueError` | `{...}: w_update must be a mapping with keys {...}; got {...}` |
| `ValueError` | `{...}: w_update holds unknown key(s) {...}; allowed keys are {...}` |
| `ValueError` | `{...}: w_update is missing {...}` |
| `ValueError` | `{...}: w_update['rule'] must be {...}; got {...}` |
| `ValueError` | `{...}: w_update['buffer_capacity'] must be an integer >= 1; got {...}` |
| `ValueError` | `{...}: w_update[{...}] must be finite reals of shape ({...}, {...}); got shape {...} dtype {...}` |
| `ValueError` | `{...}: w_update['decay'] must be finite reals of shape ({...},); got shape {...} dtype {...}` |
| `ValueError` | `{...}: w_update['simple'] is missing {...}; simple mode {...} with lateral_on={...}, topdown_on={...` |
| `ValueError` | `{...}: w_update['simple'][{...}] is set but simple mode {...} with lateral_on={...}, topdown_on={...` |
| `ValueError` | `{...}: boundary must be an EventScanBoundary; got {...}` |
| `ValueError` | `{...}: boundary reset entries must be (name, value) pairs; got {...}` |
| `ValueError` | `{...}: boundary names unknown state field(s) {...}; allowed fields are {...}` |
| `ValueError` | `{...}: boundary declares state field(s) {...} more than once` |
| `ValueError` | `{...}: boundary leaves state field(s) {...} undeclared; every field must carry or reset` |
| `ValueError` | `{...}: reset value {...} is only defined for 'TH', not {...}` |
| `ValueError` | `{...}: boundary reset of {...} must be a finite real number, a bool or {...}; got {...}` |
| `ValueError` | `{...}: threshold reset needs shape {...}; got {...}` |
| `ValueError` | `{...} events must have shape (M, 4) with [t_us, x, y, p] rows; got shape {...}` |
| `ValueError` | `{...} events must hold integers; got {...}` |
| `ValueError` | `{...} tables must be the event_scan_tables mapping; got {...}` |
| `ValueError` | `{...} tables must carry whole 'ns'/'nc' geometry; got {...}` |
| `ValueError` | `{...} groups must hold one integer position per cell; got shape {...} dtype {...}` |
| `ValueError` | `{...} groups cover {...} cells but the tables span ns + nc = {...}` |
| `ValueError` | `{...} blocks must be the local_inhibition_blocks mapping; got {...}` |
| `ValueError` | `{...} blocks must carry {...}; got {...}` |
| `ValueError` | `{...} dt_ms must be a real number; got {...}` |
| `ValueError` | `{...} dt_ms must be > 0; got {...}` |
| `ValueError` | `{...} n_steps must be >= 1` |
| `ValueError` | `{...} state must be the event-scan state mapping; got {...}` |
| `ValueError` | `{...} boundary needs a state: the map resets the state it is given` |
| `ValueError` | `{...} return_state must be a bool; got {...}` |
| `ValueError` | `{...} ns/nc must be >= 1; got {...}/{...}` |
| `ValueError` | `{...} lif_simple and lif_complex must be given together: a partial per-NeuronType emitter declaratio` |

**`jaxfne/protocol_c/c3_execution.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown delay_policy {...}` |

**`jaxfne/protocol_c/c3_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `C3 spec must have status FROZEN` |
| `ValueError` | `C3 requires enable_hdp=false` |
| `ValueError` | `C3 requires enable_rbd=false` |
| `ValueError` | `design_matrix.n_conditions mismatch` |
| `ValueError` | `duplicate condition ids` |
| `ValueError` | `{...}: delay_shuffled requires delay_shuffle_reference` |
| `ValueError` | `duration_ms/dt_ms inconsistent with n_steps` |
| `ValueError` | `C3 spec must freeze prospective_run_authorized=false at specification` |

**`jaxfne/protocol_c/estimator.py`**

| Class | Message |
|---|---|
| `ValueError` | `invalid band {...} for fs={...}` |
| `ValueError` | `phi must be (T, N); got {...}` |
| `ValueError` | `positions rows must match phi sites` |

**`jaxfne/protocol_d_biological_rbs/d0_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `no D1 coordinate declared in rbs_vector_grammar` |
| `ValueError` | `D0 spec must have status FROZEN` |
| `ValueError` | `D0 spec must have checkpoint D0` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `D0 requires enable_hdp=false` |
| `ValueError` | `D0 requires dot_W=0` |
| `ValueError` | `D0 must declare exactly one D1 coordinate` |
| `ValueError` | `D1 coordinate must be H_K per frozen specification` |
| `ValueError` | `static sweep values must be {...}, got {...}` |
| `ValueError` | `D0 izhikevich_realization.map must be b_eff = H_K * b` |
| `ValueError` | `D4 must not be mandatory for 0.4.17` |
| `ValueError` | `{...} must freeze implementation_authorized=false at D0` |
| `ValueError` | `D0 must freeze implementation_authorized=false at specification` |
| `ValueError` | `H_K is_not must include {...}` |

**`jaxfne/protocol_d_biological_rbs/d1_execution.py`**

| Class | Message |
|---|---|
| `ValueError` | `drive onset_ms must be finite and >= 0; got {...}` |
| `ValueError` | `drive duration_ms must be finite and > 0; got {...}` |

**`jaxfne/protocol_d_biological_rbs/d1_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `D1 spec must have status FROZEN` |
| `ValueError` | `D1 spec must have checkpoint D1` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `D1 requires dH_K/dt=0` |
| `ValueError` | `D1 requires dot_W=0` |
| `ValueError` | `D1 coupling must be b_eff = H_K * b` |
| `ValueError` | `static sweep values must be {...}, got {...}` |
| `ValueError` | `duration_ms/dt_ms inconsistent with n_steps` |
| `ValueError` | `D1 requires noise_scale=0 for containment gate` |
| `ValueError` | `D1 spec must authorize implementation` |
| `ValueError` | `missing gate {...}` |

**`jaxfne/protocol_d_biological_rbs/d2a_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `D2a spec must have status FROZEN` |
| `ValueError` | `D2a spec must have checkpoint D2a` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `D2a requires kappa_K=0` |
| `ValueError` | `D2a requires F1_linear dynamics` |
| `ValueError` | `duration_ms/dt_ms inconsistent with n_steps` |
| `ValueError` | `tau_k_ms must be positive` |
| `ValueError` | `D2a spec must authorize implementation` |
| `ValueError` | `D2b must remain specified_not_authorized at D2a` |

**`jaxfne/protocol_d_biological_rbs/d2b_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `D2b spec must have status FROZEN` |
| `ValueError` | `D2b spec must have checkpoint D2b` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `H_A reference must be 0` |
| `ValueError` | `H_K reference must be 1` |
| `ValueError` | `H_A must not be classified as an ion` |
| `ValueError` | `D2b requires tau_A < tau_K` |
| `ValueError` | `D2b primary candidate requires kappa_AK > 0` |
| `ValueError` | `H_K update must use H_A^n causal ordering` |
| `ValueError` | `D2b must be specification_only at freeze` |
| `ValueError` | `D2b must not authorize implementation at specification` |
| `ValueError` | `must prohibit preregistered firing-down phenotype` |

**`jaxfne/protocol_d_biological_rbs/d3_execution.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown null_arm {...}` |

**`jaxfne/protocol_d_biological_rbs/d3_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `D3 spec must have status FROZEN` |
| `ValueError` | `D3 spec must have checkpoint D3` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `D3 phenomenon_label must not use fatigue` |
| `ValueError` | `D3 must list fatigue in avoid_as_formal_mechanism` |
| `ValueError` | `D3 classification labels must be ADAPTATION, NO_ADAPTATION, UNRESOLVED` |
| `ValueError` | `D3 null hierarchy must be N0, N1, N2, D` |
| `ValueError` | `D3 primary contrast must be D_minus_N2` |
| `ValueError` | `D3 requires three prospective recovery intervals` |
| `ValueError` | `recovery intervals must be strictly increasing` |
| `ValueError` | `short recovery interval must equal 2*tau_A` |
| `ValueError` | `medium recovery interval must equal tau_K` |
| `ValueError` | `duration_ms/dt_ms inconsistent with n_steps` |
| `ValueError` | `cell_grid n_cells must be {...}` |
| `ValueError` | `D3 must be specification_only at freeze` |
| `ValueError` | `D3 must not authorize implementation at specification` |
| `ValueError` | `must prohibit D3 implementation in specification checkpoint` |
| `ValueError` | `must prohibit fatigue as formal mechanism label` |

**`jaxfne/protocol_d_biological_rbs/d_closure.py`**

| Class | Message |
|---|---|
| `ValueError` | `protocol_d_closed must be true` |
| `ValueError` | `Protocol D must close at D3` |
| `ValueError` | `D4 must remain not_authorized` |
| `ValueError` | `next milestone must be 0.4.17-E` |

**`jaxfne/protocol_e_integration/e0_1_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `E0.1 spec must have status FROZEN` |
| `ValueError` | `E0.1 spec must have checkpoint E0.1` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `E0.1 ladder must be E1..E5` |
| `ValueError` | `E0.1 reduction contracts incomplete` |
| `ValueError` | `reduction tests must be required` |
| `ValueError` | `E1 authorization policy mismatch` |
| `ValueError` | `E0.1 must be specification_only at freeze` |
| `ValueError` | `E0.1 must not authorize its own implementation` |
| `ValueError` | `must prohibit monolithic E implementation` |

**`jaxfne/protocol_e_integration/e0_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `E0 spec must have status FROZEN` |
| `ValueError` | `E0 spec must have checkpoint E0` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `E0 must be composition_not_phenotype_manufacturing` |
| `ValueError` | `W3 must remain unresolved` |
| `ValueError` | `D4 must not be authorized in E0` |
| `ValueError` | `D3 NO_ADAPTATION must be frozen` |
| `ValueError` | `E0 must be specification_only at freeze` |
| `ValueError` | `E0 must not authorize implementation at specification` |
| `ValueError` | `must prohibit closed-loop HDP` |
| `ValueError` | `must prohibit D4 authorization` |

**`jaxfne/protocol_e_integration/e1_execution.py`**

| Class | Message |
|---|---|
| `ValueError` | `n_neurons_per_area must divide evenly across laminar layers` |
| `ValueError` | `E1 realizes no noise; spec noise_scale must be 0.0, got {...}` |
| `ValueError` | `identity map failed to cover all neuron_table rows` |

**`jaxfne/protocol_e_integration/e1_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `E1 spec must have status FROZEN` |
| `ValueError` | `E1 spec must have checkpoint E1` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `E1 must not enable RBS` |
| `ValueError` | `E1 requires zero delay` |
| `ValueError` | `E1 requires exactly two areas` |
| `ValueError` | `E1 heterogeneity semantics must be frozen` |
| `ValueError` | `E1 must authorize implementation` |
| `ValueError` | `must prohibit RBS at E1` |
| `ValueError` | `must prohibit nonzero delays at E1` |

**`jaxfne/protocol_e_integration/e2_execution.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown edge_class {...} for delay attachment` |

**`jaxfne/protocol_e_integration/e2_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `E2 spec must have status FROZEN` |
| `ValueError` | `E2 spec must have checkpoint E2` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `E2 must declare R_E2_to_E1 reduction contract` |
| `ValueError` | `E2 must freeze p_local=0.2 from E1 receipt` |
| `ValueError` | `p_local must document E1-receipt provenance` |
| `ValueError` | `E2 delay classes must match E1 provenance classes` |
| `ValueError` | `E2 requires tau_local <= tau_FF < tau_FB` |
| `ValueError` | `E2 nonzero-delay realization requires tau_FF, tau_FB > 0` |
| `ValueError` | `delay_steps mismatch for {...}` |
| `ValueError` | `E2 requires exactly eight gates G1–G8` |
| `ValueError` | `G1 must be E1 reduction gate` |
| `ValueError` | `G8 must prohibit scientific overinterpretation` |
| `ValueError` | `E2 must be specification_only at freeze` |
| `ValueError` | `E2 must not authorize implementation at specification freeze` |
| `ValueError` | `must prohibit RBS at E2` |
| `ValueError` | `must prohibit retroactive p_local as E1 preregistration` |

**`jaxfne/protocol_e_integration/e3_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `E3 spec must have status FROZEN` |
| `ValueError` | `E3 spec must have checkpoint E3` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `E3 must declare R_E3_to_E2 reduction contract` |
| `ValueError` | `E3 must defer D2b explicitly` |
| `ValueError` | `E3 must use D1 coupling map b_eff = H_K * b` |
| `ValueError` | `E3 requires dot_W = 0` |
| `ValueError` | `E3 owner population must have at least one node` |
| `ValueError` | `ownership n_nodes must match flat_indices length` |
| `ValueError` | `E3 owner flat_indices must match E1 identity_map lookup (expected {...}, got {...})` |
| `ValueError` | `E3 requires E3-null and E3-dynamic modes` |
| `ValueError` | `tau_K_ms must be positive` |
| `ValueError` | `E3 must inherit D2a tau_K_ms=100` |
| `ValueError` | `E3 requires exactly nine gates G1–G9` |
| `ValueError` | `G1 must be E2 reduction gate` |
| `ValueError` | `G9 must prohibit phenotype claims` |
| `ValueError` | `E3 must be specification_only at freeze` |
| `ValueError` | `E3 must not authorize implementation at specification freeze` |
| `ValueError` | `must prohibit D2b activity writing at E3` |
| `ValueError` | `must prohibit hierarchy modification for RBS ownership` |

**`jaxfne/protocol_e_integration/e4_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `E4 spec must have status FROZEN` |
| `ValueError` | `E4 spec must have checkpoint E4` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `E4 must declare R_E4_to_E3 reduction contract` |
| `ValueError` | `E4 must declare probe-independent neural/source trajectory contract` |
| `ValueError` | `trajectory invariance must include Q` |
| `ValueError` | `E4 must inherit Experiment A b0 protocol` |
| `ValueError` | `E4 must prohibit integrated-model source variant` |
| `ValueError` | `E4 must reuse Experiment A canonical relative source contract` |
| `ValueError` | `E4 primary evidence must freeze lfp_ref field operator` |
| `ValueError` | `E4 primary probes must match conservative Experiment A subset` |
| `ValueError` | `EEG/MEG must be deferred analysis_only operators` |
| `ValueError` | `E4 requires exactly ten gates G1–G10` |
| `ValueError` | `G1 must be E3 reduction / neural invariance gate` |
| `ValueError` | `G10 must prohibit phenotype claims` |
| `ValueError` | `E4 primary neural path must use E3-null reference mode` |
| `ValueError` | `E4 simulation_policy neural_mode must be E3-null` |
| `ValueError` | `E4 must inherit E3 n_steps=2000` |
| `ValueError` | `E4 must be specification_only at freeze` |
| `ValueError` | `E4 must not authorize implementation at specification freeze` |
| `ValueError` | `must prohibit new source definition at E4` |
| `ValueError` | `must prohibit observation feedback into E3 dynamics` |
| `ValueError` | `must exclude EEG/MEG from primary evidence` |

**`jaxfne/protocol_e_integration/e5_execution.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown E5 arm {...}` |

**`jaxfne/protocol_e_integration/e5_protocol.py`**

| Class | Message |
|---|---|
| `ValueError` | `E5 spec must have status FROZEN` |
| `ValueError` | `E5 spec must have checkpoint E5` |
| `ValueError` | `protocol_id must be {...}` |
| `ValueError` | `E5 must freeze arms N0, N1, D` |
| `ValueError` | `N0 must use H_K=1 reference` |
| `ValueError` | `N1 must disable Gamma_H` |
| `ValueError` | `D must enable Gamma_H` |
| `ValueError` | `N1 must use H_K(0)=1.2 on owners` |
| `ValueError` | `N1 and D must share H_K initial condition` |
| `ValueError` | `E5 primary contrast must be D - N1` |
| `ValueError` | `E5 owner flat_indices must match frozen E3 ownership` |
| `ValueError` | `E5 result classification enum mismatch` |
| `ValueError` | `E5 must declare outcomes not required for close` |
| `ValueError` | `HIERARCHICAL_PROPAGATION must not be required for E5 close` |
| `ValueError` | `E5 must inherit seeds [11, 12, 13]` |
| `ValueError` | `E5 must freeze sample size policy` |
| `ValueError` | `E5 requires exactly ten gates G1–G10` |
| `ValueError` | `E5 must declare zero-architecture evidence rule` |
| `ValueError` | `E5 must be specification_only at freeze` |
| `ValueError` | `E5 must not authorize implementation at specification freeze` |
| `ValueError` | `must prohibit new TFNE architecture at E5` |
| `ValueError` | `must prohibit spectral pipeline at E5` |
| `ValueError` | `must prohibit E6 checkpoint` |

**`jaxfne/public_surface.py`**

| Class | Message |
|---|---|
| `ValueError` | `validate_hdp_params_semantics: {...}` |
| `ValueError` | `validate_hdp_params_semantics: {...}` |

**`jaxfne/sanity_delta.py`**

| Class | Message |
|---|---|
| `ValueError` | `hierarchy {...} is not realized: the runtime wires only {...}` |
| `ValueError` | `stimulus_frequency_hz={...} is not realized: the runtime applies no stimulus frequency; only {...} i` |
| `ValueError` | `Invalid runtime_mode: {...}` |

**`jaxfne/solvers.py`**

| Class | Message |
|---|---|
| `ValueError` | `[{...}, {...}] is not a whole number of dt={...} steps (P-028): the kernel would run {...} steps. Ad` |
| `ValueError` | `Unknown solver_type: {...} (P-028). Use "dopri5", "tsit5", or None (default tsit5).` |
| `ValueError` | `rtol/atol are not consumed by method='euler' (P-028): got rtol={...}, atol={...}. Use method='diffra` |
| `ValueError` | `Unknown solver method: {...}` |

**`jaxfne/stimulus.py`**

| Class | Message |
|---|---|
| `ValueError` | `duration_ms={...} is not a whole number of dt_ms={...} steps: the drive would silently drop the rema` |

**`jaxfne/streaming.py`**

| Class | Message |
|---|---|
| `ValueError` | `dt_ms={...} exceeds SYN_TAU_MS={...} (minor batch): the synaptic decay factor (1 - dt/tau) would go ` |
| `ValueError` | `dt_ms={...} exceeds {...}={...}: the STDP trace decay factor (1 - dt/tau) would go negative.` |

**`jaxfne/tasks/_registry.py`**

| Class | Message |
|---|---|
| `ValueError` | `task name must be a str, got {...}` |
| `ValueError` | `task {...} is already registered` |
| `ValueError` | `task {...} declares {...} {...}; expected an int >= 1` |
| `ValueError` | `task {...} declares gamma {...}; expected a real 0 <= gamma < 1` |
| `ValueError` | `unknown registered task {...}; known: {...}` |

**`jaxfne/tfne.py`**

| Class | Message |
|---|---|
| `TFNEError` | `object name must be capitalized; got {...}` |
| `TFNEError` | `object name {...} is reserved (composition operator or H-state tensor); choose another name` |
| `TFNEError` | `{...} name must be lowercase; got {...}` |
| `TFNEError` | `{...} name {...} is reserved (boundary or state symbol); choose another name` |
| `TFNEError` | `unexpected character {...} at offset {...}` |
| `TFNEError` | `unexpected end of input` |
| `TFNEError` | `expected {...} {...}; got {...}` |
| `TFNEError` | `duplicate rule definition {...}` |
| `TFNEError` | `duplicate definition {...}` |
| `TFNEError` | `only one system expression per program` |
| `TFNEError` | `rule kind must be O or X; got {...}` |
| `TFNEError` | `rule {...}[{...}] body statements need an explicit direction ('>', '<', '<>'); got {...}` |
| `TFNEError` | `rule {...}[{...}] statement options support only mech=NAME and delay=MS; got {...}` |
| `TFNEError` | `rule {...}[{...}] duplicate statement option {...}` |
| `TFNEError` | `rule metavariables are $L and $R; got ${...}` |
| `TFNEError` | `interface selection on ${...} supports only .out / .in; got .{...}` |
| `TFNEError` | `expected a rule endpoint; got {...}` |
| `TFNEError` | `boundary qualifier {...} is not a type; write [NAME] or [type=NAME]` |
| `TFNEError` | `expected boundary name after ':'` |
| `TFNEError` | `expected [NAME] or [type=NAME] after the right boundary` |
| `TFNEError` | `prefix rule application is O[k](...) (S6); X[k](...) is not defined — X-joined replication is A^{nX}` |
| `TFNEError` | `O[k](...) prefix applies to plain A^n replication (S6); got {...}` |
| `TFNEError` | `replication count must be an integer` |
| `TFNEError` | `replication relation must be O or X, as in A^{nO} or A^{nX}; got {...}` |
| `TFNEError` | `replication count must be an integer` |
| `TFNEError` | `replication count must be >= 1` |
| `TFNEError` | `replication (A^n) applies to a single named reference` |
| `TFNEError` | `expected ';' or '}' in composite; got {...}` |
| `TFNEError` | `unexpected {...} here (reserved composition operator)` |
| `TFNEError` | `only one [key] selection per reference` |
| `TFNEError` | `unexpected token {...} in expression` |
| `TFNEError` | `unexpected value token {...}` |
| `TFNEError` | `expected number; got {...}` |
| `TFNEError` | `trailing input at {...}` |
| `TFNEError` | `cannot emit {...}` |
| `TFNEError` | `boolean property values are not supported` |
| `TFNEError` | `cannot emit value {...}` |
| `TFNEError` | `duplicate TFNE path {...}` |
| `TFNEError` | `empty composite group at {...}` |
| `TFNEError` | `empty composite group at {...}` |
| `TFNEError` | `cannot expand {...}` |
| `TFNEError` | `{...} is a reserved operator` |
| `TFNEError` | `{...}: specialization base {...} undefined` |
| `TFNEError` | `{...}: specialization base {...} must declare properties` |
| `TFNEError` | `definition {...} expands to nothing` |
| `TFNEError` | `projection endpoints must be references, selections or groups` |
| `TFNEError` | `selection {...} not a member of {...}` |
| `TFNEError` | `selection {...} matches no realized object` |
| `TFNEError` | `connection rule {...} undefined` |
| `TFNEError` | `rule {...} must declare direction ('>', '<', '<>'); the algebra fixes no default direction for O[k]/` |
| `TFNEError` | `rule {...} declares both a body and a top-level direction; direction lives in the body statements` |
| `TFNEError` | `TFNE path {...} has no realized slice` |
| `TFNEError` | `neuron id {...} out of range` |
| `TFNEError` | `relation {...} realized no edges` |
| `TFNEError` | `edge index {...} out of range` |
| `TFNEError` | `relation {...} unknown` |
| `TFNEError` | `E_DELAY_OUT_OF_RANGE: rule {...} delay must be a number in ms; got {...}` |
| `TFNEError` | `E_DELAY_OUT_OF_RANGE: rule {...} delay must be finite and >= 0 ms; got {...}` |
| `TFNEError` | `rule {...} {...} must be true or false; got {...}` |
| `TFNEError` | `invalid direction {...}` |
| `TFNEError` | `mechanism must be a name` |
| `TFNEError` | `probability {...} outside [0,1]` |
| `TFNEError` | `weight maps are not supported; use a scalar` |
| `TFNEError` | `rule weight must be finite` |
| `TFNEError` | `E_DELAY_ROUNDS_TO_ZERO: rule {...} declares delay_ms={...} at dt_ms={...}, which rounds to 0 steps; ` |
| `TFNEError` | `E_DELAY_OUT_OF_RANGE: rule {...}: {...}` |
| `TFNEError` | `relation scope {...} matches no realized neurons` |
| `TFNEError` | `relation {...} has unresolved direction {...}` |
| `TFNEError` | `neuron {...} outside member blocks of {...}` |
| `TFNEError` | `E_GEOMETRY_AMBIGUOUS: leaves in layer {...}/{...} declare different {...} domains ({...}); one sampl` |
| `TFNEError` | `realization carries no explicit model to construct from` |
| `TFNEError` | `E_DELAY_COMPOSITION_AMBIGUOUS: composed areas share delay rule(s) {...}; one composed record cannot ` |
| `TFNEError` | `E_DELAY_COMPOSITION_AMBIGUOUS: composed delay blocks disagree on dt_ms ({...}); steps realized at di` |
| `TFNEError` | `mechanism must be a name` |
| `TFNEError` | `mechanism must be a name` |

**`jaxfne/tutorial_utils.py`**

| Class | Message |
|---|---|
| `ValueError` | `dt_ms must be positive` |
| `ValueError` | `hann_absolute_psd expects a 1-D signal` |
| `ValueError` | `dt_ms must be positive` |
| `ValueError` | `unsupported window {...}; expected 'hann' or 'rectangular'` |
| `ValueError` | `dt_ms must be positive` |
| `ValueError` | `absolute_band_power expects 1-D or 2-D input` |
| `ValueError` | `select_neurons requires a model with a callable neuron_table; got {...} with no usable neuron_table` |
| `ValueError` | `select_neurons could not read model.neuron_table(): {...}` |
| `ValueError` | `select_neurons got no neuron table (model.neuron_table() returned None)` |
| `ValueError` | `Row must have positive mass; got sum={...}` |
| `ValueError` | `layers must be non-empty` |
| `ValueError` | `cell_types must be non-empty` |
| `ValueError` | `{...}.noise must be nonnegative; got {...}` |
| `ValueError` | `{...}.drive must be finite; got {...}` |
| `ValueError` | `{...}.{...} must be finite; got {...}` |
| `ValueError` | `simulate_laminar_trials: inhibitory cell types {...} are outside the same-layer rule (PV/SST/VIP onl` |

**`jaxfne/util.py`**

| Class | Message |
|---|---|
| `ValueError` | `validate_runtime_config: {...}` |
| `ValueError` | `merge_runtime_configs requires at least one RuntimeConfig` |
| `ValueError` | `validate_neuronal_tensor: {...}` |
| `ValueError` | `validate_model: {...}` |
| `ValueError` | `canonical_compact_summary requires Model with Izhikevich emitter (v0)` |

**`jaxfne/validation.py`**

| Class | Message |
|---|---|
| `ValueError` | `Invalid mode '{...}'. Allowed: {...}` |
| `ValueError` | `target must be a non-empty string` |
| `ValueError` | `Invalid operator_path: {...}` |
| `ValueError` | `Invalid operator_path: {...}` |
| `ValueError` | `Invalid operator_path: {...}` |
| `ValueError` | `Invalid operator_path: {...}` |
| `ValueError` | `Invalid operator_path: {...}. Allowed: 'proxy', 'physical_candidate'` |
| `ValueError` | `recompilation_guard mode must be 'warning', 'exception', or 'off'; got {...}` |

**`jaxfne/vis/atlas_suite.py`**

| Class | Message |
|---|---|
| `ValueError` | `INVALID_TIME_GRID ({...})` |
| `ValueError` | `render_atlas needs signals; it never simulates. Use build_atlas to simulate first, or pass the signa` |
| `ValueError` | `render_atlas: signals carry no time grid and no dt_ms was given` |

**`jaxfne/vis/canonical.py`**

| Class | Message |
|---|---|
| `ValueError` | `backend must be one of {...}, got {...}` |

**`jaxfne/vis/column_viewer.py`**

| Class | Message |
|---|---|
| `ValueError` | `model.params['edge_list'] is missing — construct() must have run` |

**`jaxfne/vis/core.py`**

| Class | Message |
|---|---|
| `ValueError` | `binned_population_rate_hz expects (n_steps, n_units), got {...}` |
| `ValueError` | `dt_ms must be positive, got {...}` |
| `ValueError` | `bin_ms must be positive, got {...}` |
| `ValueError` | `welch_psd needs at least one sample along axis 0` |

**`jaxfne/vis/evidence_export.py`**

| Class | Message |
|---|---|
| `ValueError` | `evidence export supports png only, got {...}` |

**`jaxfne/vis/exporters.py`**

| Class | Message |
|---|---|
| `ValueError` | `unsupported plotly export format: {...}` |
| `ValueError` | `matplotlib figures cannot export to html` |

**`jaxfne/vis/fields.py`**

| Class | Message |
|---|---|
| `ValueError` | `signals.sources is required for this proxy readout` |
| `ValueError` | `signals.field.lfp_proxy is required for this figure` |
| `ValueError` | `signals.field.csd_proxy is required for this figure` |
| `ValueError` | `Cannot generate spectrolaminar profile: signals.field is None. Ensure that record_fields was set to ` |

**`jaxfne/vis/figure_theme.py`**

| Class | Message |
|---|---|
| `ValueError` | `scheme must be 'light' or 'dark', not {...}` |
| `ValueError` | `plotly's page has no single </body> to hold the theme script` |

**`jaxfne/vis/network3d.py`**

| Class | Message |
|---|---|
| `ValueError` | `neuron_metadata is required for circuit3d` |
| `ValueError` | `visualize_network_3d: show_edges=True draws a networkx graph's edges, and {...} has none; jaxfne.vis` |
| `ValueError` | `visualize_network_3d: {...}={...} is not one of {...}` |
| `ValueError` | `visualize_network_3d: coordinate_unit and display_unit must both be 'relative' or both physical unit` |
| `ValueError` | `visualize_network_3d: no node data found in input.` |

**`jaxfne/vis/network_inspect.py`**

| Class | Message |
|---|---|
| `ValueError` | `unknown theme {...}; known: {...} or pass a Theme instance` |
| `ValueError` | `signals is required` |
| `ValueError` | `spikes must be (n_time, n_neurons); got shape {...}` |
| `ValueError` | `neuron_table() is empty; nothing to describe` |
| `ValueError` | `style must be 'blocks' or 'columns', got {...}` |
| `ValueError` | `no units selected; check the areas argument` |
| `ValueError` | `window {...}-{...} ms outside the recorded {...} ms at dt {...} ms` |

**`jaxfne/vis/plotly/_common.py`**

| Class | Message |
|---|---|
| `ValueError` | `signals.field is None — probes(['LFP','CSD',...]) was not requested` |
| `ValueError` | `signals.field is None` |

**`jaxfne/vis/plotly/network.py`**

| Class | Message |
|---|---|
| `ValueError` | `edge_color_by must be None or 'source_class', got {...}` |
| `ValueError` | `edge_color_by requires show_edges=True` |
| `ValueError` | `color_by must be one of cell_type/layer/area, got {...}` |
| `ValueError` | `n_width_classes must be a positive int, got {...}` |
| `ValueError` | `plot_area_graph needs a model with at least two areas` |
| `ValueError` | `plot_area_graph needs model.params['edge_list']` |

**`jaxfne/vis/pseudogenome_viewer.py`**

| Class | Message |
|---|---|
| `ValueError` | `model.params['edge_list'] missing — construct must have run` |

**`jaxfne/vis/rasters.py`**

| Class | Message |
|---|---|
| `ValueError` | `plot_spike_rasters: square input {...} — neuron axis is ambiguous. Pass config_params['n_neurons_axi` |
| `ValueError` | `No spikes data found in signals.` |

**`jaxfne/vis/traces.py`**

| Class | Message |
|---|---|
| `ValueError` | `signals.sources is required for this proxy readout` |
| `ValueError` | `No spikes data found for EMM proxy calculation.` |
| `ValueError` | `No sources data found for EMM proxy calculation.` |

**`jaxfne/vis/visualize.py`**

| Class | Message |
|---|---|
| `ValueError` | `no lfp_proxy` |
| `ValueError` | `backend must be 'static', 'plotly', or 'both', got {...}` |

**`jaxfne/w1a_omega_plasticity.py`**

| Class | Message |
|---|---|
| `ValueError` | `tau_w must be > 0` |
| `ValueError` | `w0 must be > 0` |
| `ValueError` | `dt must be > 0` |
| `ValueError` | `lambda_w must be >= 0` |
| `ValueError` | `omega_infinity requires lambda_w > 0` |
| `ValueError` | `n_steps must be >= 0` |
| `ValueError` | `t must be >= 0` |
| `ValueError` | `t must be >= pulse_duration in forgetting phase` |

**`jaxfne/w1b_shadow_plasticity.py`**

| Class | Message |
|---|---|
| `ValueError` | `perturbation_step must be in [0, {...})` |
| `ValueError` | `t must be >= 0` |
| `ValueError` | `use omega_f1_no_forgetting_limit for lambda_w=0` |
| `ValueError` | `omega_f1_no_forgetting_limit requires lambda_w=0` |

**`jaxfne/w3_stability_analysis.py`**

| Class | Message |
|---|---|
| `ValueError` | `no real silent fixed point for given drive` |

<!-- ERRORS-INDEX:END -->
