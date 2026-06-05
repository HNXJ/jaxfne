# 02_PUBLIC_API_CONTRACT

Canonical import: `import jaxfne as jtfne`. The manuscript/context states this canonical import and the package identity as a compact JAX-native source-to-field/readout scaffold (`2026_jtfne_arxiv.txt:L48`). Live audit reports `__all__=146` and a `sys` namespace leak in `Pasted markdown.md:L? / L140`.

## Exported root surface from uploaded ZIP

Evidence: `jaxfne/__init__.py:LNone` defines `__all__` in the ZIP; this ZIP is stale versus live according to `Pasted markdown.md:L46`.

|name|signature / form|evidence|stability note|
|---|---|---|---|
|compilation_registry|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|AxisSpec|class AxisSpec|jaxfne/core.py:L2470|stable/exported|
|BasisSpec|class BasisSpec|jaxfne/core.py:L2529|stable/exported|
|BridgeSpec|class BridgeSpec|jaxfne/bridges.py:L27|stable/exported|
|ConfigValidationResult|class ConfigValidationResult|jaxfne/core.py:L5669|stable/exported|
|Configuration|class Configuration|jaxfne/core.py:L462|stable/exported|
|default_basis_spec|def default_basis_spec() -> BasisSpec:|jaxfne/core.py:L2651|stable/exported|
|DatasetSpec|class DatasetSpec|jaxfne/core.py:L1614|stable/exported|
|JaxFNEConfig|class JaxFNEConfig|jaxfne/core.py:L5590|stable/exported|
|JaxleyEmitterBridge|class JaxleyEmitterBridge|jaxfne/bridges.py:L48|stable/exported|
|JaxleyBridge|class JaxleyBridge|jaxfne/bridges.py:L428|stable/exported|
|JaxleyTraceSpec|class JaxleyTraceSpec|jaxfne/bridges.py:L77|stable/exported|
|hh_numpy_reference_trace|def hh_numpy_reference_trace(|jaxfne/bridges.py:L361|stable/exported|
|LaminarPopulation|class LaminarPopulation|jaxfne/core.py:L2021|stable/exported|
|LaminarSourceGeometry|class LaminarSourceGeometry|jaxfne/core.py:L2078|stable/exported|
|MatrixParameterSpec|class MatrixParameterSpec|jaxfne/core.py:L37|stable/exported|
|matrix_parameter|def matrix_parameter(|jaxfne/core.py:L68|stable/exported|
|Model|class Model|jaxfne/core.py:L2714|stable/exported|
|Objective|class Objective|jaxfne/core.py:L1530|stable/exported|
|Paradigm|class Paradigm|jaxfne/experimental_hpc/contracts.py:L443|experimental/exported; verify contract status|
|ParadigmCondition|class ParadigmCondition|jaxfne/paradigm.py:L31|stable/exported|
|ParadigmEvent|class ParadigmEvent|jaxfne/paradigm.py:L6|stable/exported|
|Probe|class Probe|jaxfne/core.py:L1489|stable/exported|
|ObjectiveReport|class ObjectiveReport|jaxfne/core.py:L1841|stable/exported|
|ReadoutResult|class ReadoutResult|jaxfne/core.py:L1796|stable/exported|
|ReadoutSpec|class ReadoutSpec|jaxfne/core.py:L1761|stable/exported|
|RunReceipt|class RunReceipt|jaxfne/core.py:L1884|stable/exported|
|RuntimeConfig|class RuntimeConfig|jaxfne/core.py:L1285|stable/exported|
|Signal|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|Signals|class Signals|jaxfne/core.py:L1496|stable/exported|
|Simulation|class Simulation|jaxfne/core.py:L1448|stable/exported|
|StimulusSchedule|class StimulusSchedule|jaxfne/core.py:L1946|stable/exported|
|SurrogateConfig|class SurrogateConfig|jaxfne/core.py:L1416|stable/exported|
|TrialBatch|class TrialBatch|jaxfne/core.py:L1698|stable/exported|
|TrialBatchResult|class TrialBatchResult|jaxfne/core.py:L1740|stable/exported|
|TrialResult|class TrialResult|jaxfne/core.py:L1716|stable/exported|
|TrialSpec|class TrialSpec|jaxfne/core.py:L1679|stable/exported|
|TuneResult|class TuneResult|jaxfne/core.py:L104|stable/exported|
|config_to_configuration|def config_to_configuration(cfg: JaxFNEConfig) -> Configuration:|jaxfne/core.py:L6022|stable/exported|
|config_to_geometry|def config_to_geometry(cfg: JaxFNEConfig) -> Optional[LaminarSourceGeometry]:|jaxfne/core.py:L5927|stable/exported|
|config_to_simulation|def config_to_simulation(cfg: JaxFNEConfig) -> Simulation:|jaxfne/core.py:L5888|stable/exported|
|config_to_trial_batch|def config_to_trial_batch(|jaxfne/core.py:L6056|stable/exported|
|config_truth_boundary|def config_truth_boundary(cfg: JaxFNEConfig) -> dict[str, Any]:|jaxfne/core.py:L5837|stable/exported|
|configuration|def configuration() -> Configuration:|jaxfne/core.py:L4611|stable/exported|
|construct|def construct(self) -> dict[str, Any]:|jaxfne/bridges.py:L69|stable/exported|
|dataset_spec|def dataset_spec(**kwargs: Any) -> DatasetSpec:|jaxfne/core.py:L5425|stable/exported|
|enable_x64|def enable_x64() -> dict[str, Any]:|jaxfne/core.py:L5504|stable/exported|
|laminar_source_geometry|def laminar_source_geometry(|jaxfne/core.py:L5480|stable/exported|
|load_config|def load_config(path: Any) -> JaxFNEConfig:|jaxfne/core.py:L5694|stable/exported|
|objective|def objective() -> Objective:|jaxfne/core.py:L4654|stable/exported|
|operator_status|def operator_status() -> dict[str, str]:|jaxfne/core.py:L5058|stable/exported|
|paradigm|def paradigm(name: str = "none") -> Paradigm:|jaxfne/paradigm.py:L139|stable/exported|
|rate_targets|def rate_targets(|jaxfne/core.py:L4658|stable/exported|
|readout_spec|def readout_spec(|jaxfne/core.py:L5395|stable/exported|
|require_jaxley|def require_jaxley():|jaxfne/bridges.py:L14|stable/exported|
|jaxley_trace_to_signals|def jaxley_trace_to_signals(|jaxfne/bridges.py:L186|stable/exported|
|run_receipt|def run_receipt(|jaxfne/core.py:L5377|stable/exported|
|runtime|def runtime(|jaxfne/core.py:L4615|stable/exported|
|runtime_report|def runtime_report(runtime_config: RuntimeConfig \| None = None) -> dict[str, Any]:|jaxfne/core.py:L4646|stable/exported|
|run_trials|def run_trials(|jaxfne/core.py:L5360|stable/exported|
|simulate|def simulate(self, *args: Any, **kwargs: Any) -> Any:|jaxfne/bridges.py:L435|stable/exported|
|simulation|def simulation(**kwargs: Any) -> Simulation:|jaxfne/core.py:L4650|stable/exported|
|standard_visual_omission|def standard_visual_omission() -> Paradigm:|jaxfne/core.py:L5073|stable/exported|
|suite2_celltype_presets|def suite2_celltype_presets() -> dict[str, dict[str, float \| str]]:|jaxfne/core.py:L4739|stable/exported|
|suite2_single_neuron_config|def suite2_single_neuron_config(*, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1, cell_type: str = "E") -> Configuration:|jaxfne/core.py:L4749|stable/exported|
|suite2_four_celltype_config|def suite2_four_celltype_config(*, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1) -> Configuration:|jaxfne/core.py:L4766|stable/exported|
|suite2_net1_config|def suite2_net1_config(|jaxfne/core.py:L4779|stable/exported|
|suite2_v1_v4_config|def suite2_v1_v4_config(|jaxfne/core.py:L4801|stable/exported|
|suite2_simulation|def suite2_simulation(|jaxfne/core.py:L4836|stable/exported|
|suite2_tune_noise_agsdr_adam|def suite2_tune_noise_agsdr_adam(|jaxfne/core.py:L4863|stable/exported|
|suite2_run_bundle|def suite2_run_bundle(model: Model, *, seed: int = 7, duration_ms: float = 1000.0, dt_ms: float = 0.1, noise_amplitude: float \| None = None) -> dict[str, Any]:|jaxfne/core.py:L4956|stable/exported|
|stimulus_schedule|def stimulus_schedule(|jaxfne/core.py:L5435|stable/exported|
|surrogate_config|def surrogate_config(**kwargs: Any) -> SurrogateConfig:|jaxfne/core.py:L5430|stable/exported|
|trial_batch|def trial_batch(|jaxfne/core.py:L5312|stable/exported|
|validate_config|def validate_config(cfg: JaxFNEConfig) -> ConfigValidationResult:|jaxfne/core.py:L5735|stable/exported|
|vis|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|with_emitter_parameters|def with_emitter_parameters(|jaxfne/core.py:L4581|stable/exported|
|build_tutorial_laminar_column|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|select_neurons|def select_neurons(model, area: str \| None = None, layer: str \| None = None,|jaxfne/tutorial_utils.py:L266|stable/exported|
|kappa_synchrony|def kappa_synchrony(spikes: np.ndarray, dt_ms: float = 0.1) -> float:|jaxfne/tutorial_utils.py:L310|stable/exported|
|rate_synchrony_targets|def rate_synchrony_targets(|jaxfne/tutorial_utils.py:L368|stable/exported|
|_KNOWN_METRICS|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|EdgeList|class EdgeList|jaxfne/emitters.py:L439|stable/exported|
|EIGNetwork|class EIGNetwork|jaxfne/emitters.py:L165|stable/exported|
|IzhikevichParams|class IzhikevichParams|jaxfne/emitters.py:L79|stable/exported|
|ReceptorSpec|class ReceptorSpec|jaxfne/emitters.py:L39|stable/exported|
|SynapseSpec|class SynapseSpec|jaxfne/emitters.py:L52|stable/exported|
|Emitter|class Emitter|jaxfne/emitters.py:L1029|stable/exported|
|IzhikevichEmitter|class IzhikevichEmitter|jaxfne/emitters.py:L1039|stable/exported|
|GLIFEmitter|class GLIFEmitter|jaxfne/emitters.py:L1086|exported loud stub risk; verify before use|
|LIFEmitter|class LIFEmitter|jaxfne/emitters.py:L1091|exported loud stub risk; verify before use|
|SynapseState|class SynapseState|jaxfne/emitters.py:L1096|stable/exported|
|SynapseLayer|class SynapseLayer|jaxfne/emitters.py:L1100|stable/exported|
|make_edge_list_from_dense|def make_edge_list_from_dense(|jaxfne/emitters.py:L480|stable/exported|
|make_eig_network|def make_eig_network(|jaxfne/emitters.py:L315|stable/exported|
|izhikevich_params_from_labels|def izhikevich_params_from_labels(|jaxfne/emitters.py:L254|stable/exported|
|simulate_edge_recurrent_izhikevich|def simulate_edge_recurrent_izhikevich(|jaxfne/emitters.py:L509|stable/exported|
|simulate_eig_izhikevich|def simulate_eig_izhikevich(|jaxfne/emitters.py:L340|stable/exported|
|simulate_receptor_exponential_izhikevich|def simulate_receptor_exponential_izhikevich(|jaxfne/emitters.py:L654|stable/exported|
|standard_receptor_specs|def standard_receptor_specs() -> dict[str, ReceptorSpec]:|jaxfne/emitters.py:L60|stable/exported|
|standard_receptor_tau_table|def standard_receptor_tau_table(dtype: str = "float32") -> jax.Array:|jaxfne/emitters.py:L624|stable/exported|
|CELL_TYPE_PRESETS|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|DEFAULT_SPIKE_IMPULSE_GAIN|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|RECEPTOR_KINETICS|(imported/constant or unresolved by AST)|jaxfne/__init__.py:L?|exported; verify source before patch|
|compute_conservation_proxy_diagnostics|def compute_conservation_proxy_diagnostics(|jaxfne/fields/proxy.py:L387|stable/exported|
|eeg_proxy_transform|def eeg_proxy_transform(|jaxfne/fields/probes.py:L271|stable/exported|
|emm_proxy_transform|def emm_proxy_transform(|jaxfne/fields/probes.py:L319|stable/exported|
|meg_proxy_transform|def meg_proxy_transform(|jaxfne/fields/probes.py:L295|stable/exported|
|FieldOutput|class FieldOutput|jaxfne/fields/proxy.py:L20|stable/exported|
|project_laminar_sources|def project_laminar_sources(|jaxfne/fields/proxy.py:L88|stable/exported|
|project_sources_to_laminar_field|def project_sources_to_laminar_field(|jaxfne/fields/proxy.py:L189|stable/exported|
|probe_laminar_modes|def probe_laminar_modes(|jaxfne/fields/proxy.py:L518|stable/exported|
|validate_projection_invariants|def validate_projection_invariants(|jaxfne/fields/diagnostics.py:L43|stable/exported|
|validate_source_field_status|def validate_source_field_status(|jaxfne/fields/proxy.py:L315|stable/exported|
|construct_source_tensor|def construct_source_tensor(|jaxfne/fields/proxy.py:L991|stable/exported|
|LinearReadout|class LinearReadout|jaxfne/fields/proxy.py:L959|stable/exported|
|config_hash|def config_hash(self) -> str:|jaxfne/core.py:L5654|stable/exported|
|json_safe|def json_safe(obj: Any) -> Any:|jaxfne/io.py:L15|stable/exported|
|manifest|def manifest(|jaxfne/core.py:L4176|stable/exported|
|save_json|def save_json(obj: Any, path: str \| Path) -> None:|jaxfne/io.py:L177|stable/exported|
|save_receipt|def save_receipt(receipt: Any, path: "str \| Path", *, overwrite: bool = False) -> None:|jaxfne/io.py:L158|stable/exported|
|sha256_file|def sha256_file(path: str \| Path) -> str:|jaxfne/io.py:L55|stable/exported|
|sha256_text|def sha256_text(text: str) -> str:|jaxfne/io.py:L49|stable/exported|
|AGSDR|class AGSDR|jaxfne/optim/core.py:L439|stable/exported|
|AGSDROptimizerSpec|class AGSDROptimizerSpec|jaxfne/optim/core.py:L174|stable/exported|
|AGSDRState|class AGSDRState|jaxfne/optim/agsdr.py:L17|stable/exported|
|GSDRState|class GSDRState|jaxfne/optim/core.py:L771|stable/exported|
|OptimizerSpec|class OptimizerSpec|jaxfne/optim/core.py:L97|stable/exported|
|SDRState|class SDRState|jaxfne/optim/core.py:L754|stable/exported|
|agsdr|def agsdr(|jaxfne/optim/core.py:L239|stable/exported|
|agsdr_transform|def agsdr_transform(|jaxfne/optim/core.py:L1044|stable/exported|
|gsdr|def gsdr(|jaxfne/optim/core.py:L154|stable/exported|
|gsdr_transform|def gsdr_transform(|jaxfne/optim/core.py:L930|stable/exported|
|optax_adam|def optax_adam(|jaxfne/optim/core.py:L332|stable/exported|
|optax_sgd|def optax_sgd(|jaxfne/optim/core.py:L355|stable/exported|
|random_search|def random_search(metadata: Optional[dict[str, Any]] = None) -> OptimizerSpec:|jaxfne/optim/core.py:L321|stable/exported|
|require_optax|def require_optax() -> Any:|jaxfne/optim/core.py:L461|stable/exported|
|sdr_transform|def sdr_transform(|jaxfne/optim/core.py:L805|stable/exported|
|get_sharding_context|def get_sharding_context() -> Optional[dict]:|jaxfne/sharding_utils.py:L119|stable/exported|
|make_candidate_sharding|def make_candidate_sharding(mesh: Mesh) -> NamedSharding:|jaxfne/sharding_utils.py:L75|stable/exported|
|make_population_mesh|def make_population_mesh() -> Optional[Mesh]:|jaxfne/sharding_utils.py:L44|stable/exported|
|make_replicated_sharding|def make_replicated_sharding(mesh: Mesh) -> NamedSharding:|jaxfne/sharding_utils.py:L94|stable/exported|


## Loud stubs / honesty risks to verify live

|path|line|text|
|---|---|---|


## Do NOT use / verify before use

| Name/pattern | Replacement / action | Evidence |
|---|---|---|
| `jtfne.Config.four_celltype` | Use `jtfne.suite2_four_celltype_config(seed=0, duration_ms=..., dt_ms=...)` then `jtfne.construct(cfg)` | verified audit-smoke correction from prior session; bundle should re-run live smoke before mutation |
| `jtfne.Model(cfg)` | Use `jtfne.construct(cfg)` | same as above |
| Unsupported `set_emitter(family='lif'/'glif')` expecting simulation | Must fail loudly or use supported `izhikevich` path | live audit says emitter-family guard was merged/tested at `Pasted markdown.md:L144` |
| Core physical EEG/MEG/calibrated amplitude claims | Use proxy/status wording unless run has geometry/calibration/solver/boundary/gauge/residual/validation | `2026_jtfne_arxiv.txt:L89` |
