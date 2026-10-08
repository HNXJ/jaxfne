# Claim ledger 0.5.5 (DRAFT scaffold, generated from `atlas_coverage.json`)

Status: scaffold for human review (fact stack: claim ledger). State = coverage state; `kind` is a path heuristic (tests only = CAPABILITY, otherwise RESULT; MIXED/UNSET need a human call). Wording strength is the permissible claim class (2026-10-08): `numerical test confirms`, `model produced`, `within specified tolerance`, `tolerance comparison, failures kept`, `under the tested conditions`, `supported only`, `do not promote`. VALIDATED is an evidence-status label, not biological validation. Negative and failed results stay in with the same prominence. Regenerate state and evidence from the coverage file; do not hand-drift them.

| id | claim (Atlas requirement) | kind | state | evidence path | path exists | wording strength |
|---|---|---|---|---|---|---|
| AT-00-R1 | Common measurement vector Y = {X,H,W,Q,Phi_E,Phi_B,SPK,PSD,C,phi,E_reduction,T_compute,M_compute}; absent quantities recorded as OMITTED, never synthesized | CAPABILITY | VALIDATED | `tests/test_atlas_v4_records.py` | yes | numerical test confirms |
| AT-00-R2 | (dt, T, dr) chosen per phenomenon, not one resolution for all | RESULT | VALIDATED | `artifacts/perf/matrix_051.json` | yes | numerical test confirms |
| AT-00-R3 | Inheritance S1 -> S2-4 -> S5-7 -> S8-9 -> S10: each adds declared objects, same computational language | CAPABILITY | VALIDATED | `tests/test_atlas_inheritance_055.py` | yes | numerical test confirms |
| AT-00-R4 | Observable-specific equivalence: reductions compared only on declared observations and tolerances | RESULT | VALIDATED | `artifacts/atlas/results/at_reduction_055.json` | yes | tolerance comparison, failures kept |
| AT-00-R5 | Every figure uses the columns structure -> dynamics -> state/plasticity -> source -> field -> observation -> computation | RESULT | VALIDATED | `artifacts/atlas/figure_columns.py` | yes | numerical test confirms |
| AT-00-R6 | Manuscript progression: one physical neuron -> interacting neurons -> emergent population field -> adaptive interacting areas -> genome-defined multiarea model | RESULT | VALIDATED | `artifacts/atlas/manuscript_progression.py` | yes | numerical test confirms |
| AT-00-R7 | Every simulation defined as data over the public surface; one simulation, many views; no simulation inside visualization | CAPABILITY | VALIDATED | `tests/test_atlas_firewall.py` | yes | numerical test confirms |
| AT-00-R8 | T_compute and M_compute recorded per simulation | RESULT | VALIDATED | `artifacts/perf/matrix_051.json` | yes | numerical test confirms |
| AT-01-R1 | One full HH neuron; current injection from subthreshold to AP | RESULT | VALIDATED | `artifacts/atlas/results/at01_extraction_055.json` | yes | model produced |
| AT-01-R2 | Extract V_m, I_Na, I_K, I_L, Q, Phi(r,t) (I_m not applicable: Jaxley HH has no M-current; Phi relative ionic-current proxy) | RESULT | VALIDATED | `artifacts/atlas/results/at01_extraction_055.json` | yes | model produced |
| AT-01-R3 | Local E/Phi from transmembrane currents | RESULT | VALIDATED | `artifacts/atlas/results/at01_extraction_055.json` | yes | model produced |
| AT-01-R4 | Magnetic field B where justified | RESULT | OUT_OF_SCOPE | `artifacts/atlas/at01_at06_052.py` | yes | do not promote |
| AT-01-R5 | Establish calibration/reference model at physical units | RESULT | OUT_OF_SCOPE | `artifacts/atlas/at01_at06_052.py` | yes | do not promote |
| AT-01-R6 | Reduction M_HH -> M_reduced -> M_population: which quantities survive | RESULT | VALIDATED | `artifacts/atlas/results/at_reduction_055.json` | yes | tolerance comparison, failures kept |
| AT-01-R7 | No claim of full electrodiffusion | RESULT | VALIDATED | `artifacts/atlas/results/at01_055.json` | yes | do not promote |
| AT-02-R1 | Driven pair N1 -> N2 over 10-10^3 um; vary distance, delay (ms), synaptic strength | CAPABILITY | VALIDATED | `tests/test_atlas_at0203_052.py` | yes | numerical test confirms |
| AT-02-R2 | Individual versus superposed fields | CAPABILITY | VALIDATED | `tests/test_atlas_at0203_052.py` | yes | numerical test confirms |
| AT-02-R3 | Distance law of the field | CAPABILITY | VALIDATED | `tests/test_atlas_at0203_052.py` | yes | numerical test confirms |
| AT-03-R1 | E<->I pair oscillator; vary coupling, delay, drive | CAPABILITY | VALIDATED | `tests/test_atlas_at0203_052.py` | yes | numerical test confirms |
| AT-03-R2 | Phase, synchrony, cancellation/reinforcement, frequency-dependent field | CAPABILITY | VALIDATED | `tests/test_atlas_at0203_052.py` | yes | numerical test confirms |
| AT-04-R1 | Geometry/orientation manipulation of a coupled pair | CAPABILITY | VALIDATED | `tests/test_atlas_at04_052.py` | yes | numerical test confirms |
| AT-04-R2 | H perturbation distinguishing correlation X -> Phi from causal state effect on excitability | CAPABILITY | VALIDATED | `tests/test_atlas_at04r2_053.py` | yes | numerical test confirms |
| AT-04-R3 | Field feedback Phi -> X only if physically implemented | RESULT | OUT_OF_SCOPE | `artifacts/atlas/at01_at06_052.py` | yes | do not promote |
| AT-05-R1 | E/I population over 0.1-2 mm; vary N, rho, synchrony rho_sync 0 -> 1 | CAPABILITY | VALIDATED | `tests/test_atlas_at0506_052.py` | yes | numerical test confirms |
| AT-05-R2 | Measure A_Phi(N, rho_sync, r); record coupled vs isolated-twin field (superposition within 1% for the arms run; Phi_N != N Phi_1 not demonstrated) | RESULT | VALIDATED | `artifacts/atlas/results/at05_isolation_055.json` | yes | within specified tolerance |
| AT-05-R3 | Coherent vs incoherent summation of microscopic sources | CAPABILITY | VALIDATED | `tests/test_atlas_at0506_052.py` | yes | numerical test confirms |
| AT-06-R1 | Structured population at layer/column scale; vary arrangement, E/I composition, source orientation | CAPABILITY | VALIDATED | `tests/test_atlas_at0506_052.py` | yes | numerical test confirms |
| AT-06-R2 | Electrode chain Q_i -> Phi -> P_electrode -> V_LFP with contact, reference, filtering | CAPABILITY | VALIDATED | `tests/test_atlas_at0506_052.py` | yes | numerical test confirms |
| AT-06-R3 | Locality C(R,f) = P[sum_{r_i<R} Phi_i(f)] / P[sum_i Phi_i(f)] | CAPABILITY | VALIDATED | `tests/test_atlas_at0506_052.py` | yes | numerical test confirms |
| AT-06-R4 | Conductivity and distance assumptions declared | CAPABILITY | VALIDATED | `tests/test_atlas_at0506_052.py` | yes | numerical test confirms |
| AT-07-R1 | Plastic population; W fixed vs Hebbian HDP vs noisy HDP under identical stimulation | CAPABILITY | VALIDATED | `tests/test_atlas_at07_053.py` | yes | numerical test confirms |
| AT-07-R2 | Chain X,H -> dW -> dX -> dQ -> dPhi observed | CAPABILITY | VALIDATED | `tests/test_atlas_at07_053.py` | yes | numerical test confirms |
| AT-07-R3 | Adaptation/stability distinguished from mere attenuation | RESULT | VALIDATED | `artifacts/atlas/results/at10_r5_twin_055.json` | yes | under the tested conditions |
| AT-07-R4 | B as input to neural/plastic dynamics (X,H,B,W) -> Q | CAPABILITY | OUT_OF_SCOPE | `tests/test_atlas_at07_053.py` | yes | do not promote |
| AT-08-R1 | Two areas A1 -> A2 over mm-cm; repeated stimulus to A1 | RESULT | VALIDATED | `artifacts/atlas/at08_at09_054.py` | yes | model produced |
| AT-08-R2 | Adaptation enabled vs clamped; causal H test | RESULT | VALIDATED | `artifacts/atlas/at08_at09_054.py` | yes | under the tested conditions |
| AT-08-R3 | Record SPK, Phi, H in both areas plus dphi_12(f), C_12(f) | RESULT | VALIDATED | `artifacts/atlas/at08_at09_054.py` | yes | model produced |
| AT-08-R4 | Answer: does adaptation stay local, propagate, or alter downstream fields without equivalent spike-count change | RESULT | VALIDATED | `artifacts/atlas/at08_at09_054.py` | yes | model produced |
| AT-09-R1 | Plastic coupling W_12(t), W_21(t) under HDP/Hebbian/noisy rules with delays | RESULT | VALIDATED | `artifacts/atlas/at08_at09_054.py` | yes | model produced |
| AT-09-R2 | Distinguish fast X, relative state H, plastic W; H may evolve while dW/dt = 0 | RESULT | VALIDATED | `artifacts/atlas/at08_at09_054.py` | yes | model produced |
| AT-09-R3 | Local vs inter-area adaptation; effective connectivity change; field/coherence consequences | RESULT | VALIDATED | `artifacts/atlas/at08_at09_054.py` | yes | model produced |
| AT-10-R1 | Compact PseudoGenome G_20 -> D(K_D) -> N_20 with areas, populations, geometry, connectivity, delays, neural models, H, HDP, field/probe definitions | RESULT | VALIDATED | `artifacts/publication/atlas/AT-10-N20/baseline/manifest.json` | yes | model produced |
| AT-10-R2 | Three phases on the same realized network: baseline (W = W0, dW/dt = 0), Hebbian HDP, noisy HDP with explicit K_t | RESULT | VALIDATED | `artifacts/atlas/results/at10_n20_055.json` | yes | model produced |
| AT-10-R3 | Baseline measures: spontaneous/noisy activity, rate distributions, population fields, spectra, inter-area propagation, boundedness, spatial field structure | RESULT | VALIDATED | `artifacts/atlas/results/at10_n20_055.json` | yes | model produced |
| AT-10-R4 | HDP phases measure W(t), H(t), X(t), Phi(t) and network-level organization | RESULT | OUT_OF_SCOPE | `artifacts/atlas/results/at10_n20_055.json` | yes | do not promote |
| AT-10-R5 | Perturbation/control assay distinguishing bounded != returning != homeostatically stabilized | RESULT | VALIDATED | `artifacts/atlas/results/at10_r5_twin_055.json` | yes | under the tested conditions |
| AT-10-R6 | Reduced fast model with very long T | RESULT | SUPPORTED | `artifacts/atlas/results/at10_r6_long_055.json` | yes | supported only |
| AT-10-R7 | F_W(X,H,W,B) with B input | CAPABILITY | OUT_OF_SCOPE | `tests/test_atlas_at07_053.py` | yes | do not promote |
