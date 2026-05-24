# v0.3 Scenario Index and Learning Spine

**Version:** v0.3.0+  
**Last updated:** 2026-05-23  
**truth_mode:** truth_safe_unverified  

---

## 15 Core Scenarios (v0.3.1–v0.3.15)

### **v0.3.1: Single Neuron I — Izhikevich Phenomenology**

**Notebook:** `tutorials/v030_01_single_neuron_izhikevich.ipynb`  
**Duration:** ~15 min (10ms simulation, CPU)  
**Prerequisites:** None

**Learning objectives:**
- Membrane potential dynamics: V_m(t) and recovery variable u(t)
- Spike generation and reset mechanism
- Parameter space: a, b, c, d (timescales and coupling)
- Numerical integration: JAX scan-based recurrence

**Questions answered:**
- How does Izhikevich model generate action potentials?
- What is the relationship between voltage reset and recovery?
- How do parameters tune firing behavior (latency, frequency, adaptation)?

**Computational concepts:**
- JAX random keys and PRNG seeding
- vmap for batch neurons
- jit compilation of recurrent simulation
- Raster plot visualization

**Figure artifacts:**
- Spike raster (single neuron, 100ms)
- V_m trace (voltage dynamics)
- Phase plane (V_m vs u)

**Acceptance gates:**
- Firing rate 2–25 Hz
- Finite values (no NaN/Inf)
- JSON-safe outputs
- JAX-native simulation path

**Truth status:** Phenomenological model; not biophysically calibrated

---

### **v0.3.2: Single Neuron II — Hodgkin-Huxley Conductance**

**Notebook:** `tutorials/v030_02_single_neuron_hodgkin_huxley.ipynb`  
**Duration:** ~20 min (100ms simulation, CPU)  
**Prerequisites:** v0.3.1

**Learning objectives:**
- Ionic conductances: Na, K, leak
- Gating variables (m, h, n) and voltage-dependent kinetics
- Equilibrium potentials and Nernst equation
- Comparison with phenomenological Izhikevich model

**Questions answered:**
- What are the biophysical mechanisms of action potentials?
- How do gating variables encode ion channel kinetics?
- Why does Hodgkin-Huxley produce realistic spike shapes?

**Computational concepts:**
- Higher-dimensional state space (4D: V, m, h, n)
- Voltage-dependent rate constants (alpha/beta)
- Numerical stability and dt requirements
- Custom JAX update rules

**Figure artifacts:**
- Spike waveform (V_m, Na, K currents)
- Gating dynamics (m, h, n traces)
- I-V curve (voltage clamp simulation)
- Comparison with v0.3.1 Izhikevich waveform

**Acceptance gates:**
- Firing rate 2–25 Hz
- Finite values throughout state space
- Sodium/potassium equilibrium potentials realistic (Na ~+60 mV, K ~-90 mV)
- JSON-safe metrics

**Truth status:** Biophysically more realistic; still not calibrated to experimental data

---

### **v0.3.3: Synaptic Dynamics — Receptor Kinetics**

**Notebook:** `tutorials/v030_03_synaptic_dynamics_receptors.ipynb`  
**Duration:** ~20 min (100ms, 1000 synaptic events)  
**Prerequisites:** v0.3.1, v0.3.2

**Learning objectives:**
- Receptor types: AMPA, NMDA, GABA_A, GABA_B
- Exponential decay kinetics: tau_rise, tau_decay
- Receptor-indexed synaptic state
- Current injection from synaptic conductance

**Questions answered:**
- How do receptor types differ in kinetics?
- What is the relationship between spike timing and synaptic strength?
- How do tau parameters shape synaptic integration windows?

**Computational concepts:**
- SynapseSpec and receptor lookup tables
- Scan over presynaptic spikes and postsynaptic state
- Exponential convolution (JAX-native, jittable)
- Multi-receptor systems

**Figure artifacts:**
- PSC waveforms (AMPA, NMDA, GABA_A, GABA_B)
- tau_rise vs tau_decay comparison
- Cumulative synaptic current (multiple events)
- Postsynaptic voltage response to synaptic input

**Acceptance gates:**
- Synaptic currents finite and physiological scale (pA to nA range)
- Receptor kinetics match declared tau values (±5%)
- JSON-safe spike times and conductances
- No negative conductances

**Truth status:** Standard receptor kinetics from literature; not measured in preparation

---

### **v0.3.4: Two-Neuron E/I Circuit**

**Notebook:** `tutorials/v030_04_two_neuron_ei_circuit.ipynb`  
**Duration:** ~20 min (500ms, 2-neuron network)  
**Prerequisites:** v0.3.1, v0.3.3

**Learning objectives:**
- Excitatory-inhibitory balance and network dynamics
- Feedback inhibition and recurrent connectivity
- Oscillatory and chaotic regimes
- Phase locking and entrainment

**Questions answered:**
- How do E and I neurons interact to generate network rhythms?
- What is the role of inhibitory feedback in stability?
- How do synaptic weights tune oscillatory behavior?

**Computational concepts:**
- Two-neuron network connectivity (2×2 weight matrix)
- Recurrent loop: spikes → synaptic currents → voltage → spikes
- JAX scan over full network state
- Time-dependent cross-correlation analysis

**Figure artifacts:**
- E and I spike rasters (temporal dynamics)
- Voltage traces (V_m_E, V_m_I)
- Cross-correlogram (spike timing relationships)
- Phase plane (E vs I firing rate)

**Acceptance gates:**
- Both neurons fire at 2–25 Hz
- Voltage traces finite and within physiological range (-100 to +40 mV)
- Cross-correlogram peaks indicate phase locking (if present)
- JSON-safe spike times and correlation metrics

**Truth status:** Network dynamics demonstrate excitatory-inhibitory balance; not validated against cortical recordings

---

### **v0.3.5: Laminar Population Structure**

**Notebook:** `tutorials/v030_05_laminar_population.ipynb`  
**Duration:** ~25 min (500ms, 100-neuron 5-layer network)  
**Prerequisites:** v0.3.1–v0.3.4

**Learning objectives:**
- Cortical laminar structure (L1, L2/3, L4, L5, L6)
- Intra- and inter-laminar connectivity
- Population-level statistics (rate, synchrony, oscillations)
- Laminar current source density (CSD) from population activity

**Questions answered:**
- How do laminar connectivity patterns shape network activity?
- What is the relationship between population firing and LFP/CSD?
- How do layer-specific inputs drive laminar responses?

**Computational concepts:**
- Laminar population organization [time, neuron, layer]
- Source reconstruction from population current
- TFNE forward-field kernel projection
- Layer-resolved metrics (rate per layer, layer-specific CSD)

**Figure artifacts:**
- Laminar raster (neurons × time, color by layer)
- Layer-resolved firing rates (L1–L6 dynamics)
- Current source density (CSD) heatmap (layer × time)
- Population synchrony index (layer-dependent)

**Acceptance gates:**
- Each layer fires at 2–25 Hz mean (within layer variation OK)
- CSD values finite and physiological range (pA/mm³ scale, no extreme outliers)
- JSON-safe layer statistics and CSD metrics
- Laminar connectivity sparse (>50% zeros, <dense coupling)

**Truth status:** Laminar structure and intra-laminar coupling illustrative; not fitted to cortical anatomy

---

### **v0.3.6: Three-Area Hierarchical Network**

**Notebook:** `tutorials/v030_06_three_area_hierarchy.ipynb`  
**Duration:** ~35 min (1000ms, 300-neuron 3-area network)  
**Prerequisites:** v0.3.1–v0.3.5

**Learning objectives:**
- Multi-area feedforward (area 1 → 2 → 3), feedback, and lateral connectivity
- Hierarchical information flow and cross-area synchronization
- Area-specific time constants and response latencies
- Three-way interaction: local dynamics + cross-area coupling

**Questions answered:**
- How do feedforward and feedback connections shape inter-area dynamics?
- What is the latency of information transfer across areas?
- How do areas synchronize across different frequencies?

**Computational concepts:**
- Multi-area network adjacency (3×3 blocks, sparse)
- Delayed cross-area projections (synaptic delay + axonal propagation)
- JAX vmap over areas, scan over time
- Cross-area coherence and phase lag analysis

**Figure artifacts:**
- Three-area raster (areas stacked, cross-area connectivity highlighted)
- Area-resolved time series (firing rate, oscillation power)
- Cross-area coherence matrix (frequency-dependent)
- Effective connectivity (information transfer direction)

**Acceptance gates:**
- Each area fires at 2–25 Hz
- Cross-area spike latencies consistent with delay parameters
- Coherence values between 0 and 1
- JSON-safe area-level metrics and connectivity

**Truth status:** Three-area architecture and connectivity patterns illustrative; not fitted to connectomics data

---

### **v0.3.7: Field Proxy I — Voltage Dipole Projection**

**Notebook:** `tutorials/v030_07_field_proxy_voltage_dipole.ipynb`  
**Duration:** ~25 min (500ms, 100-neuron population, projection)  
**Prerequisites:** v0.3.5

**Learning objectives:**
- Source representation: population current as dipole moment
- Volume conductor model: resistive media assumption
- Kernel-based projection (1/r kernel, laminar multipole)
- Extracellular potential (φ_e) reconstruction from source

**Questions answered:**
- How is population current projected to extracellular field?
- What is the relationship between source dipole and recorded potential?
- How do geometry and conductivity affect field pattern?

**Computational concepts:**
- Source vector [time, neurons] → current dipole [time, x, y, z]
- Kernel projection matrix (geometry × neural positions)
- JAX einsum for efficient kernel application
- Field validation: divergence-free (∇·σ∇φ = 0 in laminar_proxy_no_pde)

**Figure artifacts:**
- Source current trace (single neuron current)
- Extracellular potential trace (single recording location)
- Field potential heatmap (space × time)
- Source-field cross-correlation (coupling strength)

**Acceptance gates:**
- Source current finite and matches population rate
- Field potential finite (no NaN/Inf)
- Field amplitude scales with population size (expected)
- JSON-safe source/field metrics and geometry parameters

**Truth status:** Kernel-based approximation, not PDE solution; laminar_proxy_no_pde mode

---

### **v0.3.8: Field Proxy II — CSD/LFP Multimodal Readout**

**Notebook:** `tutorials/v030_08_field_proxy_csd_lfp.ipynb`  
**Duration:** ~30 min (500ms, 100-neuron, 16-channel probe)  
**Prerequisites:** v0.3.7

**Learning objectives:**
- Current source density (CSD): spatial second derivative of φ_e
- Local field potential (LFP): filtered and re-referenced φ_e
- Multimodal readout pipeline: source → field → probe
- Probe geometry and channel montage effects

**Questions answered:**
- How does CSD reveal synaptic current flow?
- What is the relationship between LFP and population activity?
- How do probe geometry and reference affect recordings?

**Computational concepts:**
- CSD kernel: -σ ∇²φ (JAX-native spatial derivative)
- LFP filtering: band-pass (0.3–300 Hz, common reference)
- Multi-probe projection (linear algebra, einsum)
- Power spectral density (PSD) analysis (Welch periodogram)

**Figure artifacts:**
- CSD profile (depth × time, heatmap)
- LFP traces (16 channels, time series)
- Frequency spectrum (PSD per channel)
- Channel-averaged LFP power (delta, theta, alpha, beta, gamma)

**Acceptance gates:**
- CSD magnitudes physiological range (pA/mm³ scale)
- LFP amplitudes realistic (μV scale at electrode distances)
- Power spectrum peaks in expected frequency ranges
- JSON-safe CSD/LFP metrics and frequency decomposition

**Truth status:** Kernel-based proxy; not measured in experiment; no Maxwell/Poisson solver

---

### **v0.3.9: Oddball Stimulus and Global Context (Advanced)**

**Notebook:** `tutorials/v030_09_oddball_stimulus_context.ipynb`  
**Duration:** ~45 min (5000ms, 3-area, global vs. oddball condition)  
**Prerequisites:** v0.3.6, v0.3.8

**Learning objectives:**
- Stimulus-driven network responses (sensory area response latency)
- Oddball detection: deviance from standard stimulus sequence
- Cross-area prediction and prediction error signals
- Context-dependent plasticity and gain modulation

**Questions answered:**
- How do networks respond differently to standard vs. oddball stimuli?
- What is the time course of prediction error signals?
- How do higher areas modulate sensory responses?

**Computational concepts:**
- Stimulus schedule: trains of pulses, occasionally perturbed
- Conditional stimulus timing and response lockedness
- Trial averaging and event-related potential (ERP) analysis
- Permutation testing for significance (oddball vs. standard)

**Figure artifacts:**
- Stimulus raster (event markers)
- Trial-averaged raster (standard vs. oddball)
- Cross-area ERP traces (latency, polarity)
- Spectral power (standard vs. oddball frequency content)

**Acceptance gates:**
- Standard and oddball responses differentiated (permutation p < 0.05)
- Latencies consistent across trials (< 50ms jitter)
- Cross-area prediction signals present (if hierarchical)
- JSON-safe ERPs and statistical tests

**Truth status:** Illustrative model of oddball effect; not validated against experimental data

---

### **v0.3.10: Omission Response and Prediction Error (Advanced)**

**Notebook:** `tutorials/v030_10_omission_response_prediction_error.ipynb`  
**Duration:** ~45 min (5000ms, 3-area, stimulus with omissions)  
**Prerequisites:** v0.3.6, v0.3.9

**Learning objectives:**
- Predictive coding: temporal expectancy and surprise
- Omission response: neural activity to absence of expected stimulus
- Prediction error circuits: cross-area error signals
- Active sensing and uncertainty reduction

**Questions answered:**
- How do networks respond to omitted (expected but not delivered) stimuli?
- What are the signatures of prediction error in population activity?
- How do higher areas signal expectation violations?

**Computational concepts:**
- Stimulus timing prediction: internal model
- Conditional response (if stimulus expected, omission triggers response)
- Difference waves (omission − expected stimulus response)
- Error signal propagation (bottom-up vs. top-down)

**Figure artifacts:**
- Omission response raster (time-locked to expected stimulus time)
- Difference waveforms (omission − stimulus response, per area)
- Prediction error latency (cross-area comparisons)
- Spectral signatures of expectancy (alpha/beta power changes)

**Acceptance gates:**
- Omission response amplitude > 0 (prediction error present)
- Latency differences between areas reflect hierarchy
- Cross-area difference waves uncorrelated (independent prediction channels)
- JSON-safe omission metrics and statistical tests

**Truth status:** Illustration of predictive coding framework; not fitted to mismatch negativity (MMN) data

---

### **v0.3.11: Spike-Timing-Dependent Plasticity I (Advanced)**

**Notebook:** `tutorials/v030_11_stdp_plasticity.ipynb`  
**Duration:** ~60 min (repeated 5000ms trials, plasticity learning)  
**Prerequisites:** v0.3.4, v0.3.6

**Learning objectives:**
- Hebbian learning and causality: "neurons that fire together wire together"
- STDP window: potentiation (pre before post) vs. depression (post before pre)
- Weight evolution during paired stimulus trains
- Homeostatic stability mechanisms

**Questions answered:**
- How do synaptic weights change based on spike timing correlations?
- What is the time window for causality detection?
- How do networks maintain stability during learning?

**Computational concepts:**
- Spike pair detection: find pre- and post-spike pairs
- STDP rule application: ΔW = A+ exp(-Δt/τ+) if Δt > 0, A- exp(Δt/τ-) if Δt < 0
- Sliding window learning: update during simulation
- Weight decay (homeostatic constraint)

**Figure artifacts:**
- STDP window (Δt vs. ΔW, characteristic function)
- Weight evolution (individual synapses, trials)
- Population connectivity before/after learning (heatmap)
- Firing rate stability (homeostasis mechanism)

**Acceptance gates:**
- STDP windows match declared A+/A- and tau+/tau- parameters
- Weights stay bounded (no runaway potentiation or silencing)
- Learning produces structured connectivity (some synapses strengthen, others weaken)
- JSON-safe weight evolution and STDP metrics

**Truth status:** Canonical STDP rule; not fitted to synaptic patch-clamp data

---

### **v0.3.12: Homeostatic Scaling and Learning Rules II (Advanced)**

**Notebook:** `tutorials/v030_12_homeostatic_scaling.ipynb`  
**Duration:** ~60 min (repeated trials, homeostatic correction)  
**Prerequisites:** v0.3.11

**Learning objectives:**
- Synaptic scaling: multiplicative global weight adjustment
- Firing rate targets and error correction
- Interaction of STDP and homeostatic regulation
- Network stability during extended learning

**Questions answered:**
- How do networks maintain stable firing rates during STDP?
- What is the time course of homeostatic adjustment?
- Can STDP and homeostasis coexist without instability?

**Computational concepts:**
- Sliding window firing rate estimation (low-pass filter)
- Multiplicative scaling rule: W_new = W × (target_rate / current_rate)
- Dual learning: STDP + homeostatic regulation
- Convergence analysis (weight distribution stability)

**Figure artifacts:**
- Firing rate dynamics (with and without homeostasis)
- Weight distribution before/after homeostatic correction
- Learning curves (error vs. trial number)
- Spectral stability (power spectrum changes during learning)

**Acceptance gates:**
- Firing rates stay within 2–25 Hz target band throughout learning
- Weights remain finite and bounded
- Homeostatic adjustment magnitude reasonable (±50% scaling max)
- JSON-safe learning curves and firing rate metrics

**Truth status:** Illustrative homeostatic mechanism; not fitted to intracellular recording data

---

### **v0.3.13: Optimization I — GSDR Parameter Fitting (Advanced)**

**Notebook:** `tutorials/v030_13_optimization_gsdr_fitting.ipynb`  
**Duration:** ~120 min (GSDR evolution, 50–100 generations)  
**Prerequisites:** v0.3.5, v0.3.8

**Learning objectives:**
- Objective function design: mismatch between target and model output
- GSDR (Genetic Search with Derivative-free Ranking) optimization
- Fitness landscape exploration and local/global optima
- Convergence metrics and termination criteria

**Questions answered:**
- How can we fit network parameters to target population statistics?
- What is the efficiency of derivative-free optimization vs. gradient-based methods?
- How sensitive are parameters to objective function definition?

**Computational concepts:**
- Objective function: L2 distance between target and simulated firing rates/spectra
- Population-based search: maintain diverse candidate solutions
- Fitness ranking and selection (Pareto optimization if multi-objective)
- JAX vmap for parallel fitness evaluation

**Figure artifacts:**
- Fitness evolution (best/median/worst per generation)
- Parameter trajectories (select parameters vs. generation)
- Firing rate convergence (target vs. final fitted model)
- Pareto front (if multi-objective: firing rate vs. spectral power)

**Acceptance gates:**
- Fitness converges (no improvement for >10 generations)
- Fitted model reproduces target statistics (R² > 0.8 typical)
- Final parameters within plausible biological ranges
- JSON-safe fitness curves and parameter values

**Truth status:** Optimization demonstrates feasibility of parameter inference; not biological calibration

---

### **v0.3.14: Optimization II — Multi-Objective and Evolutionary Search (Advanced)**

**Notebook:** `tutorials/v030_14_optimization_multiobjective.ipynb`  
**Duration:** ~150 min (AGSDR evolution, 100–200 generations, multiple objectives)  
**Prerequisites:** v0.3.13

**Learning objectives:**
- Multi-objective optimization: balance competing fitness functions
- Pareto optimality: trade-off curves between objectives
- AGSDR (Adaptive Genetic Search with Derivative-free Ranking)
- Ensemble solutions and robustness analysis

**Questions answered:**
- How do networks balance competing demands (firing rate, synchrony, power spectrum)?
- What is the Pareto frontier of possible solutions?
- Which solutions are robust to parameter perturbations?

**Computational concepts:**
- Multiple objectives: firing rate, spectral power, cross-area coherence, sparsity
- Pareto ranking: dominated vs. non-dominated solutions
- Adaptive selection pressure (AGSDR vs. GSDR)
- Robustness testing: sensitivity analysis post-optimization

**Figure artifacts:**
- Pareto front 2D/3D (objectives vs. each other)
- Population diversity over generations
- Robustness histograms (parameter sensitivity ranges)
- Trade-off curves (objective A vs. objective B, color by objective C)

**Acceptance gates:**
- Pareto front shows clear trade-offs (not degenerate points)
- Population diversity maintained (>5 unique non-dominated solutions)
- Robustness analysis shows ±10–30% parameter tolerance typical
- JSON-safe Pareto solutions and sensitivity metrics

**Truth status:** Multi-objective optimization illustrates biological design space; not evolutionary selection simulation

---

### **v0.3.15: Whole-Scenario Review and Integration (Capstone)**

**Notebook:** `tutorials/v030_15_review_integration_benchmarks.ipynb`  
**Duration:** ~60 min (integration test + benchmarks)  
**Prerequisites:** v0.3.1–v0.3.14

**Learning objectives:**
- Integration: assemble all previous scenarios into one large model
- Benchmarking: performance metrics (runtime, memory, scaling)
- Truth boundary review: explicit statement of computational scaffold status
- Next steps and extensions

**Questions answered:**
- How do the 14 concepts combine into a unified framework?
- What are the computational limits and bottlenecks?
- Where can v0.3.x extend without violating claim gates?

**Computational concepts:**
- Large-scale network: 1000+ neurons, 5 areas, plasticity, multiple readouts
- Profiling: JAX JIT time, device memory, wall-clock runtime
- Scaling analysis: runtime vs. network size (expected linear/quadratic)
- Future roadmap: Poisson solver, Maxwell fields, GPU acceleration (deferred)

**Figure artifacts:**
- Large-network raster (1000 neurons, visual sample)
- Benchmarking results (wall-clock vs. network size, device comparison)
- Integrated analysis (all 8 readout modalities side-by-side)
- Truth boundary diagram (what is computed, what is declared proxy)

**Acceptance gates:**
- Full integrated model runs to completion without NaN/Inf
- Benchmarks report realistic timings (all Pareto-optimal code paths covered)
- Truth status statement clear and complete (no ambiguous claims)
- JSON-safe manifest for large model, all 8 readouts present

**Truth status:** Final confirmation: TFNE is a computational scaffold for teaching/design, not biological validation

---

## Audit Phases (v0.3.16–v0.3.31)

### **v0.3.16–v0.3.20: Performance and Reproducibility Audit**
- Baseline CPU runtime per scenario
- Memory profiling and optimization opportunities
- PRNG reproducibility (same seed → same trajectory)
- CI/CD integration and regression detection

### **v0.3.21–v0.3.25: Documentation and Clarity Audit**
- Tutorial language audit (no overclaiming)
- Cross-reference consistency
- Accessibility for new users (prerequisites clear, learning path verified)
- Figure clarity and artifact integrity (SHA256 validation)

### **v0.3.26–v0.3.31: Scientific Boundary and Next-Steps Audit**
- Truth status validation (claim gates frozen)
- No accidental scope creep (no new solvers, no proof-of-mechanism)
- Identified extensions for v0.4+ (Maxwell, Poisson, GPU, whole-brain)
- Community feedback incorporation and issue triage

---

## See Also

- [README](README.md) — Overview and acceptance gates
- [Visualization Doctrine](visualization_doctrine.md) — PNG/Plotly policy, figure panels, manifest contract
- [Template](template.md) — 13-section required structure
- [Plotly Policy](plotly_policy.md) — Interactive figure guidelines
- [Canonical Imports](canonical_imports.md) — Import conventions
- [v0.3 Tutorial-Scenario Plan](../v030_tutorial_scenario_plan.md) — Doctrine
