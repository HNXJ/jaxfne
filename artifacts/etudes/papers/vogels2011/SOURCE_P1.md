# vogels2011: P1 source record

Target: Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W (2011), Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks, Science 334(6062):1569-1573, doi:10.1126/science.1211095. The paper text is paywalled (no legal full text reachable; nothing saved in the repo). Read instead: the paper abstract via ModelDB accession 143751; the authors' MATLAB single-cell script `inhib_plasticity.m` and Brian recurrent-network script `isp_brian.py` from the ModelDBRepository/143751 mirror (the ModelDB readme states the MATLAB code was included in the Science SOM); the open-access workshop review Vogels et al. 2013, Front Neural Circuits 7:119 (first author Vogels, co-author Zenke); the Zenke and Vogels lab publication pages (no author PDF reachable).

Pointers below name the file and symbol read. Status is observed (a source read states it), inferred (computed from stated values, derivation shown), or unknown (no source read gives it). No passage is quoted.

## 1. Model equations (symbols from the authors' code and the 2013 review)

Single cell, feedforward E/I inputs (observed, `inhib_plasticity.m`):

- Leak-normalized LIF (`gLeak = 1`): conductances decay per step, `gEx *= expGEx`, `gIn *= expGIn` with `exp(-dt/tauEx)`, `exp(-dt/tauIn)`.
- Subthreshold update: `gTot = gLeak + gEx + gIn`; `tauEff = taumem / gTot` (`taumem = 20` ms); `VInf = (gLeak*VRest + gEx*EAMPA + gIn*EGABA) / gTot`; `V` relaxes toward `VInf` over `dt = 0.1` ms. Spike at `Vth = -50` mV, reset to `VRest = -60` mV, absolute refractory `tRef = 5` ms with `V` held at rest.
- iSTDP traces: `pre(i) += eta` on each inhibitory presynaptic spike, `post += eta` on each postsynaptic spike, both decaying with `tauPlasticity = 20` ms (`eta = 0.001`).
- Rule, inhibitory synapses only: on an inhibitory presynaptic spike `Synapse(i) += post - alpha` with `alpha = 0.25*eta`; on a postsynaptic spike every inhibitory synapse takes `Synapse(i) += pre(i)`; weights floored at 0. Excitatory weights are fixed.

Recurrent network (observed, `isp_brian.py`, Brian 1.3.0):

- Conductance LIF: `dv/dt = (-gl*(v-el) - (g_ampa*w*v + g_gaba*(v-er)*w) + bgcurrent)/memc` with `gl = 10.0*nS`, `el = -60*mV`, `er = -80*mV`, `vt = -50*mV`, `memc = 200.0*pF`, `bgcurrent = 200*pA`; `dg_ampa/dt = -g_ampa/tau_ampa` (`5.0*ms`); `dg_gaba/dt = -g_gaba/tau_gaba` (`10.0*ms`); refractory `5*ms`, reset `el`; basic weight unit `w = 1.*nS`.
- `NE = 8000` excitatory plus `NI = NE/4` (2000) inhibitory cells, sparseness `epsilon = 0.02`: E to all `g_ampa` weight 0.3; I to E `g_gaba` from weight `1e-10`; I to I `g_gaba` weight 3 (fixed).
- iSTDP on I-to-E only: traces `dA/dt = -A/tau_stdp` (`tau_stdp = 20*ms`); on a presynaptic spike `A_pre += 1`, `w += (A_post - alpha)*eta` (`eta = 1e-2`); on a postsynaptic spike `A_post += 1`, `w += A_pre*eta`; cap `gmax = 100`. Coded `alpha = 3*Hz*tau_stdp*2`, i.e. `alpha = 2*rho0*tau_stdp` with target rate `rho0 = 3` Hz.
- Protocol: `1*second` without plasticity, then plasticity to `simtime = 10*second`; raster panels "Before" (0.8-1.0 s) vs "After" (last 0.2 s).

Rule in words (observed, 2013 review, "stabilize network dynamics" section): near-coincident pre- and postsynaptic spikes potentiate the inhibitory synapse while every lone presynaptic spike depresses it (the review's Fig. 1K form). Weak synapses that fail to prevent postsynaptic spiking strengthen; strong ones that reliably prevent postsynaptic spikes weaken. Over time this yields a detailed balance of excitatory and inhibitory weights per correlated input channel, acting homeostatically in feedforward and in recurrent networks. The review states the two requirements: postsynaptic activity must potentiate activated inhibitory synapses, and synapses must decay in the absence of postsynaptic firing.

## 2. Parameters

Single cell (observed, `inhib_plasticity.m` setup block):

| Parameter | Paper value | Status |
|---|---|---|
| `eta` (learning rate) | 0.001 | observed |
| `alpha` (decay term) | `0.25*eta` | observed |
| `tauPlasticity` | 20 ms | observed |
| `duration`, `dt`, `NRuns` | 1500 ms, 0.1 ms, 100 runs (150 s total) | observed |
| `tRef` | 5 ms | observed |
| `gBarEx`, `gBarIn` (units of leak, `gleak = 10 nS`) | 0.014, 0.035 | observed |
| `VRest`, `Vth`, `taumem` | -60 mV, -50 mV, 20 ms | observed |
| `EAMPA`, `EGABA`, `tauEx`, `tauIn` | 0 mV, -80 mV, 5 ms, 10 ms | observed |
| Input: `noisetau`, background, `Maxrate`, `NSigs`, `NCells`, `ExFrac` | 50 ms, 5 Hz, 500 Hz, 8 signals, 1000 trains, 0.8 excitatory | observed |
| Excitatory tuning | `0.3 + 1.1/(1 + (group-5)^4)` plus `0.1*rand` | observed |
| Inhibitory init | 0.1, uniform and weak | observed |
| Homogeneous-rate fixed point | 6.25 Hz (`alpha/(2*eta*tauPlasticity)` postsynaptic rate zeroes mean drift) | inferred |
| MATLAB integrator details | Euler update as coded; solver/version unstated | unknown |

Recurrent network (observed, `isp_brian.py` parameter block):

| Parameter | Paper value | Status |
|---|---|---|
| `NE`, `NI` | 8000, 2000 (`NI = NE/4`) | observed |
| `epsilon` | 0.02 | observed |
| `w`, `tau_ampa`, `tau_gaba` | 1 nS, 5 ms, 10 ms | observed |
| `eta`, `tau_stdp` | 1e-2, 20 ms | observed |
| `gl`, `el`, `er`, `vt`, `memc`, `bgcurrent` | 10 nS, -60 mV, -80 mV, -50 mV, 200 pF, 200 pA | observed |
| refractory | 5 ms | observed |
| E-to-all weight, I-to-E init, I-to-I weight | 0.3, 1e-10, 3 | observed |
| `alpha`, `gmax` | `2*rho0*tau_stdp` with coded `rho0 = 3` Hz, 100 | observed |
| protocol | 1 s without plasticity, plasticity to 10 s | observed |
| Excitatory fixed point | `rho_E = rho0 = 3` Hz (mean drift zeroes at `alpha = 2*rho*tau_stdp`) | inferred |
| Paper's realized rates, CVs, synchrony numbers | not in any source read | unknown |

## 3. Claims per target figure

Fig. 4, recurrent network (rule and protocol observed in `isp_brian.py`; mechanism observed in the 2013 review):

- Inhibitory STDP on I-to-E synapses drives excitatory firing to the coded target `rho0 = 3` Hz (observed constant, inferred equilibrium above).
- The network moves from its initial state to an inhibition-stabilized state, shown as "Before" (0.8-1.0 s) vs "After" (last 0.2 s) rasters (observed code protocol).
- Realized rate values, CV values and synchrony measures as the paper states them: unknown (full text paywalled; see §5 G-source).

Figs. 1-2, single cell (mechanism observed in the 2013 review; tuning routine observed in `inhib_plasticity.m`):

- Inhibitory weights align per input channel to the excitatory tuning, so excitatory and inhibitory currents balance in detail; the paper links this to sparse firing and to excitatory-inhibitory receptive-field plasticity (observed, ModelDB abstract).
- Numeric values as the paper states them: unknown (full text paywalled; see §5 G-source).

## 4. Data, code and licence

- Paper licence: subscription journal (Science, observed via Europe PMC record `isOpenAccess: N`); an author copy is listed on the Vogels lab publications page (`neurotheory.ox.ac.uk/~timv/pub/VogelsScience2011.pdf`) but the host is unreachable (transport error, observed 2026-10-19); no licence statement observed for that copy.
- Authors' code location and licence: ModelDB accession 143751 (observed, `modeldb.science/143751`: simulators MATLAB/Brian/Python, implementers Zenke and Vogels), mirrored at `github.com/ModelDBRepository/143751` (observed file list: `inhib_plasticity.m`, `InputCurrents.m`, `isp_brian.py`); read as text only, never run (checklist S1). No licence statement observed on either page read.
- Data location and licence: none (simulation paper; no recordings in any source read).

## 5. Gaps (each item observed or inferred; jaxfne paths checked in this tree)

Have (observed): current-based LIF emitter family (`jaxfne/emitters_lif.py`); shared STDP weight kernel with pre/post traces (`stdp_weight_update`, `update_stdp_weights_jax`, `run_stdp_stream` in `jaxfne/plasticity.py`, on the public surface per `jaxfne/public_surface.py`); HDP rule families (`simulate_edge_recurrent_izhikevich_hdp` in `jaxfne/emitters.py`, `jaxfne/hdp_rule.py`); per-edge delays and external Poisson drive (per the brunel2000 record).

- G-source (observed): the paywalled full text and SOM PDF are unreachable from here (science.org article and SOM URLs return 403; the author's copy host fails transport; OSB project page is 404; PubMed page blocked; web search unavailable). Until a legal copy is readable, the paper's stated Fig. 4 rates/CVs and Figs. 1-2 numbers stay unknown; the P2 pilot reproduces the authors' code behavior first.
- G1 (inferred): conductance synapses with reversal potentials (`er = -80*mV`, `EAMPA`/`EGABA`) against the current-based core LIF; P2 checks what zeroing the difference costs.
- G2 (observed): the core STDP kernel updates only excitatory-presynaptic weights (`update_mask = exc_mask[None, :]` in `stdp_weight_update`, `jaxfne/plasticity.py:93`) with asymmetric LTP/LTD; the Vogels rule needs plastic I-to-E weights with pairing potentiation plus a per-presynaptic-spike depression term. No matching rule observed in the core.
- G3 (observed): I-to-E weights start near zero (`1e-10`) and grow under plasticity with cap `gmax = 100`; the port declares init, growth and cap handling at P2.
- G4 (inferred): a 10,000-cell sparse network through the public API is a P2 scale question, not a claim.
- G5 (observed): no licence statement observed on the authors' code; reuse terms unknown.
