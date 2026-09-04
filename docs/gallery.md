# Gallery

Eight reproducible reference panels generated entirely by repository code from JaxFNE.

All panels represent computational proxies and relative uncalibrated states (`calibration_status = relative_proxy_readout`). No direct physical or clinical claims are asserted.

---

## 01: Realized Network Structure

![01 Realized Network](_static/gallery/01_realized_network.png)

**Configuration:** `suite2_v1_v4_config` · **Seed:** `42`  
**Status:** `relative_proxy_coordinates`  
**Description:** Realized spatial locations and cell-class identities in a two-area (V1-V4) laminar network.

---

## 02: Population Activity and Spike Raster

![02 Spikes Activity](_static/gallery/02_spikes_activity.png)

**Configuration:** `suite2_net1_config(n=100)` · **Seed:** `10`  
**Status:** `uncalibrated_computational_scaffold`  
**Description:** Spike raster and instantaneous population firing rate for a recurrent 100-neuron network.

---

## 03: Fast Neural State Trajectories ($V_m$)

![03 Fast State Vm](_static/gallery/03_fast_state_vm.png)

**Configuration:** `suite2_net1_config(n=100)` · **Seed:** `10`  
**Status:** `native_izhikevich_millivolts`  
**Description:** Membrane potential traces demonstrating fast spiking dynamics and subthreshold integration.

---

## 04: Relative Biophysical State (RBD Dynamics)

![04 H RBD State](_static/gallery/04_h_rbd_state.png)

**Configuration:** `suite2_net1_config (enable_hdp=True)` · **Seed:** `7`  
**Status:** `relative_dimensionless_state`  
**Description:** Dynamic evolution of the Relative Biophysical State $H$ under activity-dependent drain and restorative control.

---

## 05: Transmembrane Current Source $Q$ to Extracellular LFP Proxy

![05 Source to Field](_static/gallery/05_source_to_field.png)

**Configuration:** `suite2_net1_config (record_fields=True)` · **Seed:** `10`  
**Status:** `uncalibrated_source_and_field_proxy`  
**Description:** Transmembrane relative source current density and projected laminar field potential (LFP) proxy across linear contacts.

---

## 06: Finite-Delay Timing — Event, Axonal Latency, and Postsynaptic Response

![06 Finite Delay Timing](_static/gallery/06_finite_delay_timing.png)

**Configuration:** `two_neuron_delay_circuit(delay=10ms)` · **Seed:** `1`  
**Status:** `discrete_delay_buffer_exact`  
**Description:** Explicit temporal decomposition: presynaptic spike emission, 10 ms axonal transmission buffer latency, arriving synaptic current, and subsequent postsynaptic EPSP integration.

---

## 07: Multi-Area Laminar Connectivity Matrix

![07 Multiarea Laminar Connectivity](_static/gallery/07_multiarea_laminar_connectivity.png)

**Configuration:** `suite2_v1_v4_config` · **Seed:** `42`  
**Status:** `uncalibrated_sparse_weights`  
**Description:** Realized sparse inter-column and intra-column synaptic connection matrix for V1-V4 multi-area model.

---

## 08: Multiscale State Evolution — Fast Observable $X$ ($V_m$) vs Slower $H$ and $W$

![08 Fast vs Slow State](_static/gallery/08_fast_vs_slow_state.png)

**Configuration:** `suite2_net1_config(n=100, enable_hdp=True)` · **Seed:** `7`  
**Status:** `multiscale_state_coupling`  
**Description:** Co-registered multiscale trajectories: sub-millisecond membrane state $X$ ($V_m$) alongside slower hidden biophysical state $H$ (RBD dynamics) and plastic synaptic weight coupling $W$ (HDP).
