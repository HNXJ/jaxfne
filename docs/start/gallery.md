# Gallery

Nine reproducible reference panels generated entirely by repository code from JaxFNE.

All panels represent computational proxies and relative uncalibrated states (`calibration_status = relative_proxy_readout`). No direct physical or clinical claims are asserted.

---

## 01: Realized Network Structure

![01 Realized Network](../_static/gallery/01_realized_network.png)

**Configuration:** `suite2_v1_v4_config` · **Seed:** `42`  
**Status:** `relative_proxy_coordinates`  
**Description:** Realized spatial locations and cell-class identities in a two-area (V1-V4) laminar network.

---

## 02: Population Activity and Spike Raster

![02 Spikes Activity](../_static/gallery/02_spikes_activity.png)

**Configuration:** `suite2_net1_config(n=100)` · **Seed:** `10`  
**Status:** `uncalibrated_computational_scaffold`  
**Description:** Spike raster and instantaneous population firing rate for a recurrent 100-neuron network.

---

## 03: Fast Neural State Trajectories ($V_m$)

![03 Fast State Vm](../_static/gallery/03_fast_state_vm.png)

**Configuration:** `suite2_net1_config(n=100)` · **Seed:** `10`  
**Status:** `native_izhikevich_millivolts`  
**Description:** Membrane potential traces demonstrating fast spiking dynamics and subthreshold integration.

---

## 04: Relative Biophysical State (RBD Dynamics)

![04 H RBD State](../_static/gallery/04_h_rbd_state.png)

**Configuration:** `suite2_net1_config (enable_hdp=True)` · **Seed:** `7`  
**Status:** `relative_dimensionless_state`  
**Description:** Dynamic evolution of the Relative Biophysical State $H$ under activity-dependent drain and restorative control.

---

## 05: Transmembrane Current Source $Q$ to Extracellular LFP Proxy

![05 Source to Field](../_static/gallery/05_source_to_field.png)

**Configuration:** `suite2_net1_config (record_fields=True)` · **Seed:** `10`  
**Status:** `uncalibrated_source_and_field_proxy`  
**Description:** Transmembrane relative source proxy and projected laminar field potential (LFP) proxy across linear contacts.

---

## 06: Finite-Delay Timing — Event, Axonal Latency, and Postsynaptic Response

![06 Finite Delay Timing](../_static/gallery/06_finite_delay_timing.png)

**Configuration:** `two_neuron_delay_circuit(delay=10ms)` · **Seed:** `1`  
**Status:** `discrete_delay_buffer_exact`  
**Description:** Explicit temporal decomposition: presynaptic spike emission, 10 ms axonal transmission buffer latency, arriving synaptic current, and subsequent postsynaptic EPSP integration.

---

## 07: Multi-Area Laminar Connectivity Matrix

![07 Multiarea Laminar Connectivity](../_static/gallery/07_multiarea_laminar_connectivity.png)

**Configuration:** `suite2_v1_v4_config` · **Seed:** `42`  
**Status:** `uncalibrated_sparse_weights`  
**Description:** Realized sparse inter-column and intra-column synaptic connection matrix for V1-V4 multi-area model.

---

## 08: Multiscale State Evolution — Fast Observable $X$ ($V_m$) vs Slower $H$ and $W$

![08 Fast vs Slow State](../_static/gallery/08_fast_vs_slow_state.png)

**Configuration:** `suite2_net1_config(n=100, enable_hdp=True)` · **Seed:** `7`  
**Status:** `multiscale_state_coupling`  
**Description:** Co-registered multiscale trajectories: sub-millisecond membrane state $X$ ($V_m$) alongside slower hidden biophysical state $H$ (RBD dynamics) and plastic synaptic weight coupling $W$ (HDP).

---

## 09: Three-Area Hierarchy (V1–V4–PFC)

![09 Three Area Hierarchy](../_static/gallery/09_three_area_hierarchy.png)

**Configuration:** `build_multi_area_columns(["V1", "V4", "PFC"], n_per_area=100, ei_profile="canonical")` · **Seed:** `0`  
**Status:** `relative_proxy_coordinates`  
**Description:** Realized spatial locations and cell-class identities in a three-area laminar hierarchy with bidirectional feedforward/feedback connectivity.

Seven fixed panels, always emitted (`schema`, `network_3d`, `raster` as **OBSERVED**; `lfp`, `h_dynamics`, `hdp`, `oscillatory` as **DERIVED**), each carrying a provenance card (`config_hash`, N, edges, steps, dt, jaxfne version). A panel whose declared inputs cannot be met renders an explicit omission card:

![V1-V4-PFC three-area hierarchy (100/area)](https://hnxj.github.io/jaxfne/figures/three_area/raster.png)
*V1-V4-PFC three-area hierarchy (100/area): spike raster.*

Interactive figures: [Circuit schematic](https://hnxj.github.io/jaxfne/figures/three_area/schema.html) · [Network 3D](https://hnxj.github.io/jaxfne/figures/three_area/network_3d.html) · [Spike raster](https://hnxj.github.io/jaxfne/figures/three_area/raster.html) · [LFP proxy](https://hnxj.github.io/jaxfne/figures/three_area/lfp.html) · [Oscillatory response](https://hnxj.github.io/jaxfne/figures/three_area/oscillatory.html)

```python
import jaxfne as jtfne
from jaxfne.vis import build_atlas

cfg = (
    jtfne.build_multi_area_columns(["V1", "V4", "PFC"], n_per_area=100, ei_profile="canonical")
    .runtime(seed=0)
    .set_emitter("izhikevich", "cortical_eig")
    .probes(["spikes", "V_m"])
    .field(domain="laminar_column", conductivity="proxy")
)
model = jtfne.construct(cfg)
sim = jtfne.simulation(duration_ms=500.0, dt_ms=0.5, seed=0)
signals = jtfne.simulate(model, sim)
manifest = build_atlas(
    model, signals,
    out_dir="docs/_static/atlas_three_area",
    title="V1-V4-PFC three-area hierarchy (100/area)",
)
```

## Atlas evidence index (generated)

<!-- ATLAS-INDEX:START -->

Evidence index of validated atlases (25 models). Generated from committed `manifest.json` files — do not edit by hand; run `python scripts/generate_gallery.py --write`.

### `atlas/calibration_100` — Calibration-ready column, minimal run (100n, 200 ms)

State `validated` · N=100 · edges=9900 · steps=400 · config `96761084e110` · jaxfne 0.5.0

=== "Still"

    ![Calibration-ready column, minimal run (100n, 200 ms)](https://hnxj.github.io/jaxfne/figures/calibration_100/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/calibration_100/raster.html" loading="lazy" title="Calibration-ready column, minimal run (100n, 200 ms)"></iframe>

*Calibration-ready column, minimal run (100n, 200 ms): spike raster.*

### `atlas/canonical_etude_1000` — Canonical-column etude, uniform drive (1000n)

State `validated` · N=1000 · edges=999000 · steps=2000 · config `b8ec3bc1536d` · jaxfne 0.5.0

=== "Still"

    ![Canonical-column etude, uniform drive (1000n)](https://hnxj.github.io/jaxfne/figures/canonical_etude_1000/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/canonical_etude_1000/raster.html" loading="lazy" title="Canonical-column etude, uniform drive (1000n)"></iframe>

*Canonical-column etude, uniform drive (1000n): spike raster.*

### `atlas/config_grammar_1000` — Configuration-grammar column, smoke-scale (1000n, 200 ms)

State `validated` · N=1000 · edges=999000 · steps=400 · config `4665fb608374` · jaxfne 0.5.0

=== "Still"

    ![Configuration-grammar column, smoke-scale (1000n, 200 ms)](https://hnxj.github.io/jaxfne/figures/config_grammar_1000/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/config_grammar_1000/raster.html" loading="lazy" title="Configuration-grammar column, smoke-scale (1000n, 200 ms)"></iframe>

*Configuration-grammar column, smoke-scale (1000n, 200 ms): spike raster.*

### `atlas/ei_population_100` — Chainable E/I population (100n)

State `validated` · N=100 · edges=9900 · steps=2000 · config `8c242da2a84d` · jaxfne 0.5.0

=== "Still"

    ![Chainable E/I population (100n)](https://hnxj.github.io/jaxfne/figures/ei_population_100/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/ei_population_100/raster.html" loading="lazy" title="Chainable E/I population (100n)"></iframe>

*Chainable E/I population (100n): spike raster.*

### `atlas/evoked_l4` — Evoked L4 drive, evoked condition (100n)

State `validated` · N=100 · edges=9900 · steps=3000 · config `872f598151ed` · jaxfne 0.5.0

=== "Still"

    ![Evoked L4 drive, evoked condition (100n)](https://hnxj.github.io/jaxfne/figures/evoked_l4/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/evoked_l4/raster.html" loading="lazy" title="Evoked L4 drive, evoked condition (100n)"></iframe>

*Evoked L4 drive, evoked condition (100n): spike raster.*

### `atlas/hdp_10` — HDP circuit, full recording (10n, 100 ms)

State `validated` · N=10 · edges=90 · steps=200 · config `14c5b4f994b9` · jaxfne 0.5.0

=== "Still"

    ![HDP circuit, full recording (10n, 100 ms)](https://hnxj.github.io/jaxfne/figures/hdp_10/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/hdp_10/raster.html" loading="lazy" title="HDP circuit, full recording (10n, 100 ms)"></iframe>

*HDP circuit, full recording (10n, 100 ms): spike raster.*

### `atlas/hdp_1000` — HDP column, short run (1000n, 200 ms)

State `validated` · N=1000 · edges=999000 · steps=400 · config `a6746c86b439` · jaxfne 0.5.5

=== "Still"

    ![HDP column, short run (1000n, 200 ms)](https://hnxj.github.io/jaxfne/figures/hdp_1000/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/hdp_1000/raster.html" loading="lazy" title="HDP column, short run (1000n, 200 ms)"></iframe>

*HDP column, short run (1000n, 200 ms): spike raster.*

### `atlas/homeostasis_1000` — Homeostasis column (1000n)

State `validated` · N=1000 · edges=999000 · steps=2000 · config `4dbc78e75818` · jaxfne 0.5.5

=== "Still"

    ![Homeostasis column (1000n)](https://hnxj.github.io/jaxfne/figures/homeostasis_1000/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/homeostasis_1000/raster.html" loading="lazy" title="Homeostasis column (1000n)"></iframe>

*Homeostasis column (1000n): spike raster.*

### `atlas/lfp_csd_12` — LFP/CSD laminar column (12n)

State `validated` · N=12 · edges=132 · steps=2000 · config `a07c380b8304` · jaxfne 0.5.0

=== "Still"

    ![LFP/CSD laminar column (12n)](https://hnxj.github.io/jaxfne/figures/lfp_csd_12/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/lfp_csd_12/raster.html" loading="lazy" title="LFP/CSD laminar column (12n)"></iframe>

*LFP/CSD laminar column (12n): spike raster.*

### `atlas` — Canonical V1 Column (1000n)

State `validated` · N=1000 · edges=215785 · steps=2000 · config `91f037cf383e` · jaxfne 0.5.8

=== "Still"

    ![Canonical V1 Column (1000n)](https://hnxj.github.io/jaxfne/figures/readme/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/readme/raster.html" loading="lazy" title="Canonical V1 Column (1000n)"></iframe>

*Canonical V1 Column (1000n): spike raster.*

### `atlas/network_100_ei` — Balanced E/I population (100n)

State `validated` · N=100 · edges=9900 · steps=1000 · config `e58d76183666` · jaxfne 0.5.0

=== "Still"

    ![Balanced E/I population (100n)](https://hnxj.github.io/jaxfne/figures/network_100_ei/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/network_100_ei/raster.html" loading="lazy" title="Balanced E/I population (100n)"></iframe>

*Balanced E/I population (100n): spike raster.*

### `atlas/objective_60` — Objective-grammar chain, pre-tune (60n)

State `validated` · N=60 · edges=3540 · steps=400 · config `48f956974352` · jaxfne 0.5.0

=== "Still"

    ![Objective-grammar chain, pre-tune (60n)](https://hnxj.github.io/jaxfne/figures/objective_60/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/objective_60/raster.html" loading="lazy" title="Objective-grammar chain, pre-tune (60n)"></iframe>

*Objective-grammar chain, pre-tune (60n): spike raster.*

### `atlas/omission_60` — Omission-oddball column, plain drive (60n)

State `validated` · N=60 · edges=3540 · steps=2000 · config `f3305c133965` · jaxfne 0.5.0

=== "Still"

    ![Omission-oddball column, plain drive (60n)](https://hnxj.github.io/jaxfne/figures/omission_60/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/omission_60/raster.html" loading="lazy" title="Omission-oddball column, plain drive (60n)"></iframe>

*Omission-oddball column, plain drive (60n): spike raster.*

### `atlas/operator_chain_40` — Operator-composition column (40n)

State `validated` · N=40 · edges=1560 · steps=200 · config `d22bc040c652` · jaxfne 0.5.0

=== "Still"

    ![Operator-composition column (40n)](https://hnxj.github.io/jaxfne/figures/operator_chain_40/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/operator_chain_40/raster.html" loading="lazy" title="Operator-composition column (40n)"></iframe>

*Operator-composition column (40n): spike raster.*

### `atlas/probe_32` — Probe-operators circuit, minimal run (32n, 200 ms)

State `validated` · N=32 · edges=992 · steps=400 · config `5690a65f088b` · jaxfne 0.5.0

=== "Still"

    ![Probe-operators circuit, minimal run (32n, 200 ms)](https://hnxj.github.io/jaxfne/figures/probe_32/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/probe_32/raster.html" loading="lazy" title="Probe-operators circuit, minimal run (32n, 200 ms)"></iframe>

*Probe-operators circuit, minimal run (32n, 200 ms): spike raster.*

### `atlas/scale_100` — Suite No. 3 scale N=100 (async patch)

State `validated` · N=100 · edges=9900 · steps=2000 · config `96a6c894ecad` · jaxfne 0.5.0

=== "Still"

    ![Suite No. 3 scale N=100 (async patch)](https://hnxj.github.io/jaxfne/figures/scale_100/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/scale_100/raster.html" loading="lazy" title="Suite No. 3 scale N=100 (async patch)"></iframe>

*Suite No. 3 scale N=100 (async patch): spike raster.*

### `atlas/single_neuron` — Single neuron (1n)

State `validated` · N=1 · edges=0 · steps=1000 · config `0cb7b98c0189` · jaxfne 0.5.0

=== "Still"

    ![Single neuron (1n)](https://hnxj.github.io/jaxfne/figures/single_neuron/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/single_neuron/raster.html" loading="lazy" title="Single neuron (1n)"></iframe>

*Single neuron (1n): spike raster.*

### `atlas/source_column_48` — Source-bookkeeping column (48n)

State `validated` · N=48 · edges=2256 · steps=2000 · config `e29a9e61e9eb` · jaxfne 0.5.0

=== "Still"

    ![Source-bookkeeping column (48n)](https://hnxj.github.io/jaxfne/figures/source_column_48/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/source_column_48/raster.html" loading="lazy" title="Source-bookkeeping column (48n)"></iframe>

*Source-bookkeeping column (48n): spike raster.*

### `atlas/suite1_column` — Suite No. 1 laminar column, smoke-scale (48n, 1000 ms of 5000 ms)

State `validated` · N=48 · edges=2256 · steps=2000 · config `316023c3edc9` · jaxfne 0.5.0

=== "Still"

    ![Suite No. 1 laminar column, smoke-scale (48n, 1000 ms of 5000 ms)](https://hnxj.github.io/jaxfne/figures/suite1_column/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/suite1_column/raster.html" loading="lazy" title="Suite No. 1 laminar column, smoke-scale (48n, 1000 ms of 5000 ms)"></iframe>

*Suite No. 1 laminar column, smoke-scale (48n, 1000 ms of 5000 ms): spike raster.*

### `atlas/suite2_net1` — Suite No. 2 net1 (100n)

State `validated` · N=100 · edges=9900 · steps=2000 · config `2cc90a7c1ef0` · jaxfne 0.5.0

=== "Still"

    ![Suite No. 2 net1 (100n)](https://hnxj.github.io/jaxfne/figures/suite2_net1/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/suite2_net1/raster.html" loading="lazy" title="Suite No. 2 net1 (100n)"></iframe>

*Suite No. 2 net1 (100n): spike raster.*

### `atlas/two_neuron_ei` — Two-neuron E/I (2n)

State `validated` · N=2 · edges=2 · steps=1000 · config `8ae5b8435e85` · jaxfne 0.5.0

=== "Still"

    ![Two-neuron E/I (2n)](https://hnxj.github.io/jaxfne/figures/two_neuron_ei/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/two_neuron_ei/raster.html" loading="lazy" title="Two-neuron E/I (2n)"></iframe>

*Two-neuron E/I (2n): spike raster.*

### `atlas/v1_column` — 600n population, laminar readout

State `validated` · N=600 · edges=359400 · steps=2000 · config `c79b0c87047f` · jaxfne 0.5.0

=== "Still"

    ![600n population, laminar readout](https://hnxj.github.io/jaxfne/figures/v1_column/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/v1_column/raster.html" loading="lazy" title="600n population, laminar readout"></iframe>

*600n population, laminar readout: spike raster.*

### `atlas/v1_pfc_dual` — V1-PFC dual column, single AAAB trial (200n, no HDP carryover)

State `validated` · N=200 · edges=240 · steps=2000 · config `f9433f57d4d5` · jaxfne 0.5.0

=== "Still"

    ![V1-PFC dual column, single AAAB trial (200n, no HDP carryover)](https://hnxj.github.io/jaxfne/figures/v1_pfc_dual/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/v1_pfc_dual/raster.html" loading="lazy" title="V1-PFC dual column, single AAAB trial (200n, no HDP carryover)"></iframe>

*V1-PFC dual column, single AAAB trial (200n, no HDP carryover): spike raster.*

### `atlas/v1v4_80` — V1-V4 scaffold (80/area)

State `validated` · N=160 · edges=12891 · steps=2000 · config `0699b80ad44b` · jaxfne 0.5.0

=== "Still"

    ![V1-V4 scaffold (80/area)](https://hnxj.github.io/jaxfne/figures/v1v4_80/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/v1v4_80/raster.html" loading="lazy" title="V1-V4 scaffold (80/area)"></iframe>

*V1-V4 scaffold (80/area): spike raster.*

### `atlas_three_area` — V1-V4-PFC three-area hierarchy (100/area)

State `validated` · N=300 · edges=30105 · steps=1000 · config `84f67ccf815f` · jaxfne 0.5.5

=== "Still"

    ![V1-V4-PFC three-area hierarchy (100/area)](https://hnxj.github.io/jaxfne/figures/three_area/raster.png)

=== "Interactive"

    <iframe class="jx-frame" src="https://hnxj.github.io/jaxfne/figures/three_area/raster.html" loading="lazy" title="V1-V4-PFC three-area hierarchy (100/area)"></iframe>

*V1-V4-PFC three-area hierarchy (100/area): spike raster.*

<!-- ATLAS-INDEX:END -->


