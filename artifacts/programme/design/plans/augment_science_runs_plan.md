# Model Augmentation Science Runs Plan: Scale Ladder and Heterogeneity

Status: PROPOSED PLAN ONLY (2026-10-01 revision 2). Authority: item 0a of `artifacts/todo_stack.md`.
No long simulation runs are executed until approved by Hamm.

---

## 1. Scope & Theoretical Framework

Verify model augmentation (`jaxfne.augment`) dynamics across a 10-fold scale ladder
($10^3 \to 10^4$ neurons) under controlled parameter heterogeneity across all six augmentation axes:
$$N \to G \to \Theta_C \to \Theta_X \to W_0 \to H_0$$

### 1.1 Balanced Scaling & External Drive Realization
In `jaxfne`, model construction realizes:
1. **Recurrent Synaptic Weights**: Synaptic connection weights scale as $w \sim 1/\sqrt{N_{total}}$
   via `_connection_edge_weight`.
2. **Connectivity Density**: Rule-based layer-to-layer connections compile full bipartite graphs
   ($p = 1.0$), so the number of recurrent presynaptic inputs per neuron scales linearly with
   network size: $K \propto N$. Total recurrent input therefore scales as:
   $$I_{rec} \sim K \cdot w \propto N \cdot \frac{1}{\sqrt{N}} = \sqrt{N}$$
3. **External Drive**: Unlike classic theoretical balanced network models (van Vreeswijk &
   Sompolinsky 1996, 1998) where external drive is scaled as $I_{ext} \propto \sqrt{K}$ so that
   $\mathcal{O}(\sqrt{K})$ excitation and inhibition cancel to yield exact rate invariance, `jaxfne`
   realizes a **fixed external drive** ($I_{ext} = \text{const}$, configured via `params.drive` or
   `drive()` per cell type / neuron, $\mathcal{O}(1)$).

**Consequence for Scaling Hypothesis**:
Because recurrent input grows as $\sqrt{N}$ while external drive remains constant $\mathcal{O}(1)$,
exact rate invariance across $N$ is **not theoretically predicted**. Mean population rates will
drift as recurrent feedback strengthens relative to intrinsic drive.
Therefore, rate trajectories across scale are treated as a **measured empirical observable** rather
than a rigid pass/fail criterion. The pass criterion for scaling is **dynamical stability**: the
network must remain in a stable active regime without collapsing into quiescence ($\nu_E < 0.5$ Hz)
or runaway saturation ($\nu_E > 35.0$ Hz).

---

## 2. Scale Ladder & Experimental Design

### 2.1 Network Ladder Steps
All models derive from the canonical column specification (`canonical-v1-column-1000n.json`):
- **Level 0 (Base Reference)**: $N = 1{,}000$ neurons ($215{,}785$ edges, measured).
- **Level 1**: $N = 2{,}000$ neurons ($s_N = 2.0$, $863{,}140$ edges, measured).
- **Level 2**: $N = 5{,}000$ neurons ($s_N = 5.0$, $\sim 5{,}394{,}625$ edges, estimated).
- **Level 3 (Target)**: $N = 10{,}000$ neurons ($s_N = 10.0$, $\sim 21{,}578{,}500$ edges, estimated).

Each simulation runs for $T = 1{,}000.0$ ms with $dt = 0.5$ ms ($2{,}000$ integration steps).

### 2.2 Heterogeneity Arms & Realized Jitter Distributions
In accordance with `docs/api/augment.md`, `jaxfne.augment` implements bounded uniform draws
(no log-normal distributions):
- **Multiplicative Relative Uniform Jitter**: $x \sim U[(1 - j) x_0, (1 + j) x_0]$ for positive quantities:
  - $\Theta_X$: `g_mech` and `dT_ms` (`g_jitter=j`, `tau_jitter=j`).
  - $\Theta_C$: `delay_ms` (`delay_jitter=j`) and `probability` (`probability_jitter=j`).
    *Note*: `construct` quantizes continuous `delay_ms` to `round(delay_ms / dt_ms)` steps and
    refuses delays rounding to 0 steps; `augment` does not bound delays by $dt$.
  - $W_0$: Synaptic weight gain `w_mech` (`jitter=j`).
- **Additive Uniform Jitter**: $H \sim H_0 + U[-j, +j]$:
  - $H_0$: RBS initial state `plastic.H` (`jitter=j`).

Three experimental conditions per scale tier:
1. **Arm A (Homogeneous Baseline, $j = 0$)**: Pure `ScaleN(s_N)` with zero parameter dispersion.
2. **Arm B (Mild Biological Jitter, $j = 0.05$)**: All active axes perturbed with uniform jitter $j = 0.05$.
3. **Arm C (Substantial Biological Heterogeneity, $j = 0.15$)**: All active axes perturbed with uniform jitter $j = 0.15$.

Seeds: 3 distinct random seeds ($K_V \in \{101, 202, 303\}$) per condition.

---

## 3. Ensemble Observables & Acceptance Tolerances

Empirical baseline measured on $N=1{,}000$ canonical column ($T=1{,}000$ ms, $dt=0.5$ ms, seed 0):

| Observable | Measure | Empirical 1k Baseline Reference | Scale Scaling Criterion ($10^3 \to 10^4$) | Heterogeneity Tolerance (Arm A $\to$ C) | Rationale |
|---|---|---|---|---|---|
| **E Rate ($\nu_E$)** | Population mean spike rate of E units | **$11.00$ Hz** | Measured observable; stability bound $[0.5, 35.0]$ Hz | Ratio in $[0.70, 1.40]$ vs Arm A | Fixed drive with $w \sim 1/\sqrt{N}$ alters E/I net balance; stability requires avoiding silence or seizure. |
| **PV Rate ($\nu_{PV}$)** | Population mean spike rate of PV units | **$0.84$ Hz** | Measured observable; stability bound $[0.1, 25.0]$ Hz | Ratio in $[0.60, 1.60]$ vs Arm A | Tracks recurrent inhibition; canonical scaffold PV units fire sparsely at baseline. |
| **SST Rate ($\nu_{SST}$)** | Population mean spike rate of SST units | **$47.88$ Hz** | Measured observable; stability bound $[20.0, 75.0]$ Hz | Ratio in $[0.70, 1.40]$ vs Arm A | SST units maintain primary pacemaker/inhibition tone in canonical column scaffold. |
| **VIP Rate ($\nu_{VIP}$)** | Population mean spike rate of VIP units | **$0.56$ Hz** | Measured observable; stability bound $[0.05, 15.0]$ Hz | Ratio in $[0.50, 2.00]$ vs Arm A | Disinhibitory interneurons remain in low-rate baseline regime. |
| **Spike Irregularity ($CV_{ISI}$)** | $std(ISI) / mean(ISI)$ for units with $\ge 5$ spikes | **$0.036$** (Rhythmic / clock-like) | Monitored trajectory; $CV_{ISI} \le 1.8$ | Measure desynchronization (Arm C $CV_{ISI} > 0.10$) | Unperturbed canonical column fires rhythmically ($CV \approx 0.04$); heterogeneity breaks lockstep. |
| **LFP Peak Frequency ($f_{peak}$)** | Frequency of max PSD in $15-80$ Hz via `jnwb.compute_psd` | **$21.0$ Hz** (Beta resonance) | Shift $\le \pm 4.0$ Hz | Shift $\le \pm 6.0$ Hz | Beta rhythm frequency is determined by synaptic time constants and delays, invariant to $N$. |
| **Pairwise Correlation ($r_{sc}$)** | Mean Pearson correlation of $20$ ms binned spike counts | **$0.609$** (Synchronous population) | Monitored trajectory; $|r_{sc}(10^4) - r_{sc}(10^3)| \le 0.25$ | Measure decorrelation (Arm C $r_{sc} < 0.50$) | Baseline canonical scaffold exhibits high synchrony; jitter provides desynchronizing dispersion. |
| **Population Fano ($FF_{pop}$)** | Variance-to-mean ratio of $50$ ms binned population counts | **$211.8$** | Bounded; ratio in $[0.3, 3.0]$ vs 1k | Bounded; ratio in $[0.3, 3.0]$ vs Arm A | High baseline Fano reflects synchronous population bursts across the column. |

---

## 4. Computational Resource & Memory Measurements & Estimates

| Scale ($N$) | Realized Edges | Peak Memory | Run Duration (CPU) | Measurement Status | Sub-suite Cost (3 seeds x 3 arms) |
|---|---|---|---|---|---|
| $1{,}000$ | $215{,}785$ | $530.6$ MB | $\sim 2.9$ s sim ($\sim 5.0$ s cold const) | **MEASURED** | $9 \times 3.5\text{s} \approx 32$ s |
| $2{,}000$ | $863{,}140$ | $567.0$ MB | $\sim 5.7$ s sim ($\sim 9.8$ s cold const) | **MEASURED** | $9 \times 8.0\text{s} \approx 72$ s |
| $5{,}000$ | $\sim 5{,}394{,}625$ | $\sim 1.2$ GB | $\sim 18.0$ s sim ($\sim 25.0$ s const) | **ESTIMATED** | $9 \times 25.0\text{s} \approx 3.8$ min |
| $10{,}000$ | $\sim 21{,}578{,}500$ | $\sim 3.8$ GB | $\sim 65.0$ s sim ($\sim 80.0$ s const) | **ESTIMATED** | $9 \times 80.0\text{s} \approx 12.0$ min |
| **Total** | — | **Peak $\sim 3.8$ GB** | — | — | **Total Wall-Time: $\sim 18$ minutes** |

### Safety Constraints & Gates
1. **Memory Ceiling**: Enforce `assert psutil.virtual_memory().available > 6.0 * 1024**3` before launching the $N=10{,}000$ arm to prevent swapping or memory exhaustion.
2. **Single-Run Timeout**: Each simulation is capped at 180 seconds.
3. **Deterministic Seed Binding**: Every run records `seed`, `spec_digest`, `config_hash`, and git commit SHA in its receipt.

---

## 5. Decision & Execution Options for Hamm

Grade per Operating Contract §2a:

| Grade | Confidence | Option | Details |
|---|---|---|---|
| **Recommended** | **80%** | **Phased Ladder ($10^3 \to 2{,}000 \to 5{,}000$) first** | Execute levels 0, 1, and 2 (max runtime $\sim 5.5$ min total); verify observable scaling and memory footprint before running the final $10{,}000$-neuron tier. |
| Alternative | 60% | Single-seed probe ($N=10^3$ vs $N=10^4$) | Run one seed at $10^3$ and one at $10^4$ under Arm B to measure concrete timing and memory before full matrix. |
| Full Sweep | 50% | Complete 36-run matrix ($10^3 \to 10^4$, all arms & seeds) | Execute the entire suite in one automated batch ($\sim 18$ min wall-clock). |

Awaiting Hamm's instruction before initiating any simulation runs.
