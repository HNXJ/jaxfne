# Model Augmentation Science Runs Plan: Scale Ladder and Heterogeneity

Status: PROPOSED PLAN ONLY (2026-10-01). Authority: item 0a of `artifacts/todo_stack.md`.
No long simulation runs are executed until approved by Hamm.

---

## 1. Scope & Objective

Verify that model augmentation (`jaxfne.augment`) preserves biologically coherent,
asynchronous-irregular network dynamics across a 10-fold scale ladder ($10^3 \to 10^4$ neurons)
under controlled parameter heterogeneity across all six augmentation axes:
$$N \to G \to \Theta_C \to \Theta_X \to W_0 \to H_0$$

The central theoretical hypothesis:
Because `ScaleN` enforces balanced synaptic scaling ($w \sim 1/\sqrt{N}$), ensemble firing
rates, inter-spike interval regularities ($CV_{ISI}$), pairwise spike correlations, and LFP
spectral motifs should remain invariant or within tight, declared bounds across the ladder.

---

## 2. Scale Ladder & Experimental Design

### 2.1 Network Ladder Steps
All models derive from the canonical column specification (`canonical-v1-column-1000n.json`):
- **Level 0 (Base)**: $N = 1{,}000$ neurons ($\sim 215{,}785$ edges, baseline reference).
- **Level 1**: $N = 2{,}000$ neurons ($s_N = 2.0$, $\sim 863{,}000$ edges).
- **Level 2**: $N = 5{,}000$ neurons ($s_N = 5.0$, $\sim 5{,}400{,}000$ edges).
- **Level 3 (Target)**: $N = 10{,}000$ neurons ($s_N = 10.0$, $\sim 21{,}578{,}500$ edges).

Each simulation runs for $T = 1{,}000.0$ ms with $dt = 0.5$ ms ($2{,}000$ integration steps)
under fixed baseline Poisson drive.

### 2.2 Heterogeneity Arms
For each scale level $N$, evaluate three heterogeneity conditions:
1. **Arm A (Homogeneous Baseline)**: Pure `ScaleN(s_N)` with zero parameter dispersion ($\sigma = 0$).
2. **Arm B (Mild Biological Jitter, $\sigma = 0.05$)**:
   - $\Theta_X$: Conductance $g_{mech}$ and time constant $\tau_{mech}$ perturbed with log-normal jitter ($\sigma = 0.05$).
   - $\Theta_C$: Connection delay $\Delta_{ms}$ jittered by $\pm 0.05 \Delta$ (bounded $\ge dt$).
   - $W_0$: Synaptic weight gain jittered log-normally with $\sigma_w = 0.05$.
   - $H_0$: RBS initial state additive jitter with $\sigma_H = 0.05$.
3. **Arm C (Substantial Biological Heterogeneity, $\sigma = 0.15$)**:
   - Same axes perturbed with $\sigma = 0.15$, testing dynamical stability against broad biophysical dispersion.

Seeds: 3 distinct random seeds ($K_V \in \{101, 202, 303\}$) per condition to distinguish
structural convergence from sample noise.

---

## 3. Ensemble Observables & Acceptance Tolerances

| Observable | Measure | Target / Baseline Range | Scale Invariance Tolerance ($10^3 \to 10^4$) | Heterogeneity Tolerance (Arm A $\to$ C) | Rationale |
|---|---|---|---|---|---|
| **E Rate ($\nu_E$)** | Population mean spike rate of E units | $3.0 - 15.0$ Hz | Ratio in $[0.75, 1.33]$ ($\pm 25\%$) | Ratio in $[0.70, 1.40]$ | Balanced E/I scaling ($1/\sqrt{N}$) maintains fixed net mean background current (van Vreeswijk & Sompolinsky 1998). |
| **PV Rate ($\nu_{PV}$)** | Population mean spike rate of PV fast-spiking units | $15.0 - 45.0$ Hz | Ratio in $[0.75, 1.33]$ ($\pm 25\%$) | Ratio in $[0.70, 1.40]$ | Interneuron tracking of pyramidal activity maintains balance without runaway inhibition. |
| **SST / VIP Rates** | Population mean rates of SST / VIP interneurons | $2.0 - 15.0$ Hz | Ratio in $[0.70, 1.40]$ | Ratio in $[0.65, 1.50]$ | Disinhibitory and feedback interneurons stay within active physiological regime. |
| **Spike Irregularity ($CV_{ISI}$)** | $std(ISI) / mean(ISI)$ for units with $\ge 5$ spikes | $0.80 - 1.40$ (Asynchronous-Irregular) | Max abs diff $\le 0.15$ | Max abs diff $\le 0.20$ | $CV \approx 1$ confirms asynchronous irregular regime; prevents transition to synchronous bursting ($CV > 2$) or clock-like firing ($CV < 0.5$). |
| **LFP Peak Frequency ($f_{peak}$)** | Frequency of max PSD in $15-80$ Hz via `jnwb.compute_psd` | Canonical gamma/beta peak ($30-50$ Hz) | Shift $\le \pm 4.0$ Hz | Shift $\le \pm 6.0$ Hz | PING/ING resonance frequency is primarily determined by synaptic time constants ($\tau_{GABA_A}$, $\tau_{AMPA}$) and delays, invariant to $N$. |
| **Pairwise Correlation ($r_{sc}$)** | Mean Pearson correlation of $20$ ms binned spike counts | $-0.01 \le r_{sc} \le 0.04$ | $|r_{sc}(10^4) - r_{sc}(10^3)| \le 0.02$ | $|r_{sc}(Arm C) - r_{sc}(Arm A)| \le 0.03$ | Fast negative feedback actively cancels shared fluctuations in balanced networks (Renart et al. 2010). |
| **Population Fano ($FF_{pop}$)** | Variance-to-mean ratio of $50$ ms binned population counts | $0.80 - 2.50$ | Ratio in $[0.70, 1.40]$ | Ratio in $[0.65, 1.50]$ | Prevents large-scale population hypersynchrony while allowing realistic rate variance. |

---

## 4. Computational Resource & Memory Estimates

| Scale ($N$) | Realized Edges | Est. Memory Peak | Est. Run Duration (CPU) | Sub-suite Cost (3 seeds x 3 arms) |
|---|---|---|---|---|
| $1{,}000$ | $215{,}785$ | $\sim 85$ MB | $\sim 2.5$ s | $9 \times 2.5\text{s} \approx 23$ s |
| $2{,}000$ | $863{,}140$ | $\sim 220$ MB | $\sim 6.0$ s | $9 \times 6.0\text{s} \approx 54$ s |
| $5{,}000$ | $5{,}394{,}625$ | $\sim 1.1$ GB | $\sim 22.0$ s | $9 \times 22.0\text{s} \approx 3.3$ min |
| $10{,}000$ | $21{,}578{,}500$ | $\sim 3.8$ GB | $\sim 65.0$ s | $9 \times 65.0\text{s} \approx 9.8$ min |
| **Total** | — | **Peak $\sim 3.8$ GB** | — | **Total Wall-Time: $\sim 14.5$ minutes** |

### Safety Constraints & Gates
1. **Memory Ceiling**: Enforce `assert psutil.virtual_memory().available > 6.0 * 1024**3` before launching the $N=10{,}000$ arm to prevent swapping or memory exhaustion.
2. **Single-Run Timeout**: Each simulation is capped at 180 seconds.
3. **Deterministic Seed Binding**: Every run records `seed`, `spec_digest`, `config_hash`, and git commit SHA in its receipt.

---

## 5. Decision & Execution Options for Hamm

Grade per Operating Contract §2a:

| Grade | Confidence | Option | Details |
|---|---|---|---|
| **Recommended** | **80%** | **Phased Ladder ($10^3 \to 2{,}000 \to 5{,}000$) first** | Execute levels 0, 1, and 2 (max runtime $\sim 4.5$ min total); verify observable convergence and memory footprint before running the final $10{,}000$-neuron tier. |
| Alternative | 60% | Single-seed probe ($N=10^3$ vs $N=10^4$) | Run one seed at $10^3$ and one at $10^4$ under Arm B to measure concrete timing and memory before full matrix. |
| Full Sweep | 50% | Complete 36-run matrix ($10^3 \to 10^4$, all arms & seeds) | Execute the entire suite in one automated batch ($\sim 15$ min wall-clock). |

Awaiting Hamm's instruction before initiating any simulation runs.
