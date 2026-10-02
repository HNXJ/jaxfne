# Atlas

> **Status:** Canonical Atlas source for the JaxFNE 0.5.x programme
> (project source 8 of 8).
>
> **Identity:** S1-S10 correspond to AT-01-AT-10. Requirement IDs (`AT-0n-R<k>`) and mutable coverage state are maintained separately in `artifacts/programme/atlas_coverage.json`.
>
> **Boundary:** This source distinguishes the Atlas design and required experiments from capabilities it requests but does not itself establish. In particular: proxy != calibrated; correlation != causal field feedback; attenuation != adaptation; bounded != active stability. Calibrated HH/field behavior and field feedback (`Phi -> X`) require independent evidence and authorization where not already established.
>
> **Release ownership:** The 0.5.x programme maps shared engine capabilities and Atlas integration across releases; mutable progress/validation state belongs in the coverage file, not this source.
>
> **Symbol B (human decision 2026-09-23):** `B` is the magnetic field everywhere in this source (`\Phi_B` in `Y`). Where `B` appears as an input to neural or plastic dynamics — `(X,H,W,B)\rightarrow Q`, `F_W(X,H,W,B)` — it is a field-derived input and, like `\Phi\rightarrow X`, a candidate capability, not an established one.
>
> **Transcription:** the header rows of the two tables below were merged into one cell when the text was pasted; they are restored into columns. No other change to the source text.

---

## Atlas structure

Yes. I would compress the previous Atlas into **10 canonical simulations**, not 10 mechanisms. Each simulation should expose the *same* JaxFNE objects at the spacetime scale where they matter:

```math
\boxed{ G \xrightarrow{D} N \rightarrow (X,H,W,B) \rightarrow Q \rightarrow \Phi(\mathbf r,t) \rightarrow Y }
```

Complexity increases monotonically:

```math
1N\rightarrow2N\rightarrow\text{population}\rightarrow2\text{ areas}\rightarrow20\text{ areas}.
```

This directly matches TFNE's intended recursive scale invariance, from single cell through populations and larger neural systems.

### Proposed 10 simulations

| # | System | Space / time of interest | Manipulation | Main scientific demonstration |
| --- | --- | --- | --- | --- |
| **S1** | **1 full HH neuron** | `\mu m`, `\mu s-ms` | current injection; subthreshold → AP | physical membrane dynamics → transmembrane currents → `Q` → local `E/\Phi`, and where justified `B`; establish calibration/reference model |
| **S2** | **2 neurons: E→E / driven pair** | `10-10^3\,\mu m`, ms | distance, delay, synaptic strength | spike → synaptic current → second neuron; individual versus superposed fields; distance law |
| **S3** | **2 neurons: E↔I oscillator** | `10-10^3\,\mu m`, ms–100 ms | E/I coupling, delay, drive | minimal recurrent oscillation; phase, cancellation/reinforcement, frequency-dependent field |
| **S4** | **2 neurons + field/state coupling** | `10-10^3\,\mu m`, ms–s | geometry/orientation + `H` or justified `\Phi\rightarrow X` perturbation | distinguish correlation `X\to\Phi` from causal field/state effect on excitability |
| **S5** | **E/I population** | `0.1-2` mm, ms–s | `N,\rho,` synchrony | emergence of population field/LFP from microscopic sources; coherent vs incoherent summation |
| **S6** | **structured cortical population** | layer/column scale, ms–s | spatial arrangement, E/I composition, source orientation | electrode locality: source-distance contribution, geometry, contact/reference/filtering |
| **S7** | **plastic population** | column scale, seconds–minutes/model-time | Hebbian/HDP/noisy HDP | `X,H\rightarrow\Delta W\rightarrow\Delta X\rightarrow\Delta Q\rightarrow\Delta\Phi`; adaptation/stability versus mere attenuation |
| **S8** | **2 areas: repeated-input adaptation** | mm–cm, ms–minutes | repeated stimulus; adaptation enabled/clamped | local adaptation and transmission: area 1 → area 2; spikes + field attenuation; causal `H` test |
| **S9** | **2 areas: adaptive/plastic coupling** | mm–cm, ms–long horizon | HDP/Hebbian/noisy rule, delays | local vs inter-area adaptation; changing effective connectivity; field/coherence consequences |
| **S10** | **20-area PseudoGenome/JDNA system** | cm/network scale, ms–long horizon | compact developmental specification; baseline → Hebbian HDP → noisy HDP | compact genome → realized multiarea system → stable long execution; multiscale observables and plastic stabilization |

That gives exactly:

```math
\boxed{1+3+3+2+1=10}.
```

---

# The connecting logic

The important part is that these should **not be ten independent demos**.

Each should inherit the previous scientific objects:

```math
\begin{array}{rcl} S_1 &: & X,Q,\Phi\\ S_{2-4} &: & +\,W,B,\mathbf r,\text{interaction}\\ S_{5-7} &: & +\,N,\rho,\text{collective state}\\ S_{8-9} &: & +\,\text{area hierarchy, long delays}\\ S_{10} &: & +\,G,D,\text{large-scale composition}. \end{array}
```

Thus scale changes, but the computational language does not.

That is exactly the TFNE proposition:

```math
\boxed{\text{factor at specification time; flatten at execution time}}
```

and eventually

```math
\mathcal A \rightarrow (s,h_0,\mathcal I) \rightarrow \text{JaxFNE execution}.
```

## Spacetime should deliberately change

This is important. We should **not run everything at the same** **`\Delta t,T,\Delta r`** merely for consistency.

Instead:

```math
(\Delta t,T,\Delta r) = f(\text{phenomenon being resolved}).
```

Conceptually:

| Scale | Resolve | Typical simulation emphasis |
| --- | --- | --- |
| HH | AP/current kinetics | very fine `\Delta t`, short `T` |
| pair | synapse/delay/phase | fine `\Delta t`, short-medium `T` |
| population | oscillations/LFP | ms resolution, seconds |
| adaptation | `H,W` dynamics | sufficient fast resolution + long `T` |
| 20-area | propagation/plasticity/stability | reduced fast model + very long `T` |

This becomes a methodological result itself:

```math
\boxed{\text{model detail should follow the spacetime scale of the observable}}
```

rather than assuming maximum detail is universally better.

That agrees with the project's observable-specific reduction principle: two models are equivalent only relative to declared observations and tolerances.

---

# S1 should be the physical anchor

I particularly like putting **full HH first**.

It gives us a physically interpretable reference from which later reductions can be measured:

```math
M_{\rm HH} \rightarrow M_{\rm reduced} \rightarrow M_{\rm population}.
```

For S1 we should extract at minimum:

```math
V_m,\quad I_{\rm Na},I_{\rm K},I_L,\quad I_m,\quad Q,\quad \Phi(\mathbf r,t).
```

Then ask which quantities survive reduction.

This prevents the paper from starting with arbitrary dimensionless neural units and only later claiming physical interpretation.

But we must preserve the existing boundary: JaxFNE's reduced emitters are **not currently claims to solve full electrodiffusion**.

---

# S2–S4: three pair experiments should answer different questions

I would avoid merely changing connectivity.

### S2 — transmission

```math
N_1\rightarrow N_2
```

Question:

```math
\boxed{\text{How does activity propagate between discrete neural sources?}}
```

Vary distance/delay/weight.

### S3 — recurrence

```math
E\rightleftarrows I
```

Question:

```math
\boxed{\text{How do interacting currents produce oscillatory neural + field dynamics?}}
```

This establishes phase, synchrony and cancellation.

### S4 — causality

```math
X\rightarrow Q\rightarrow\Phi
```

versus, **only if physically implemented**,

```math
X\rightarrow\Phi\rightarrow X.
```

Question:

```math
\boxed{\text{Is the field merely observed, or can it alter neural dynamics?}}
```

This is scientifically much more useful than a third connectivity configuration.

---

# S5–S7: emergence

These should establish the transition

```math
\boxed{\text{microscopic sources}\rightarrow\text{mesoscopic field}}.
```

### S5 — scaling

Control synchrony:

```math
\rho_{\rm sync}:0\rightarrow1.
```

Measure something like

```math
A_\Phi(N,\rho_{\rm sync},r).
```

The key result should show why

```math
\Phi_N\neq N\Phi_1
```

generally.

### S6 — measurement

Introduce the electrode:

```math
Q_i \rightarrow \Phi(\mathbf r,t) \rightarrow P_{\rm electrode} \rightarrow V_{\rm LFP}(t).
```

Then quantify locality:

```math
C(R,f)= \frac{ P_{\rm electrode}\left[\sum_{r_i<R}\Phi_i(f)\right] }{ P_{\rm electrode}\left[\sum_i\Phi_i(f)\right] }.
```

This directly asks **“how local is local field potential?”**

The project's current source explicitly says LFP interpretation depends on geometry, correlation, spatial arrangement, conductivity, distance and frequency.

### S7 — adaptation/plasticity

Now:

```math
X,H,B,W \rightarrow Q \rightarrow\Phi.
```

Compare:

```math
\begin{cases} W=\text{fixed}\\ \text{Hebbian HDP}\\ \text{noisy HDP} \end{cases}
```

under identical stimulation.

This exposes **slow biological state through fast electromagnetic observations**.

---

# S8–S9: two-area systems

These should be the bridge between local LFP and large-scale network dynamics.

## S8: adaptation propagates

```math
A_1\rightarrow A_2.
```

Repeated input to `A_1`.

Measure simultaneously:

```math
\begin{aligned} &SPK_1,\Phi_1,H_1\\ &SPK_2,\Phi_2,H_2\\ &\Delta\phi_{12}(f),\quad C_{12}(f). \end{aligned}
```

Question:

> Does adaptation remain local, propagate through connectivity, or alter downstream fields without equivalent changes in spike count?

## S9: connectivity itself adapts

```math
A_1 \underset{W_{21}(t)}{\stackrel{W_{12}(t)}{\rightleftarrows}} A_2.
```

Now distinguish:

```math
\text{fast dynamics }X \quad|\quad \text{relative state }H \quad|\quad \text{plastic parameters }W.
```

That distinction is foundational to JaxFNE: `H\neq\mathrm{HDP}`, and `H` may evolve while `\dot W=0`.

---

# S10 — the synthesis

This should be visually spectacular but scientifically controlled.

Start from a **compact PseudoGenome**:

```math
G_{20}
```

and development:

```math
\boxed{ G_{20}\xrightarrow{D(K_D)} N_{20} }
```

where `N_{20}` contains 20 areas, populations, geometry, connectivity, delays, neural models, `H`, HDP and field/probe definitions.

PseudoGenome development is already canonical JaxFNE semantics.

Then three phases using the **same realized network**:

### S10a — baseline

```math
W=W_0,\qquad \dot W=0.
```

Establish:

- spontaneous/noisy activity;
- firing-rate distributions;
- population fields;
- spectra;
- inter-area propagation;
- boundedness;
- spatial field structure.

### S10b — Hebbian HDP

Enable:

```math
\dot W=F_W(X,H,W,B).
```

Measure:

```math
W(t),H(t),X(t),\Phi(t)
```

and network-level organization.

### S10c — noisy HDP

Add stochastic parameter dynamics through explicit `K_t`:

```math
W_{t+1} = F_W(X_t,H_t,W_t,B_t,K_t).
```

Then ask the strong question:

```math
\boxed{ \text{Can local plasticity rules maintain useful bounded network dynamics under persistent stochastic perturbation?} }
```

But terminology matters: **bounded trajectories are not automatically evidence of active stability.** The project explicitly preserves that distinction.

So S10 should include a perturbation/control assay capable of distinguishing:

```math
\text{bounded} \neq \text{returning} \neq \text{homeostatically stabilized}.
```

---

## One common measurement vector

This will make the ten simulations feel like **one experiment at ten scales**.

For every simulation record whatever subset exists of

```math
\boxed{ Y= \{ X,H,W,Q,\Phi_E,\Phi_B, SPK,PSD,C,\phi, E_{\rm reduction}, T_{\rm compute},M_{\rm compute} \}. }
```

Then every figure can use the same conceptual columns:

**Structure → dynamics → state/plasticity → source → field → observation → computation.**

That gives the manuscript a clean progression:

```math
\boxed{ \begin{array}{c} \textbf{one physical neuron}\\ \downarrow\\ \textbf{interacting neurons}\\ \downarrow\\ \textbf{emergent population field}\\ \downarrow\\ \textbf{adaptive interacting areas}\\ \downarrow\\ \textbf{genome-defined multiarea brain model} \end{array}}
```

while underneath everything remains

```math
\boxed{ (h_t,x_t;s)\mapsto(h_{t+1},y_t). }
```

That is the connection I think makes the Atlas much stronger: **10 simulations, but one mathematical object progressively exposed across** **`\sim\mu m\rightarrow cm`** **and** **`\sim\mu s\rightarrow`** **long adaptive timescales.** The paper then demonstrates not that JaxFNE has many features, but that **one small simulation grammar survives the change of scale.**
