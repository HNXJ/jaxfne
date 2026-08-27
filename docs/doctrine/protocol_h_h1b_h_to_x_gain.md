# H-to-X Gain

*H-to-X gain interface — protocol H1b, specification. Typed gain map from hidden state to emitter; protocol identifier in provenance.*

**Status:** CLOSED (H1c-C implemented); see `protocol_h_rbd_memory.md`  
**Architecture:** `docs/doctrine/tfne_containment_architecture.md` — H1b is the
**first emitter-local** typed gain map; RBS may eventually couple to \(S,F,P\)
with strict typing.

## 0. Architectural scope (refinement post-H1a)

RBS is not intrinsically emitter-centered. The containment formulation allows

\[
\mathcal T_H:(E,S,F,P,O,A)_t\rightarrow(E,S,F,P,O,A)_{t+1},
\]

with declared maps such as \(H_{\mathrm{ion}}\rightarrow\gamma(H)\) or
\(H_{\mathrm{ATP}}\rightarrow\theta(H)\). **H1b addresses only**
\(H\rightarrow x\) on the emitter — the smallest coupling needed before neural
memory tests. Field/source couplings are out of scope until explicitly specified.

## 1. Scientific issue (why H1b exists)

H1 implements

\[
\tau_H \dot H_i = R(H_i) + \kappa_H I_i^{\mathrm{rel}},
\qquad
\dot{\mathbf x}_i = F_x(\mathbf x_i, \mathbf I_i)\;\text{(unchanged)},
\]

so with \(\kappa_H=0\) a localized RBS perturbation \(H_k = 1+\delta_H\) can relax in
**\(H\) alone** without affecting spikes, voltages, or synaptic state. That is a valid
**RBS state test**, but it is **not** yet the RBD hypothesis we need for network memory:

\[
\boxed{H \rightarrow x}
\]

must be realized somewhere if hidden biophysical state is to influence emitter
behavior and thereby express adaptation or memory through **neural observables**.

Building \(M(\Delta)\) in H3 before \(H \rightarrow x\) is specified risks measuring
mostly **direct access to \(H\)**, not memory expressed through dynamics.

## 2. Coupling grammar (frozen distinction)

Generic RBD separates two directions:

\[
\dot H_i = F_H(H_i, x_i, I_i;\kappa_{x\rightarrow H}),
\qquad
\dot x_i = F_x(x_i, I_i; G(H_i;\kappa_{H\rightarrow x})).
\]

| Checkpoint | Direction | H1 status |
|------------|-----------|-----------|
| **H1a** | \(x/I \rightarrow H\) | **Implemented** — \(\kappa_H I^{\mathrm{rel}}\) in `simulate_edge_recurrent_izhikevich_rbd` |
| **H1b** | \(H \rightarrow x\) | **This document** — gain interface to be selected, then implemented as H1c |
| **H2** | Full-state continuation incl. \(\mathcal B_t\) | Blocked until H1b frozen |
| **H3** | Localized RBS perturbation + \(M(\Delta)\) | Blocked until H1c implements minimal \(H \rightarrow x\) |

Ladder:

\[
\boxed{\text{H1a }(x/I\!\to\! H)\;\checkmark}
\rightarrow
\boxed{\text{H1b spec }(H\!\to\! x)}
\rightarrow
\boxed{\text{H1c impl}}
\rightarrow
\text{H2}\rightarrow\text{H3}.
\]

## 3. Gain interface doctrine (not biological interpretation)

\(H \rightarrow x\) must be a **gain interface** on emitter quantities, not a
special biological reading baked into RBS coordinates.

Template:

\[
p_i^{\mathrm{eff}} = g_p(H_i;\kappa_{H\rightarrow x})\, p_i,
\qquad
g_p(1)=1,
\qquad
\kappa_{H\rightarrow x}=0 \Rightarrow g_p(H)\equiv 1.
\]

Candidate forms ( **not selected** in H1b):

| ID | Sketch | Notes |
|----|--------|-------|
| **G-exp** | \(g(H;\beta)=\exp[\beta(H-1)]\) | Positive, \(g(1)=1\), \(g(H)\approx 1+\beta(H-1)\) locally |
| **G-lin** | \(g(H;\beta)=1+\beta(H-1)\) | Simplest local gain; can violate positivity for large \(\|H-1\|\) |
| **G-inv** | \(g(H;\beta)=H^\beta\) | \(g(1)=1\); requires \(H>0\) if non-integer \(\beta\) |

**Do not select** a gain family in H1b. First inventory modulatable emitter
quantities; then choose the smallest interface that preserves Izhikevich semantics.

## 4. Inventory — edge-recurrent Izhikevich kernel (`simulate_edge_recurrent_izhikevich_rbd`)

Activity step (current H1):

```text
syn_i     = segment_sum(weight * syn_state, post=i)
current_i = drive_i + sched_i + syn_i + noise_coef * noise_i
dv, du    = izhikevich_dv_du(v, u, current_i, a, b)
```

Per-neuron quantities that can host \(g(H)\) without changing graph topology:

| Target | Effective form | Affects \(F_x\)? | Affects probe? | Semantics risk |
|--------|----------------|------------------|----------------|----------------|
| **A. Total deterministic current** | \(I_i^{\mathrm{eff}} = g_I(H_i)(drive_i+sched_i+syn_i) + noise\) | Yes | Via source | Bundles external + recurrent; null at \(g_I\equiv 1\) is clean |
| **B. Extrinsic drive only** | \(g_d(H_i)(drive_i+sched_i) + syn_i + noise\) | Yes | Via source | Preserves recurrent coupling calibration; HDP uses asymmetric `H_boost` on this path only |
| **C. Recurrent synaptic only** | \(drive_i+sched_i + g_s(H_i)\,syn_i + noise\) | Yes | Via source | Isolates \(H\) effect on network feedback; extrinsic stimuli unchanged |
| **D. Noise channel** | \(\cdots + g_n(H_i)\,noise\) | Stochastic only | Weak | Poor primary memory carrier |
| **E. Parameter `b`** | \(b_i^{\mathrm{eff}}=g_b(H_i)\,b_i\) | Yes (via \(du/dt\)) | Indirect | Alters subthreshold integration; spike threshold unchanged |
| **F. Parameter `a` / `d`** | \(a_i^{\mathrm{eff}}, d_i^{\mathrm{eff}}\) | Yes (reset kinetics) | Indirect | High risk of disguised cell-type change |
| **G. Spike threshold** | \(V_{\mathrm{th},i}^{\mathrm{eff}}(H_i)\) | Yes | Via spikes | Not currently a public per-neuron field (hardcoded 30 mV) |
| **H. `source_scale`** | \(Q \propto source\_scale\) | **No** | Yes only | Readout gain, not plant — **unsuitable** for \(H\rightarrow x\) on dynamics |

**Precedent (HDP, not Protocol H):** `H_boost_gain` applies
`boost = 1 + H_boost_gain * max(0, 1-H)` to `(drive + sched)` only — a
**one-sided** extrinsic boost, not a generic bidirectional gain with \(g(1)=1\) in
the multiplicative sense. Useful ancestry, not the H1b template.

**Recommended shortlist for H1c decision** (pending review):

1. **B** — extrinsic drive gain (closest to calibrated external input)
2. **C** — recurrent synaptic gain (closest to network memory hypothesis)
3. **A** — total deterministic current (simplest single-knob null)

Implementation must expose \(\kappa_{H\rightarrow x}\) with explicit null
(\(g\equiv 1\)) recoverable **independently** of \(\kappa_H\).

## 5. \(I^{\mathrm{rel}}\) semantics and baseline-one (H1a)

**Definition (implemented):**

\[
I_i^{\mathrm{rel}} = \frac{I_{\mathrm{syn},i}}{i_{\mathrm{ref}}},
\qquad
I_{\mathrm{syn},i} = \sum_{e:\,\mathrm{post}(e)=i} w_e\,s_e,
\]

with \(s_e\ge 0\) the nonnegative edge synaptic filter state and native weights
\(w_e\) signed.

**Zero-centering:** \(I^{\mathrm{rel}}\) is **not** zero-centered. For excitatory
recurrent activity with positive filter states, typically
\(I_{\mathrm{syn},i}\ge 0\) and \(\mathbb E[I_i^{\mathrm{rel}}]>0\) under ongoing
drive. There is no subtraction of a running baseline in the current kernel.

**Two concepts Protocol H must keep separate:**

| Concept | Meaning |
|---------|---------|
| **Coordinate reference** | RBS relative baseline \(H=1\) (e.g. \(z/z^\*\)); the restoring term satisfies \(R(1)=0\) |
| **Driven steady state** | \(H^\*\) solving \(R(H^\*) + \kappa_H\,\mathbb E[I^{\mathrm{rel}}]=0\) under ongoing input |

When \(\kappa_H>0\) and \(\mathbb E[I^{\mathrm{rel}}]\neq 0\), generally \(H^\*\neq 1\).
That is not a bug: **baseline-one is a reference calibration**, not a claim that
the driven attractor sits at \(H=1\). H3 receipts must record which baseline is used
for perturbations (\(H_k=1+\delta_H\) about coordinate 1, not necessarily about
\(H^\*\)).

With \(\kappa_H=0\) and no input coupling, \(H^\*=1\) for both F1 and F2 (where defined).

## 6. H1 phase-portrait expectations (\(I^{\mathrm{rel}}=0\))

Isolated relaxation (\(\kappa_H=0\), no synaptic input): compare F1 vs F2 at
\(H_0\in\{0.5,0.8,1,1.2,2\}\).

Restoring velocities \(R_1(H)=1-H\), \(R_2(H)=H^{-1}-1\):

| Regime | F1 | F2 |
|--------|----|----|
| \(H<1\) | positive \(\dot H\) | **stronger** positive: \(H^{-1}-1 > 1-H\) |
| \(H>1\) | negative \(\dot H\) | **weaker** magnitude: \(\|H^{-1}-1\| < \|1-H\|\) |
| Symmetry about 1 | \(R_1(1-\delta)=-R_1(1+\delta)\) | **asymmetric** |

Same local Jacobian at \(H=1\); nonlinear distinction appears away from equilibrium.
**Stronger recovery from depletion, slower decay from surplus** (F2 vs F1).

Executable receipt: `tests/test_protocol_h_rbd_h1_phase_portrait.py`.

## 7. H1c implementation gate (authorized: C — postsynaptic recurrent gain)

**Frozen coupling (H1c-C):**

\[
I_i^{\mathrm{drive}} = I_i^{\mathrm{ext}} + G_H(H_i;\beta_H)\,I_i^{\mathrm{rec}} + \text{noise},
\qquad
G_H(H;\beta_H)=1+\beta_H(H-1).
\]

| Rule | Requirement |
|------|-------------|
| External drive | Untouched: \(I^{\mathrm{ext,eff}}=I^{\mathrm{ext}}\) |
| \(F_H\) input | **Pre-gain** \(I^{\mathrm{rec}}\) only (not \(G_H I^{\mathrm{rec}}\)) |
| Admissibility | \(G_H>0\); nonpositive gain invalidates (no clip) |
| Null F0 | \(H\equiv 1 \Rightarrow G_H=1\) |
| Null \(\beta_H=0\) | H1a activity |
| Null \(I^{\mathrm{rec}}=0\) | \(H\) invisible to \(x\) via this map |
| Null \(W\) | Fixed throughout |

**Sign-symmetry test (pre-H2/H3):** \(H_k=1\pm\delta\) with \(\beta_H>0\) must
respectively increase/decrease recurrent susceptibility on neuron \(k\).

A and B (total deterministic / extrinsic drive gain) remain **later typed maps**,
not replacements.

Implementation: `beta_h` on `simulate_edge_recurrent_izhikevich_rbd`;
receipt: `tests/test_protocol_h_rbd_h1c.py`.

## 8. Stop rules

- Do not open H3 \(M(\Delta)\) without H1c \(H\rightarrow x\)
- Do not bake exponential (or any) gain into doctrine without inventory sign-off
- Do not conflate coordinate reference \(H=1\) with driven \(H^\*\)

## 9. References

- `docs/doctrine/protocol_h_rbd_memory.md` — parent protocol
- `jaxfne/emitters.py` — `simulate_edge_recurrent_izhikevich_rbd`, HDP `H_boost_gain`
- `tests/test_protocol_h_rbd_h1.py` — H1a unit tests
- `tests/test_protocol_h_rbd_h1_phase_portrait.py` — isolated F1/F2 trajectories
