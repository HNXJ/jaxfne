# Protocol W — HDP parameter memory (W0 frozen; implementation closed)

**Status:** **W0 FROZEN** · **W1a IMPLEMENTED** · **W1b IMPLEMENTED** · W2+ **not** authorized  
**Prerequisite:** Protocol H **closed** at H4 (`docs/doctrine/protocol_h_rbd_memory.md`)  
**W0 receipt:** `artifacts/protocol_w/w0_mathematical_contract.json`  
**Out of scope:** H4 rescue, Protocol H extensions, W implementation, conservation/competition in W1

Protocol W is a **new dynamical problem** — not an extension of the H4 topology/delay experiment.

## 1. Scientific question

\[
\boxed{
\text{Can transient RBS history be written into a slower parameter state while preserving bounded, physically sane dynamics?}
}
\]

Protocol W must **not** be introduced to “fix” H4.

## 2. Causal hierarchy

| Path | Grammar | Protocol |
|------|---------|----------|
| RBD | \(u \rightarrow H \rightarrow X\) | H (closed) |
| HDP | \(u \rightarrow H \rightarrow W \rightarrow X\) | W |

**Conceptual unification:** HDP is a slower **RBD realization** on parameter coordinates. Hidden biophysical history is read out through typed plasticity maps:

\[
H \xrightarrow{F_W} \omega \xrightarrow{\exp} W.
\]

STDP, BCM, neuromodulated learning, and scaling are future alternative choices for the drive \(D\), not separate architectural species.

## 3. Inputs from Protocol H

| Proposition | Status |
|-------------|--------|
| **P1** RBS retains perturbation history | Supported (H1–H3) |
| **P2** RBS history influences activity | Supported (H1c, H3) |
| **P3** Geometry/delays extend activity memory (preregistered) | Not supported (H4) |

W does not reopen H4 design choices.

## 4. W0 design checkpoint — lesson from toy experiments

The naive rule \(\dot W_{ij}\propto H_i-H_j\) has a desirable directional interpretation but **insufficient stability structure**. Under asynchronous activity it can:

- manufacture total coupling,
- cross zero,
- couple increasing \(W\) to increasing transmission cost.

W0 therefore **separates three previously conflated ingredients**:

\[
\boxed{
\text{plastic drive}
\;+\;
\text{admissibility}
\;+\;
\text{forgetting/stabilization}
\;[\;+\;\text{optional conservation/competition}\;].
}
\]

Each ingredient is **independently switchable** so falsification can identify what is actually necessary.

## 5. Minimal admissible HDP grammar (W0 — **frozen**)

### 5.1 Generic edge equation

\[
\boxed{
\tau_W\dot W_{ij}
=
D_{ij}(H_i,H_j,X_i,X_j)
-
R_{ij}(W_{ij})
+
C_{ij}(W)
}
\]

| Term | Role | W1 status |
|------|------|-----------|
| \(D_{ij}\) | Hidden-state-dependent **plastic drive** | \(\kappa_W(H_{\mathrm{pre}}-H_{\mathrm{post}})\) |
| \(R_{ij}\) | Intrinsic restoring / regularization | mapped to \(R_\omega=\lambda_W\omega\) |
| \(C_{ij}\) | Optional conservation / competition | **absent** |

**Conservation is a separate hypothesis.** W1 does **not** impose \(\sum W=\mathrm{const}\). Exponential admissibility plus linear \(\omega\)-restoration already address positivity and zero-drive drift; adding conservation simultaneously would confound which constraint produced stability.

### 5.2 Structural positivity via log-parameterization (admissibility)

Do **not** rely on clipping, projection, or \(1/W\) singularities. Freeze:

\[
\boxed{
W_{ij} = W_{ij}^{0}\,e^{\omega_{ij}},
\qquad
\omega_{ij}\in\mathbb R.
}
\]

HDP evolves \(\omega\):

\[
\boxed{
\tau_W\dot\omega_{ij}
=
D_{ij} - R_\omega(\omega_{ij}) + C_\omega.
}
\]

Immediately \(W_{ij}>0\) without discontinuous projection. Relative interpretation:

\[
\boxed{
\omega_{ij} = \log\frac{W_{ij}}{W_{ij}^{0}}.
}
\]

- \(\omega=0\) — baseline coupling  
- \(\omega>0\) — potentiation  
- \(\omega<0\) — depression  

This is the preferred W0 admissibility architecture (replaces any earlier implicit “clip \(W\)” boundedness sketch).

### 5.3 Excitatory / inhibitory sign (typed, not via \(W\) sign)

\(W_{ij}>0\) is appropriate for **magnitude / gain**. Synaptic **sign** is a separate typed property:

\[
s_{ij}\in\{-1,+1\},
\qquad
\text{effective coupling} = s_{ij}\,W_{ij}.
\]

Inhibitory versus excitatory plasticity must not be encoded by the sign of \(W\) under \(W=W_0 e^\omega\). This separation is mandatory for eventual E/I HDP.

### 5.4 W1 restoring law (forgetting / stabilization)

Do **not** use the nonlinear \((1/W - W)\) rule in W1. Freeze linear restoration in \(\omega\):

\[
\boxed{R_\omega = \lambda_W\,\omega.}
\]

Thus the W1 edge law is:

\[
\boxed{
\tau_W\dot\omega_{ij}
=
\kappa_W\,(H_{\mathrm{pre},ij}-H_{\mathrm{post},ij})
-
\lambda_W\,\omega_{ij}.
}
\]

With RBS baseline \(H=1\), the drive is the **explicit normalized difference** \(\kappa_W(H_i-H_j)\). There is no legacy `/100` scaling from toy experiments.

### 5.5 Known memory timescale

If the \(H\)-gradient drive vanishes (\(H_i-H_j=0\)):

\[
\omega(t) = \omega(t_0)\,e^{-\lambda_W(t-t_0)/\tau_W}.
\]

Parameter memory timescale:

\[
\boxed{\tau_{\mathrm{mem},W} = \frac{\tau_W}{\lambda_W}.}
\]

### 5.6 Relation to RBD (timescale separation)

\[
\boxed{\tau_H\dot H = F_H(H,\ldots)}
\qquad\text{then}\qquad
\boxed{
\tau_W\dot\omega = \kappa_W\,\Delta H - \lambda_W\,\omega,
\qquad
W = W_0 e^\omega.
}
\]

If \(\tau_W \gg \tau_H\), a transient \(H\) perturbation can decay while \(\omega\neq 0\) — the parameter-memory phenomenon W is designed to test.

### 5.7 W1a analytic ground truth (prescribed \(\Delta H\))

**Preferred W1a topology:** a **single plastic directed edge** with **prescribed**

\[
H_{\mathrm{pre}}(t)-H_{\mathrm{post}}(t)=\Delta H(t),
\]

not a full network. Verify \(\omega(t)\) integrates prescribed history before closing the loop to emitters.

Rectangular RBS-gradient pulse (\(\omega(0)=0\)):

\[
\Delta H(t)=
\begin{cases}
\delta, & 0<t<T_p \\
0, & t\ge T_p
\end{cases}
\]

During pulse:

\[
\omega(t)=\frac{\kappa_W\delta}{\lambda_W}
\left(1-e^{-\lambda_W t/\tau_W}\right).
\]

At offset:

\[
\omega(T_p)=\frac{\kappa_W\delta}{\lambda_W}
\left(1-e^{-\lambda_W T_p/\tau_W}\right).
\]

Afterward:

\[
\boxed{
\omega(t)=\omega(T_p)\,e^{-\lambda_W(t-T_p)/\tau_W}.
}
\]

This is the W1 analytic receipt — analogous to F1 transparency in Protocol H.

**W1a implementation:** `jaxfne/w1a_omega_plasticity.py`, `tests/test_protocol_w_w1a.py`.

**Preregistered W1a gates (all tested):** zero-drive null; \(\lambda_W=0\) no-forgetting
limit (supported scientific mode); sign symmetry and \(W_+W_-=W_0^2\); structural
positivity without clipping; reference state; \(\tau_{\mathrm{mem},W}\) \(1/e\) decay;
pulse-duration linear/saturation; discrete-Euler bit-exact recurrence; continuous
limit convergence. Euler stability: \(0<\lambda_W\Delta t/\tau_W<2\); monotonic
regime \(\le 1\) recorded but not required for rejection.

Causal chain for W1a:

\[
\Delta H(t) \rightarrow \omega(t) \rightarrow W(t).
\]

**Do not** connect \(W\rightarrow X\) until **W2** (W1b remains shadow-only).

### 5.8 W1b — RBD-generated shadow plasticity (**implemented**)

**Question:** can simulated RBD history write the same well-characterized \(\omega\) state?

Shadow contract: simulation at fixed \(W_0\); passive
\(\tau_W\dot\omega_{AB}=\kappa_W(H_A-H_B)-\lambda_W\omega_{AB}\) (and BA with reversed
gradient). **No** \(\omega\rightarrow W\rightarrow X\).

**Update ordering:** \(\omega_{n+1}=F(\omega_n,H_n)\) using pre-step \(H_n\).

**Emergent symmetry:** \(\omega_{AB}=-\omega_{BA}\); \(W_{AB}W_{BA}=W^0_{AB}W^0_{BA}\).

**Principal phenomenon:** timescale-separated parameter retention when
\(\tau_{\mathrm{mem},W}\gg\tau_H\).

Implementation: `jaxfne/w1b_shadow_plasticity.py`, `tests/test_protocol_w_w1b.py`.

## 6. Experimental ladder (frozen order)

\[
\boxed{
\begin{aligned}
\text{W1a} &: H \rightarrow \omega && \text{parameter writing (prescribed }\Delta H\text{, analytic)} \\
\text{W1b} &: H \rightarrow \omega && \text{parameter writing (minimal 2-node or single edge, simulated }H\text{)} \\
\text{W2}  &: \omega \rightarrow W \rightarrow X && \text{parameter expression} \\
\text{W3}  &: H \rightarrow W \rightarrow X && \text{closed HDP loop} \\
\text{W4}  &: \text{parameter-memory quantification / nulls} && \text{prospective assay}
\end{aligned}
}
\]

| Phase | Status |
|-------|--------|
| **W1a** | **IMPLEMENTED** — prescribed \(\Delta H\to\omega\), analytic + Euler receipts |
| **W1b** | **IMPLEMENTED** — RBD shadow \(\omega\) on \(A\rightleftarrows B\); no \(W\to X\) |
| **W2** | not authorized |
| **W3** | not authorized |
| **W4** | not authorized |

\(\text{W2}_{\mathrm{competition}}\): competition/conservation — **only if necessary** after W1.

## 7. Implementation authorization gates

| Phase | Authorized |
|-------|------------|
| W0 | frozen |
| W1a | **yes** — scalar integrator only |
| W1b | **yes** — shadow plasticity, no feedback |
| W2+ | **no** |

W1b+ remains closed until W1a review.

### 7.1 Stability receipts (pre-code)

| Receipt | Criterion |
|---------|-----------|
| **Positivity** | \(W_{ij}>0\) for all finite \(\omega_{ij}\) |
| **Zero-drive decay** | \(\Delta H=0 \Rightarrow \omega\to 0\) (or fixed point \(\omega^\*\)) |
| **Finite perturbation** | finite \(\Delta H\) pulse \(\Rightarrow\) finite \(\omega(T_p)\) |
| **No manufactured coupling** | \(\omega=0 \Rightarrow W=W_0\) exactly |
| **RBD recovery** | \(\kappa_W=0, \omega\equiv 0 \Rightarrow\) RBD kernel |

### 7.2 Forbidden shortcuts

- No tuning \(\kappa_W,\lambda_W,\tau_W\) after viewing W4 memory metrics
- No H4-topology experiments in W1–W3
- No conflation of RBD fading memory with \(W\) retention without separate nulls
- No conservation and restoration enabled simultaneously in W1

## 8. Relation to existing code

| Artifact | Role |
|----------|------|
| `jaxfne/w1a_omega_plasticity.py` | W1a scalar \(\omega\) integrator + analytic references |
| `jaxfne/w1b_shadow_plasticity.py` | W1b shadow \(\omega\) from RBD-recorded \(\Delta H\) |
| `tests/test_protocol_w_w1b.py` | W1b composition, symmetry, F1, nulls, timescale receipts |
| `simulate_edge_recurrent_izhikevich_rbd` | RBD substrate (\(\dot W=0\)) |
| `simulate_edge_recurrent_izhikevich_hdp` | Legacy reference — **not** canonical W; lacks W0 \(\omega\) grammar |
| `docs/doctrine/rbs_rbd_hdp.md` | RBS/RBD/HDP authority |

## 9. Evidence and receipts

W0 exports:

- `artifacts/protocol_w/w0_mathematical_contract.json`

Future W runs (when authorized) add prospective receipts per phase.

## 10. Stop rules

- Implementation before W0 sign-off — **blocked** (W0 now frozen; W1 still blocked)
- Using long heterogeneous rings in W1
- Encoding E/I sign in \(\mathrm{sign}(W)\) under log-parameterization
- Enabling \(C_{ij}\) in W1 without separate W2\(_{\mathrm{competition}}\) protocol ID

## 11. References

- `docs/doctrine/protocol_h_rbd_memory.md` — Protocol H (closed)
- `artifacts/protocol_h_rbd/h4_matrix/h4_interpretation_receipt.json`
- `docs/doctrine/rbs_rbd_hdp.md`
- `docs/doctrine/tfne_containment_architecture.md` — typed coupling maps
