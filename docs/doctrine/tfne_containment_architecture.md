# TFNE Containment and Composition

**Status:** FROZEN (rule; implementation follows incrementally)  
**Baseline:** `dev` @ `6003958` (H1a/H1b)  
**Authority:** complements `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md`

## 1. Central claim

\[
\boxed{\text{TFNE is a containment and composition model for neural biophysics.}}
\]

TFNE does **not** compete with Hodgkin–Huxley, Izhikevich, Jaxley, STDP, BCM,
Hebbian learning, electromagnetic forward models, metabolic models, or statistical
dynamics. Those are **realizations inside a common mathematical grammar**.

> **TFNE does not prescribe a neural model; it provides a common relative state,
> operator, and geometry grammar in which neural models of different physical
> resolution can be composed.**

## 2. Three separable concepts

| Pillar | Symbol | Role |
|--------|--------|------|
| **State** | \(\mathcal X=(X,H,\Theta,\mathcal B,\mathcal G,\ldots)\) | Complete Markov state; \(H\) is the unified **relative biophysical dependency state** |
| **Operators** | \(E,S,F,P,O,A\) | Typed roles: how physics evolves and transforms |
| **Geometry** | \(\mathcal G\) | Where quantities live and how they couple spatially |

Tensor Fields bridge state and geometry:

\[
(H,X,\Theta)\otimes\mathcal G \rightarrow \text{structured neural physics}.
\]

The canonical operator chain remains:

\[
E_\theta \rightarrow S_\psi \rightarrow F_\gamma \rightarrow P_\eta
\qquad\text{(plus objective/optimizer/evidence)}.
\]

Individual emitters, source maps, fields, and probes are **replaceable
implementations of typed roles**, not one universal physical equation.

## 3. RBS — unified dependency state (not one equation)

\[
\boxed{\mathbf H(t)\in\mathbb R^{d_H}}
\]

is a **finite-dimensional state-space container**, not a single homeostatic
equation.

> **RBS is a finite-dimensional relative biophysical state whose coordinates may
> represent explicit physical quantities or reduced sufficient states, and whose
> influence on TFNE operators is declared through typed gain/coupling maps.**

Coordinates may represent **different physics** while sharing one interface:

\[
\mathbf H=
\begin{bmatrix}
H_{\mathrm{Na}}\\ H_{\mathrm{K}}\\ H_{\mathrm{Ca}}\\ H_{\mathrm{ATP}}\\
H_{\mathrm{vesicle}}\\ H_{\mathrm{GABA}}\\ H_{\mathrm{DA}}\\ H_{\mathrm{5HT}}\\
H_{\mathrm{NE}}\\ H_{\mathrm{ACh}}\\ H_{\mathrm{STDP,+}}\\ H_{\mathrm{STDP,-}}\\
\vdots
\end{bmatrix}.
\]

**Unification is not semantic equivalence:**

\[
\boxed{
H_{\mathrm{Na}},\,H_{\mathrm{DA}},\,H_{\mathrm{STDP}}
\in
\text{the same finite-dimensional state-space grammar}.
}
\]

not \(H_{\mathrm{Na}}=H_{\mathrm{DA}}=H_{\mathrm{STDP}}\).

Each coordinate admits a typed physical interpretation (or explicit reduced
interpretation \(\mathcal R_k(\mathbf z)\)). Coordinates are not required to be
mutually identifiable or calibrated to the same units.

## 4. RBD — state-space dynamics container

\[
\boxed{
\dot{\mathbf H}=F_H(\mathbf H,X,\Theta,U,\mathcal G,\ldots)
}
\]

is the general **Relative Biophysical Dynamics** container. Examples as
**realizations**, not definitions:

| Realization | Sketch |
|-------------|--------|
| Ionic / metabolic | \(\dot H_{\mathrm{ion}} = F_{\mathrm{ion}}(\ldots)\) |
| STDP traces | \(\tau_+\dot H_+ = -H_+ + S_{\mathrm{pre}}\), \(\tau_-\dot H_- = -H_- + S_{\mathrm{post}}\) |
| BCM threshold | \(H_\theta\) sliding activity threshold in \(\dot W = F_W(x,H_\theta,W)\) |
| Neuromodulated plasticity | \(\dot W = F_W(H_+,H_-,H_{\mathrm{DA}},W)\) |
| Scalar toy (Protocol H F1/F2) | \(\tau_H\dot H = R(H) + \kappa I^{\mathrm{rel}}\) |

Instead of parallel subsystems (STDP, BCM, homeostasis, neuromodulation,
adaptation), the grammar is:

\[
\boxed{
\text{H-state dynamics} + \text{parameter/state readout}.
}
\]

**HDP** is the readout class in which persistent parameters evolve from \(H\):

\[
\dot W = F_W(\mathbf H, X, W, \ldots).
\]

## 5. Typed gain/coupling maps (essential)

One mathematical container \(\mathbf H\in\mathbb R^{d_H}\) does **not** mean one
scalar controls everything. Influence on operators is **declared per coupling**:

\[
p^{\mathrm{eff}} = g_p(\mathbf H;\kappa)\, p,
\qquad
g|_{\kappa=0}\equiv 1\ \text{where required},
\]

with **typing** so that:

- emitter-local \(H_{\mathrm{ion}}\) may map to channel parameters \(\theta(H)\)
- \(H_{\mathrm{transmitter}}\) may map to release/receptor/synaptic dynamics
- \(H_{\mathrm{trace}}\) may map to \(W\) via \(F_W\)
- \(H_{\mathrm{extracellular}}\) may map to field operator \(\gamma(H)\) **only
  through an explicit declared coupling**

**Strict typing rule:** an emitter-local coordinate must **not** modulate a field
operator without an explicit, documented coupling map. Prevents accidental
cross-operator leakage.

### 5.1 Operator-parameterized TFNE (target architecture)

RBS may parameterize **any physically appropriate TFNE operator**, not only the
emitter:

\[
\boxed{
\mathcal T_H:
(E,S,F,P,O,A)_t
\longrightarrow
(E,S,F,P,O,A)_{t+1}.
}
\]

Examples (declared couplings only):

| Coupling | Sketch |
|----------|--------|
| \(H_{\mathrm{ion}}\rightarrow\gamma(H)\) | Extracellular ionic state alters field conductivity operator |
| \(H_{\mathrm{ATP}}\rightarrow\theta(H)\) | Metabolic state modulates emitter parameters |
| \(H\rightarrow E,S,F,P\) | Full operator chain may depend on selected \(H\) coordinates |

Current implementation (H1a/H1b) centers the **first** coupling on the emitter
(\(H\leftrightarrow x\)). That is a minimal slice of this architecture, not its
full scope.

## 6. Parameter sharing template

Local deviations from a shared biophysical template:

\[
\Theta_i = \Theta_{\mathrm{type}(i)} \odot G(\mathbf H_i),
\]

giving approximately:

\[
\boxed{
\text{shared biophysical template}
+
\text{relative local state}
+
\text{geometry}.
}
\]

Computationally: thousands of neurons may share a canonical parameter set while
RBS produces local deviations — without storing a fully independent detailed
model per element unless the user chooses that resolution.

## 7. Fidelity as explicit user choice

TFNE does not decide how much biology a user can afford:

\[
\boxed{
\text{fidelity}
\leftrightarrow
d_H + \text{emitter complexity} + \text{field complexity} + \text{geometry resolution}.
}
\]

Progressive resolution ladder (illustrative):

| Tier | Sketch |
|------|--------|
| Minimal | \(d_H=0\), Izhikevich, fixed \(W\), proxy field |
| Effective adaptation | \(d_H=1\) scalar RBS |
| Selected biophysics | \(d_H\sim 10\) ions/resources/modulators |
| Rich reduction | \(d_H\sim 100+\) |
| Detailed compartments | Jaxley-class emitters inside same TFNE grammar |

## 8. Relation to Protocol H

| Layer | Content |
|-------|---------|
| **Architecture (this doc)** | \(H\) as container; typed maps to any operator |
| **Protocol H1a** | First \(x/I\rightarrow H\) realization on emitter |
| **Protocol H1b** | First \(H\rightarrow x\) gain spec (emitter-local) |
| **Protocol H1c** | Implement selected emitter gain (not authorized) |
| **Future** | Field/source couplings \(H\rightarrow S,F\) with explicit typing |

Do not open H3 \(M(\Delta)\) until emitter-local \(H\rightarrow x\) exists (H1c).
Broader operator couplings are **architectural target**, not H1 scope.

## 9. References

- `docs/doctrine/rbs_rbd_hdp.md` — RBS/RBD/HDP index
- `docs/doctrine/protocol_h_rbd_memory.md` — Protocol H
- `docs/doctrine/protocol_h_h1b_h_to_x_gain.md` — emitter gain inventory
- `artifacts/project_sources/4_tfne_theory_and_neural_tensor.md` — upstream math
