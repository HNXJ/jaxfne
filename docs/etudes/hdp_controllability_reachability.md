# Etude: HDP controllability, reachability, and TFNE phenotype

**Status:** consolidated computational evidence (relative/proxy readouts)  
**Bundle:** `artifacts/etudes/hdp_controllability_reachability/`  
**Frozen protocol SHA256:** see `manifest.json`  
**Branch / SHA at consolidation:** see `manifest.json`

---

## 1. Question

How do **latent-state dimensionality**, **actuator controllability**, **restorative alignment**, **local stability**, and **finite-amplitude reachability** determine adaptation of a minimal excitatory/inhibitory (E/PV) neural circuit—and how does successful adaptation appear across spikes, membrane potentials, sources, laminar field proxies, and spectra?

The circuit is the MCC-3 minimal TFNE motif: 5 E + 5 PV Izhikevich neurons with heterogeneous drive, synaptic weights \(\Theta\), and optional population H-state adaptation. Observables are measured through the canonical package chain

\[
(X,H,\Theta)\;\Rightarrow\; Q(t)\;\Rightarrow\;\Phi(z,t)\;\Rightarrow\;\text{readouts},
\]

with explicit proxy readout status (`physical_amplitude_calibrated = False`).

---

## 2. Mathematical formulation

**Plant and adaptation grammar**

\[
\dot X = F_X(X,H,\Theta,U),\qquad
\dot H = F_H(H,X,\Theta,U),\qquad
\dot\Theta = F_\Theta(H,X,\Theta).
\]

**Local observable sensitivity** (operating-point linearization)

\[
\dot r \approx J_\Theta\, B\, H.
\]

**Four conditions** (necessary / design targets for restorative 2D E/I regulation):

| condition | criterion | role |
|-----------|-----------|------|
| controllability | \(\mathrm{rank}(J_\Theta B)\ge d_e\) | independent actuator directions |
| restorative direction | \(\dot{\mathcal E}=e^\top J_\Theta B H<0\) | error decreases along actuation |
| local stability | \(\max_i \Re\lambda_i(J_{\mathrm{slow}})<0\) | bounded slow error dynamics |
| finite-amplitude authority | \(r^\*\in\mathcal R(U,\mathcal C)\) | setpoint reachable under \(U\) and parameter box \(\mathcal C\) |

**Frozen vector-H controller** (derived analytically from measured \(J_S\), not trained):

\[
\tau_H \dot H = -e - \Lambda H,\qquad
\tau_\Theta \dot\Theta_S = B H,\qquad
H=(h_E,h_I)^\top,\qquad
\Theta_S=(m_{EI},\eta_{a_E}).
\]

Design (from `hdp_mvc_frozen_controller_spec.json`):

- \(\zeta = 1.2\)
- \(\tau_{\mathrm{slow}} \approx 2\,\mathrm{s}\)
- \(G^\* \approx 0.347\, I\)
- \(\tau_H = 0.2\,\mathrm{s}\), \(\tau_\Theta = 2.0\,\mathrm{s}\)

**Slow-pole verification:** \(\max\Re\lambda(J_{\mathrm{slow}}) = -0.5\,\mathrm{s}^{-1}\) (stable).

---

## 3. Scalar-H falsification

At the scientific operating point (`scientific_agsdr_theta_star`), the synaptic-only plant Jacobian has **rank deficiency**:

\[
\mathrm{rank}(J_W)=1
\]

so \(\mathrm{rank}(J_W B)\le 1\) for all actuator maps \(B\). Scalar per-neuron H-state adds at most one additional adaptive direction, insufficient for independent 2D E/I regulation.

**Receipt:** `artifacts/msvc_hdp_diagnostic/hdp_mvc_alignment_diagnostic.json`  
**MVC #2 scalar-H control:** \(R_{EI}^{\mathrm{scalar}}=0.434\), terminal weighted error \(0.525\) — marginal improvement over HDP-off.

---

## 4. Minimal vector-H controller derivation

Actuator inventory expands \(W\to\Theta\) and measures intrinsic biophysical sensitivities. The minimal full-rank pair selected for implementation:

\[
\Theta_S=(m_{EI},\eta_{a_E}),
\qquad
J_S=
\begin{bmatrix}
0 & 9.018\\
7.016 & 8.617
\end{bmatrix},
\qquad
\mathrm{rank}(J_S)=2.
\]

**Receipts:**

- `hdp_mvc_j_theta_actuator_inventory.json` — \(\mathrm{rank}(J_\Theta)=2\) where \(\mathrm{rank}(J_W)=1\)
- `hdp_mvc_intrinsic_actuator_inventory.json` — \((m_{EI},a_E)\) pair
- `hdp_mvc_frozen_controller_spec.json` — frozen \(B\), \(\Lambda\), pole placement

Structural gate: \(\mathrm{rank}(J_\Theta B)=2\) (Level A). Local alignment \(\dot{\mathcal E}<0\) at perturbation onset (Level B, prior frozen validation receipt).

---

## 5. Reachability and prospective MVCs

**Authority map** (`hdp_mvc_authority_boundary.json`): independently measured \(\mathcal R(U,\mathcal C)\) over \((m_{EI},\eta_{a_E})\) at fixed heterogeneous drive.

| MVC | \(\alpha_U\) | \(r_0\in\mathcal R(U,\mathcal C)\)? | outcome |
|-----|-------------|--------------------------------------|---------|
| #1 | 1.5 | **no** (\(d_R=1.12\)) | actuator saturation; partial E recovery, I remains elevated |
| #2 | 1.2 | **yes** (\(\alpha_U^{\mathrm{crit}}\approx 1.2\)) | prospective validation |

**MVC #1 (falsification of finite reachability):**

\[
\boxed{\text{local controllability}\not\Rightarrow\text{finite reachability}}
\]

Vector-H at \(U_1=1.5U_0\): \(R_E=0.988\), \(R_I=0.561\), \(R_{EI}=0.570\) — E rate recovers while I does not; target outside authority set.

**MVC #2 (prospective validation at \(\alpha_U=1.20\)):**

| condition | \(R_{EI}\) | terminal \(\|D(r-r_0)\|\) |
|-----------|-----------|---------------------------|
| HDP-off | 0.381 | 0.515 |
| scalar-H | 0.434 | 0.525 |
| **vector-H** | **0.960** | **0.030** |

**Receipts:** `hdp_mvc_validation_v2.json`, `hdp_mvc_posthoc_scalar_vector.json`, `hdp_mvc_frozen_validation.json`

Protocol frozen before execution (`hdp_mvc_protocol_v2.json`); setpoint \(r_0\) measured pre-perturbation, not design-target AGS DR.

---

## 6. Neurophysiological phenotype

Same frozen MVC #2 trajectories; epochs (ms): baseline \([20,3000)\), early \([3000,5000)\), late \([13000,15000)\).

**Pipeline:** spikes \(\to V_m\) (native Izhikevich coordinate) \(\to Q(t)\) (source proxy) \(\to \Phi(z,t)\) (laminar field proxy) \(\to\) Welch PSD.

| quantity | baseline | late off | late scalar | late vector |
|----------|----------|----------|-------------|-------------|
| \(r_E\) (Hz) | 10.90 | 13.72 | 13.68 | **10.90** |
| \(r_I\) (Hz) | 9.14 | 13.93 | 14.05 | **8.85** |
| source RMS (rel.) | 9.26 | 10.93 | 10.95 | 10.51 |
| \(E_P\) (log PSD distance) | 0 | 10.13 | 9.70 | **2.10** |
| \(R_P\) | — | −0.04 | 0.05 | **0.77** |
| phenotype class | — | poor rate | poor rate | **rate OK, altered phenotype** |

**Central coupling test:**

\[
r_{\mathrm{late}}\approx r_0 \quad\stackrel{?}{\Longrightarrow}\quad P_{\mathrm{late}}(f)\approx P_0(f).
\]

**Result:** For vector-H, population rates recover (\(R_{EI}=0.96\)) and source PSD moves substantially toward baseline (\(R_P=0.77\), dominant peak returns to \(\sim 19.8\,\mathrm{Hz}\)). However, weighted physiological distance remains elevated. **Controlled E/I rates recover; the broader TFNE state settles on an alternative compensated trajectory**—scientifically distinct from both failed off/scalar conditions and from perfect spectral identity.

**Phenotype classification (`rate_recovery_altered_phenotype`):** Assigned when terminal weighted rate error \(\|D(r_{\mathrm{late}}-r_0)\|_2 < 0.15\) (rates near setpoint) but weighted phenotype distance \(d_{\mathrm{phys}} \ge 0.15\). The phenotype vector is

\[
y = (r_E, r_I, \mathrm{SD}(V_m), \mathrm{RMS}(Q), \mathrm{RMS}(\Phi_e), P_{1\text{--}30}, P_{30\text{--}80}, P_{80\text{--}150}),
\]

with weights \((1/15, 1/10, 0.05, 1, 1, 1, 1, 1)\) on relative/proxy units. For vector-H late epoch: \(d_{\mathrm{phys}}=1.88\) driven primarily by elevated source/field RMS (10.51 vs baseline 9.23) and residual band-power offsets despite near-baseline rates. Spectral log-distance \(E_P=2.10\) (vs off 10.13) confirms partial—not full—spectral rewind.

\[
\boxed{\text{observable restoration}\neq\text{state restoration}.}
\]

Package-native operators used: `Signals` field proxy (`phi_e_proxy`, `csd_proxy`), `spectrolaminar_psd`, `kappa_synchrony`. No notebook-local engines.

**Figure:** `artifacts/etudes/hdp_controllability_reachability/figure.png` (panels A–L; panel E = OFF | scalar | vector rasters; MVC #1 inset in D).

---

## 7. Conclusions and scope

> In this minimal E/I TFNE circuit, scalar synaptic H-state adaptation is insufficient for independent two-dimensional E/I regulation. Expanding both latent-state dimension and the adaptive actuator basis yields a full-rank restorative controller. When the desired state lies within the finite-amplitude authority set, the prospectively frozen vector-H controller restores E/I activity substantially better than HDP-off and scalar-H controls.

The neurophysiology extends this claim: vector-H achieves near-complete rate recovery with substantial but incomplete spectral/field recovery—rate restoration does not imply identical oscillatory or laminar-proxy phenotype.

**Scope limits (non-negotiable):**

- Relative/proxy readouts with `physical_amplitude_calibrated = False`; empirical LFP/CSD calibration out of scope
- Single seed (17), single minimal circuit topology
- Controller derived from local Jacobian at one operating point; no anti-windup
- Diagnostic receipts used at consolidation (local provenance; not committed): hashes in `manifest.json`
- This Etude bundle is the compact publication-facing consolidation

**Reproduce committed bundle:**

```bash
python scripts/consolidate_hdp_controllability_etude.py
```

The committed `metrics.json`, `figure.png`, and `manifest.json` are the durable evidence surface. Manifest SHA256 entries record frozen protocol/controller receipts used at consolidation time.
