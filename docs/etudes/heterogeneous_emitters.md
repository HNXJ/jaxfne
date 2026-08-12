# Étude: Heterogeneous emitters in one TFNE composition (frozen protocol)

**Protocol identity:** `heterogeneous_emitters_v0415b`  
**Supersedes:** `v0415` (n=10 HEI, `bound_mode=minimal`, \(U_{\rm hei}=0.5\)), which produced nonfinite \(Q\) (`hei_error=true`). Failed receipt: `artifacts/etudes/heterogeneous_emitters/failed_v0415/`.  
**Revision reason:** the composition claim requires a finite declared source, not a prettier spectrum. `homeostatic_ei` at n=10 under extra drive with unbounded \(x\) is a known numerical regime of that kernel; the canonical mature circuit is n=2 with `bound_mode=stable`.
**Package baseline:** `jaxfne/` unchanged since `d5cf9a6`; repo HEAD at etude start recorded in the bundle  
**Bundle:** `artifacts/etudes/heterogeneous_emitters/`  
**Package code:** not modified for this étude  
**Jaxley:** installed locally but **not** a primary emitter (deferred; see §6)

This document is the experimental protocol. Do not retune after seeing
outcomes. A scientific change requires a new protocol identity.

---

## 1. Question

\[
\boxed{\text{Can heterogeneous neural equations participate in the same TFNE composition?}}
\]

Not “which neuron model is best,” and not “are their sources physically
equivalent?”

For emitters \(E_a,E_b\):

\[
X_a=E_a(U),\qquad X_b=E_b(U),
\qquad
Q_a=S_a(X_a),\qquad Q_b=S_b(X_b),
\qquad
Y_{a,k}=O_k(Q_a),\qquad Y_{b,k}=O_k(Q_b).
\]

Closure:

\[
\boxed{
E\text{ may change while }S\rightarrow F\rightarrow P
\text{ remains composable}.
}
\]

Checkmarks mean **composable under declared semantics**, not numerical
equality. In particular

\[
Q_{\mathrm{Izh}}^{(r)}\neq Q_{\mathrm{HEI}}^{(r)}
\]

unless an explicit calibration map exists (none is applied).

---

## 2. Emitters (smallest mature distinct pair)

| id | \(F_X\) | declared source | \(N\) |
|----|---------|-----------------|-------|
| `izh` | Izhikevich quadratic IF + reset (`cortical_eig`) | `uncalibrated_izhikevich_native_current` | 10 |
| `hei` | continuous bounded E/I state \(x\) with \(G,H\) (`homeostatic_ei`, canonical cubic/hebbian/linear, `bound_mode=stable`) | `uncalibrated_homeostatic_ei_native_current` | 2 |

LIF/GLIF are unimplemented placeholders and are not used.
HDP-on Izhikevich is the same \(F_X\) plus \(F_H\); it is not a second family.

`hei` \(V_m\) is native \(x\), not millivolts. Spikes are threshold-crossing
indicators on \(x\), not Izhikevich resets.

---

## 3. Common input family \(U_j(t)\)

Shared **shape** (relative native drive added to each family's baseline):

| epoch | time (ms) | shape |
|-------|-----------|-------|
| baseline | 0–100 | 0 |
| step | 100–600 | 1 |
| gap | 600–700 | 0 |
| pulse | 700–750 | 1 |
| tail | 750–1000 | 0 |

Family-native scales (not a calibration):

- `izh`: \(U = 6.0 \times \mathrm{shape}\)
- `hei`: \(U = 0.2 \times \mathrm{shape}\)

Duration `1000` ms, \(dt=0.5\) ms, seed `11`, dtype `float32`, JIT off.

**F–I characterization** (separate short runs, 400 ms, last 300 ms scored):
constant extra drive. `izh` amplitudes `{0,2,4,8,12}`; `hei` amplitudes
`{0,0.05,0.1,0.2,0.3}`. Same shape (constant), different native scale.

`izh` receives \(U\) through `Model.simulate(..., paradigm=StimulusSchedule)`.
`hei` uses the supported kernel path `simulate_homeostatic_ei(..., drive_schedule=U)`.
`Model.simulate`'s live docstring is the Izhikevich vertical slice; the HEI
branch returns before paradigm resolution. Classification:
**ANALYSIS_GAP** (family-specific drive surface), not a 0.4.15 observation
defect. Do not patch in this étude. Closure may add an HEI caveat to
`docs/api/core.md` or reject unused `paradigm` on HEI; that is out of scope
here.

---

## 4. Source and observation

Each family's **declared** \(S\) is used. Do not rescale \(Q\) into a common
physical unit.

Post-hoc observation stack (same operators as the Multiscale Étude, applied
independently to each \(Q\)):

| id | operator |
|----|----------|
| `lfp_ref` | `project_laminar_sources`, \(n=16\), width \(0.10\) |
| `lfp_wide` | same, width \(0.25\) |
| `csd` | `csd_proxy` from `lfp_ref` (fused \(D_{zz}KQ\)) |
| `eeg_sup` | declared 3-row Gaussian leadfield at \(z=0.25\), widths \(0.18,0.20,0.22\); `leadfield_status=toy_or_declared_proxy` |

Spectra: `spectrolaminar_psd_jax`, burn-in 100 ms, \(f\in[1,150]\) Hz, 96 bins.

Negative control (report, do not force): similar mean spike rates can coexist
with different \(Q(t)\) and different \(P_Y(f)\). That is why TFNE retains
\(Q\) rather than collapsing neural activity to rate.

---

## 5. Gates (predeclared)

- **A — emitter heterogeneity:** two different \(F_X\) execute.
- **B — source closure:** each yields a declared \(Q\) with distinct
  `source_calibration_status`; no cross-family physical-equivalence claim.
- **C — observation closure:** the same \(O_k\) stack runs post-hoc on each \(Q\).
- **D — semantic preservation:** provenance names emitter family, source
  status, and observation receipts.
- **E — neurophysiological consequence:** at least one meaningful difference
  in dynamics survives into \(Q\) or \(Y\); a rate-similar / \(Q\)-different
  pattern is a success. A total null is acceptable if reported as such.

Optional AGSDR fit of a reduced emitter to a richer target is **omitted**
(would require a cross-family observable contract that is not 0.4.15 work).

---

## 6. Explicitly out of scope

- Jaxley HH as a third primary row (optional dependency, distinct stimulus
  API, distinct source_mode). Deferred to a later biophysical-bridge étude.
- Universal approximation claims.
- Package mutation to unify `Model.simulate` stimulus injection for `hei`.

---

## 7. Reproduce

```bash
PYTHONPATH=. python3 scripts/run_heterogeneous_emitters_etude.py
```
