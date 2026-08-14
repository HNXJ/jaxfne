# Étude: Experiment A — canonical multiscale observation (0.4.17-B)

**Status:** prospectively frozen at B0 before the decisive dataset  
**Protocol spec:** `artifacts/etudes/experiment_a/b0_protocol_spec.json`  
**Bundle:** `artifacts/etudes/experiment_a/`  
**Parent audit:** `artifacts/audit/v0417_a_capability_audit.json` (rank-1 delta)

This étude establishes one reproducible source-of-truth trajectory for the TFNE
observation chain. It is **not** a showcase of independently generated modality
examples.

## Causal architecture (frozen)

\[
(X,H)\xrightarrow{S}Q\xrightarrow{F_{\mathcal G,\mathcal M}}\Phi\xrightarrow{P}Y
\]

- **First-class frozen artifacts:** \(X(t)\), \(H(t)\), \(Q(t)\), geometry,
  metadata.
- **One neural simulate per seed.** All readouts are post-hoc on frozen arrays.
- **Source \(Q\)** is inspectable separately from plotting helpers.

## Semantic classification (every output)

| Class | Meaning in Experiment A |
|-------|-------------------------|
| `native` | Direct simulate output (e.g. \(V_m\), spikes, \(Q\)) |
| `relative_proxy` | Laminar LFP/CSD and declared linear laminar maps |
| `analysis_only` | EEG/MEG/EMM toy leadfields — operator demos only |
| `calibrated_physical` | **Excluded** — no calibrated EEG/MEG claims |

EEG/MEG remain **analysis_only** even though Experiment A includes them for
operator comparison. They must not be promoted to manuscript physical claims.

## Checkpoints

| ID | Deliverable |
|----|-------------|
| B0 | Frozen protocol spec (this document + JSON) |
| B1 | Canonical `canonical_source.npz` per seed: \(X,H,Q\) + geometry |
| B2 | Independent probe operators at fixed \((Q,\mathcal G,\mathcal M)\) |
| B3 | Immutable receipt bundle (manifest, metrics, hashes; no manuscript figures) |

## Frozen neural system

Identical to [multiscale observation](multiscale_observation.md) group A:
seed `7`, \(N=40\), Izhikevich `cortical_eig`, HDP/AGSDR off, 2000 ms,
\(dt=0.5\) ms, float32, eager simulate.

When HDP is off, \(H(t)\equiv 1\) documents the identity RBS container.

## Invariants (predeclared)

1. \(Q=0 \Rightarrow Y=0\) for linear zero-offset operators.
2. \(F(aQ_1+bQ_2)=aF(Q_1)+bF(Q_2)\) for declared linear \(F\).
3. Probe passes do not mutate \(Q\) or neural state.
4. Metrics and visualization inputs share the same frozen arrays.
5. B2: \(Q^{(1)}=Q^{(2)}\), \(P_1\neq P_2 \Rightarrow Y_1\neq Y_2\) where expected.

## Reproduce

```bash
python scripts/run_experiment_a.py
```

## Relation to multiscale_observation_v0415

Experiment A **supersedes** the 0.4.15 étude packaging for publication
development: it adds explicit \(X,H,Q\) freezing, factorized \(F\) then \(P\)
demonstration, and a B3 receipt. The underlying operator stack is unchanged;
semantic status of EEG/MEG is unchanged (`analysis_only`).
