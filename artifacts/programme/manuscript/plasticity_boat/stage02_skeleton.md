# Stage 2: complete skeleton (draft 1, 2026-11-19)

Limit: 8 pages total including references (internal target; owner, 2026-11-19). Author block: placeholders only (HN, #001, #002, #003, AB); resolved at stage 13.

## Page budget (owner table)

| Section | Pages | Claims carried |
|---|---|---|
| Title, abstract, introduction | 1 | A1, A2, A3 |
| §2 Task, cortical model, plasticity mechanisms | 1.5 | none (definitions only) |
| §3 Experimental design and controls | 1 | none (protocol) |
| §4 Results, three main figures | 3 | C2 to C8; R2, R3, R4 pending |
| §5 Discussion, limitations, conclusion | 0.5 | limitations L1 to L5 |
| References | 1 | — |

## Section outline

**§1 Introduction.** Passive stimuli cannot test goal-directed control of a plasticity rule. Closed-loop tasks for spiking models are rare and usually introduce confounds. Statement of Plasticity Boat, scope (10-neuron validation and falsification; 100-neuron competition), and the directional-observability question. Ends with contributions, each tied to a result or a gate.

**§2 Task, cortical model, plasticity mechanisms.**
- Plant: point mass p, v0 = 0, p* = 0, dt = 0.02, damping 0.5, actuator identity; start R = 5 (protocol_v0.md §5).
- Circuit (10 neurons): LIF, fixture from contract §12; motor populations (+x, −x, +y, −y) = neurons 0 to 3.
- Circuit (100 neurons): not yet built; E/I 80/20, four motor groups; EX-25.
- Sensing: F1 distance-only encoder, I_i = b_i + g_i·e/(e + d0), g = (24, 0, 24, 24, 24, 24, 24, 24, 0, 0), d0 = 1. Directional encoder (Option A, provisional, EX-23): u(z) = max(0, z)/(max(0, z) + d0) applied to −p_x, p_x, −p_y, p_y on motor neurons 0 to 3.
- Decoder: f0·(c+x − c−x, c+y − c−y), f0 = 5.
- H-state (F2, F2b, F4, F6): H(0) = 0, τ = 2.0 s, κ = 1.0, clip [0, 4].
- Eight strategies S1 to S8: protocol_v0.md §3.

**§3 Experimental design and controls.**
- Starts: development 20 (θ ~ U[0, 2π), default_rng 2025), held-out 40 (default_rng 2026, unused until the owner's lock).
- Episode K = 600 control intervals; primary window = final 20% (errors[480:600], pre-step).
- Pairing: same starts, same circuit initialisation and seed across strategies.
- Controls: zero drive, stationary input, known-direction force probe, S1 static, S7 shuffled, S6 baseline; directional positive control (EX-24).
- Gates: effective variation ≥ 0.9·N before any CI; bootstrap over starts (10 000, rng 0); no stopping rule; a negative result is reported unchanged.

**§4 Results.**
- R1 (C2, C3, C5): F1 pushes +x; 0 of 20 dev starts end within R; 8 of 8 starts with p_x < 0 dip below R, 0 of 12 with p_x ≥ 0. Figure 2.
- R2 (A2): directional positive control on 10 neurons. [PENDING: EX-24.] Figure 2, panel b.
- R3: 100-neuron zero-drive and stationary controls, runtime. [PENDING: EX-25.]
- R4: eight-strategy competition on the 100-neuron circuit. [PENDING: EX-26 freeze; not claimed.] Figure 3.

**§5 Discussion, limitations, conclusion.** Limitations:
- L1: LIF point neurons, not biophysically detailed.
- L2: the directional encoder is an engineered input; the paper states it.
- L3: single-start results for C4, C6, C7, C8; no CI.
- L4: the sign split is post hoc on the development starts.
- L5: 100-neuron scale and the competition are not yet verified.

**References.** [MISSING: bibliography not compiled; stage 9.]

**Supplementary (stage 5, not in the 8 pages).** Runaway at K = 3000 (R-10); F3 saturation (R-08); envA start sweep (R-09); dev-start table (R-12); Lane C controls; F6 re-run (EX-17).

## Figures

| No. | Message | Data source | Status |
|---|---|---|---|
| 1 | Task and circuit schematic | contract §12, §13.3 (not data) | stage 4 |
| 2 | Distance versus interval for the 20 dev starts, coloured by sign(p_x); panel b, positive control | `_temp/crazy_sailor_explore/falsifier/dev/*.json`; EX-24 | data ready for panel a; panel b pending |
| 3 | Eight-strategy outcomes on the 100-neuron circuit | not yet produced | pending R4 |

## Claim map (carried from stage 1)

- A1 = C2, C3, C5 (§4 R1). Status: observed, descriptive, post hoc split.
- A2 = R2. Status: pending.
- A3 = protocol_v0.md. Status: proposal; not a result.
