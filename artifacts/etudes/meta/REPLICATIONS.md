# Replication index

One row per scored preregistration (release plan 0.5.6–0.5.10, §5). Verdicts stay in,
pass or fail (B2). Ledgers and results live on the lane branch until it merges to dev.

| Release | Paper, figure | Tier | Stage | Verdict | Ledger and result | Receipts | Gallery |
|---|---|---|---|---|---|---|---|
| 0.5.6 | N'dri et al. 2025, PCL Fig. 3 | T-port | P5, prereg v5 | in progress; scored without a pass: v5 T2 (sf curve vs digitized Fig. 3) fails both sets; T1, T3 pass; T4 pass on a post hoc criterion (changed after v4); v4 FAIL stands; PCL-R1 fidelity audit closed 2026-10-24 (`artifacts/etudes/papers/pcl/r1_fidelity_audit_2026-10-24.md`: six findings, not reproduced); next: rescore on the core with T4 declared up front (Hamm 2026-10-18) | `pcl-port` 8f75acd0: `artifacts/etudes/papers/pcl/LEDGER.md`, `prereg/P3_fig3_v5_result.md`, `prereg/P3_fig3_result.md`, `prereg/P3_fig3_diagnostic.md` | `F:\warehouse\pcl\scores\` | none yet (P7) |
| 0.5.6 (P5); P6 in 0.5.7 | Tahvili, Vinck, di Volo 2026, CAMINOS Fig. 1 | T-jaxfne | P6, prereg v6 + amendment v7 (R4); P6 v2 library path through the public API | mixed (Hamm 2026-10-14; primary 2 ms model, `CALIBRATION.md` §10): reproduced f_pk 33.80 Hz (e 0.034), E rate inside 1–3 Hz, PV/SOM/E rate order, R4 onset p = 0.04 after the v7 fidelity fix; finding that needs X: PV→SOM lag 6.47 ms on seeds 1–3 (e 0.239), 6.32 ms on seeds 4–8, P3 estimator 7.22 ms, all off against 8.5 ms; recorded sensitivity: 0.1 ms delay gives the preprint ≈ 7 ms (e ≤ 0.014) with f_pk 43–46 Hz | `l2-caminos` 1e768f9d: `artifacts/etudes/papers/caminos/LEDGER.md`, `RESULT_P3.md`, `CALIBRATION.md` | `F:\warehouse\caminos\scored\`, `…\v7_20261012\`, `F:\warehouse\caminos\p6_scored_v2\` | none yet (P7) |
| 0.5.7 | Izhikevich 2003, Fig. 2 and 3 | T-jaxfne | P5, prereg v7 (signed 2026-10-13) | in progress; 6 of 8 under signed v7; 8 of 8 under post hoc amendment v8.2 (release notes cite v7). Fig. 2 C4 (FS ISI ratio 1.86) and C8 (RZ spikes once) fail, 6 of 8 pass; Fig. 3 N1, N3, N4 pass, N2 fails (median ISI CV 0.40, band 0.6–1.5; 0.39 on the S1 reference arm too). Amendment v8.2 (post hoc, signed 2026-10-13): Fig. 2 passes 8 of 8 (C8 pulse I = 2; C4 as the least-adapting type, FS 1.86 against RS 2.53 and LTS 4.39); N2 stays a failed paper claim (`net.m` gives 0.39). P3b (`net.m` update order) stopped at its equivalence gate | `izh2003` 34c38273: `artifacts/etudes/papers/izhikevich2003/RESULT_P3.md`, `scores/p3_v7_score.json`, `scores/p3_v7_s1_report.json`; `scores/p3_v8_score.json` | `F:\warehouse\izh2003\scored_v7\`, `F:\warehouse\izh2003\v8\` | none yet (P7) |
| 0.5.8 | Destexhe 2009, Fig. 6B self-sustained AI (AdEx) | T-port | P3 (steps 1–6 scored against the unsigned draft `PREREG_P3.md`; unscored by ruling 2026-10-18, B8) | no verdict; pilot step 6: E1 alive on 4 of 8 seeds (2, 3, 6 sustained), E2 off on every alive seed, E4 rate order off (about 5x hyperactive); reserved seeds 4–8 spent, new reserved seeds before signing | `artifacts/etudes/papers/destexhe2009/CALIBRATION.md` steps 1–6 | — | — |
| 0.5.8 | Nejat, Sherfey, Bastos 2026, GSDR Fig. 6 | T-jaxfne | P1 (steps 4–6.13 scored against the unsigned draft `PREREG_P3.md`; unscored by ruling 2026-10-18, B8) | no verdict; pilot fits ran the gradient-free lane loop without MCDP (P-039) and stay saturated (kappa 0.60 on the 500-generation reference refit against y 0.0156) | `artifacts/etudes/papers/gsdr2026/CALIBRATION.md` steps 4–6.13 | `F:\warehouse\gsdr\gsdr_fit500_reference_2026-10-17\` | — |
| 0.5.7 | Brunel 2000, Fig. 8 (SR, SI fast, AI, SI slow) | T-jaxfne | P5, prereg v4 (signed 2026-10-19) | reproduced (conditional): E1-E5 in band on seeds 4-8 (B rate 59.33 Hz, e 0.023; B global 182.4 Hz, e 0.013; C rate 37.87 Hz, e 0.0044; D rate 5.63 Hz, e 0.024; D global 20.6 Hz, e 0.064); Q1-Q3 pass on every seed; condition: Q3 reads the paper's wording (stationary global, CV > 0.1), chosen after the pilot failed the classifier's asynchronous bounds (C: CV 0.40-0.41, kappa 0.20); float64 at C agrees | `artifacts/etudes/papers/brunel2000/RESULT_P3.md`, `artifacts/etudes/papers/brunel2000/score_p3.json` | `C:/workspace/jaxfne_receipts/brunel2000_p3v4/` | none yet (P7) |

Decisions (Hamm 2026-10-13; CAMINOS ruling 2026-10-14; status tags 2026-10-16):

- Status tags (Hamm 2026-10-16; mixed 2026-10-19): reproduced, reproduced (conditional),
  mixed, in progress, not started, failed (rare). "Not reproduced" is no status: a preregistration scored without a
  pass keeps the paper in progress, and the scored criteria stay visible (B2). The former
  demonstrations (PCL Fig. 2, HH AT-01, Q-learning) are in progress; CAMINOS stays mixed
  (Hamm 2026-10-18); the dashboard separates the P6 library-path PASS from the paper verdict.

- PCL Fig. 3 stays in progress (Hamm 2026-10-13: it must replicate). The v4 FAIL stands as
  recorded; v5 (signed 2026-10-13) also fails, on T2. The next attempt is a new,
  separately declared protocol, and the P6 library path follows (B1).
- CAMINOS final (Hamm 2026-10-14, `CALIBRATION.md` §10): PLoS CB is primary; f_pk,
  rates and R4 reproduced; the PV→SOM lag miss is a finding that needs X; no residual
  fit ran; the 0.1 ms preprint sensitivity is recorded.
- CAMINOS P6 moves to 0.5.7 (Hamm 2026-10-13, todo stack section 5): its P6 prereg goes to
  signature and its equivalence test lands in that release.
- Izhikevich 2003 scored in 0.5.7 without a pass, in progress (Fig. 2 C4/C8, Fig. 3 N2; post hoc
  amendment v8.2; P3b stopped at its equivalence gate).
- GSDR and Destexhe 2009 (Hamm 2026-10-18): steps scored against unsigned draft preregs are
  unscored pilots, kept visible (B2). Next per B8: pilot, then signature, then fresh seeds.
  Both in progress, ship in 0.5.8.

B1: library-path attempts: CAMINOS P6 v2 ran through the public API (mixed verdict per §10); PCL has none yet.

## Queue

| Paper | Tier | Stage | Note |
|---|---|---|---|
| Potjans and Diesmann 2014, Cereb Cortex 24(3):785-806, "The cell-type specific cortical microcircuit" | T-jaxfne (Hamm 2026-10-18: jaxfne core only) | P2 | Fig. 6 spontaneous rates/CV/synchrony and the Table 5 connectivity map. P1 source record: `artifacts/etudes/papers/potjans2014/SOURCE_P1.md`. Rule B8: pilot, then signature, then fresh seeds. In progress, ships in 0.5.9 |
| Nejat, Sherfey, Bastos 2026, PLoS One 21(8): e0354021, "A genetic algorithm for self-supervised models of oscillatory neurodynamics" (GSDR) | T-jaxfne from the start (Hamm 2026-10-13: "just do it with jaxfne") | P1; pilot row above | 50-cell HH E/PV/SST circuit, GSDR (stochastic delta rule plus MCDP plus genetic revert) on recurrent conductances; Figs. 2–4 and 6 are model claims, Fig. 5 is macaque MT/MST data. Code public (`HNXJ/GSDR`, DynaSim `devDL`), read as data only. P1 source record: `artifacts/etudes/papers/gsdr2026/SOURCE_P1.md`. In progress, ships in 0.5.8 |
| Vogels TP, Sprekeler H, Zenke F, Clopath C, Gerstner W 2011, Science 334(6062):1569-1573, "Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks" | T-jaxfne | P2 | Inhibitory STDP (I-to-E, pairing potentiation plus per-prespike depression) on the jaxfne core only; targets Fig. 4 (recurrent network driven to rho0 = 3 Hz) and Figs. 1-2 (single-cell detailed balance). P1 source record: `artifacts/etudes/papers/vogels2011/SOURCE_P1.md`. Rule B8: pilot, then signature, then fresh seeds. In progress, ships in 0.5.8 |
| Mejias, Murray, Kennedy and Wang 2016, Sci Adv 2(11):e1601335, "Feedforward and feedback frequency-dependent interactions in a large-scale laminar network of the primate cortex" | T-jaxfne (Hamm 2026-10-18: jaxfne core only) | P2 | Two-module (gamma supra, alpha infra) Wilson-Cowan laminar circuit in 30 areas on FLN/SLN/wiring-delay anatomy; targets Fig. 2 (local gamma), Fig. 3 (interlaminar coupling), Fig. 6-7 (functional hierarchy and jumps). P1 source record: `artifacts/etudes/papers/mejias2016/SOURCE_P1.md`. Rule B8: pilot, then signature, then fresh seeds |
| Batuev and Sukhov 2026, Dynamics 6(4):42, "Predictive-Coding-Inspired Reconstruction of Nonlinear Dynamics in Spiking Neural Networks", doi:10.3390/dynamics6040042, CC BY 4.0 | T-jaxfne (queued by Hamm 2026-10-20) | P0 | Two prediction-error architectures (external comparator; a spiking error population) in the Neural Engineering Framework, trained with the local PES rule, reconstructing a 2-D oscillator and the Lorenz system; scored by reconstruction error, envelope and instantaneous-frequency correlation, PLV and phase difference. Core gaps before P1: NEF encoders and decoders, the PES rule, a continuous-signal readout. Methods and code availability unchecked (P1) |
| Jansen and Rit 1995, Biol Cybern 73(4):357-366, "Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns", doi:10.1007/BF00199471 | T-jaxfne (opens the EEG/MEG lane, Hamm 2026-10-20) | P0 | Three-population column (pyramidal, excitatory and inhibitory interneurons) with second-order synaptic kernels and a sigmoid rate; alpha rhythm and the VEP of coupled columns; the EEG proxy then goes through the core head forward model. Core gap before P1: a neural-mass emitter. Methods and code availability unchecked (P1) |
