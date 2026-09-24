# 0.5.3 ENGINE item 5 receipt — plasticity controls (W2)

Branch `w2-53` (after 7b commit `6b9595a`). Authority: H5 (adversarial),
H7 (dead-param classification), H8 (`d_H=1` valid; RBS != homeostasis).

## Controls

`jaxfne/hdp_network.py`: `enable_plasticity` / `disable_plasticity` /
`clamp_plasticity` / `pin_projection_weights` / `projection_mask`
(indices XOR 0/1-values, never ambiguous). Mechanism: optional per-edge
`plasticity_mask` on both HDP kernels (None = bit-exact legacy path via a
static Python branch; entries > 0.5 plastic, else `w_next = w` exactly).
Threaded through `_hdp_kernel_kwargs` (both branches), `compile_step_fn`
(signature-derived allowlist admits it for `hdp`, refuses for `baseline`),
`KNOWN_HDP_PARAM_KEYS` + regenerated contract artifact. `hdp_network.run`
validates via the 7b helper.

Per rule: legacy disable zeroes `K_HDP` AND `K_w_ctrl` together (the
K_HDP=0 template: `K_HDP=0` alone leaves the restoring term live, so the
template is the pair); registered disable zeroes the rule's `k_w`
coefficient, or raises directing to an explicit mask when the rule has no
`k_w`. Per projection: mask (gains kept, rest stays plastic); clamp =
mask + pinned `w0` (`pin_projection_weights`, one mask object, True =
plastic everywhere). H dynamics are never frozen by these controls.

## fixedW identity proof (dedicated test, not assumed)

`test_fixedW_identity_proof_disabled_equals_null_bit_for_bit`, 2-edge
fixture, 40 steps: explicit null gains vs all-zero mask (gains ON) vs
`disable_plasticity` output — all three `w_trace` array-equal AND equal
to `w0`; V/spikes/sources/H_trace agree across all three. Structural
basis: null takes the `w_next = w` gate (no float ops on w); mask=None
takes a static branch containing byte-identical code to pre-0.5.3.
H≠W: in the same disabled run H evolves under drive while W is bit-fixed.

## Verification

- New `tests/test_plasticity_controls_053.py`: 24 passed (API unit 10,
  identity/clamp/isolation 7, adversarial masks 5, Model/run level 3).
  Measured: unmasked edges move (legacy max|dw| ~1e-4 over 40 steps on the
  fixture; 56/56 move on suite2/40 steps), masked/clamped edges bit-fixed.
- Existing, affected: hygiene+contract+hdp01+kernel_standalone+gen01+law01
  (101) / dispatch+delayed_registrable+population+finite_delay+audit01+
  barrier+equiv01+rec01 (57) / ei_hebbian+ei_rho+config_baseline+phaseC+
  continuation_contract (36) — all passed.
- Invariants: K_HDP=0 null path untouched (gate still `w_next = w`);
  `d_H=1` throughout (no `h_state_dim` set anywhere); H≠HDP untouched.

## Boundaries (fail closed, documented)

- `plasticity_mask` + population locality: refused (per-edge weights there
  derive from theta channels; 0.5.4 cross-area territory).
- `plasticity_mask` + `kernel="baseline"`: refused by the 7b allowlist
  (mask is meaningless without plasticity).
- Mask honored iff the HDP kernel runs; null-gains runs are fully fixed-W
  regardless (controls-level promise, receipt-stated, not a silent drop).
- Manifest arrays: mask rides `hdp_params` like the existing
  `controller_B`/`m_ei_edge_mask` precedent; full serialize+roundtrip is
  item 6 (H4).
