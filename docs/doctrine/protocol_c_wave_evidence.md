# Protocol C wave evidence (0.4.17-C)

**Status:** Protocol C **CLOSED** at C4 (`c4_interpretation_receipt.json`)

## Evidence tiers

| Layer | Status | Receipt |
|-------|--------|---------|
| C0 wave protocol spec | SPECIFIED | `c0_wave_protocol_spec.json` |
| C1 estimator \(\widehat{\mathcal W}\) on synthetic \(\Phi\) | TESTED (synthetic only) | `c1_synthetic_validation_receipt.json` |
| C2 `delay_state` runtime continuation | TESTED | `c2_delay_continuation_receipt.json` |
| C3 neural geometry/delay prospective run | OBSERVED (60 cells) | `c3_execution_receipt.json` |
| C4 interpretation freeze | FROZEN | `c4_interpretation_receipt.json` |

## What is evidenced

- **C1:** When a field is injected with known planar or negative-control structure, the frozen estimator returns the preregistered three-way classification with frozen gates.
- **C2:** Baseline recurrent `Model.simulate` can continue \(\mathcal B_t\) without losing delay history under segmentation.
- **C3/C4:** Under frozen ring geometry/delay factorial conditions, band-limited \(V_m\) was passed to the **same** estimator; per-cell classifications and quality diagnostics were recorded prospectively without post-hoc retuning.

## What is **not** evidenced

- That jaxfne neural dynamics **generate** traveling waves in vivo or in full biophysical models.
- That detected classifications imply field-mediated propagation or axonal conduction velocity \(v_c\).
- That heterogeneous delays alone imply propagation.
- Inferential significance of factorial contrasts (descriptive counts/proportions only; \(n=10\) seeds per condition).

## Causal distinction (frozen)

\[
\text{Protocol D: how signals are delayed;}
\qquad
\text{Protocol C: whether activity exhibits propagation signatures under }\widehat{\mathcal W}.
\]

Estimator output does not feed upstream.

## Follow-up science

Further wave claims require a **new protocol identifier**. Protocol C is closed; do not open C3b/C5 under this ID. Proceed to **0.4.17-D** (biological RBS) without delaying on wave-positive outcomes.
