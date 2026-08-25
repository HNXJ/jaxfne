SCIENTIFIC NOTE FOR E2b PROVENANCE (attached per freeze-audit condition)

Source: E2a receipt artifacts/e2/preregistration/E2a_search/e2a_search_receipt.json
        independently recomputed by pre-verification worker A at E2b open.

Finding: All six Theta_search candidates achieved G_adequate rate 1.0
        (5/5 D_dev seeds each, seeds 9000-9004). Therefore

    theta* = {id: theta0, drive_E: 4.0, drive_I: 2.0,
              weight_mu: 0.25, noise_scale: 0.0, W_ms: 60}

is NOT a unique optimum. It was selected solely by the preregistered
lexicographic tie-break (lowest drive_E, then lowest weight_mu) applied to a
six-way tie. E2a did not identify a narrow physiological operating region.

Supported downstream claim:  "all six candidates were adequate in D_dev;
theta0 is the tie-break-preferred representative."
Unsupported claim (forbidden): "theta* is the best operating point."

Additional disclosure carried into E2b provenance: the E2a adequacy numbers
were synthetic proxy computations (executor line 48 discloses this); they are
valid for the blinded selection mechanics that were frozen, and E2b now
measures real dynamics. The E2b receipt's calibration_generalization field
compares confirmatory C0 adequacy against the E2a proxy 6/6 and must report
any material difference as calibration-generalization evidence, never absorb
it silently into the phenotype result.
