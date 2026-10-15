# 24-ENT-01 receipt — ENTRY_GRANTED

**Branch:** `dev` (on top of `a9a3ee9`).

## Entry criteria verification
1. v0.4.23 shipped: tag `v0.4.23` peel `7e91da3` == `origin/main`;
   PyPI `0.4.23` exists (trusted publishing run `34703754053`);
   GitHub release with CI bytes (`ad23dafb…`/`d3805f40…`); RTD stable
   build `34526536` @ `7e91da3` success. All independently verified
   during 23-PUB-01 closure.
2. W8 baseline present: `artifacts/audit/w8_baseline.json` (schema
   `jaxfne.w8.baseline.v1`, 2026-09-10) plus cumulative programme
   receipts `artifacts/programme/v0_4_23_*_receipt.md`.
3. Version coherence: `test_docs_version_alignment` 13/13 green on the
   published state (source 0.4.23, PyPI 0.4.23, previous 0.4.22).

Verdict: ENTRY_GRANTED for the v0.4.24 programme. Next: 24-W16-6.
