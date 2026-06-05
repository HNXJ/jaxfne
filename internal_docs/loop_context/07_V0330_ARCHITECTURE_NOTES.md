# 07_V0330_ARCHITECTURE_NOTES

The assessment plan defines v0.3.30 as the final atlas/evidence-bundle step plus connectivity compiler, Net/FlatNet boundary, optional trainer wrapper, and PyNWB export bridge. It states stop conditions: no dense O(N^2) compiler except explicit matrix mode, SHA256 validation for artifact_ref, no Python objects in traced FlatNet kernels, PyNWB must be lazy, and proxy NWB data must not use calibrated units.

|target|current state with evidence|contract to satisfy|human design questions|proposed phase plan|
|---|---|---|---|---|
|Connectivity compiler|partial; builders/EdgeList present in live audit; assessment says v0.3.30 deliverable|MechanismSpec/ConnectionRuleSpec/WeightInitSpec/compile_connection_rules; deterministic sparse edge arrays; artifact_ref SHA256|schema module location; sparse default vs explicit matrix; public names/wrappers|Phase 1 freeze contract; Phase 2 tests; Phase 3 implementation; Phase 4 full validation|
|Net/FlatNet transform|partial; PyTree sites present, FlatNet not public per audit|Config->Net->FlatNet, TrackingMaps, PyTree roundtrip, simulate_flat smoke; no Python objects in JIT|whether FlatNet is stable public or experimental; relationship to existing Model|Design doc -> tests -> wrapper -> smoke|
|Final atlas bundle|docs/notebooks mostly pass; Etude long cells remain|all notebooks listed; docs links valid; artifacts named/hashed; proxy boundary clear|whether to split 397-line cell or leave as accepted debt|Index files -> link tests -> notebook receipts -> asset hash manifest|
|Optional PyNWB export|absent/guarded only|lazy optional dependency; explicit units/status/provenance; read/write round trip; proxy not calibrated units|schema exactness, units/status names, session metadata fields|Design schema -> optional tests -> writer -> readback|
|Strict validation|strong but improve release cleanup and public install smoke|compile/full tests/audit/mkdocs/build/twine/install smoke/artifact validators|where release script lives; TestPyPI workflow path|harden scripts -> CI job -> receipt template|


## No-code rule

Do not open a new public API for any RED item without: frozen contract, tests first, wrapper/deprecation plan, validation command receipts, and scope/status metadata unchanged.
