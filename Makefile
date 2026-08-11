.PHONY: test-dev test-broad test-release test-publication surface-contract

# Validation gates — file lists live in scripts/run_test_gate.py, not here.
test-dev:
	python3 scripts/run_test_gate.py dev

test-broad:
	python3 scripts/run_test_gate.py broad

test-release:
	python3 scripts/run_test_gate.py release

test-publication:
	python3 scripts/run_test_gate.py publication

# Regenerate local development surface contract (gitignored artifacts/developer/).
surface-contract:
	python3 scripts/generate_surface_contract.py
