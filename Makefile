.PHONY: test-dev test-broad test-release test-rc test-publication surface-contract public-surface-contract readme-atlas

# Validation gates — file lists live in scripts/run_test_gate.py, not here.
test-dev:
	python3 scripts/run_test_gate.py dev

test-broad:
	python3 scripts/run_test_gate.py broad

test-release:
	python3 scripts/run_test_gate.py release

# Release-candidate gate. Run this on the candidate SHA itself: it writes the
# untracked attestation that scripts/release/reconcile_release_target.py needs
# to authorize publication. See docs/ci_policy.md.
test-rc:
	python3 scripts/run_test_gate.py rc

test-publication:
	python3 scripts/run_test_gate.py publication

# Regenerate local development surface contract (gitignored artifacts/developer/).
surface-contract:
	python3 scripts/generate_surface_contract.py

# Regenerate the tracked public surface contract artifact from the live module.
public-surface-contract:
	python3 scripts/generate_public_surface_contract.py

# Regenerate the canonical atlas: interactive docs panels + README stills.
readme-atlas:
	python3 scripts/generate_readme_atlas.py
