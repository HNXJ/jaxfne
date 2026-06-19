# API Contracts Skill

## Purpose
Keep public APIs finished, compatible, and unambiguous.

## Rules
- Public helpers must be implemented, wrapped for compatibility, or clearly marked experimental.
- A public `NotImplementedError` is acceptable only if the surface is intentionally fenced.
- Prefer backward-compatible extensions over breaking changes.
- If a helper exists, its contract must be testable from the public entry point.

## Acceptance checks
- No public function appears finished while still acting as a hidden stub.
- Compatibility wrappers exist when signatures evolve.
- Tests cover documented public paths rather than only internals.
