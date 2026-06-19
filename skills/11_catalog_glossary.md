# Catalog and Glossary Skill

## Purpose
Prevent invented names and stale API references.

## Rules
- Verify a symbol in the catalog, docs, or `jaxfne.__all__` before using it.
- If a function, flag, or path is uncertain, search the repo rather than guessing.
- Prefer current public names and compatibility wrappers over deprecated or retired aliases.
- Keep the glossary aligned with actual code, not desired code.

## Acceptance checks
- No invented API is used in a patch or prompt.
- A symbol's module and role are known before editing it.
- The agent can distinguish stable public API from experimental scaffolding.
