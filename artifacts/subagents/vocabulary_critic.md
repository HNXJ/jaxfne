# Vocabulary critic (reusable subagent)

Purpose: review JaxFNE prose against `artifacts/vocabulary/JAXFNE_VOCABULARY.md`
while preserving scientific meaning.

Infrastructure target (W14): one vocabulary source + one critic + one
deterministic audit — not overlapping vocabulary skills.

## Procedure

```text
inspect → classify → propose/change → preserve → verify → stop
```

### 1. Inspect

- Read `artifacts/vocabulary/JAXFNE_VOCABULARY.md`.
- Work only on the **bounded file set** assigned by the main agent.
- Do not edit the same file as another concurrent critic pass.

### 2. Classify each candidate occurrence

| Label | Action |
| --- | --- |
| REPLACE | Noncanonical prose; canonical term preserves meaning |
| TECHNICALLY_REQUIRED | Keep; term is precise in context (see controlled words) |
| API_IDENTIFIER | Keep; code/API/type name |
| MATHEMATICAL_TERM | Keep; equation or operator notation |
| HISTORICAL | Keep; changelog or frozen evidence |
| CITATION | Keep; bibliography or external title |
| UNCERTAIN | Surface to main agent; do not replace |

### 3. Propose / change

**Public entry pages** (`README.md`, `docs/index.md`): minimum words; no
named-tool comparisons, competitive positioning, defensive "not X" prose, or long
novelty lists (see `artifacts/context.md`).

- Apply **only** `REPLACE` automatically.
- Prefer canonical terms from §1.
- Priority replacements (semantic, not mechanical): unnecessary `truth`,
  `doctrine`, `ontology`, `grammar`, `contract`, `protocol`, `gate`,
  `framework`, `architecture` in generic prose.

Examples:

- "scientific truth" → "scientific evidence" or "current state"
- "execution grammar" → "execution pipeline"
- "public contract" (prose) → "public API"
- "doctrine" (public prose) → remove or "rule" / "method"
- "protocol" → keep only for named experimental protocols

### 4. Preserve

Do **not** modify for vocabulary alone:

- code identifiers, API names, file paths
- equations and mathematical symbols
- citations and bibliography titles
- historical changelog entries
- test literals and gate command names
- allowlisted fixed phrases in `scripts/audit_vocabulary.py`

### 5. Verify

After edits on assigned files:

```bash
python scripts/audit_vocabulary.py --check
mkdocs build --strict   # if public docs changed
pytest tests/test_agent_context_hygiene.py -q
```

### 6. Stop

Return report:

```text
FILES_REVIEWED
TERMS_FOUND
REPLACED
PRESERVED_TECHNICAL
UNCERTAIN
SEMANTIC_CONFLICTS
TESTS
```

## Evolution

Improve this file or `JAXFNE_VOCABULARY.md` only from **observed failures**
(changed meaning, missed jargon class, new inconsistency). Record:

`trigger` → `cause` → `repair` → `evidence` → `scope`

Do not grow from hypothetical concerns.

## Pass assignments (typical)

| Pass | Scope |
| --- | --- |
| A | `README.md`, `docs/index.md`, `docs/scope_and_status.md`, `docs/quickstart.md`, `docs/api/index.md` |
| B | Remaining mkdocs `nav:` pages under `docs/` |
| C | `artifacts/context.md`, `artifacts/AGENTS.md`, `artifacts/skills/`, `docs/for_ai_agents.md` |
