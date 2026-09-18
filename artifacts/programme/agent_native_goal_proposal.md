# PROPOSAL — "Agent-native JaxFNE" as a durable project goal

**Status: AWAITING HUMAN APPROVAL. Not adopted.**

Project sources and `artifacts/fact_stack.md` are human-edit-only, so this file
holds the exact proposed text rather than writing it into a human-owned source.
Nothing in the repository treats this as authority yet.

Authorized as a direction by Hamm, with the instruction that the durable goal
belongs in a project source while `artifacts/todo_stack.md` keeps only the
executable work needed to reach it. The executable work is in the todo stack
under "Long-term goal — agent-native JaxFNE"; the rationale is not duplicated
there.

## Proposed text, verbatim

> **Agent-native JaxFNE.** JaxFNE should support reliable AI-assisted
> mathematical-biophysics modeling in which scientific intent is translated
> through explicit TFNE specifications into tested JaxFNE operations. TFNE,
> code, documentation, skills, inspection, and tests should expose one
> scientific system. Hierarchical TFNE is the human/agent specification
> representation; realized flat state and parameters are the efficient
> execution representation. Agent workflows must preserve configured,
> realized, executed, and observed semantics, fail explicitly on consequential
> missing or unsupported information, and match verification strength to the
> scientific claim. AI assists construction, execution, inspection, and
> verification; it does not replace scientific judgment.

## What remains for the human to decide

1. Whether to adopt the paragraph at all.
2. Which source owns it — a new project source, an existing one, or
   `artifacts/fact_stack.md`.
3. Whether the wording above is final. It is Hamm's own text, recorded
   unaltered; I have not edited it.

## Why it is worth durable status

The goal outlives the work items that serve it. Kept only as pending work, the
architectural reason for the later skills, tools and tests would be lost once
those items closed, and the constraint that distinguishes this from generic
"AI-assisted modeling" — that natural language must pass through an explicit,
inspectable TFNE specification rather than becoming generated simulator code —
is exactly the part that has to survive.

Two defects found during TFNE-PARAM-01 are the concrete argument. A declared
weight of 0.5, 0.25 or 0.125 all executed at 0.353553 while every edge count
matched, and a declared `AMPA` mechanism still executes with a 0.1 ms time
constant rather than AMPA's 2.0 ms. Both are the same failure class:

```
researcher's intended model != agent's executable interpretation
```

An agent with direct repository access would have produced a perfectly
executable simulation in either case, and reported success. The goal above is
what makes that class of failure mechanically difficult rather than merely
discouraged.

## Retained lesson

Verification must perturb parameters with asymmetric, non-neutral values and
inspect downstream consumption. Equality of counts, names, shapes or
default-valued outputs is too weak to establish semantic preservation, and
`structure identity != parameter identity` just as
`stored parameter != consumed parameter` and
`mechanism identity != mechanism kinetics`.
