# Patch note

This bundle is a structural cleanup of the earlier skill package.

## What changed
- flattened the bundle into root-level markdown files only
- removed nested skill-folder dependency from the bundle format
- separated repo orientation from enforcement rules
- added catalog and objective-grammar guidance so the skills also teach repo usage
- kept the enforcement skills focused on the concrete weak points:
  - analysis fallback masking
  - dense connectivity construction
  - batch/vectorization defaults
  - projection semantics
  - runtime fallback transparency
  - API contract completeness
  - parameter scope clarity
  - experimental fencing

## How this should be applied
- copy the markdown files into repo-root `skills/` (not `jaxfne/skills/`)
- keep the bundle flat in the repository as well
- update the repo README or agent docs only if a path or rule changed
- run the relevant tests after each meaningful edit
