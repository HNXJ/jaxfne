# Adversarial review packet (template)

Launch: `python <opencode-delegate skill dir>/oc_delegate.py PACKET.md --tier review --dir <repo> --context <diff file>`
with the diff written by `git diff <base>..<head> > <scratch>/review.diff`. Read-only; the reviewer
has none of the session context, so the packet must stand alone.

```text
GOAL      Find defects in the change below. You are an adversary: assume it is wrong
          until the code and tests show otherwise.
BASE      <base sha>   HEAD <head sha>   DIFF attached as context.
CLAIMS    <each claim the change makes, one per line, with the test that supports it>
CHECK     1. Does each test fail on the pre-change code? (named-for-case fixtures, no
             vacuous passes, selectors that collect tests)
          2. Stored vs consumed: does every new parameter change the realized object or
             get refused?
          3. Scientific honesty: selection status of tuned constants, in-sample vs
             out-of-sample, negative results reported.
          4. Scope: unrelated behavior changed? frozen artifacts touched?
          5. Docs/results/registry consistent with the code?
OUT       Findings ranked by severity, each with file:line, the concrete failing
          scenario, and a proposed fix. Say "no finding" per check when clean.
```

After the review: fix each finding or record it in `artifacts/issue_log/ISSUE_LOG.md`
with its reason; verify every claimed defect yourself before acting on it.
