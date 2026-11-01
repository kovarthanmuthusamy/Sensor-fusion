## What
<!-- What changes, in two or three bullets. -->

## Why
<!-- The problem or decision behind it. Link the ADR if there is one. -->

## How tested
<!-- Tests added or changed, commands run, numbers measured. -->

## Not verified
<!-- Be plain about what you could not run or check (for example: the container image, a GPU path). -->

## Checklist
- [ ] Branched from the latest `main`; one task in this PR
- [ ] `make check` passes (lint, types, tests)
- [ ] Behaviour that matters in deployment has a test that fails if it is removed
- [ ] `CHANGELOG.md` updated for anything a user would notice
- [ ] ADR added or updated for a real decision; no data, checkpoints or secrets committed
