# 001 — Repo skeleton and the branch → PR workflow

**PR:** #1 `feat/repo-skeleton` · **Date:** 2026-10-26 · **Tags:** code-craft, mlops, design

## What was built
- Packaging and tooling: `pyproject.toml`, `requirements.lock.txt` (exact versions), ruff, pre-commit, mypy, pytest + coverage.
- Hydra configs (`configs/`) with RGB-baseline and fusion model variants and mini/trainval data variants ([ADR 0003](../adr/0003-hydra-config.md)).
- Package skeleton: seeding, GPU-architecture check for Volta (`sm_70`), and a `train.py` entry point that logs the resolved config.
- 28 tests, including a test that the baseline and fusion configs differ *only* in the input.
- GitHub Actions CI: lint job and test job on Linux with CPU PyTorch.

## Decisions made (by you)
- Hydra over dataclasses or argparse. Quality gates: pre-commit, mypy, coverage.
- MIT license; university email kept as the commit identity.
- Squash-merge only, no direct pushes to `main`, 0 required approvals (solo repo).

## Senior instinct
- **Check the commit identity before the first push.** The author is stamped into the commit hash; changing it after publishing means rewriting shared history.
- **Branch from `main`, keep branches short-lived, name them `type/short-description`** (`feat/`, `fix/`, `chore/`, `docs/`, `test/`). Small PRs get real reviews.
- **A protection rule must fit the team size.** "1 approval" on a one-person repo locks you out of your own `main`.
- **Never run a destructive command without reading what it destroys.** `git reset --hard` removes uncommitted work, not just a commit. Use `--soft` to undo only the commit, or check `git status` first.
- **Commit early.** Staging stores file contents in git's database; that is the only reason the lost edits could be recovered (`git fsck` lists unreferenced objects).
- **Machine-wide defaults hide per-project choices:** the cached GitHub login and the git identity were both picked up from the machine, not from the project. Pin them per repo.
- **Push the base branch first.** The first branch pushed to an empty repo becomes its default branch. Push `main`, check the default branch in Settings, then push feature branches.
- **Test a protection rule after creating it.** Try a direct push to `main` and confirm it is rejected. Don't assume the rule landed on the branch you meant.
- **Read your commit messages before the first push, not only the diff.** Trailers (`Co-Authored-By:`) and the author identity become permanent history. Changing pushed history needs a force push, which branch protection blocks, so clean up before the ruleset exists.
- **Rewrite history with a safety net.** Make a backup branch first, then prove the rewrite changed only the messages by comparing each commit's tree hash (the snapshot id of its files) before and after.
- **Put a pure rename in its own commit.** Git then records renames, so `git log --follow` and `git blame` keep each file's history, and a reviewer sees "names changed, content didn't".

## Errors hit
| Symptom | Root cause | How to diagnose next time |
|---|---|---|
| `git push` → 403 as the wrong account | Git Credential Manager caches one login per host (`github.com`) for the whole Windows user | The remote URL has no username. Put it in the URL, and remove stale logins from Credential Manager |
| Commit author was `...@domain.local` | No `user.email` configured, so git invented one | `git config --show-origin user.email`; use `git log -1 --format='%an <%ae>'` before pushing |
| Tracked-file edits vanished after "undoing" a test commit | `git reset --hard HEAD~1` also discards uncommitted changes | Read the reflog (`git reflog`), then `git fsck --no-reflogs` to find unreferenced blobs and restore them with `git cat-file blob <hash> > file` |
| `.coverage` appeared as untracked | Not in `.gitignore` | Run `git status --ignored` after the first test run and add generated files |
| No `main` on GitHub, so no base branch for the PR | `feat/repo-skeleton` was pushed first, and the first branch pushed to an empty repo becomes the default | `git ls-remote --heads origin` lists the branches GitHub has. Then `git push -u origin main` and switch the default branch in Settings → General |
| Claude listed as a co-author on GitHub | Every agent-made commit ended with a `Co-Authored-By:` trailer, which GitHub reads as a co-author | Search the messages with `git log --format=%B` for `co-authored-by` before the first push. Fix: `git rebase <base> --exec <script>` to strip the line, then `git push --force-with-lease` |

## Quiz
**Q1 (credential stores):** What is the security risk of GCM keeping logins machine-wide?
**Your answer:** Another user can share my credentials even with 2FA, and push or pull my repos.
**Feedback:** Right that 2FA doesn't help: the stored OAuth token was issued *after* 2FA. Correction: credentials are encrypted per Windows user, so the real exposure is anyone in your Windows session, administrators, and any program running as you. Revoke tokens at github.com/settings/applications.

**Q2 (merge vs rebase):** How do you update a feature branch when `main` moved on?
**Your answer:** Merge `main` into the branch, or rebase onto `main`.
**Feedback:** Correct. Missing the reason: rebase gives a linear history but rewrites commits, so a pushed branch then needs `--force-with-lease`. With squash merge, a plain merge is the safe default.

**Q3 (approvals on a solo repo):** What happens with 1 required approval and one person on the repo?
**Your answer:** A pull request is opened for merge.
**Feedback:** Missed the point. The merge button stays blocked: GitHub ignores the author's own approval, so no valid approval can ever arrive. Fix: set approvals to 0 (keep PR + CI required).

**Q4 (why amend is easy before pushing):** Why is fixing a commit author hard after pushing?
**Your answer:** It's safer to amend now; later we can't navigate back and change it.
**Feedback:** Mostly right. The mechanism: an amend creates a new hash, so your history and GitHub's disagree and fixing it needs a force-push, which the `main` ruleset forbids. (Note the difference between *committed* and *pushed*.)

**Q5 (rename in its own commit):** Why is the `Docs` → `docs` rename a separate commit on `main` instead of part of the feature branch?
**Your answer:** A branch is a separate folder in Windows, and I enabled case sensitivity for only this folder.
**Feedback:** Incorrect, and the starting idea is off. A branch is not a folder: git keeps all branches in `.git`, and the project folder shows one branch at a time. The folder's case-sensitivity flag was in fact *disabled* when checked (`fsutil file queryCaseSensitiveInfo`), and it would not matter anyway, because GitHub and Linux CI are always case-sensitive. The real reasons: (1) a rename-only commit is recorded as renames, so file history survives; (2) the wrong name was already committed in `main`'s first commit, so fixing it there means every later branch starts from correct paths; (3) the reviewer sees at a glance that no content changed.

**Q6 (default-branch ruleset):** Why does it matter that the default branch is `main` when the ruleset targets "the default branch"?
**Your answer:** Because `main` is the branch we aim to protect, and another branch would collapse the name order.
**Feedback:** Core is right. Concretely: while `feat/repo-skeleton` was the default, the rule would have locked the branch we work on and left `main` unprotected. Clones and new PRs also point at the default branch.

## Interview angle
"I run my projects the way teams do: protected `main`, short-lived branches, a PR per change with CI, and decision records. I once lost uncommitted work to a `git reset --hard`, recovered it from git's object store, and now commit early on a branch."
