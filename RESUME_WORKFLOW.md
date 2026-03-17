# Resume Workflow (FiKing-Trader)

Use this checklist whenever you reopen VS Code so your branch and context stay current.

1. Open the workspace file: `FiKing-Trader.code-workspace`
2. Pull latest branch state:
   - `git checkout reza`
   - `git pull origin reza`
3. Verify working tree:
   - `git status`
4. If you need previous temporary work, inspect stash first:
   - `git stash list`
   - `git stash show --name-status stash@{0}`
5. Before closing VS Code, commit or stash local changes:
   - `git add -A && git commit -m "wip"`
   - or `git stash push -u -m "wip-<date>"`

## Current branch sync note
- `reza` has been merged with latest `origin/Kian` and pushed.
- Merge commit: `3ce3ebe`.

## Why chat sometimes feels reset
Chat continuity depends on reopening the same workspace/session context. Opening another unrelated workspace can start a fresh context. Keeping this workspace file and this resume checklist helps avoid that.
