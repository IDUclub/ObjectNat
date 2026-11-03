## What and why

<!-- What this PR changes and the reason. Link the issue: Closes #... -->

## Checks

- [ ] PR title follows Conventional Commits (`feat:`, `fix:`, `perf:`, `docs:` ...); it becomes the squash commit
- [ ] `make format-check` and `make lint` pass
- [ ] `uv run pytest -q -m "not network"` passes; new behaviour has a test
- [ ] `make rust-check` passes (if `rust/` changed)
- [ ] Docs and docstrings updated for public API changes
