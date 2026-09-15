# Contributing to ObjectNat

This guide covers the day-to-day workflow for humans and coding agents. The
short version: **write [Conventional Commits](https://www.conventionalcommits.org/),
open a PR, squash-merge it into the official `IDUclub/ObjectNat` `master` branch, and the release happens
automatically.**
Never bump the version or create a tag by hand.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (package manager; `uv.lock` is committed)
- `make` (optional but every command below has a `make` shortcut)

## Setup

```bash
uv sync --all-groups     # or: make install-dev
```

## Local development loop

| Task         | Command             | Notes                                             |
|--------------|---------------------|---------------------------------------------------|
| Run tests    | `make test`         | builds the OSM graph via Overpass (needs internet)|
| Coverage     | `make coverage`     | terminal report                                   |
| Coverage CI  | `make coverage-xml` | writes `coverage.xml`                              |
| Format       | `make format`       | isort + black                                      |
| Format check | `make format-check` | what CI enforces                                  |
| Lint         | `make lint`         | pylint                                            |
| Build docs   | `make docs`         | Sphinx → `docs/_build/html`                       |

Line length is 120. Imports are isort-sorted (`__init__.py` files are skipped).

The test suite downloads an intermodal graph from the Overpass API once and caches it under
`tests/test_cache/` (git-ignored) as an `.urbangraph` archive, so subsequent runs are fast.

## Commit messages (Conventional Commits)

Every commit — and, because we **squash-merge**, every **PR title** — must follow
Conventional Commits. The PR title becomes the squash commit on `master`, and that
message is what drives the next version.

```
<type>(<optional scope>): <description>

[optional body]

[optional footer, e.g. BREAKING CHANGE: ...]
```

The squash commit body is copied into the GitHub Release and `CHANGELOG.md`. python-semantic-release joins
consecutive lines of a paragraph into one, so separate every point with a blank line; a wrapped list without
blank lines becomes a single run-on paragraph.

### How the version is bumped

The next version is computed from the commit types since the last release tag:

| Commit                                                       | Example                                | Version effect              |
|--------------------------------------------------------------|----------------------------------------|-----------------------------|
| `fix:` / `perf:`                                             | `fix(noise): clamp air resistance`     | **patch** (`2.0.1 → 2.0.2`) |
| `feat:`                                                      | `feat(coverage): add ways geometry`    | **minor** (`2.0.1 → 2.1.0`) |
| `feat!:` or `BREAKING CHANGE:` footer                        | `feat(graph)!: require UrbanGraph`      | **major** (`2.0.1 → 3.0.0`) |
| `docs:` `test:` `chore:` `refactor:` `ci:` `style:` `build:` | `test(coverage): expand cases`         | **no release**              |

If a batch of merged commits contains several types, the **highest** bump wins.
A push to `master` that contains only no-release types produces no release at all.

Preview what would be released without changing anything:

```bash
make version-next     # prints the next version PSR would compute (no-op)
make version          # prints the current version
```

## Releases (automated — do not do this by hand)

In the official `IDUclub/ObjectNat` repository, every successful `Tests and Coverage` run for a push to
`master` starts `.github/workflows/release.yml`:

1. runs [python-semantic-release](https://python-semantic-release.readthedocs.io/),
   which computes the next version, updates `objectnat/_version.py` and
   `CHANGELOG.md`, commits `chore(release): vX.Y.Z [skip ci]`, tags `vX.Y.Z`, and
   creates the GitHub Release;
2. publishes the built distributions to PyPI via Trusted Publishing (OIDC);
3. lets `.github/workflows/docs.yml` rebuild and publish the official documentation to `gh-pages`.

Development forks may run tests and documentation previews, but they do not publish releases or deploy
the official documentation.

The version lives in a **single source of truth**, `objectnat/_version.py`
(`VERSION = "x.y.z"`); `pyproject.toml` reads it dynamically through hatchling.
There is no manual `pyproject.toml` version and no manual `git tag`. Do not edit
`objectnat/_version.py` by hand — the release automation owns it. `CHANGELOG.md` is updated in place: each
release is inserted below `<!-- version list -->` and earlier entries are left untouched, so a released entry
may be rewritten by hand when the generated text needs it.

## Pull request checklist

- [ ] PR title is a valid Conventional Commit (it becomes the squash commit).
- [ ] `make format-check` and `make test` pass locally.
- [ ] New behavior has tests.
- [ ] Public API changes are reachable via `from objectnat import ...`.
- [ ] Breaking changes use `!` / `BREAKING CHANGE:` and note the migration.
