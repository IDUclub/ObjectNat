# AGENTS.md

Guidance for any coding agent (Claude Code, Cursor, Copilot, Codex, Aider, Windsurf, etc.) working in this
repository. All commands below are plain shell/CLI and assume no specific agent or IDE.

## graphify (recommended for exploring this codebase)

[graphify](https://github.com/Graphify-Labs/graphify) turns this repository into a navigable knowledge graph
(god nodes, community structure, cross-file relationships) and is the recommended way to answer questions about
the codebase — prefer it over blind grep/file-walking.

The graph is **not committed** (`graphify-out/` is git-ignored); each contributor generates it locally.

Activate it:

```bash
pip install graphifyy          # or: uv tool install graphifyy
graphify .                     # build the graph into graphify-out/ (AST is free; docs use an LLM)
```

If your runtime exposes a `/graphify` command or a graphify skill, use that instead of the raw CLI.

Rules (once `graphify-out/graph.json` exists):

- For codebase questions, run `graphify query "<question>"`. Use `graphify path "<A>" "<B>"` for relationships and
  `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than
  `GRAPH_REPORT.md` or raw grep output.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review, or when query/path/explain do not surface
  enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

# Project guide

## Project Overview

**ObjectNat** is an open-source library for object-level geospatial analysis on city networks: accessibility
isochrones, coverage zones, service provision, visibility analysis, and noise simulation.

Graphs come from **[IduEdu](https://github.com/IDUclub/IduEdu)** (`UrbanGraph`); ObjectNat consumes them and never
builds networks itself. Python 3.11–3.12 only. Package manager: **uv** (lockfile: `uv.lock`, committed). Build
backend: hatchling.

## Commands

```bash
# Install all dependency groups
uv sync --all-groups            # or: make install-dev

# Run the test suite (downloads an OSM graph via Overpass on first run)
uv run pytest                   # or: make test

# Run a single test file / a single test by name
uv run pytest tests/test_coverage_zones.py -v
uv run pytest tests/test_isochrones.py -k "radius" -v

# Coverage
uv run pytest --cov=objectnat --cov-report=term-missing   # or: make coverage

# Format / lint
uv run isort objectnat tests    # or: make format
uv run black objectnat tests
uv run pylint objectnat         # or: make lint

# Build docs
uv run sphinx-build -b html docs docs/_build/html   # or: make docs
```

## Architecture

### Central input: IduEdu `UrbanGraph`

Every graph-based method takes an `iduedu.UrbanGraph` (not NetworkX). It stores the network as two GeoDataFrames —
`nodes_gdf` (point geometries, unique index = node id) and `edges_gdf` (columns `u`, `v`, `geometry`,
`length_meter`, `time_min`) — and exposes Numba-accelerated Dijkstra (`multi_source_dijkstra_path_length`,
`multi_source_dijkstra_nearest_source`, `dijkstra_path_length_parallel`). ObjectNat calls these directly; it does not
reimplement routing. Default weight: `time_min`; alternative: `length_meter`.

### Module layout

```
objectnat/
  __init__.py        — re-exports the public surface (config, _api.*, __version__)
  _api.py            — single import hub; all public symbols come from here
  _config.py         — global config (logger level, tqdm toggle)
  _version.py        — VERSION single source of truth (semantic-release owns it)
  methods/
    accessibility/   — isochrones + coverage on UrbanGraph
      isochrones.py  — get_graph_isochrones, get_stepped_graph_isochrones
      coverage.py    — get_graph_coverage, get_stepped_graph_coverage
      radius.py      — get_radius_coverage (graph-free: points + radius + Voronoi)
      _utils.py      — shared geometry builders (Voronoi cells, radius/ways/separate/stepped)
    provision/       — get_service_provision, recalculate_links, clip_provision, ProvisionResult
                       (provision_calculating.py + provision_result.py + exceptions)
    noise/           — simulate_noise, calculate_simplified_noise_frame, noise_reduce helpers
    visibility/      — get_visibility (accurate/simple methods)
    utils/
      geom_utils.py  — geometry helpers shared across methods
```

### Public accessibility API (all keyword-only after the graph)

- `get_graph_isochrones(urban_graph, *, weight_value_cutoff, gdf_origins=..., geometry_type="radius"|"ways"|"separate", ...)`
- `get_stepped_graph_isochrones(urban_graph, *, geometry_type=None|"radius"|"ways"|"separate", step=..., ...)`
- `get_graph_coverage(urban_graph, *, gdf_destinations=..., geometry_type=None|"radius"|"ways", ...)`
- `get_stepped_graph_coverage(urban_graph, *, geometry_type=None|"radius"|"ways"|"separate", step=..., ...)`

Coverage runs the search **reversed** (from destinations); isochrones run **outward** from origins. For
`geometry_type="ways"` on intermodal/walk graphs only pedestrian (`type == "walk"`) edges shape the geometry.

### Coding conventions

- Line length: 120 characters (black + pylint).
- Imports sorted with isort (multi-line style 3, trailing comma); `isort` skips `__init__.py`.
- All public API must be reachable via `from objectnat import ...` through `_api.py`.
- Graph methods accept `iduedu.UrbanGraph`; do not add NetworkX-based graph handling to method code.

### Tests

`tests/` (flat, not a package). `conftest.py` builds a session-scoped intermodal `UrbanGraph` for OSM relation
1114252 via Overpass and caches it as `tests/test_cache/*.urbangraph`. Matplotlib runs on the `Agg` backend
(set in conftest). Example data (buildings/services/etc.) lives under `docs/methods/examples/examples_data/`.

### Releases

Version lives only in `objectnat/_version.py`; `pyproject.toml` reads it dynamically. Releases are automated on
push to `master` via python-semantic-release from Conventional Commits — see CONTRIBUTING.md. Never bump the
version or tag by hand.
