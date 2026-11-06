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
builds networks itself. Python 3.11–3.14. Package manager: **uv** (lockfile: `uv.lock`, committed). Build
backend: maturin (Python package + Rust extension `objectnat._rust` from `rust/`).

## Commands

```bash
# Install all dependency groups
uv sync --all-groups            # or: make install-dev

# Run the test suite (first run downloads OSM data: Overpass, Geofabrik as fallback)
uv run pytest                   # or: make test

# Run all tests that do not need the network
uv run pytest -q -m "not network"

# Run a single test file / a single test by name
uv run pytest tests/test_coverage_zones.py -v
uv run pytest tests/test_isochrones.py -k "radius" -v

# Coverage
uv run pytest --cov=objectnat --cov-report=term-missing   # or: make coverage

# Format / lint
uv run isort objectnat tests    # or: make format
uv run black objectnat tests
uv run pylint objectnat         # or: make lint

# Build docs (English, then Russian into html/ru)    # or: make docs
uv run sphinx-build -b html docs docs/_build/html
uv run sphinx-build -b html -D language=ru -d docs/_build/doctrees-ru docs docs/_build/html/ru

# Russian translation of the docs: add new/changed text to the catalogues, then check nothing is untranslated
uv run python docs/i18n.py update                   # or: make docs-i18n
uv run python docs/i18n.py check                    # or: make docs-i18n-check (CI fails on untranslated text)

# Rust extension (built automatically by `uv sync`; needs a Rust toolchain): checks and unit tests
(cd rust && cargo fmt --check && cargo clippy --release --all-targets -- -D warnings && cargo test --release)

```

## Architecture

### Central input: IduEdu `UrbanGraph`

Every graph-based method takes an `iduedu.UrbanGraph` (not NetworkX). It stores the network as two GeoDataFrames —
`nodes_gdf` (point geometries, unique index = node id) and `edges_gdf` (columns `u`, `v`, `geometry`,
`length_meter`, `time_min`) — and exposes Numba-accelerated Dijkstra (`multi_source_dijkstra`,
`single_source_dijkstra`, `dijkstra_parallel` with `output="distances"`, `multi_source_dijkstra_nearest_source`).
ObjectNat calls these directly; it does not reimplement routing. Default weight: `time_min`; alternative: `length_meter`.

### Module layout

```
objectnat/
  __init__.py        — re-exports the public surface (config, _api.*, __version__)
  _api.py            — single import hub; all public symbols come from here
  _config.py         — global config (logger level, tqdm toggle)
  _version.py        — VERSION for objectnat.__version__ (semantic-release bumps it)
  methods/
    accessibility/   — isochrones + coverage on UrbanGraph
      isochrones.py  — get_graph_isochrones, get_stepped_graph_isochrones
      coverage.py    — get_graph_coverage, get_stepped_graph_coverage
      radius.py      — get_radius_coverage (graph-free: points + radius + Voronoi)
      _utils.py      — shared geometry builders (Voronoi cells, radius/ways/separate/stepped)
    provision/       — get_service_provision, recalculate_links, clip_provision, ProvisionResult
                       (provision_calculating.py + provision_result.py + exceptions; _matrix.py reads
                       dense or pandas-sparse OD matrices without densifying them)
    noise/           — simulate_noise_grid (Rust receiver grid), simulate_noise, calculate_simplified_noise_frame,
                       noise_reduce helpers
    insolation/      — get_insolation (sun hours on a ground grid), get_facade_insolation (per floor on facades);
                       buildings with heights + terrain, NOAA sun position, single days or periods
    visibility/      — get_visibility (exact visibility; method="simple" is a deprecated alias), get_isovist_grid
                       (isovist metrics on a grid), get_viewshed / get_viewshed_catchment (2.5D: lines of sight
                       over building heights and terrain), visibility catchment
    utils/
      geom_utils.py  — geometry helpers shared across methods
      scene.py       — 2.5D scene for insolation and viewsheds: building heights, terrain, local CRS
      raster.py      — Raster (values, transform, crs) and rasterize: grid points of insolation, noise, catchment
                       and isovist grids to rasters (matplotlib, GeoTIFF, folium); grid methods record their grid
                       in attrs; insolation and noise take output="zones" | "receivers" | "both"
rust/                — Rust crate compiled by maturin into the extension `objectnat._rust` (PyO3, abi3)
  src/visibility.rs  — exact batch visibility polygons (grid index, critical directions, rayon)
  src/provision.rs   — gravity-probabilistic provision allocation on sparse OD entries (rayon)
  src/insolation.rs  — 2.5D scene (walls with roof heights + terrain grid): sunlight rays, lines of sight (rayon)
  src/viewshed.rs    — footprint rasterization, per-observer visible cells and catchment counts (rayon)
  src/noise.rs       — receiver-grid noise: direct paths, first-order reflections and diffraction around one or
                       two building corners (Maekawa / ISO 9613-2), energy sum (rayon)
  src/lib.rs         — Python bindings (`visibility_polygons`, `provision_allocate`, `noise_grid_energy`)
```

### Rust extension

ObjectNat is built with maturin (`pyproject.toml` → `[tool.maturin]`): the Python package plus one compiled
module, `objectnat._rust`, from `rust/`. Python code converts GeoPandas/shapely data to numpy arrays and
calls it (see `objectnat/methods/visibility/_native.py`); the public API stays in Python and does not change.
There is no pure-Python fallback, so any machine building from source needs Rust. Versions in `pyproject.toml`, `rust/Cargo.toml`,
`objectnat/_version.py` and `CITATION.cff` are bumped together by semantic-release.

### Public accessibility API (all keyword-only after the graph)

- `get_graph_isochrones(urban_graph, *, weight_value_cutoff, gdf_origins=..., geometry_type="radius"|"ways"|"separate", ...)`
- `get_stepped_graph_isochrones(urban_graph, *, geometry_type=None|"radius"|"ways"|"separate", step=..., ...)`
- `get_graph_coverage(urban_graph, *, gdf_destinations=..., geometry_type=None|"radius"|"ways", ...)`
- `get_stepped_graph_coverage(urban_graph, *, geometry_type=None|"radius"|"ways"|"separate", step=..., ...)`

Coverage runs the search **reversed** (from destinations); isochrones run **outward** from origins. On
intermodal/walk graphs only pedestrian (`type == "walk"`) edges shape `geometry_type="ways"`, and only their end
nodes shape any geometry (`select_geometry_nodes` in `_utils.py`): transit route nodes carry the search but their
distance includes the boarding wait, and several of them can share one stop location.

### Coding conventions

- Line length: 120 characters (black + pylint).
- Imports sorted with isort (multi-line style 3, trailing comma); `isort` skips `__init__.py`.
- All public API must be reachable via `from objectnat import ...` through `_api.py`.
- Graph methods accept `iduedu.UrbanGraph`; do not add NetworkX-based graph handling to method code.

### Tests

`tests/` (flat, not a package). `conftest.py` builds a session-scoped intermodal `UrbanGraph` for OSM relation
1114252 via Overpass, falling back to the Geofabrik extract (`iduedu.download_osm_pbf`, cut file cached in
`tests/test_cache/geofabrik/`) when Overpass does not answer, and caches the graph as `tests/test_cache/*.urbangraph`.
Matplotlib runs on the `Agg` backend
(set in conftest). Example data (buildings/services/etc.) lives under `docs/methods/examples/examples_data/`.

### Documentation in two languages

English is the source; Russian lives in gettext catalogues `docs/locale/ru/LC_MESSAGES/<page>.po` (pages, notebook
markdown and the docstrings autodoc shows). `docs/conf.py` sets `locale_dirs`, one catalogue per page and the
language switch (`docs/_templates/sidebar/lang.html`: the Russian site is the `ru/` subdirectory of the English one).
**Whenever English text changes — a page, a notebook text cell, a public docstring — run `docs/i18n.py update`,
translate every empty or `#, fuzzy` entry (keep markup, roles, literals and links; decimal commas in Russian prose)
and finish with `docs/i18n.py check`.** Containers of nested lists (`Text:  - a - b`) stay identical to the
English; the update step keeps them so. `README_RU.rst` mirrors `README.rst`.

### Releases

The version lives in `pyproject.toml`, `rust/Cargo.toml`, `objectnat/_version.py` and `CITATION.cff`;
python-semantic-release bumps all four (and refreshes `uv.lock`) on push to `master` from Conventional Commits,
then the release workflow builds the sdist and abi3 wheels for Linux/macOS/Windows and publishes them — see
CONTRIBUTING.md. Never bump the version or tag by hand.
