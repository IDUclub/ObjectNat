# CHANGELOG


## v2.0.0 (2026-07-10)

### Features

- Objectnat 2.0 — UrbanGraph migration, structured provision, clustering removal
  ([#16](https://github.com/IDUclub/ObjectNat/pull/16),
  [`8d9ff49`](https://github.com/IDUclub/ObjectNat/commit/8d9ff4926875b27c32f47a11c5bc70d3274b498e))

* feat!: migrate graph methods to IduEdu 2.0 UrbanGraph; uv build; flat layout

Full ObjectNat 2.0 migration off NetworkX onto the IduEdu 2.0 UrbanGraph model, plus a build-system
  and repository overhaul aligned with IduEdu.

Accessibility (isochrones + coverage) - New objectnat/methods/accessibility/ package (coverage,
  isochrones, radius, shared _utils) built directly on iduedu.UrbanGraph and its Numba Dijkstra
  (multi_source / nearest_source / parallel), replacing the deleted methods/isochrones/ and
  methods/coverage_zones/ NetworkX code. - "ways" geometry uses pedestrian (type=="walk") edges only
  on intermodal/walk graphs so transit legs no longer distort the shape.

Build system & packaging - poetry -> uv + hatchling; version is dynamic from objectnat/_version.py.
  - PEP 621 [project] + PEP 735 [dependency-groups]; python-semantic-release. - CI split into
  quality.yml / release.yml / docs.yml (mirrors IduEdu), uv-based; test-image publishing to the
  assets branch preserved. - src/objectnat + src/tests -> flat objectnat/ + tests/ (tests no longer
  a package).

Cleanup - Removed objectnat.gdf_to_graph, dead graph utils (reverse_graph,
  remove_weakly_connected_nodes, get_closest_nodes_from_gdf) and math_utils; dropped the dead
  commented get_visibilities_from_points block in visibility. - Exported get_air_resist_ratio from
  objectnat.methods.noise. - Added AGENTS.md, CONTRIBUTING.md, CHANGELOG.md, .editorconfig,
  .gitattributes.

Tests validated on a live intermodal graph: accessibility 14/14 and noise 5/5 green under pandas
  3.0. Known issue: clustering (HDBSCAN cluster_selection_epsilon) fails on numpy 2.4 — upstream
  scikit-learn bug (#33219), tracked separately.

BREAKING CHANGE: graph-based methods now take an iduedu.UrbanGraph, not a networkx.Graph.
  get_accessibility_isochrones -> get_graph_isochrones and get_accessibility_isochrone_stepped ->
  get_stepped_graph_isochrones; isochrones return a single GeoDataFrame instead of (isochrones,
  pt_stops, pt_routes). Arguments are keyword-only and renamed: nx_graph -> urban_graph,
  points/point -> gdf_origins, gdf_to -> gdf_destinations, weight_value -> weight_value_cutoff,
  isochrone_type/step_type -> geometry_type. Removed objectnat.gdf_to_graph and the math_utils
  module. Requires iduedu>=2.0.0, pandas>=3.0, numpy>=2.4.

* feat!: return structured provision results and improve noise attenuation

Refactor service provision to return a ProvisionResult with sparse flow storage, separate
  demand/capacity summaries, and helper functions for materializing buildings, services, and link
  GeoDataFrames.

Split provision calculation and result formatting into dedicated modules, remove the old Provision
  model, export the new provision helpers, and update provision tests, docs, and example notebook
  for the new API.

Improve noise propagation through vegetation by using layered tree attenuation, robust signed-angle
  shadow sectors, and an optional source_position_buffer_r for sources located inside buildings or
  trees.

Add ObjectNat 2.0 migration planning notes and allow lock files to be tracked.

BREAKING CHANGE: get_service_provision now a returns ProvisionResult instead of a tuple of
  GeoDataFrames. recalculate_links now accepts and returns ProvisionResult.

* feat!: remove clustering and finalize ObjectNat 2.0 docs/tests

- remove point clustering from the public API and runtime dependencies - drop networkx and
  scikit-learn from project dependencies - add focused unit coverage for provision, noise, geom
  utils, and config - make provision allocation seed configurable and document ProvisionResult -
  update CI quality workflow with lint and Python 3.11/3.12 test matrix - refresh README, Sphinx
  docs, migration guide, and ObjectNat 2.0 examples - update notebooks for UrbanGraph, OD matrix
  parquet export, coverage joins, noise edges, and visibility API

### Breaking Changes

- Graph-based methods now take an iduedu.UrbanGraph, not a networkx.Graph.
  get_accessibility_isochrones -> get_graph_isochrones and get_accessibility_isochrone_stepped ->
  get_stepped_graph_isochrones; isochrones return a single GeoDataFrame instead of (isochrones,
  pt_stops, pt_routes). Arguments are keyword-only and renamed: nx_graph -> urban_graph,
  points/point -> gdf_origins, gdf_to -> gdf_destinations, weight_value -> weight_value_cutoff,
  isochrone_type/step_type -> geometry_type. Removed objectnat.gdf_to_graph and the math_utils
  module. Requires iduedu>=2.0.0, pandas>=3.0, numpy>=2.4.

- Get_service_provision now a returns ProvisionResult instead of a tuple of GeoDataFrames.
  recalculate_links now accepts and returns ProvisionResult.


## v1.4.1 (2025-11-27)

### Bug Fixes

- **get_visibility**: Added filter on obstacles geometry
  ([`3e671ab`](https://github.com/IDUclub/ObjectNat/commit/3e671abbda4f342774eb3ecd61d1d573a3ed775d))


## v1.4.0 (2025-11-26)

### Bug Fixes

- **0.5.1**: Pt data getter updated according to IduEdu 0.5.0
  ([`e451dde`](https://github.com/IDUclub/ObjectNat/commit/e451dde16f3950a374a445b7021979dae549ee2e))

- **coverage_zone**: Wrong filter
  ([`2066343`](https://github.com/IDUclub/ObjectNat/commit/20663438c7954676678c45c3c9f9f47726d508df))

- **invalid crs**: Reproject data to same crs
  ([`1f063b7`](https://github.com/IDUclub/ObjectNat/commit/1f063b7c0756ce8d06cd1effdfc81669ae9f32c3))

- **simplified_noise_frame**: - when no obstacles, point was providing to visibility not as gdf in
  utm crs, but expected in 4326
  ([`cd239cf`](https://github.com/IDUclub/ObjectNat/commit/cd239cfae77572349667c598c8ec7c6df7169d75))

### Chores

- _version.py changed
  ([`b9caa14`](https://github.com/IDUclub/ObjectNat/commit/b9caa1499b2d0c305fa2623305a0f71d3ac6dffe))

- Changed test images naming + added refs to new branch
  ([`93289e3`](https://github.com/IDUclub/ObjectNat/commit/93289e36e16b8bce5c4f58081cd25debfab48e2a))

- Readme_ru.md update
  ([`34b72ff`](https://github.com/IDUclub/ObjectNat/commit/34b72ffa06d9fcc9feef6f8e984f8d4787277ffe))

### Code Style

- Fix wrong code source in make and linted
  ([`541b8a3`](https://github.com/IDUclub/ObjectNat/commit/541b8a3f17939e86bd55784ec87fa8474a755a15))

### Continuous Integration

- Added file removing before commiting new in assets
  ([`a41cd04`](https://github.com/IDUclub/ObjectNat/commit/a41cd0476b85ce39314b1e270fd838a2ad1059d5))

- Added git removing
  ([`f9d3517`](https://github.com/IDUclub/ObjectNat/commit/f9d351782ddab31a0dfdff29aa36b97e786990cb))

- Added permission to write
  ([`9057f79`](https://github.com/IDUclub/ObjectNat/commit/9057f79bffee0b8901b5b75ba56db47b9edbb6ca))

- Assets orphan automation
  ([`23e3cc3`](https://github.com/IDUclub/ObjectNat/commit/23e3cc30ebd12e11c26888792d5456999c3462a9))

- Changed workflow activation branch
  ([`5790e0b`](https://github.com/IDUclub/ObjectNat/commit/5790e0bbdc379d28299f84313e492657abba7dce))

- Removed pyarrow install step
  ([`d551d82`](https://github.com/IDUclub/ObjectNat/commit/d551d82edd40b99e75188794166f487957a4c912))

- **deploy-docs**: - added new step for docs deployment
  ([`4fbc058`](https://github.com/IDUclub/ObjectNat/commit/4fbc05826e8915e145f44c5e472f9f376fbbf83f))

- **jobs united in one**: No more venv rebuilding
  ([`8f2132f`](https://github.com/IDUclub/ObjectNat/commit/8f2132f6648fcbdbf578297eba628dfa53527d0c))

### Documentation

- Readme links changed to iduclub by default
  ([`b834317`](https://github.com/IDUclub/ObjectNat/commit/b83431726a4f728389413e32c78d4031d99b68ca))

- Wrong ref to badges
  ([`5e6e601`](https://github.com/IDUclub/ObjectNat/commit/5e6e601502a6f3cfd525bf76c518f220e9d896e2))

- **init**: - refactored all docstrings to Google style
  ([`3896e61`](https://github.com/IDUclub/ObjectNat/commit/3896e612accb1050dd1525b884d902313f46f2b9))

- **readme**: Fixed links to docs
  ([`1e3930a`](https://github.com/IDUclub/ObjectNat/commit/1e3930a8bb7362d272cb21855d88b0db66410b13))

### Features

- **1.1.0**: - added pre commits
  ([`9252e1b`](https://github.com/IDUclub/ObjectNat/commit/9252e1b1c50a2ef0db4b82415cf6f82981759371))

- **coverage_zone**: - new metric instead old
  ([`1626820`](https://github.com/IDUclub/ObjectNat/commit/16268205a1e3ae573ae3a5b73e5c28bcaa727f88))

- **coverage_zone**: Added exception catches
  ([`3d8afba`](https://github.com/IDUclub/ObjectNat/commit/3d8afba9896a69b9a5223c1b969cd5871864df8a))

- **dev**: Stepped coverage zones
  ([`d584c65`](https://github.com/IDUclub/ObjectNat/commit/d584c65ddf841dab4a259a584eeb94f367ea5804))

- **gdf2graph+noise_frame**: - added efficient gdf2graph method for future work
  ([`1f22973`](https://github.com/IDUclub/ObjectNat/commit/1f22973558e17d013b60614f850a830ffc4dc6fd))

- **noise simulation**: - added ability to simulate with trees (only on 1st iter)
  ([`b5d0936`](https://github.com/IDUclub/ObjectNat/commit/b5d093688de23f667700e7f093b83f60fed2bce2))

- **noise simulation**: - added noise reduce from trees
  ([`366ceb3`](https://github.com/IDUclub/ObjectNat/commit/366ceb3a629d037f68aa45195e5295531810a024))

- **noise simulation**: - Reflections are done
  ([`1014649`](https://github.com/IDUclub/ObjectNat/commit/101464957836e53abf6f5cbbee5eba4105089fb4))

- **Noise simulation + lint + new example data**: - implemented parallel noise_sim.py
  ([`19ebddb`](https://github.com/IDUclub/ObjectNat/commit/19ebddb8325e69a4ec016a048efbdcc6683f0b55))

- **visibility_analysis**: - bugfix
  ([`9b808bb`](https://github.com/IDUclub/ObjectNat/commit/9b808bb9097dfbad74ff0123e513aff490b8234e))

- **visibility_analysis**: - fixed when point_from is on building wall
  ([`2d5fc70`](https://github.com/IDUclub/ObjectNat/commit/2d5fc70043bf22ad06f589cf369dce74f5ce9d72))

- **visibility_analysis**: - new geom utils
  ([`75e34c5`](https://github.com/IDUclub/ObjectNat/commit/75e34c5ed1c78c3f5432261cdf416c5cc7da4e75))

- **visibility_analysis**: - performance increasement
  ([`fb25e3f`](https://github.com/IDUclub/ObjectNat/commit/fb25e3f1aa4882f5a0ee6482556466be08369592))

### Refactoring

- **noise frame and etc**: - much increased the noise_frame performance
  ([`403d8de`](https://github.com/IDUclub/ObjectNat/commit/403d8de3b11c34a57fef3c8474394f50ed1ecf66))

- **visibility**: - merged two dif funcs in one with param `method`
  ([`2afd2fe`](https://github.com/IDUclub/ObjectNat/commit/2afd2fedb675022266d18694a9b50484ee0e63d9))

- **visibility**: Work in progress
  ([`0008e78`](https://github.com/IDUclub/ObjectNat/commit/0008e78f57796ca357caab9fb59aa5463f11f214))

### Testing

- Bug fixes and new tests
  ([`21d61a3`](https://github.com/IDUclub/ObjectNat/commit/21d61a30fb64d5cb9cebc81bc0bde0601d403638))
