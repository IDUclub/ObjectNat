Migration 1.x to 2.0
====================

ObjectNat 2.0 is a breaking release focused on the transition from NetworkX-based
graph handling to ``iduedu.UrbanGraph`` and on clearer public results for service
provision.

The sections below summarize the user-facing changes that affect notebooks,
scripts, and downstream applications.

Graph Model
-----------

Graph-based accessibility and coverage methods now consume
``iduedu.UrbanGraph``. ObjectNat no longer builds or mutates routing graphs
itself. Network structure comes from IduEdu through ``nodes_gdf``,
``edges_gdf``, and accelerated Dijkstra helpers.

Common changes:

- ``nx_graph`` arguments are replaced by the first positional ``urban_graph``
  argument.
- The default graph weight remains ``time_min``; ``length_meter`` is available
  for distance-based analysis.
- Old graph conversion helpers such as ``gdf_to_graph`` and ``graph_to_gdf``
  are removed from the public workflow.

Accessibility API
-----------------

Old accessibility names were replaced with graph-explicit names:

.. list-table::
   :header-rows: 1

   * - ObjectNat 1.x
     - ObjectNat 2.0
   * - ``get_accessibility_isochrones``
     - :func:`objectnat.get_graph_isochrones`
   * - ``get_accessibility_isochrone_stepped``
     - :func:`objectnat.get_stepped_graph_isochrones`
   * - ``get_graph_coverage`` with NetworkX input
     - :func:`objectnat.get_graph_coverage` with ``UrbanGraph`` input
   * - ``get_stepped_graph_coverage`` with NetworkX input
     - :func:`objectnat.get_stepped_graph_coverage` with ``UrbanGraph`` input

Isochrones return ``GeoDataFrame`` results directly. They no longer return
auxiliary public transport stop/route tables from the old NetworkX workflow.

Service Provision
-----------------

Service provision now returns :class:`objectnat.ProvisionResult` instead of a
tuple of materialized GeoDataFrames.

Typical ObjectNat 2.0 workflow:

.. code-block:: python

   from objectnat import (
       get_service_provision,
       get_provision_buildings,
       get_provision_services,
       get_provision_links,
   )

   result = get_service_provision(
       buildings=buildings,
       services=services,
       distance_matrix=distance_matrix,
       threshold=15,
   )

   buildings_with_metrics = get_provision_buildings(buildings, result)
   services_with_metrics = get_provision_services(services, result)
   links = get_provision_links(buildings, services, result)

``ProvisionResult.flow`` stores the sparse building-service allocation matrix.
``demand_rows`` and ``capacity_rows`` store tabular metrics that can be joined
back to source objects when needed.

Visibility API
--------------

``get_visibility_accurate`` and ``get_visibilities_from_points`` are replaced by :func:`objectnat.get_visibility`,
which accepts one or many observers and computes exact visibility polygons in the Rust extension, multi-threaded.
``method="simple"`` is deprecated: both values of ``method`` return the exact result, and ``resolution`` and
``parallel`` have no effect. Releases of the 2.x series added :func:`objectnat.get_visibility_catchment`,
:func:`objectnat.get_observer_points`, isovist metrics (``isovist_metrics=True``) and the 2.5D functions
:func:`objectnat.get_viewshed` and :func:`objectnat.get_viewshed_catchment`; see :doc:`methods/visibility`.

Removed in 2.0
--------------

The following APIs and dependencies have been removed in the 2.0 release:

- ``gdf_to_graph`` and ``graph_to_gdf`` graph conversion helpers.
- ``math_utils``.
- ``get_clusters_polygon`` together with the point clusterization docs/notebook.
- Direct ``networkx`` and ``scikit-learn`` dependencies (they are no longer
  runtime requirements of ObjectNat).

Noise
-----

:func:`objectnat.simulate_noise` (wave-front model for point sources) and
:func:`objectnat.calculate_simplified_noise_frame` (visibility masks and distance decay for point, line and polygon
sources) keep their signatures. Releases of the 2.x series added :func:`objectnat.simulate_noise_grid`, a
receiver-grid model with direct paths, first-order reflections and energy summation; see :doc:`methods/noise`.
