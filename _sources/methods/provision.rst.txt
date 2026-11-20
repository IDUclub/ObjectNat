Service Provision
=================

Overview
--------

Service provision analysis allocates the **demand** of buildings (for example, the number of school-age children)
to **services** with limited **capacity** (school places), given a matrix of travel costs between them and a
normative **threshold** (minutes or metres). It shows which buildings are served within the threshold, which are
served only by services farther away, and which services are overloaded or underused.

:func:`objectnat.get_service_provision` computes the allocation and returns a :class:`objectnat.ProvisionResult`.
Helper functions join the metrics back to the buildings and services and build link geometries;
:func:`objectnat.recalculate_links` and :func:`objectnat.clip_provision` adjust a computed result.

Inputs
------

- ``buildings``: GeoDataFrame with a numeric demand column (``buildings_demand_column``, default ``"demand"``);
- ``services``: GeoDataFrame with a numeric capacity column (``services_capacity_column``, default ``"capacity"``);
- ``distance_matrix``: costs with rows ``buildings.index`` and columns ``services.index``, in the units of
  ``threshold``. It may be a dense DataFrame (``inf`` or ``NaN`` for unreachable pairs) or a pandas sparse
  DataFrame (``SparseDtype`` with ``NaN`` or ``inf`` as fill value) that stores only reachable pairs. Sparse
  matrices are read entry by entry without densifying, so large areas fit in memory. IduEdu's ``od_matrix``
  returns such a sparse matrix (see :doc:`examples/calculate_od_matrix`).

Method
------

The allocation is a gravity-based probabilistic model; it runs in ObjectNat's Rust extension over the entries of
the (sparse) matrix. Only building–service pairs with a finite cost ``d`` of at most ``3 * threshold`` can receive
demand. Each such pair has a **gravity weight** ``1 / (d + 1)²``: the closer the building, the larger the weight,
and a building twice as close gets on average about four times as much. The 1 is added so that a building at zero
cost, in the same place as the service, gets a finite weight. The allocation runs in rounds:

1. **Search radius.** The first round considers the pairs with a cost of up to about half the threshold,
   ``(threshold + 1) / 2``; every round widens this radius 1.5 times, so nearby buildings are served first.
2. **Offers.** Every service with free places takes the buildings with unmet demand inside the radius, keeps the
   ones with the largest weights (first the top tenth of them; the share grows as buildings and services drop
   out) and distributes its free places among them at random, in proportion to their weights.
3. **Acceptance.** A building that got more offers than it needs accepts its unmet demand, choosing among the
   offers at random in proportion to their size; the rest of the places stay free.
4. **Next round.** Buildings with no unmet demand and services with no free places drop out, and the rounds go on
   until nothing more can be allocated.

The random streams are seeded per object and iteration from ``seed``, so a fixed seed gives the same result
regardless of the number of CPU cores. Different seeds give alternative allocations with the same model.

Outputs
-------

:class:`objectnat.ProvisionResult` contains

- ``flow``: sparse building × service matrix of allocated demand;
- ``demand_rows``: per building ``demand``, ``demand_left``, ``supplied_demands_within`` and
  ``supplied_demands_without`` (allocated from services within / beyond ``threshold``), ``min_dist`` (cost to the
  nearest service), ``avg_dist`` (flow-weighted mean cost) and ``provision_value`` =
  ``supplied_demands_within / demand``;
- ``capacity_rows``: per service ``capacity``, ``capacity_left``, ``carried_capacity_within``,
  ``carried_capacity_without`` and ``service_load`` (total allocated demand);
- ``distance_matrix`` and ``threshold`` used for the calculation.

:func:`objectnat.get_provision_buildings` and :func:`objectnat.get_provision_services` join these metrics to copies
of the input GeoDataFrames; :func:`objectnat.get_provision_links` returns one straight line per non-zero flow, from the
representative point of the building to the service, with ``building_index``, ``service_index``, ``demand``
(allocated units) and ``distance`` (cost).

.. autofunction:: objectnat.get_service_provision

.. autoclass:: objectnat.ProvisionResult
   :members:

.. autofunction:: objectnat.get_provision_buildings

.. autofunction:: objectnat.get_provision_services

.. autofunction:: objectnat.get_provision_links

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/provision_map_value_v4.png
   :alt: Provision value of residential buildings
   :align: center
   :width: 100%

   Provision value of residential buildings for the example schools with a 10-minute threshold on the intermodal
   travel-time matrix (``seed=0``); dots are schools. Non-residential buildings and streets are
   grey; the black line is the district boundary.

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/provision_map_links_v4.png
   :alt: Provision with building–school links
   :align: center
   :width: 100%

   The same result with :func:`objectnat.get_provision_links`: one line per non-zero flow, width proportional to
   the allocated demand. Dark links lie within the threshold, light ones beyond it (up to ``3 * threshold``); the
   light links carry the demand that the schools within 10 minutes cannot accommodate.

Recalculation and clipping
--------------------------

:func:`objectnat.recalculate_links` removes the flows whose cost exceeds ``new_max_dist`` and rebuilds the
aggregate metrics without a new allocation. The removed demand is not redistributed to other services; run
:func:`objectnat.get_service_provision` again for a full reallocation.

:func:`objectnat.clip_provision` keeps the buildings that intersect ``selection_zone``, the links of these buildings
and the services the links refer to. All four inputs must share one CRS.

.. autofunction:: objectnat.recalculate_links

.. autofunction:: objectnat.clip_provision

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/provision_map_recalculated_v4.png
   :alt: Provision after recalculate_links
   :align: center
   :width: 100%

   ``recalculate_links(result, 10)``: the links longer than 10 minutes are removed and the metrics are rebuilt;
   their demand stays unallocated.

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/provision_map_clipped_v4.png
   :alt: Provision clipped to a selection zone
   :align: center
   :width: 100%

   :func:`objectnat.clip_provision` with a 450 m circle: the buildings in the zone, all their links and the schools
   these links lead to, including schools outside the zone.

Limitations
-----------

- The allocation is stochastic: individual flows depend on ``seed``; aggregate metrics vary little between seeds
  for large inputs but can vary for small ones.
- Demand and capacity are integers (allocated in units); fractional values are truncated.
- The model has no explicit choice behaviour (quality, preference, catchment rules); every reachable service
  within ``3 * threshold`` can receive demand.
- The result is only as accurate as the cost matrix; the example matrix contains travel times in minutes on an
  intermodal graph.

Example notebooks
-----------------

:doc:`examples/calculate_od_matrix` (cost matrix with IduEdu) and :doc:`examples/provision`.
