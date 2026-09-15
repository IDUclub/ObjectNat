Service Provision Analysis
==========================

This module evaluates how well **services** (e.g., schools, clinics, shops)
cover **residential buildings** based on their **capacity** and **accessibility**.
It models **demand–supply relationships** and provides tools to analyze, visualize,
and adjust service coverage.

----

    Service provision analysis helps estimate how effectively urban infrastructure
    meets population needs.

----

Evaluate Initial Provision
--------------------------

Calculates **provision scores** between population points and service facilities
and returns a :class:`objectnat.ProvisionResult`. The result keeps the sparse
building-service allocation matrix together with per-building and per-service
summary tables. This separates the calculation step from the materialization of
GeoDataFrames for mapping or export.

The calculation considers:

- **Distance or time thresholds**
- **Facility capacity**
- **Demand distribution**

.. autofunction:: objectnat.get_service_provision

Provision Result
----------------

``ProvisionResult`` is the central output of service provision analysis. It is
designed to be lightweight enough for recalculation and explicit enough for
downstream joins.

.. autoclass:: objectnat.ProvisionResult
   :members:

The main fields are:

- ``flow`` — sparse building-service matrix. Rows match building indices,
  columns match service indices, and non-zero values are allocated demand.
- ``demand_rows`` — per-building metrics, including original demand, remaining
  demand, supplied demand inside/outside the normative threshold, minimum
  distance, average weighted distance, and provision value.
- ``capacity_rows`` — per-service metrics, including capacity, remaining
  capacity, carried capacity inside/outside the threshold, and total service
  load.
- ``distance_matrix`` — aligned cost matrix used for the calculation.
- ``threshold`` — normative distance or time threshold used to classify
  within-threshold and outside-threshold flows.

Materialize GeoDataFrames
-------------------------

Use the helper functions below to join calculated provision metrics back to
source objects and to create link geometries between buildings and services.

.. autofunction:: objectnat.get_provision_buildings
.. autofunction:: objectnat.get_provision_services
.. autofunction:: objectnat.get_provision_links

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/service_provision_initial.png
   :alt: service_provision_initial
   :align: center
   :width: 80%

   Initial service provision analysis — demand–supply balance based on accessibility.

----

Recalculate Provision
---------------------

Allows recalculation of provision results after tightening the maximum allowed
link distance **without recomputing the full OD-matrix**. Existing flows whose
cost exceeds ``new_max_dist`` are removed, and aggregate metrics are rebuilt.
Removed demand is not redistributed to other services; run
:func:`objectnat.get_service_provision` again when a full reallocation is needed.

.. autofunction:: objectnat.recalculate_links

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/service_provision_recalculated.png
   :alt: service_provision_recalculated
   :align: center
   :width: 80%

   Recalculated provision results using adjusted travel-time thresholds.

----

Clip to Analysis Area
---------------------

Restricts provision outputs to a given **geographic boundary**
(e.g., administrative region, neighborhood, planning area).

.. autofunction:: objectnat.clip_provision

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/service_provision_clipped.png
   :alt: service_provision_clipped
   :align: center
   :width: 80%

   Provision results clipped to a selected administrative boundary.

----

Example notebook
----------------

:doc:`examples/calculate_od_matrix`
:doc:`examples/provision`
