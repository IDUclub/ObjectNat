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
considering:

- **Distance or time thresholds**
- **Facility capacity**
- **Demand distribution**

.. autofunction:: objectnat.get_service_provision

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/service_provision_initial.png
   :alt: service_provision_initial
   :align: center
   :width: 80%

   Initial service provision analysis — demand–supply balance based on accessibility.

----

Recalculate Provision
---------------------

Allows recalculation of provision results with **new accessibility thresholds**
**without recomputing the full OD-matrix**, saving computation time.

.. autofunction:: objectnat.recalculate_links

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/service_provision_recalculated.png
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

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/service_provision_clipped.png
   :alt: service_provision_clipped
   :align: center
   :width: 80%

   Provision results clipped to a selected administrative boundary.

----

Example notebook
----------------

:doc:`examples/calculate_adjacency_matrix`
:doc:`examples/provision`