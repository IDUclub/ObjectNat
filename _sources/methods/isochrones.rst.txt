Isochrones and Transport Accessibility
======================================

Isochrones represent areas reachable from a starting point within a given time limit
along a transport network.
This functionality enables analysis of transport accessibility using **pedestrian**, **automobile**, **public transport** graphs, or their combination.

----

    The library provides several methods for generating isochrones depending on the required level of detail and visualization.

----

Baseline Isochrones
-------------------

Show a single area reachable within a specified time.

.. autofunction:: objectnat.get_accessibility_isochrones

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/isochrone_ways_15_min.png
   :alt: isochrone_ways_15_min
   :align: center
   :width: 80%

   Isochrone for **road network** within 15 minutes.

----

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/isochrone_radius_15_min.png
   :alt: isochrone_radius_15_min
   :align: center
   :width: 80%

   Isochrone using **radius-based** method (15 min).

----

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/isochrone_3points_radius_8_min.png
   :alt: isochrone_3points_radius_8_min
   :align: center
   :width: 80%

   Isochrones for **three start points** (8 min).

----

Stepped Isochrones
------------------

Show accessibility ranges divided into time intervals (e.g., 5, 10, 15 minutes).

.. autofunction:: objectnat.get_accessibility_isochrone_stepped

----

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/stepped_isochrone_ways_15_min.png
   :alt: stepped_isochrone_ways_15_min
   :align: center
   :width: 80%

   Stepped isochrones for **road network** (5–15 min).

----

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/stepped_isochrone_radius_15_min.png
   :alt: stepped_isochrone_radius_15_min
   :align: center
   :width: 80%

   Stepped **radius-based** isochrones (5–15 min).

----

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/stepped_isochrone_separate_15_min.png
   :alt: stepped_isochrone_separate_15_min
   :align: center
   :width: 80%

   **Separate stepped zones** visualized per time interval.

----

Example notebook
----------------

:doc:`examples/isochrones`