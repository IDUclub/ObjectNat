Visibility Analysis
===================

**Visibility analysis** estimates which **buildings** or **areas** are visible from
a given observer point (or set of points) within a specified distance.
It is useful for studying **visual accessibility**, **urban morphology**, and
**perceptual exposure** in public spaces.

----

    The module supports several modes of visibility computation — from precise
    raster-based modeling to fast vector-based approximations and multi-point visibility grids.

----

Accurate Method
----------------

Performs visibility analysis using **fine-grained raster-based algorithms**.
Provides high spatial accuracy but is more computationally intensive.
Best suited for detailed, local-scale visibility studies.

.. autofunction:: objectnat.get_visibility_accurate

----

Fast Approximate Method
-----------------------

Optimized for **large datasets** or **regional-scale studies**.
Uses geometric simplifications and **vector-based visibility** estimation,
providing fast yet informative results.

.. autofunction:: objectnat.get_visibility

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/visibility_comparison_methods.png
   :alt: visibility_comparison_methods
   :align: center
   :width: 80%

   Comparison between raster-based and vector-based visibility methods.

----

Catchment Visibility from Multiple Points
-----------------------------------------

Performs visibility analysis for a **dense grid of observer points**,
producing combined **catchment visibility zones** — areas showing where specific
objects (e.g., landmarks, buildings) can be seen from.

.. autofunction:: objectnat.get_visibilities_from_points

.. figure:: https://github.com/user-attachments/assets/b5b0d4b3-a02f-4ade-8772-475703cd6435
   :alt: visibility-catchment-area
   :align: center
   :width: 80%

   Example of visibility polygons aggregated into **visibility pools** —
   zones most visible from multiple locations in an urban environment.

----

Example notebook
----------------

:doc:`examples/visibility`