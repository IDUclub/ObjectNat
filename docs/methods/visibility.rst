Visibility Analysis
===================

**Visibility analysis** estimates which **buildings**, **barriers**, or **areas**
are visible from one or multiple observer points within a given distance.
It is commonly used in studies of **visual accessibility**, **urban perception**,
**noise propagation**, and **urban form analysis**.

----

    The module provides a unified interface for visibility computation through
    the :func:`objectnat.get_visibility` function. The user can compute exact
    visibility from a single point or from a large batch of points in parallel.

----

Method
------

:func:`objectnat.get_visibility` returns, for each observer, the exact area visible within its
viewing radius: the view disc minus everything hidden behind obstacle walls. Obstacles may be
polygons (with holes) or lines. The calculation runs in ObjectNat's compiled Rust extension
(``objectnat._rust``), multi-threaded over observers, and handles thousands of observers per second.

``method="simple"`` (a ray-based approximation) is deprecated: the exact algorithm is now faster than
it was, so both values of ``method`` return the exact result.

.. autofunction:: objectnat.get_visibility

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/visibility_example.png
   :alt: visibility_example
   :align: center
   :width: 80%

   Area visible from one point within 800 m among the example buildings.

----

Catchment Visibility from Multiple Points
-----------------------------------------

Performs visibility analysis for a **dense grid of observer points**,
producing combined **catchment visibility zones** — areas showing where specific
objects (e.g., landmarks, buildings) can be seen from.


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