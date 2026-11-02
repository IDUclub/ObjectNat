Visibility Analysis
===================

**Visibility analysis** estimates which **buildings**, **barriers**, or **areas**
are visible from one or multiple observer points within a given distance.
It is commonly used in studies of **visual accessibility**, **urban perception**,
**noise propagation**, and **urban form analysis**.

----

    The module computes exact visibility from a single point or from a large batch of
    points in parallel with :func:`objectnat.get_visibility`, and aggregates the visibility
    of many observers into catchment zones with :func:`objectnat.get_visibility_catchment`.

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

Visibility Catchment Areas
--------------------------

:func:`objectnat.get_visibility_catchment` shows **where things can be seen from most often**. Observers are
placed densely, typically every 10–20 m along the streets with :func:`objectnat.get_observer_points`. The
area around them is split into square cells, and each cell gets the number of observers that see its centre
(their exact visibility polygons are accumulated in the Rust extension, without building them in Python), or
the sum of their weights when ``weight_column`` is given (e.g. pedestrian flow). Cells seen by nobody,
including cells inside buildings, are left out. The rest are grouped into classes and every class is
returned as one zone:

- ``classes=5`` (the default) builds classes from quantiles of the cell values; it adapts to the data, so
  class numbers are relative to one run;
- ``classes=[1, 10, 50, 100]`` sets the class lower bounds explicitly, in observer counts (or weight
  sums), so zones of different areas or scenarios share one scale.

Each zone carries ``visibility_class`` (``1`` is the least visible) and the range of cell values in it
(``count_min`` / ``count_max``, or ``weight_min`` / ``weight_max``). Zone outlines follow the cell grid, so
``cell_size`` sets their detail; time and memory grow with the number of cells. Visibility is symmetric, so
a cell's value is also the number of observers visible *from* it.

.. autofunction:: objectnat.get_visibility_catchment

.. autofunction:: objectnat.get_observer_points

.. figure:: https://github.com/user-attachments/assets/b5b0d4b3-a02f-4ade-8772-475703cd6435
   :alt: visibility-catchment-area
   :align: center
   :width: 80%

   Visibility catchment zones: places seen from many locations in an urban environment.

----

Example notebook
----------------

:doc:`examples/visibility`