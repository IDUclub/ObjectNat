Isochrones
==========

Overview
--------

An **isochrone** is the area reachable from an origin within a travel-time or distance budget along a transport
network. ObjectNat builds isochrones on an ``iduedu.UrbanGraph`` prepared by `IduEdu
<https://github.com/IDUclub/IduEdu>`__ (walk, drive, public transport or intermodal graph).
:func:`objectnat.get_graph_isochrones` returns one polygon per origin; :func:`objectnat.get_stepped_graph_isochrones`
splits the reachable area into bands of width ``step``.

Method
------

1. Every origin is snapped to its nearest graph node, or node ids are taken from ``graph_node_column`` or
   ``origin_nodes``.
2. A Dijkstra search runs outward from the origins along the edge direction, with edge weights ``time_min``
   (minutes, default) or ``length_meter`` (metres), up to ``weight_value_cutoff``. The searches of all origins run in
   one parallel call of IduEdu's Numba routines.
3. The reachable nodes are converted into a polygon according to ``geometry_type``. The residual budget of a
   node (cutoff minus its cost) is converted to a distance, for ``time_min`` with the median edge speed of the graph
   (``length_meter / time_min``):

   - ``"radius"`` (default): union of buffers around the reachable nodes with their residual distance as radius;
   - ``"ways"``: reachable edges buffered by ``road_buffer_size`` metres, limited by the residual-radius buffers;
   - ``"separate"``: buffers of ``buffer_factor`` × residual distance around the reachable nodes, merged into one
     polygon.

4. The result is clipped to ``zone`` when it is given.

On walk and intermodal graphs only pedestrian edges (``type == "walk"``) shape ``"ways"``, and only their end nodes
shape any geometry: transit route nodes carry the search, but their cost includes the boarding wait and several
route nodes can share one stop location.

Stepped isochrones assign every reachable node to the interval ``[0, step), [step, 2·step), …`` that contains its
cost, up to ``weight_value_cutoff`` (by default the farthest reachable node), and build one geometry per interval:
``geometry_type=None`` uses Voronoi cells around the nodes, ``"radius"`` and ``"ways"`` clip these cells to the
residual-radius buffers or to the buffered streets, and ``"separate"`` builds independent buffers per interval.
With several origins, the bands describe their combined reachability: each node keeps the cost from its nearest
origin. The column ``dist`` holds the upper bound of the interval.

Outputs
-------

GeoDataFrames in the CRS of the input origins. :func:`objectnat.get_graph_isochrones` has one row per origin with
``source_node`` and the input attributes; :func:`objectnat.get_stepped_graph_isochrones` has one row per interval
and the column ``dist`` (minutes or metres).

.. autofunction:: objectnat.get_graph_isochrones

.. autofunction:: objectnat.get_stepped_graph_isochrones

Figures
-------

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/isochrone_ways_15_min.png
   :alt: Isochrone with geometry_type ways, 15 minutes
   :align: center
   :width: 70%

   15-minute isochrone on the intermodal graph of Vasilyevsky Island, ``geometry_type="ways"``.

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/isochrone_radius_15_min.png
   :alt: Isochrone with geometry_type radius, 15 minutes
   :align: center
   :width: 70%

   The same isochrone with ``geometry_type="radius"``.

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/isochrone_3points_radius_8_min.png
   :alt: Isochrones of three origins, 8 minutes
   :align: center
   :width: 70%

   8-minute isochrones of three origins, ``geometry_type="radius"``.

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/stepped_isochrone_ways_15_min.png
   :alt: Stepped isochrones, ways
   :align: center
   :width: 70%

   Stepped isochrones up to 15 minutes, ``geometry_type="ways"``.

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/stepped_isochrone_radius_15_min.png
   :alt: Stepped isochrones, radius
   :align: center
   :width: 70%

   Stepped isochrones up to 15 minutes, ``geometry_type="radius"``.

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/stepped_isochrone_separate_15_min.png
   :alt: Stepped isochrones, separate
   :align: center
   :width: 70%

   Stepped isochrones up to 15 minutes, ``geometry_type="separate"``.

Limitations
-----------

- Travel times are those stored in the graph (``time_min``); congestion, timetables beyond the waiting times
  modelled by IduEdu, and turn penalties are not considered.
- The polygon outside the network is an interpolation: ``"radius"`` and ``"separate"`` assume straight-line
  movement at the median edge speed for the residual budget, which overestimates reach across blocks without paths.
- Origins are snapped to the nearest graph node; the access distance from the origin to that node is not added.

Example notebook
----------------

:doc:`examples/isochrones`
