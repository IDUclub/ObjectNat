Point Clusterization
====================

Clusterization groups nearby points into polygons based on spatial proximity and minimum cluster size.  
It can be used for identifying dense urban areas, service hubs, or catchment zones.

----

**Cluster Generation**
~~~~~~~~~~~~~~~~~~~~~~

Clusters are generated using spatial rules:

- **Minimum distance** between points to be included in the same cluster.
- **Minimum number of points** required to form a valid cluster.

.. currentmodule:: objectnat

.. autofunction:: get_clusters_polygon

.. image:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/building_clusters.png
   :alt: building_clusters
   :align: center

----

Example notebook
----------------

:doc:`examples/clustering`