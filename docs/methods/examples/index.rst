Examples
========

Each notebook runs end to end on the example data in ``docs/methods/examples/examples_data/`` (buildings with
``storeys_count``, services with ``capacity``, vegetation polygons and a building-to-service travel-time matrix for
Vasilyevsky Island, Saint Petersburg). Notebooks that use a street graph build it with IduEdu from a Geofabrik
extract. Every notebook ends with a section *Your own area*, which shows what to replace to get the same results
for another territory.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - Content
   * - :doc:`isochrones`
     - Isochrones and stepped isochrones on an intermodal graph, all geometry types
   * - :doc:`coverage`
     - Coverage zones by time and distance, stepped coverage, graph-free radius coverage
   * - :doc:`calculate_od_matrix`
     - Building-to-service travel-time matrix with IduEdu, stored for the provision example
   * - :doc:`provision`
     - Service provision, recalculation with a shorter threshold, clipping, sparse matrices
   * - :doc:`visibility`
     - Exact 2D visibility polygons, isovist metrics and isovist fields, visibility catchment zones
   * - :doc:`viewshed`
     - 2.5D viewsheds over building heights, comparison with 2D visibility, viewshed catchment, terrain
   * - :doc:`noise`
     - Receiver-grid noise simulation, wave-front simulation and simplified noise frame
   * - :doc:`insolation`
     - Ground sun hours for a day and a period, terrain, facade insolation per floor with a norm check

.. toctree::
   :hidden:
   :maxdepth: 1

   isochrones
   coverage
   calculate_od_matrix
   provision
   visibility
   viewshed
   noise
   insolation
