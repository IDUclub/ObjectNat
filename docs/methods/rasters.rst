Rasters
=======

Overview
--------

Several methods compute values on a regular grid of square cells and return the grid as points:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - Grid points
     - Columns
   * - :func:`objectnat.get_insolation`
     - ``output="receivers"`` or ``"both"``
     - ``sun_hours``, ``min_sun_hours``, ``max_continuous_hours``
   * - :func:`objectnat.simulate_noise_grid`
     - ``output="receivers"`` or ``"both"``
     - ``noise_level``
   * - :func:`objectnat.get_visibility_catchment`, :func:`objectnat.get_viewshed_catchment`
     - ``output="cells"`` or ``"both"``
     - ``count`` (``weight``), ``visibility_class``
   * - :func:`objectnat.get_isovist_grid`
     - the result
     - isovist metrics (``view_share``, ``compactness``, ``occlusivity``, ``drift``, ...)

``output="receivers"`` (``"cells"``) skips the zone polygons, which is faster when only the values are needed:
for a single day of insolation the polygons take most of the run time.

:func:`objectnat.rasterize` turns one column of such points into a :class:`objectnat.Raster`: a 2-D array with an
affine transform and a CRS. A raster shows a continuous quantity cell by cell, without the class bounds of the zone
polygons, and it can be drawn with matplotlib, written to a GeoTIFF file or put on a folium web map.

Method
------

The grid methods record their grid in the ``attrs`` of the returned points: the local metric CRS of the
calculation (a UTM zone), the cell size and the position of one cell centre. :func:`objectnat.rasterize` reprojects
the points to that CRS, puts every point into its cell and fills cells without a point (buildings, cells below
``target_noise_db``, cells seen by nobody) with ``NaN``. The raster reproduces the calculation grid exactly, in
whatever CRS the points were returned, and also after rows were filtered. Points without the record, for example
read from a file, need ``cell_size``; they are then placed in their own projected CRS or in an estimated UTM zone.

:class:`objectnat.Raster` is the tuple ``(values, transform, crs)``. ``values`` has the northern row first;
``transform`` holds the affine coefficients ``(a, b, c, d, e, f)`` in the order of rasterio and GDAL. It is also
the format of ``terrain`` arguments, so a raster can be passed back as terrain.

- ``plot(ax, **imshow_kwargs)`` draws it with ``imshow`` in the raster CRS. Other layers on the same axes must be
  reprojected to ``raster.crs``.
- ``to_geotiff(path)`` writes a float32 GeoTIFF (requires ``rasterio``).
- ``to_folium(cmap=..., vmin=..., vmax=..., opacity=...)`` returns a ``folium.raster_layers.ImageOverlay``. Web
  maps draw images in Web Mercator, so the grid is resampled to a Mercator grid with about the same number of cells
  (nearest cell) and coloured with a matplotlib colormap; cells without a value are transparent. The overlay is
  added to any folium map, including the maps of ``GeoDataFrame.explore()`` (requires ``folium`` and
  ``matplotlib``).
- ``explore(m=None, cmap=..., vmin=..., vmax=..., name=...)`` does the same on a new map (or on ``m``) and adds a
  colour bar, like ``GeoDataFrame.explore()``.

.. code-block:: python

   from objectnat import get_insolation, rasterize

   receivers = get_insolation(buildings, "2026-03-21", area=area, output="receivers")
   sun = rasterize(receivers, "sun_hours")

   image = sun.plot(cmap="inferno", vmin=0, vmax=14)          # matplotlib
   sun.to_geotiff("sun_hours_2026-03-21.tif")                 # GIS
   sun.explore(cmap="inferno", vmin=0, vmax=14, name="sun hours")   # web map with a colour bar
   m = buildings.explore(color="grey")                               # or on top of another map
   sun.explore(m, cmap="inferno", vmin=0, vmax=14, name="sun hours")

.. autofunction:: objectnat.rasterize

.. autoclass:: objectnat.Raster
   :members: cell_size, bounds, extent, plot, explore, to_folium, to_geotiff

Limitations
-----------

- A raster has the resolution of the calculation grid; finer rasters need a smaller ``grid_step`` or
  ``cell_size`` in the calculation itself.
- ``to_folium`` embeds the image in the map as a PNG, so very large rasters make heavy HTML files; for large areas
  write a GeoTIFF and serve it as tiles.
- Points without grid ``attrs`` must lie on a regular grid of the given ``cell_size``.

Example notebooks
-----------------

:doc:`examples/insolation` (sun hours as rasters and on a web map) and :doc:`examples/visibility` (isovist field).
