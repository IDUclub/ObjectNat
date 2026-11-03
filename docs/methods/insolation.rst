Insolation
==========

**Insolation analysis** shows how long places get direct sunlight during a day: which yards, playgrounds
and streets are sunny, and which stay in shadow all day long. It is used to check sanitary norms for
playgrounds and recreation areas, to compare building designs, and to choose places for greenery that
needs sun.

----

    :func:`objectnat.get_insolation` follows the sun through a chosen day and computes, on a regular grid
    of ground points, the hours of direct sunlight among buildings of known height and terrain.

----

Method
------

The position of the sun is computed for every ``time_step_min`` minutes of the day with the NOAA solar
calculator (accurate to about 0.01°, including atmospheric refraction) at the centre of the area. For
every ground point and every sun position above ``min_sun_elevation`` a ray is cast towards the sun. The
point is sunlit when the ray passes above every building wall it crosses and above the terrain. The ray
is only followed while it is lower than the highest roof or hill of the scene, so the high midday sun
costs little. The calculation runs in ObjectNat's Rust extension, in parallel over points.

- **Buildings** are extruded footprints with flat roofs. Heights come from ``height_column``, from
  ``storeys_column`` × ``storey_height`` where the height is missing, or from ``default_height``. Include
  the buildings around the area too: their shadows reach into it.
- **Terrain** is an optional height raster: a GeoTIFF path (needs ``rasterio``) or ``(heights, transform,
  crs)``. Buildings stand on the terrain at their footprint, points lie on it. Hills shade the ground and
  the buildings behind them.
- **Directions** are measured from true north; the difference between true and grid north of the
  projection (meridian convergence) is taken into account.

The result groups the points into classes of sun hours (``hour_breaks``). Class ``0`` is the ground that
gets no direct sun at all that day. With ``return_receivers=True`` every point also carries
``max_continuous_hours``, the longest uninterrupted sunny period, which is what insolation norms usually
limit (for example, SanPiN 1.2.3685-21 in Russia asks for continuous insolation of playgrounds). Use
``min_sun_elevation`` to leave out the low morning and evening sun the norms do not count.

.. autofunction:: objectnat.get_insolation

Limitations
-----------

- Roofs are flat; pitched roofs, balconies and trees are not modelled.
- Only direct sunlight is counted, not diffuse light or reflections.
- Insolation of windows and facades (points on walls at floor heights) is not available yet.
