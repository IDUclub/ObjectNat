Insolation
==========

**Insolation analysis** shows how long places get direct sunlight during a day: which yards, playgrounds
and streets are sunny, and which stay in shadow all day long. It is used to check sanitary norms for
playgrounds and recreation areas, to compare building designs, and to choose places for greenery that
needs sun.

----

    :func:`objectnat.get_insolation` follows the sun through a chosen day or period and computes, on a
    regular grid of ground points, the hours of direct sunlight among buildings of known height and terrain.
    :func:`objectnat.get_facade_insolation` does the same for points on building facades, floor by floor.

----

Method
------

The position of the sun is computed for every ``time_step_min`` minutes of the day with the NOAA solar
calculator (accurate to about 0.01°, including atmospheric refraction) for the date and the location of
the area, so the sun's path follows the latitude and the season. For
every ground point and every sun position above ``min_sun_elevation`` a ray is cast towards the sun. The
point is sunlit when the ray passes above every building wall it crosses and above the terrain. The ray
is only followed while it is lower than the highest roof or hill of the scene, so the high midday sun
costs little. The calculation runs in ObjectNat's Rust extension, in parallel over points.

- **Buildings** are extruded footprints with flat roofs. Heights come from ``height_column``, from
  ``storeys_column`` × ``storey_height`` where the height is missing, or from ``default_height``. Include
  the buildings around the area too: their shadows reach into it.
- **Terrain** is optional: a GeoDataFrame of polygons with heights (rasterized with ``terrain_cell``
  cells), a GeoTIFF path (needs ``rasterio``) or ``(heights, transform, crs)``. Buildings stand on the
  terrain at their footprint, points lie on it. Hills shade the ground and the buildings behind them.
- **Periods**: ``date=("2026-04-22", "2026-08-22")`` simulates every ``day_step``-th day of the period
  (and its last day) with the scene built once.
- **Directions** are measured from true north; the difference between true and grid north of the
  projection (meridian convergence) is taken into account.

The result groups the points into classes of daily sun hours (``hour_breaks``, mean over the simulated
days). Class ``0`` is the ground that gets no direct sun on any simulated day. With
``return_receivers=True`` every point also carries ``min_sun_hours`` (the worst day) and
``max_continuous_hours``, the longest uninterrupted sunny period on the worst day, which is what
insolation norms usually limit (for example, SanPiN 1.2.3685-21 in Russia asks for continuous insolation of playgrounds). Use
``min_sun_elevation`` to leave out the low morning and evening sun the norms do not count.

.. autofunction:: objectnat.get_insolation

Facades and Windows
-------------------

Insolation norms for housing check the **windows** of living rooms: how long the sun shines on them without
a break on the calculation day. :func:`objectnat.get_facade_insolation` places calculation points along
every wall of the chosen buildings (outer and courtyard walls) every ``spacing`` metres, on every floor at
``window_height`` above the floor, just in front of the wall. The sun counts for a point only while it is
in front of that facade (and at least ``min_facade_angle`` away from the facade plane) and the ray towards
it is not blocked by other buildings or the terrain. Walls shared with a neighbouring building get no
points.

Every point carries the building, the floor, the direction the facade faces and the sun hours as on the
ground; with ``norm_hours`` it also says whether the continuous insolation meets the norm. Group the points
by building and floor to see which floors and facades fail it, or compare a scenario with and without a
planned building to see whose windows it darkens.

.. autofunction:: objectnat.get_facade_insolation

Limitations
-----------

- Roofs are flat; pitched roofs, balconies, loggias and trees are not modelled.
- Facade points stand on the outer wall surface; window reveals (the depth of the wall) are not modelled.
- Only direct sunlight is counted, not diffuse light or reflections.
