Upgrading to 3.0
================

ObjectNat 3.0 runs its computationally intensive parts in a compiled Rust engine, ``objectnat._rust``, and adds
2.5D and grid methods on top of it. No function is removed or renamed, but the parameters of the former Python
implementations are removed, some results differ and installing from source needs a Rust toolchain. Release 2.2.0
already has the new engine and still accepts the removed parameters with a warning, so this page applies to
upgrades from any 2.x release.

What changes for existing code
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Change
     - Effect
     - What to do
   * - Radius isochrones and coverage follow the network
     - With ``geometry_type="radius"`` every reachable node is buffered by its remaining budget, but by no more
       than ``off_network_distance`` (default 150 m), so the area no longer grows into a straight-line circle
       around the start
     - ``off_network_distance=None`` keeps the 2.x shape
   * - :func:`~objectnat.get_visibility` has one exact method
     - ``method``, ``resolution`` and ``parallel`` are removed: the visibility is always exact and computed in
       parallel threads; ``max_workers`` limits the threads
     - Remove these arguments; ``get_visibility(points, obstacles, 100)`` replaces
       ``get_visibility(points, obstacles, 100, "simple")``
   * - :func:`~objectnat.calculate_simplified_noise_frame` always runs the visibility in parallel threads
     - ``visibility_parallel`` is removed
     - Remove the argument; ``visibility_max_workers`` limits the threads
   * - Provision allocation runs in Rust
     - For a given ``seed`` the allocation differs from 2.1, but matches it statistically, and is the same on any
       number of cores; :func:`~objectnat.get_service_provision` also accepts a pandas sparse OD matrix
     - None needed: the model and the signature are unchanged
   * - Noise reflections are deterministic
     - :func:`~objectnat.simulate_noise` gives the same result on every run
     - None needed
   * - Installation
     - Wheels contain the compiled extension; building from source requires Rust; IduEdu 2.4.2 or newer
     - ``pip install -U objectnat`` installs a wheel, no compiler needed

What is new
-----------

- :doc:`methods/insolation`: direct sun hours on the ground (:func:`~objectnat.get_insolation`) and on facades
  floor by floor (:func:`~objectnat.get_facade_insolation`), among buildings and terrain, for a day or a period.
- :doc:`methods/visibility`: 2.5D viewsheds over building heights and terrain (:func:`~objectnat.get_viewshed`),
  isovist metrics in :func:`~objectnat.get_visibility`, isovist fields on a grid
  (:func:`~objectnat.get_isovist_grid`), catchments of many observers in 2D and 2.5D
  (:func:`~objectnat.get_visibility_catchment`, :func:`~objectnat.get_viewshed_catchment`) and observers along a
  pedestrian graph (:func:`~objectnat.get_observer_points`).
- :doc:`methods/noise`: :func:`~objectnat.simulate_noise_grid`, sound levels on a receiver grid around point, line
  and polygon sources with reflections and diffraction around building corners.
- :doc:`methods/rasters`: :func:`~objectnat.rasterize` and :class:`~objectnat.Raster` turn grid results into
  matplotlib plots, GeoTIFF files and web maps.
- Speed: exact visibility and provision allocation are many times faster (provision on the example data takes
  0.06 s instead of 2.6 s), and isochrones for many origins are faster with the same geometry.
