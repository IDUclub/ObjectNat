Noise Simulation & Noise Frame
==============================

**Noise Simulation** models how sound propagates from one or more **source points**,
taking into account **obstacles**, **vegetation**, and **environmental conditions**.
The outputs are **noise exposure maps** that are useful for **urban planning**,
**environmental impact assessments**, and **acoustic zoning**.

----

    The module provides several simulation methods depending on the required level of detail and computational performance.

    - **Receiver-grid simulation** — sound levels on a regular grid of receivers from point
      sources, with direct paths, first-order reflections, vegetation and energy summation;
      computed in ObjectNat's Rust extension.
    - **Full wave-based simulation** — detailed point-source modeling with reflections,
      obstacle absorption, and vegetation attenuation.
    - **Simplified geometric frame** — faster approximate results for point, line,
      and polygon sources using visibility masks and distance decay.

----

Receiver-Grid Simulation
------------------------

Computes the level at every receiver of a regular grid (``grid_step`` metres) and groups the
levels into bands of ``db_sim_step`` dB, returned in the same format as
:func:`objectnat.simulate_noise`. Sources may be points, lines (roads, railways) or polygons
(industrial sites, noisy buildings, radiating from their outline). For each source and receiver it
sums, as energies:

- the **direct path**, when no obstacle blocks the line of sight;
- **first-order reflections** from obstacle walls within ``reflection_max_dist`` (default 50 m) of the
  source (image-source method), with a loss of ``-10 lg(1 - a)`` for the wall's absorption ratio ``a``.

Every path loses ``20 lg r`` for geometric spreading, ``k r`` for air absorption (``k`` from
temperature and frequency) and, through vegetation, a frequency-dependent rate times the length of
foliage crossed. Lines are sampled every ``source_spacing`` metres with levels calibrated so that a long
line follows the line-source law ``L - 10 lg d - k d``. Two equal sources add +3 dB, unlike the band
maximum used by the other two functions. Ground effect, diffraction and higher-order reflections are
not modelled. With ``return_receivers=True`` the exact receiver levels are returned too.

Computation time grows with the number of receivers times the number of point and line samples in
reach; for large road networks use a coarser ``grid_step`` / ``source_spacing`` or
:func:`objectnat.calculate_simplified_noise_frame` for a quick estimate.

.. autofunction:: objectnat.simulate_noise_grid

----

Full Wave-Based Simulation
--------------------------

Performs detailed noise modeling from point sources using acoustic decay layers.
This approach accounts for reflections, obstacle absorption coefficients, and
vegetation attenuation. Source noise level and frequency can be provided either
as scalar arguments or through per-point ``source_noise_db`` and
``geometric_mean_freq_hz`` columns.

Useful configuration parameters include:

- ``absorb_ratio_column`` and ``standart_absorb_ratio`` for obstacle absorption.
- ``trees`` and ``tree_resolution`` for vegetation zones.
- ``source_position_buffer_r`` for sources that start inside an obstacle or tree
  polygon.
- ``use_parallel`` for process-based task distribution.

Vegetation attenuation is evaluated as layered sound reduction through the
shadow sector behind tree geometries. The model estimates how deeply the sound
path passes through vegetation and applies frequency-dependent reduction for
each distance layer.

.. autofunction:: objectnat.simulate_noise

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/noise_simulation_1point.png
   :alt: noise_simulation_1point
   :align: center
   :width: 80%

   Example of full wave-based simulation for a single noise source.

----

Simplified Noise Frame
----------------------

Generates a **simplified noise exposure map** using **geometric visibility**
and **sound decay with distance**, without running a full reflection simulation.
It accepts ``Point``, ``LineString``, and ``Polygon`` source geometries. Lines
and polygons are sampled into representative points for visibility calculation,
then recombined into source-level noise frames.

The simplified frame is ideal for **rapid assessments** or large-scale analyses
where precision is less critical. It does not model material absorption,
reflections, or detailed vegetation attenuation; obstacles are used as
line-of-sight masks.

.. autofunction:: objectnat.calculate_simplified_noise_frame

.. figure:: https://raw.githubusercontent.com/IDUclub/ObjectNat/assets/noise_frame.png
   :alt: noise_frame
   :align: center
   :width: 80%

   Simplified geometric noise exposure frame — fast and efficient.

----

Additional Resources
--------------------

The historical project wiki contains additional background on the noise model,
but the API reference on this page is the source of truth for ObjectNat 2.0:

`Noise Simulation on GitHub <https://github.com/IDUclub/ObjectNat/wiki/Noise-simulation>`_

----

Example notebook
----------------

:doc:`examples/noise`
