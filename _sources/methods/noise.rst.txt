Noise Simulation & Noise Frame
==============================

**Noise Simulation** models how sound propagates from one or more **source points**,
taking into account **obstacles**, **vegetation**, and **environmental conditions**.
The outputs are **noise exposure maps** that are useful for **urban planning**,
**environmental impact assessments**, and **acoustic zoning**.

----

    The module provides several simulation methods depending on the required level of detail and computational performance.

    - **Full wave-based simulation** — detailed point-source modeling with reflections,
      obstacle absorption, and vegetation attenuation.
    - **Simplified geometric frame** — faster approximate results for point, line,
      and polygon sources using visibility masks and distance decay.

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
