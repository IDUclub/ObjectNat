Noise Simulation & Noise Frame
==============================

**Noise Simulation** models how sound propagates from one or more **source points**,
taking into account **obstacles**, **vegetation**, and **environmental conditions**.
The outputs are **noise exposure maps** that are useful for **urban planning**,
**environmental impact assessments**, and **acoustic zoning**.

----

    The module provides several simulation methods depending on the required level of detail and computational performance.

    - **Full wave-based simulation** — physically detailed modeling of sound propagation.
    - **Simplified geometric frame** — faster approximate results for general assessments.

----

Full Wave-Based Simulation
--------------------------

Performs detailed noise modeling using **wave-based acoustic calculations**.
This approach accounts for reflections, diffractions, and absorption by materials.

.. autofunction:: objectnat.simulate_noise

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/noise_simulation_1point.png
   :alt: noise_simulation_1point
   :align: center
   :width: 80%

   Example of full wave-based simulation for a single noise source.

----

Simplified Noise Frame
----------------------

Generates a **simplified noise exposure map** using only **geometric visibility**
and **sound decay with distance**, without running a full wave simulation.
Ideal for **rapid assessments** or large-scale analyses where precision is less critical.

.. autofunction:: objectnat.calculate_simplified_noise_frame

.. figure:: https://raw.githubusercontent.com/DDonnyy/ObjectNat/assets/noise_frame.png
   :alt: noise_frame
   :align: center
   :width: 80%

   Simplified geometric noise exposure frame — fast and efficient.

----

Additional Resources
--------------------

For comprehensive documentation and advanced configuration options,
see the project Wiki:

`Noise Simulation on GitHub <https://github.com/DDonnyy/ObjectNat/wiki/Noise-simulation>`_

----

Example notebook
----------------

:doc:`examples/noise`