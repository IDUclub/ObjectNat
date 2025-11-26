.. toctree::
    :hidden:
    :maxdepth: 1

    methods/isochrones
    methods/coverage
    methods/provision
    methods/visibility
    methods/noise
    methods/clustering
    methods/examples/index

ObjectNat
=========

Object-oriented Network Analysis Tools
--------------------------------------

**ObjectNat** — an open-source Python library for **object-oriented network analysis** and **spatial accessibility modeling**,  
developed by the **IDU team** at ITMO University.


|badge-black| |badge-pypi| |badge-ci| |badge-cov| |badge-license| |badge-github|

.. |badge-black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. |badge-pypi| image:: https://img.shields.io/pypi/v/objectnat.svg
   :target: https://pypi.org/project/objectnat/
   :alt: PyPI version

.. |badge-ci| image:: https://github.com/IDUclub/ObjectNat/actions/workflows/ci_pipeline.yml/badge.svg
   :target: https://github.com/IDUclub/ObjectNat/actions/workflows/ci_pipeline.yml
   :alt: CI

.. |badge-cov| image:: https://codecov.io/gh/DDonnyy/ObjectNat/graph/badge.svg?token=K6JFSJ02GU
   :target: https://codecov.io/gh/DDonnyy/ObjectNat
   :alt: Coverage

.. |badge-license| image:: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License

.. |badge-github| image:: https://img.shields.io/badge/GitHub-IDUclub%2FObjectNat-181717?logo=github
   :target: https://github.com/IDUclub/ObjectNat
   :alt: GitHub

----

Overview
--------

**ObjectNat** extends standard network analysis with a focus on **object-level geospatial computation**.  
It provides a unified set of tools for analyzing **coverage**, **provision**, **accessibility**,  
**visibility**, **noise simulation**, and **spatial clustering** — all based on graph connectivity.

The library integrates seamlessly with:
- **GeoPandas**, **Shapely**, and **NetworkX** for spatial operations;
- **IduEdu** for graph preparation and multimodal routing;
- Python's scientific ecosystem (NumPy, Pandas, Matplotlib, etc).

----

Features
--------

- **Isochrones & Accessibility**

  - :func:`objectnat.get_accessibility_isochrones`, :func:`objectnat.get_accessibility_isochrone_stepped`
- **Coverage Zones**

  - :func:`objectnat.get_graph_coverage`, :func:`objectnat.get_radius_coverage`, :func:`objectnat.get_stepped_graph_coverage`
- **Service Provision**

  - :func:`objectnat.get_service_provision` , :func:`objectnat.recalculate_links`, :func:`objectnat.clip_provision`
- **Noise Simulation**

  - :func:`objectnat.simulate_noise`, :func:`objectnat.calculate_simplified_noise_frame`
- **Visibility Analysis**

  - :func:`objectnat.get_visibility_accurate`, :func:`objectnat.get_visibility`, :func:`objectnat.get_visibilities_from_points`
- **Clustering**

  - :func:`objectnat.get_clusters_polygon`
- **Utilities**

  - Graph and geometry helpers: `geom_utils`, `graph_utils`, `math_utils`

----

Installation
------------

```
pip install objectnat
```

- > Requires Python 3.11+ and the standard geospatial stack
- > (Pandas, GeoPandas, Shapely, NetworkX, NumPy).

----

Quickstart
----------

To ensure optimal performance of ObjectNat's geospatial analysis functions, it's recommended
to utilize urban graphs sourced from the `IduEdu <https://pypi.org/project/iduedu/>`_ library.
**IduEdu** is an open-source Python library designed for the creation and manipulation of complex
city networks derived from OpenStreetMap data.

.. code-block:: python

   # Install required packages (uncomment if needed)
   # !pip install iduedu objectnat

   from iduedu import get_4326_boundary, get_intermodal_graph
   from objectnat import get_accessibility_isochrones

   # Load boundary and build graph for a region (OSM ID 1114252)
   poly = get_4326_boundary(osm_id=1114252)
   G = get_intermodal_graph(territory=poly, clip_by_territory=True)

   # Compute accessibility isochrones from several points
    stepped_isochrone, pt_stops, pt_routes = get_accessibility_isochrone_stepped(
        isochrone_type='separate',
        point=point,
        weight_type="time_min",
        weight_value=10,
        nx_graph=G_intermodal,
        step=2
    )

   stepped_isochrone.explore()

----

> Contributions are very welcome!  
> Open an issue or PR on GitHub to suggest new features or improvements.

----

Contacts
--------

- `NCCR <https://actcognitive.org/>`_ — National Center for Cognitive Research  
- `IDU <https://idu.itmo.ru/>`_ — Institute of Design and Urban Studies  
- `Natalya Chichkova <https://t.me/nancy_nat>`_ — Project Manager  
- `Danila Oleynikov (Donny) <https://t.me/ddonny_dd>`_ — Lead Software Engineer

----

License
-------

This project is open-source. See the :file:`LICENSE.txt` file for details.

----

Publications
------------

Coming soon…
