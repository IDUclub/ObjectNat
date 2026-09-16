.. toctree::
    :hidden:
    :maxdepth: 1

    methods/isochrones
    methods/coverage
    methods/provision
    methods/visibility
    methods/noise
    methods/examples/index
    migration_1_to_2

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

.. |badge-ci| image:: https://github.com/IDUclub/ObjectNat/actions/workflows/quality.yml/badge.svg
   :target: https://github.com/IDUclub/ObjectNat/actions/workflows/quality.yml
   :alt: Tests and coverage

.. |badge-cov| image:: https://raw.githubusercontent.com/IDUclub/ObjectNat/python-coverage-comment-action-data/badge.svg
   :target: https://github.com/IDUclub/ObjectNat/tree/python-coverage-comment-action-data
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

**ObjectNat** extends urban network analysis with a focus on **object-level geospatial computation**.
It provides a unified set of tools for analyzing **coverage**, **provision**, **accessibility**,  
**visibility**, and **noise simulation** on city-scale geospatial data.

The library integrates seamlessly with:

- **GeoPandas** and **Shapely** for spatial operations;
- **IduEdu** ``UrbanGraph`` for graph preparation and multimodal routing;
- Python's scientific ecosystem (NumPy, Pandas, Matplotlib, etc).

----

Features
--------

- **Isochrones & Accessibility**

  - :func:`objectnat.get_graph_isochrones`, :func:`objectnat.get_stepped_graph_isochrones`
- **Coverage Zones**

  - :func:`objectnat.get_graph_coverage`, :func:`objectnat.get_radius_coverage`, :func:`objectnat.get_stepped_graph_coverage`
- **Service Provision**

  - :func:`objectnat.get_service_provision`, :func:`objectnat.get_provision_buildings`,
    :func:`objectnat.get_provision_services`, :func:`objectnat.get_provision_links`,
    :func:`objectnat.recalculate_links`, :func:`objectnat.clip_provision`
- **Noise Simulation**

  - :func:`objectnat.simulate_noise`, :func:`objectnat.calculate_simplified_noise_frame`
- **Visibility Analysis**

  - :func:`objectnat.get_visibility`
- **Utilities**

  - Geometry helpers used by accessibility, visibility, and noise workflows

----

Installation
------------

::

   pip install objectnat

Requires Python 3.11 or 3.12 and the standard geospatial stack (Pandas, GeoPandas,
Shapely, NumPy, SciPy). Graph-based accessibility methods consume ``UrbanGraph``
objects from IduEdu, which is installed with ObjectNat.

----

Quickstart
----------

To ensure optimal performance of ObjectNat's geospatial analysis functions, it's recommended
to utilize urban graphs sourced from the `IduEdu <https://pypi.org/project/iduedu/>`_ library.
**IduEdu** is an open-source Python library designed for the creation and manipulation of complex
city networks from OpenStreetMap and GTFS Schedule data.

.. code-block:: python

   # Install required packages (uncomment if needed)
   # !pip install iduedu objectnat

   import geopandas as gpd
   from shapely.geometry import Point

   from iduedu import get_4326_boundary, get_intermodal_graph
   from objectnat import get_stepped_graph_isochrones

   # Load boundary and build graph for a region (OSM ID 1114252)
   poly = get_4326_boundary(osm_id=1114252)
   graph = get_intermodal_graph(territory=poly, clip_by_territory=True)

   # Compute stepped accessibility isochrones from one or more origin points.
   origins = gpd.GeoDataFrame(
       geometry=[Point(30.3141, 59.9386)],
       crs=4326,
   )

   stepped_isochrones = get_stepped_graph_isochrones(
       graph,
       gdf_origins=origins,
       weight_type="time_min",
       weight_value_cutoff=10,
       geometry_type="separate",
       step=2,
   )

   stepped_isochrones.explore()

----

Contributions are very welcome!
Open an issue or PR on GitHub to suggest new features or improvements.

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

Citing ObjectNat
----------------

If you use ObjectNat in research, please cite it. Citation metadata is kept in
`CITATION.cff <https://github.com/IDUclub/ObjectNat/blob/master/CITATION.cff>`__; on GitHub,
**Cite this repository** in the repository sidebar copies it as APA or BibTeX.
Publications describing ObjectNat will be listed below.

----

Publications
------------

Coming soon…
