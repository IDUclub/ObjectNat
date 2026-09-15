ObjectNat
=========

Object-oriented Network Analysis Tools
--------------------------------------

.. |badge-black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. |badge-pypi| image:: https://img.shields.io/pypi/v/objectnat.svg
   :target: https://pypi.org/project/objectnat/
   :alt: PyPI version

.. |badge-ci| image:: https://github.com/IDUclub/ObjectNat/actions/workflows/quality.yml/badge.svg
   :target: https://github.com/IDUclub/ObjectNat/actions/workflows/quality.yml
   :alt: Tests and coverage

.. |badge-coverage| image:: https://raw.githubusercontent.com/IDUclub/ObjectNat/python-coverage-comment-action-data/badge.svg
   :target: https://github.com/IDUclub/ObjectNat/tree/python-coverage-comment-action-data
   :alt: Coverage

.. |badge-license| image:: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License

.. |badge-docs| image:: https://img.shields.io/badge/docs-latest-4aa0d5?logo=readthedocs
   :target: https://iduclub.github.io/ObjectNat/
   :alt: Docs

|badge-black| |badge-pypi| |badge-ci| |badge-coverage| |badge-license| |badge-docs|

`РИДМИ (Russian) <https://github.com/IDUclub/ObjectNat/blob/master/README_RU.rst>`__

.. image:: https://raw.githubusercontent.com/IDUclub/ObjectNat/master/docs/_static/ONlogo.svg
   :align: center
   :width: 400
   :alt: ObjectNat logo


**ObjectNat** is an open-source library developed by the **IDU** team
for spatial and network analysis in urban studies.
The library provides tools for analyzing **accessibility**, **visibility**,
**noise propagation**, and **service provision**.

----

Key Features
------------

Each feature includes a **Jupyter Notebook example** and **full documentation**.

1. **Isochrones and Transport Accessibility**  

   Isochrones represent areas reachable from an origin point within a specified time along a transport network.
   This feature allows the analysis of transport accessibility using pedestrian, road,
   public transport, or multimodal ``UrbanGraph`` objects prepared by IduEdu.

   The library supports several methods for building isochrones:

   - **Basic isochrones**: display a single zone reachable within a specified time.
   - **Step isochrones**: divide the accessibility area into time intervals (e.g., 3, 5, 10 minutes).


   📘 `Example <https://iduclub.github.io/ObjectNat/methods/examples/isochrones.html>`__
   🔗 `Documentation <https://iduclub.github.io/ObjectNat/methods/isochrones.html>`__

2. **Graph Coverage Zones from Points**

   A function for generating **coverage areas** from a set of origin points using a transport network.
   It computes the area reachable from each point by **travel time** or **distance**,
   then builds polygons using **Voronoi diagrams** and clips them by a given boundary if specified.

   📘 `Example <https://iduclub.github.io/ObjectNat/methods/examples/coverage.html>`__
   🔗 `Documentation <https://iduclub.github.io/ObjectNat/methods/coverage.html>`__

3. **Service Provision Analysis**  

   A function to evaluate how well residential buildings and their populations are provided
   with services (e.g., schools, clinics) that have limited **capacity**
   and a defined **accessibility threshold** (in minutes or meters).
   The function models the **balance between supply and demand**,
   assessing how well services meet the needs of nearby buildings within an acceptable time.
   ObjectNat 2.0 returns a structured ``ProvisionResult`` with a sparse flow matrix
   and materialization helpers for buildings, services, and link geometries.

   📘 `Example <https://iduclub.github.io/ObjectNat/methods/examples/provision.html>`__
   🔗 `Documentation <https://iduclub.github.io/ObjectNat/methods/provision.html>`__

4. **Visibility Analysis**  

   A function for evaluating visibility from a given point or set of points to nearby buildings within a given radius.
   It is used to assess visual accessibility in urban environments.
   The unified ``get_visibility`` API supports accurate and simplified methods,
   including parallel execution for batches of observer points.

   📘 `Example <https://iduclub.github.io/ObjectNat/methods/examples/visibility.html>`__
   🔗 `Documentation <https://iduclub.github.io/ObjectNat/methods/visibility.html>`__

5. **Noise Simulation & Noise Frame**

   Simulation of noise propagation from sources, taking into account **obstacles**, **vegetation**,
   and **environmental factors**.

   📘 `Example <https://iduclub.github.io/ObjectNat/methods/examples/noise.html>`__
   🔗 `Documentation <https://iduclub.github.io/ObjectNat/methods/noise.html>`__
   🧠 `Detailed theory <https://github.com/IDUclub/ObjectNat/wiki/Noise-simulation>`__

----

City Graphs via *IduEdu*
------------------------

For optimal performance, **ObjectNat** is recommended to be used with graphs
created by the `IduEdu <https://github.com/IDUclub/IduEdu>`_ library.
Graph-based ObjectNat methods consume ``iduedu.UrbanGraph`` directly; ObjectNat
does not build NetworkX graphs internally.

**IduEdu** is an open-source Python library designed for building and processing
complex urban networks from OpenStreetMap and GTFS Schedule data.


**IduEdu** can be installed via ``pip``::

    pip install IduEdu

Example usage::

    from iduedu import get_4326_boundary, get_intermodal_graph

    poly = get_4326_boundary(osm_id=1114252)
    urban_graph = get_intermodal_graph(territory=poly, clip_by_territory=True)

----

Installation
------------

**ObjectNat** can be installed via ``pip``::

    pip install ObjectNat

----

Configuration
-------------

You can adjust logging and progress bar output using the config module::

    from objectnat import config

    config.change_logger_lvl("INFO")   # mute debug logs
    config.set_enable_tqdm(False)      # disable tqdm progress bars

Migration to ObjectNat 2.0
--------------------------

ObjectNat 2.0 replaces NetworkX-based graph inputs with ``iduedu.UrbanGraph``,
renames accessibility functions, and returns ``ProvisionResult`` from service
provision. See the migration guide in the documentation:
`Migration 1.x to 2.0 <https://iduclub.github.io/ObjectNat/migration_1_to_2.html>`__.

----

Contacts
--------

- `NCCR <https://actcognitive.org/>`_ — National Center for Cognitive Research  
- `IDU <https://idu.itmo.ru/>`_ — Institute of Design and Urban Studies  
- `Natalya Chichkova <https://t.me/nancy_nat>`_ — Project Manager  
- `Danila Oleynikov (Donny) <https://t.me/ddonny_dd>`_ — Lead Software Engineer

----

Citing ObjectNat
----------------

If you use ObjectNat in research, please cite it. Citation metadata is kept in
`CITATION.cff <https://github.com/IDUclub/ObjectNat/blob/master/CITATION.cff>`__; on GitHub,
**Cite this repository** in the repository sidebar copies it as APA or BibTeX.
Publications describing ObjectNat will be listed below.

Publications
------------

Coming soon.
