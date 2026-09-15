ObjectNat
=========

Object-oriented Network Analysis Tools
--------------------------------------

.. |badge-black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Стиль кода: black

.. |badge-pypi| image:: https://img.shields.io/pypi/v/objectnat.svg
   :target: https://pypi.org/project/objectnat/
   :alt: Версия PyPI

.. |badge-ci| image:: https://github.com/IDUclub/ObjectNat/actions/workflows/quality.yml/badge.svg
   :target: https://github.com/IDUclub/ObjectNat/actions/workflows/quality.yml
   :alt: Тесты и покрытие

.. |badge-coverage| image:: https://raw.githubusercontent.com/IDUclub/ObjectNat/python-coverage-comment-action-data/badge.svg
   :target: https://github.com/IDUclub/ObjectNat/tree/python-coverage-comment-action-data
   :alt: Покрытие тестами

.. |badge-license| image:: https://img.shields.io/badge/license-BSD--3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: Лицензия

.. |badge-docs| image:: https://img.shields.io/badge/docs-latest-4aa0d5?logo=readthedocs
   :target: https://iduclub.github.io/ObjectNat/
   :alt: Документация

|badge-black| |badge-pypi| |badge-ci| |badge-coverage| |badge-license| |badge-docs|

`README (English) <https://github.com/IDUclub/ObjectNat/blob/master/README.rst>`_

.. image:: https://raw.githubusercontent.com/IDUclub/ObjectNat/master/docs/_static/ONlogo.svg
   :align: center
   :width: 400
   :alt: ObjectNat logo

----

**ObjectNat** — это библиотека с открытым исходным кодом, разработанная командой **IDU**
для пространственного и сетевого анализа в городских исследованиях.
Библиотека предоставляет инструменты для анализа **доступности**, **видимости**,
**распространения шума** и **обеспеченности сервисами**.

----

Основные функции
----------------

Каждая функция сопровождается **примером в Jupyter Notebook** и **документацией**.

1. **Изохроны и транспортная доступность**

   Изохроны представляют собой области, достижимые из исходной точки за заданное время по транспортной сети.
   Эта функция позволяет анализировать транспортную доступность с использованием графов пешеходного, автомобильного,
   общественного транспорта или их комбинации, подготовленных в формате ``UrbanGraph`` через IduEdu.

   Библиотека поддерживает несколько методов построения изохрон:

   - **Базовые изохроны**: отображают одну зону, достижимую за заданное время.
   - **Шаговые изохроны**: делят зону доступности на интервалы времени (например, 3, 5, 10 минут).

   📘 `Пример <https://iduclub.github.io/ObjectNat/methods/examples/isochrones.html>`__
   🔗 `Документация <https://iduclub.github.io/ObjectNat/methods/isochrones.html>`__

2. **Зоны покрытия**

   Функция генерации **зон покрытия** от набора исходных точек с использованием транспортной сети. Вычисляет область,
   достижимую из каждой точки по **времени в пути** или **дистанции**, затем строит полигоны с помощью
   **диаграмм Вороного** и обрезает их по заданной границе, если она указана.

   📘 `Пример <https://iduclub.github.io/ObjectNat/methods/examples/coverage.html>`__
   🔗 `Документация <https://iduclub.github.io/ObjectNat/methods/coverage.html>`__

3. **Анализ обеспеченности сервисами**

   Функция оценки обеспеченности жилых зданий и их населения услугами (например, школы, поликлиники),
   которые имеют ограниченную **вместимость** и заданный **порог доступности** (в минутах или метрах).
   Функция моделирует **баланс спроса и предложения**, оценивая, насколько хорошо услуги удовлетворяют потребности
   близлежащих зданий в пределах допустимого времени.
   В ObjectNat 2.0 результат возвращается как структурированный ``ProvisionResult`` со sparse-матрицей потоков
   и helper-функциями для зданий, сервисов и геометрий связей.

   📘 `Пример <https://iduclub.github.io/ObjectNat/methods/examples/provision.html>`__
   🔗 `Документация <https://iduclub.github.io/ObjectNat/methods/provision.html>`__

4. **Анализ видимости**

   Функция оценки видимости от заданной точки или множества точек до близлежащих зданий в пределах заданного радиуса.
   Применяется для оценки визуальной доступности в городской среде. Единый API ``get_visibility`` поддерживает
   точный и упрощённый методы, а также параллельный расчёт для набора точек наблюдения.

   📘 `Пример <https://iduclub.github.io/ObjectNat/methods/examples/visibility.html>`__
   🔗 `Документация <https://iduclub.github.io/ObjectNat/methods/visibility.html>`__

5. **Моделирование шума**

   Симуляция распространения шума от источников с учётом **препятствий**, **растительности** и **факторов окружающей среды**.

   📘 `Пример <https://iduclub.github.io/ObjectNat/methods/examples/noise.html>`__
   🔗 `Документация <https://iduclub.github.io/ObjectNat/methods/noise.html>`__
   🧠 `Подробное описание <https://github.com/IDUclub/ObjectNat/wiki/Noise-simulation>`__

----

Городские графы с помощью *IduEdu*
----------------------------------

Для оптимальной работы **ObjectNat** рекомендуется использовать графы,
созданные библиотекой `IduEdu <https://github.com/IDUclub/IduEdu>`__.
Графовые методы ObjectNat принимают ``iduedu.UrbanGraph`` напрямую; ObjectNat больше не строит
NetworkX-графы внутри библиотеки.

**IduEdu** — это библиотека на Python с открытым исходным кодом, предназначенная для построения и обработки
сложных городских сетей на основе данных OpenStreetMap и расписаний GTFS Schedule.


**IduEdu** можно установить с помощью ``pip``::

    pip install IduEdu

Пример использования::

    from iduedu import get_4326_boundary, get_intermodal_graph

    poly = get_4326_boundary(osm_id=1114252)
    urban_graph = get_intermodal_graph(territory=poly, clip_by_territory=True)

----

Установка
---------

**ObjectNat** можно установить с помощью ``pip``::

    pip install ObjectNat

----

Конфигурация
------------

Настройте вывод логов и прогресс-бары через модуль конфигурации::

    from objectnat import config

    config.change_logger_lvl("INFO")   # отключить отладочные логи
    config.set_enable_tqdm(False)      # отключить прогресс-бары tqdm

Миграция на ObjectNat 2.0
-------------------------

ObjectNat 2.0 заменяет NetworkX-входы на ``iduedu.UrbanGraph``, переименовывает
функции доступности и возвращает ``ProvisionResult`` из анализа обеспеченности.
См. migration guide в документации:
`Migration 1.x to 2.0 <https://iduclub.github.io/ObjectNat/migration_1_to_2.html>`__.

----

Контакты
--------

- `НЦКР <https://actcognitive.org/>`_ — Национальный центр когнитивных исследований
- `ИДУ <https://idu.itmo.ru/>`_ — Институт дизайна и урбанистики
- `Наталья Чичкова <https://t.me/nancy_nat>`_ — менеджер проекта
- `Данила Олейников (Donny) <https://t.me/ddonny_dd>`_ — ведущий инженер-разработчик

----

Цитирование ObjectNat
---------------------

Если вы используете ObjectNat в исследованиях, пожалуйста, цитируйте библиотеку. Метаданные для цитирования
хранятся в `CITATION.cff <https://github.com/IDUclub/ObjectNat/blob/master/CITATION.cff>`__; на GitHub кнопка
**Cite this repository** на боковой панели репозитория копирует их в формате APA или BibTeX.
Публикации об ObjectNat будут перечислены ниже.

Публикации
----------

Скоро будут опубликованы.
