import os
from contextlib import nullcontext

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
import shapely
from iduedu import UrbanGraph, config, get_4326_boundary, get_intermodal_graph
from shapely import Point

# Tests only render plots to PNG files; force the non-interactive backend so the
# suite never needs a display/Tk (headless CI, Windows without Tk, etc.).
matplotlib.use("Agg")

logger = config.logger

path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../docs/methods/examples/examples_data/")
output_dir = os.path.join(os.path.dirname(__file__), "test_output")
cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


@pytest.fixture(scope="session")
def buildings_data():
    data_path = os.path.join(path_to_data, "buildings.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")
    buildings_data = gpd.read_parquet(data_path)
    buildings_data.index = buildings_data.index.astype(int)
    return buildings_data


@pytest.fixture(scope="session")
def services_data():
    data_path = os.path.join(path_to_data, "services.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")
    services_data = gpd.read_parquet(data_path)
    services_data.index = services_data.index.astype(int)
    return services_data


@pytest.fixture(scope="session")
def matrix_time_data():
    data_path = os.path.join(path_to_data, "matrix_time.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")
    matrix_time_data = pd.read_parquet(data_path)
    matrix_time_data.index = matrix_time_data.index.astype(int)
    matrix_time_data.columns = matrix_time_data.columns.astype(int)
    return matrix_time_data


@pytest.fixture(scope="session")
def trees_data():
    data_path = os.path.join(path_to_data, "trees.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")
    return gpd.read_parquet(data_path)


TERRITORY_OSM_ID = 1114252  # Vasileostrovsky district, Saint Petersburg
GEOFABRIK_REGION = "northwestern-fed-district"
# order in which OSM sources are tried; e.g. OBJECTNAT_TEST_OSM_SOURCES=overpass,pbf
OSM_SOURCES = tuple(s.strip() for s in os.getenv("OBJECTNAT_TEST_OSM_SOURCES", "pbf,overpass").split(",") if s.strip())


def _from_geofabrik():
    """Read OSM data from the test territory cut out of the Geofabrik extract (cached in test_cache/geofabrik)."""
    from iduedu import download_osm_pbf  # pylint: disable=import-outside-toplevel

    path = download_osm_pbf(
        osm_id=TERRITORY_OSM_ID, region=GEOFABRIK_REGION, out_dir=os.path.join(cache_dir, "geofabrik")
    )
    return config.use_pbf(path)


_OSM_SOURCES = {"pbf": _from_geofabrik, "overpass": nullcontext}


def _build_territory() -> tuple:
    """Boundary and intermodal graph of the test territory from the first OSM source that works."""
    errors = []
    for name in OSM_SOURCES:
        if name not in _OSM_SOURCES:
            raise ValueError(f"Unknown OSM source {name!r} in OBJECTNAT_TEST_OSM_SOURCES; use pbf and/or overpass")
        try:
            with _OSM_SOURCES[name]():
                boundary = get_4326_boundary(osm_id=TERRITORY_OSM_ID)
                graph = get_intermodal_graph(territory=boundary, clip_by_territory=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(f"OSM source {name} failed: {exc!r}")
            errors.append(f"{name}: {exc!r}")
            continue
        logger.info(f"Test territory built from {name}")
        return boundary, graph
    raise RuntimeError("No OSM source worked for the test territory: " + "; ".join(errors))


@pytest.fixture(scope="session")
def osm_1114252() -> tuple:
    """``(boundary, intermodal graph)`` of the test territory, cached in test_cache so later runs are offline."""
    boundary_file = os.path.join(cache_dir, "boundary_1114252.wkt")
    graph_file = os.path.join(cache_dir, "intermodal_graph_1114252.urbangraph")
    if os.path.exists(boundary_file) and os.path.exists(graph_file):
        try:
            logger.info(f"Loading cached test territory from {cache_dir}")
            with open(boundary_file, encoding="utf-8") as file:
                return shapely.from_wkt(file.read()), UrbanGraph.read(graph_file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to load the cached test territory: {e}. Regenerating...")
    boundary, graph = _build_territory()
    try:
        graph.write(graph_file)
        with open(boundary_file, "w", encoding="utf-8") as file:
            file.write(boundary.wkt)
    except OSError as e:
        logger.error(f"Failed to cache the test territory: {e}")
    return boundary, graph


@pytest.fixture(scope="session")
def boundary_osm_1114252(osm_1114252):
    return osm_1114252[0]


@pytest.fixture(scope="session")
def intermodal_osm_1114252(osm_1114252) -> UrbanGraph:
    return osm_1114252[1]


@pytest.fixture(scope="session")
def intermodal_osm_1114252_edges_gdf(intermodal_osm_1114252) -> gpd.GeoDataFrame:
    return intermodal_osm_1114252.edges_gdf


@pytest.fixture(scope="session")
def gdf_1point():
    return gpd.GeoDataFrame(geometry=[Point(30.27060176, 59.93546846)], crs=4326)


@pytest.fixture(scope="session")
def gdf_3points():
    points = [Point(30.27060176, 59.93546846), Point(30.29586657, 59.94410918), Point(30.2312112, 59.9482336)]
    return gpd.GeoDataFrame(geometry=points, crs=4326)
