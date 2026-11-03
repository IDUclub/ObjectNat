import os
from contextlib import nullcontext

import geopandas as gpd
import matplotlib
import pandas as pd
import pytest
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


@pytest.fixture(scope="session")
def osm_pbf_path():
    """OSM file of the test territory cut from the Geofabrik extract and cached in ``test_cache/geofabrik``,
    or ``None`` (data then comes from Overpass) when the installed IduEdu cannot download extracts."""
    try:
        from iduedu import download_osm_pbf  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.info("This IduEdu cannot download Geofabrik extracts; using Overpass")
        return None
    return download_osm_pbf(
        osm_id=TERRITORY_OSM_ID, region=GEOFABRIK_REGION, out_dir=os.path.join(cache_dir, "geofabrik")
    )


def _reading(osm_pbf_path):
    return config.use_pbf(osm_pbf_path) if osm_pbf_path is not None else nullcontext()


@pytest.fixture(scope="session")
def boundary_osm_1114252(osm_pbf_path):
    with _reading(osm_pbf_path):
        return get_4326_boundary(osm_id=TERRITORY_OSM_ID)


@pytest.fixture(scope="session")
def intermodal_osm_1114252(boundary_osm_1114252, osm_pbf_path) -> UrbanGraph:
    cache_file = os.path.join(cache_dir, "intermodal_graph_1114252.urbangraph")
    if os.path.exists(cache_file):
        try:
            logger.info(f"Loading cached graph from {cache_file}")
            return UrbanGraph.read(cache_file)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Failed to load cached graph: {e}. Regenerating...")
            os.remove(cache_file)
    logger.info("Generating new intermodal graph")
    with _reading(osm_pbf_path):
        graph = get_intermodal_graph(territory=boundary_osm_1114252, clip_by_territory=True)
    try:
        logger.info(f"Saving graph to cache: {cache_file}")
        graph.write(cache_file)
    except OSError as e:
        logger.error(f"Failed to cache graph: {e}")
    return graph


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
