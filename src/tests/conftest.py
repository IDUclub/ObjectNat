import os
import pickle

import geopandas as gpd
import pandas as pd
import pytest
from iduedu import config, get_4326_boundary, get_intermodal_graph
from shapely import Point

from objectnat import graph_to_gdf

logger = config.logger

path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docs/methods/examples/examples_data/")
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


@pytest.fixture(scope="session")
def boundary_osm_1114252():
    return get_4326_boundary(osm_id=1114252)


@pytest.fixture(scope="session")
def intermodal_osm_1114252(boundary_osm_1114252):
    cache_file = os.path.join(cache_dir, "intermodal_graph_1114252.pickle")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                logger.info(f"Loading cached graph from {cache_file}")
                return pickle.load(f)
        except (pickle.PickleError, EOFError) as e:
            logger.warning(f"Failed to load cached graph: {e}. Regenerating...")
            os.remove(cache_file)
    logger.info("Generating new intermodal graph")
    graph = get_intermodal_graph(territory=boundary_osm_1114252, clip_by_territory=True)
    try:
        with open(cache_file, "wb") as f:
            logger.info(f"Saving graph to cache: {cache_file}")
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    except IOError as e:
        logger.error(f"Failed to cache graph: {e}")
    return graph


@pytest.fixture(scope="session")
def intermodal_osm_1114252_edges_gdf(intermodal_osm_1114252):
    graph_gdf = graph_to_gdf(intermodal_osm_1114252, nodes=False, restore_edge_geom=True)
    return graph_gdf


@pytest.fixture(scope="session")
def gdf_1point():
    return gpd.GeoDataFrame(geometry=[Point(30.27060176, 59.93546846)], crs=4326)


@pytest.fixture(scope="session")
def gdf_3points():
    points = [Point(30.27060176, 59.93546846), Point(30.29586657, 59.94410918), Point(30.2312112, 59.9482336)]
    return gpd.GeoDataFrame(geometry=points, crs=4326)
