import geopandas as gpd
import pandas as pd

from .provision_calculating import calculate_provision, recalculate_provision
from .provision_result import ProvisionResult


def get_service_provision(
    *,
    buildings: gpd.GeoDataFrame,
    distance_matrix: pd.DataFrame,
    services: gpd.GeoDataFrame,
    threshold: float,
    buildings_demand_column: str = "demand",
    services_capacity_column: str = "capacity",
    seed: int = 0,
) -> ProvisionResult:
    """
    Compute service provision between demand locations and service facilities.

    The function implements a **gravity-based allocation model**: service capacity is
    distributed across nearby demand points with weights that **decay with the square
    of distance (or generalized cost)**. Closer buildings receive proportionally
    higher shares of the available capacity. Only building-service pairs within
    ``3 * threshold`` take part in the allocation, which runs in ObjectNat's Rust
    extension over those pairs only.

    Args:
        buildings:
            GeoDataFrame of demand locations. Must include a numeric demand column.
        distance_matrix:
            OD cost matrix where rows are ``buildings.index`` and columns are
            ``services.index``. Units must match ``threshold``. Either a dense
            DataFrame (``inf``/``NaN`` = unreachable) or, for large areas, a pandas
            sparse DataFrame (``SparseDtype`` with a ``NaN`` or ``inf`` fill value)
            that stores only the reachable pairs.
        services:
            GeoDataFrame of service facilities. Must include a numeric capacity column.
        threshold:
            Normative cost threshold. Units are the same as in ``distance_matrix``.
        buildings_demand_column:
            Column name of building demand values. Default is ``"demand"``.
        services_capacity_column:
            Column name of service capacity values. Default is ``"capacity"``.
        seed:
            Seed of the random draws in the allocation. A fixed seed yields
            reproducible results (independent of the number of CPU cores); vary it
            to sample alternative allocations. Default is ``0``.

    Returns:
        ProvisionResult:
            Dataclass with a sparse building-service flow matrix, building metrics,
            service metrics, the aligned distance matrix, and the provision threshold.
    """
    return calculate_provision(
        buildings=buildings,
        services=services,
        distance_matrix=distance_matrix,
        threshold=threshold,
        buildings_demand_column=buildings_demand_column,
        services_capacity_column=services_capacity_column,
        seed=seed,
    )


def clip_provision(
    buildings: gpd.GeoDataFrame, services: gpd.GeoDataFrame, links: gpd.GeoDataFrame, selection_zone: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clip service provision outputs to a specific geographic boundary.

    Keeps only buildings that intersect ``selection_zone``, links that connect
    to the kept buildings, and services referenced by those links.
    """
    if not selection_zone.crs == buildings.crs == services.crs == links.crs:
        raise ValueError(
            f"CRS mismatch: buildings_crs:{buildings.crs}, "
            f"links_crs:{links.crs} , "
            f"services_crs:{services.crs}, "
            f"selection_zone_crs:{selection_zone.crs}"
        )
    buildings = buildings.copy()
    links = links.copy()
    services = services.copy()

    s = buildings.intersects(selection_zone.union_all())
    buildings = buildings.loc[s[s].index]
    links = links[links["building_index"].isin(buildings.index.tolist())]
    services_to_keep = set(links["service_index"].tolist())
    services.drop(list(set(services.index.tolist()) - services_to_keep), inplace=True)
    return buildings, services, links


def recalculate_links(provision_result: ProvisionResult, new_max_dist: float) -> ProvisionResult:
    """
    Recalculate provision result after tightening the maximum allowed link distance.

    Flows whose cost exceeds ``new_max_dist`` are removed, and building/service
    aggregates are recomputed without redistributing removed demand.
    """
    return recalculate_provision(provision_result, new_max_dist)
